"""提供显式连接、使能、目标预检、执行和停止的 ROS 位置接口."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from functools import partial
import json
from threading import Lock
import time
from typing import Callable

import rclpy
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from .hex_adapter import six_values
from .position_adapter import HexPositionAdapter
from .position_plan import JointPositionMove, PositionLimits, positive
from .position_worker import PositionBackend, PositionWorker


@dataclass(frozen=True)
class StagedTarget:
    """已经预检但尚未执行的一次性目标."""

    positions: tuple[float, ...]
    duration: float
    received: float


def vector_parameter(value: object, name: str) -> tuple[float, ...]:
    """读取六维数组，避免将 ROS 参数中的空值直接转换为数值."""
    if not isinstance(value, (list, tuple, array)):
        raise ValueError(f'{name} 必须为六维数组')
    return six_values(value)


class PositionControlNode(Node):
    """双臂各有独立执行线程，ROS 服务只受理请求并发布状态."""

    def __init__(self) -> None:
        """校验所有启动参数后再创建线程，默认不连接任何设备."""
        super().__init__('archer_position_driver')
        descriptor = ParameterDescriptor(read_only=True, dynamic_typing=True)
        defaults = {
            'arms': 'both', 'connect_on_start': False, 'control_hz': 250,
            'startup_timeout': 10.0, 'target_timeout': 5.0, 'return_duration': 5.0,
            'max_excursion': 0.05, 'tracking_tolerance': 0.10,
            'feedback_tolerance': 0.005,
            'left_url': 'ws://192.168.1.100:8439',
            'right_url': 'ws://192.168.1.100:9439',
            'left_limits_confirmed': False, 'right_limits_confirmed': False,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value, descriptor)
        for side in ('left', 'right'):
            for suffix in ('lower', 'upper', 'max_velocity', 'max_acceleration'):
                self.declare_parameter(f'{side}_{suffix}', [], descriptor)
        if self.get_parameter('use_sim_time').value is not False:
            raise ValueError('实机位置节点要求 use_sim_time=false')
        self.add_on_set_parameters_callback(self._validate_runtime_parameters)
        arms = self.get_parameter('arms').value
        if arms not in ('left', 'right', 'both'):
            raise ValueError('arms 仅接受 left、right 或 both')
        self.sides = ('left', 'right') if arms == 'both' else (str(arms),)
        frequency = self.get_parameter('control_hz').value
        if isinstance(frequency, bool) or not isinstance(frequency, int):
            raise ValueError('control_hz 必须为整数')
        if frequency not in (250, 500):
            raise ValueError('control_hz 仅接受 250 或 500')
        startup = positive(self.get_parameter('startup_timeout').value, 'startup_timeout')
        self.target_timeout = positive(
            self.get_parameter('target_timeout').value, 'target_timeout',
        )
        self.return_duration = positive(
            self.get_parameter('return_duration').value, 'return_duration',
        )
        self.max_excursion = positive(
            self.get_parameter('max_excursion').value, 'max_excursion',
        )
        tolerance = positive(
            self.get_parameter('tracking_tolerance').value, 'tracking_tolerance',
        )
        feedback_tolerance = positive(
            self.get_parameter('feedback_tolerance').value, 'feedback_tolerance',
        )
        autoconnect = self.get_parameter('connect_on_start').value
        if not isinstance(autoconnect, bool):
            raise ValueError('connect_on_start 必须是布尔值')
        self._lock = Lock()
        self._interlock_reason = ''
        self._targets: dict[str, StagedTarget] = {}
        self._target_messages: dict[str, str] = {}
        self.limits: dict[str, PositionLimits] = {}
        self.workers: dict[str, PositionWorker] = {}
        urls = []
        for side in self.sides:
            confirmed = self.get_parameter(f'{side}_limits_confirmed').value
            if not isinstance(confirmed, bool):
                raise ValueError(f'{side}_limits_confirmed 必须为布尔值')
            url = self.get_parameter(f'{side}_url').value
            if not isinstance(url, str):
                raise ValueError(f'{side}_url 必须是字符串')
            urls.append(url)
            factory: Callable[[], PositionBackend] | None = None
            if confirmed:
                vectors = {
                    suffix: vector_parameter(self.get_parameter(f'{side}_{suffix}').value, suffix)
                    for suffix in ('lower', 'upper', 'max_velocity', 'max_acceleration')
                }
                limits = PositionLimits(
                    vectors['lower'], vectors['upper'], vectors['max_velocity'],
                    vectors['max_acceleration'], feedback_tolerance=feedback_tolerance,
                )
                self.limits[side] = limits
                factory = partial(
                    HexPositionAdapter, url, limits, control_hz=frequency,
                    max_excursion=self.max_excursion, tracking_tolerance=tolerance,
                )
            elif autoconnect:
                raise ValueError(f'{side} 真实限位尚未确认，不能自动连接')
            self.workers[side] = PositionWorker(
                side, factory, self._trip, control_hz=frequency, startup_timeout=startup,
            )
        if len(urls) != len(set(urls)):
            raise ValueError('左右臂不能使用相同地址')
        qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_pub = self.create_publisher(String, '/hardware/position/status', qos)
        self.feedback_pubs = {
            side: self.create_publisher(
                JointState, f'/hardware/position/{side}/joint_states', 10,
            ) for side in self.sides
        }
        self._service_handles = []
        self._subscription_handles = []
        for side in self.sides:
            for kind in (
                'connect', 'enable', 'move', 'return_initial', 'disconnect', 'stop', 'halt',
            ):
                self._service_handles.append(self.create_service(
                    Trigger, f'/hardware/position/{side}/{kind}',
                    self._service_callback(side, kind),
                ))
            self._subscription_handles.append(self.create_subscription(
                JointTrajectory, f'/hardware/position/{side}/target',
                self._target_callback(side), 1,
            ))
        for kind in ('stop', 'halt', 'reset'):
            self._service_handles.append(self.create_service(
                Trigger, f'/hardware/position/{kind}', self._service_callback(None, kind),
            ))
        self._last_status = ''
        self._last_status_time = 0.0
        self._published_source: dict[str, int] = {}
        self.create_timer(0.02, self.publish_state)
        try:
            for worker in self.workers.values():
                worker.start()
            if autoconnect:
                for worker in self.workers.values():
                    worker.submit('connect')
        except Exception:
            self.close_workers()
            raise
        self.publish_state()

    def _validate_runtime_parameters(self, parameters: list[Parameter]) -> SetParametersResult:
        result = SetParametersResult()
        result.successful = not any(
            p.name == 'use_sim_time' and p.value is not False for p in parameters
        )
        result.reason = '' if result.successful else '实机位置节点禁止使用仿真时间'
        return result

    def _trip(self, reason: str) -> None:
        with self._lock:
            if not self._interlock_reason:
                self._interlock_reason = reason
            self._targets.clear()
        for worker in self.workers.values():
            worker.request_stop()

    def _service_callback(
        self, side: str | None, kind: str,
    ) -> Callable[[Trigger.Request, Trigger.Response], Trigger.Response]:
        def callback(request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
            try:
                response.message = self.handle_request(side, kind)
                response.success = True
            except (ValueError, RuntimeError) as error:
                response.success = False
                response.message = str(error)
            return response
        return callback

    def handle_request(self, side: str | None, kind: str) -> str:
        """受理结果与执行结果分开，最终结果通过状态话题确认."""
        if kind in ('stop', 'halt'):
            self._trip('用户请求停止，需断开连接并显式复位')
            return '已请求两臂停止后续目标；不代表物理停止已确认'
        if kind == 'reset':
            if any(w.snapshot().state not in ('DISCONNECTED', 'UNCONFIGURED')
                   for w in self.workers.values()):
                raise RuntimeError('先等待两臂断开连接，再复位软件联锁')
            with self._lock:
                self._interlock_reason = ''
                self._targets.clear()
            return '软件联锁已复位；未连接、未使能、未清除设备故障'
        if side is None or side not in self.workers:
            raise ValueError('未配置该机械臂')
        with self._lock:
            if self._interlock_reason and kind != 'disconnect':
                raise RuntimeError(f'软件联锁：{self._interlock_reason}')
            if kind == 'move':
                target = self._targets.get(side)
                if target is None:
                    raise RuntimeError('请先发布一个有效的单点目标')
                if time.monotonic() - target.received > self.target_timeout:
                    del self._targets[side]
                    raise RuntimeError('预检目标已过期，请重新发布')
                identifier = self.workers[side].submit(
                    'move', goal=target.positions, duration=target.duration,
                )
                del self._targets[side]
            else:
                identifier = self.workers[side].submit(
                    'return' if kind == 'return_initial' else kind,
                    duration=self.return_duration,
                )
                if kind in ('disconnect', 'return_initial'):
                    self._targets.pop(side, None)
        return f'已受理编号 {identifier}，请查看状态话题；不代表动作已完成'

    def _target_callback(self, side: str) -> Callable[[JointTrajectory], None]:
        def callback(message: JointTrajectory) -> None:
            try:
                self.stage_target(side, message)
            except (ValueError, RuntimeError) as error:
                with self._lock:
                    self._targets.pop(side, None)
                    self._target_messages[side] = f'REJECTED: {error}'
        return callback

    def stage_target(self, side: str, message: JointTrajectory) -> None:
        """单点消息只暂存；必须另行调用 move 服务才可能执行."""
        with self._lock:
            if self._interlock_reason:
                raise RuntimeError('软件联锁尚未复位')
            if side not in self.limits:
                raise RuntimeError('本臂真实限位尚未确认')
            points = list(message.points)
            if len(points) != 1:
                raise ValueError('当前接口仅接受一个目标点，不接收整段轨迹')
            if message.header.stamp.sec != 0 or message.header.stamp.nanosec != 0:
                raise ValueError('当前单点接口不支持预约时间戳')
            if message.header.frame_id:
                raise ValueError('关节目标不使用笛卡尔 frame_id')
            expected = [f'{side}_joint_{i}' for i in range(1, 7)]
            names = list(message.joint_names)
            if len(names) != 6 or set(names) != set(expected):
                raise ValueError('必须提供本臂六个唯一关节名，禁止混入另一只臂')
            point = points[0]
            values = six_values(point.positions)
            if len(point.velocities) or len(point.accelerations) or len(point.effort):
                raise ValueError('当前只接受位置目标，速度和加速度由预检轨迹生成')
            sec = point.time_from_start.sec
            ns = point.time_from_start.nanosec
            if sec < 0 or not 0 <= ns < 1_000_000_000:
                raise ValueError('目标时间格式错误')
            duration = positive(sec + ns / 1e9, '目标时长')
            ordered = tuple(values[names.index(name)] for name in expected)
            snapshot = self.workers[side].snapshot()
            frame = snapshot.feedback
            if snapshot.state != 'HOLDING' or frame is None:
                raise RuntimeError('先显式使能并等待 HOLDING 和新鲜反馈')
            age = (time.perf_counter_ns() - frame.source_ns) / 1e9
            if not 0 <= age <= 0.1:
                raise RuntimeError('预检使用的真实反馈已过期')
            JointPositionMove.prepare(
                frame.positions, ordered, duration, self.limits[side], self.max_excursion,
            )
            self._targets[side] = StagedTarget(ordered, duration, time.monotonic())
            self._target_messages[side] = 'STAGED：已预检，尚未执行'

    def publish_state(self) -> None:
        """发布独立反馈和可供晚加入订阅者读取的执行状态."""
        states = {}
        with self._lock:
            interlock = self._interlock_reason
            targets = {side: side in self._targets for side in self.sides}
            target_messages = dict(self._target_messages)
        for side, worker in self.workers.items():
            snapshot = worker.snapshot()
            frame = snapshot.feedback
            fresh = (
                frame is not None
                and 0 <= (time.perf_counter_ns() - frame.source_ns) / 1e9 <= 0.1
            )
            states[side] = {
                'state': snapshot.state, 'error': snapshot.error,
                'feedback_fresh': fresh, 'initial_positions': snapshot.initial_positions,
                'command_id': snapshot.command_id, 'command': snapshot.command,
                'phase': snapshot.phase, 'message': snapshot.message,
                'target_staged': targets[side], 'target_message': target_messages.get(side, ''),
                'within_nominal_limits': (self.limits[side].within_nominal(frame.positions)
                                          if fresh and frame is not None and side in self.limits
                                          else None),
                'return_target': (self.limits[side].return_goal(snapshot.initial_positions)
                                  if snapshot.initial_positions is not None and side in self.limits
                                  else None),
            }
            if fresh and frame is not None and self._published_source.get(side) != frame.source_ns:
                message = JointState()
                message.header.stamp = self.get_clock().now().to_msg()
                message.name = [f'{side}_joint_{i}' for i in range(1, 7)]
                message.position = list(frame.positions)
                message.velocity = list(frame.velocities)
                message.effort = list(frame.efforts)
                self.feedback_pubs[side].publish(message)
                self._published_source[side] = frame.source_ns
        text = json.dumps({'interlock': interlock, 'arms': states}, ensure_ascii=False)
        now = time.monotonic()
        if text != self._last_status or now - self._last_status_time >= 1.0:
            status = String()
            status.data = text
            self.status_pub.publish(status)
            self._last_status = text
            self._last_status_time = now

    def close_workers(self) -> None:
        """两臂先同时请求停止，再分别等待线程退出."""
        for worker in self.workers.values():
            worker.request_stop()
        errors = []
        for worker in self.workers.values():
            try:
                worker.shutdown()
            except RuntimeError as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError('；'.join(errors))


def main(args: list[str] | None = None) -> None:
    """运行 ROS 服务节点；退出不自动归位."""
    rclpy.init(args=args)
    node: PositionControlNode | None = None
    try:
        node = PositionControlNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            if node is not None:
                try:
                    node.close_workers()
                finally:
                    node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == '__main__':
    main()
