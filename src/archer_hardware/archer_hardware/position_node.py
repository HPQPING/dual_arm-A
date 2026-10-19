"""提供显式连接、使能、目标预检、执行和停止的 ROS 位置接口."""

from __future__ import annotations

from array import array
from concurrent.futures import Executor, Future, ProcessPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from importlib import import_module
import json
from multiprocessing import get_context
from multiprocessing.managers import SyncManager
from pathlib import Path
import re
from threading import Event, Lock
import time
from typing import Any, Callable, Protocol

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

from .hex_adapter import JointFeedback, six_values
from .commissioning import review_joint_plan
from .position_adapter import HexPositionAdapter
from .position_plan import (
    JointPositionMove, JointPositionTrajectory, PairedTrajectoryStart, PositionLimits, positive,
    ContinuousPositionTrajectory, prepare_continuous_pair,
)
from .position_worker import PositionBackend, PositionWorker


class PreparationCancellation(Protocol):
    """兼容本机事件和跨进程事件代理，只暴露取消所需的最小接口."""

    def is_set(self) -> bool:
        """读取是否已经取消."""
        ...

    def set(self) -> None:
        """通知计算任务取消."""
        ...


@dataclass(frozen=True)
class StagedTarget:
    """已经预检但尚未执行的一次性目标."""

    positions: tuple[float, ...]
    duration: float
    received: float


@dataclass(frozen=True)
class StagedPair:
    """同一时间轴及两侧不可变位置数组，尚未执行."""

    times_ns: tuple[int, ...]
    positions: dict[str, tuple[tuple[float, ...], ...]]
    received: float
    continuous: dict[str, ContinuousPositionTrajectory] | None = None
    details: dict[str, Any] | None = None


def vector_parameter(value: object, name: str) -> tuple[float, ...]:
    """读取六维数组，避免将 ROS 参数中的空值直接转换为数值."""
    if not isinstance(value, (list, tuple, array)):
        raise ValueError(f'{name} 必须为六维数组')
    return six_values(value)


def prepare_pair_message(
    text: str, limits: dict[str, PositionLimits], max_excursion: float,
    allow_continuous: bool, cancelled: PreparationCancellation,
) -> StagedPair:
    """后台仅解析数据与限制，不读取ROS节点或SDK；停止后结果不得进入暂存."""
    data = json.loads(text)
    if cancelled.is_set():
        raise ValueError('成对预检已取消')
    if isinstance(data, dict) and data.get('version') == 2:
        if not allow_continuous:
            raise ValueError('版本二连续轨迹尚未显式允许')
        plans = prepare_continuous_pair(
            data, limits, max_excursion, cancelled=cancelled.is_set,
        )
        return StagedPair(plans['left'].times_ns, {}, 0., plans)
    review_joint_plan(data, limits)
    assert isinstance(data, dict)
    times = tuple(p['time_from_start']['sec'] * 10 ** 9 + p['time_from_start']['nanosec']
                  for p in data['left']['points'])
    positions = {side: tuple(six_values(p['positions']) for p in data[side]['points'])
                 for side in ('left', 'right')}
    return StagedPair(times, positions, 0.)


def create_preparation_resources() -> tuple[Executor, SyncManager | None]:
    """使用spawn进程隔离数值计算，避免fork带有ROS及设备线程的进程."""
    context = get_context('spawn')
    manager = context.Manager()
    try:
        return ProcessPoolExecutor(max_workers=1, mp_context=context), manager
    except Exception:
        manager.shutdown()
        raise


def file_request_id(text: str) -> str:
    """只解析至多16Ki字符的定位请求编号，不读取轨迹文件或SDK."""
    from .task_file import json_mapping

    if not isinstance(text, str) or len(text) > 16384:
        raise ValueError('文件定位请求无效或超过16Ki字符上限')
    data = json_mapping(text.encode('utf-8'))
    identifier = data.get('request_id')
    if not isinstance(identifier, str) or re.fullmatch('[0-9a-f]{32}', identifier) is None:
        raise ValueError('文件请求需要唯一的32位小写十六进制编号')
    return identifier


def prepare_file_task_message(
    text: str, limits: dict[str, PositionLimits], max_excursion: float,
    start: dict[str, tuple[float, ...]], cancelled: PreparationCancellation,
) -> StagedPair:
    """在计算进程核对文件快照并生成已检完整路径，不读取SDK、不自行发送运动."""
    from .task_file import json_mapping, load_reviewed_task
    from .task_geometry import assemble_task_cycle, check_task_geometry, prepare_approach
    from .task_geometry import prepare_return

    data = json_mapping(text.encode('utf-8'))
    required = {'request_id', 'run_dir', 'limits_file', 'prepared_file', 'report_file',
                'transition_max_excursion'}
    if set(data) != required:
        raise ValueError('文件预检请求字段不完整或含未知字段')
    identifier = file_request_id(text)
    paths = {}
    for name in ('run_dir', 'limits_file', 'prepared_file', 'report_file'):
        value = data[name]
        if not isinstance(value, str) or not value.strip() or '\x00' in value:
            raise ValueError(name + '需要有效路径')
        path = Path(value).expanduser()
        if not path.is_absolute():
            raise ValueError(name + '必须是驱动所在电脑的绝对路径')
        paths[name] = path
    transition = positive(data['transition_max_excursion'], 'transition_max_excursion')
    if transition > max_excursion:
        raise ValueError('过渡范围不能大于驱动允许的完整任务范围')
    reviewed = load_reviewed_task(
        paths['run_dir'], paths['limits_file'], paths['prepared_file'], paths['report_file'],
        max_excursion=max_excursion, cancelled=cancelled.is_set,
    )
    if reviewed.report['task_type'] != 'split_circle':
        raise ValueError('抓取文件含虚拟附着事件，真实夹爪未接入；当前仅允许画圆文件进入位置执行')
    if set(limits) != {'left', 'right'} or any(
        reviewed.plans[side].limits != limits[side] for side in ('left', 'right')
    ):
        raise ValueError('准备文件使用的限位与当前驱动限制不一致')
    approach = prepare_approach(reviewed.plans, start, max_excursion=transition)
    home = prepare_return(reviewed.plans, start, max_excursion=transition)
    cycle = assemble_task_cycle(reviewed.plans, approach=approach, return_home=home,
                                max_excursion=max_excursion, cancelled=cancelled.is_set)
    geometry = check_task_geometry(reviewed.plans, approach=approach, return_home=home,
                                   cancelled=cancelled.is_set)
    if cancelled.is_set():
        raise ValueError('文件预检已取消，禁止返回可执行对象')
    if geometry.get('sampled_cross_arm_table_passed') is not True:
        raise ValueError('完整文件的离散几何检查没有返回通过结果')
    details = {
        'request_id': identifier, 'task_type': 'split_circle',
        'duration_s': cycle['left'].duration, 'sha256': dict(reviewed.report['sha256']),
        'initial_positions': {s: list(p.initial) for s, p in cycle.items()},
        'phase_start_ns': {'APPROACH_FILE': 0, 'TASK': approach['left'].times_ns[-1],
                           'RETURN_ACTUAL': (approach['left'].times_ns[-1]
                                             + reviewed.plans['left'].times_ns[-1])},
        'sampled_geometry': geometry, 'preserve_path': True,
        'gripper_commands_included': False,
    }
    return StagedPair(cycle['left'].times_ns, {}, 0., cycle, details)


class PositionControlNode(Node):
    """双臂各有独立执行线程，ROS 服务只受理请求并发布状态."""

    def __init__(self) -> None:
        """校验所有启动参数后再创建线程，默认不连接任何设备."""
        super().__init__('archer_position_driver')
        descriptor = ParameterDescriptor(read_only=True, dynamic_typing=True)
        defaults = {
            'arms': 'both', 'connect_on_start': False, 'control_hz': 250,
            'startup_timeout': 10.0, 'target_timeout': 5.0,
            'max_excursion': 0.05, 'tracking_tolerance': 0.10,
            'feedback_tolerance': 0.005,
            'allow_api_calibration': False, 'readiness_timeout': 10.0,
            'allow_paired_trajectory': False,
            'allow_continuous_trajectory': False,
            'allow_task_file_trajectory': False,
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
        self.allow_api_calibration = self.get_parameter('allow_api_calibration').value
        if type(self.allow_api_calibration) is not bool:
            raise ValueError('allow_api_calibration 必须为布尔值')
        self.allow_paired_trajectory = self.get_parameter('allow_paired_trajectory').value
        if type(self.allow_paired_trajectory) is not bool:
            raise ValueError('allow_paired_trajectory 必须为布尔值')
        allow_continuous = self.get_parameter('allow_continuous_trajectory').value
        if not isinstance(allow_continuous, bool):
            raise ValueError('allow_continuous_trajectory 必须为布尔值')
        self.allow_continuous_trajectory: bool = allow_continuous
        allow_files = self.get_parameter('allow_task_file_trajectory').value
        if not isinstance(allow_files, bool):
            raise ValueError('allow_task_file_trajectory必须是布尔值')
        if allow_files and (not allow_continuous or not self.allow_paired_trajectory):
            raise ValueError('完整文件需同时明确允许成对轨迹和连续轨迹')
        self.allow_task_file_trajectory: bool = allow_files
        self._paired_details: dict[str, Any] = {}
        self._file_request_id = ''
        self._file_rejection: dict[str, str] = {}
        if allow_continuous:
            # 在启动工作线程及连接设备之前加载数值库，避免持有控制时首次导入争用GIL。
            import_module('archer_control.continuous_trajectory')
        if allow_files:
            # 设备线程启动前载入定位请求所需模块，避免首次文件回调导入争用GIL。
            import_module('archer_hardware.task_file')
            import_module('archer_hardware.task_geometry')
        readiness_timeout = positive(
            self.get_parameter('readiness_timeout').value, 'readiness_timeout',
        )
        if readiness_timeout > 30.0:
            raise ValueError('readiness_timeout 不能超过 30 秒')
        self._lock = Lock()
        self._interlock_reason = ''
        self._paired_target: StagedPair | None = None
        self._paired_active = False
        self._paired_gate: PairedTrajectoryStart | None = None
        self._paired_ids: dict[str, str] = {}
        self._paired_requested_at = 0.0
        self._paired_phase = 'IDLE'
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
                    allow_api_calibration=self.allow_api_calibration,
                    readiness_timeout=readiness_timeout,
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
                'connect', 'enable', 'move', 'disconnect',
            ):
                self._service_handles.append(self.create_service(
                    Trigger, f'/hardware/position/{side}/{kind}',
                    self._service_callback(side, kind),
                ))
            self._subscription_handles.append(self.create_subscription(
                JointTrajectory, f'/hardware/position/{side}/target',
                self._target_callback(side), 1,
            ))
        self._subscription_handles.append(self.create_subscription(
            String, '/hardware/position/paired_target', self._paired_target_callback, 1,
        ))
        self._subscription_handles.append(self.create_subscription(
            String, '/hardware/position/task_file', self._file_target_callback, 1,
        ))
        self._subscription_handles.append(self.create_subscription(
            String, '/hardware/position/task_execute', self._file_execute_callback, 1,
        ))
        for kind in ('halt', 'paired_move'):
            self._service_handles.append(self.create_service(
                Trigger, f'/hardware/position/{kind}', self._service_callback(None, kind),
            ))
        self._last_status = ''
        self._last_status_time = 0.0
        self._published_source: dict[str, int] = {}
        self._preparer: Executor | None = None
        self._preparation_manager: SyncManager | None = None
        if self.allow_paired_trajectory:
            self._preparer, self._preparation_manager = create_preparation_resources()
        self._preparation: Future[StagedPair] | None = None
        self._preparation_cancel: PreparationCancellation = Event()
        self._preparation_valid = False
        self._preparation_started = 0.
        self._preparation_timeout = 30.
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
            self._paired_target = None
            self._cancel_preparation_locked(signal=False)
            self._paired_active = False
            self._paired_phase = 'CANCELLED'
            if self._paired_gate is not None:
                self._paired_gate.cancel()
        for worker in self.workers.values():
            worker.request_stop()
        self._signal_preparation_cancel()

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
        if kind == 'halt':
            self._trip('立即停止：本次运行已结束，后续由人工断电及复位')
            return '已请求两臂停止后续目标；不代表物理停止已确认'
        if kind == 'paired_move':
            if side is not None:
                raise ValueError('成对执行必须使用全局入口')
            return self.request_paired_trajectory()
        if kind not in ('connect', 'enable', 'move', 'disconnect'):
            raise ValueError('该操作已移除；本次运行不提供软件复位或恢复入口')
        if side is None or side not in self.workers:
            raise ValueError('未配置该机械臂')
        with self._lock:
            if self._interlock_reason and kind != 'disconnect':
                raise RuntimeError(f'本次运行已结束：{self._interlock_reason}')
            if self._preparation is not None:
                if kind != 'disconnect':
                    raise RuntimeError('后台预检尚未结束；如需取消请使用 halt')
                self._cancel_preparation_locked()
                self._paired_phase = 'REJECTED: 预检期间显式断开'
            if self._paired_active:
                raise RuntimeError('成对轨迹正在准备或执行；如遇异常请使用 halt')
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
                identifier = self.workers[side].submit(kind)
                if kind == 'disconnect':
                    self._targets.pop(side, None)
        return f'已受理编号 {identifier}，请查看状态话题；不代表动作已完成'

    def _paired_target_callback(self, message: String) -> None:
        """ROS回调仅提交字符串；耗时JSON解析及多项式检查放在独立后台进程."""
        try:
            self.queue_pair_preparation(message.data)
        except (ValueError, RuntimeError, TypeError) as error:
            with self._lock:
                self._paired_target = None
                if not self._paired_active:
                    self._paired_phase = f'REJECTED: {error}'

    def _file_target_callback(self, message: String) -> None:
        """仅提交完整任务的文件定位请求，拒绝使用离线起点文件冒充实际反馈."""
        identifier = ''
        try:
            identifier = file_request_id(message.data)
            self.queue_pair_preparation(message.data, task_file=True)
        except (ValueError, RuntimeError, TypeError) as error:
            with self._lock:
                self._file_rejection = {'request_id': identifier, 'reason': str(error)}
                if not self._paired_active:
                    self._paired_target = None
                    self._paired_phase = f'REJECTED: {error}'

    def _file_execute_callback(self, message: String) -> None:
        """按预检编号显式消费一次文件计划，不会把另一次暂存当成本轮任务."""
        identifier = message.data if re.fullmatch('[0-9a-f]{32}', message.data) else ''
        try:
            if not identifier:
                raise ValueError('执行请求必须为32位预检编号')
            self.request_paired_trajectory(expected_request_id=identifier)
        except (ValueError, RuntimeError) as error:
            with self._lock:
                self._file_rejection = {'request_id': identifier, 'reason': str(error)}

    def _cancel_preparation_locked(self, signal: bool = True) -> None:
        """只作废后台结果，不排队新的预检；调用者已持有节点锁."""
        self._preparation_valid = False
        if self._preparation is not None:
            self._preparation.cancel()
        if signal:
            self._signal_preparation_cancel()

    def _signal_preparation_cancel(self) -> None:
        """取消计算进程失败也不能恢复旧结果，故障联停先请求设备线程停止."""
        try:
            self._preparation_cancel.set()
        except Exception as error:
            self.get_logger().error('后台取消信号未确认，结果仍已作废：' + str(error))

    def _pair_feedback(self, *, coherent: bool = False) -> dict[str, JointFeedback]:
        """同时核对两侧新鲜静止反馈；不会主动连接或申请控制."""
        result = {}
        for side in self.sides:
            snapshot = self.workers[side].snapshot()
            frame = snapshot.feedback
            if (snapshot.state != 'HOLDING' or frame is None
                    or not 0 <= (time.perf_counter_ns() - frame.source_ns) / 1e9 <= .1
                    or max(abs(v) for v in frame.velocities) > .02):
                raise RuntimeError(f'{side} 需要新鲜静止反馈及HOLDING状态')
            result[side] = frame
        if (coherent and len(result) == 2
                and abs(result['left'].source_ns - result['right'].source_ns) > 20000000):
            raise RuntimeError('双臂反馈时间差超过20ms，不能作为同一次起点快照')
        return result

    def _check_pair_starts(self, plan: StagedPair) -> None:
        """后台完成后用最新反馈复核起点，不沿用提交时的旧反馈."""
        for side, frame in self._pair_feedback(coherent=plan.details is not None).items():
            if plan.continuous is None:
                JointPositionTrajectory.prepare(
                    plan.times_ns, plan.positions[side], self.limits[side], self.max_excursion,
                    actual_start=frame.positions,
                )
            else:
                # 此处只检查；执行线程仍须相对文件原起点独立对齐，避免累计放宽容差。
                plan.continuous[side].aligned_start(
                    frame.positions, self.limits[side], self.max_excursion,
                )

    def queue_pair_preparation(self, text: str, *, task_file: bool = False) -> None:
        """至多保留一个后台任务；新消息到来先作废旧暂存，忙时拒绝而不堆积."""
        identifier = file_request_id(text) if task_file else ''
        with self._lock:
            self._paired_target = None
            if self._preparation is not None:
                self._cancel_preparation_locked()
                raise RuntimeError('后台预检尚未结束，已作废旧结果，请等待后重新提交')
            if (not self.allow_paired_trajectory or self.sides != ('left', 'right')
                    or set(self.limits) != {'left', 'right'}):
                raise RuntimeError('需允许成对轨迹并配置两臂已确认限位')
            if self._interlock_reason or self._paired_active:
                raise RuntimeError('停止或成对执行期间禁止预检新计划')
            if task_file and not self.allow_task_file_trajectory:
                raise RuntimeError('完整任务文件入口默认关闭，需显式允许')
            if not isinstance(text, str) or len(text) > 50 * 1024 * 1024:
                raise ValueError('计划字符串无效或超过50Mi字符上限')
            if task_file and len(text) > 16384:
                raise ValueError('文件定位请求超过16Ki字符上限')
            frames = self._pair_feedback(coherent=task_file)
            self._targets.clear()
            self._paired_ids = {}
            self._paired_details = {}
            self._file_request_id = identifier
            self._file_rejection = {}
            self._preparation_cancel = (self._preparation_manager.Event()
                                        if self._preparation_manager is not None else Event())
            self._preparation_valid = True
            self._preparation_started = time.monotonic()
            self._preparation_timeout = 120. if task_file else 30.
            self._paired_phase = 'VALIDATING'
            assert self._preparer is not None
            if task_file:
                self._preparation = self._preparer.submit(
                    prepare_file_task_message, text, dict(self.limits), self.max_excursion,
                    {side: frame.positions for side, frame in frames.items()},
                    self._preparation_cancel,
                )
            else:
                self._preparation = self._preparer.submit(
                    prepare_pair_message, text, dict(self.limits), self.max_excursion,
                    self.allow_continuous_trajectory, self._preparation_cancel,
                )

    def check_preparation_progress(self) -> None:
        """只消费已完成结果；取消、超时或起点变动都不能产生可执行暂存."""
        with self._lock:
            future = self._preparation
            if future is None:
                return
            if (self._preparation_valid
                    and time.monotonic() - self._preparation_started > self._preparation_timeout):
                self._cancel_preparation_locked()
                self._paired_phase = f'REJECTED: 后台预检超过{self._preparation_timeout:g}秒'
            if not future.done():
                return
            self._preparation = None
            if not self._preparation_valid or self._interlock_reason:
                return
            self._preparation_valid = False
            try:
                plan = future.result()
                if self._file_request_id and (
                    plan.details is None or plan.details.get('request_id') != self._file_request_id
                    or plan.continuous is None or set(plan.continuous) != {'left', 'right'}
                    or any(not curve.preserve_path for curve in plan.continuous.values())
                ):
                    raise ValueError('后台文件结果编号或路径保护标记不符，禁止暂存')
                if plan.continuous is not None:
                    # NumPy数组跨进程序列化后可能恢复可写；进入暂存前重新冻结。
                    for curve in plan.continuous.values():
                        for polynomial in curve.polynomials:
                            polynomial.setflags(write=False)
                self._check_pair_starts(plan)
                self._paired_target = replace(plan, received=time.monotonic())
                self._paired_details = dict(plan.details or {})
                self._paired_phase = 'STAGED'
            except Exception as error:
                self._paired_target = None
                self._paired_phase = f'REJECTED: {error}'

    def stage_paired_target(self, data: object) -> None:
        """沿用离线计划格式，使用两臂实际反馈和当前位移约束重新预检."""
        if isinstance(data, dict) and data.get('version') == 2:
            self.queue_pair_preparation(json.dumps(data, allow_nan=False))
            return
        with self._lock:
            if self._preparation is not None:
                self._cancel_preparation_locked()
                self._paired_target = None
                self._paired_phase = 'REJECTED: 新计划作废正在预检的旧结果'
                raise RuntimeError('后台预检尚未结束，请等待后重新提交')
            if not self.allow_paired_trajectory or self.sides != ('left', 'right'):
                raise RuntimeError('需显式允许成对轨迹并配置both，两臂不会自动连接或使能')
            if self._interlock_reason or self._paired_active:
                raise RuntimeError('停止或成对执行期间禁止暂存计划')
            if set(self.limits) != {'left', 'right'}:
                raise RuntimeError('两臂限位尚未确认')
            review_joint_plan(data, self.limits)
            assert isinstance(data, dict)
            times = tuple(p['time_from_start']['sec'] * 10 ** 9
                          + p['time_from_start']['nanosec'] for p in data['left']['points'])
            positions = {
                side: tuple(six_values(p['positions']) for p in data[side]['points'])
                for side in self.sides
            }
            for side in self.sides:
                snapshot = self.workers[side].snapshot()
                frame = snapshot.feedback
                if (snapshot.state != 'HOLDING' or frame is None
                        or not 0 <= (time.perf_counter_ns() - frame.source_ns) / 1e9 <= 0.1
                        or max(abs(v) for v in frame.velocities) > 0.02):
                    raise RuntimeError(f'{side} 需要新鲜静止反馈及HOLDING状态')
                JointPositionTrajectory.prepare(
                    times, positions[side], self.limits[side], self.max_excursion,
                    actual_start=frame.positions,
                )
            self._targets.clear()
            self._paired_target = StagedPair(times, positions, time.monotonic())
            self._paired_details = {}
            self._file_request_id = ''
            self._file_rejection = {}
            self._paired_ids = {}
            self._paired_phase = 'STAGED'

    def request_paired_trajectory(self, *, expected_request_id: str | None = None) -> str:
        """消费暂存计划，两线程都确认准备后才确定共同单调启动时刻."""
        failure: Exception | None = None
        identifiers: dict[str, str] = {}
        with self._lock:
            if (not self.allow_paired_trajectory or self.sides != ('left', 'right')
                    or self._interlock_reason):
                raise RuntimeError('成对执行未允许或本次运行已结束')
            if self._paired_active:
                raise RuntimeError('正在成对执行，禁止重复提交')
            if self._preparation is not None:
                raise RuntimeError('后台预检尚未结束，不能开始执行')
            plan = self._paired_target
            if plan is None or time.monotonic() - plan.received > self.target_timeout:
                self._paired_target = None
                self._paired_phase = 'REJECTED: 暂存缺失或过期'
                raise RuntimeError('没有有效成对暂存计划，或计划已过期')
            if plan.details is not None:
                if expected_request_id != plan.details.get('request_id'):
                    raise RuntimeError('完整文件必须通过匹配预检编号的task_execute入口执行')
                self._check_pair_starts(plan)
            elif expected_request_id is not None:
                raise RuntimeError('当前暂存不是该编号的完整任务文件')
            self._paired_target = None
            gate = PairedTrajectoryStart()
            self._paired_gate = gate
            self._paired_active = True
            self._paired_phase = 'PREPARING'
            self._paired_requested_at = time.monotonic()
            try:
                for side in self.sides:
                    if plan.continuous is None:
                        identifiers[side] = self.workers[side].submit(
                            'trajectory', times_ns=plan.times_ns, positions=plan.positions[side],
                            gate=gate,
                        )
                    else:
                        identifiers[side] = self.workers[side].submit(
                            'trajectory', continuous=plan.continuous[side], gate=gate,
                        )
                self._paired_ids = dict(identifiers)
            except Exception as error:
                gate.cancel()
                failure = error
        if failure is not None:
            self._trip(f'成对轨迹未完整受理：{failure}')
            raise RuntimeError('成对请求未完整受理，已取消两臂') from failure
        return json.dumps(
            {'operation': 'paired_move', 'commands': identifiers}, ensure_ascii=False,
        )

    def check_paired_progress(self) -> None:
        """按请求编号确认两侧结果，任一拒绝、故障或丢失请求即联停."""
        with self._lock:
            if not self._paired_active:
                return
            identifiers = dict(self._paired_ids)
            requested = self._paired_requested_at
            gate = self._paired_gate
        if gate is None or gate.cancelled:
            self._trip('成对轨迹的共同启动门控已取消')
            return
        complete = True
        running = False
        for side, identifier in identifiers.items():
            snapshot = self.workers[side].snapshot()
            if snapshot.command_id != identifier:
                complete = False
                if time.monotonic() - requested > 1.0:
                    self._trip(f'{side} 成对轨迹请求未进入执行线程')
                    return
            elif (snapshot.phase in ('FAILED', 'REJECTED', 'CANCELLED')
                  or snapshot.state in ('STOPPED', 'FAULT', 'DISCONNECTED')):
                self._trip(f'{side} 成对轨迹失败：{snapshot.message}')
                return
            elif (snapshot.feedback is None or not 0 <= (
                time.perf_counter_ns() - snapshot.feedback.source_ns
            ) / 1e9 <= 0.1):
                self._trip(f'{side} 成对轨迹缺少新鲜反馈，已取消两侧')
                return
            elif snapshot.phase != 'SUCCEEDED' or snapshot.state != 'HOLDING':
                complete = False
                running = running or snapshot.state == 'RUNNING'
        with self._lock:
            if self._paired_active:
                self._paired_phase = (
                    'FINISHED' if complete else ('RUNNING' if running else 'PREPARING')
                )
                if complete:
                    self._paired_active = False

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
                raise RuntimeError('本次运行已结束，禁止暂存目标')
            if self._preparation is not None:
                raise RuntimeError('后台预检期间禁止暂存单臂目标')
            if self._paired_active:
                raise RuntimeError('成对轨迹期间禁止暂存单臂目标')
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
            self._paired_target = None
            self._paired_phase = 'IDLE'
            self._paired_details = {}
            self._file_request_id = ''
            self._file_rejection = {}
            self._target_messages[side] = 'STAGED：已预检，尚未执行'

    def publish_state(self) -> None:
        """发布独立反馈和可供晚加入订阅者读取的执行状态."""
        self.check_preparation_progress()
        self.check_paired_progress()
        states = {}
        with self._lock:
            interlock = self._interlock_reason
            targets = {side: side in self._targets for side in self.sides}
            target_messages = dict(self._target_messages)
            paired = {'active': self._paired_active, 'phase': self._paired_phase,
                      'staged': self._paired_target is not None,
                      'commands': dict(self._paired_ids),
                      'allowed': self.allow_paired_trajectory,
                      'continuous_allowed': self.allow_continuous_trajectory,
                      'task_files_allowed': self.allow_task_file_trajectory,
                      'task_file': dict(self._paired_details),
                      'request_id': self._file_request_id,
                      'last_file_rejection': dict(self._file_rejection),
                      'preflight_busy': self._preparation is not None}
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
                'readiness_stage': snapshot.readiness_stage,
                'allow_api_calibration': self.allow_api_calibration,
                'trajectory_start_at_monotonic': snapshot.trajectory_start_at,
                'trajectory_observed_start_at_monotonic': snapshot.trajectory_observed_start_at,
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
        text = json.dumps({'interlock': interlock, 'session_ended': bool(interlock),
                           'paired_trajectory': paired,
                           'arms': states}, ensure_ascii=False)
        now = time.monotonic()
        if text != self._last_status or now - self._last_status_time >= 1.0:
            status = String()
            status.data = text
            self.status_pub.publish(status)
            self._last_status = text
            self._last_status_time = now

    def close_workers(self) -> None:
        """两臂先同时请求停止，再分别等待线程退出."""
        with self._lock:
            self._cancel_preparation_locked(signal=False)
            self._paired_target = None
        for worker in self.workers.values():
            worker.request_stop()
        self._signal_preparation_cancel()
        errors = []
        for worker in self.workers.values():
            try:
                worker.shutdown()
            except RuntimeError as error:
                errors.append(str(error))
        if self._preparer is not None:
            self._preparer.shutdown(wait=True, cancel_futures=True)
        if self._preparation_manager is not None:
            self._preparation_manager.shutdown()
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
