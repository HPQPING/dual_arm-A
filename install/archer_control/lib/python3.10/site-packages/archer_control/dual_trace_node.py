"""只读观察 ROS 双臂仿真：发布世界坐标轨迹，逐次任务保存 CSV 和误差摘要。"""

from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import TextIO

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Point
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray
import yaml

from archer_mujoco.sim_node import prepare_plan, sample_plan
from .check_circle import validate_tool_chain
from .circle_planner import semicircle_points
from .dual_arm_trajectory import rigid_transform
from .kinematics import ArmKinematics
from .single_arm_node import number, numbers
from .trajectory import FloatArray, joint_names, validate_joints


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError('参数文件必须使用字符串键的 YAML 映射')
    return {str(key): item for key, item in value.items()}


class DualTrace(Node):
    def __init__(self) -> None:
        super().__init__('dual_trace')
        self.declare_parameter('config_file', '')
        self.declare_parameter('output_dir', str(Path.home() / 'archer_ws/dual_results'))
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        if self.get_parameter('use_sim_time').value is not False:
            raise ValueError('记录节点必须使用 use_sim_time:=false')
        source = self.get_parameter('config_file').value
        output = self.get_parameter('output_dir').value
        if not isinstance(source, str) or not source or not isinstance(output, str) or not output:
            raise ValueError('config_file 和 output_dir 必须是非空路径')
        with Path(source).expanduser().open(encoding='utf-8') as stream:
            raw: object = yaml.safe_load(stream)
        params = _mapping(_mapping(_mapping(raw)['dual_arm_task'])['ros__parameters'])
        if params.get('use_sim_time') is not False or params.get('circle_plane') != 'xz':
            raise ValueError('记录配置要求系统时间和 xz 圆平面')
        self.output = Path(output).expanduser()
        self.center = np.asarray(numbers(params['circle_center'], 'circle_center'), dtype=np.float64)
        self.radius = number(params['circle_radius'], 'circle_radius')
        left_arc, right_arc = semicircle_points(self.center, self.radius, 64)
        self.reference = np.concatenate((left_arc, right_arc[1:]), axis=0)
        self.speeds = numbers(params['speed_limits'], 'speed_limits')
        tcp = numbers(params['tcp_xyz'], 'tcp_xyz')
        mount = numbers(params['mount_xyz'], 'mount_xyz')
        rotation = numbers(params['mount_rpy'], 'mount_rpy')
        self.models: dict[str, ArmKinematics] = {}
        self.world: dict[str, FloatArray] = {}
        for side in ('left', 'right'):
            path = params.get(side + '_urdf', '')
            if not isinstance(path, str):
                raise ValueError('URDF 路径必须是字符串')
            model = (ArmKinematics.from_urdf(path, side) if path else
                     ArmKinematics.from_description(side, tcp, mount, rotation))
            validate_tool_chain(model)
            self.models[side] = model
            self.world[side] = rigid_transform(
                numbers(params[side + '_base_xyz'], side + '_base_xyz'),
                numbers(params[side + '_base_rpy'], side + '_base_rpy'))
        self.feedback: dict[str, tuple[int, FloatArray]] = {}
        self.plans: dict[str, tuple[FloatArray, FloatArray]] = {}
        self.traces: dict[str, list[FloatArray]] = {'left': [], 'right': []}
        self.last_saved: dict[str, int] = {}
        self.counts = {'left': 0, 'right': 0}
        self.maximum_tracking = {'left': 0., 'right': 0.}
        self.maximum_circle = {'left': 0., 'right': 0.}
        self.arc_counts = {'left': 0, 'right': 0}
        self.start_ns: int | None = None
        self.stream: TextIO | None = None
        self.directory: Path | None = None
        self.task_state = 'WAITING'
        self.last_display_ns = 0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, '/dual/tcp_markers', qos)
        self.status_pub = self.create_publisher(String, '/dual/trace/status', qos)
        self.subscriptions_kept = [
            self.create_subscription(JointState, '/left/joint_states', self.on_left_feedback, 10),
            self.create_subscription(JointState, '/right/joint_states', self.on_right_feedback, 10),
            self.create_subscription(JointTrajectory, '/left/trajectory_preview', self.on_left_command, qos),
            self.create_subscription(JointTrajectory, '/right/trajectory_preview', self.on_right_command, qos),
            self.create_subscription(String, '/dual/task/status', self.on_status, qos),
        ]
        self.timer = self.create_timer(.02, self.tick)
        self.publish_status('READY')

    def publish_status(self, text: str) -> None:
        message = String()
        message.data = text
        self.status_pub.publish(message)
        self.get_logger().info(text)

    def on_left_feedback(self, message: JointState) -> None:
        self.on_feedback('left', message)

    def on_right_feedback(self, message: JointState) -> None:
        self.on_feedback('right', message)

    def on_left_command(self, message: JointTrajectory) -> None:
        self.on_command('left', message)

    def on_right_command(self, message: JointTrajectory) -> None:
        self.on_command('right', message)

    def on_feedback(self, side: str, message: JointState) -> None:
        names = list(message.name)
        expected = joint_names(side)
        if len(names) != len(set(names)) or len(message.position) != len(names):
            return
        if not all(name in names for name in expected):
            return
        try:
            q = validate_joints([message.position[names.index(name)] for name in expected],
                                self.models[side].limits)
        except (ValueError, TypeError):
            return
        stamp = int(message.header.stamp.sec) * 10**9 + int(message.header.stamp.nanosec)
        if stamp <= 0:
            return
        previous = self.feedback.get(side)
        if previous is None or stamp > previous[0]:
            self.feedback[side] = (stamp, q)

    def on_command(self, side: str, message: JointTrajectory) -> None:
        feedback = self.feedback.get(side)
        now = self.get_clock().now().nanoseconds
        if feedback is None or not 0 <= now - feedback[0] <= 500_000_000:
            self.get_logger().warning(side + ' 缺少新鲜初始反馈，无法重建本次目标轨迹')
            return
        try:
            plan = prepare_plan(message, joint_names(side), feedback[1],
                                self.models[side].limits, self.speeds, allow_scheduled=True)
            stamp = int(message.header.stamp.sec) * 10**9 + int(message.header.stamp.nanosec)
            if stamp <= now:
                raise ValueError('记录节点只接受尚未开始的定时轨迹')
            if self.start_ns is not None and stamp != self.start_ns:
                raise ValueError('左右轨迹时间戳不一致，拒绝混合记录')
            if side in self.plans:
                raise ValueError('本次记录已收到该侧轨迹，拒绝覆盖')
            if self.stream is None:
                self.begin(stamp)
            self.plans[side] = plan
        except (ValueError, TypeError, OverflowError) as exc:
            self.get_logger().warning('无法记录轨迹：' + str(exc))

    def begin(self, stamp: int) -> None:
        self.start_ns = stamp
        self.directory = self.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stream = (self.directory / 'tracking.csv').open('w', newline='', encoding='utf-8-sig')
        header = ['feedback_stamp_ns', 'elapsed_s', 'side', 'phase']
        header += [f'actual_J{i}_rad' for i in range(1, 7)]
        header += ['target_x_m', 'target_y_m', 'target_z_m', 'actual_x_m', 'actual_y_m',
                   'actual_z_m', 'tracking_error_mm', 'distance_to_circle_mm']
        csv.writer(self.stream).writerow(header)
        self.publish_status('RECORDING: ' + str(self.directory))

    def on_status(self, message: String) -> None:
        state = message.data
        if state == 'PLANNING':
            self.finish('SUPERSEDED')
            self.plans.clear()
            self.start_ns = None
            self.traces = {'left': [], 'right': []}
            self.last_saved.clear()
            self.counts = {'left': 0, 'right': 0}
            self.arc_counts = {'left': 0, 'right': 0}
            self.maximum_tracking = {'left': 0., 'right': 0.}
            self.maximum_circle = {'left': 0., 'right': 0.}
        self.task_state = state
        if state == 'COMPLETED' or state.startswith(('FAILED', 'STOP_')):
            self.record()
            self.finish(state)

    def record(self) -> None:
        if self.stream is None or self.start_ns is None:
            return
        now = self.get_clock().now().nanoseconds
        for side, plan in self.plans.items():
            feedback = self.feedback.get(side)
            if feedback is None:
                continue
            stamp, q = feedback
            if stamp <= self.last_saved.get(side, 0) or not 0 <= now - stamp <= 500_000_000:
                continue
            elapsed = (stamp - self.start_ns) * 1e-9
            if elapsed < 0:
                continue
            target_q, _ = sample_plan(plan[0], plan[1], elapsed)
            actual = (self.world[side] @ self.models[side].forward(q))[:3, 3]
            desired = (self.world[side] @ self.models[side].forward(target_q))[:3, 3]
            error = float(np.linalg.norm(actual - desired)) * 1000.
            # 第一段为接近圆弧起点；圆距离统计只覆盖随后画圆及终点稳定阶段。
            on_arc = elapsed >= float(plan[0][1])
            circle_error: float | None = None
            if on_arc:
                radius = float(np.hypot(actual[0] - self.center[0], actual[2] - self.center[2]))
                circle_error = float(np.hypot(radius - self.radius, actual[1] - self.center[1])) * 1000.
                self.maximum_circle[side] = max(self.maximum_circle[side], circle_error)
                self.arc_counts[side] += 1
                self.traces[side].append(actual.copy())
            self.maximum_tracking[side] = max(self.maximum_tracking[side], error)
            self.counts[side] += 1
            self.last_saved[side] = stamp
            csv.writer(self.stream).writerow([stamp, elapsed, side, 'ARC' if on_arc else 'APPROACH',
                *[float(v) for v in q], *[float(v) for v in desired], *[float(v) for v in actual],
                error, circle_error])

    def tick(self) -> None:
        self.record()
        now = self.get_clock().now().nanoseconds
        if now - self.last_display_ns >= 100_000_000:
            self.last_display_ns = now
            self.publish_markers()
            if self.stream is not None:
                self.stream.flush()

    def publish_markers(self) -> None:
        messages: list[Marker] = []
        for index, points, rgb in (
            (0, list(self.reference), (0.8, 0.8, 0.8)),
            (1, self.traces['left'], (1., .45, .05)),
            (2, self.traces['right'], (0., .85, 1.)),
        ):
            marker = Marker()
            marker.header.frame_id = 'world'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'dual_tcp'; marker.id = index
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD if len(points) >= 2 else Marker.DELETE
            marker.pose.orientation.w = 1.
            marker.scale.x = .003
            marker.color.r, marker.color.g, marker.color.b = rgb
            marker.color.a = 1.
            # 均匀抽样仅限制 RViz 消息大小，CSV 仍保留全部已记录数据。
            indices = np.linspace(0, len(points) - 1, min(len(points), 1200), dtype=np.intp)
            marker.points = [Point(x=float(points[int(i)][0]), y=float(points[int(i)][1]),
                                   z=float(points[int(i)][2])) for i in indices]
            messages.append(marker)
        array = MarkerArray()
        array.markers = messages
        self.marker_pub.publish(array)

    def finish(self, state: str) -> None:
        if self.stream is None or self.directory is None:
            return
        self.stream.close()
        self.stream = None
        summary = {
            'task_state': state, 'start_stamp_ns': self.start_ns,
            'records': self.counts, 'arc_records': self.arc_counts,
            'maximum_tracking_error_mm': {
                side: self.maximum_tracking[side] if self.counts[side] else None for side in self.counts},
            'maximum_distance_to_circle_mm': {
                side: self.maximum_circle[side] if self.arc_counts[side] else None for side in self.counts},
            'source': '实际 TCP 由 ROS 关节反馈和模型正运动学计算，不是实机外部测量',
        }
        (self.directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                    encoding='utf-8')
        self.publish_status('SAVED: ' + str(self.directory))

    def destroy_node(self):
        self.record()
        self.finish('NODE_SHUTDOWN')
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DualTrace()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()