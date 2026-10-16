"""只读观察 ROS 双臂仿真：发布世界坐标轨迹，逐次任务保存 CSV 和误差摘要。"""

from __future__ import annotations

import csv
from datetime import datetime
from dataclasses import dataclass
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
from .dual_arm_trajectory import checked_pose, rigid_transform
from .cooperative_planner import interpolate_pose, pose_error
from .kinematics import ArmKinematics, rotation_error
from .single_arm_node import number, numbers
from .trajectory import FloatArray, joint_names, validate_joints


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError('参数文件必须使用字符串键的 YAML 映射')
    return {str(key): item for key, item in value.items()}


def _matrix(value: object) -> FloatArray:
    if not isinstance(value, list):
        raise ValueError('位姿必须是四行数值列表')
    rows = [numbers(row, '位姿行') for row in value]
    return checked_pose(rows)


@dataclass(frozen=True)
class TaskProfile:
    mode: str
    start_ns: int
    approach: float
    times: FloatArray
    objects: tuple[FloatArray, ...]
    left_grasp: FloatArray
    right_grasp: FloatArray

    def object_at(self, elapsed: float) -> FloatArray:
        if not self.objects:
            raise ValueError('当前任务没有物体参考路径')
        t = max(0., elapsed - self.approach)
        if t >= self.times[-1]:
            return self.objects[-1].copy()
        index = max(0, int(np.searchsorted(self.times, t, side='right')) - 1)
        u = float((t - self.times[index]) / (self.times[index + 1] - self.times[index]))
        progress = min(1., max(0., 6. * u**5 - 15. * u**4 + 10. * u**3))
        return interpolate_pose(self.objects[index], self.objects[index + 1], progress)


class DualTrace(Node):
    def __init__(self) -> None:
        super().__init__('dual_trace')
        self.declare_parameter('config_file', '')
        self.declare_parameter('output_dir', str(Path.home() / 'archer_ws/dual_results'))
        self.declare_parameter('max_pair_skew_s', .005)
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
        self.max_pair_skew = number(self.get_parameter('max_pair_skew_s').value, 'max_pair_skew_s')
        if not 0. <= self.max_pair_skew <= .05:
            raise ValueError('反馈配对允许时差必须为 0～0.05 s')
        self.geometry = {name: numbers(params[name], name) for name in (
            'left_base_xyz', 'right_base_xyz', 'left_base_rpy', 'right_base_rpy',
            'tcp_xyz', 'mount_xyz', 'mount_rpy')}
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
        self.maximum_object = {'left': 0., 'right': 0.}
        self.profile: TaskProfile | None = None
        self.pending: dict[str, tuple[str, JointTrajectory]] = {}
        self.relative_stream: TextIO | None = None
        self.pair_count = 0
        self.maximum_relative = [0., 0.]
        self.last_pair = (0, 0)
        self.visual_elapsed = 0.
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
            self.create_subscription(JointTrajectory, '/left/cooperative_preview', self.on_left_cooperative, qos),
            self.create_subscription(JointTrajectory, '/right/cooperative_preview', self.on_right_cooperative, qos),
            self.create_subscription(String, '/dual/task/profile', self.on_profile, qos),
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
        self.on_command('left', message, 'split_circle')

    def on_right_command(self, message: JointTrajectory) -> None:
        self.on_command('right', message, 'split_circle')

    def on_left_cooperative(self, message: JointTrajectory) -> None:
        self.on_command('left', message, 'cooperative')

    def on_right_cooperative(self, message: JointTrajectory) -> None:
        self.on_command('right', message, 'cooperative')

    def on_profile(self, message: String) -> None:
        try:
            data = _mapping(json.loads(message.data))
            mode, stamp = data['task_type'], data['start_stamp_ns']
            if data.get('version') != 1 or mode not in ('split_circle', 'cooperative'):
                raise ValueError('任务说明的版本或类型不受支持')
            if isinstance(stamp, bool) or not isinstance(stamp, int) or stamp <= 0:
                raise ValueError('任务说明的起始时间无效')
            geometry = _mapping(data['geometry'])
            for name, expected in self.geometry.items():
                actual = numbers(geometry[name], name)
                if len(actual) != 3 or not np.allclose(actual, expected, atol=1e-9, rtol=0.):
                    raise ValueError('记录模型与任务模型不一致：' + name)
            speeds = numbers(data['speed_limits'], 'speed_limits')
            if len(speeds) != 6 or min(speeds) <= 0.:
                raise ValueError('任务速度限制无效')
            approach = number(data['approach_duration_s'], 'approach_duration_s')
            if approach < 0.:
                raise ValueError('接近时间不能为负数')
            poses: tuple[FloatArray, ...] = ()
            times = np.zeros(0, dtype=np.float64)
            left, right = np.eye(4, dtype=np.float64), np.eye(4, dtype=np.float64)
            center, radius = self.center.copy(), self.radius
            if mode == 'cooperative':
                raw_poses = data['world_object_poses']
                if not isinstance(raw_poses, list) or not 2 <= len(raw_poses) <= 1000:
                    raise ValueError('物体路径长度无效')
                poses = tuple(_matrix(pose) for pose in raw_poses)
                times = np.asarray(numbers(data['object_times_s'], 'object_times_s'), dtype=np.float64)
                if len(times) != len(poses) or times[0] != 0. or np.any(np.diff(times) <= 0.):
                    raise ValueError('物体路径与分段时间不一致')
                left, right = _matrix(data['object_from_left_tcp']), _matrix(data['object_from_right_tcp'])
                reference = np.asarray([pose[:3, 3] for pose in poses], dtype=np.float64)
            else:
                center = np.asarray(numbers(data['circle_center'], 'circle_center'), dtype=np.float64)
                radius = number(data['circle_radius'], 'circle_radius')
                a, b = semicircle_points(center, radius, 64)
                reference = np.asarray(np.concatenate((a, b[1:]), axis=0), dtype=np.float64)
            profile = TaskProfile(str(mode), stamp, approach, times, poses, left, right)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            self.get_logger().error('任务说明无法用于记录：' + str(exc))
            return
        if self.profile is not None and self.profile.start_ns == stamp:
            return
        self.finish('SUPERSEDED')
        self.profile = profile
        self.start_ns = None
        self.plans.clear()
        self.traces = {'left': [], 'right': []}
        self.last_saved.clear()
        self.counts = {'left': 0, 'right': 0}
        self.arc_counts = {'left': 0, 'right': 0}
        self.maximum_tracking = {'left': 0., 'right': 0.}
        self.maximum_circle = {'left': 0., 'right': 0.}
        self.maximum_object = {'left': 0., 'right': 0.}
        self.maximum_relative = [0., 0.]
        self.pair_count = 0
        self.last_pair = (0, 0)
        self.visual_elapsed = max(0., (self.get_clock().now().nanoseconds - stamp) * 1e-9)
        self.reference = reference
        self.speeds = speeds
        if mode == 'split_circle':
            self.center, self.radius = center, radius
        # 不同话题之间可能先收到预览，说明到达后再处理；最多缓存两侧各一条。
        cached = self.pending.copy()
        self.pending.clear()
        for side, (kind, trajectory) in cached.items():
            self.on_command(side, trajectory, kind)

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

    def on_command(self, side: str, message: JointTrajectory, mode: str = 'split_circle') -> None:
        stamp = int(message.header.stamp.sec) * 10**9 + int(message.header.stamp.nanosec)
        profile = self.profile
        if profile is None or profile.start_ns != stamp:
            self.pending[side] = (mode, message)
            return
        if profile.mode != mode:
            self.get_logger().warning('预览话题与任务类型不一致，拒绝混合记录')
            return
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
        header += ['object_tcp_error_mm']
        csv.writer(self.stream).writerow(header)
        if self.profile is not None and self.profile.mode == 'cooperative':
            self.relative_stream = (self.directory / 'relative.csv').open('w', newline='', encoding='utf-8-sig')
            csv.writer(self.relative_stream).writerow([
                'left_stamp_ns', 'right_stamp_ns', 'elapsed_s', 'feedback_skew_ms',
                'relative_position_error_mm', 'relative_rotation_error_rad'])
        self.publish_status('RECORDING: ' + str(self.directory))

    def on_status(self, message: String) -> None:
        state = message.data
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
            profile = self.profile
            if profile is None:
                return
            cooperative = profile.mode == 'cooperative'
            boundary = profile.approach if cooperative else float(plan[0][1])
            on_arc = elapsed >= boundary
            object_error: float | None = None
            phase = 'APPROACH' if not on_arc else ('COOPERATIVE' if cooperative else 'ARC')
            self.visual_elapsed = elapsed
            if cooperative and on_arc:
                grasp = profile.left_grasp if side == 'left' else profile.right_grasp
                object_error = float(np.linalg.norm(actual - (profile.object_at(elapsed) @ grasp)[:3, 3])) * 1000.
                self.maximum_object[side] = max(self.maximum_object[side], object_error)
            circle_error: float | None = None
            if on_arc and not cooperative:
                radius = float(np.hypot(actual[0] - self.center[0], actual[2] - self.center[2]))
                circle_error = float(np.hypot(radius - self.radius, actual[1] - self.center[1])) * 1000.
                self.maximum_circle[side] = max(self.maximum_circle[side], circle_error)
            if on_arc:
                self.arc_counts[side] += 1
                self.traces[side].append(actual.copy())
            self.maximum_tracking[side] = max(self.maximum_tracking[side], error)
            self.counts[side] += 1
            self.last_saved[side] = stamp
            csv.writer(self.stream).writerow([stamp, elapsed, side, phase,
                *[float(v) for v in q], *[float(v) for v in desired], *[float(v) for v in actual],
                error, circle_error, object_error])
        self.record_relative()

    def record_relative(self) -> None:
        profile = self.profile
        if profile is None or profile.mode != 'cooperative' or self.relative_stream is None:
            return
        if 'left' not in self.feedback or 'right' not in self.feedback:
            return
        left_stamp, left_q = self.feedback['left']
        right_stamp, right_q = self.feedback['right']
        pair = (left_stamp, right_stamp)
        now = self.get_clock().now().nanoseconds
        if pair == self.last_pair or not all(0 <= now - t <= 500_000_000 for t in pair):
            return
        # 不把相隔太久的两帧误当成同一时刻；CSV 明确记录实际时间差。
        skew = abs(left_stamp - right_stamp) * 1e-9
        if skew > self.max_pair_skew:
            return
        elapsed = ((left_stamp - profile.start_ns) + (right_stamp - profile.start_ns)) * .5e-9
        if min(left_stamp, right_stamp) < profile.start_ns + int(round(profile.approach * 1e9)):
            return
        left = self.world['left'] @ self.models['left'].forward(left_q)
        right = self.world['right'] @ self.models['right'].forward(right_q)
        expected = np.linalg.inv(profile.left_grasp) @ profile.right_grasp
        distance, angle = pose_error(np.linalg.inv(left) @ right, expected)
        self.maximum_relative[0] = max(self.maximum_relative[0], distance * 1000.)
        self.maximum_relative[1] = max(self.maximum_relative[1], angle)
        self.pair_count += 1
        self.last_pair = pair
        csv.writer(self.relative_stream).writerow([left_stamp, right_stamp, elapsed, skew * 1000.,
                                                   distance * 1000., angle])

    def tick(self) -> None:
        self.record()
        now = self.get_clock().now().nanoseconds
        if now - self.last_display_ns >= 100_000_000:
            self.last_display_ns = now
            self.publish_markers()
            if self.stream is not None:
                self.stream.flush()
            if self.relative_stream is not None:
                self.relative_stream.flush()

    def publish_markers(self) -> None:
        messages: list[Marker] = []
        profile = self.profile
        cooperative = profile is not None and profile.mode == 'cooperative'
        for index, points, rgb in (
            (0, list(self.reference), (0.2, 1., .3) if cooperative else (0.8, 0.8, 0.8)),
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
        cube = Marker()
        cube.header.frame_id = 'world'
        cube.header.stamp = self.get_clock().now().to_msg()
        cube.ns = 'dual_tcp'; cube.id = 3
        cube.type = Marker.CUBE
        cube.action = Marker.ADD if cooperative else Marker.DELETE
        if cooperative and profile is not None:
            pose = profile.object_at(self.visual_elapsed)
            cube.pose.position = Point(x=float(pose[0, 3]), y=float(pose[1, 3]), z=float(pose[2, 3]))
            vector = rotation_error(pose[:3, :3], np.eye(3, dtype=np.float64))
            angle = float(np.linalg.norm(vector))
            quaternion_xyz = np.zeros(3, dtype=np.float64) if angle < 1e-12 else vector / angle * np.sin(angle / 2.)
            cube.pose.orientation.x = float(quaternion_xyz[0])
            cube.pose.orientation.y = float(quaternion_xyz[1])
            cube.pose.orientation.z = float(quaternion_xyz[2])
            cube.pose.orientation.w = float(np.cos(angle / 2.))
            # 仅为 4×4×40 cm 的参考示意，不参与 MuJoCo 接触或碰撞计算。
            cube.scale.x = .04; cube.scale.y = .04; cube.scale.z = .40
            cube.color.r = .2; cube.color.g = 1.; cube.color.b = .3; cube.color.a = .25
        messages.append(cube)
        array = MarkerArray()
        array.markers = messages
        self.marker_pub.publish(array)

    def finish(self, state: str) -> None:
        if self.stream is None or self.directory is None:
            return
        self.stream.close()
        self.stream = None
        if self.relative_stream is not None:
            self.relative_stream.close()
            self.relative_stream = None
        profile = self.profile
        cooperative = profile is not None and profile.mode == 'cooperative'
        summary = {
            'task_state': state, 'start_stamp_ns': self.start_ns,
            'task_type': profile.mode if profile is not None else None,
            'records': self.counts,
            'arc_records': {side: self.arc_counts[side] if not cooperative else 0 for side in self.counts},
            'cooperative_records': {side: self.arc_counts[side] if cooperative else 0 for side in self.counts},
            'maximum_tracking_error_mm': {
                side: self.maximum_tracking[side] if self.counts[side] else None for side in self.counts},
            'maximum_distance_to_circle_mm': {
                side: self.maximum_circle[side] if self.arc_counts[side] and not cooperative else None for side in self.counts},
            'maximum_object_tcp_error_mm': {
                side: self.maximum_object[side] if cooperative and self.arc_counts[side] else None for side in self.counts},
            'paired_records': self.pair_count,
            'maximum_relative_position_error_mm': self.maximum_relative[0] if self.pair_count else None,
            'maximum_relative_rotation_error_rad': self.maximum_relative[1] if self.pair_count else None,
            'max_pair_skew_s': self.max_pair_skew,
            'reference_object_only': cooperative,
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