"""ROS 2 双臂仿真任务节点：后台规划、跨臂预检、共同时间戳和执行反馈。

使用 python3 -m archer_control.dual_arm_node 启动；默认不运动，需调用开始服务。
本节点只用于共享 MuJoCo 仿真，不连接实机，不实现力控制。
"""

from __future__ import annotations

import time
import json
from threading import Event
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import cast

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory

from .dual_arm_trajectory import (
    DualArmKinematics, DualArmPlan, rigid_transform, peak_target_speeds, plan_paired_joints,
    append_return_home,
)
from .kinematics import ArmKinematics
from .circle_planner import plan_circle, semicircle_points, inward_rotation
from .model_checks import CollisionModel, validate_tool_chain, cross_arm_failure
from .circle_sim import preflight
from archer_mujoco.dual_model_loader import load_dual_arm
from archer_mujoco.model_loader import mujoco
from .cooperative_planner import CooperativePlan, interpolate_pose, plan_cooperative, sample_errors, sample_object, pose_error
from .parameter_validation import number, numbers
from .trajectory import FloatArray, JointPlan, joint_names, sample_joint_plan, to_ros_trajectory, validate_joints


@dataclass(frozen=True)
class PreparedTask:
    paired: DualArmPlan
    approach_duration: float = 0.
    cooperative: CooperativePlan | None = None
    return_home_start: float | None = None


class DualArmTask(Node):
    def __init__(self) -> None:
        super().__init__('dual_arm_task')
        defaults = {
            'task_type': 'split_circle', 'start_delay': 3.0, 'feedback_timeout': 0.5,
            'minimum_duration': 0.20, 'time_margin': 1.2,
            'goal_tolerance': 0.01, 'start_velocity_limit': 0.01,
            'left_base_xyz': [-0.21, 0.0, 0.0], 'right_base_xyz': [0.21, 0.0, 0.0],
            'left_base_rpy': [0.0, 0.0, float(np.pi / 2)], 'right_base_rpy': [0.0, 0.0, float(np.pi / 2)],
            'tcp_xyz': [0.0, 0.0, 0.147021696], 'mount_xyz': [0.0, 0.0, 0.0],
            'mount_rpy': [0.0, 0.0, 0.0],
            'speed_limits': [0.1, 0.5, 0.5, 0.5, 0.5, 0.5],
            'circle_center': [0.0, 0.40, 0.31], 'circle_radius': 0.20,
            'circle_plane': 'xz', 'arc_samples': 64, 'inward_tilt_deg': 10.0,
            'clearance': 0.01, 'left_urdf': '', 'right_urdf': '',
            'object_start_xyz': [0., .40, .31], 'object_start_rpy': [0., 0., 0.],
            'object_goal_xyz': [0., .42, .32], 'object_goal_rpy': [0., 0., float(np.deg2rad(5.))],
            'left_grasp_xyz': [0., 0., .20], 'right_grasp_xyz': [0., 0., -.20],
            'cooperative_samples': 32, 'cooperative_position_tolerance': .01,
            'cooperative_rotation_tolerance': .03,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        if self.get_parameter('use_sim_time').value is not False:
            raise ValueError('Use use_sim_time:=false; all nodes must share the same system clock')
        task_type = self.get_parameter('task_type').value
        if task_type not in ('split_circle', 'cooperative'):
            raise ValueError('task_type 必须为 split_circle 或 cooperative')
        self.task_type = task_type
        self.circle_center = np.asarray(self.read_vector('circle_center', 3), dtype=np.float64)
        self.circle_radius = self.read_number('circle_radius')
        plane = self.get_parameter('circle_plane').value
        arc_samples = self.get_parameter('arc_samples').value
        if plane != 'xz' or isinstance(arc_samples, bool) or not isinstance(arc_samples, int):
            raise ValueError('当前半圆任务只使用竖直 xz 平面，采样数必须是整数')
        self.arc_samples = arc_samples
        semicircle_points(self.circle_center, self.circle_radius, arc_samples)
        self.inward_tilt = np.deg2rad(self.read_number('inward_tilt_deg'))
        if not 0. <= self.inward_tilt <= np.pi / 4:
            raise ValueError('夹爪内倾角必须在 0～45 度内')
        self.clearance = self.read_number('clearance')
        if not .001 <= self.clearance <= .05:
            raise ValueError('跨臂检查间隙必须在 0.001～0.05 m 内')
        self.delay = self.read_number('start_delay')
        self.feedback_timeout = self.read_number('feedback_timeout')
        self.minimum = self.read_number('minimum_duration')
        self.margin = self.read_number('time_margin')
        self.tolerance = self.read_number('goal_tolerance')
        self.velocity_limit = self.read_number('start_velocity_limit')
        if not 1.0 <= self.delay <= 20.0 or self.margin < 1.0:
            raise ValueError('start_delay must be 1 to 20 seconds; time_margin must be >= 1')
        if min(self.feedback_timeout, self.minimum, self.tolerance, self.velocity_limit) <= 0:
            raise ValueError('Timeout, duration and tolerances must be positive')
        self.speeds = self.read_vector('speed_limits', 6)
        if min(self.speeds) <= 0:
            raise ValueError('Speed limits must be positive')
        tcp = self.read_vector('tcp_xyz', 3)
        mount = self.read_vector('mount_xyz', 3)
        mount_rpy = self.read_vector('mount_rpy', 3)
        self.geometry = {name: self.read_vector(name, 3) for name in (
            'left_base_xyz', 'right_base_xyz', 'left_base_rpy', 'right_base_rpy',
            'tcp_xyz', 'mount_xyz', 'mount_rpy')}
        self.paths: dict[str, str | None] = {}
        for side in ('left', 'right'):
            value = self.get_parameter(side + '_urdf').value
            if not isinstance(value, str):
                raise ValueError('URDF 路径必须是字符串')
            self.paths[side] = value or None
        left = (ArmKinematics.from_urdf(self.paths['left'], 'left') if self.paths['left'] else
                ArmKinematics.from_description('left', tcp, mount, mount_rpy))
        right = (ArmKinematics.from_urdf(self.paths['right'], 'right') if self.paths['right'] else
                 ArmKinematics.from_description('right', tcp, mount, mount_rpy))
        validate_tool_chain(left)
        validate_tool_chain(right)
        self.scene_matches = False
        self.models = {'left': left, 'right': right}
        self.kinematics = DualArmKinematics(
            left, right,
            rigid_transform(self.read_vector('left_base_xyz', 3), self.read_vector('left_base_rpy', 3)),
            rigid_transform(self.read_vector('right_base_xyz', 3), self.read_vector('right_base_rpy', 3)),
        )
        self.object_start = rigid_transform(self.read_vector('object_start_xyz', 3), self.read_vector('object_start_rpy', 3))
        self.object_goal = rigid_transform(self.read_vector('object_goal_xyz', 3), self.read_vector('object_goal_rpy', 3))
        self.grasps: dict[str, FloatArray] = {}
        for side in ('left', 'right'):
            grasp = rigid_transform(self.read_vector(side + '_grasp_xyz', 3), [0., 0., 0.])
            grasp[:3, :3] = inward_rotation(side, float(self.inward_tilt))
            self.grasps[side] = grasp
        samples = self.get_parameter('cooperative_samples').value
        if isinstance(samples, bool) or not isinstance(samples, int) or not 16 <= samples <= 128:
            raise ValueError('cooperative_samples 必须是 16～128 的整数')
        self.cooperative_samples = samples
        self.cooperative_position_tolerance = self.read_number('cooperative_position_tolerance')
        self.cooperative_rotation_tolerance = self.read_number('cooperative_rotation_tolerance')
        if min(self.cooperative_position_tolerance, self.cooperative_rotation_tolerance) <= 0.:
            raise ValueError('协同执行误差阈值必须为正数')
        self.approach_duration = 0.
        self.return_home_start: float | None = None
        self.cooperative_plan: CooperativePlan | None = None
        self.stage = ''
        self.current: dict[str, FloatArray] = {}
        self.velocities: dict[str, float] = {}
        self.feedback_times: dict[str, float] = {}
        self.backend_status: dict[str, str] = {}
        self.accepted: set[str] = set()
        self.finished_arms: set[str] = set()
        self.plan: DualArmPlan | None = None
        self.planning: Future[PreparedTask] | None = None
        self.planning_started = 0.0
        self.cancel_event = Event()
        self.cancelled_work: Future[PreparedTask] | None = None
        self.planner = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dual_ik')
        self.start_ns: int | None = None
        self.last_clock_ns: int | None = None
        self.published_at = 0.0
        self.expected_end = 0.0
        self.goal_since: float | None = None
        self.start_positions: dict[str, FloatArray] = {}
        self.observed_motion: dict[str, float] = {}
        self.required_motion: dict[str, float] = {}
        self.state = 'WAITING_FOR_FEEDBACK'
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, '/dual/task/status', qos)
        self.stage_pub = self.create_publisher(String, '/dual/task/stage', qos)
        self.profile_pub = self.create_publisher(String, '/dual/task/profile', qos)
        self.commands = {
            side: self.create_publisher(JointTrajectory, '/' + side + '/joint_trajectory', 10)
            for side in ('left', 'right')
        }
        # 观察者使用独立预览话题，不占用控制话题的唯一后端订阅名额。
        self.previews = {
            side: self.create_publisher(JointTrajectory, '/' + side + (
                '/cooperative_preview' if self.task_type == 'cooperative' else '/trajectory_preview'), qos)
            for side in ('left', 'right')
        }
        self.stop_clients = {
            side: self.create_client(Trigger, '/' + side + '/stop') for side in ('left', 'right')
        }
        self.subscriptions_kept = [
            self.create_subscription(
                JointState, '/left/joint_states', self.on_left_feedback, 10,
            ),
            self.create_subscription(
                JointState, '/right/joint_states', self.on_right_feedback, 10,
            ),
            self.create_subscription(
                String, '/left/trajectory_status', self.on_left_backend_status, qos,
            ),
            self.create_subscription(
                String, '/right/trajectory_status', self.on_right_backend_status, qos,
            ),
        ]
        self.config_sub = self.create_subscription(String, '/dual/sim/config', self.on_scene_config, qos)
        self.start_service = self.create_service(Trigger, '/dual/task/start', self.on_start)
        self.stop_service = self.create_service(Trigger, '/dual/task/stop', self.on_stop)
        self.timer = self.create_timer(0.02, self.monitor)
        self.set_status('WAITING_FOR_FEEDBACK')
        self.get_logger().info('任务类型：' + self.task_type + '；等待开始服务，不自动运动。')

    def on_left_feedback(self, message: JointState) -> None:
        self.on_feedback('left', message)

    def on_right_feedback(self, message: JointState) -> None:
        self.on_feedback('right', message)

    def on_left_backend_status(self, message: String) -> None:
        self.on_backend_status('left', message)

    def on_right_backend_status(self, message: String) -> None:
        self.on_backend_status('right', message)

    def read_number(self, name: str) -> float:
        return number(self.get_parameter(name).value, name)

    def read_vector(self, name: str, length: int) -> list[float]:
        result = numbers(self.get_parameter(name).value, name)
        if len(result) != length:
            raise ValueError(f'{name} needs {length} numbers')
        return result

    def set_status(self, state: str) -> None:
        self.state = state
        message = String()
        message.data = state
        self.status_pub.publish(message)
        self.get_logger().info(state)
        if state != 'RUNNING':
            self.set_stage(state)

    def set_stage(self, stage: str) -> None:
        if stage != self.stage:
            self.stage = stage
            message = String()
            message.data = stage
            self.stage_pub.publish(message)

    def on_feedback(self, side: str, message: JointState) -> None:
        names = list(message.name)
        expected = joint_names(side)
        if len(names) != len(set(names)) or not all(name in names for name in expected):
            return
        if len(message.position) != len(names) or len(message.velocity) != len(names):
            return
        indices = [names.index(name) for name in expected]
        try:
            q = validate_joints([message.position[index] for index in indices], self.models[side].limits)
            velocity = np.asarray([message.velocity[index] for index in indices], dtype=np.float64)
            if not np.all(np.isfinite(velocity)):
                return
        except (ValueError, TypeError):
            return
        self.current[side] = q
        self.velocities[side] = float(np.max(np.abs(velocity)))
        self.feedback_times[side] = time.monotonic()

    def feedback_fresh(self) -> bool:
        now = time.monotonic()
        return all(side in self.current and now - self.feedback_times.get(side, -float('inf')) <= self.feedback_timeout
                   for side in ('left', 'right'))

    def on_backend_status(self, side: str, message: String) -> None:
        status = message.data
        self.backend_status[side] = status
        if self.plan is None or self.start_ns is None:
            return
        if status in (f'ACCEPTED:{self.start_ns}', f'RUNNING:{self.start_ns}'):
            self.accepted.add(side)
        elif status == f'FINISHED:{self.start_ns}':
            self.finished_arms.add(side)
        elif status.startswith(('REJECTED', 'STOPPED')):
            self.fail(f'{side}: {status}')

    def on_scene_config(self, message: String) -> None:
        try:
            config = json.loads(message.data)
            geometry = config['geometry']
            if config['backend'] != 'archer_shared_mujoco_v2':
                raise ValueError('后端不是本轮共享双臂仿真节点')
            for name, expected in self.geometry.items():
                actual = np.asarray(geometry[name], dtype=np.float64)
                if actual.shape != (3,) or not np.allclose(actual, expected, atol=1e-9, rtol=0.):
                    raise ValueError('任务与仿真模型参数不一致：' + name)
            if not np.isclose(float(config['clearance']), self.clearance, atol=1e-9, rtol=0.):
                raise ValueError('任务与仿真的跨臂间隙阈值不一致')
        except (ValueError, TypeError, KeyError) as exc:
            self.scene_matches = False
            self.get_logger().error(str(exc))
            if self.plan is not None or self.planning is not None:
                self.fail('仿真配置改变或不匹配')
            return
        self.scene_matches = True

    def build_plan(self, starts: dict[str, FloatArray] | None = None) -> PreparedTask:
        # 后台线程仅使用启动时保存的配置和关节快照，不调用 ROS 参数或发布接口。
        snapshot = self.current if starts is None else starts
        left_start, right_start = snapshot['left'].copy(), snapshot['right'].copy()
        if self.task_type == 'cooperative':
            return self.build_cooperative(left_start, right_start)
        result = plan_circle(
            self.kinematics, left_start, right_start, self.circle_center, self.circle_radius,
            self.arc_samples, float(self.inward_tilt), self.minimum, self.margin,
            cancelled=self.cancel_event.is_set,
        )
        if self.cancel_event.is_set():
            raise ValueError('规划已取消')
        return_start = result.plan.duration
        result = replace(result, plan=append_return_home(
            result.plan, self.kinematics.left.limits, self.kinematics.right.limits,
            self.speeds, self.speeds, self.minimum, self.margin,
        ))
        for arm_plan in (result.plan.left, result.plan.right):
            if np.any(peak_target_speeds(arm_plan) > np.asarray(self.speeds) + 1e-10):
                raise ValueError('规划速度超出任务 speed_limits，请提高 time_margin 后重试')
        loaded = load_dual_arm(left_urdf=self.paths['left'], right_urdf=self.paths['right'],
            left_base_xyz=self.geometry['left_base_xyz'], right_base_xyz=self.geometry['right_base_xyz'],
            left_base_rpy=self.geometry['left_base_rpy'], right_base_rpy=self.geometry['right_base_rpy'],
            tcp_xyz=self.geometry['tcp_xyz'], mount_xyz=self.geometry['mount_xyz'],
            mount_rpy=self.geometry['mount_rpy'], enable_contacts=True)
        preflight(loaded, result, self.clearance)
        if self.cancel_event.is_set():
            raise ValueError('规划已取消')
        return PreparedTask(
            result.plan, approach_duration=float(result.plan.left.times[result.arc_first_index]),
            return_home_start=return_start,
        )

    def build_cooperative(self, left_start: FloatArray, right_start: FloatArray) -> PreparedTask:
        """先空载接近示例抓取位姿，再拼接固定抓取关系的协同移动。

        本阶段不夹紧、不创建物体刚性连接；接近段不约束物体抓取关系。
        """
        starts = {'left': left_start, 'right': right_start}
        bases = {'left': self.kinematics.world_from_left, 'right': self.kinematics.world_from_right}
        seeds: dict[str, FloatArray] = {}
        for side, random_seed in (('left', 7), ('right', 8)):
            arm = self.models[side]
            target = np.linalg.inv(bases[side]) @ self.object_start @ self.grasps[side]
            rng = np.random.default_rng(random_seed)
            candidates: list[FloatArray] = []
            for index in range(33):
                if self.cancel_event.is_set():
                    raise ValueError('协同规划已取消')
                seed = starts[side] if index == 0 else rng.uniform(arm.limits[:, 0], arm.limits[:, 1])
                solution = arm.solve_ik(target, seed)
                if solution.success:
                    candidates.append(solution.joints.copy())
            if not candidates:
                raise ValueError(side + ' 示例抓取位姿未找到 IK 解')
            seeds[side] = min(candidates, key=lambda q: float(np.max(
                np.abs(q - starts[side]) / np.asarray(self.speeds))))
        approach = plan_paired_joints(
            left_start, right_start, [seeds['left']], [seeds['right']], self.speeds, self.speeds,
            self.models['left'].limits, self.models['right'].limits,
            minimum_duration=max(2., self.minimum), time_margin=self.margin)
        objects = [interpolate_pose(self.object_start, self.object_goal, float(u))
                   for u in np.linspace(0., 1., self.cooperative_samples + 1)]
        cooperative = plan_cooperative(self.kinematics, seeds['left'], seeds['right'], objects,
            self.grasps['left'], self.grasps['right'], minimum_duration=self.minimum, time_margin=self.margin)
        plans: list[JointPlan] = []
        for first, second in ((approach.left, cooperative.paired.left), (approach.right, cooperative.paired.right)):
            times = np.asarray(np.concatenate((first.times, approach.duration + second.times[1:])), dtype=np.float64)
            positions = np.asarray(np.concatenate((first.positions, second.positions[1:]), axis=0), dtype=np.float64)
            plans.append(JointPlan(first.joint_names, times, positions))
        paired = DualArmPlan(plans[0], plans[1])
        if any(np.any(peak_target_speeds(p) > np.asarray(self.speeds) + 1e-10) for p in plans):
            raise ValueError('协同目标速度超限，请提高 time_margin')
        loaded = load_dual_arm(left_urdf=self.paths['left'], right_urdf=self.paths['right'],
            left_base_xyz=self.geometry['left_base_xyz'], right_base_xyz=self.geometry['right_base_xyz'],
            left_base_rpy=self.geometry['left_base_rpy'], right_base_rpy=self.geometry['right_base_rpy'],
            tcp_xyz=self.geometry['tcp_xyz'], mount_xyz=self.geometry['mount_xyz'],
            mount_rpy=self.geometry['mount_rpy'], enable_contacts=True)
        cast(CollisionModel, loaded.model).geom_margin[:] = self.clearance * .6
        times_to_check = set(float(t) for t in paired.left.times)
        times_to_check.update(float(t) for t in np.linspace(0., paired.duration,
                              int(np.ceil(paired.duration / .02)) + 1))
        for elapsed in sorted(times_to_check):
            if self.cancel_event.is_set():
                raise ValueError('协同预检已取消')
            for side, arm_plan in (('left', paired.left), ('right', paired.right)):
                loaded.data.qpos[loaded.arms[side].qpos_indices] = sample_joint_plan(arm_plan, elapsed)[0]
            mujoco.mj_forward(loaded.model, loaded.data)
            failure = cross_arm_failure(loaded, self.clearance)
            if failure is not None:
                raise ValueError(f'接近或协同路径跨臂间隙不足，t={elapsed:.3f}s：{failure}')
            if elapsed >= approach.duration:
                errors = sample_errors(self.kinematics, cooperative, elapsed - approach.duration)
                if max(errors['world_position_m'], errors['relative_position_m']) > .0005:
                    raise ValueError('协同路径采样位置误差超过 0.5 mm')
                if max(errors['world_rotation_rad'], errors['relative_rotation_rad']) > .003:
                    raise ValueError('协同路径采样姿态误差超过 0.003 rad')
        return PreparedTask(paired, approach.duration, cooperative)

    def on_start(self, request, response):
        if self.plan is not None or self.planning is not None or (
                self.cancelled_work is not None and not self.cancelled_work.done()):
            response.success = False
            response.message = 'A paired task is already active'
            return response
        if (not self.scene_matches or self.count_publishers('/dual/sim/config') != 1):
            response.success = False
            response.message = '需唯一共享仿真后端，且配置必须一致'
            return response
        if (not self.feedback_fresh() or any(self.velocities.get(side, float('inf')) > self.velocity_limit
                                            for side in ('left', 'right'))):
            response.success = False
            response.message = 'Both arms need fresh stationary feedback'
            return response
        if any(self.commands[side].get_subscription_count() != 1
               or not (self.backend_status.get(side) == 'READY'
                       or self.backend_status.get(side, '').startswith('FINISHED:'))
               or not self.stop_clients[side].service_is_ready() for side in ('left', 'right')):
            response.success = False
            response.message = 'Require exactly one ready simulator per arm, with status and stop service'
            return response
        snapshot = {side: self.current[side].copy() for side in ('left', 'right')}
        self.cancel_event = Event()
        self.cooperative_plan = None
        self.approach_duration = 0.
        self.return_home_start: float | None = None
        self.planning_started = time.monotonic()
        self.planning = self.planner.submit(self.build_plan, snapshot)
        self.set_status('PLANNING')
        response.success = True
        response.message = '后台规划与跨臂预检已开始；尚未发送轨迹，请查看 /dual/task/status'
        return response

    def dispatch_plan(self, plan: DualArmPlan) -> None:
        if not self.feedback_fresh() or not self.scene_matches:
            raise ValueError('规划期间关节反馈过期或仿真配置不匹配')
        for side, arm_plan in (('left', plan.left), ('right', plan.right)):
            if (self.velocities.get(side, float('inf')) > self.velocity_limit
                    or float(np.max(np.abs(self.current[side] - arm_plan.positions[0]))) > 0.002):
                raise ValueError(side + ' moved during planning; trajectory not sent')
            if (self.commands[side].get_subscription_count() != 1
                    or not self.stop_clients[side].service_is_ready()
                    or not (self.backend_status.get(side) == 'READY'
                            or self.backend_status.get(side, '').startswith('FINISHED:'))):
                raise ValueError(side + ' backend became unavailable during planning')
        messages = [to_ros_trajectory(plan.left), to_ros_trajectory(plan.right)]
        now_ns = self.get_clock().now().nanoseconds
        start_ns = now_ns + int(round(self.delay * 1e9))
        for message in messages:
            message.points = list(message.points)[1:]
            message.header.stamp.sec = start_ns // 10**9
            message.header.stamp.nanosec = start_ns % 10**9
        self.plan = plan
        self.start_ns = start_ns
        self.last_clock_ns = now_ns
        self.published_at = time.monotonic()
        self.expected_end = self.published_at + self.delay + plan.duration
        self.goal_since = None
        self.accepted.clear()
        self.finished_arms.clear()
        for side, arm_plan in (('left', plan.left), ('right', plan.right)):
            self.start_positions[side] = self.current[side].copy()
            excursion = float(np.max(np.abs(arm_plan.positions - self.current[side])))
            self.required_motion[side] = min(0.003, excursion * 0.1)
            self.observed_motion[side] = 0.0
        self.set_status('WAITING_FOR_ACCEPTANCE')
        self.publish_profile(start_ns)
        for side, message in zip(('left', 'right'), messages):
            self.previews[side].publish(message)
            self.commands[side].publish(message)

    def publish_profile(self, start_ns: int) -> None:
        """发布本次任务实际使用的参考量，避免记录节点猜测启动参数或任务类型。"""
        profile: dict[str, object] = {
            'version': 1, 'task_type': self.task_type, 'start_stamp_ns': start_ns,
            'geometry': self.geometry, 'speed_limits': self.speeds,
            'approach_duration_s': self.approach_duration,
            'return_home_start_s': self.return_home_start,
            'circle_center': self.circle_center.tolist(), 'circle_radius': self.circle_radius,
        }
        cooperative = self.cooperative_plan
        if cooperative is not None:
            profile.update({
                'object_times_s': cooperative.paired.left.times.tolist(),
                'world_object_poses': [pose.tolist() for pose in cooperative.object_poses],
                'object_from_left_tcp': cooperative.object_from_left_tcp.tolist(),
                'object_from_right_tcp': cooperative.object_from_right_tcp.tolist(),
            })
        message = String()
        message.data = json.dumps(profile, ensure_ascii=False)
        self.profile_pub.publish(message)

    def cancel_planning(self) -> None:
        future = self.planning
        self.planning = None
        if future is not None:
            # 正在执行的计算会在取消检查处退出，其结果不会再发布。
            self.cancel_event.set()
            self.cancelled_work = future
            future.cancel()

    def request_stops(self) -> bool:
        all_sent = True
        for side in ('left', 'right'):
            client = self.stop_clients[side]
            if not client.service_is_ready():
                all_sent = False
                self.get_logger().error(side + ' stop service unavailable')
                continue
            future = client.call_async(Trigger.Request())
            future.add_done_callback(lambda result, arm=side: self.on_stop_result(arm, result))
        return all_sent

    def on_stop_result(self, side: str, future) -> None:
        try:
            response = future.result()
            if response is None or not response.success:
                self.get_logger().error(side + ' stop not confirmed')
            else:
                self.get_logger().info(side + ' stop confirmed')
        except Exception as exc:
            self.get_logger().error(f'{side} stop failed: {exc}')

    def fail(self, reason: str) -> None:
        self.cancel_planning()
        self.plan = None
        self.start_ns = None
        self.set_status('FAILED: ' + reason)
        self.request_stops()

    def on_stop(self, request, response):
        self.cancel_planning()
        self.plan = None
        self.start_ns = None
        sent = self.request_stops()
        self.set_status('STOP_REQUESTED' if sent else 'STOP_UNCONFIRMED')
        response.success = sent
        response.message = 'Check both stop confirmations; re-enable both simulators before another task'
        return response

    def monitor(self) -> None:
        future = self.planning
        if future is not None:
            if not self.feedback_fresh():
                self.fail('Joint feedback expired during planning')
            elif time.monotonic() - self.planning_started > 120.0:
                self.fail('Planning timed out')
            elif future.done():
                self.planning = None
                try:
                    prepared = future.result()
                    self.approach_duration = prepared.approach_duration
                    self.return_home_start = prepared.return_home_start
                    self.cooperative_plan = prepared.cooperative
                    self.dispatch_plan(prepared.paired)
                except Exception as exc:
                    self.fail('Planning failed: ' + str(exc))
            return
        plan = self.plan
        start_ns = self.start_ns
        if plan is None or start_ns is None:
            if self.state == 'WAITING_FOR_FEEDBACK' and self.feedback_fresh() and self.scene_matches:
                self.set_status('READY')
            return
        if not self.feedback_fresh():
            self.fail('Joint feedback expired')
            return
        now_ns = self.get_clock().now().nanoseconds
        previous = self.last_clock_ns
        self.last_clock_ns = now_ns
        if previous is not None and not 0 <= now_ns - previous <= 250_000_000:
            self.fail('Clock jump or executor delay')
            return
        if len(self.accepted) < 2:
            if now_ns >= start_ns - 300_000_000:
                self.fail('Both simulators did not acknowledge before the start deadline')
            return
        if now_ns < start_ns:
            if self.state != 'ARMED':
                self.set_status('ARMED')
            return
        if self.state != 'RUNNING':
            self.set_status('RUNNING')
        elapsed = (now_ns - start_ns) * 1e-9
        cooperative = self.cooperative_plan
        if cooperative is None:
            if elapsed < self.approach_duration:
                self.set_stage('APPROACH')
            elif self.return_home_start is not None and elapsed > self.return_home_start:
                self.set_stage('RETURN_HOME')
            else:
                self.set_stage('DRAW_CIRCLE')
        elif elapsed < self.approach_duration:
            self.set_stage('APPROACH')
        else:
            self.set_stage('COOPERATIVE')
            obj = sample_object(cooperative, elapsed - self.approach_duration)
            left_pose, right_pose = self.kinematics.forward_world(self.current['left'], self.current['right'])
            errors = [pose_error(left_pose, obj @ cooperative.object_from_left_tcp),
                      pose_error(right_pose, obj @ cooperative.object_from_right_tcp),
                      pose_error(np.linalg.inv(left_pose) @ right_pose,
                                 np.linalg.inv(cooperative.object_from_left_tcp) @ cooperative.object_from_right_tcp)]
            if (max(pair[0] for pair in errors) > self.cooperative_position_tolerance
                    or max(pair[1] for pair in errors) > self.cooperative_rotation_tolerance):
                self.fail('协同执行的 TCP 或相对位姿误差超限')
                return
        now = time.monotonic()
        if now > self.expected_end + 10.0:
            self.fail('Timed out waiting for both arms to converge')
            return
        for side in ('left', 'right'):
            self.observed_motion[side] = max(self.observed_motion[side], float(np.max(
                np.abs(self.current[side] - self.start_positions[side]))))
        converged = now >= self.expected_end and len(self.finished_arms) == 2
        for side, arm_plan in (('left', plan.left), ('right', plan.right)):
            converged = (converged and float(np.max(np.abs(self.current[side] - arm_plan.positions[-1]))) <= self.tolerance
                         and self.velocities[side] <= self.velocity_limit
                         and self.observed_motion[side] >= self.required_motion[side])
        if converged:
            if self.goal_since is None:
                self.goal_since = now
            elif now - self.goal_since >= 0.3:
                self.plan = None
                self.start_ns = None
                self.set_status('COMPLETED')
        else:
            self.goal_since = None

    def destroy_node(self):
        self.cancel_planning()
        self.planner.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = None
    try:
        node = DualArmTask()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node is not None and (node.plan is not None or node.planning is not None):
            node.cancel_planning()
            node.plan = None
            node.request_stops()
            deadline = time.monotonic() + 0.5
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            print('Stop requests sent; verify simulator confirmations.')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()