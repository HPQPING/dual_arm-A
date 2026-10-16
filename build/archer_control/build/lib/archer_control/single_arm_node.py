"""Single-arm joint and sampled Cartesian tasks for the simulation interface."""

from array import array
import math
import time

import numpy as np
from numpy.typing import ArrayLike
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .trajectory import (
    DEFAULT_JOINT_LIMITS, DEFAULT_SPEED_LIMITS, FloatArray, JointPlan,
    joint_names, plan_joint_path, to_ros_trajectory, validate_joints,
)
from .kinematics import ArmKinematics


def number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be a number')
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def numbers(value: object, name: str) -> list[float]:
    if not isinstance(value, (list, tuple, array)):
        raise ValueError(f'{name} must be an array of numbers')
    return [number(item, name) for item in value]


def circle_positions(start: ArrayLike, radius: float, plane: str, samples: int) -> FloatArray:
    """Circle samples starting and ending at the current TCP, in arm base.

    The center is offset by -radius along the plane's first axis. This avoids
    an initial jump to the circle. Joint interpolation between samples is
    an approximation, not exact continuous Cartesian circle tracking.
    """
    position = np.asarray(start, dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError('Circle start must contain three finite coordinates')
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('circle_radius must be positive and finite')
    if isinstance(samples, bool) or not isinstance(samples, int) or not 8 <= samples <= 1000:
        raise ValueError('circle_samples must be an integer from 8 to 1000')
    axes = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
    if plane not in axes:
        raise ValueError('circle_plane must be xy, xz or yz')
    first, second = axes[plane]
    angles = np.linspace(0., 2. * np.pi, samples + 1)[1:]
    points = np.repeat(position[None, :], samples, axis=0)
    points[:, first] += radius * (np.cos(angles) - 1.)
    points[:, second] += radius * np.sin(angles)
    points[-1] = position
    return points


def rotation_from_rpy(values: ArrayLike) -> FloatArray:
    rpy = np.asarray(values, dtype=np.float64)
    if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
        raise ValueError('target_rpy must contain three finite radians')
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)


def cartesian_joint_path(
    model: ArmKinematics, start: ArrayLike, positions: ArrayLike, rotation: ArrayLike,
) -> FloatArray:
    """Solve all TCP samples with a continuous joint seed before publishing."""
    points = np.asarray(positions, dtype=np.float64)
    orientation = np.asarray(rotation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not 1 <= len(points) <= 9999:
        raise ValueError('Cartesian positions must have shape (N, 3)')
    if (not np.all(np.isfinite(points)) or orientation.shape != (3, 3)
            or not np.all(np.isfinite(orientation))):
        raise ValueError('Invalid Cartesian target values')
    seed = np.asarray(start, dtype=np.float64).copy()
    solutions: list[FloatArray] = []
    for index, position in enumerate(points):
        target = np.eye(4, dtype=np.float64)
        target[:3, :3] = orientation
        target[:3, 3] = position
        result = model.solve_ik(target, seed)
        if not result.success:
            raise ValueError(
                f'IK failed at sample {index + 1}: {result.message}; '
                f'position error={result.position_error:.6f} m, '
                f'rotation error={result.orientation_error:.6f} rad. No trajectory published.'
            )
        if float(np.max(np.abs(result.joints - seed))) > 0.35:
            raise ValueError('IK step exceeds 0.35 rad; add intermediate Cartesian samples')
        seed = result.joints.copy()
        solutions.append(seed)
    return np.asarray(solutions, dtype=np.float64)


class SingleArmTask(Node):
    def __init__(self):
        super().__init__('single_arm_task')
        defaults = {
            'side': 'left',
            'task_type': 'joint',
            'target_frame': 'base',
            'keep_current_orientation': True,
            'target_rpy': [0.0, 0.0, 0.0],
            'cartesian_waypoints': [0.30, 0.0, 0.40],
            'circle_radius': 0.005,
            'circle_plane': 'xz',
            'circle_samples': 16,
            'tcp_xyz': [0.0, 0.0, 0.147],
            'mount_xyz': [0.0, 0.0, 0.0],
            'mount_rpy': [0.0, 0.0, 0.0],
            'minimum_duration': 4.0,
            'time_margin': 1.1,
            'feedback_timeout': 0.5,
            'start_velocity_limit': 0.01,
            'goal_tolerance': 0.02,
            'goal_hold_time': 0.3,
            'timeout_scale': 2.0,
            'timeout_extra': 10.0,
            'speed_limits': list(DEFAULT_SPEED_LIMITS),
            'joint_limits': [value for pair in DEFAULT_JOINT_LIMITS for value in pair],
            # ROS 2 parameters do not support nested numeric lists. Six numbers per waypoint.
            'waypoints': [0.1, -0.7, 1.5, -0.8, 0.0, 0.0,
                          0.0, -0.8, 1.6, -0.8, 0.0, 0.0],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        side = self.get_parameter('side').value
        if not isinstance(side, str) or side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        self.names = joint_names(side)
        self.side = side
        task_type = self.get_parameter('task_type').value
        if not isinstance(task_type, str) or task_type not in ('joint', 'cartesian', 'circle'):
            raise ValueError('task_type must be joint, cartesian or circle')
        self.task_type = task_type
        frame = self.get_parameter('target_frame').value
        if frame != 'base':
            raise ValueError('This version accepts only target_frame: base')
        keep_orientation = self.get_parameter('keep_current_orientation').value
        if not isinstance(keep_orientation, bool):
            raise ValueError('keep_current_orientation must be true or false')
        self.keep_orientation = keep_orientation
        self.target_rpy = numbers(self.get_parameter('target_rpy').value, 'target_rpy')
        rotation_from_rpy(self.target_rpy)
        flat_cartesian = numbers(self.get_parameter('cartesian_waypoints').value, 'cartesian_waypoints')
        if not flat_cartesian or len(flat_cartesian) % 3:
            raise ValueError('cartesian_waypoints must contain three numbers per position')
        self.cartesian_waypoints = np.asarray(flat_cartesian, dtype=np.float64).reshape(-1, 3)
        self.circle_radius = self.read_number('circle_radius')
        circle_plane = self.get_parameter('circle_plane').value
        samples = self.get_parameter('circle_samples').value
        if not isinstance(circle_plane, str) or isinstance(samples, bool) or not isinstance(samples, int):
            raise ValueError('circle_plane must be a string and circle_samples must be an integer')
        self.circle_plane = circle_plane
        self.circle_samples = samples
        circle_positions([0., 0., 0.], self.circle_radius, circle_plane, samples)
        self.minimum = self.read_number('minimum_duration')
        self.margin = self.read_number('time_margin')
        self.feedback_timeout = self.read_number('feedback_timeout')
        self.start_velocity_limit = self.read_number('start_velocity_limit')
        self.tolerance = self.read_number('goal_tolerance')
        self.hold_time = self.read_number('goal_hold_time')
        self.timeout_scale = self.read_number('timeout_scale')
        self.timeout_extra = self.read_number('timeout_extra')
        if any(value <= 0 for value in (
            self.minimum, self.feedback_timeout, self.start_velocity_limit,
            self.tolerance, self.hold_time, self.timeout_extra,
        )) or self.margin < 1 or self.timeout_scale < 1:
            raise ValueError('Durations/tolerances must be positive; margins/scales must be >= 1')
        self.speeds = numbers(self.get_parameter('speed_limits').value, 'speed_limits')
        flat_limits = numbers(self.get_parameter('joint_limits').value, 'joint_limits')
        if len(flat_limits) != 12:
            raise ValueError('joint_limits must contain twelve numbers: lower, upper for each joint')
        self.limits = np.asarray(flat_limits, dtype=np.float64).reshape(6, 2)
        self.kinematics: ArmKinematics | None = None
        if self.task_type != 'joint':
            self.kinematics = ArmKinematics.from_description(
                side,
                numbers(self.get_parameter('tcp_xyz').value, 'tcp_xyz'),
                numbers(self.get_parameter('mount_xyz').value, 'mount_xyz'),
                numbers(self.get_parameter('mount_rpy').value, 'mount_rpy'),
            )
            self.limits[:, 0] = np.maximum(self.limits[:, 0], self.kinematics.limits[:, 0])
            self.limits[:, 1] = np.minimum(self.limits[:, 1], self.kinematics.limits[:, 1])
        flat_points = numbers(self.get_parameter('waypoints').value, 'waypoints')
        if not flat_points or len(flat_points) % 6:
            raise ValueError('waypoints must contain six numbers per waypoint')
        self.waypoints = np.asarray(flat_points, dtype=np.float64).reshape(-1, 6)
        # Validate configuration without sending anything.
        plan_joint_path(side, self.waypoints[0], self.waypoints,
                        self.speeds, self.limits, self.minimum, self.margin)

        self.current: np.ndarray | None = None
        self.last_feedback: float | None = None
        self.max_velocity = float('inf')
        self.plan: JointPlan | None = None
        self.started: float | None = None
        self.within_goal_since: float | None = None
        self.task_start_position: np.ndarray | None = None
        self.required_motion = 0.0
        self.observed_motion = 0.0
        self.state = 'WAITING_FOR_FEEDBACK'
        base = '/' + side
        from trajectory_msgs.msg import JointTrajectory
        self.command_pub = self.create_publisher(JointTrajectory, base + '/joint_trajectory', 10)
        self.feedback_sub = self.create_subscription(JointState, base + '/joint_states', self.on_state, 10)
        status_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                                durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = self.create_publisher(String, base + '/task/status', status_qos)
        self.stop_client = self.create_client(Trigger, base + '/stop')
        self.start_service = self.create_service(Trigger, base + '/task/start', self.on_start)
        self.stop_service = self.create_service(Trigger, base + '/task/stop', self.on_stop)
        self.timer = self.create_timer(0.05, self.monitor)
        self.set_status('WAITING_FOR_FEEDBACK')
        self.get_logger().info(
            f'{side} task ready. Start service: {base}/task/start. '
            'This topic client has no action-server acceptance acknowledgement.'
        )
        self.get_logger().info(f'Task type: {self.task_type}; Cartesian frame: {side}_base_link')

    def read_number(self, name: str) -> float:
        return number(self.get_parameter(name).value, name)

    def set_status(self, state: str):
        self.state = state
        message = String()
        message.data = state
        self.status_pub.publish(message)
        self.get_logger().info(state)

    def feedback_fresh(self) -> bool:
        received = self.last_feedback
        return received is not None and time.monotonic() - received <= self.feedback_timeout

    def on_state(self, message: JointState):
        names = list(message.name)
        if len(names) != len(set(names)) or not all(name in names for name in self.names):
            return
        if len(message.position) != len(names):
            return
        order = [names.index(name) for name in self.names]
        try:
            position = validate_joints([message.position[index] for index in order], self.limits)
        except (ValueError, TypeError):
            return
        now = time.monotonic()
        previous = self.current
        previous_time = self.last_feedback
        if len(message.velocity) == len(names):
            velocity = np.asarray([message.velocity[index] for index in order], dtype=np.float64)
            if not np.all(np.isfinite(velocity)):
                return
            self.max_velocity = float(np.max(np.abs(velocity)))
        elif previous is not None and previous_time is not None and now > previous_time:
            self.max_velocity = float(np.max(np.abs(position - previous)) / (now - previous_time))
        else:
            self.max_velocity = float('inf')
        self.current = position
        self.last_feedback = now

    def on_start(self, request, response):
        current = self.current
        if self.plan is not None:
            response.success = False
            response.message = 'A task is already running'
            return response
        if current is None or not self.feedback_fresh():
            response.success = False
            response.message = 'No fresh joint feedback'
            return response
        if self.max_velocity > self.start_velocity_limit:
            response.success = False
            response.message = 'Wait until the arm is stationary before starting'
            return response
        if self.command_pub.get_subscription_count() == 0:
            response.success = False
            response.message = 'No trajectory subscriber; start the simulation first'
            return response
        try:
            waypoints = self.waypoints
            if self.task_type != 'joint':
                model = self.kinematics
                if model is None:
                    raise ValueError('Kinematics model is not loaded')
                start_pose = model.forward(current)
                rotation = start_pose[:3, :3] if self.keep_orientation else rotation_from_rpy(self.target_rpy)
                positions = (circle_positions(start_pose[:3, 3], self.circle_radius,
                                              self.circle_plane, self.circle_samples)
                             if self.task_type == 'circle' else self.cartesian_waypoints)
                waypoints = cartesian_joint_path(model, current, positions, rotation)
            plan = plan_joint_path(self.side, current, waypoints,
                                   self.speeds, self.limits, self.minimum, self.margin)
            message = to_ros_trajectory(plan)
            # The simulator anchors interpolation to its own latest joint state.
            # Do not require exact equality with this client's slightly older feedback.
            message.points = list(message.points)[1:]
        except (ValueError, TypeError) as exc:
            response.success = False
            response.message = str(exc)
            return response
        self.plan = plan
        self.task_start_position = current.copy()
        excursion = float(np.max(np.abs(plan.positions - current)))
        self.required_motion = min(0.005, excursion * 0.1)
        self.observed_motion = 0.0
        self.started = time.monotonic()
        self.within_goal_since = None
        self.command_pub.publish(message)
        self.set_status('RUNNING')
        response.success = True
        response.message = f'Published trajectory, duration {plan.duration:.2f}s; monitoring feedback'
        return response

    def request_backend_stop(self):
        if not self.stop_client.service_is_ready():
            self.get_logger().error('Backend stop service is unavailable; stop was not confirmed')
            return False
        future = self.stop_client.call_async(Trigger.Request())
        future.add_done_callback(self.on_backend_stop)
        return True

    def on_backend_stop(self, future):
        try:
            response = future.result()
            if response is None or not response.success:
                self.get_logger().error('Backend did not confirm stop')
            else:
                self.get_logger().info('Backend confirmed stop/hold; explicitly re-enable before another task')
        except Exception as exc:
            self.get_logger().error(f'Backend stop failed: {exc}')

    def on_stop(self, request, response):
        self.plan = None
        self.started = None
        requested = self.request_backend_stop()
        self.set_status('STOP_REQUESTED' if requested else 'STOP_UNCONFIRMED')
        response.success = requested
        response.message = 'Stop request sent; check backend confirmation' if requested else 'Stop service unavailable'
        return response

    def fail(self, reason: str):
        self.plan = None
        self.started = None
        self.within_goal_since = None
        self.set_status('FAILED: ' + reason)
        self.request_backend_stop()

    def monitor(self):
        plan = self.plan
        started = self.started
        current = self.current
        if plan is None or started is None:
            if self.state == 'WAITING_FOR_FEEDBACK' and self.feedback_fresh() and self.max_velocity <= self.start_velocity_limit:
                self.set_status('READY')
            return
        if current is None or not self.feedback_fresh():
            self.fail('Joint feedback became stale')
            return
        now = time.monotonic()
        elapsed = now - started
        task_start_position = self.task_start_position
        if task_start_position is not None:
            self.observed_motion = max(
                self.observed_motion,
                float(np.max(np.abs(current - task_start_position))),
            )
        if elapsed > plan.duration * self.timeout_scale + self.timeout_extra:
            self.fail('Task timed out; trajectory may have been rejected or blocked')
            return
        # This checks final convergence, not Cartesian or per-waypoint tracking.
        error = float(np.max(np.abs(current - plan.positions[-1])))
        if (elapsed >= plan.duration and error <= self.tolerance
                and self.max_velocity <= self.start_velocity_limit
                and self.observed_motion >= self.required_motion):
            if self.within_goal_since is None:
                self.within_goal_since = now
            if now - self.within_goal_since >= self.hold_time:
                self.plan = None
                self.started = None
                self.set_status('COMPLETED')
        else:
            self.within_goal_since = None


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = SingleArmTask()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node is not None and node.plan is not None:
            print('Task client exited. Backend may continue; use the backend stop service if needed.')
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()