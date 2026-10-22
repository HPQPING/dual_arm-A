"""提供ROS单臂MuJoCo仿真、关节反馈与位置轨迹接口，不连接真机."""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from array import array
from importlib import import_module
import sys
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from .model_loader import DataInterface, FloatArray, ModelInterface, load_single_arm, mujoco
from .joint_interpolation import coefficients, evaluate, _extrema

ParsedPlan = tuple[FloatArray, FloatArray] | tuple[FloatArray, FloatArray, FloatArray, FloatArray]


class ViewerCamera(Protocol):
    lookat: FloatArray
    distance: float
    azimuth: float
    elevation: float


class ViewerHandle(Protocol):
    cam: ViewerCamera

    def lock(self) -> AbstractContextManager[object]: ...

    def sync(self) -> None: ...

    def is_running(self) -> bool: ...

    def close(self) -> None: ...


class ViewerModule(Protocol):
    def launch_passive(self, model: ModelInterface, data: DataInterface) -> ViewerHandle: ...


def prepare_plan(
    message: JointTrajectory,
    names: tuple[str, ...],
    start: ArrayLike,
    limits: ArrayLike,
    speed_limits: ArrayLike,
    allow_scheduled: bool = False,
) -> ParsedPlan:
    """校验停车点或带导数的连续轨迹，并统一关节顺序。"""
    message_names = list(message.joint_names)
    start_array = np.asarray(start, dtype=np.float64)
    bounds = np.asarray(limits, dtype=np.float64)
    speeds = np.asarray(speed_limits, dtype=np.float64)
    if start_array.shape != (6,) or not np.all(np.isfinite(start_array)):
        raise ValueError("Start must contain six finite positions")
    if (
        bounds.shape != (6, 2)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[:, 0] >= bounds[:, 1])
    ):
        raise ValueError("Invalid joint-limit matrix")
    if speeds.shape != (6,) or not np.all(np.isfinite(speeds)) or np.any(speeds <= 0):
        raise ValueError("Invalid speed limits")
    if len(message_names) != 6 or set(message_names) != set(names):
        raise ValueError("joint_names must contain all six joints exactly once")
    if message.header.stamp.sec < 0 or not 0 <= message.header.stamp.nanosec < 10**9:
        raise ValueError("Invalid header timestamp")
    if not allow_scheduled and (
        message.header.stamp.sec != 0 or message.header.stamp.nanosec != 0
    ):
        raise ValueError("Scheduled timestamps require allow_scheduled=True")
    points = list(message.points)
    if not points or len(points) > 10000:
        raise ValueError("Trajectory must contain 1 to 10000 points")
    order = [message_names.index(name) for name in names]
    if any(p.velocities or p.accelerations for p in points):
        return prepare_continuous_plan(points, order, start_array, bounds, speeds)
    times = [0.0]
    positions: list[FloatArray] = [start_array.copy()]
    for point_index, point in enumerate(points):
        if len(point.positions) != 6:
            raise ValueError("Each point must contain six positions")
        if point.velocities or point.accelerations or point.effort:
            raise ValueError("Only positions are supported in this first version")
        if point.time_from_start.sec < 0 or not 0 <= point.time_from_start.nanosec < 10**9:
            raise ValueError("Invalid time_from_start")
        t = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        q = np.asarray(list(point.positions), dtype=np.float64)[order]
        if not np.all(np.isfinite(q)):
            raise ValueError("Joint positions must be finite")
        if np.any(q < bounds[:, 0]) or np.any(q > bounds[:, 1]):
            raise ValueError("Trajectory point is outside joint limits")
        if t == 0.0 and point_index == 0:
            if not np.allclose(q, start_array, atol=1e-6, rtol=0):
                raise ValueError("A point at time zero must match the current pose")
            continue
        if t <= times[-1] or t > 3600:
            raise ValueError("Point times must increase and be at most 3600 seconds")
        # Quintic interpolation has a maximum normalized slope of 1.875.
        peak = 1.875 * np.abs(q - positions[-1]) / (t - times[-1])
        if np.any(peak > speeds + 1e-9):
            raise ValueError("Trajectory duration is too short for the speed limits")
        times.append(t)
        positions.append(q)
    if len(times) < 2:
        raise ValueError("At least one point must have a positive duration")
    return np.asarray(times, dtype=np.float64), np.asarray(positions, dtype=np.float64)


def prepare_continuous_plan(
    points: list[JointTrajectoryPoint], order: list[int], start: FloatArray,
    bounds: FloatArray, speeds: FloatArray,
) -> ParsedPlan:
    """保留全部节点导数，解析核对段内位置和速度，不重新改成逐点停车。"""
    times, positions = [0.], [start.copy()]
    velocities, accelerations = [np.zeros(6)], [np.zeros(6)]
    for index, point in enumerate(points):
        if (any(len(values) != 6 for values in
                (point.positions, point.velocities, point.accelerations)) or point.effort):
            raise ValueError('连续轨迹每点必须同时包含六维位置、速度、加速度且不含力矩')
        q, v, a = [np.asarray(list(values), dtype=np.float64)[order] for values in
                   (point.positions, point.velocities, point.accelerations)]
        if any(not np.all(np.isfinite(x)) for x in (q, v, a)):
            raise ValueError('连续轨迹的节点及导数必须有限')
        stamp = point.time_from_start
        if stamp.sec < 0 or not 0 <= stamp.nanosec < 10**9:
            raise ValueError('连续轨迹时间无效')
        t = stamp.sec + stamp.nanosec * 1e-9
        if index == 0 and t == 0.:
            if (not np.allclose(q, start, atol=1e-6, rtol=0.)
                    or np.any(v != 0.) or np.any(a != 0.)):
                raise ValueError('连续轨迹起点必须匹配当前姿态并从零导数开始')
            continue
        if t <= times[-1] or t > 3600.:
            raise ValueError('连续轨迹时间必须递增且不超过一小时')
        h = float(t - times[-1])
        c = coefficients(positions[-1], q, velocities[-1], v, accelerations[-1], a, h)
        extent = _extrema(c, h, 0)
        if (np.any(extent[:, 0] < bounds[:, 0] - 1e-12)
                or np.any(extent[:, 1] > bounds[:, 1] + 1e-12)
                or np.any(np.max(np.abs(_extrema(c, h, 1)), axis=1) > speeds + 1e-9)):
            raise ValueError('连续轨迹段内位置或速度超限')
        times.append(t)
        positions.append(q)
        velocities.append(v)
        accelerations.append(a)
    if len(times) < 2 or np.any(velocities[-1] != 0.) or np.any(accelerations[-1] != 0.):
        raise ValueError('连续轨迹需要正时长并以零导数结束')
    return (np.asarray(times, dtype=np.float64), np.asarray(positions, dtype=np.float64),
            np.asarray(velocities, dtype=np.float64), np.asarray(accelerations, dtype=np.float64))


def sample_plan(
    times: FloatArray, positions: FloatArray, elapsed: float,
    velocities: FloatArray | None = None, accelerations: FloatArray | None = None,
) -> tuple[FloatArray, bool]:
    """使用节点导数连续插值；旧位置接口仍按零导数停车。"""
    if not np.isfinite(elapsed):
        raise ValueError("Elapsed time must be finite")
    if elapsed >= times[-1]:
        return positions[-1].copy(), True
    elapsed = max(0.0, float(elapsed))
    index = min(int(np.searchsorted(times, elapsed, side="right")) - 1, len(times) - 2)
    u = (elapsed - times[index]) / (times[index + 1] - times[index])
    if velocities is not None and accelerations is not None:
        h = float(times[index + 1] - times[index])
        c = coefficients(positions[index], positions[index + 1],
                         velocities[index], velocities[index + 1],
                         accelerations[index], accelerations[index + 1], h)
        return evaluate(c, float(u), h), False
    s = 6 * u**5 - 15 * u**4 + 10 * u**3
    return positions[index] + s * (positions[index + 1] - positions[index]), False


def sample_prepared_plan(plan: ParsedPlan, elapsed: float) -> tuple[FloatArray, bool]:
    """单臂、双臂和记录器使用同一份已校验曲线。"""
    if len(plan) == 4:
        return sample_plan(plan[0], plan[1], elapsed, plan[2], plan[3])
    return sample_plan(plan[0], plan[1], elapsed)


def checked_number(value: object, name: str) -> float:
    """Reject missing, boolean, non-numeric and non-finite parameter values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def checked_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false, got {value!r}")
    return value


def checked_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def checked_vector(value: object, name: str, length: int) -> list[float]:
    if not isinstance(value, (list, tuple, array)) or len(value) != length:
        raise ValueError(f"{name} must contain {length} numbers")
    return [checked_number(item, name) for item in value]


class SingleArmSim(Node):
    def __init__(self):
        super().__init__("single_arm_sim")
        defaults = {
            "side": "left",
            "timestep": 0.002,
            "state_hz": 50.0,
            "viewer_hz": 30.0,
            "show_viewer": True,
            "enable_contacts": False,
            "gravity_compensation": True,
            "position_kp": 80.0,
            "position_kv": 8.0,
            "tcp_xyz": [0.0, 0.0, 0.147],
            "mount_xyz": [0.0, 0.0, 0.0],
            "mount_rpy": [0.0, 0.0, 0.0],
            "speed_limits": [0.1, 0.5, 0.5, 0.5, 0.5, 0.5],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        self.side = checked_text(self.get_parameter("side").value, "side")
        dt = checked_number(self.get_parameter("timestep").value, "timestep")
        state_hz = checked_number(self.get_parameter("state_hz").value, "state_hz")
        viewer_hz = checked_number(self.get_parameter("viewer_hz").value, "viewer_hz")
        if not all(np.isfinite(v) and v > 0 for v in (dt, state_hz, viewer_hz)):
            raise ValueError("timestep, state_hz and viewer_hz must be positive")
        if state_hz > 1 / dt or viewer_hz > 1 / dt:
            raise ValueError("State/viewer rate cannot exceed the physics step rate")
        if checked_bool(self.get_parameter("use_sim_time").value, "use_sim_time"):
            raise ValueError("Use use_sim_time:=false for this standalone simulation")
        self.speed_limits = np.asarray(
            checked_vector(self.get_parameter("speed_limits").value, "speed_limits", 6),
            dtype=float,
        )
        if (
            self.speed_limits.shape != (6,)
            or not np.all(np.isfinite(self.speed_limits))
            or np.any(self.speed_limits <= 0)
        ):
            raise ValueError("speed_limits must contain six positive finite values")
        self.compensate = checked_bool(
            self.get_parameter("gravity_compensation").value, "gravity_compensation"
        )
        self.arm = load_single_arm(
            side=self.side,
            timestep=dt,
            position_kp=checked_number(self.get_parameter("position_kp").value, "position_kp"),
            position_kv=checked_number(self.get_parameter("position_kv").value, "position_kv"),
            tcp_xyz=checked_vector(self.get_parameter("tcp_xyz").value, "tcp_xyz", 3),
            mount_xyz=checked_vector(self.get_parameter("mount_xyz").value, "mount_xyz", 3),
            mount_rpy=checked_vector(self.get_parameter("mount_rpy").value, "mount_rpy", 3),
            enable_contacts=checked_bool(
                self.get_parameter("enable_contacts").value, "enable_contacts"
            ),
        )
        self.initial = self.arm.data.qpos[self.arm.qpos_indices].copy()
        self.target = self.initial.copy()
        self.plan: ParsedPlan | None = None
        self.plan_start = 0.0
        self.plan_start_ros_ns: int | None = None
        self.last_schedule_clock: int | None = None
        self.schedule_running = False
        self.enabled = True
        self.paused = False
        self.finished = False
        self.viewer: ViewerHandle | None = None
        topic = "/" + self.side
        self.state_pub = self.create_publisher(JointState, topic + "/joint_states", 10)
        status_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.trajectory_status_pub = self.create_publisher(
            String, topic + "/trajectory_status", status_qos
        )
        self.command_sub = self.create_subscription(
            JointTrajectory, topic + "/joint_trajectory", self.on_trajectory, 10
        )
        self.stop_service = self.create_service(Trigger, topic + "/stop", self.on_stop)
        self.enable_service = self.create_service(SetBool, topic + "/set_enabled", self.on_enabled)
        self.pause_service = self.create_service(SetBool, topic + "/pause", self.on_pause)
        self.reset_service = self.create_service(Trigger, topic + "/reset", self.on_reset)
        # A steady timer avoids dependence on /clock. Simulation advances one dt per callback.
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.physics_timer = self.create_timer(dt, self.step, clock=self.steady_clock)
        self.state_timer = self.create_timer(
            1 / state_hz, self.publish_state, clock=self.steady_clock
        )
        self.viewer_timer = self.create_timer(
            1 / viewer_hz, self.sync_viewer, clock=self.steady_clock
        )
        if checked_bool(self.get_parameter("show_viewer").value, "show_viewer"):
            viewer_module = cast(ViewerModule, import_module("mujoco.viewer"))
            viewer = viewer_module.launch_passive(self.arm.model, self.arm.data)
            self.viewer = viewer
            with viewer.lock():
                viewer.cam.lookat[:] = [0.0, 0.0, 0.35]
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -20
            viewer.sync()
        self.get_logger().info(
            f"{self.side} simulation ready. No hardware connection. "
            f"Commands: {topic}/joint_trajectory; states: {topic}/joint_states"
        )
        self.publish_trajectory_status("READY")

    def publish_trajectory_status(self, status: str):
        message = String()
        message.data = status
        self.trajectory_status_pub.publish(message)

    def viewer_lock(self) -> AbstractContextManager[object]:
        viewer = self.viewer
        return nullcontext() if viewer is None else viewer.lock()

    def on_trajectory(self, message: JointTrajectory):
        if not self.enabled or self.paused or self.plan is not None:
            self.get_logger().warning("Rejected: stopped, paused, or another trajectory is active")
            self.publish_trajectory_status("REJECTED: stopped, paused, or busy")
            return
        try:
            stamp = message.header.stamp
            start_ns = int(stamp.sec) * 10**9 + int(stamp.nanosec)
            now_ns = self.get_clock().now().nanoseconds
            if start_ns and not 200_000_000 <= start_ns - now_ns <= 30_000_000_000:
                raise ValueError("Scheduled start must be 0.2 to 30 seconds in the future")
            plan = prepare_plan(
                message,
                self.arm.joint_names,
                self.arm.data.qpos[self.arm.qpos_indices],
                self.arm.joint_limits,
                self.speed_limits,
                allow_scheduled=True,
            )
        except (ValueError, TypeError, OverflowError) as exc:
            self.get_logger().error(f"Rejected trajectory: {exc}")
            self.publish_trajectory_status("REJECTED: " + str(exc))
            return
        self.plan = plan
        self.plan_start = float(self.arm.data.time)
        self.plan_start_ros_ns = start_ns if start_ns else None
        self.last_schedule_clock = now_ns if start_ns else None
        self.schedule_running = False
        self.publish_trajectory_status(f"ACCEPTED:{start_ns}")
        self.get_logger().info(f"Trajectory accepted: {plan[0][-1]:.2f} simulation seconds")

    def hold_current(self):
        self.plan = None
        self.plan_start_ros_ns = None
        self.last_schedule_clock = None
        with self.viewer_lock():
            self.target = np.clip(
                self.arm.data.qpos[self.arm.qpos_indices],
                self.arm.joint_limits[:, 0],
                self.arm.joint_limits[:, 1],
            ).copy()
            self.arm.data.ctrl[self.arm.actuator_indices] = self.target

    def on_stop(self, request, response):
        self.hold_current()
        self.enabled = False
        self.publish_trajectory_status("STOPPED")
        response.success = True
        response.message = (
            "Trajectory cancelled; holding. Call set_enabled with true to accept new commands."
        )
        return response

    def on_enabled(self, request, response):
        if not request.data:
            self.hold_current()
        self.enabled = bool(request.data)
        if self.plan is None:
            self.publish_trajectory_status("READY" if self.enabled else "STOPPED")
        response.success = True
        response.message = "Commands enabled" if self.enabled else "Holding; commands disabled"
        return response

    def on_pause(self, request, response):
        # A shared wall-clock schedule cannot be resumed after a local pause.
        if request.data and self.plan_start_ros_ns is not None:
            self.hold_current()
            self.enabled = False
            self.publish_trajectory_status("STOPPED: scheduled trajectory cancelled by pause")
        self.paused = bool(request.data)
        response.success = True
        response.message = "Physics paused" if self.paused else "Physics resumed"
        return response

    def on_reset(self, request, response):
        # Simulation-only teleport. Keep simulation time monotonic.
        with self.viewer_lock():
            simulation_time = float(self.arm.data.time)
            mujoco.mj_resetData(self.arm.model, self.arm.data)
            self.arm.data.time = simulation_time
            self.arm.data.qpos[self.arm.qpos_indices] = self.initial
            self.arm.data.ctrl[self.arm.actuator_indices] = self.initial
            mujoco.mj_forward(self.arm.model, self.arm.data)
        self.plan = None
        self.plan_start_ros_ns = None
        self.last_schedule_clock = None
        self.target = self.initial.copy()
        self.enabled = False
        self.publish_trajectory_status("STOPPED: reset")
        response.success = True
        response.message = "Simulation reset. Commands disabled; explicitly enable before moving."
        return response

    def step(self):
        if self.paused or self.finished:
            return
        with self.viewer_lock():
            plan = self.plan
            if plan is not None:
                elapsed = float(self.arm.data.time) - self.plan_start
                start_ns = self.plan_start_ros_ns
                if start_ns is not None:
                    now_ns = self.get_clock().now().nanoseconds
                    previous_ns = self.last_schedule_clock
                    self.last_schedule_clock = now_ns
                    if previous_ns is not None and not 0 <= now_ns - previous_ns <= 250_000_000:
                        self.plan = None
                        self.plan_start_ros_ns = None
                        self.last_schedule_clock = None
                        self.target = self.arm.data.qpos[self.arm.qpos_indices].copy()
                        self.arm.data.ctrl[self.arm.actuator_indices] = self.target
                        self.enabled = False
                        self.publish_trajectory_status("STOPPED: clock jump or execution delay")
                        return
                    elapsed = (now_ns - start_ns) * 1e-9
                    if elapsed >= 0 and not self.schedule_running:
                        self.schedule_running = True
                        self.publish_trajectory_status(f"RUNNING:{start_ns}")
                self.target, done = sample_prepared_plan(plan, elapsed)
                if done:
                    self.plan = None
                    self.plan_start_ros_ns = None
                    self.last_schedule_clock = None
                    self.publish_trajectory_status(
                        f"FINISHED:{start_ns if start_ns is not None else 0}"
                    )
                    self.get_logger().info("Target trajectory finished; holding final target")
            self.arm.data.ctrl[self.arm.actuator_indices] = self.target
            self.arm.data.qfrc_applied[:] = 0
            if self.compensate:
                mujoco.mj_forward(self.arm.model, self.arm.data)
                # Simulation bias-force compensation only; no hardware torque commands.
                self.arm.data.qfrc_applied[self.arm.dof_indices] = self.arm.data.qfrc_bias[
                    self.arm.dof_indices
                ]
            mujoco.mj_step(self.arm.model, self.arm.data)
            if not np.all(np.isfinite(self.arm.data.qpos)) or not np.all(
                np.isfinite(self.arm.data.qvel)
            ):
                self.plan = None
                self.paused = True
                self.enabled = False
                self.publish_trajectory_status("STOPPED: non-finite simulation state")
                self.get_logger().error(
                    "Non-finite state; simulation paused. Reset before continuing."
                )

    def publish_state(self):
        message = JointState()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "world"
        message.name = list(self.arm.joint_names)
        message.position = [float(value) for value in self.arm.data.qpos[self.arm.qpos_indices]]
        message.velocity = [float(value) for value in self.arm.data.qvel[self.arm.dof_indices]]
        # Effort reports actuator-generated torque, excluding compensation/contact forces.
        message.effort = [
            float(value) for value in self.arm.data.qfrc_actuator[self.arm.dof_indices]
        ]
        self.state_pub.publish(message)

    def sync_viewer(self):
        viewer = self.viewer
        if viewer is not None:
            if not viewer.is_running():
                self.finished = True
                return
            viewer.sync()

    def destroy_node(self):
        viewer = self.viewer
        if viewer is not None:
            viewer.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None
    failed = False
    try:
        node = SingleArmSim()
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception as exc:
        failed = True
        print(f"Single-arm simulation failed: {exc}", file=sys.stderr)
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
