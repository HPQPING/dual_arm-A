"""
一个共享 MuJoCo 世界、一次物理积分、两个 ROS 关节位置控制端点.

使用 python3 -m archer_mujoco.dual_sim_node 启动，不连接实机。
"""

from __future__ import annotations

from contextlib import nullcontext
from importlib import import_module
import json
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectory

from .dual_model_loader import LoadedDualArm, load_dual_arm
from .model_loader import FloatArray, mujoco
from .sim_node import (
    SingleArmSim,
    ViewerHandle,
    ViewerModule,
    checked_bool,
    checked_number,
    checked_vector,
    sample_prepared_plan,
    ParsedPlan,
)


class SceneArmEndpoint(SingleArmSim):
    """复用单臂轨迹校验，物理积分由共同场景统一执行."""

    def __init__(self, scene: DualArmScene, side: str, speed_limits: FloatArray) -> None:
        # 不调用单臂构造函数，以免额外加载另一套物理世界。
        Node.__init__(self, side + "_scene_sim")
        self.scene = scene
        self.side = side
        self.arm = scene.loaded.arms[side]
        self.speed_limits = speed_limits.copy()
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
        topic = "/" + side
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.state_pub = self.create_publisher(JointState, topic + "/joint_states", 10)
        self.trajectory_status_pub = self.create_publisher(
            String, topic + "/trajectory_status", qos
        )
        self.command_sub = self.create_subscription(
            JointTrajectory, topic + "/joint_trajectory", self.on_trajectory, 10
        )
        self.stop_service = self.create_service(Trigger, topic + "/stop", self.on_stop)
        self.enable_service = self.create_service(SetBool, topic + "/set_enabled", self.on_enabled)
        self.pause_service = self.create_service(SetBool, topic + "/pause", self.on_pause)
        self.reset_service = self.create_service(Trigger, topic + "/reset", self.on_reset)
        self.publish_trajectory_status("READY")

    def on_stop(self, request, response):
        self.scene.stop_all("stop requested on " + self.side)
        response.success = True
        response.message = "Both arms holding; explicitly re-enable both before moving"
        return response

    def on_pause(self, request, response):
        if request.data:
            self.scene.stop_all("shared scene paused")
        self.scene.paused = bool(request.data)
        for endpoint in self.scene.endpoints.values():
            endpoint.paused = self.scene.paused
        response.success = True
        response.message = (
            "Shared physics paused" if request.data else "Resumed; both arms still need enabling"
        )
        return response

    def on_reset(self, request, response):
        self.scene.reset_all()
        response.success = True
        response.message = "Both simulation arms reset; commands disabled"
        return response

    def on_enabled(self, request, response):
        if not request.data:
            self.scene.stop_all("一侧取消使能")
        elif self.scene.paused:
            response.success = False
            response.message = "场景暂停中，请先恢复场景"
            return response
        else:
            self.enabled = True
            self.publish_trajectory_status("READY" if self.plan is None else "BUSY")
        response.success = True
        response.message = "本臂已使能" if request.data else "双臂已停止"
        return response

    def update_target(self, now_ns: int) -> None:
        plan = self.plan
        if plan is None:
            return
        start_ns = self.plan_start_ros_ns
        elapsed = (
            float(self.arm.data.time) - self.plan_start
            if start_ns is None
            else (now_ns - start_ns) * 1e-9
        )
        if start_ns is not None and elapsed >= 0 and not self.schedule_running:
            self.schedule_running = True
            self.publish_trajectory_status(f"RUNNING:{start_ns}")
        self.target, done = sample_prepared_plan(plan, elapsed)
        if done:
            self.plan = None
            self.plan_start_ros_ns = None
            self.last_schedule_clock = None
            self.publish_trajectory_status(f"FINISHED:{start_ns if start_ns is not None else 0}")


class ContactInfo(Protocol):
    geom1: int
    geom2: int
    dist: float


class ContactList(Protocol):
    def __getitem__(self, index: int) -> ContactInfo: ...


class ContactData(Protocol):
    ncon: int
    contact: ContactList


class ContactModel(Protocol):
    geom_margin: FloatArray
    geom_bodyid: NDArray[np.int32]


class BodyKinds(Protocol):
    mjOBJ_BODY: object


class NameAPI(Protocol):
    mjtObj: BodyKinds

    def mj_id2name(self, model: object, kind: object, index: int) -> str | None: ...


class DualArmScene(Node):
    def __init__(self) -> None:
        super().__init__("dual_arm_scene")
        defaults = {
            "timestep": 0.002,
            "state_hz": 50.0,
            "viewer_hz": 30.0,
            "show_viewer": True,
            "enable_contacts": False,
            "gravity_compensation": True,
            "position_kp": 80.0,
            "position_kv": 8.0,
            "left_base_xyz": [-0.21, 0.0, 0.0],
            "right_base_xyz": [0.21, 0.0, 0.0],
            "left_base_rpy": [0.0, 0.0, float(np.pi / 2)],
            "right_base_rpy": [0.0, 0.0, float(np.pi / 2)],
            "tcp_xyz": [0.0, 0.0, 0.147021696],
            "mount_xyz": [0.0, 0.0, 0.0],
            "mount_rpy": [0.0, 0.0, 0.0],
            "left_urdf": "",
            "right_urdf": "",
            "clearance": 0.01,
            "speed_limits": [0.1, 0.5, 0.5, 0.5, 0.5, 0.5],
            "left_initial_joints": [0.0, -0.8, 1.6, -0.8, 0.0, 0.0],
            "right_initial_joints": [0.0, -0.8, 1.6, -0.8, 0.0, 0.0],
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        if not self.has_parameter("use_sim_time"):
            self.declare_parameter("use_sim_time", False)
        if checked_bool(self.get_parameter("use_sim_time").value, "use_sim_time"):
            raise ValueError("Use use_sim_time:=false for this scene")
        dt = self.read_number("timestep")
        state_hz = self.read_number("state_hz")
        viewer_hz = self.read_number("viewer_hz")
        if min(dt, state_hz, viewer_hz) <= 0 or max(state_hz, viewer_hz) > 1 / dt:
            raise ValueError(
                "Positive rates required; state/viewer rates must not exceed physics rate"
            )
        self.compensate = checked_bool(
            self.get_parameter("gravity_compensation").value, "gravity_compensation"
        )
        self.clearance = self.read_number("clearance")
        if not 0.001 <= self.clearance <= 0.05:
            raise ValueError("跨臂检查间隙必须在 0.001～0.05 m 内")
        self.scene_config = {
            name: self.read_vector(name, 3)
            for name in (
                "left_base_xyz",
                "right_base_xyz",
                "left_base_rpy",
                "right_base_rpy",
                "tcp_xyz",
                "mount_xyz",
                "mount_rpy",
            )
        }
        paths: dict[str, str | None] = {}
        for side in ("left", "right"):
            value = self.get_parameter(side + "_urdf").value
            if not isinstance(value, str):
                raise ValueError("URDF 路径必须是字符串")
            paths[side] = value or None
        self.loaded: LoadedDualArm = load_dual_arm(
            left_urdf=paths["left"],
            right_urdf=paths["right"],
            left_base_xyz=self.read_vector("left_base_xyz", 3),
            right_base_xyz=self.read_vector("right_base_xyz", 3),
            left_base_rpy=self.read_vector("left_base_rpy", 3),
            right_base_rpy=self.read_vector("right_base_rpy", 3),
            tcp_xyz=self.read_vector("tcp_xyz", 3),
            mount_xyz=self.read_vector("mount_xyz", 3),
            mount_rpy=self.read_vector("mount_rpy", 3),
            timestep=dt,
            position_kp=self.read_number("position_kp"),
            position_kv=self.read_number("position_kv"),
            enable_contacts=checked_bool(
                self.get_parameter("enable_contacts").value, "enable_contacts"
            ),
        )
        speeds = np.asarray(self.read_vector("speed_limits", 6), dtype=np.float64)
        if np.any(speeds <= 0):
            raise ValueError("Speed limits must be positive")
        # 使用合法仿真初值，不裁剪实测反馈，也不擅自扩大关节限位。
        for side in ("left", "right"):
            initial = np.asarray(self.read_vector(side + "_initial_joints", 6), dtype=np.float64)
            arm = self.loaded.arms[side]
            bounds = arm.joint_limits
            invalid = np.flatnonzero((initial < bounds[:, 0]) | (initial > bounds[:, 1]))
            if len(invalid):
                index = int(invalid[0])
                raise ValueError(
                    f"{side} initial J{index + 1}={initial[index]:.8f} rad exceeds "
                    f"URDF limits [{bounds[index, 0]:.8f}, {bounds[index, 1]:.8f}]"
                )
            self.loaded.data.qpos[arm.qpos_indices] = initial
            self.loaded.data.ctrl[arm.actuator_indices] = initial
        mujoco.mj_forward(self.loaded.model, self.loaded.data)
        self.contact_model = load_dual_arm(
            left_urdf=paths["left"],
            right_urdf=paths["right"],
            left_base_xyz=self.scene_config["left_base_xyz"],
            right_base_xyz=self.scene_config["right_base_xyz"],
            left_base_rpy=self.scene_config["left_base_rpy"],
            right_base_rpy=self.scene_config["right_base_rpy"],
            tcp_xyz=self.scene_config["tcp_xyz"],
            mount_xyz=self.scene_config["mount_xyz"],
            mount_rpy=self.scene_config["mount_rpy"],
            enable_contacts=True,
        )
        cast(ContactModel, self.contact_model.model).geom_margin[:] = self.clearance * 0.6
        self.contact_ticks = 0
        self.paused = False
        self.finished = False
        self.viewer: ViewerHandle | None = None
        self.last_clock_ns: int | None = None
        self.endpoints = {side: SceneArmEndpoint(self, side, speeds) for side in ("left", "right")}
        self.steady_clock = Clock(clock_type=ClockType.STEADY_TIME)
        self.physics_timer = self.create_timer(dt, self.step, clock=self.steady_clock)
        self.state_timer = self.create_timer(
            1 / state_hz, self.publish_states, clock=self.steady_clock
        )
        self.viewer_timer = self.create_timer(
            1 / viewer_hz, self.sync_viewer, clock=self.steady_clock
        )
        config_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.config_pub = self.create_publisher(String, "/dual/sim/config", config_qos)
        self.publish_config()
        if checked_bool(self.get_parameter("show_viewer").value, "show_viewer"):
            module = cast(ViewerModule, import_module("mujoco.viewer"))
            viewer = module.launch_passive(self.loaded.model, self.loaded.data)
            self.viewer = viewer
            with viewer.lock():
                viewer.cam.lookat[:] = [0.0, 0.18, 0.30]
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 90
                viewer.cam.elevation = -12
            viewer.sync()
        self.get_logger().info(
            "双臂共享场景就绪：中点 world，左 X=-0.21，右 X=+0.21，前方 +Y，上方 +Z。"
        )

    def publish_config(self) -> None:
        message = String()
        message.data = json.dumps(
            {
                "backend": "archer_shared_mujoco_v2",
                "geometry": self.scene_config,
                "clearance": self.clearance,
            }
        )
        self.config_pub.publish(message)

    def cross_arm_failure(self) -> str | None:
        """按实际姿态采样检查跨臂间隙，不含自碰撞、桌面或线缆."""
        self.contact_model.data.qpos[:] = self.loaded.data.qpos
        mujoco.mj_forward(self.contact_model.model, self.contact_model.data)
        data = cast(ContactData, self.contact_model.data)
        geometry = cast(ContactModel, self.contact_model.model)
        api = cast(NameAPI, import_module("mujoco"))
        for index in range(data.ncon):
            contact = data.contact[index]
            a = (
                api.mj_id2name(
                    self.contact_model.model,
                    api.mjtObj.mjOBJ_BODY,
                    int(geometry.geom_bodyid[contact.geom1]),
                )
                or ""
            )
            b = (
                api.mj_id2name(
                    self.contact_model.model,
                    api.mjtObj.mjOBJ_BODY,
                    int(geometry.geom_bodyid[contact.geom2]),
                )
                or ""
            )
            cross = (a.startswith("left_") and b.startswith("right_")) or (
                a.startswith("right_") and b.startswith("left_")
            )
            if cross and contact.dist < self.clearance:
                return f"{a}/{b}：间距 {contact.dist * 1000:.3f} mm"
        return None

    def read_number(self, name: str) -> float:
        return checked_number(self.get_parameter(name).value, name)

    def read_vector(self, name: str, length: int) -> list[float]:
        return checked_vector(self.get_parameter(name).value, name, length)

    def stop_all(self, reason: str) -> None:
        for endpoint in self.endpoints.values():
            endpoint.hold_current()
            endpoint.enabled = False
            endpoint.publish_trajectory_status("STOPPED: " + reason)

    def reset_all(self) -> None:
        self.stop_all("shared scene reset")
        viewer = self.viewer
        with nullcontext() if viewer is None else viewer.lock():
            simulation_time = float(self.loaded.data.time)
            mujoco.mj_resetData(self.loaded.model, self.loaded.data)
            self.loaded.data.time = simulation_time
            for endpoint in self.endpoints.values():
                self.loaded.data.qpos[endpoint.arm.qpos_indices] = endpoint.initial
                self.loaded.data.ctrl[endpoint.arm.actuator_indices] = endpoint.initial
                endpoint.target = endpoint.initial.copy()
                endpoint.paused = False
            mujoco.mj_forward(self.loaded.model, self.loaded.data)
        self.paused = False
        self.contact_ticks = 0
        self.last_clock_ns = None

    def scheduled_pair_valid(self, now_ns: int) -> bool:
        left, right = self.endpoints["left"], self.endpoints["right"]
        scheduled = [
            endpoint for endpoint in (left, right) if endpoint.plan_start_ros_ns is not None
        ]
        if not scheduled:
            return True
        if len(scheduled) == 1:
            start_ns = scheduled[0].plan_start_ros_ns
            other = right if scheduled[0] is left else left
            if other.plan is not None or (
                start_ns is not None and now_ns >= start_ns - 100_000_000
            ):
                self.stop_all("scheduled pair incomplete")
                return False
            return True
        left_plan, right_plan = left.plan, right.plan
        if (
            left.plan_start_ros_ns != right.plan_start_ros_ns
            or left_plan is None
            or right_plan is None
            or not np.array_equal(left_plan[0], right_plan[0])
        ):
            self.stop_all("paired start times or segment times differ")
            return False
        return True

    def step(self) -> None:
        if self.paused or self.finished:
            return
        now_ns = self.get_clock().now().nanoseconds
        scheduled = any(
            endpoint.plan_start_ros_ns is not None for endpoint in self.endpoints.values()
        )
        previous = self.last_clock_ns
        self.last_clock_ns = now_ns
        if scheduled and previous is not None and not 0 <= now_ns - previous <= 250_000_000:
            self.stop_all("clock jump or execution delay")
        self.scheduled_pair_valid(now_ns)
        viewer = self.viewer
        with nullcontext() if viewer is None else viewer.lock():
            for endpoint in self.endpoints.values():
                endpoint.update_target(now_ns)
                self.loaded.data.ctrl[endpoint.arm.actuator_indices] = endpoint.target
            self.loaded.data.qfrc_applied[:] = 0.0
            if self.compensate:
                mujoco.mj_forward(self.loaded.model, self.loaded.data)
                self.loaded.data.qfrc_applied[:] = self.loaded.data.qfrc_bias
            # 整个双臂模型只积分一次，不能为每条手臂分别推进共享时间。
            mujoco.mj_step(self.loaded.model, self.loaded.data)
            finite = np.all(np.isfinite(self.loaded.data.qpos)) and np.all(
                np.isfinite(self.loaded.data.qvel)
            )
        if not finite:
            self.paused = True
            self.stop_all("non-finite physics state; reset required")
            return
        self.contact_ticks += 1
        if self.contact_ticks >= 10:
            self.contact_ticks = 0
            failure = self.cross_arm_failure()
            if failure is not None:
                self.stop_all("跨臂间隙不足：" + failure)
                self.paused = True
                for endpoint in self.endpoints.values():
                    endpoint.paused = True

    def publish_states(self) -> None:
        for endpoint in self.endpoints.values():
            endpoint.publish_state()

    def sync_viewer(self) -> None:
        viewer = self.viewer
        if viewer is not None:
            if not viewer.is_running():
                self.finished = True
            else:
                viewer.sync()

    def destroy_node(self):
        if self.viewer is not None:
            self.viewer.close()
        for endpoint in self.endpoints.values():
            endpoint.destroy_node()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    scene = None
    executor = SingleThreadedExecutor()
    try:
        scene = DualArmScene()
        executor.add_node(scene)
        for endpoint in scene.endpoints.values():
            executor.add_node(endpoint)
        while rclpy.ok() and not scene.finished:
            executor.spin_once(timeout_sec=0.1)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        executor.shutdown()
        if scene is not None:
            scene.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
