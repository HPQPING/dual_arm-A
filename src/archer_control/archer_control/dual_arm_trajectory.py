"""
双臂轨迹规划与文件导出，不访问硬件或发送 ROS 指令.

世界坐标与基座变换由调用者传入；共享时间轴不代表硬件同步启动。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable
import json
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from .kinematics import ArmKinematics
from .trajectory import (
    DEFAULT_JOINT_LIMITS,
    DEFAULT_SPEED_LIMITS,
    FloatArray,
    JointPlan,
    joint_names,
    plan_joint_path,
)


def rigid_transform(xyz: ArrayLike, rpy: ArrayLike) -> FloatArray:
    """Transform from local base coordinates to world; meters/radians."""
    translation = np.asarray(xyz, dtype=np.float64)
    angles = np.asarray(rpy, dtype=np.float64)
    if (
        translation.shape != (3,)
        or angles.shape != (3,)
        or not np.all(np.isfinite(translation))
        or not np.all(np.isfinite(angles))
    ):
        raise ValueError("xyz and rpy must each contain three finite numbers")
    roll, pitch, yaw = angles
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    result[:3, 3] = translation
    return result


def checked_pose(values: ArrayLike) -> FloatArray:
    pose = np.asarray(values, dtype=np.float64).copy()
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("Pose must be a finite 4 by 4 matrix")
    rotation = pose[:3, :3]
    if (
        not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0.0)
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.0)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-8, rtol=0.0)
    ):
        raise ValueError("Pose must contain a proper rigid rotation and homogeneous last row")
    return pose


@dataclass(frozen=True)
class DualArmPlan:
    left: JointPlan
    right: JointPlan

    def __post_init__(self) -> None:
        if self.left.joint_names != joint_names("left") or self.right.joint_names != joint_names(
            "right"
        ):
            raise ValueError("Expected a left plan and a right plan, in joint order 1 to 6")
        if not np.array_equal(self.left.times, self.right.times):
            raise ValueError("Both arms must have identical waypoint times")

    @property
    def duration(self) -> float:
        return self.left.duration


def export_paired_plan(plan: DualArmPlan, destination: str | Path) -> Path:
    """导出共享时间轴的关节计划；只创建新文件，不覆盖已有计划."""
    payload: dict[str, object] = {"version": 1, "interpolation": "quintic_stop"}
    continuous = plan.left.velocities is not None
    if continuous != (plan.right.velocities is not None):
        raise ValueError('左右计划必须使用相同插值格式')
    if continuous:
        payload.update(version=2, interpolation='quintic_c2')
        stops = np.ones(len(plan.left.times), dtype=bool)
        for arm in (plan.left, plan.right):
            assert arm.velocities is not None and arm.accelerations is not None
            stops &= np.all(arm.velocities == 0., axis=1) & np.all(arm.accelerations == 0., axis=1)
        payload['stop_indices'] = np.flatnonzero(stops).tolist()
    shared_ns: list[int] | None = None
    for side, arm in (("left", plan.left), ("right", plan.right)):
        times = np.asarray(arm.times, dtype=np.float64)
        positions = np.asarray(arm.positions, dtype=np.float64)
        if (
            arm.joint_names != joint_names(side)
            or times.ndim != 1
            or len(times) < 2
            or positions.shape != (len(times), 6)
            or not np.all(np.isfinite(times))
            or not np.all(np.isfinite(positions))
            or times[0] != 0.0
            or any(float(b) <= float(a) for a, b in zip(times, times[1:]))
            or times[-1] > 3600.0
        ):
            raise ValueError(f"{side} 计划必须包含从零开始的递增时间和有限六关节角，最长一小时")
        rounded = [int(round(float(value) * 1_000_000_000)) for value in times]
        if any(b <= a for a, b in zip(rounded, rounded[1:])):
            raise ValueError("时间点舍入为纳秒后重合，不能导出")
        if shared_ns is not None and rounded != shared_ns:
            raise ValueError("左右臂导出的时间轴必须一致")
        shared_ns = rounded
        payload[side] = {
            "joint_names": list(arm.joint_names),
            "points": [
                {
                    "positions": [float(value) for value in position],
                    **({'velocities': arm.velocities[index].tolist(),
                        'accelerations': arm.accelerations[index].tolist()}
                       if arm.velocities is not None and arm.accelerations is not None else {}),
                    "time_from_start": {
                        "sec": stamp // 1_000_000_000,
                        "nanosec": stamp % 1_000_000_000,
                    },
                }
                for index, (stamp, position) in enumerate(zip(rounded, positions))
            ],
        }
    # 提前完成序列化和校验，避免无效输入留下半份文件。
    content = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    target = Path(destination).expanduser()
    with target.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    return target.resolve()


def plan_paired_joints(
    left_start: ArrayLike,
    right_start: ArrayLike,
    left_waypoints: Iterable[ArrayLike],
    right_waypoints: Iterable[ArrayLike],
    left_speeds: ArrayLike = DEFAULT_SPEED_LIMITS,
    right_speeds: ArrayLike = DEFAULT_SPEED_LIMITS,
    left_limits: ArrayLike = DEFAULT_JOINT_LIMITS,
    right_limits: ArrayLike = DEFAULT_JOINT_LIMITS,
    minimum_duration: float = 1.0,
    time_margin: float = 1.1,
) -> DualArmPlan:
    """
    Use the slower arm's duration for EACH matching waypoint segment.

    Unlike stretching only the total duration, this aligns every paired waypoint. Quintic
    interpolation gives both arms the same phase per segment.
    """
    left = plan_joint_path(
        "left", left_start, left_waypoints, left_speeds, left_limits, minimum_duration, time_margin
    )
    right = plan_joint_path(
        "right",
        right_start,
        right_waypoints,
        right_speeds,
        right_limits,
        minimum_duration,
        time_margin,
    )
    if len(left.times) != len(right.times):
        raise ValueError("Both arms must have the same number of waypoints")
    durations = np.maximum(np.diff(left.times), np.diff(right.times))
    times = np.concatenate((np.array([0.0], dtype=np.float64), np.cumsum(durations)))
    return DualArmPlan(
        JointPlan(left.joint_names, times, left.positions),
        JointPlan(right.joint_names, times, right.positions),
    )


def append_return_home(
    plan: DualArmPlan,
    left_limits: ArrayLike,
    right_limits: ArrayLike,
    left_speeds: ArrayLike = DEFAULT_SPEED_LIMITS,
    right_speeds: ArrayLike = DEFAULT_SPEED_LIMITS,
    minimum_duration: float = 1.0,
    time_margin: float = 1.1,
) -> DualArmPlan:
    """将本次起点作为归位目标；返回新计划，调用者须对完整路径重新预检."""
    home = plan_paired_joints(
        plan.left.positions[-1],
        plan.right.positions[-1],
        [plan.left.positions[0]],
        [plan.right.positions[0]],
        left_speeds,
        right_speeds,
        left_limits,
        right_limits,
        minimum_duration,
        time_margin,
    )
    times = np.asarray(
        np.concatenate((plan.left.times, home.left.times[1:] + plan.duration)), dtype=np.float64
    )
    return DualArmPlan(
        *(
            JointPlan(
                original.joint_names,
                times.copy(),
                np.asarray(
                    np.concatenate((original.positions, returning.positions[1:]), axis=0),
                    dtype=np.float64,
                ),
                None if original.velocities is None else np.concatenate(
                    (original.velocities, np.zeros_like(returning.positions[1:])), axis=0),
                None if original.accelerations is None else np.concatenate(
                    (original.accelerations, np.zeros_like(returning.positions[1:])), axis=0),
            )
            for original, returning in ((plan.left, home.left), (plan.right, home.right))
        )
    )


def continuous_paired_plan(
    plan: DualArmPlan, left_limits: ArrayLike, right_limits: ArrayLike,
    speed_limits: ArrayLike, acceleration_limits: ArrayLike, stops: tuple[int, ...],
    margin: float = 1.2,
) -> DualArmPlan:
    """保留几何路径点，重算共同时间与连续导数，仅在指定阶段边界停车."""
    from .continuous_trajectory import plan_continuous_pair

    bounds = [np.asarray(x, dtype=np.float64) for x in (left_limits, right_limits)]
    curve, _ = plan_continuous_pair(
        plan.left.positions, plan.right.positions, [b[:, 0] for b in bounds],
        [b[:, 1] for b in bounds], [speed_limits] * 2, [acceleration_limits] * 2,
        stops=stops, margin=margin,
    )
    plans = [JointPlan(joint_names(s), curve.times, curve.positions[s],
                       curve.velocities[s], curve.accelerations[s]) for s in ('left', 'right')]
    return DualArmPlan(*plans)


class DualArmKinematics:
    """
    Convert world TCP poses to each arm's base, then solve paired IK.

    This checks IK, joint limits, target speeds and joint branch jumps. It does not check
    collisions, grip forces or coupled object dynamics.
    """

    def __init__(
        self,
        left: ArmKinematics,
        right: ArmKinematics,
        world_from_left: ArrayLike | None = None,
        world_from_right: ArrayLike | None = None,
    ) -> None:
        if left.names != joint_names("left") or right.names != joint_names("right"):
            raise ValueError("Models must be left and right arms respectively")
        self.left = left
        self.right = right
        self.world_from_left = checked_pose(
            rigid_transform([0.0, 0.42, 0.0], [0.0, 0.0, 0.0])
            if world_from_left is None
            else world_from_left
        )
        self.world_from_right = checked_pose(
            np.eye(4) if world_from_right is None else world_from_right
        )
        self.world_from_left.setflags(write=False)
        self.world_from_right.setflags(write=False)

    def forward_world(
        self, left_joints: ArrayLike, right_joints: ArrayLike
    ) -> tuple[FloatArray, FloatArray]:
        return (
            self.world_from_left @ self.left.forward(left_joints),
            self.world_from_right @ self.right.forward(right_joints),
        )

    def plan_world_poses(
        self,
        left_start: ArrayLike,
        right_start: ArrayLike,
        left_poses: Iterable[ArrayLike],
        right_poses: Iterable[ArrayLike],
        left_speeds: ArrayLike = DEFAULT_SPEED_LIMITS,
        right_speeds: ArrayLike = DEFAULT_SPEED_LIMITS,
        minimum_duration: float = 1.0,
        time_margin: float = 1.1,
        max_joint_step: float = 0.35,
    ) -> DualArmPlan:
        """Solve every target first; failure returns no partial trajectory."""
        left_targets = [checked_pose(pose) for pose in left_poses]
        right_targets = [checked_pose(pose) for pose in right_poses]
        if not 1 <= len(left_targets) <= 9999 or len(left_targets) != len(right_targets):
            raise ValueError("Provide 1 to 9999 paired world poses")
        if (
            isinstance(max_joint_step, bool)
            or not np.isfinite(max_joint_step)
            or max_joint_step <= 0
        ):
            raise ValueError("max_joint_step must be positive and finite")
        # Validate starting states and speed/limit configuration before running IK.
        plan_joint_path(
            "left",
            left_start,
            [left_start],
            left_speeds,
            self.left.limits,
            minimum_duration,
            time_margin,
        )
        plan_joint_path(
            "right",
            right_start,
            [right_start],
            right_speeds,
            self.right.limits,
            minimum_duration,
            time_margin,
        )
        left_seed = np.asarray(left_start, dtype=np.float64).copy()
        right_seed = np.asarray(right_start, dtype=np.float64).copy()
        left_points: list[FloatArray] = []
        right_points: list[FloatArray] = []
        left_from_world = np.linalg.inv(self.world_from_left)
        right_from_world = np.linalg.inv(self.world_from_right)
        for index, (left_pose, right_pose) in enumerate(zip(left_targets, right_targets)):
            left_result = self.left.solve_ik(left_from_world @ left_pose, left_seed)
            right_result = self.right.solve_ik(right_from_world @ right_pose, right_seed)
            for side, result, seed in (
                ("left", left_result, left_seed),
                ("right", right_result, right_seed),
            ):
                if not result.success:
                    raise ValueError(
                        f"{side} IK failed at waypoint {index + 1}: {result.message}; "
                        f"position error={result.position_error:.6f} m; no paired plan returned"
                    )
                if float(np.max(np.abs(result.joints - seed))) > max_joint_step:
                    raise ValueError(
                        f"{side} IK step is too large at waypoint {index + 1}; "
                        "add intermediate poses"
                    )
            left_seed = left_result.joints.copy()
            right_seed = right_result.joints.copy()
            left_points.append(left_seed)
            right_points.append(right_seed)
        return plan_paired_joints(
            left_start,
            right_start,
            left_points,
            right_points,
            left_speeds,
            right_speeds,
            self.left.limits,
            self.right.limits,
            minimum_duration,
            time_margin,
        )


def peak_target_speeds(plan: JointPlan) -> FloatArray:
    """解析计算每段关节峰值速度，同时支持停车点和连续导数."""
    if plan.velocities is not None and plan.accelerations is not None:
        from archer_mujoco.joint_interpolation import coefficients, _extrema

        return np.stack([
            np.max(np.abs(_extrema(coefficients(
                plan.positions[i], plan.positions[i + 1],
                plan.velocities[i], plan.velocities[i + 1],
                plan.accelerations[i], plan.accelerations[i + 1], float(h)), float(h), 1)), axis=1)
            for i, h in enumerate(np.diff(plan.times))
        ])
    return 1.875 * np.abs(np.diff(plan.positions, axis=0)) / np.diff(plan.times)[:, None]
