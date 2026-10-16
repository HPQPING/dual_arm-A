"""Paired trajectory planning only. No ROS commands or hardware access.

World convention: X forward, Y left, Z up. The provisional default has
the right base at world origin and the left base at Y=0.42 m.
Shared segment times align target progress, not backend start clocks.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from .kinematics import ArmKinematics
from .trajectory import (
    DEFAULT_JOINT_LIMITS, DEFAULT_SPEED_LIMITS, FloatArray, JointPlan,
    joint_names, plan_joint_path,
)


def rigid_transform(xyz: ArrayLike, rpy: ArrayLike) -> FloatArray:
    """Transform from local base coordinates to world; meters/radians."""
    translation = np.asarray(xyz, dtype=np.float64)
    angles = np.asarray(rpy, dtype=np.float64)
    if (translation.shape != (3,) or angles.shape != (3,)
            or not np.all(np.isfinite(translation)) or not np.all(np.isfinite(angles))):
        raise ValueError('xyz and rpy must each contain three finite numbers')
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
        raise ValueError('Pose must be a finite 4 by 4 matrix')
    rotation = pose[:3, :3]
    if (not np.allclose(pose[3], [0., 0., 0., 1.], atol=1e-8, rtol=0.)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8, rtol=0.)
            or not np.isclose(np.linalg.det(rotation), 1., atol=1e-8, rtol=0.)):
        raise ValueError('Pose must contain a proper rigid rotation and homogeneous last row')
    return pose


@dataclass(frozen=True)
class DualArmPlan:
    left: JointPlan
    right: JointPlan

    def __post_init__(self) -> None:
        if self.left.joint_names != joint_names('left') or self.right.joint_names != joint_names('right'):
            raise ValueError('Expected a left plan and a right plan, in joint order 1 to 6')
        if not np.array_equal(self.left.times, self.right.times):
            raise ValueError('Both arms must have identical waypoint times')

    @property
    def duration(self) -> float:
        return self.left.duration


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
    """Use the slower arm's duration for EACH matching waypoint segment.

    Unlike stretching only the total duration, this aligns every paired
    waypoint. Quintic interpolation gives both arms the same phase per segment.
    """
    left = plan_joint_path('left', left_start, left_waypoints, left_speeds,
                           left_limits, minimum_duration, time_margin)
    right = plan_joint_path('right', right_start, right_waypoints, right_speeds,
                            right_limits, minimum_duration, time_margin)
    if len(left.times) != len(right.times):
        raise ValueError('Both arms must have the same number of waypoints')
    durations = np.maximum(np.diff(left.times), np.diff(right.times))
    times = np.concatenate((np.array([0.], dtype=np.float64), np.cumsum(durations)))
    return DualArmPlan(
        JointPlan(left.joint_names, times, left.positions),
        JointPlan(right.joint_names, times, right.positions),
    )


class DualArmKinematics:
    """Convert world TCP poses to each arm's base, then solve paired IK.

    This checks IK, joint limits, target speeds and joint branch jumps.
    It does not check collisions, grip forces or coupled object dynamics.
    """

    def __init__(
        self,
        left: ArmKinematics,
        right: ArmKinematics,
        world_from_left: ArrayLike | None = None,
        world_from_right: ArrayLike | None = None,
    ) -> None:
        if left.names != joint_names('left') or right.names != joint_names('right'):
            raise ValueError('Models must be left and right arms respectively')
        self.left = left
        self.right = right
        self.world_from_left = checked_pose(
            rigid_transform([0., 0.42, 0.], [0., 0., 0.])
            if world_from_left is None else world_from_left
        )
        self.world_from_right = checked_pose(
            np.eye(4) if world_from_right is None else world_from_right
        )
        self.world_from_left.setflags(write=False)
        self.world_from_right.setflags(write=False)

    def forward_world(self, left_joints: ArrayLike, right_joints: ArrayLike) -> tuple[FloatArray, FloatArray]:
        return (self.world_from_left @ self.left.forward(left_joints),
                self.world_from_right @ self.right.forward(right_joints))

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
            raise ValueError('Provide 1 to 9999 paired world poses')
        if isinstance(max_joint_step, bool) or not np.isfinite(max_joint_step) or max_joint_step <= 0:
            raise ValueError('max_joint_step must be positive and finite')
        # Validate starting states and speed/limit configuration before running IK.
        plan_joint_path('left', left_start, [left_start], left_speeds,
                        self.left.limits, minimum_duration, time_margin)
        plan_joint_path('right', right_start, [right_start], right_speeds,
                        self.right.limits, minimum_duration, time_margin)
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
                ('left', left_result, left_seed), ('right', right_result, right_seed),
            ):
                if not result.success:
                    raise ValueError(
                        f'{side} IK failed at waypoint {index + 1}: {result.message}; '
                        f'position error={result.position_error:.6f} m; no paired plan returned'
                    )
                if float(np.max(np.abs(result.joints - seed))) > max_joint_step:
                    raise ValueError(f'{side} IK step is too large at waypoint {index + 1}; add intermediate poses')
            left_seed = left_result.joints.copy()
            right_seed = right_result.joints.copy()
            left_points.append(left_seed)
            right_points.append(right_seed)
        return plan_paired_joints(
            left_start, right_start, left_points, right_points,
            left_speeds, right_speeds, self.left.limits, self.right.limits,
            minimum_duration, time_margin,
        )


def peak_target_speeds(plan: JointPlan) -> FloatArray:
    """Quintic peak target speeds for each segment and joint, radians/s."""
    return 1.875 * np.abs(np.diff(plan.positions, axis=0)) / np.diff(plan.times)[:, None]