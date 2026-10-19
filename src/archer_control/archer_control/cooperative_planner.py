"""双臂固定抓取关系的运动学规划，不包含夹紧力、接触动力学或实机驱动."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from .dual_arm_trajectory import DualArmKinematics, DualArmPlan, checked_pose
from .kinematics import rotation_error
from .trajectory import FloatArray, sample_joint_plan


def interpolate_pose(start: ArrayLike, goal: ArrayLike, fraction: float) -> FloatArray:
    """平移线性插值，旋转沿 SO(3) 最短旋转插值，不直接插值欧拉角."""
    a, b = checked_pose(start), checked_pose(goal)
    if isinstance(fraction, bool) or not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("位姿插值比例必须在 0～1 内")
    vector = rotation_error(b[:3, :3], a[:3, :3])
    angle = float(np.linalg.norm(vector))
    rotation = np.eye(3, dtype=np.float64)
    if angle > 1e-12:
        x, y, z = vector / angle
        skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
        theta = angle * fraction
        rotation += np.sin(theta) * skew + (1.0 - np.cos(theta)) * (skew @ skew)
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation @ a[:3, :3]
    result[:3, 3] = a[:3, 3] + fraction * (b[:3, 3] - a[:3, 3])
    return result


def pose_error(actual: ArrayLike, expected: ArrayLike) -> tuple[float, float]:
    """返回位置误差（米）和姿态误差（弧度）."""
    a, b = checked_pose(actual), checked_pose(expected)
    return (
        float(np.linalg.norm(a[:3, 3] - b[:3, 3])),
        float(np.linalg.norm(rotation_error(b[:3, :3], a[:3, :3]))),
    )


@dataclass(frozen=True)
class CooperativePlan:
    paired: DualArmPlan
    object_poses: tuple[FloatArray, ...]
    object_from_left_tcp: FloatArray
    object_from_right_tcp: FloatArray


def plan_cooperative(
    model: DualArmKinematics,
    left_start: ArrayLike,
    right_start: ArrayLike,
    object_poses: Iterable[ArrayLike],
    object_from_left_tcp: ArrayLike,
    object_from_right_tcp: ArrayLike,
    minimum_duration: float = 0.10,
    time_margin: float = 1.2,
) -> CooperativePlan:
    """
    把物体路径转换为配对 TCP 路径，再进行六轴 IK 与共同时间规划.

    输入变换含义：T_world_TCP = T_world_object × T_object_TCP。 第一项物体位姿必须与两臂当前抓取位姿一致，不自动生成接近或夹紧动作。
    固定抓取关系只在离散目标点约束；插值段误差必须另外采样检查。
    """
    objects = tuple(checked_pose(pose) for pose in object_poses)
    if not 2 <= len(objects) <= 1000:
        raise ValueError("物体路径必须包含 2～1000 个位姿，第一项为当前物体位姿")
    left_grasp, right_grasp = checked_pose(object_from_left_tcp), checked_pose(
        object_from_right_tcp
    )
    actual = model.forward_world(left_start, right_start)
    for side, pose, grasp in zip(("left", "right"), actual, (left_grasp, right_grasp)):
        distance, angle = pose_error(pose, objects[0] @ grasp)
        if distance > 0.0002 or angle > 0.002:
            raise ValueError(
                f"{side} 初始 TCP 与抓取关系不一致：{distance * 1000:.3f} mm，"
                f"{angle:.6f} rad；请先单独验证接近及抓取阶段"
            )
    paired = model.plan_world_poses(
        left_start,
        right_start,
        [pose @ left_grasp for pose in objects[1:]],
        [pose @ right_grasp for pose in objects[1:]],
        minimum_duration=minimum_duration,
        time_margin=time_margin,
    )
    for array in (*objects, left_grasp, right_grasp):
        array.setflags(write=False)
    return CooperativePlan(paired, objects, left_grasp, right_grasp)


def sample_object(plan: CooperativePlan, elapsed: float) -> FloatArray:
    """使用与关节轨迹相同的分段时间和五次进度，得到对应时刻的物体参考位姿."""
    if not np.isfinite(elapsed):
        raise ValueError("采样时间必须为有限数值")
    times = plan.paired.left.times
    if elapsed <= 0.0:
        return plan.object_poses[0].copy()
    if elapsed >= times[-1]:
        return plan.object_poses[-1].copy()
    index = int(np.searchsorted(times, elapsed, side="right")) - 1
    u = float((elapsed - times[index]) / (times[index + 1] - times[index]))
    progress = 6.0 * u**5 - 15.0 * u**4 + 10.0 * u**3
    return interpolate_pose(
        plan.object_poses[index], plan.object_poses[index + 1], min(1.0, max(0.0, progress))
    )


def sample_errors(
    model: DualArmKinematics, plan: CooperativePlan, elapsed: float
) -> dict[str, float]:
    """同时检查世界 TCP 跟踪和两 TCP 相对位姿，避免仅检查一项."""
    left_q = sample_joint_plan(plan.paired.left, elapsed)[0]
    right_q = sample_joint_plan(plan.paired.right, elapsed)[0]
    left, right = model.forward_world(left_q, right_q)
    obj = sample_object(plan, elapsed)
    left_distance, left_angle = pose_error(left, obj @ plan.object_from_left_tcp)
    right_distance, right_angle = pose_error(right, obj @ plan.object_from_right_tcp)
    expected_relative = np.linalg.inv(plan.object_from_left_tcp) @ plan.object_from_right_tcp
    relative_distance, relative_angle = pose_error(np.linalg.inv(left) @ right, expected_relative)
    return {
        "world_position_m": max(left_distance, right_distance),
        "world_rotation_rad": max(left_angle, right_angle),
        "relative_position_m": relative_distance,
        "relative_rotation_rad": relative_angle,
    }
