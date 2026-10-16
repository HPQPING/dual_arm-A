"""离线规划双臂半圆轨迹及夹爪内倾姿态，不访问 ROS 接口或实机。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

from .dual_arm_trajectory import DualArmKinematics, DualArmPlan, plan_paired_joints
from .kinematics import ArmKinematics
from .trajectory import DEFAULT_SPEED_LIMITS, FloatArray, validate_joints


# 工程采用值：8 mm 放在模型的 link_6 → flange，不再作为夹爪安装偏移。
TCP_XYZ: tuple[float, float, float] = (0., 0., .147021696)
MOUNT_XYZ: tuple[float, float, float] = (0., 0., 0.)
CIRCLE_CENTER: tuple[float, float, float] = (0., .40, .31)
CIRCLE_RADIUS: float = .20


@dataclass(frozen=True)
class CircleResult:
    plan: DualArmPlan
    left_targets: FloatArray
    right_targets: FloatArray
    left_rotation: FloatArray
    right_rotation: FloatArray
    # plan.positions[0] 是起始关节反馈，[1] 是第一个圆弧目标点。
    arc_first_index: int = 1


def inward_rotation(side: str, tilt_radians: float) -> FloatArray:
    """工具 +Z 主要朝向世界 +Y，左右夹爪沿 X 方向分别向内倾斜。

    IK 在保持目标 TCP 位置的同时改变工具姿态。相对各自的 TCP，左夹爪主体
    向 -X 偏移，右夹爪主体向 +X 偏移。需要六个关节联动，不能只调整 J5。
    """
    if side not in ('left', 'right') or not np.isfinite(tilt_radians) or not 0 <= tilt_radians <= np.pi / 4:
        raise ValueError('side must be left/right; inward tilt must be 0..45 degrees')
    angle = tilt_radians if side == 'left' else -tilt_radians
    normal = np.array([np.sin(angle), np.cos(angle), 0.], dtype=np.float64)
    tool_x = np.array([np.cos(angle), -np.sin(angle), 0.], dtype=np.float64)
    return np.asarray(
        np.column_stack((tool_x, np.cross(normal, tool_x), normal)),
        dtype=np.float64,
    )


def semicircle_points(center: ArrayLike, radius: float, samples: int) -> tuple[FloatArray, FloatArray]:
    c = np.asarray(center, dtype=np.float64)
    if c.shape != (3,) or not np.all(np.isfinite(c)):
        raise ValueError('circle center must contain three finite world coordinates')
    if not np.isfinite(radius) or radius <= 0 or c[2] <= radius:
        raise ValueError('Radius must be positive; circle bottom must be above world Z=0')
    if isinstance(samples, bool) or not isinstance(samples, int) or not 16 <= samples <= 256:
        raise ValueError('arc_samples must be an integer from 16 to 256')
    paths: list[FloatArray] = []
    for first_angle in (np.pi / 2., 3. * np.pi / 2.):
        theta = np.linspace(first_angle, first_angle + np.pi, samples + 1)
        points = np.repeat(c[None, :], samples + 1, axis=0)
        points[:, 0] += radius * np.cos(theta)
        points[:, 2] += radius * np.sin(theta)
        paths.append(points)
    return paths[0], paths[1]


def _solve_arc(
    model: ArmKinematics, world_from_base: FloatArray, start: FloatArray,
    points: FloatArray, rotation: FloatArray, rng: np.random.Generator,
    attempts: int, cancelled: Callable[[], bool],
) -> FloatArray:
    inverse = np.linalg.inv(world_from_base)
    target = np.eye(4, dtype=np.float64)
    target[:3, :3] = rotation
    target[:3, 3] = points[0]
    local = inverse @ target
    candidates: list[FloatArray] = []
    for index in range(attempts + 1):
        if cancelled():
            raise ValueError('Planning cancelled')
        seed = start if index == 0 else rng.uniform(model.limits[:, 0], model.limits[:, 1])
        result = model.solve_ik(local, seed)
        if result.success:
            candidates.append(result.joints.copy())
    if not candidates:
        raise ValueError('No start-point IK solution found for ' + model.names[0].split('_')[0])
    candidates.sort(key=lambda q: float(np.max(np.abs(q - start) / np.asarray(DEFAULT_SPEED_LIMITS))))
    last_error = ''
    for seed in candidates:
        joints = [seed.copy()]
        for index, position in enumerate(points[1:], start=1):
            if cancelled():
                raise ValueError('Planning cancelled')
            target[:3, 3] = position
            result = model.solve_ik(inverse @ target, joints[-1])
            if not result.success:
                last_error = f'IK failed at arc sample {index}: {result.message}'
                break
            if float(np.max(np.abs(result.joints - joints[-1]))) > 0.35:
                last_error = f'Joint branch jump at arc sample {index}'
                break
            joints.append(result.joints.copy())
        if len(joints) == len(points):
            return np.asarray(joints, dtype=np.float64)
    raise ValueError('No continuous arc branch: ' + last_error)


def plan_circle(
    model: DualArmKinematics, left_start: ArrayLike, right_start: ArrayLike,
    center: ArrayLike = CIRCLE_CENTER, radius: float = CIRCLE_RADIUS,
    arc_samples: int = 64, inward_tilt: float = np.pi / 18.,
    minimum_duration: float = 0.20, time_margin: float = 1.2,
    ik_attempts: int = 32, random_seed: int = 7,
    cancelled: Callable[[], bool] | None = None,
) -> CircleResult:
    """规划相同分段时长：先在关节空间接近起点，再跟踪两个离散半圆。

    使用同时约束位置和姿态的 IK。起点采用可复现的多初值搜索，但不保证
    找到所有可行解分支。碰撞预检由独立模块执行。
    每个插值段的起点和终点目标速度均为零。
    """
    if isinstance(ik_attempts, bool) or not isinstance(ik_attempts, int) or not 1 <= ik_attempts <= 64:
        raise ValueError('ik_attempts must be 1..64')
    if isinstance(random_seed, bool) or not isinstance(random_seed, int) or random_seed < 0:
        raise ValueError('random_seed must be a nonnegative integer')
    left = validate_joints(left_start, model.left.limits)
    right = validate_joints(right_start, model.right.limits)
    left_points, right_points = semicircle_points(center, radius, arc_samples)
    left_rotation = inward_rotation('left', inward_tilt)
    right_rotation = inward_rotation('right', inward_tilt)
    check_cancel = cancelled if cancelled is not None else lambda: False
    rng = np.random.default_rng(random_seed)
    left_joints = _solve_arc(model.left, model.world_from_left, left, left_points,
                             left_rotation, rng, ik_attempts, check_cancel)
    right_joints = _solve_arc(model.right, model.world_from_right, right, right_points,
                              right_rotation, rng, ik_attempts, check_cancel)
    plan = plan_paired_joints(left, right, left_joints, right_joints,
                             DEFAULT_SPEED_LIMITS, DEFAULT_SPEED_LIMITS,
                             model.left.limits, model.right.limits, minimum_duration, time_margin)
    return CircleResult(plan, left_points, right_points, left_rotation, right_rotation)