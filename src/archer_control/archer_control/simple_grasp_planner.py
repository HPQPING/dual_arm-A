"""简化抓取演示规划：接近、仅 J6 对齐、沿 world +Y 插入 1 cm、搬运和归位。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
import math
from typing import cast

import numpy as np

from .circle_planner import inward_rotation
from .model_checks import cross_arm_failure
from .cooperative_planner import interpolate_pose, plan_cooperative, sample_errors
from .dual_arm_trajectory import DualArmKinematics, DualArmPlan, plan_paired_joints, rigid_transform
from .grasp_task_planner import CollisionGeometry, GraspMotion, _join, _reverse
from .kinematics import ArmKinematics
from .trajectory import FloatArray, JointPlan, sample_joint_plan, validate_joints
from archer_mujoco.dual_model_loader import LoadedDualArm, load_dual_arm
from archer_mujoco.grasp_model_loader import GraspData, GraspModel, physics
from archer_mujoco.grasp_scene import GraspConfig
from archer_mujoco.model_loader import mujoco


class PlanningCancelled(RuntimeError):
    """仿真规划收到取消请求，不属于路径不可达。"""


def check_planning_cancelled(cancelled: Callable[[], bool] | None) -> None:
    """在有界计算步骤之间响应取消，禁止采纳未完成的规划。"""
    if cancelled is not None and cancelled():
        raise PlanningCancelled('抓取仿真规划已取消')


@dataclass(frozen=True)
class SimpleGraspMotion(GraspMotion):
    alignment: DualArmPlan
    insertion: DualArmPlan
    collision: LoadedDualArm


def joint6_zero_target(arm: ArmKinematics, actual: FloatArray) -> FloatArray:
    """保留到位实际 J1～J5，将 J6 的绝对目标明确设为零。"""
    result = actual.copy()
    result[5] = 0.
    result = validate_joints(result, arm.limits)
    if np.linalg.norm(arm.forward(result)[:3, 3] - arm.forward(actual)[:3, 3]) > 1e-8:
        raise ValueError('当前 TCP 不在 J6 轴线上，不能只转 J6 保持位置')
    return result


def zero_joint6(paired: DualArmPlan) -> DualArmPlan:
    """进给、搬运及退开阶段均保持 J6 目标为精确零；归位阶段恢复起点角度。"""
    plans: list[JointPlan] = []
    for plan in (paired.left, paired.right):
        positions = plan.positions.copy()
        positions[:, 5] = 0.
        plans.append(JointPlan(plan.joint_names, plan.times, positions))
    return DualArmPlan(plans[0], plans[1])


def check_runtime_paths(contact: LoadedDualArm, paths: tuple[tuple[str, DualArmPlan], ...]) -> None:
    """在独立模型上检查到位反馈修正后的轨迹，不改写运行场景。"""
    floor = physics.mj_name2id(cast(GraspModel, contact.model), physics.mjtObj.mjOBJ_GEOM, 'floor')
    if floor < 0:
        raise ValueError('到位修正预检缺少台面')
    for phase, paired in paths:
        times = set(float(t) for t in paired.left.times)
        times.update(float(t) for t in np.linspace(0., paired.duration, math.ceil(paired.duration / .03) + 1))
        for elapsed in sorted(times):
            for side, plan in (('left', paired.left), ('right', paired.right)):
                contact.data.qpos[contact.arms[side].qpos_indices] = sample_joint_plan(plan, elapsed)[0]
            mujoco.mj_forward(contact.model, contact.data)
            failure = cross_arm_failure(contact, .01)
            if failure:
                raise ValueError(phase + ' 到位修正跨臂预检失败：' + failure)
            hits = cast(GraspData, contact.data)
            if any(floor in (hits.contact[i].geom1, hits.contact[i].geom2) and hits.contact[i].dist < .001
                   for i in range(hits.ncon)):
                raise ValueError(phase + ' 到位修正台面间隙小于 1 mm')


def prepare_simple_motion(model: DualArmKinematics, current: dict[str, FloatArray], config: GraspConfig,
                          left_urdf: str | None = None, right_urdf: str | None = None, *,
                          cancelled: Callable[[], bool] | None = None) -> SimpleGraspMotion:
    """只规划和预检；不设置真实仿真起点，不使用接触力判定夹取。"""
    check_planning_cancelled(cancelled)
    config.validate()
    starts = {side: validate_joints(current[side], arm.limits)
              for side, arm in (('left', model.left), ('right', model.right))}
    obj = rigid_transform([0., .4, config.center_z], [0., 0., 0.])
    grasps = {side: rigid_transform([0., 0., height], [0., 0., 0.])
              for side, height in (('left', config.grasp_offset), ('right', -config.grasp_offset))}
    for grasp in grasps.values():
        # 真 GP80 的夹爪平面是局部 YZ；其法向 X 最终朝 world -Z，长度 Z 朝 +Y。
        # 不能把简化指腹的局部 X 开合轴误当成原 GP80 网格的夹爪平面。
        roll = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=np.float64)
        grasp[:3, :3] = inward_rotation('left', 0.) @ roll
    targets = {side: obj @ grasp for side, grasp in grasps.items()}
    seeds: dict[str, FloatArray] = {}
    for side, arm, base, random_seed in (
        ('left', model.left, model.world_from_left, 7), ('right', model.right, model.world_from_right, 8),
    ):
        generator = np.random.default_rng(random_seed)
        candidates: list[FloatArray] = []
        for index in range(25):
            check_planning_cancelled(cancelled)
            guess = starts[side] if index == 0 else generator.uniform(arm.limits[:, 0], arm.limits[:, 1])
            result = arm.solve_ik(np.linalg.inv(base) @ targets[side], guess)
            if result.success:
                candidates.append(result.joints.copy())
        if not candidates:
            raise ValueError(side + ' 平行夹爪抓取姿态不可达')
        seeds[side] = min(candidates, key=lambda q: float(np.max(np.abs(q - starts[side]))))
        seeds[side][5] = 0.

    # 从抓取点退回 1 cm，构造对齐点。这样再沿 +Y 前进 1 cm 才到工件，避免越过工件。
    pre = {side: targets[side].copy() for side in targets}
    for pose in pre.values():
        pose[1, 3] -= .01
    check_planning_cancelled(cancelled)
    backward = model.plan_world_poses(seeds['left'], seeds['right'],
        [interpolate_pose(targets['left'], pre['left'], float(u)) for u in np.linspace(0., 1., 9)[1:]],
        [interpolate_pose(targets['right'], pre['right'], float(u)) for u in np.linspace(0., 1., 9)[1:]],
        minimum_duration=.125)
    backward = zero_joint6(backward)
    aligned = {'left': backward.left.positions[-1].copy(), 'right': backward.right.positions[-1].copy()}
    before = {side: q.copy() for side, q in aligned.items()}
    for side, arm in (('left', model.left), ('right', model.right)):
        # 到位前使用本轮读取的 J6 角度，不指定 YOZ 平面或固定偏转量。
        before[side][5] = starts[side][5]
        before[side] = validate_joints(before[side], arm.limits)
        # 当前工具链的 TCP 必须位于 J6 轴线上，才允许只动 J6 保持 TCP 位置。
        if np.linalg.norm(arm.forward(before[side])[:3, 3] - arm.forward(aligned[side])[:3, 3]) > 1e-8:
            raise ValueError(side + ' TCP 不在 J6 转轴上，不能仅旋转 J6 对齐')
    alignment = plan_paired_joints(before['left'], before['right'], [aligned['left']], [aligned['right']],
        left_limits=model.left.limits, right_limits=model.right.limits, minimum_duration=1.5)
    insertion = DualArmPlan(_reverse(backward.left), _reverse(backward.right))

    # 张开夹爪的接近先在更远位置调整手臂，再到达对齐点。
    start_poses = model.forward_world(before['left'], before['right'])
    far = {side: pose.copy() for side, pose in zip(('left', 'right'), start_poses)}
    for side in far:
        far[side][1, 3] -= .05 if side == 'left' else .03
    check_planning_cancelled(cancelled)
    outward = model.plan_world_poses(before['left'], before['right'],
        [interpolate_pose(start_poses[0], far['left'], float(u)) for u in np.linspace(0., 1., 13)[1:]],
        [interpolate_pose(start_poses[1], far['right'], float(u)) for u in np.linspace(0., 1., 13)[1:]],
        minimum_duration=.15)
    via = {'left': outward.left.positions[-1].copy(), 'right': outward.right.positions[-1].copy()}
    for side in via:
        via[side][0] = starts[side][0]
    transit = plan_paired_joints(starts['left'], starts['right'],
        [via['left'], outward.left.positions[-1]], [via['right'], outward.right.positions[-1]],
        left_limits=model.left.limits, right_limits=model.right.limits, minimum_duration=3.)
    approach = DualArmPlan(_join(transit.left, _reverse(outward.left)), _join(transit.right, _reverse(outward.right)))

    corners = [obj.copy() for _ in range(5)]
    corners[1][2, 3] += config.lift_height
    corners[2][2, 3] += config.lift_height
    corners[2][1, 3] += config.move_distance
    corners[3][2, 3] += config.lift_height
    poses = [corners[0]]
    for start, goal in zip(corners[:-1], corners[1:]):
        poses.extend(interpolate_pose(start, goal, float(u)) for u in np.linspace(0., 1., 17)[1:])
    check_planning_cancelled(cancelled)
    cooperative = plan_cooperative(model, seeds['left'], seeds['right'], poses, grasps['left'], grasps['right'],
                                   minimum_duration=.25)
    cooperative = replace(cooperative, paired=zero_joint6(cooperative.paired))
    retreat_poses = {side: targets[side].copy() for side in targets}
    for side in retreat_poses:
        retreat_poses[side][1, 3] = far[side][1, 3]
    check_planning_cancelled(cancelled)
    retreat = model.plan_world_poses(cooperative.paired.left.positions[-1], cooperative.paired.right.positions[-1],
        [interpolate_pose(targets['left'], retreat_poses['left'], float(u)) for u in np.linspace(0., 1., 17)[1:]],
        [interpolate_pose(targets['right'], retreat_poses['right'], float(u)) for u in np.linspace(0., 1., 17)[1:]],
        minimum_duration=.15)
    retreat = zero_joint6(retreat)
    home_via = {'left': retreat.left.positions[-1].copy(), 'right': retreat.right.positions[-1].copy()}
    for side in home_via:
        home_via[side][0] = starts[side][0]
    home = plan_paired_joints(retreat.left.positions[-1], retreat.right.positions[-1],
        [home_via['left'], starts['left']], [home_via['right'], starts['right']],
        left_limits=model.left.limits, right_limits=model.right.limits, minimum_duration=3.)

    check_planning_cancelled(cancelled)
    contact = load_dual_arm(left_urdf, right_urdf, left_base_xyz=(-.21, 0., 0.), right_base_xyz=(.21, 0., 0.),
        left_base_rpy=(0., 0., np.pi / 2.), right_base_rpy=(0., 0., np.pi / 2.),
        tcp_xyz=(0., 0., .147021696), enable_contacts=True)
    geometry = cast(CollisionGeometry, contact.model)
    geometry.geom_margin[:] = .006
    floor = physics.mj_name2id(cast(GraspModel, contact.model), physics.mjtObj.mjOBJ_GEOM, 'floor')
    if floor < 0:
        raise ValueError('离散预检缺少台面')
    geometry.geom_pos[floor, 2] = config.table_z
    for phase, paired in (('APPROACH', approach), ('ALIGN_JOINT6', alignment), ('INSERT', insertion),
                           ('COOPERATIVE', cooperative.paired), ('RETREAT', retreat), ('RETURN_HOME', home)):
        print(f'简化演示预检 {phase}，时长 {paired.duration:.2f} s')
        times = set(float(t) for t in paired.left.times)
        times.update(float(t) for t in np.linspace(0., paired.duration, math.ceil(paired.duration / .03) + 1))
        for elapsed in sorted(times):
            check_planning_cancelled(cancelled)
            for side, plan in (('left', paired.left), ('right', paired.right)):
                contact.data.qpos[contact.arms[side].qpos_indices] = sample_joint_plan(plan, elapsed)[0]
            mujoco.mj_forward(contact.model, contact.data)
            failure = cross_arm_failure(contact, .01)
            if failure:
                raise ValueError(phase + ' 跨臂预检失败：' + failure)
            hits = cast(GraspData, contact.data)
            if any(floor in (hits.contact[i].geom1, hits.contact[i].geom2) and hits.contact[i].dist < .001
                   for i in range(hits.ncon)):
                raise ValueError(phase + ' 机械臂与台面间隙小于 1 mm')
            if phase == 'COOPERATIVE':
                errors = sample_errors(model, cooperative, elapsed)
                if max(errors['world_position_m'], errors['relative_position_m']) > .0005:
                    raise ValueError('简化协同插值位置误差超过 0.5 mm')
                if max(errors['world_rotation_rad'], errors['relative_rotation_rad']) > .003:
                    raise ValueError('简化协同插值姿态误差超过 0.003 rad')
    check_planning_cancelled(cancelled)
    for value in starts.values():
        value.setflags(write=False)
    return SimpleGraspMotion(starts, approach, cooperative, retreat, home, alignment, insertion, contact)
