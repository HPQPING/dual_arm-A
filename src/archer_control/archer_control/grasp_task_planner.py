"""从反馈关节快照规划接近、上下协同搬运、张开退开及返回起点."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol, cast

import numpy as np

from .model_checks import CollisionModel, cross_arm_failure
from .circle_planner import inward_rotation
from .cooperative_planner import (
    CooperativePlan,
    interpolate_pose,
    plan_cooperative,
    sample_errors,
    sample_object,
)
from .dual_arm_trajectory import (
    DualArmKinematics,
    DualArmPlan,
    plan_paired_joints,
    rigid_transform,
)
from .trajectory import FloatArray, JointPlan, sample_joint_plan, validate_joints
from archer_mujoco.dual_model_loader import load_dual_arm
from archer_mujoco.grasp_model_loader import GraspData, GraspModel, load_grasp_model, physics
from archer_mujoco.grasp_scene import GraspConfig
from archer_mujoco.model_loader import mujoco


class CollisionGeometry(CollisionModel, Protocol):
    geom_pos: FloatArray


@dataclass(frozen=True)
class GraspMotion:
    start_joints: dict[str, FloatArray]
    approach: DualArmPlan
    cooperative: CooperativePlan
    retreat: DualArmPlan
    home: DualArmPlan


def _reverse(plan: JointPlan) -> JointPlan:
    return JointPlan(plan.joint_names, plan.duration - plan.times[::-1], plan.positions[::-1])


def _join(first: JointPlan, second: JointPlan) -> JointPlan:
    if not np.allclose(first.positions[-1], second.positions[0], atol=1e-9, rtol=0.0):
        raise ValueError("接近轨迹的连接位置不连续")
    return JointPlan(
        first.joint_names,
        np.asarray(
            np.concatenate((first.times, first.duration + second.times[1:])), dtype=np.float64
        ),
        np.asarray(np.concatenate((first.positions, second.positions[1:])), dtype=np.float64),
    )


def prepare_grasp_motion(
    model: DualArmKinematics,
    current: dict[str, FloatArray],
    config: GraspConfig,
    left_urdf: str | None = None,
    right_urdf: str | None = None,
) -> GraspMotion:
    """只计算和预检，不改写实际仿真状态，也不发布运动命令."""
    config.validate()
    starts = {
        side: validate_joints(current[side], arm.limits)
        for side, arm in (("left", model.left), ("right", model.right))
    }
    object_start = rigid_transform([0.0, 0.4, config.center_z], [0.0, 0.0, 0.0])
    grasps: dict[str, FloatArray] = {}
    for side, height in (("left", config.grasp_offset), ("right", -config.grasp_offset)):
        grasp = rigid_transform([0.0, 0.0, height], [0.0, 0.0, 0.0])
        grasp[:3, :3] = inward_rotation(side, float(np.deg2rad(10.0)))
        grasps[side] = grasp
    seeds: dict[str, FloatArray] = {}
    for side, arm, base, random_seed in (
        ("left", model.left, model.world_from_left, 7),
        ("right", model.right, model.world_from_right, 8),
    ):
        target = np.linalg.inv(base) @ object_start @ grasps[side]
        generator = np.random.default_rng(random_seed)
        candidates: list[FloatArray] = []
        for index in range(33):
            guess = (
                starts[side]
                if index == 0
                else generator.uniform(arm.limits[:, 0], arm.limits[:, 1])
            )
            solution = arm.solve_ik(target, guess)
            if solution.success:
                candidates.append(solution.joints.copy())
        if not candidates:
            raise ValueError(side + " 抓取姿态没有找到 IK 解")
        seeds[side] = min(candidates, key=lambda q: float(np.max(np.abs(q - starts[side]))))

    # 沿各自工具接近方向退开；上部 6 cm、下部 4.5 cm，兼顾张开指腹和下臂可达性。
    targets = {side: object_start @ grasps[side] for side in ("left", "right")}
    backoff: dict[str, list[FloatArray]] = {"left": [], "right": []}
    for side in ("left", "right"):
        distance = 0.06 if side == "left" else 0.045
        for fraction in np.linspace(0.0, 1.0, 17)[1:]:
            pose = targets[side].copy()
            pose[:3, 3] -= pose[:3, 2] * distance * float(fraction)
            backoff[side].append(pose)
    outward = model.plan_world_poses(
        seeds["left"], seeds["right"], backoff["left"], backoff["right"], minimum_duration=0.15
    )
    # 先在当前基座转角下调整其余关节，再转向物件，避免所有关节同时转动时扫过圆柱。
    via = {"left": outward.left.positions[-1].copy(), "right": outward.right.positions[-1].copy()}
    for side in ("left", "right"):
        via[side][0] = starts[side][0]
    transit = plan_paired_joints(
        starts["left"],
        starts["right"],
        [via["left"], outward.left.positions[-1]],
        [via["right"], outward.right.positions[-1]],
        left_limits=model.left.limits,
        right_limits=model.right.limits,
        minimum_duration=3.0,
    )
    approach = DualArmPlan(
        _join(transit.left, _reverse(outward.left)), _join(transit.right, _reverse(outward.right))
    )

    corners = [object_start.copy() for _ in range(5)]
    corners[1][2, 3] += config.lift_height
    corners[2][2, 3] += config.lift_height
    corners[2][1, 3] += config.move_distance
    corners[3][2, 3] += config.lift_height
    poses = [corners[0]]
    for start, end in zip(corners[:-1], corners[1:]):
        poses.extend(interpolate_pose(start, end, float(u)) for u in np.linspace(0.0, 1.0, 17)[1:])
    cooperative = plan_cooperative(
        model,
        seeds["left"],
        seeds["right"],
        poses,
        grasps["left"],
        grasps["right"],
        minimum_duration=0.25,
    )
    retreat = model.plan_world_poses(
        cooperative.paired.left.positions[-1],
        cooperative.paired.right.positions[-1],
        backoff["left"],
        backoff["right"],
        minimum_duration=0.15,
    )
    home_via = {
        "left": retreat.left.positions[-1].copy(),
        "right": retreat.right.positions[-1].copy(),
    }
    for side in ("left", "right"):
        home_via[side][0] = starts[side][0]
    home = plan_paired_joints(
        retreat.left.positions[-1],
        retreat.right.positions[-1],
        [home_via["left"], starts["left"]],
        [home_via["right"], starts["right"]],
        left_limits=model.left.limits,
        right_limits=model.right.limits,
        minimum_duration=3.0,
    )

    # 独立预检模型：原机械臂网格检查跨臂间隙，专用指腹模型检查非抓取阶段误碰物件。
    contact = load_dual_arm(
        left_urdf,
        right_urdf,
        left_base_xyz=(-0.21, 0.0, 0.0),
        right_base_xyz=(0.21, 0.0, 0.0),
        left_base_rpy=(0.0, 0.0, np.pi / 2.0),
        right_base_rpy=(0.0, 0.0, np.pi / 2.0),
        tcp_xyz=(0.0, 0.0, 0.147021696),
        enable_contacts=True,
    )
    geometry = cast(CollisionGeometry, contact.model)
    geometry.geom_margin[:] = 0.006
    # 按名称查找台面，并使用本轮确定的世界高度。
    floor = physics.mj_name2id(cast(GraspModel, contact.model), physics.mjtObj.mjOBJ_GEOM, "floor")
    if floor < 0:
        raise ValueError("预检场景缺少工作平面")
    geometry.geom_pos[floor, 2] = config.table_z
    proxy = load_grasp_model(starts["left"], starts["right"], left_urdf, right_urdf, config)
    for phase, paired in (
        ("APPROACH", approach),
        ("COOPERATIVE", cooperative.paired),
        ("RETREAT", retreat),
        ("RETURN_HOME", home),
    ):
        print(f"预检 {phase}，轨迹时长 {paired.duration:.2f} s")
        times = set(float(t) for t in paired.left.times)
        times.update(
            float(t)
            for t in np.linspace(0.0, paired.duration, math.ceil(paired.duration / 0.02) + 1)
        )
        for elapsed in sorted(times):
            for side, plan in (("left", paired.left), ("right", paired.right)):
                q = sample_joint_plan(plan, elapsed)[0]
                contact.data.qpos[contact.arms[side].qpos_indices] = q
                proxy.data.qpos[proxy.arms[side].qpos_indices] = q
                proxy.data.qpos[proxy.finger_positions[side]] = (
                    0.0 if phase != "COOPERATIVE" else 0.04 - config.object_diameter / 2.0
                )
            mujoco.mj_forward(contact.model, contact.data)
            failure = cross_arm_failure(contact, 0.01)
            if failure is not None:
                raise ValueError(phase + " 离散跨臂预检失败：" + failure)
            hits = cast(GraspData, contact.data)
            for index in range(hits.ncon):
                hit = hits.contact[index]
                if floor in (hit.geom1, hit.geom2) and hit.dist < 0.001:
                    raise ValueError(f"{phase} t={elapsed:.3f}s 机械臂与台面间隙小于 1 mm")
            obj = sample_object(cooperative, elapsed) if phase == "COOPERATIVE" else object_start
            proxy.data.qpos[proxy.object_qpos: proxy.object_qpos + 3] = obj[:3, 3]
            physics.mj_forward(proxy.model, proxy.data)
            if phase != "COOPERATIVE":
                for index in range(proxy.data.ncon):
                    hit = proxy.data.contact[index]
                    if (
                        proxy.object_geom in (hit.geom1, hit.geom2)
                        and (hit.geom1 in proxy.finger_geoms or hit.geom2 in proxy.finger_geoms)
                        and hit.dist < 0.0005
                    ):
                        other = hit.geom2 if hit.geom1 == proxy.object_geom else hit.geom1
                        raise ValueError(
                            f"{phase} t={elapsed:.3f}s 张开指腹 {proxy.finger_geoms[other]} "
                            f"会误碰圆柱，距离={hit.dist * 1000.:.3f}mm，请调整接近路径"
                        )
            else:
                errors = sample_errors(model, cooperative, elapsed)
                if (
                    max(errors["world_position_m"], errors["relative_position_m"]) > 0.0005
                    or max(errors["world_rotation_rad"], errors["relative_rotation_rad"]) > 0.003
                ):
                    raise ValueError("协同插值预检超限")
    for q in starts.values():
        q.setflags(write=False)
    return GraspMotion(starts, approach, cooperative, retreat, home)
