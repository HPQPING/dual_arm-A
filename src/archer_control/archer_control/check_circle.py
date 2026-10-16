"""离线预检双臂圆轨迹，不发布运动指令，不连接实机。

模型应已将 link_6 → flange 设置为局部 +Z 8 mm；法兰 → 夹爪零平移。
实机安装偏航角尚未确认，本脚本仅验证这组仿真假设。
仅检查两臂之间的采样间隙，不提供连续碰撞或实机安全保证。
"""

from __future__ import annotations

import argparse
from importlib import import_module
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .circle_planner import CIRCLE_CENTER, CIRCLE_RADIUS, MOUNT_XYZ, TCP_XYZ, plan_circle
from .dual_arm_trajectory import DualArmKinematics, rigid_transform, peak_target_speeds
from .kinematics import ArmKinematics, rotation_error
from .trajectory import sample_joint_plan, DEFAULT_SPEED_LIMITS


class CollisionModel(Protocol):
    geom_margin: NDArray[np.float64]
    geom_bodyid: NDArray[np.int32]


class Contact(Protocol):
    geom1: int
    geom2: int
    dist: float


class Contacts(Protocol):
    def __getitem__(self, index: int) -> Contact: ...


class CollisionData(Protocol):
    ncon: int
    contact: Contacts


class BodyKinds(Protocol):
    mjOBJ_BODY: object


class NameAPI(Protocol):
    mjtObj: BodyKinds
    def mj_id2name(self, model: object, kind: object, index: int) -> str | None: ...


def validate_tool_chain(model: ArmKinematics) -> None:
    """检查工程坐标约定，避免旧模型或重复偏移被悄悄使用；不代表实机标定。"""
    prefix = model.base_frame.removesuffix('base_link')
    expected = (
        (prefix + 'link_6', prefix + 'flange', (0., 0., .008)),
        (prefix + 'flange', prefix + 'gp80_base_link', MOUNT_XYZ),
        (prefix + 'gp80_base_link', prefix + 'tcp', TCP_XYZ),
    )
    for parent, child, translation in expected:
        matches = [joint for joint in model.chain if joint.parent == parent and joint.child == child]
        if len(matches) != 1 or matches[0].kind != 'fixed':
            raise ValueError(f'模型缺少固定坐标关系：{parent} → {child}，请先更新 Xacro')
        joint = matches[0]
        if not np.allclose(joint.origin[:3, 3], translation, atol=1e-8, rtol=0.):
            raise ValueError(f'{parent} → {child} 平移应为 {translation} m；'
                             f'实际为 {joint.origin[:3, 3].tolist()}，请检查偏移是否遗漏或重复')
        # 夹爪安装旋转可按模型调整；其余两个参考帧按当前约定保持同向。
        if child != prefix + 'gp80_base_link' and not np.allclose(
                joint.origin[:3, :3], np.eye(3), atol=1e-8, rtol=0.):
            raise ValueError(f'{parent} → {child} 的旋转应为零，请检查模型坐标约定')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-urdf')
    parser.add_argument('--right-urdf')
    parser.add_argument('--center', nargs=3, type=float, default=list(CIRCLE_CENTER))
    parser.add_argument('--radius', type=float, default=CIRCLE_RADIUS)
    parser.add_argument('--tilt-deg', type=float, default=10.)
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--mount-xyz', nargs=3, type=float, default=list(MOUNT_XYZ))
    parser.add_argument('--mount-yaw-deg', type=float, default=0.)
    parser.add_argument('--tcp-xyz', nargs=3, type=float, default=list(TCP_XYZ))
    parser.add_argument('--left-start', nargs=6, type=float, default=[0., -.8, 1.6, -.8, 0., 0.])
    parser.add_argument('--right-start', nargs=6, type=float, default=[0., -.8, 1.6, -.8, 0., 0.])
    parser.add_argument('--clearance', type=float, default=.01)
    parser.add_argument('--check-dt', type=float, default=.02)
    args = parser.parse_args()
    if (not np.isfinite(args.clearance) or not .001 <= args.clearance <= .05
            or not np.isfinite(args.check_dt) or not .001 <= args.check_dt <= .05):
        parser.error('clearance must be .001..05 m; check-dt must be .001..05 s')
    try:
        mount_rpy = [0., 0., float(np.deg2rad(args.mount_yaw_deg))]
        print(f'Candidate mount XYZ (m): {args.mount_xyz}; mount yaw: {args.mount_yaw_deg:.3f} deg')
        if args.left_urdf or args.right_urdf:
            print('Explicit URDFs use their embedded mounting and TCP definitions.')
        left = (ArmKinematics.from_urdf(args.left_urdf, 'left') if args.left_urdf else
                ArmKinematics.from_description('left', args.tcp_xyz, args.mount_xyz, mount_rpy))
        right = (ArmKinematics.from_urdf(args.right_urdf, 'right') if args.right_urdf else
                 ArmKinematics.from_description('right', args.tcp_xyz, args.mount_xyz, mount_rpy))
        validate_tool_chain(left)
        validate_tool_chain(right)
        world_left = rigid_transform([-.21, 0., 0.], [0., 0., np.pi / 2.])
        world_right = rigid_transform([.21, 0., 0.], [0., 0., np.pi / 2.])
        model = DualArmKinematics(left, right, world_left, world_right)
        result = plan_circle(model, args.left_start, args.right_start, args.center,
                             args.radius, args.samples, float(np.deg2rad(args.tilt_deg)), random_seed=args.seed)
        errors: list[float] = []
        angle_errors: list[float] = []
        for side, plan, targets in (('left', result.plan.left, result.left_targets),
                                     ('right', result.plan.right, result.right_targets)):
            for q, position in zip(plan.positions[1:], targets):
                pose = (world_left @ left.forward(q) if side == 'left' else world_right @ right.forward(q))
                errors.append(float(np.linalg.norm(pose[:3, 3] - position)))
                expected_rotation = result.left_rotation if side == 'left' else result.right_rotation
                angle_errors.append(float(np.linalg.norm(rotation_error(expected_rotation, pose[:3, :3]))))
            if np.any(peak_target_speeds(plan) > np.asarray(DEFAULT_SPEED_LIMITS) + 1e-10):
                raise ValueError('Target speed limit exceeded')
            print(f'{side} J5 range: {np.rad2deg(plan.positions[:, 4].min()):.2f}..'
                  f'{np.rad2deg(plan.positions[:, 4].max()):.2f} deg')
        if max(errors) > 1e-4:
            raise ValueError('TCP waypoint error exceeds 0.1 mm')
        if max(angle_errors) > 1e-3:
            raise ValueError('TCP waypoint orientation error exceeds 0.001 rad')
        # 在几何预检阶段再导入仿真模块，让纯轨迹规划器不依赖 MuJoCo 或 ROS 消息。
        from archer_mujoco.dual_model_loader import load_dual_arm
        from archer_mujoco.model_loader import mujoco
        loaded = load_dual_arm(
            left_urdf=args.left_urdf, right_urdf=args.right_urdf,
            left_base_xyz=[-.21, 0., 0.], right_base_xyz=[.21, 0., 0.],
            left_base_rpy=[0., 0., np.pi / 2.], right_base_rpy=[0., 0., np.pi / 2.],
            tcp_xyz=args.tcp_xyz, mount_xyz=args.mount_xyz, mount_rpy=mount_rpy, enable_contacts=True,
        )
        geometry = cast(CollisionModel, loaded.model)
        collision_data = cast(CollisionData, loaded.data)
        names = cast(NameAPI, import_module('mujoco'))
        # 设置两侧几何体的接触检测裕量，以覆盖所要求的间隙范围。
        geometry.geom_margin[:] = args.clearance * 0.6
        times = set(float(t) for t in result.plan.left.times)
        times.update(float(t) for t in np.linspace(0., result.plan.duration,
                     int(np.ceil(result.plan.duration / args.check_dt)) + 1))
        for elapsed in sorted(times):
            for side, plan in (('left', result.plan.left), ('right', result.plan.right)):
                loaded.data.qpos[loaded.arms[side].qpos_indices] = sample_joint_plan(plan, elapsed)[0]
            mujoco.mj_forward(loaded.model, loaded.data)
            for index in range(collision_data.ncon):
                contact = collision_data.contact[index]
                a = names.mj_id2name(loaded.model, names.mjtObj.mjOBJ_BODY,
                                    int(geometry.geom_bodyid[contact.geom1])) or 'world'
                b = names.mj_id2name(loaded.model, names.mjtObj.mjOBJ_BODY,
                                    int(geometry.geom_bodyid[contact.geom2])) or 'world'
                cross = ((a.startswith('left_') and b.startswith('right_'))
                         or (a.startswith('right_') and b.startswith('left_')))
                if cross and contact.dist < args.clearance:
                    raise ValueError(f'Cross-arm clearance failed at {elapsed:.3f}s: '
                                     f'{a} / {b}, distance {contact.dist * 1000:.3f} mm')
        print(f'PASS: paired circle IK, target speeds, {len(times)} cross-arm clearance samples')
        print(f'Max TCP waypoint error: {max(errors) * 1000:.4f} mm')
        print(f'Max TCP waypoint orientation error: {max(angle_errors):.6f} rad')
        print(f'Approach duration: {result.plan.left.times[1]:.2f} s; total: {result.plan.duration:.2f} s')
        print('Candidate mounting model only; no motion sent. Self-collision, table and cables are not checked here.')
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print('FAIL: ' + str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())