"""离线演示同一物体上、下两处抓取关系，不发布轨迹，不控制夹爪或实机。"""

from __future__ import annotations

import argparse
from typing import cast

import numpy as np

from archer_mujoco.dual_model_loader import load_dual_arm
from archer_mujoco.model_loader import mujoco
from ..model_checks import CollisionModel, validate_tool_chain, cross_arm_failure
from ..circle_planner import MOUNT_XYZ, TCP_XYZ, inward_rotation
from ..cooperative_planner import interpolate_pose, plan_cooperative, sample_errors
from ..dual_arm_trajectory import DualArmKinematics, rigid_transform, peak_target_speeds, export_paired_plan
from ..kinematics import ArmKinematics
from ..trajectory import DEFAULT_SPEED_LIMITS, FloatArray, sample_joint_plan


def find_start(model: ArmKinematics, local_target: FloatArray, seed: int) -> FloatArray:
    """为离线示例寻找假定已抓取时的初值，不把结果作为实机接近轨迹。"""
    rng = np.random.default_rng(seed)
    initial = np.array([0., -.8, 1.6, -.8, 0., 0.], dtype=np.float64)
    solutions: list[FloatArray] = []
    for index in range(33):
        start = initial if index == 0 else rng.uniform(model.limits[:, 0], model.limits[:, 1])
        result = model.solve_ik(local_target, start)
        if result.success:
            solutions.append(result.joints.copy())
    if not solutions:
        raise ValueError('示例抓取起点未找到 IK 解；不代表所有初值下都不可达')
    return min(solutions, key=lambda q: float(np.max(np.abs(q - initial)
                                                  / np.asarray(DEFAULT_SPEED_LIMITS))))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-urdf')
    parser.add_argument('--right-urdf')
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--export-plan', help='预检通过后导出关节计划，仅创建新文件')
    args = parser.parse_args()
    if not 16 <= args.samples <= 128:
        parser.error('samples 必须在 16～128 内')
    try:
        left = (ArmKinematics.from_urdf(args.left_urdf, 'left') if args.left_urdf else
                ArmKinematics.from_description('left', TCP_XYZ, MOUNT_XYZ))
        right = (ArmKinematics.from_urdf(args.right_urdf, 'right') if args.right_urdf else
                 ArmKinematics.from_description('right', TCP_XYZ, MOUNT_XYZ))
        validate_tool_chain(left); validate_tool_chain(right)
        world_left = rigid_transform([-.21, 0., 0.], [0., 0., np.pi / 2.])
        world_right = rigid_transform([.21, 0., 0.], [0., 0., np.pi / 2.])
        model = DualArmKinematics(left, right, world_left, world_right)
        # 虚拟物体中心沿用圆心；左 TCP 位于上部，右 TCP 位于下部，相隔 40 cm。
        # 这只是可替换的运动学示例，不是物体尺寸、接触点或夹紧力的实测值。
        obj_start = rigid_transform([0., .40, .31], [0., 0., 0.])
        obj_goal = rigid_transform([0., .42, .32], [0., 0., float(np.deg2rad(5.))])
        grasps = []
        for side, height in (('left', .20), ('right', -.20)):
            grasp = rigid_transform([0., 0., height], [0., 0., 0.])
            grasp[:3, :3] = inward_rotation(side, float(np.deg2rad(10.)))
            grasps.append(grasp)
        left_start = find_start(left, np.linalg.inv(world_left) @ obj_start @ grasps[0], 7)
        right_start = find_start(right, np.linalg.inv(world_right) @ obj_start @ grasps[1], 8)
        objects = [interpolate_pose(obj_start, obj_goal, float(u))
                   for u in np.linspace(0., 1., args.samples + 1)]
        result = plan_cooperative(model, left_start, right_start, objects, grasps[0], grasps[1])
        for arm_plan in (result.paired.left, result.paired.right):
            if np.any(peak_target_speeds(arm_plan) > np.asarray(DEFAULT_SPEED_LIMITS) + 1e-10):
                raise ValueError('关节目标速度超限')
        loaded = load_dual_arm(
            left_urdf=args.left_urdf, right_urdf=args.right_urdf,
            left_base_xyz=[-.21, 0., 0.], right_base_xyz=[.21, 0., 0.],
            left_base_rpy=[0., 0., np.pi / 2.], right_base_rpy=[0., 0., np.pi / 2.],
            tcp_xyz=TCP_XYZ, mount_xyz=MOUNT_XYZ, enable_contacts=True,
        )
        cast(CollisionModel, loaded.model).geom_margin[:] = .006
        maxima = {key: 0. for key in sample_errors(model, result, 0.)}
        times = set(float(t) for t in result.paired.left.times)
        times.update(float(t) for t in np.linspace(0., result.paired.duration,
                     int(np.ceil(result.paired.duration / .02)) + 1))
        for elapsed in sorted(times):
            for key, error in sample_errors(model, result, elapsed).items():
                maxima[key] = max(maxima[key], error)
            for side, arm_plan in (('left', result.paired.left), ('right', result.paired.right)):
                loaded.data.qpos[loaded.arms[side].qpos_indices] = sample_joint_plan(arm_plan, elapsed)[0]
            mujoco.mj_forward(loaded.model, loaded.data)
            failure = cross_arm_failure(loaded, .01)
            if failure is not None:
                raise ValueError(f't={elapsed:.3f}s 跨臂间隙不足：{failure}')
        if max(maxima['world_position_m'], maxima['relative_position_m']) > .0005:
            raise ValueError('采样位置误差超过示例阈值 0.5 mm，请加密路径或调整 IK')
        if max(maxima['world_rotation_rad'], maxima['relative_rotation_rad']) > .003:
            raise ValueError('采样姿态误差超过示例阈值 0.003 rad')
        print(f'通过：配对协同路径，时长 {result.paired.duration:.3f} s，预检 {len(times)} 个时间点')
        print(f"最大世界 TCP 位置误差：{maxima['world_position_m'] * 1000:.4f} mm")
        print(f"最大两 TCP 相对位置误差：{maxima['relative_position_m'] * 1000:.4f} mm")
        print(f"最大两 TCP 相对姿态误差：{maxima['relative_rotation_rad']:.6f} rad")
        print('仅运动学与离散跨臂预检：未模拟物体、夹紧力、自碰撞、桌面或线缆，未发送动作。')
        if args.export_plan:
            target = export_paired_plan(result.paired, args.export_plan)
            print(f'已导出：{target}；仍需真机限位预检与实际起点核对，未发送动作')
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print('预检失败：' + str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())