"""Offline paired planning check. Does not publish or connect to robots.

Run: python3 -m archer_control.check_dual_plan
Or pass --left-urdf PATH --right-urdf PATH to use expanded URDF files.
"""

from __future__ import annotations

import argparse

import numpy as np

from .dual_arm_trajectory import DualArmKinematics, peak_target_speeds, plan_paired_joints
from .kinematics import ArmKinematics, rotation_error
from .trajectory import DEFAULT_SPEED_LIMITS


def check(model: DualArmKinematics) -> bool:
    start = np.array([0., -0.8, 1.6, -0.8, 0., 0.], dtype=np.float64)
    left_start_pose, right_start_pose = model.forward_world(start, start)
    # Small independent TCP excursions with constant world orientation.
    # This tests coordinate conversion and pairing, not a common-object grasp.
    left_targets = []
    right_targets = []
    for offset in (0.003, 0.0):
        left_pose = left_start_pose.copy()
        right_pose = right_start_pose.copy()
        left_pose[:3, 3] += [-offset, 0., offset]
        right_pose[:3, 3] += [-offset, 0., -offset]
        left_targets.append(left_pose)
        right_targets.append(right_pose)
    plan = model.plan_world_poses(start, start, left_targets, right_targets)
    speeds = np.asarray(DEFAULT_SPEED_LIMITS, dtype=np.float64)
    if not np.array_equal(plan.left.times, plan.right.times):
        raise ValueError('Paired waypoint times differ')
    if np.any(peak_target_speeds(plan.left) > speeds + 1e-12) or np.any(peak_target_speeds(plan.right) > speeds + 1e-12):
        raise ValueError('A target speed limit was exceeded')
    maximum_position_error = 0.0
    maximum_rotation_error = 0.0
    for index, (left_target, right_target) in enumerate(zip(left_targets, right_targets), start=1):
        left_pose, right_pose = model.forward_world(plan.left.positions[index], plan.right.positions[index])
        for pose, target in ((left_pose, left_target), (right_pose, right_target)):
            maximum_position_error = max(maximum_position_error, float(np.linalg.norm(pose[:3, 3] - target[:3, 3])))
            maximum_rotation_error = max(maximum_rotation_error, float(np.linalg.norm(rotation_error(target[:3, :3], pose[:3, :3]))))
    if maximum_position_error > 1e-4 or maximum_rotation_error > 1e-3:
        raise ValueError('World TCP pose error exceeds the IK tolerance')
    # Different arms govern different segments: align every segment, not just totals.
    q1 = start.copy(); q1[0] += 0.1
    q2 = q1.copy(); q2[1] += 0.1
    paired = plan_paired_joints(start, start, [q1, q2], [start, q1], minimum_duration=0.1)
    if not np.array_equal(paired.left.times, paired.right.times):
        raise ValueError('Joint pairing failed')
    unreachable = left_start_pose.copy(); unreachable[:3, 3] += [3., 3., 3.]
    try:
        model.plan_world_poses(start, start, [unreachable], [right_start_pose])
    except ValueError:
        pass
    else:
        raise ValueError('Unreachable target was incorrectly accepted')
    print('PASS: paired world IK, shared segment times, target speed limits, unreachable target rejection')
    print(f'World TCP maximum position error: {maximum_position_error:.8f} m')
    print(f'World TCP maximum orientation error: {maximum_rotation_error:.8f} rad')
    print(f'Paired example duration: {plan.duration:.3f} s')
    print('Planning check only; no trajectory was sent to simulation or hardware.')
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-urdf')
    parser.add_argument('--right-urdf')
    parser.add_argument('--left-base-xyz', type=float, nargs=3, default=[0., 0.42, 0.])
    parser.add_argument('--right-base-xyz', type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument('--left-base-rpy', type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument('--right-base-rpy', type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument('--tcp-xyz', type=float, nargs=3, default=[0., 0., 0.147])
    parser.add_argument('--mount-xyz', type=float, nargs=3, default=[0., 0., 0.])
    parser.add_argument('--mount-rpy', type=float, nargs=3, default=[0., 0., 0.])
    args = parser.parse_args()
    try:
        from .dual_arm_trajectory import rigid_transform
        left = (ArmKinematics.from_urdf(args.left_urdf, 'left') if args.left_urdf
                else ArmKinematics.from_description('left', args.tcp_xyz, args.mount_xyz, args.mount_rpy))
        right = (ArmKinematics.from_urdf(args.right_urdf, 'right') if args.right_urdf
                 else ArmKinematics.from_description('right', args.tcp_xyz, args.mount_xyz, args.mount_rpy))
        model = DualArmKinematics(
            left, right, rigid_transform(args.left_base_xyz, args.left_base_rpy),
            rigid_transform(args.right_base_xyz, args.right_base_rpy),
        )
        check(model)
    except (ValueError, OSError, RuntimeError) as exc:
        print(f'FAIL: {exc}')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())