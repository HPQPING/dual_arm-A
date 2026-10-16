"""Offline FK/Jacobian/IK check. Does not publish robot commands."""

import argparse

import numpy as np

from .kinematics import ArmKinematics, rotation_error


def check(model: ArmKinematics) -> bool:
    seed = np.array([0., -.8, 1.6, -.8, 0., 0.], dtype=np.float64)
    pose, jacobian = model.forward_with_jacobian(seed)
    print('Base frame:', model.base_frame)
    print('TCP frame:', model.tcp_frame)
    print('TCP position in base (m):', pose[:3, 3])
    print('TCP transform:\n', pose)
    finite_difference = np.zeros((6, 6), dtype=np.float64)
    epsilon = 1e-6
    for index in range(6):
        shifted = seed.copy()
        shifted[index] += epsilon
        next_pose = model.forward(shifted)
        finite_difference[:3, index] = (next_pose[:3, 3] - pose[:3, 3]) / epsilon
        finite_difference[3:, index] = rotation_error(next_pose[:3, :3], pose[:3, :3]) / epsilon
    error = float(np.max(np.abs(jacobian - finite_difference)))
    print('Jacobian finite-difference max error:', error)
    known = seed + np.array([.08, .06, -.04, .03, .08, -.06])
    target = model.forward(known)
    result = model.solve_ik(target, seed)
    print('IK success:', result.success)
    print('IK position error (m):', result.position_error)
    print('IK rotation error (rad):', result.orientation_error)
    print('IK joints (rad):', result.joints)
    print('IK message:', result.message)
    passed = error < 1e-5 and result.success
    print('PASS' if passed else 'FAIL')
    return passed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--side', choices=['left', 'right'], default='left')
    parser.add_argument('--urdf', help='Expanded URDF; otherwise use installed project Xacro')
    parser.add_argument('--tcp-xyz', nargs=3, type=float, default=[0., 0., .147])
    parser.add_argument('--mount-xyz', nargs=3, type=float, default=[0., 0., 0.])
    parser.add_argument('--mount-rpy', nargs=3, type=float, default=[0., 0., 0.])
    args = parser.parse_args()
    model = (ArmKinematics.from_urdf(args.urdf, args.side) if args.urdf else
             ArmKinematics.from_description(args.side, args.tcp_xyz, args.mount_xyz, args.mount_rpy))
    if not check(model):
        raise SystemExit(1)


if __name__ == '__main__':
    main()