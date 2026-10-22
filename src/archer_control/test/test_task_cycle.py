"""检查任务归位的共享时间轴、限位及输入不变性，不连接机械臂."""

import unittest

import numpy as np

from archer_control.dual_arm_trajectory import (
    append_return_home, plan_paired_joints, continuous_paired_plan, peak_target_speeds,
)


class TaskCycleTests(unittest.TestCase):
    """使用合成关节路径检查归位段，碰撞预检仍由具体任务执行."""

    def test_shared_home_and_unchanged_input(self):
        """较慢的一侧决定归位时间，原计划与已规划任务段保持不变."""
        limits = np.asarray([[-2.0, 2.0]] * 6)
        plan = plan_paired_joints(
            [0.0] * 6,
            [0.1] * 6,
            [[0.2] * 6],
            [[0.4] * 6],
            left_limits=limits,
            right_limits=limits,
        )
        original_times = plan.left.times.copy()
        original_positions = plan.left.positions.copy()
        cycle = append_return_home(plan, limits, limits, [0.2] * 6, [0.1] * 6)
        self.assertGreater(cycle.duration, plan.duration)
        np.testing.assert_array_equal(cycle.left.times, cycle.right.times)
        self.assertTrue(np.all(np.diff(cycle.left.times) > 0.0))
        for before, after in ((plan.left, cycle.left), (plan.right, cycle.right)):
            np.testing.assert_array_equal(after.positions[-1], before.positions[0])
            np.testing.assert_array_equal(after.positions[:-1], before.positions)
        np.testing.assert_array_equal(plan.left.times, original_times)
        np.testing.assert_array_equal(plan.left.positions, original_positions)
        self.assertFalse(np.shares_memory(plan.left.positions, cycle.left.positions))

    def test_outside_home_rejected(self):
        """不得通过追加归位把越界初值作为正式轨迹指令."""
        limits = np.asarray([[-2.0, 2.0]] * 6)
        plan = plan_paired_joints(
            [0.0] * 6,
            [0.0] * 6,
            [[0.2] * 6],
            [[0.2] * 6],
            left_limits=limits,
            right_limits=limits,
        )
        narrow = np.asarray([[0.1, 1.0]] * 6)
        with self.assertRaises(ValueError):
            append_return_home(plan, narrow, limits)


class ContinuousCircleExecutionTests(unittest.TestCase):
    """核对规划、ROS消息、仿真和记录器使用同一条连续曲线。"""

    def setUp(self):
        """构造含接近、圆弧示意和归位的合成路径。"""
        self.bounds = np.asarray([[-1., 1.]] * 6)
        self.raw = plan_paired_joints(
            [0.] * 6, [0.] * 6, [[q] * 6 for q in (.1, .15, .2, .25, 0.)],
            [[q] * 6 for q in (.1, .15, .2, .25, 0.)],
            left_limits=self.bounds, right_limits=self.bounds,
        )
        self.plan = continuous_paired_plan(
            self.raw, self.bounds, self.bounds, [.5] * 6, [.3] * 6, (0, 1, 4, 5))

    def test_only_stage_boundaries_stop(self):
        """位置原样保留，普通圆弧节点速度非零，阶段边界导数为零。"""
        for side in ('left', 'right'):
            arm = getattr(self.plan, side)
            np.testing.assert_array_equal(arm.positions, getattr(self.raw, side).positions)
            np.testing.assert_array_equal(arm.velocities[[0, 1, 4, 5]], 0.)
            np.testing.assert_array_equal(arm.accelerations[[0, 1, 4, 5]], 0.)
            self.assertTrue(np.all(np.linalg.norm(arm.velocities[2:4], axis=1) > 0.))
            self.assertLessEqual(float(np.max(peak_target_speeds(arm))), .5)

    def test_ros_and_simulation_match_planning(self):
        """ROS往返后密集取样一致，仿真不得静默丢掉非零中间导数。"""
        from archer_control.trajectory import to_ros_trajectory, sample_joint_plan
        from archer_mujoco.sim_node import prepare_plan, sample_prepared_plan

        for arm in (self.plan.left, self.plan.right):
            message = to_ros_trajectory(arm)
            parsed = prepare_plan(message, arm.joint_names, arm.positions[0],
                                  self.bounds, [.5] * 6)
            self.assertEqual(len(parsed), 4)
            for t in np.linspace(0., arm.duration, 300):
                np.testing.assert_allclose(sample_joint_plan(arm, float(t))[0],
                                           sample_prepared_plan(parsed, float(t))[0], atol=1e-9)

    def test_incomplete_derivatives_and_segment_overshoot_rejected(self):
        """拒绝缺少导数和仅端点合法但段内越界的消息。"""
        from archer_control.trajectory import to_ros_trajectory
        from archer_mujoco.sim_node import prepare_plan

        arm = self.plan.left
        message = to_ros_trajectory(arm)
        message.points[2].accelerations = []
        with self.assertRaises(ValueError):
            prepare_plan(message, arm.joint_names, arm.positions[0], self.bounds, [.5] * 6)
        message = to_ros_trajectory(arm)
        message.points[2].velocities = [100.] * 6
        with self.assertRaises(ValueError):
            prepare_plan(message, arm.joint_names, arm.positions[0], self.bounds, [.5] * 6)

    def test_legacy_position_message_retains_original_behavior(self):
        """旧单臂位置消息继续可用，但连续画圆入口不会采用旧零速曲线。"""
        from archer_control.trajectory import to_ros_trajectory, sample_joint_plan
        from archer_mujoco.sim_node import prepare_plan, sample_prepared_plan

        arm = self.raw.left
        parsed = prepare_plan(to_ros_trajectory(arm), arm.joint_names, arm.positions[0],
                              self.bounds, [.5] * 6)
        self.assertEqual(len(parsed), 2)
        for t in np.linspace(0., arm.duration, 100):
            np.testing.assert_allclose(sample_joint_plan(arm, float(t))[0],
                                       sample_prepared_plan(parsed, float(t))[0], atol=1e-12)

    def test_export_and_stretch_preserve_derivatives(self):
        """导出正确标记连续格式，延长时长后按比例调整导数。"""
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from archer_control.dual_arm_trajectory import export_paired_plan
        from archer_control.trajectory import stretch_plan

        with TemporaryDirectory() as folder:
            p = export_paired_plan(self.plan, Path(folder) / 'plan.json')
            payload = json.loads(p.read_text())
            self.assertEqual(payload['version'], 2)
            self.assertEqual(payload['stop_indices'], [0, 1, 4, 5])
            self.assertEqual(payload['left']['points'][2]['velocities'],
                             self.plan.left.velocities[2].tolist())
        stretched = stretch_plan(self.plan.left, self.plan.duration * 2.)
        np.testing.assert_allclose(stretched.velocities, self.plan.left.velocities / 2.)
        np.testing.assert_allclose(stretched.accelerations, self.plan.left.accelerations / 4.)


if __name__ == "__main__":
    unittest.main()
