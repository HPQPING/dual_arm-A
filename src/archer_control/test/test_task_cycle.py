"""检查任务归位的共享时间轴、限位及输入不变性，不连接机械臂."""

import unittest

import numpy as np

from archer_control.dual_arm_trajectory import append_return_home, plan_paired_joints


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


if __name__ == "__main__":
    unittest.main()
