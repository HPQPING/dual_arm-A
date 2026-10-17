"""验证小幅试动文件生成，不连接 SDK 或驱动机械臂."""

import unittest

from sensor_msgs.msg import JointState

from archer_hardware.commissioning import read_positions, target_message, trial_target
from archer_hardware.position_plan import PositionLimits


class JointTrialTests(unittest.TestCase):
    """检查单关节修改、边界恢复、时间限制和实际反馈校验."""

    def setUp(self) -> None:
        """构造虚构约束和反馈，不能作为实机参数使用."""
        self.limits = PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.05,) * 6,
                                     (0.1,) * 6, 0.002)
        self.message = JointState()
        self.message.name = [f'left_joint_{i}' for i in range(1, 7)]
        self.message.position = [0.1 * i for i in range(6)]
        self.message.velocity = [0.0] * 6
        self.message.header.stamp.sec = 100

    def test_single_joint_and_direction(self) -> None:
        """负增量只修改选中的关节，其他关节保留实测值."""
        start = tuple(self.message.position)
        result = trial_target('left', 3, -0.01, 2.0, start, self.limits)
        goal = result['points'][0]['positions']
        self.assertAlmostEqual(goal[2], start[2] - 0.01)
        self.assertEqual(goal[:2] + goal[3:], list(start[:2] + start[3:]))

    def test_duration_extends_for_constraints(self) -> None:
        """短时长自动延长到满足解析速度及加速度的值."""
        result = trial_target('right', 1, 0.02, 0.001, (0.0,) * 6, self.limits)
        duration = result['points'][0]['time_from_start']
        self.assertGreater(duration['sec'] + duration['nanosec'] / 1e9, 1.0)

    def test_invalid_request(self) -> None:
        """拒绝零增量、过大增量、非有限数及错误关节号."""
        for delta in (0.0, 0.021, float('nan'), float('inf'), True):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                trial_target('left', 1, delta, 2.0, (0.0,) * 6, self.limits)
        with self.assertRaises(ValueError):
            trial_target('left', 0, 0.01, 2.0, (0.0,) * 6, self.limits)

    def test_limit_overrun_only_recovers_inward(self) -> None:
        """实际轻微越界允许向内恢复，禁止继续向外移动."""
        start = (-1.001,) + (0.0,) * 5
        trial_target('left', 1, 0.01, 2.0, start, self.limits)
        with self.assertRaises(ValueError):
            trial_target('left', 1, -0.001, 2.0, start, self.limits)

    def test_multiple_outside_joints_require_preparation(self) -> None:
        """多个关节越界时先生成共同收回目标，不允许先动范围内关节."""
        start = (-1.001, 1.001) + (0.0,) * 4
        with self.assertRaises(ValueError):
            trial_target('left', 3, 0.01, 2.0, start, self.limits)
        result = target_message('left', start, self.limits.recovery_goal(start), 2.0, self.limits)
        self.assertEqual(result['points'][0]['positions'], [-0.995, 0.995] + [0.0] * 4)

    def test_feedback_reorder(self) -> None:
        """按照名称重排，不能依赖消息数组原始顺序."""
        original = tuple(self.message.position)
        self.message.name = list(reversed(list(self.message.name)))
        self.message.position = list(reversed(original))
        self.assertEqual(read_positions(self.message, 'left', 100_000_000_000), original)

    def test_stale_moving_or_wrong_arm_rejected(self) -> None:
        """过期、正在运动或混臂的反馈都不能生成试动文件."""
        with self.assertRaises(ValueError):
            read_positions(self.message, 'left', 101_000_000_000)
        self.message.velocity = [0.03] * 6
        with self.assertRaises(ValueError):
            read_positions(self.message, 'left', 100_000_000_000)
        self.message.velocity = [0.0] * 6
        with self.assertRaises(ValueError):
            read_positions(self.message, 'right', 100_000_000_000)
