"""离线验证试动文件、普通双臂计划校验，不连接SDK或驱动机械臂."""

from copy import deepcopy
from typing import Any
import unittest

from sensor_msgs.msg import JointState

from archer_hardware.commissioning import (
    read_positions, review_joint_plan,
    target_message, trial_target,
)
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


class JointPlanReviewTests(unittest.TestCase):
    """预检完整计划的解析约束，不创建SDK连接或ROS节点."""

    def setUp(self) -> None:
        """虚构共用时间轴和六维位置，用于验证边界与导数峰值."""
        bound = PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.05,) * 6, (0.1,) * 6)
        self.limits = {'left': bound, 'right': bound}
        self.plan: dict[str, Any] = {'version': 1, 'interpolation': 'quintic_stop'}
        for side in ('left', 'right'):
            self.plan[side] = {
                'joint_names': [f'{side}_joint_{i}' for i in range(1, 7)],
                'points': [
                    {'positions': [position] * 6,
                     'time_from_start': {'sec': seconds, 'nanosec': 0}}
                    for seconds, position in ((0, 0.0), (2, 0.01), (4, 0.02))
                ],
            }

    def test_joint_constraints_and_scope(self) -> None:
        """检查解析峰值及范围说明，不把数学通过当作当前驱动可执行."""
        report = review_joint_plan(self.plan, self.limits)
        self.assertTrue(report['passed'])
        self.assertEqual(report['duration_s'], 4.0)
        self.assertAlmostEqual(report['arms']['left']['peak_velocity_rad_s'][0], 0.009375)
        self.assertFalse(report['execution_supported_by_current_driver'])

    def test_fast_segment_and_limit_violation(self) -> None:
        """分别拒绝超速轨迹与越界位置，不改变输入来使其通过."""
        plan = deepcopy(self.plan)
        for side in ('left', 'right'):
            plan[side]['points'][1]['time_from_start'] = {'sec': 0, 'nanosec': 10_000_000}
        with self.assertRaisesRegex(ValueError, '速度超限|加速度超限'):
            review_joint_plan(plan, self.limits)
        plan = deepcopy(self.plan)
        plan['left']['points'][2]['positions'][0] = 1.001
        with self.assertRaisesRegex(ValueError, '超出'):
            review_joint_plan(plan, self.limits)

    def test_acceleration_is_checked_separately(self) -> None:
        """允许峰值速度但加速度超限的短段也必须拒绝."""
        bounds = PositionLimits((-1.0,) * 6, (1.0,) * 6, (1.0,) * 6, (0.0001,) * 6)
        with self.assertRaisesRegex(ValueError, '加速度超限'):
            review_joint_plan(self.plan, {'left': bounds, 'right': bounds})

    def test_preflight_shares_execution_duration_limit(self) -> None:
        """离线预检也使用执行核心的一小时上限，不出现两套规则."""
        for side in ('left', 'right'):
            self.plan[side]['points'][-1]['time_from_start']['sec'] = 3601
        with self.assertRaisesRegex(ValueError, '一小时'):
            review_joint_plan(self.plan, self.limits)

    def test_shared_grid_checks_intermediate_times(self) -> None:
        """两侧总时长相同但中间时间不同仍拒绝."""
        self.plan['right']['points'][1]['time_from_start'] = {'sec': 1, 'nanosec': 800_000_000}
        with self.assertRaisesRegex(ValueError, '相同的纳秒时间轴'):
            review_joint_plan(self.plan, self.limits)

    def test_invalid_identity_interpolation_and_nonfinite_position(self) -> None:
        """拒绝错误关节顺序、未知插值和非有限角度."""
        for kind in ('names', 'interpolation', 'nan'):
            plan = deepcopy(self.plan)
            if kind == 'names':
                plan['left']['joint_names'].reverse()
            elif kind == 'interpolation':
                plan['interpolation'] = 'linear'
            else:
                plan['left']['points'][1]['positions'][0] = float('nan')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                review_joint_plan(plan, self.limits)

    def test_time_must_start_at_zero_and_increase(self) -> None:
        """首点非零、重复时间和布尔秒数均拒绝."""
        for index, stamp in ((0, {'sec': 1, 'nanosec': 0}),
                             (1, {'sec': 0, 'nanosec': 0}),
                             (1, {'sec': True, 'nanosec': 0})):
            plan = deepcopy(self.plan)
            plan['left']['points'][index]['time_from_start'] = stamp
            with self.subTest(index=index, stamp=stamp), self.assertRaises(ValueError):
                review_joint_plan(plan, self.limits)
