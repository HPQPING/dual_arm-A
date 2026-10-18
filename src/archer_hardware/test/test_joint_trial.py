"""离线验证试动文件、双臂计划预检与时间调整，不连接SDK或驱动机械臂."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch

import yaml

from sensor_msgs.msg import JointState

from archer_hardware.commissioning import (
    plan_check_main, plan_retime_main, read_positions, retime_joint_plan, review_joint_plan,
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

    def test_cli_writes_evidence_without_ros_and_preserves_existing_report(self) -> None:
        """离线输出报告哈希，错误计划报告失败且不覆盖旧文件."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            plan, limits, output = root / 'plan.yaml', root / 'limits.yaml', root / 'report.json'
            plan.write_text(yaml.safe_dump(self.plan), encoding='utf-8')
            parameters: dict[str, Any] = {}
            for side in ('left', 'right'):
                bound = self.limits[side]
                for field in ('lower', 'upper', 'max_velocity', 'max_acceleration'):
                    parameters[f'{side}_{field}'] = list(getattr(bound, field))
            parameters['feedback_tolerance'] = 0.0
            limits.write_text(yaml.safe_dump({
                'archer_position_driver': {'ros__parameters': parameters},
            }), encoding='utf-8')
            args = ['--plan-file', str(plan), '--limits-file', str(limits),
                    '--output', str(output)]
            with patch('rclpy.init') as initialize:
                self.assertEqual(plan_check_main(args), 0)
                initialize.assert_not_called()
            report = json.loads(output.read_text(encoding='utf-8'))
            self.assertEqual(len(report['plan_sha256']), 64)
            self.assertFalse(report['commands_sent'])
            saved = output.read_bytes()
            with self.assertRaises(SystemExit):
                plan_check_main(args)
            self.assertEqual(output.read_bytes(), saved)
            plan.write_text('not_a_plan: true', encoding='utf-8')
            args[-1] = str(root / 'failed.json')
            self.assertEqual(plan_check_main(args), 1)
            self.assertFalse(json.loads((root / 'failed.json').read_text())['passed'])

    def test_retime_uses_tighter_arm_and_preserves_positions(self) -> None:
        """左右臂使用同一放慢倍数，右臂更严格的速度约束也必须生效."""
        bounds = dict(self.limits)
        bounds['right'] = PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.001,) * 6,
                                         (0.1,) * 6)
        original = deepcopy(self.plan)
        output, report = retime_joint_plan(self.plan, bounds)
        self.assertTrue(review_joint_plan(output, bounds)['passed'])
        self.assertGreater(report['duration_s'], 4.0)
        self.assertEqual(self.plan, original)
        for side in ('left', 'right'):
            self.assertEqual(output[side]['joint_names'], original[side]['joint_names'])
            for new, old in zip(output[side]['points'], original[side]['points']):
                self.assertEqual(new['positions'], old['positions'])
        self.assertEqual([p['time_from_start'] for p in output['left']['points']],
                         [p['time_from_start'] for p in output['right']['points']])
        self.assertAlmostEqual(report['time_scale'], 9.84375)

    def test_retime_acceleration_and_already_slow_plan(self) -> None:
        """加速度需要按时间平方降低，原本足够慢的计划不加速也不改写时间."""
        output, report = retime_joint_plan(self.plan, self.limits)
        self.assertEqual(output, self.plan)
        self.assertEqual(report['time_scale'], 1.0)
        bound = PositionLimits((-1.0,) * 6, (1.0,) * 6, (1.0,) * 6, (0.0001,) * 6)
        output, report = retime_joint_plan(self.plan, {'left': bound, 'right': bound})
        self.assertTrue(review_joint_plan(output, {'left': bound, 'right': bound})['passed'])
        self.assertLessEqual(max(report['arms']['left']['peak_acceleration_rad_s2']), 0.0001)
        self.assertGreater(report['time_scale'], 10.0)

    def test_retime_does_not_fix_invalid_positions_or_grids(self) -> None:
        """越界角度、错位时间轴和未知插值不能靠放慢绕过."""
        for kind in ('position', 'grid', 'interpolation'):
            plan = deepcopy(self.plan)
            if kind == 'position':
                plan['left']['points'][1]['positions'][0] = 1.001
            elif kind == 'grid':
                plan['right']['points'][1]['time_from_start']['sec'] = 3
            else:
                plan['interpolation'] = 'linear'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                retime_joint_plan(plan, self.limits)
        for margin in (True, 1.0, 2.001, float('nan')):
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                retime_joint_plan(self.plan, self.limits, margin)

    def test_retime_rejects_result_longer_than_execution_limit(self) -> None:
        """极低配置速度需要超过一小时时拒绝生成，不提高配置限制."""
        bound = PositionLimits((-1.0,) * 6, (1.0,) * 6, (1e-9,) * 6, (0.1,) * 6)
        with self.assertRaisesRegex(ValueError, '一小时'):
            retime_joint_plan(self.plan, {'left': bound, 'right': bound})

    def test_retime_cli_hashes_no_ros_and_no_overwrite(self) -> None:
        """只读源文件、排他生成计划与报告，主入口不初始化ROS或SDK."""
        from archer_hardware.operator_client import main
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, config = root / 'source.json', root / 'limits.yaml'
            output, report = root / 'new.json', root / 'report.json'
            source.write_text(json.dumps(self.plan), encoding='utf-8')
            parameters: dict[str, Any] = {'feedback_tolerance': 0.0}
            for side in ('left', 'right'):
                for field in ('lower', 'upper', 'max_velocity', 'max_acceleration'):
                    parameters[f'{side}_{field}'] = list(getattr(self.limits[side], field))
            config.write_text(yaml.safe_dump({
                'archer_position_driver': {'ros__parameters': parameters},
            }), encoding='utf-8')
            args = ['--plan-file', str(source), '--limits-file', str(config),
                    '--output', str(output), '--report', str(report)]
            before = source.read_bytes()
            with patch('rclpy.init') as initialize, patch(
                'archer_hardware.hex_adapter.verified_sdk', side_effect=AssertionError('SDK'),
            ):
                self.assertEqual(main(['plan-retime'] + args), 0)
                initialize.assert_not_called()
            evidence = json.loads(report.read_text(encoding='utf-8'))
            self.assertTrue(evidence['passed'])
            self.assertFalse(evidence['commands_sent'])
            self.assertEqual(len(evidence['output_sha256']), 64)
            self.assertEqual(source.read_bytes(), before)
            saved = output.read_bytes(), report.read_bytes()
            with self.assertRaises(SystemExit):
                plan_retime_main(args)
            self.assertEqual((output.read_bytes(), report.read_bytes()), saved)
            args[5] = str(source)
            with self.assertRaises(SystemExit):
                plan_retime_main(args)
            self.assertEqual(source.read_bytes(), before)
