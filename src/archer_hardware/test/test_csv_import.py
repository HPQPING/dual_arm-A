"""验证仿真CSV离线导入、原始时间和角度保留，不连接设备或执行运动."""

import csv
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch

import yaml

from archer_hardware.commissioning import review_joint_plan
from archer_hardware.csv_import import import_samples, prepare_samples
from archer_hardware.operator_client import main
from archer_hardware.position_plan import PositionLimits


def state_csv(times=(0., .05, .1), positions=(0., .001, .002)) -> bytes:
    """生成含有故意无效驱动输入的合成实际状态，证明导入器不读取该列."""
    stream = io.StringIO(newline='')
    names = [f'{side}_joint_{i}_position_rad' for side in ('left', 'right') for i in range(1, 7)]
    writer = csv.DictWriter(stream, fieldnames=[
        'simulation_time_s', 'stage', 'gripper_command', 'virtual_attached',
        *names, 'left_joint_1_drive_input_rad',
    ])
    writer.writeheader()
    for index, (elapsed, q) in enumerate(zip(times, positions)):
        writer.writerow({
            'simulation_time_s': elapsed,
            'stage': 'COMPLETED' if index == len(times) - 1 else 'RUNNING',
            'gripper_command': 2., 'virtual_attached': 0,
            **{name: q for name in names}, 'left_joint_1_drive_input_rad': 'nan',
        })
    return stream.getvalue().encode('utf-8-sig')


def tracking_csv(skew=2000000, omit=False) -> bytes:
    """生成左右分行的合成ROS反馈记录，保持大整数纳秒时间戳."""
    stream = io.StringIO(newline='')
    writer = csv.writer(stream)
    writer.writerow(['feedback_stamp_ns', 'side', 'phase',
                     *[f'actual_J{i}_rad' for i in range(1, 7)]])
    for index in range(3):
        for side, offset in (('left', 0), ('right', skew)):
            if omit and side == 'right' and index == 2:
                continue
            writer.writerow([1792316000123456789 + index * 50000000 + offset,
                             side, 'ARC', *([.001 * index] * 6)])
    return stream.getvalue().encode('utf-8')


class CsvImportTests(unittest.TestCase):
    """检查输入身份、精确时间、双臂配对、边界拒绝与旧执行器隔离."""

    def setUp(self) -> None:
        """仅使用虚构测试限位，不作为实机配置."""
        self.bound = PositionLimits((-1.,) * 6, (1.,) * 6, (.05,) * 6, (.1,) * 6)
        self.limits = {'left': self.bound, 'right': self.bound}

    def test_actual_columns_and_timing_are_preserved(self) -> None:
        """仿真驱动输入为nan也不影响实际角度导入，间隔不因250Hz而缩短."""
        result, report = import_samples(state_csv(), self.limits)
        self.assertTrue(report['import_succeeded'])
        self.assertFalse(report['execution_ready'])
        self.assertEqual(report['duration_s'], .1)
        self.assertEqual(result['left']['points'][1]['positions'], [.001] * 6)
        self.assertEqual(result['right']['points'][1]['time_from_start'],
                         {'sec': 0, 'nanosec': 50000000})
        self.assertEqual(result['interpolation'], 'unassigned')
        with self.assertRaises(ValueError):
            review_joint_plan(result, self.limits)

    def test_same_time_same_position_can_be_deduplicated(self) -> None:
        """阶段切换时同刻同角只合并重复采样，保留后续时间和参考事件."""
        result, report = import_samples(
            state_csv((0., .05, .05, .1), (0., .001, .001, .002)), self.limits,
        )
        self.assertEqual(len(result['left']['points']), 3)
        self.assertEqual(report['duplicate_identical_arm_samples_removed'], 2)
        self.assertEqual(report['duration_s'], .1)

    def test_bad_times_gaps_and_actual_angles_are_rejected(self) -> None:
        """倒退、同刻不同角、采样缺口、非有限数和名义越界均拒绝."""
        cases = [state_csv((0., .05, .04)), state_csv((0., .05, .05)),
                 state_csv((0., 1., 2.)), state_csv(positions=(0., float('nan'), .002)),
                 state_csv(positions=(0., 1.001, .002))]
        for content in cases:
            with self.subTest(content=content[:30]), self.assertRaises(ValueError):
                import_samples(content, self.limits)

    def test_ros_feedback_pairing_keeps_nanosecond_precision(self) -> None:
        """小于配置时差才配对，时间原点不经过大整数浮点转换."""
        result, report = import_samples(tracking_csv(), self.limits)
        self.assertEqual(result['source_time_origin_ns'], 1792316000124456789)
        self.assertEqual(report['max_pair_skew_ms'], 2.)
        self.assertEqual(result['left']['points'][2]['time_from_start']['nanosec'], 100000000)
        for content in (tracking_csv(skew=6000000), tracking_csv(omit=True)):
            with self.assertRaises(ValueError):
                import_samples(content, self.limits)

    def test_rate_diagnostics_do_not_retime_or_claim_continuous_proof(self) -> None:
        """快速采样路径保留原时序并报告需调整，不假装已满足连续执行约束."""
        result, report = import_samples(state_csv(positions=(0., .1, .2)), self.limits)
        self.assertTrue(report['requires_time_adjustment_by_sample_differences'])
        self.assertFalse(report['execution_ready'])
        self.assertEqual(result['left']['points'][-1]['time_from_start']['nanosec'], 100000000)

    def test_wrong_columns_and_duplicate_header_are_rejected(self) -> None:
        """目标角或驱动输入不能冒充实际关节角，重复列名不能被字典静默覆盖."""
        with self.assertRaises(ValueError):
            import_samples(b'time,drive_input_rad\n0,0\n', self.limits)
        content = state_csv().replace(b'stage,', b'stage,stage,', 1)
        with self.assertRaises(ValueError):
            import_samples(content, self.limits)

    def test_prepare_preserves_angles_and_retimes_events(self) -> None:
        """连续准备保留全部实际角度，事件时间同步放慢但不声明可以执行夹爪."""
        samples, _ = import_samples(state_csv(positions=(0., .01, .02)), self.limits)
        before = json.dumps(samples, sort_keys=True)
        plan, report = prepare_samples(samples, self.limits)
        self.assertEqual(plan['version'], 2)
        self.assertEqual(plan['interpolation'], 'quintic_c2')
        self.assertGreater(report['uniform_time_scale'], 1.)
        self.assertTrue(report['continuous_constraints_passed'])
        self.assertFalse(report['execution_ready'])
        self.assertFalse(report['phase_execution_policy_verified'])
        self.assertFalse(report['actual_start_verified'])
        self.assertFalse(report['geometry_preflight_passed'])
        for side in ('left', 'right'):
            self.assertEqual([p['positions'] for p in plan[side]['points']],
                             [p['positions'] for p in samples[side]['points']])
        self.assertGreater(report['reference_events'][-1]['prepared_time_ns'], 100000000)
        self.assertEqual(json.dumps(samples, sort_keys=True), before)
        with self.assertRaises(ValueError):
            review_joint_plan(plan, self.limits)

    def test_gripper_event_remains_reference_and_requires_a_stop(self) -> None:
        """阶段未变但夹爪参考值改变时，也不能跨过事件不停顿."""
        content = state_csv().replace(b'0.05,RUNNING,2.0', b'0.05,RUNNING,1.0')
        samples, _ = import_samples(content, self.limits)
        plan, report = prepare_samples(samples, self.limits)
        self.assertIn(1, plan['stop_indices'])
        self.assertEqual(plan['left']['points'][1]['velocities'], [0.] * 6)
        self.assertEqual(report['reference_events'][1]['gripper_command'], 1.)
        self.assertNotIn('gripper_command', plan)
        self.assertFalse(report['commands_sent'])

    def test_prepare_rejects_unpaired_or_unusable_candidates(self) -> None:
        """准备步骤重新检查共享时间，不静默修复不同步的输入."""
        samples, _ = import_samples(state_csv(), self.limits)
        samples['right']['points'][1]['time_from_start']['nanosec'] += 1
        with self.assertRaisesRegex(ValueError, '共用时间轴'):
            prepare_samples(samples, self.limits)
        samples['version'] = 1
        with self.assertRaises(ValueError):
            prepare_samples(samples, self.limits)

    def test_cli_uses_completed_run_no_ros_sdk_or_overwrite(self) -> None:
        """仅在同一成功目录增加两个JSON，保留原始文件，禁止初始化ROS和SDK."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            source, summary = root / 'state.csv', root / 'summary.json'
            config = root / 'limits.yaml'
            source.write_bytes(state_csv())
            summary.write_text(json.dumps({'state': 'COMPLETED', 'ros_stage': 'COMPLETED'}))
            params: dict[str, Any] = {'feedback_tolerance': 0.}
            for side in ('left', 'right'):
                for field in ('lower', 'upper', 'max_velocity', 'max_acceleration'):
                    params[f'{side}_{field}'] = list(getattr(self.bound, field))
            config.write_text(yaml.safe_dump({
                'archer_position_driver': {'ros__parameters': params},
            }))
            before = source.read_bytes(), summary.read_bytes()
            args = ['csv-import', '--run-dir', str(root), '--limits-file', str(config)]
            with patch('rclpy.init') as initialize, patch(
                'archer_hardware.hex_adapter.verified_sdk', side_effect=AssertionError('SDK'),
            ):
                self.assertEqual(main(args), 0)
                initialize.assert_not_called()
            report = json.loads((root / 'recorded_path_report.json').read_text())
            self.assertEqual(len(report['csv_sha256']), 64)
            self.assertFalse(report['commands_sent'])
            self.assertEqual((source.read_bytes(), summary.read_bytes()), before)
            saved = (root / 'recorded_path.json').read_bytes()
            with self.assertRaises(SystemExit):
                main(args)
            self.assertEqual((root / 'recorded_path.json').read_bytes(), saved)
            with patch('rclpy.init') as initialize, patch(
                'archer_hardware.hex_adapter.verified_sdk', side_effect=AssertionError('SDK'),
            ):
                self.assertEqual(main(args + ['--prepare']), 0)
                initialize.assert_not_called()
            prepared = json.loads((root / 'prepared_path.json').read_text())
            self.assertEqual(prepared['version'], 2)
            prepared_report = json.loads((root / 'prepared_path_report.json').read_text())
            self.assertTrue(prepared_report['preparation_succeeded'])
            self.assertFalse(prepared_report['execution_ready'])
            self.assertEqual((source.read_bytes(), summary.read_bytes()), before)
            params['left_max_velocity'] = [1e-8] * 6
            config.write_text(yaml.safe_dump({
                'archer_position_driver': {'ros__parameters': params},
            }))
            self.assertEqual(main(args + [
                '--prepare', '--output', str(root / 'too_slow.json'),
                '--report', str(root / 'too_slow_report.json'),
            ]), 1)
            self.assertFalse((root / 'too_slow.json').exists())
            rejected = json.loads((root / 'too_slow_report.json').read_text())
            self.assertFalse(rejected['import_succeeded'])
            summary.write_text(json.dumps({'state': 'FAILED'}))
            failure_args = args + ['--output', str(root / 'rejected.json'),
                                   '--report', str(root / 'failed_report.json')]
            self.assertEqual(main(failure_args), 1)
            self.assertFalse((root / 'rejected.json').exists())


if __name__ == '__main__':
    unittest.main()
