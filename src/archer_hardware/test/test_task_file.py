"""验证整轮任务文件核对及篡改拒绝，不连接设备或执行动作."""

import csv
import hashlib
import io
import json
from pathlib import Path
from threading import Event
from tempfile import TemporaryDirectory
from typing import Any
import unittest
from unittest.mock import patch

import yaml

from archer_hardware.csv_import import main as prepare_main
from archer_hardware.operator_client import main as operator_main
from archer_hardware.task_file import json_mapping, review_task_files
from archer_hardware.position_node import prepare_file_task_message
from archer_hardware.position_plan import PositionLimits


class TaskFileTests(unittest.TestCase):
    """用完整合成阶段检查文件一致性，不把哈希核对冒充执行许可."""

    def setUp(self) -> None:
        """生成同一轮成功抓取、配置与连续准备文件."""
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.config = self.root / 'limits.yaml'
        params: dict[str, Any] = {'feedback_tolerance': 0.}
        for side in ('left', 'right'):
            for field, value in (('lower', -1.), ('upper', 1.),
                                 ('max_velocity', .05), ('max_acceleration', .1)):
                params[f'{side}_{field}'] = [value] * 6
        self.config.write_text(yaml.safe_dump({
            'archer_position_driver': {'ros__parameters': params},
        }))
        labels = ['APPROACH', 'ALIGN_JOINT6', 'INSERT', 'CLOSE_DEMO', 'LIFT',
                  'RELEASE', 'RETREAT', 'RETURN_HOME', 'COMPLETED']
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['simulation_time_s', 'stage', 'gripper_command', 'virtual_attached',
                         *[f'{s}_joint_{i}_position_rad' for s in ('left', 'right')
                           for i in range(1, 7)]])
        for index, phase in enumerate(labels):
            writer.writerow([index * .05, phase, 0., 0, *([index * .001] * 12)])
        (self.root / 'state.csv').write_text(stream.getvalue())
        self.write('summary.json', {
            'state': 'COMPLETED', 'ros_stage': 'COMPLETED',
            'recording': {'task_name': '协同抓取', 'complete': True, 'saved': True,
                          'task_completed': True, 'sample_count': len(labels)},
        })
        self.assertEqual(prepare_main(['--run-dir', str(self.root), '--limits-file',
                                      str(self.config), '--prepare']), 0)

    def write(self, name: str, data: object) -> None:
        """写入测试JSON，保持有限数值."""
        (self.root / name).write_text(json.dumps(data, allow_nan=False), encoding='utf-8')

    def read(self, name: str) -> dict[str, Any]:
        """读取测试映射."""
        return json.loads((self.root / name).read_text(encoding='utf-8'))

    def review(self) -> dict[str, Any]:
        """使用明确的离线幅度限制审阅这份文件."""
        return review_task_files(self.root, self.config, self.root / 'prepared_path.json',
                                 self.root / 'prepared_path_report.json', max_excursion=.05)

    def repin(self, filename: str, field: str) -> None:
        """重算测试哈希，证明语义拒绝并非仅靠原报告哈希."""
        report = self.read('prepared_path_report.json')
        report[field] = hashlib.sha256((self.root / filename).read_bytes()).hexdigest()
        self.write('prepared_path_report.json', report)

    def test_valid_bundle_is_read_only_and_never_ready(self) -> None:
        """成功只代表文件与关节约束一致，不授予几何、起点或抓取许可."""
        before = {p.name: p.read_bytes() for p in self.root.iterdir()}
        result = self.review()
        self.assertTrue(result['task_files_verified'])
        self.assertEqual(result['task_type'], 'cooperative_grasp')
        for name in ('execution_ready', 'commands_sent', 'geometry_preflight_passed',
                     'actual_start_verified', 'gripper_execution_verified'):
            self.assertIs(result[name], False)
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.root.iterdir()})

    def test_mismatched_snapshots_are_rejected(self) -> None:
        """原CSV、摘要、配置或准备文件变化，原报告均不能继续使用."""
        for name in ('state.csv', 'summary.json', 'limits.yaml', 'prepared_path.json'):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_bytes()
                path.write_bytes(original + b'\n')
                with self.assertRaisesRegex(ValueError, '不一致'):
                    self.review()
                path.write_bytes(original)

    def test_modified_sample_rejected_even_with_rehashed_payload(self) -> None:
        """重新计算哈希仍不能改写原始实际角度."""
        payload = self.read('prepared_path.json')
        payload['left']['points'][4]['positions'][0] += .00001
        self.write('prepared_path.json', payload)
        self.repin('prepared_path.json', 'output_sha256')
        with self.assertRaisesRegex(ValueError, '关节角被改写'):
            self.review()

    def test_failed_summary_cannot_be_promoted_by_new_hash(self) -> None:
        """失败摘要不能通过改报告哈希变成成功任务."""
        summary = self.read('summary.json')
        summary['state'] = 'FAILED'
        self.write('summary.json', summary)
        self.repin('summary.json', 'summary_sha256')
        with self.assertRaisesRegex(ValueError, 'COMPLETED'):
            self.review()

    def test_missing_or_reordered_home_phase_rejected(self) -> None:
        """删除归位阶段并同步篡改阶段报告仍会拒绝."""
        source = self.root / 'state.csv'
        source.write_bytes(source.read_bytes().replace(b'RETURN_HOME', b'APPROACH'))
        report = self.read('prepared_path_report.json')
        for side in ('left', 'right'):
            report['source_stages'][side][7] = 'APPROACH'
        for event in report['reference_events']:
            if event['stage'] == 'RETURN_HOME':
                event['stage'] = 'APPROACH'
        self.write('prepared_path_report.json', report)
        self.repin('state.csv', 'csv_sha256')
        with self.assertRaisesRegex(ValueError, 'RETURN_HOME'):
            self.review()

    def test_reference_event_change_rejected(self) -> None:
        """夹爪参考事件变化必须重新准备，不能仅修改说明字段."""
        report = self.read('prepared_path_report.json')
        report['reference_events'][0]['gripper_command'] = 2.
        self.write('prepared_path_report.json', report)
        with self.assertRaisesRegex(ValueError, '参考事件'):
            self.review()

    def test_uniform_time_claim_and_counts_rechecked(self) -> None:
        """时间伸长与保留点数不能直接相信报告中的自称."""
        original = self.read('prepared_path_report.json')
        for field, value in (
            ('uniform_time_scale', 1.), ('samples_preserved', 8),
            ('source_rows', 100), ('paired_samples', True),
        ):
            with self.subTest(field=field):
                report = dict(original)
                report[field] = value
                self.write('prepared_path_report.json', report)
                with self.assertRaises(ValueError):
                    self.review()
        self.write('prepared_path_report.json', original)

    def test_ambiguous_csv_and_legacy_object_rejected(self) -> None:
        """不自动选择混合CSV，也不把旧参考物协同示意当作圆柱抓取."""
        (self.root / 'tracking.csv').write_text('unused')
        with self.assertRaisesRegex(ValueError, '恰有一种'):
            self.review()
        (self.root / 'tracking.csv').unlink()
        summary = self.read('summary.json')
        summary['recording']['task_name'] = '旧参考物示意'
        self.write('summary.json', summary)
        self.repin('summary.json', 'summary_sha256')
        with self.assertRaisesRegex(ValueError, '元数据'):
            self.review()

    def test_json_ambiguity_rejected(self) -> None:
        """重复字段和非有限JSON不允许依赖读取器的覆盖规则."""
        for content in (b'{"state":1,"state":2}', b'{"q":NaN}', b'[]'):
            with self.assertRaises(ValueError):
                json_mapping(content)

    def test_event_stop_cannot_be_removed_by_repinning(self) -> None:
        """阶段停止点即使从两份报告一起删除，仍须按原CSV重新推导."""
        payload = self.read('prepared_path.json')
        payload['stop_indices'].remove(4)
        self.write('prepared_path.json', payload)
        report = self.read('prepared_path_report.json')
        report['stop_indices'] = payload['stop_indices']
        self.write('prepared_path_report.json', report)
        self.repin('prepared_path.json', 'output_sha256')
        with self.assertRaisesRegex(ValueError, '必要停顿'):
            self.review()

    def test_circle_files_and_legacy_reference_object(self) -> None:
        """完整画圆可通过同一入口，旧cooperative参考物文件不能混作第二任务."""
        (self.root / 'state.csv').unlink()
        (self.root / 'prepared_path.json').unlink()
        (self.root / 'prepared_path_report.json').unlink()
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['feedback_stamp_ns', 'side', 'phase',
                         *[f'actual_J{i}_rad' for i in range(1, 7)]])
        for index, phase in enumerate(('APPROACH', 'ARC', 'RETURN_HOME')):
            for side in ('left', 'right'):
                writer.writerow([1792316000123456789 + index * 50000000,
                                 side, phase, *([.001 * index] * 6)])
        (self.root / 'tracking.csv').write_text(stream.getvalue())
        summary = {'task_state': 'COMPLETED', 'task_name': '画圆',
                   'task_type': 'split_circle', 'reference_object_only': False,
                   'records': {'left': 3, 'right': 3}}
        self.write('summary.json', summary)
        self.assertEqual(prepare_main(['--run-dir', str(self.root), '--limits-file',
                                      str(self.config), '--prepare']), 0)
        self.assertEqual(self.review()['task_type'], 'split_circle')
        summary['task_type'] = 'cooperative'
        summary['reference_object_only'] = True
        self.write('summary.json', summary)
        self.repin('summary.json', 'summary_sha256')
        with self.assertRaisesRegex(ValueError, '参考物'):
            self.review()

    def test_command_does_not_initialize_ros_or_sdk(self) -> None:
        """统一操作入口在初始化ROS之前转入只读任务检查."""
        args = ['task-review', '--run-dir', str(self.root), '--limits-file', str(self.config),
                '--prepared-file', str(self.root / 'prepared_path.json'), '--report-file',
                str(self.root / 'prepared_path_report.json'), '--max-excursion', '.05']
        with patch('rclpy.init') as initialize, patch(
            'archer_hardware.hex_adapter.verified_sdk', side_effect=AssertionError('SDK'),
        ):
            self.assertEqual(operator_main(args), 0)
            initialize.assert_not_called()

    def file_request(self) -> str:
        """定位合成记录，不提供伪造反馈起点字段."""
        return json.dumps({
            'request_id': 'a' * 32, 'run_dir': str(self.root), 'limits_file': str(self.config),
            'prepared_file': str(self.root / 'prepared_path.json'),
            'report_file': str(self.root / 'prepared_path_report.json'),
            'transition_max_excursion': .05,
        })

    def file_limits(self) -> dict[str, PositionLimits]:
        """与合成配置完全相同的约束，用来检查配置串用拒绝."""
        limit = PositionLimits((-1.,) * 6, (1.,) * 6, (.05,) * 6, (.1,) * 6, 0.)
        return {'left': limit, 'right': limit}

    def make_circle_record(self) -> None:
        """替换本测试临时记录为完整画圆阶段，并重新生成绑定哈希的候选."""
        for name in ('state.csv', 'prepared_path.json', 'prepared_path_report.json'):
            (self.root / name).unlink()
        stream = io.StringIO(newline='')
        writer = csv.writer(stream)
        writer.writerow(['feedback_stamp_ns', 'side', 'phase',
                         *[f'actual_J{i}_rad' for i in range(1, 7)]])
        for index, phase in enumerate(('APPROACH', 'ARC', 'RETURN_HOME')):
            for side in ('left', 'right'):
                writer.writerow([1792316000123456789 + index * 50000000,
                                 side, phase, *([.001 * index] * 6)])
        (self.root / 'tracking.csv').write_text(stream.getvalue())
        self.write('summary.json', {
            'task_state': 'COMPLETED', 'task_name': '画圆', 'task_type': 'split_circle',
            'reference_object_only': False, 'records': {'left': 3, 'right': 3},
        })
        self.assertEqual(prepare_main(['--run-dir', str(self.root), '--limits-file',
                                      str(self.config), '--prepare']), 0)

    def test_file_prepare_reads_once_and_preserves_checked_geometry(self) -> None:
        """使用真实文件核对与插值；几何替身只验证调用链，不宣称合成姿态无碰撞."""
        from archer_hardware.task_file import load_reviewed_task, read_snapshot

        self.make_circle_record()
        original = load_reviewed_task(self.root, self.config, self.root / 'prepared_path.json',
                                      self.root / 'prepared_path_report.json', max_excursion=.05)
        starts = {'left': (-.01,) * 6, 'right': (-.02,) * 6}
        with patch('archer_hardware.task_file.read_snapshot', wraps=read_snapshot) as read, \
                patch('archer_hardware.task_geometry.check_task_geometry', return_value={
                    'sampled_cross_arm_table_passed': True,
                }) as geometry:
            result = prepare_file_task_message(self.file_request(), self.file_limits(),
                                               .05, starts, Event())
        self.assertEqual(read.call_count, 5)
        geometry.assert_called_once()
        assert result.continuous is not None and result.details is not None
        self.assertEqual(result.details['request_id'], 'a' * 32)
        self.assertFalse(result.details['gripper_commands_included'])
        for side, plan in result.continuous.items():
            self.assertTrue(plan.preserve_path)
            self.assertEqual(plan.initial, starts[side])
            self.assertEqual(plan.goal, starts[side])
            for expected, actual in zip(original.plans[side].polynomials, plan.polynomials[1:-1]):
                self.assertEqual(expected.tobytes(), actual.tobytes())

    def test_file_prepare_rejects_grasp_without_real_gripper(self) -> None:
        """禁止把虚拟附着事件忽略后当作真实抓取执行."""
        with patch('archer_hardware.task_geometry.check_task_geometry') as geometry:
            with self.assertRaisesRegex(ValueError, '真实夹爪未接入'):
                prepare_file_task_message(self.file_request(), self.file_limits(), .05,
                                          {'left': (0.,) * 6, 'right': (0.,) * 6}, Event())
            geometry.assert_not_called()

    def test_file_prepare_rejects_config_change_or_injected_start(self) -> None:
        """驱动约束串用、额外静态起点和未限定过渡范围均不能进入执行暂存."""
        from dataclasses import replace

        self.make_circle_record()
        limits = self.file_limits()
        changed = {**limits, 'right': replace(limits['right'], max_velocity=(.06,) * 6)}
        with self.assertRaisesRegex(ValueError, '驱动限制不一致'):
            prepare_file_task_message(self.file_request(), changed, .05,
                                      {'left': (0.,) * 6, 'right': (0.,) * 6}, Event())
        request = json.loads(self.file_request())
        request['start'] = {'left': [0.] * 6, 'right': [0.] * 6}
        with self.assertRaisesRegex(ValueError, '未知字段'):
            prepare_file_task_message(json.dumps(request), limits, .05, {}, Event())
        request.pop('start')
        request['transition_max_excursion'] = .06
        with self.assertRaisesRegex(ValueError, '不能大于'):
            prepare_file_task_message(json.dumps(request), limits, .05, {}, Event())

    def test_file_prepare_cancelled_before_or_during_geometry(self) -> None:
        """取消发生在读取前或几何返回前，都不能返回完整可执行对象."""
        cancelled = Event()
        cancelled.set()
        with self.assertRaisesRegex(ValueError, '取消'):
            prepare_file_task_message(self.file_request(), self.file_limits(), .05, {}, cancelled)
        self.make_circle_record()
        cancelled.clear()
        with patch('archer_hardware.task_geometry.check_task_geometry',
                   side_effect=lambda *args, **kwargs: cancelled.set() or {}):
            with self.assertRaisesRegex(ValueError, '禁止返回'):
                prepare_file_task_message(self.file_request(), self.file_limits(), .05,
                                          {'left': (0.,) * 6, 'right': (0.,) * 6}, cancelled)


if __name__ == '__main__':
    unittest.main()
