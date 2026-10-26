"""检查协同事件、连续时间轴、命令分发与失败保留旧转换，不连接真实设备。"""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest import TestCase
from unittest.mock import patch

from archer_hardware.grasp_events import GRASP_MOTION_STAGES, event_schedule
from archer_hardware.grasp_import import main
from archer_hardware.operator_client import main as operator_main
from archer_hardware.task_file import check_task_stages


def inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """构造阶段完整、等待五秒的双侧样本与准备文件。"""
    labels = ['READY', *GRASP_MOTION_STAGES]
    times, elapsed = [], 0
    for name in labels:
        times.append(elapsed)
        elapsed += 5 if name == 'GRASP_WAIT' else 1
    raw = {side: {'points': [{'source_stage': name} for name in labels]}
           for side in ('left', 'right')}
    prepared = {
        side: {'points': [{'time_from_start': {'sec': t, 'nanosec': 0}} for t in times]}
        for side in ('left', 'right')
    }
    summary = {'simple_grasp_rule': {
        'version': 3, 'closed_command': 5.65, 'open_command': 5.5, 'wait_s': 5.,
        'insertion_world_m': [0., .01, 0.], 'retreat_world_m': [0., -.01, 0.],
        'grasp_confirmation': 'assumed_for_demo',
    }}
    return raw, prepared, summary


def test_schedule_never_turns_virtual_attachment_into_a_command():
    raw, prepared, summary = inputs()
    raw['reference_events'] = [{'virtual_attached': 1}]
    schedule = event_schedule(raw, prepared, summary)
    assert [e['target'] for e in schedule['events']] == [5.65, 5.5, 5.5, 5.5, 5.65]
    assert schedule['events'][2]['minimum_wall_s'] == 5.
    assert schedule['execution_ready'] is False
    assert schedule['grasp_confirmation'] == 'assumed_for_demo'
    assert all('virtual_attached' not in e for e in schedule['events'])


def test_desynchronized_arm_stages_are_rejected():
    raw, prepared, summary = inputs()
    raw['right']['points'][3]['source_stage'] = 'WRONG'
    with TestCase().assertRaisesRegex(ValueError, '完全相同'):
        event_schedule(raw, prepared, summary)


def test_shortened_wait_is_rejected():
    raw, prepared, summary = inputs()
    first = next(i for i, p in enumerate(raw['left']['points']) if p['source_stage'] == 'INSERT')
    for side in ('left', 'right'):
        for point in prepared[side]['points'][first:]:
            point['time_from_start']['sec'] -= 1
    with TestCase().assertRaisesRegex(ValueError, '五秒等待'):
        event_schedule(raw, prepared, summary)


def test_obsolete_or_ambiguous_rules_are_rejected():
    case = TestCase()
    for field, value in [('version', True), ('open_command', 2.),
                         ('retreat_world_m', [0., -.002, 0.])]:
        raw, prepared, summary = inputs()
        summary['simple_grasp_rule'][field] = value
        with case.subTest(field=field), case.assertRaisesRegex(ValueError, '新版协同规则'):
            event_schedule(raw, prepared, summary)


def test_task_review_supports_new_stages_without_loosening_circle():
    labels = list(GRASP_MOTION_STAGES)
    check_task_stages(labels, 'cooperative_grasp', simplified_grasp=True)
    with TestCase().assertRaisesRegex(ValueError, 'CLOSE_DEMO'):
        check_task_stages(labels, 'cooperative_grasp')
    with TestCase().assertRaisesRegex(ValueError, 'ARC'):
        check_task_stages(labels, 'split_circle')


def test_operator_dispatch_does_not_initialize_ros():
    with patch('archer_hardware.grasp_import.main', return_value=0) as entry, \
            patch('rclpy.init') as initialize:
        assert operator_main(['grasp-import', '--input', '/tmp/state.csv']) == 0
        entry.assert_called_once_with(['--input', '/tmp/state.csv'])
        initialize.assert_not_called()


def test_failed_conversion_preserves_previous_pair():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        output, report = root / 'state.json', root / 'state_report.json'
        output.write_bytes(b'previous trajectory')
        report.write_bytes(b'previous report')
        with patch('archer_hardware.grasp_import.convert_grasp', side_effect=ValueError('bad')):
            assert main(['--input', str(root / 'state.csv'), '--limits-file',
                         str(root / 'limits.yaml'), '--output-dir', str(root)]) == 1
        assert output.read_bytes() == b'previous trajectory'
        assert report.read_bytes() == b'previous report'


def test_successful_conversion_uses_matching_input_stem():
    with TemporaryDirectory() as folder:
        root = Path(folder)
        payload = json.dumps({'version': 2}).encode()
        report = json.dumps({'duration_s': 10.}).encode()
        with patch('archer_hardware.grasp_import.convert_grasp',
                   return_value=(payload, report)):
            assert main(['--input', str(root / 'state.csv'), '--limits-file',
                         str(root / 'limits.yaml'), '--output-dir', str(root)]) == 0
        assert (root / 'state.json').read_bytes() == payload
        assert (root / 'state_report.json').read_bytes() == report


def test_schedule_does_not_modify_inputs():
    raw, prepared, summary = inputs()
    original = deepcopy((raw, prepared, summary))
    event_schedule(raw, prepared, summary)
    assert (raw, prepared, summary) == original
