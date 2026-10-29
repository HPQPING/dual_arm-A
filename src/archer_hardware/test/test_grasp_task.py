"""检查协同入口许可、输入文件绑定及夹爪连续时序，不创建SDK连接。"""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from archer_hardware import grasp_task
from archer_hardware.file_task import resolve_trajectory_input
from archer_hardware.grasp_events import target_at


def test_full_replay_uses_one_error_bound_for_geometry_and_runtime():
    """两任务各自的预检与运行边界一致；预检失败时绝不能创建SDK连接。"""
    from archer_hardware.file_task import run

    for task_type, expected in (('split_circle', .05), ('cooperative_grasp', .045)):
        future = MagicMock()
        future.done.return_value = True
        future.result.side_effect = RuntimeError('offline preflight deliberately rejected')
        node = MagicMock()
        node.workers, node.limits = {}, {}
        node._interlock_reason, node._paired_gate = '', None
        node._control_period, node.max_excursion = .004, 3.5
        node._preparer.submit.return_value = future
        with patch('rclpy.init') as initialize, patch('rclpy.ok', return_value=True), \
                patch('rclpy.shutdown'), \
                patch('archer_hardware.position_node.PositionControlNode', return_value=node):
            result = run(Path('/prepared'), Path('/source'), Path('/limits'), task_type=task_type)
            assert result == 1
        args = initialize.call_args.kwargs['args']
        params = dict(args[i+1].split(':=', 1) for i, value in enumerate(args) if value == '-p')
        assert params['task_joint_error_rad'] == str(expected)
        assert params['task_startup_joint_error_rad'] == str(expected)
        assert params['tracking_tolerance'] == str(expected)
        assert node._preparer.submit.call_args.kwargs['joint_error_rad'] == expected
        node.handle_request.assert_not_called()


def test_peer_stopped_cannot_hide_fault_or_pending_fault_snapshot():
    """左臂正常取消和旧线程快照不能盖过SDK先锁存的右臂J2故障。"""
    from archer_hardware.file_task import worker_failure
    from archer_hardware.position_worker import WorkerSnapshot

    stopped = WorkerSnapshot('STOPPED', message='按共同参考时间推进轨迹')
    for published in (True, False, None):
        right = WorkerSnapshot('FAULT', error='J2 tracking fault') if published else (
            WorkerSnapshot('RUNNING', message='按共同参考时间推进轨迹'))
        workers = {
            'left': SimpleNamespace(snapshot=lambda: stopped,
                                    _backend=SimpleNamespace(state='STOPPED', last_error='')),
            'right': SimpleNamespace(snapshot=lambda: right, _backend=SimpleNamespace(
                state='RUNNING' if published is None else 'FAULT',
                last_error='J2 tracking fault', _native_error='J2 tracking fault')),
        }
        assert worker_failure(workers) == 'right: J2 tracking fault'


def test_failure_selection_keeps_plain_cancel_and_success():
    """人工取消继续退出，正常运行没有伪造故障。"""
    from archer_hardware.file_task import worker_failure
    from archer_hardware.position_worker import WorkerSnapshot

    running = SimpleNamespace(snapshot=lambda: WorkerSnapshot('RUNNING'), _backend=None)
    assert worker_failure({'left': running, 'right': running}) is None
    stopped = SimpleNamespace(snapshot=lambda: WorkerSnapshot('STOPPED'), _backend=None)
    assert worker_failure({'left': stopped}) == 'left: 任务已停止'


def test_gripper_phase_feedback_reports_values_without_commands():
    """开爪/等待结束报告实际值，不能把阶段名当物理动作完成。"""
    from archer_hardware.file_task import show_gripper_feedback

    gripper = MagicMock()
    gripper.status.return_value = {'command': 5.5, 'sent_command': 5.5,
                                   'actual_position': 5.65, 'feedback_age_s': .002}
    workers = {'left': SimpleNamespace(_backend=SimpleNamespace(gripper=gripper))}
    with patch('builtins.print') as output:
        show_gripper_feedback(workers)
    text = output.call_args.args[0]
    assert '5.5' in text and '5.65' in text and 'left' in text
    gripper.status.assert_called_once()
    gripper.request_enable.assert_not_called()
    gripper.stop.assert_not_called()


class GraspTaskTests(unittest.TestCase):
    def setUp(self):
        self.folder = TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.source = self.root / 'archer_results' / 'example_协同抓取'
        self.converted = self.root / 'archer_results_trans' / self.source.name
        self.source.mkdir(parents=True)
        self.converted.mkdir(parents=True)
        self.config = self.root / 'grasp.yaml'
        self.config.write_text('configuration')
        for name in ('state.csv', 'summary.json'):
            (self.source / name).write_text('test input')
        self.payload = self.converted / 'state.json'
        self.payload.write_text('test prepared input')
        report = {'source_csv': str(self.source / 'state.csv')}
        for file, key in ((self.payload, 'output_sha256'),
                          (self.source / 'state.csv', 'csv_sha256'),
                          (self.source / 'summary.json', 'summary_sha256'),
                          (self.config, 'limits_sha256')):
            report[key] = hashlib.sha256(file.read_bytes()).hexdigest()
        (self.converted / 'state_report.json').write_text(json.dumps(report))
        self.ros = patch('archer_hardware.grasp_task.import_module', return_value=SimpleNamespace(
            remove_ros_args=lambda *, args: args))
        self.budget = patch.object(grasp_task, 'qualify_prepared_return')
        self.budget.start()
        self.addCleanup(self.budget.stop)
        self.ros.start()
        self.addCleanup(self.ros.stop)

    def args(self):
        return ['--execute-authorized-full-grasp', '--trajectory-file', str(self.payload),
                '--limits-file', str(self.config)]

    def test_no_implicit_execution_permission(self):
        with patch.object(grasp_task, 'run') as run, self.assertRaises(SystemExit):
            grasp_task.main(self.args()[1:])
        run.assert_not_called()

    def test_same_runner_with_explicit_cooperative_type(self):
        arguments = self.args()
        with patch.object(grasp_task, 'run', return_value=0) as run:
            self.assertEqual(grasp_task.main(arguments), 0)
        run.assert_called_once_with(
            self.converted, self.source, self.config, trajectory_file=self.payload,
            ros_args=arguments, task_type='cooperative_grasp',
            backend_type=grasp_task.CarryFeedbackAdapter)

    def test_changed_input_rejected_before_creating_node(self):
        (self.source / 'state.csv').write_text('changed after conversion')
        with patch.object(grasp_task, 'run') as run, self.assertRaises(SystemExit):
            grasp_task.main(self.args())
        run.assert_not_called()

    def test_circle_default_does_not_accept_grasp_input(self):
        with self.assertRaisesRegex(ValueError, '仅执行画圆'):
            resolve_trajectory_input(self.payload, None, self.config)

    def test_report_is_not_a_trajectory(self):
        with self.assertRaisesRegex(ValueError, '不是state_report'):
            resolve_trajectory_input(self.payload.with_name('state_report.json'), None,
                                     self.config, expected_csv_name='state.csv')

    def test_gripper_profile_is_continuous_and_holds_during_carry(self):
        schedule = {'policy': 'cooperative_grasp_events_v2', 'phase_start_ns': {
            'OPEN_FOR_GRASP': 10**9, 'INSERT': 2*10**9,
            'CLOSE_BEFORE_HOME': 8*10**9, 'RETURN_HOME': 9*10**9,
        }}
        for elapsed, expected in ((0., 5.65), (1., 5.65), (1.5, 5.575),
                                  (2., 5.5), (5., 5.5), (8., 5.5), (8.5, 5.575), (10., 5.65)):
            with self.subTest(elapsed=elapsed):
                self.assertAlmostEqual(target_at(schedule, elapsed), expected)
        with self.assertRaises(ValueError):
            target_at(schedule, float('nan'))


if __name__ == '__main__':
    unittest.main()
