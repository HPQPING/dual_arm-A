"""正式协同入口的默认文件、显式许可和后端注入，不连接设备。"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
from archer_hardware import grasp_task


class PackageTests(unittest.TestCase):
    def test_trajectory_must_be_explicit_before_any_execution(self):
        from contextlib import redirect_stderr
        import io

        with patch.object(
            grasp_task,
            'import_module',
            return_value=SimpleNamespace(remove_ros_args=lambda *, args: args),
        ), patch.object(grasp_task, 'run') as run, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                grasp_task.main(['--execute-authorized-full-grasp'])
        run.assert_not_called()

    def test_check_only_does_not_run_or_connect(self):
        root = Path('/release')
        with patch.object(
            grasp_task,
            'import_module',
            return_value=SimpleNamespace(remove_ros_args=lambda *, args: args),
        ), patch.object(
            grasp_task,
            'resolve_trajectory_input',
            return_value=(root, root, root / 'limits.yaml', root / 'state.json'),
        ), patch.object(
            grasp_task, 'qualify_prepared_return'
        ), patch.object(
            grasp_task, 'run'
        ) as run:
            self.assertEqual(
                grasp_task.main(
                    [
                        '--check-only',
                        '--trajectory-file',
                        str(root / 'state.json'),
                        '--limits-file',
                        str(root / 'limits.yaml'),
                    ]
                ),
                0,
            )
        run.assert_not_called()

    def test_formal_backend_is_passed_explicitly(self):
        from archer_hardware.cooperative_feedback import CarryFeedbackAdapter

        root = Path('/release')
        with patch.object(
            grasp_task,
            'import_module',
            return_value=SimpleNamespace(remove_ros_args=lambda *, args: args),
        ), patch.object(
            grasp_task,
            'resolve_trajectory_input',
            return_value=(root, root, root / 'limits.yaml', root / 'state.json'),
        ), patch.object(
            grasp_task, 'qualify_prepared_return'
        ), patch.object(
            grasp_task, 'run', return_value=7
        ) as run:
            self.assertEqual(
                grasp_task.main(
                    [
                        '--execute-authorized-full-grasp',
                        '--trajectory-file',
                        str(root / 'state.json'),
                        '--limits-file',
                        str(root / 'limits.yaml'),
                    ]
                ),
                7,
            )
        self.assertIs(run.call_args.kwargs['backend_type'], CarryFeedbackAdapter)


class FeedbackLifetimeTests(unittest.TestCase):
    def test_unused_adapter_is_not_retained_by_observation_registry(self):
        import gc
        import weakref
        from archer_hardware.cooperative_feedback import CarryFeedbackAdapter
        from archer_hardware.position_plan import PositionLimits

        limits = PositionLimits((-4.0,) * 6, (4.0,) * 6, (0.5,) * 6, (0.3,) * 6)
        adapter = CarryFeedbackAdapter('ws://localhost:8439', limits)
        self.assertEqual(adapter._tracking_trace.maxlen, 8000)
        reference = weakref.ref(adapter)
        del adapter
        gc.collect()
        self.assertIsNone(reference(), '未连接的任务适配器不应被历史观测全局列表保留')


if __name__ == '__main__':
    unittest.main()
