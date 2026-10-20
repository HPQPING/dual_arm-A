"""离线验证显式操作客户端，服务后端仅为内存替身."""

import json
from threading import Thread
import unittest
from unittest.mock import patch

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from archer_hardware.operator_client import (
    OperatorClient, main, operation_result, request_path,
)


def status(identifier: str, phase: str, state: str = 'HOLDING') -> dict:
    """生成最小操作状态，与设备无关."""
    return {'interlock': '', 'arms': {'left': {
        'command_id': identifier, 'phase': phase, 'state': state,
        'message': '', 'error': '',
    }}}


class OperatorResultTests(unittest.TestCase):
    """检查请求作用域与结果编号匹配."""

    def test_plan_review_dispatch_never_initializes_ros(self) -> None:
        """离线子命令转交现有联调模块，不创建操作客户端或连接."""
        with patch('archer_hardware.commissioning.plan_check_main', return_value=0) as review:
            with patch('rclpy.init') as initialize:
                self.assertEqual(main(['plan-check', '--help']), 0)
            review.assert_called_once_with(['--help'])
            initialize.assert_not_called()

    def test_commissioning_subcommands_do_not_start_operator(self) -> None:
        """集中入口转发原有工具，不自动启动控制客户端或 ROS 上下文."""
        for command, function in (('audit', 'audit_main'), ('trial', 'trial_main')):
            with self.subTest(command=command), patch(
                'archer_hardware.commissioning.' + function, return_value=0,
            ) as entry, patch('archer_hardware.operator_client.OperatorClient') as client:
                self.assertEqual(main([command, '--help']), 0)
                entry.assert_called_once_with(['--help'])
                client.assert_not_called()

    def test_explicit_scope(self) -> None:
        """普通操作必须选一只臂，halt 为全局且不映射到归位."""
        self.assertEqual(request_path('halt', None), '/hardware/position/halt')
        for command, side in (('enable', None), ('halt', 'left'), ('reset', None),
                              ('return', 'left'), ('stop', None)):
            with self.assertRaises(ValueError):
                request_path(command, side)

    def test_task_run_arguments_fail_before_ros(self) -> None:
        """缺文件、混入单臂参数或把文件参数给其他操作时，不创建控制客户端."""
        for args in (['task-run'], ['task-run', '--side', 'left'],
                     ['status', '--run-dir', '/tmp/unused']):
            with self.subTest(args=args), patch('rclpy.init') as initialize:
                with self.assertRaises(SystemExit) as error:
                    main(args)
                self.assertEqual(error.exception.code, 2)
                initialize.assert_not_called()

    def test_recovery_scope_and_result_match(self) -> None:
        """限位收回明确指定一臂，受理或旧编号不能作为本次完成."""
        self.assertEqual(request_path('recover-limits', 'right'),
                         '/hardware/position/right/recover_limits')
        with self.assertRaises(ValueError):
            request_path('recover-limits', None)
        self.assertIsNone(operation_result(
            status('old', 'SUCCEEDED'), 'left', 'new', 'recover-limits',
        ))
        self.assertTrue(operation_result(
            status('new', 'SUCCEEDED'), 'left', 'new', 'recover-limits',
        ))

    def test_task_run_dispatch_is_explicit_and_does_not_enable(self) -> None:
        """文件执行只调用专用流程，不偷偷调用连接或使能命令."""
        with patch('rclpy.init'), patch('rclpy.ok', return_value=False), patch(
            'archer_hardware.operator_client.OperatorClient',
        ) as constructor:
            client = constructor.return_value
            client.run_task_file.return_value = {'result': 'SUCCEEDED'}
            self.assertEqual(main([
                'task-run', '--run-dir', '/tmp/run', '--limits-file', '/tmp/limits.yaml',
                '--prepared-file', '/tmp/plan.json', '--report-file', '/tmp/report.json',
                '--transition-max-excursion', '.05',
            ]), 0)
            client.run_command.assert_not_called()
            request, timeout = client.run_task_file.call_args.args
            self.assertEqual(timeout, 120.)
            self.assertEqual(len(request['request_id']), 32)
            self.assertEqual(request['transition_max_excursion'], .05)

    def test_old_success_and_accepted_are_not_completion(self) -> None:
        """旧编号成功及本次运行中都不能认为已经完成."""
        self.assertIsNone(operation_result(status('old', 'SUCCEEDED'), 'left', 'new', 'move'))
        self.assertIsNone(operation_result(status('new', 'RUNNING'), 'left', 'new', 'move'))
        self.assertTrue(operation_result(status('new', 'SUCCEEDED'), 'left', 'new', 'move'))

    def test_failure_or_changed_state_is_not_success(self) -> None:
        """失败、拒绝、取消及完成后状态改变都必须报告."""
        for phase in ('FAILED', 'REJECTED', 'CANCELLED'):
            with self.assertRaises(RuntimeError):
                operation_result(status('new', phase), 'left', 'new', 'move')
        with self.assertRaises(RuntimeError):
            operation_result(status('new', 'SUCCEEDED', 'FAULT'), 'left', 'new', 'move')

    def test_interlock_allows_only_disconnect_result(self) -> None:
        """软件联锁时仍允许确认显式断开完成."""
        value = status('new', 'SUCCEEDED', 'DISCONNECTED')
        value['interlock'] = 'test fault'
        self.assertTrue(operation_result(value, 'left', 'new', 'disconnect'))
        with self.assertRaises(RuntimeError):
            operation_result(value, 'left', 'new', 'enable')


class OperatorRosTests(unittest.TestCase):
    """通过真实 ROS 服务和状态话题验证，不使用 SDK 或网络设备."""

    def setUp(self) -> None:
        """独立执行器托管内存服务，客户端在测试线程等待结果."""
        rclpy.init(args=[])
        self.server = Node('fake_operator_backend')
        self.client = OperatorClient()
        self.value = status('old', 'SUCCEEDED')
        self.calls: list[str] = []
        self.reject = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.server.create_publisher(String, '/hardware/position/status', qos)
        self.service = self.server.create_service(
            Trigger, '/hardware/position/left/move', self.move,
        )
        self.halt_service = self.server.create_service(
            Trigger, '/hardware/position/halt', self.halt,
        )
        self.timer = self.server.create_timer(0.02, self.publish)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.server)
        self.thread = Thread(target=self.executor.spin, daemon=True)
        self.thread.start()

    def tearDown(self) -> None:
        """停止服务执行器后释放全部 ROS 对象."""
        self.executor.shutdown(timeout_sec=2.0)
        self.thread.join(timeout=2.0)
        self.client.destroy_node()
        self.server.destroy_node()
        rclpy.shutdown()

    def publish(self) -> None:
        """发布替身当前状态."""
        message = String()
        message.data = json.dumps(self.value)
        self.publisher.publish(message)

    def halt(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """只在内存记录立即停止，结束替身本次运行."""
        self.calls.append('halt')
        self.value['interlock'] = 'fake session ended'
        for arm in self.value['arms'].values():
            arm['state'] = 'STOPPED'
        response.success = True
        response.message = 'fake halt accepted'
        return response

    def move(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """记录一次请求并设置本次编号；不执行任何运动."""
        self.calls.append('move')
        response.success = not self.reject
        if self.reject:
            response.message = 'rejected by fake backend'
        else:
            response.message = '已受理编号 abcdef123456，请查看状态话题；不代表动作已完成'
            self.value = status('abcdef123456', 'SUCCEEDED')
        return response

    def test_status_does_not_call_services(self) -> None:
        """只查看状态不会连接、使能、归位或停止."""
        result = self.client.run_command('status', None, 3.0)
        self.assertEqual(result['result'], 'STATUS')
        self.assertEqual(self.calls, [])
        self.assertEqual(list(self.client.clients), [])

    def test_halt_waits_for_new_software_state(self) -> None:
        """受理响应不是停止确认；等待新状态且不声称物理停止已验收."""
        result = self.client.run_command('halt', None, 3.0)
        self.assertEqual(result['result'], 'SOFTWARE_HALT_OBSERVED')
        self.assertFalse(result['physical_stop_confirmed'])
        self.assertEqual(self.calls, ['halt'])

    def test_removed_entries_never_initialize_ros(self) -> None:
        """旧键盘和stop命令在参数解析时拒绝，不遗留控制请求."""
        for command in ('keyboard', 'stop'):
            with self.subTest(command=command), patch('rclpy.init') as initialize:
                with self.assertRaises(SystemExit) as result:
                    main([command])
                self.assertEqual(result.exception.code, 2)
                initialize.assert_not_called()
        self.assertEqual(self.calls, [])

    def test_single_request_and_correlated_completion(self) -> None:
        """按本次编号确认，服务只调用一次且完成后不自动归位."""
        result = self.client.run_command('move', 'left', 3.0)
        self.assertEqual(result['command_id'], 'abcdef123456')
        self.assertEqual(self.calls, ['move'])
        self.assertEqual(list(self.client.clients), [])

    def test_recovery_waits_for_result_and_enables_failure_halt(self) -> None:
        """新命令只发送一次，并沿用运动请求失败后的停止规则."""
        service = self.server.create_service(
            Trigger, '/hardware/position/left/recover_limits', self.move,
        )
        try:
            result = self.client.run_command('recover-limits', 'left', 3.0)
            self.assertEqual(result['command_id'], 'abcdef123456')
            self.assertEqual(self.calls, ['move'])
            self.assertTrue(self.client.halt_on_failure)
        finally:
            self.server.destroy_service(service)

    def test_rejection_is_not_ambiguous_timeout(self) -> None:
        """明确拒绝不算已受理，客户端不重试."""
        self.reject = True
        with self.assertRaisesRegex(RuntimeError, 'rejected'):
            self.client.run_command('move', 'left', 3.0)
        self.assertFalse(self.client.request_issued)
        self.assertEqual(self.calls, ['move'])

    def test_missing_service_not_sent(self) -> None:
        """服务不存在时不发送请求、不创建 SDK 或自动连接."""
        with self.assertRaisesRegex(RuntimeError, '服务未就绪'):
            self.client.run_command('enable', 'right', 0.1)
        self.assertFalse(self.client.request_issued)
        self.assertEqual(self.calls, [])
