"""离线验证显式操作客户端，服务后端仅为内存替身."""

import json
from threading import Thread
import time
import unittest
from unittest.mock import patch

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from archer_hardware.operator_client import (
    KeyboardStops, OperatorClient, keyboard_actions, main, operation_result, request_path,
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

    def test_keyboard_decoding_ignores_terminal_sequences(self) -> None:
        """方向键或功能键中的S不能误触归位，Ctrl+C有效，空格无操作."""
        self.assertEqual(keyboard_actions(b'sS ')[0], ['stop', 'stop'])
        actions, state = keyboard_actions(b'\x1b[')
        self.assertEqual(actions, [])
        self.assertEqual(keyboard_actions(b'1;2S', state), ([], 0))
        self.assertEqual(keyboard_actions(b'\x1bOS')[0], [])
        self.assertEqual(keyboard_actions(b' ')[0], [])
        self.assertEqual(keyboard_actions(b'\x03', 2), (['exit'], 0))
        self.assertEqual(keyboard_actions(b'\x03')[0], ['exit'])

    def test_keyboard_rejects_pipe_before_ros_start(self) -> None:
        """没有交互终端时拒绝运行，不隐式创建ROS客户端."""
        with patch('sys.stdin.isatty', return_value=False), patch('rclpy.init') as initialize:
            with self.assertRaises(SystemExit) as result:
                main(['keyboard'])
            self.assertEqual(result.exception.code, 2)
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
        self.assertEqual(request_path('stop', None), '/hardware/position/stop')
        self.assertEqual(request_path('halt', None), '/hardware/position/halt')
        for command, side in (('enable', None), ('halt', 'left'), ('reset', None),
                              ('return', 'left'), ('stop', 'right')):
            with self.assertRaises(ValueError):
                request_path(command, side)

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
        self.return_ticks = 0
        self.block_return = False
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.publisher = self.server.create_publisher(String, '/hardware/position/status', qos)
        self.service = self.server.create_service(
            Trigger, '/hardware/position/left/move', self.move,
        )
        self.stop_service = self.server.create_service(
            Trigger, '/hardware/position/stop', self.normal_stop,
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
        if self.return_ticks:
            self.return_ticks += 1
            if self.return_ticks >= 4 and not self.block_return:
                self.value['arms']['right']['phase'] = 'SUCCEEDED'
        message = String()
        message.data = json.dumps(self.value)
        self.publisher.publish(message)

    def normal_stop(
        self, request: Trigger.Request, response: Trigger.Response,
    ) -> Trigger.Response:
        """左臂先完成、右臂延后完成，检查客户端不会只等待一侧."""
        self.calls.append('stop')
        self.value = status('111111111111', 'SUCCEEDED')
        self.value['arms']['right'] = {
            'command_id': '222222222222', 'phase': 'RUNNING', 'state': 'HOLDING',
        }
        self.return_ticks = 1
        response.success = True
        response.message = json.dumps({'operation': 'stop', 'commands': {
            'left': '111111111111', 'right': '222222222222',
        }})
        return response

    def halt(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        """只在内存记录立即停止，结束替身本次运行."""
        self.calls.append('halt')
        self.return_ticks = 0
        self.value['interlock'] = 'fake session ended'
        for arm in self.value['arms'].values():
            arm['state'] = 'STOPPED'
        response.success = True
        response.message = 'fake halt accepted'
        return response

    def wait_keyboard(self, controller: KeyboardStops, predicate) -> None:
        """交替推进真实ROS和无阻塞键盘状态，最长三秒."""
        deadline = time.monotonic() + 3.0
        while not predicate() and time.monotonic() < deadline:
            rclpy.spin_once(self.client, timeout_sec=0.01)
            controller.poll()
        self.assertTrue(predicate())

    def test_keyboard_halt_preempts_return_and_latches(self) -> None:
        """归位等待中仍能处理halt，之后S不能恢复或重复提交."""
        self.block_return = True
        controller = KeyboardStops(self.client, 3.0, report=lambda text: None)
        try:
            self.wait_keyboard(controller, controller.stop_client.service_is_ready)
            self.wait_keyboard(controller, controller.halt_client.service_is_ready)
            controller.press_stop()
            self.wait_keyboard(controller, lambda: bool(controller.identifiers))
            controller.press_stop()
            self.assertEqual(self.calls, ['stop'])
            controller.press_halt()
            self.wait_keyboard(controller, lambda: controller.halt_phase == 'COMPLETE')
            controller.press_stop()
            controller.press_halt()
            controller.poll()
            self.assertEqual(self.calls, ['stop', 'halt'])
            self.assertTrue(controller.halt_latched)
        finally:
            controller.close()

    def test_keyboard_normal_return_completes_without_halt(self) -> None:
        """普通归位完成保持监听，不自动halt或关闭后端连接."""
        controller = KeyboardStops(self.client, 3.0, report=lambda text: None)
        try:
            self.wait_keyboard(controller, controller.stop_client.service_is_ready)
            controller.press_stop()
            self.wait_keyboard(controller, lambda: self.return_ticks >= 4
                               and not controller.identifiers and controller.stop_future is None)
            self.assertFalse(controller.halt_latched)
            self.assertEqual(self.calls, ['stop'])
        finally:
            controller.close()

    def test_keyboard_halt_timeout_is_not_confirmation(self) -> None:
        """超时不报告已经停住，Ctrl+C仍可退出."""
        controller = KeyboardStops(self.client, 3.0, report=lambda text: None)
        try:
            controller.press_halt(exit_requested=True)
            controller.halt_deadline = time.monotonic() - 1.0
            controller.poll()
            self.assertEqual(controller.halt_phase, 'UNCONFIRMED')
            self.assertTrue(controller.exit_requested)
        finally:
            controller.close()

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

    def test_stop_waits_for_both_return_identifiers(self) -> None:
        """普通停止必须确认两侧本次归位完成，不把单侧成功当整体成功."""
        result = self.client.run_command('stop', None, 3.0)
        self.assertEqual(result['result'], 'RETURNED_TO_INITIAL_TARGET')
        self.assertGreaterEqual(self.return_ticks, 4)
        self.assertEqual(self.calls, ['stop'])

    def test_single_request_and_correlated_completion(self) -> None:
        """按本次编号确认，服务只调用一次且完成后不自动归位."""
        result = self.client.run_command('move', 'left', 3.0)
        self.assertEqual(result['command_id'], 'abcdef123456')
        self.assertEqual(self.calls, ['move'])
        self.assertEqual(list(self.client.clients), [])

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
