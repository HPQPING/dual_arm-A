"""离线验证反馈适配器，所有测试均禁止连接实际设备."""

import asyncio
from importlib import import_module
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from archer_hardware.hex_adapter import HexFeedbackAdapter, report_only


class FakeArm:
    """模拟同一帧的设备反馈."""

    motor_count = 6
    robot_type = 16

    def __init__(self):
        """生成当前时刻的完整反馈."""
        self.set_stamp(time.perf_counter_ns())

    def set_stamp(self, stamp):
        """设置测试时间戳."""
        self.status = {
            'pos': [0.1] * 6, 'vel': [0.0] * 6, 'eff': [0.0] * 6,
            'ts': {'s': stamp // 1_000_000_000, 'ns': stamp % 1_000_000_000},
        }

    def get_simple_motor_status(self, pop=False):
        """返回一帧，检查调用者读取最新数据."""
        assert pop is False
        return self.status


class FeedbackTests(unittest.TestCase):
    """覆盖数据异常、时效性和官方发送出口."""

    def setUp(self):
        """注入模拟 SDK，不建立网络连接."""
        self.adapter = HexFeedbackAdapter('ws://127.0.0.1:8439')
        self.arm = FakeArm()
        self.api = SimpleNamespace(
            device_list=[self.arm], is_api_exit=lambda: False,
            is_websocket_recv_timeout=lambda: False,
        )
        self.adapter._api = self.api
        self.adapter._arm_type = FakeArm

    def test_complete_frame_and_duplicate(self):
        """完整帧只读取一次，不重复发布旧帧."""
        feedback = self.adapter.read()
        assert feedback is not None, '应当读到一帧完整关节反馈'
        self.assertEqual(feedback.positions, (0.1,) * 6)
        self.assertIsNone(self.adapter.read())

    def test_invalid_vectors(self):
        """拒绝维度错误、非数值及非有限反馈."""
        for values in ([0.0] * 5, [float('nan')] * 6,
                       [float('inf')] * 6, [True] * 6, ['1'] * 6):
            with self.subTest(values=values):
                self.arm.status['pos'] = values
                with self.assertRaises((ValueError, TypeError)):
                    self.adapter.read()

    def test_stale_frame(self):
        """已经读取过的帧也必须接受时效检查."""
        self.adapter.read()
        self.arm.set_stamp(time.perf_counter_ns() - 1_000_000_000)
        with self.assertRaisesRegex(RuntimeError, '过期'):
            self.adapter.read()

    def test_future_frame(self):
        """拒绝来自错误时钟基准的未来帧."""
        self.arm.set_stamp(time.perf_counter_ns() + 1_000_000_000)
        with self.assertRaisesRegex(RuntimeError, '时间基准'):
            self.adapter.read()

    def test_backwards_timestamp(self):
        """拒绝倒序数据而非将其作为新的初始角度."""
        self.adapter.read()
        self.arm.set_stamp(self.adapter._last_source_ns - 1)
        with self.assertRaisesRegex(RuntimeError, '倒退'):
            self.adapter.read()

    def test_wrong_robot(self):
        """六轴也不意味着可以接受其他系列机械臂."""
        self.arm.robot_type = 14
        with self.assertRaisesRegex(RuntimeError, 'Archer'):
            self.adapter.read()

    def test_disconnected_and_missing(self):
        """未连接及尚未发现设备时不制造默认角度."""
        self.api.device_list = []
        self.assertIsNone(self.adapter.read())
        self.adapter._api = None
        self.assertIsNone(self.adapter.read())

    def test_sdk_exit(self):
        """SDK 退出立即阻止继续提供旧反馈."""
        self.api.is_api_exit = lambda: True
        with self.assertRaises(RuntimeError):
            self.adapter.read()

    def test_invalid_urls(self):
        """拒绝缺少端口、带密码或不支持的连接格式."""
        for url in ('http://localhost:80', 'ws://localhost',
                    'ws://a:b@localhost:8439', 'ws://localhost:8439/path'):
            with self.subTest(url=url), self.assertRaises(ValueError):
                HexFeedbackAdapter(url)

    def test_official_sdk_send_gate(self):
        """使用已安装官方 protobuf 检查混合包、控制包及清理路径."""
        from hex_device.hex_device_api import HexDeviceApi
        # protobuf 消息类在运行时生成，通过动态模块读取实际类型。
        APIDown = import_module('hex_device.generated.public_api_down_pb2').APIDown

        sent = []

        async def capture(api, message):
            sent.append(message.SerializeToString())

        # 构造函数替换为无操作，绝不运行官方网络线程。
        adapter = HexFeedbackAdapter('ws://127.0.0.1:8439')
        with patch.object(HexDeviceApi, '__init__', return_value=None):
            adapter.connect()
        api = adapter._api
        allowed = APIDown()
        allowed.set_report_frequency = 1
        rejected = []
        for field in ('api_control_initialize', 'calibrate'):
            message = APIDown()
            setattr(message.arm_command.arm_exclusive_command, field, True)
            rejected.append(message)
        mixed = APIDown()
        mixed.CopyFrom(allowed)
        mixed.arm_command.arm_shared_command.clear_parking_stop = True
        rejected.extend([mixed, APIDown()])
        with patch.object(HexDeviceApi, '_send_down_message', capture):
            asyncio.run(api._send_down_message(allowed))
            for message in rejected:
                self.assertFalse(report_only(message))
                asyncio.run(api._send_down_message(message))
        self.assertEqual(len(sent), 1)
        with patch.object(HexDeviceApi, '_start_device_periodic_task') as start:
            api._start_device_periodic_task(1)
            start.assert_not_called()
        with patch.object(HexDeviceApi, 'close') as close:
            adapter.close()
            close.assert_called_once()
        self.assertIsNone(adapter._api)


if __name__ == '__main__':
    unittest.main()
