"""离线验证反馈适配器，所有测试均禁止连接实际设备."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
import logging
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from archer_hardware.hex_adapter import (
    HexFeedbackAdapter, managed_sdk_class, report_only,
)


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

    def test_local_receipt_accepts_small_sdk_clock_lead(self):
        """SDK偏移换算时间略领先时，本地收包时间仍为非未来，原始时间另存."""
        received = time.perf_counter_ns()
        sdk_stamp = received + 2000000
        self.arm.set_stamp(sdk_stamp)
        self.api._project_feedback_frames = {16: (self.arm, received, self.arm.status)}
        frame = self.adapter.read()
        assert frame is not None
        self.assertEqual(frame.source_ns, received)
        self.assertEqual(frame.sdk_source_ns, sdk_stamp)
        self.assertGreaterEqual(time.perf_counter_ns() - frame.source_ns, 0)
        self.assertIsNone(self.adapter.read())

    def test_new_receipt_does_not_refresh_repeated_sdk_sample(self):
        """同一SDK时间戳即使重复收到，也不更新可供控制使用的反馈帧."""
        received = time.perf_counter_ns()
        self.arm.set_stamp(received)
        self.api._project_feedback_frames = {16: (self.arm, received, self.arm.status)}
        frame = self.adapter.read()
        assert frame is not None
        self.api._project_feedback_frames[16] = (self.arm, received + 1, self.arm.status)
        self.assertIsNone(self.adapter.read())
        self.assertEqual(self.adapter._last_source_ns, received)

    def test_receipt_and_sdk_time_both_reject_stale_frames(self):
        """新收包也不能掩盖SDK旧时间，SDK新时间也不能掩盖长时间未收包."""
        now = time.perf_counter_ns()
        samples = ((now, now - 1000000000), (now - 1000000000, now),
                   (now, now + 1000000000))
        for received, sdk_stamp in samples:
            self.arm.set_stamp(sdk_stamp)
            self.api._project_feedback_frames = {16: (self.arm, received, self.arm.status)}
            with self.subTest(received=received, sdk=sdk_stamp), \
                    self.assertRaisesRegex(RuntimeError, '过期|时间基准'):
                self.adapter.read()

    def test_managed_hook_captures_same_frame_without_transport(self):
        """直接调用真实托管收包钩子，保存同帧时间与角度，不创建网络线程."""
        from unittest.mock import Mock

        arm = self.arm

        class Base:
            """只模拟上行更新，不提供任何连接或发送入口."""

            def _process_api_up(self, message):
                """先更新角度，由托管钩子随后保存本帧."""
                arm.status['pos'] = [.2] * 6

            def find_device_by_robot_type(self, robot_type):
                """返回当前离线替身."""
                return arm

        api = object.__new__(managed_sdk_class(Base))
        api._project_feedback_frames = {}
        message = Mock(robot_type=16)
        message.HasField.side_effect = lambda field: field == 'arm_status'
        api._process_api_up(message)
        saved_arm, received, status = api._project_feedback_frames[16]
        self.assertIs(saved_arm, arm)
        self.assertLessEqual(received, time.perf_counter_ns())
        self.assertEqual(status['pos'], [.2] * 6)

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
            close.assert_not_called()
        self.assertIsNone(adapter._api)


class ShutdownTests(unittest.TestCase):
    """运行官方主循环与内存传输，始终绕过网络构造函数."""

    def start_api(self, *, failure=False, callback_close=False, delay=0.05):
        """仅替换连接和收包，保留已安装 SDK 的线程与关闭入口."""
        from hex_device.hex_device_api import HexDeviceApi

        api = object.__new__(managed_sdk_class(HexDeviceApi))
        api._logger = logging.getLogger('offline-sdk-shutdown')
        api._stream_mode = False
        api._internal_device_list = []
        api._optional_device_list = []
        api._HexDeviceApi__loop = None
        api._HexDeviceApi__loop_ready = threading.Event()
        api._HexDeviceApi__is_closing = threading.Event()
        api._HexDeviceApi__shutdown_event = None
        api._HexDeviceApi__kcp_client = None
        finished = threading.Event()
        aborted = threading.Event()
        started = threading.Event()

        async def close_socket():
            await asyncio.sleep(delay)
            if failure:
                raise OSError('offline injected close failure')
            finished.set()

        api._HexDeviceApi__websocket = SimpleNamespace(
            close=close_socket, transport=SimpleNamespace(abort=aborted.set),
        )

        async def connect():
            started.set()

        async def parser():
            if callback_close:
                api.close()
            await asyncio.Event().wait()

        async def stop_tasks():
            return None

        api._HexDeviceApi__connect_ws = connect
        api._HexDeviceApi__websocket_data_parser = parser
        api._stop_all_device_tasks = stop_tasks
        worker = threading.Thread(target=api._HexDeviceApi__loop_start)
        api._HexDeviceApi__loop_thread = worker
        worker.start()
        self.assertTrue(started.wait(1.0))

        def cleanup():
            try:
                api.close()
            except RuntimeError:
                if not failure and delay <= 2.0:
                    raise

        self.addCleanup(cleanup)
        return api, worker, finished, aborted

    def test_handshake_before_loop_exit_and_idempotent_close(self):
        """等待真实异步挂起后才结束线程，重复关闭不遗留任务."""
        api, worker, finished, aborted = self.start_api()
        api.close()
        self.assertTrue(finished.is_set())
        self.assertFalse(aborted.is_set())
        self.assertFalse(worker.is_alive())
        self.assertTrue(api._HexDeviceApi__loop.is_closed())
        self.assertFalse(asyncio.all_tasks(api._HexDeviceApi__loop))
        api.close()

    def test_close_does_not_send_device_disable_or_wait_point_three(self):
        """退出不调用厂商设备 stop，不保留导致看门狗空窗的固定等待."""
        from unittest.mock import Mock

        api, worker, finished, aborted = self.start_api()
        device = SimpleNamespace(stop=Mock())
        api._internal_device_list = [device]
        api.close()
        device.stop.assert_not_called()
        self.assertTrue(finished.is_set())
        self.assertFalse(aborted.is_set())
        self.assertFalse(worker.is_alive())

    def test_callback_close_does_not_wait_on_own_thread(self):
        """模拟解析失败回调，关闭不在事件线程阻塞等待自己."""
        api, worker, finished, aborted = self.start_api(callback_close=True)
        worker.join(1.0)
        self.assertFalse(worker.is_alive())
        self.assertTrue(finished.is_set())
        self.assertFalse(aborted.is_set())
        self.assertTrue(api._HexDeviceApi__loop.is_closed())

    def test_concurrent_close_waits_for_one_handshake(self):
        """两个清理调用共享握手结果，不提前结束正在关闭的连接."""
        api, worker, finished, aborted = self.start_api()
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(api.close) for _ in range(2)]
            for future in futures:
                future.result(timeout=4.0)
        self.assertTrue(finished.is_set())
        self.assertFalse(aborted.is_set())
        self.assertFalse(worker.is_alive())
        self.assertTrue(api._HexDeviceApi__loop.is_closed())

    def test_transport_failure_is_reported_after_cleanup(self):
        """握手失败时中止本地传输，向上报告而非声称已正常关闭."""
        api, worker, finished, aborted = self.start_api(failure=True)
        with self.assertRaisesRegex(RuntimeError, 'OSError'):
            api.close()
        self.assertFalse(finished.is_set())
        self.assertTrue(aborted.is_set())
        self.assertFalse(worker.is_alive())
        self.assertTrue(api._HexDeviceApi__loop.is_closed())

    def test_transport_timeout_is_bounded_and_reported(self):
        """迟迟不完成的握手在限定时间内取消，避免原有五秒假等待."""
        api, worker, finished, aborted = self.start_api(delay=30.0)
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'TimeoutError'):
            api.close()
        self.assertLess(time.monotonic() - start, 4.0)
        self.assertFalse(finished.is_set())
        self.assertTrue(aborted.is_set())
        self.assertFalse(worker.is_alive())
        self.assertTrue(api._HexDeviceApi__loop.is_closed())


if __name__ == '__main__':
    unittest.main()
