"""以内存传输运行真实 SDK 周期，验证位置、取消、过期和退出，不连接设备."""

import asyncio
from importlib import import_module
from threading import Event
import time
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from archer_hardware.hex_adapter import HexFeedbackAdapter, JointFeedback
from archer_hardware.position_adapter import HexPositionAdapter
from archer_hardware.position_plan import (
    ContinuousPositionTrajectory, PairedTrajectoryStart, PositionLimits,
)
from archer_hardware.sdk_position import isolated_arm_class


class NativeCycleTests(unittest.TestCase):
    """保留官方 Arm._periodic，只将最终传输替换成内存接收器."""

    def setUp(self) -> None:
        self.sdk = import_module('hex_device.hex_device_api').HexDeviceApi
        self.limits = PositionLimits((-2.,) * 6, (2.,) * 6, (.1,) * 6, (.2,) * 6)
        self.adapter = HexPositionAdapter('ws://127.0.0.1:8439', self.limits)
        arm_type = isolated_arm_class()
        with patch('archer_hardware.position_adapter.isolated_arm_class', return_value=arm_type):
            with patch.object(self.sdk, '__init__', return_value=None):
                self.adapter.connect()
        self.api = self.adapter._api
        self.api._HexDeviceApi__is_closing = Event()
        self.arm = arm_type(robot_type=25, motor_count=6, proto_version=(1, 6), control_hz=250,
                            send_message_callback=self.api._send_down_message)
        self.arm._my_session_id = 7
        self.arm._error_codes = [None] * 6
        self.arm.get_motor_error_codes = lambda: self.arm._error_codes
        self.arm._pulse_per_rotation = np.array([100000.] * 6)
        self.arm._project_position_error = self.adapter._native_failure
        self.api._internal_device_list = [self.arm]
        self.api._optional_device_list = []
        self.api._device_id_map = {1: self.arm}
        self.sent: list[Any] = []
        self.clear_ack = True
        self.feedback_patch = patch.object(HexFeedbackAdapter, 'read', side_effect=self.frame)
        self.feedback_patch.start()
        self.addCleanup(self.feedback_patch.stop)

    def frame(self) -> JointFeedback:
        """每次生成新的离线静止反馈，绝不代表真实角度采集."""
        return JointFeedback((0.,) * 6, (0.,) * 6, (0.,) * 6, time.perf_counter_ns())

    async def capture(self, api: Any, message: Any) -> None:
        """模拟固件确认准备，不执行任何物理动作."""
        packet = type(message)()
        packet.CopyFrom(message)
        self.sent.append(packet)
        if message.arm_command.HasField('arm_shared_command'):
            if self.clear_ack:
                self.arm._parking_stop_detail.Clear()
            return
        command = message.arm_command.arm_exclusive_command
        field = command.WhichOneof('exclusive_command')
        if field == 'api_control_initialize':
            self.arm._session_holder = 7
            self.arm._api_control_initialized = True
        elif field == 'calibrate':
            self.arm._calibrated = True

    async def acquire(self) -> asyncio.Task:
        """实际跑 SDK 原生周期并推进项目准备状态至保持."""
        task = asyncio.create_task(self.arm._periodic())
        self.adapter.request_control()
        for _ in range(30):
            await asyncio.sleep(.003)
            self.adapter.tick()
            if self.adapter.state == 'HOLDING':
                await asyncio.sleep(.008)
                return task
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self.fail('离线 SDK 准备没有进入保持')

    async def finish(self, task: asyncio.Task) -> None:
        """先取消准入，再停止测试周期，不调用任何真实网络清理."""
        assert self.adapter._native is not None
        self.adapter._native.close()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    def kinds(self) -> list[str]:
        """将捕获的实际 protobuf 下行分类供断言使用."""
        result = []
        for packet in self.sent:
            if packet.arm_command.HasField('arm_shared_command'):
                result.append('clear_api_timeout')
                continue
            command = packet.arm_command.arm_exclusive_command
            field = command.WhichOneof('exclusive_command')
            if field != 'arm_api_control_command':
                result.append(field)
            else:
                targets = command.arm_api_control_command.motor_targets.targets
                result.append('empty' if not targets else targets[0].ListFields()[0][0].name)
        return result

    def set_api_timeout(self) -> None:
        """模拟控制器保留的API超时停车，不发送任何真实控制命令."""
        types = import_module('hex_device.generated.public_api_types_pb2')
        self.arm._parking_stop_detail.CopyFrom(types.ParkingStopDetail(
            reason='API communication timeout', is_remotely_clearable=True,
            category=types.ParkingStopCategory.PscAPICommunicationTimeout,
        ))

    def test_official_startup_clears_once_then_prepares_and_holds(self) -> None:
        """真实SDK周期执行单次指定清除、初始化、必要校准及位置保持."""
        async def run() -> None:
            self.set_api_timeout()
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                try:
                    kinds = self.kinds()
                    self.assertEqual(kinds.count('clear_api_timeout'), 1)
                    self.assertEqual(kinds.count('api_control_initialize'), 1)
                    self.assertEqual(kinds.count('calibrate'), 1)
                    self.assertLess(kinds.index('clear_api_timeout'),
                                    kinds.index('api_control_initialize'))
                    self.assertLess(kinds.index('calibrate'), kinds.index('position'))
                    self.assertEqual(self.adapter.initial_positions, (0.,) * 6)
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_connect_alone_never_clears_existing_timeout(self) -> None:
        """没有显式enable时，SDK自身的自动清除和重新初始化均不交付."""
        async def run() -> None:
            self.set_api_timeout()
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = asyncio.create_task(self.arm._periodic())
                try:
                    await asyncio.sleep(.025)
                    self.assertEqual(self.sent, [])
                    self.assertEqual(self.adapter.state, 'CONNECTED')
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_clear_without_confirmation_times_out_without_retry(self) -> None:
        """一次清除未获确认时只等待，超时不重发清除或申请控制."""
        async def run() -> None:
            self.set_api_timeout()
            self.clear_ack = False
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = asyncio.create_task(self.arm._periodic())
                try:
                    self.adapter.request_control()
                    for _ in range(8):
                        await asyncio.sleep(.003)
                        self.adapter.tick()
                    self.assertEqual(self.kinds(), ['clear_api_timeout'])
                    self.adapter._requested_at -= self.adapter.readiness_timeout + 1.
                    with self.assertRaisesRegex(RuntimeError, '就绪超时'):
                        self.adapter.tick()
                    await asyncio.sleep(.012)
                    self.assertEqual(self.kinds(), ['clear_api_timeout'])
                    self.assertEqual(self.adapter.state, 'FAULT')
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_running_timeout_cancels_pair_without_clear_or_reinitialization(self) -> None:
        """已保持或运动时的新超时取消双方，SDK不能自行清除并复发旧目标."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                try:
                    gate = PairedTrajectoryStart()
                    self.adapter._trajectory_gate = gate
                    before = self.kinds().count('position')
                    self.set_api_timeout()
                    await asyncio.sleep(.02)
                    self.assertEqual(self.adapter.state, 'FAULT')
                    self.assertTrue(gate.cancelled)
                    self.assertEqual(self.kinds().count('position'), before)
                    self.assertEqual(self.kinds().count('api_control_initialize'), 1)
                    self.assertNotIn('clear_api_timeout', self.kinds())
                    self.assertIn('brake', self.kinds())
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_halt_before_clear_delivery_discards_preparation(self) -> None:
        """停止优先作废排队清除，随后SDK周期不能恢复初始化或运动."""
        async def run() -> None:
            self.set_api_timeout()
            with patch.object(self.sdk, '_send_down_message', self.capture):
                self.adapter.request_control()
                self.adapter.stop(wait=False)
                task = asyncio.create_task(self.arm._periodic())
                try:
                    await asyncio.sleep(.02)
                    self.assertEqual(self.sent, [])
                    self.assertEqual(self.adapter.state, 'STOPPED')
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_native_public_position_and_halt_no_more_positions(self) -> None:
        """准备和位置都由官方周期交付；停止后只允许制动，不再下发位置."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                try:
                    self.assertIn('position', self.kinds())
                    self.assertEqual(self.adapter.commanded_positions, (0.,) * 6)
                    self.adapter.stop(wait=False)
                    count = self.kinds().count('position')
                    await asyncio.sleep(.02)
                    self.assertEqual(self.kinds().count('position'), count)
                    self.assertIn('brake', self.kinds())
                    self.assertEqual(self.kinds().count('api_control_initialize'), 1)
                    self.assertEqual(self.kinds().count('calibrate'), 1)
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_expired_cache_brakes_and_never_recovers(self) -> None:
        """规划线程不再更新时，SDK 不能无限复用旧目标或重新初始化."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                try:
                    await asyncio.sleep(.065)
                    self.assertEqual(self.adapter.state, 'FAULT')
                    self.assertIn('40ms', self.adapter.last_error)
                    count = self.kinds().count('position')
                    await asyncio.sleep(.015)
                    self.assertEqual(self.kinds().count('position'), count)
                    self.assertIn('brake', self.kinds())
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_encoding_failure_and_peer_cancel_stop_native_sender(self) -> None:
        """SDK 编码拒绝时取消双臂门控，错误不会只停留在日志中."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                try:
                    gate = PairedTrajectoryStart()
                    self.adapter._trajectory_gate = gate
                    self.adapter._target = (.01,) * 6
                    targets = self.arm._construct_custom_motor_msg(
                        self.adapter._command_types.POSITION, [.01] * 6,
                    )
                    self.adapter._send(self.arm._construct_custom_joint_command_msg(targets),
                                       'position')
                    await asyncio.sleep(.015)
                    self.assertEqual(self.adapter.state, 'FAULT')
                    self.assertTrue(gate.cancelled)
                    self.assertIn('裁剪', self.adapter.last_error)
                    self.assertIn('brake', self.kinds())
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_closed_gate_blocks_all_native_controls(self) -> None:
        """开始退出后即使厂商周期尚未取消，也不得再交付任何控制包."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                try:
                    assert self.adapter._native is not None
                    self.adapter._native.close()
                    count = len(self.sent)
                    await asyncio.sleep(.015)
                    self.assertEqual(len(self.sent), count)
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_shared_cancel_blocks_cached_native_positions(self) -> None:
        """另一臂取消后，即使本臂规划线程还未tick，发送出口也立即拦截缓存."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                try:
                    gate = PairedTrajectoryStart()
                    self.adapter._trajectory_gate = gate
                    count = self.kinds().count('position')
                    gate.cancel()
                    await asyncio.sleep(.015)
                    self.assertEqual(self.kinds().count('position'), count)
                    self.assertEqual(self.adapter.state, 'FAULT')
                    self.assertIn('brake', self.kinds())
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_native_continuous_cycle_returns_and_holds(self) -> None:
        """完整数学路径经过项目规划和官方周期，到末点后保持而不追加归位."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                try:
                    plan = ContinuousPositionTrajectory.prepare(
                        (0, 150000000, 300000000),
                        ((0.,) * 6, (.0001,) * 6, (0.,) * 6),
                        ((0.,) * 6,) * 3, ((0.,) * 6,) * 3, self.limits, .05,
                    )
                    gate = PairedTrajectoryStart()
                    self.adapter.arm_continuous(plan, 'left', gate)
                    gate.ready('right')
                    deadline = time.monotonic() + .8
                    while time.monotonic() < deadline:
                        await asyncio.sleep(.004)
                        self.adapter.tick()
                        if self.adapter.state == 'HOLDING':
                            await asyncio.sleep(.006)
                            break
                    self.assertEqual(self.adapter.state, 'HOLDING')
                    self.assertEqual(self.adapter.commanded_positions, (0.,) * 6)
                    self.assertGreater(self.kinds().count('position'), 20)
                    self.assertNotIn('brake', self.kinds())
                finally:
                    await self.finish(task)
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
