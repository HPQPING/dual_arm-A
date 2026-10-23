"""以内存传输运行真实 SDK 周期，验证位置、取消、过期和退出，不连接设备."""

import asyncio
from dataclasses import replace
from importlib import import_module
from threading import Event
import time
from typing import Any
from types import SimpleNamespace
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
        self.start_control_periodic = self.api._project_start_control_periodic
        # 本测试由acquire手动运行真实SDK周期，禁止启动网络事件循环任务。
        self.api._project_start_control_periodic = lambda: None
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

    def test_control_periodic_is_deferred_and_starts_only_once(self) -> None:
        """发现设备不启动控制周期；显式准备后启动一次，重复调用不重复建任务."""
        with patch.object(self.sdk, '_start_device_periodic_task') as start:
            self.api._start_device_periodic_task(1)
            start.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, '显式准备'):
                self.start_control_periodic()
            self.api._HexDeviceApi__loop = SimpleNamespace(is_running=lambda: True)
            self.api._device_tasks = {}
            self.adapter.state = 'REQUESTING'

            def record(device_id):
                self.api._device_tasks[device_id] = object()

            start.side_effect = record
            self.start_control_periodic()
            self.start_control_periodic()
            start.assert_called_once_with(1)

    def test_startup_request_is_queued_before_periodic_start(self) -> None:
        """控制周期第一拍之前已有明确的初始化请求，不先运行厂商自动重连."""
        observed = []

        def start():
            observed.append(self.adapter.state)
            native = self.adapter._native
            assert native is not None
            self.assertIn('initialize', native.pending)

        self.api._project_start_control_periodic = start
        self.adapter.request_control()
        self.assertEqual(observed, ['REQUESTING'])

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

    def test_close_waits_for_brake_queued_by_sdk_encoding_fault(self) -> None:
        """SDK构造异常后立即退出，也须给已缓存BRAKE发送机会，不能先清空."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                native = self.adapter._native
                assert native is not None
                before = self.kinds().count('position')
                with patch.object(self.arm, 'validate_joint_positions',
                                  side_effect=ValueError('offline encoding rejection')):
                    with self.assertRaisesRegex(ValueError, 'offline encoding rejection'):
                        self.arm._construct_joint_command_msg(self.arm._target_command)
                self.assertEqual(self.adapter.state, 'FAULT')
                self.assertIn('brake', native.pending)
                waiting = Event()
                original_wait = native.wait_pending_brake

                def wait_for_delivery() -> bool:
                    waiting.set()
                    return original_wait(.1)

                try:
                    with patch.object(native, 'wait_pending_brake', wait_for_delivery), \
                            patch.object(HexFeedbackAdapter, 'close', return_value=None):
                        close = asyncio.create_task(asyncio.to_thread(self.adapter.close))
                        for _ in range(30):
                            if waiting.is_set():
                                break
                            await asyncio.sleep(.001)
                        self.assertTrue(waiting.is_set())
                        task = asyncio.create_task(self.arm._periodic())
                        await close
                    self.assertIn('brake', self.kinds())
                    self.assertEqual(self.kinds().count('position'), before)
                    self.assertTrue(native.closed)
                    self.assertEqual(self.adapter.state, 'DISCONNECTED')
                    self.assertNotIn('底层发送未确认', self.adapter.last_error)
                finally:
                    await self.finish(task)
        asyncio.run(run())

    def test_close_reports_queued_brake_that_cannot_be_delivered(self) -> None:
        """底层周期不可用时有界退出，保留原故障与制动未确认，不伪报成功."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                self.adapter._native_failure('offline lost cycle')
                native = self.adapter._native
                assert native is not None
                self.assertFalse(native.wait_pending_brake(.001))
                with patch.object(HexFeedbackAdapter, 'close', return_value=None):
                    await asyncio.to_thread(self.adapter.close)
                self.assertTrue(native.closed)
                self.assertIn('offline lost cycle', self.adapter.last_error)
                self.assertIn('底层发送未确认', self.adapter.last_error)
                self.assertEqual(self.adapter.state, 'DISCONNECTED')
                self.assertNotIn('brake', self.kinds())
        asyncio.run(run())

    def test_position_ack_records_sent_packet_when_pending_is_replaced(self) -> None:
        """传输期间缓存更新不能丢失已发送确认，也不能用后来SDK历史冒充原包."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            first, second = (.0001,) * 6, (.0002,) * 6
            native = self.adapter._native
            assert native is not None
            self.adapter._target = first
            targets = self.arm._construct_custom_motor_msg(
                self.adapter._command_types.POSITION, list(first))
            self.adapter._send(self.arm._construct_custom_joint_command_msg(targets), 'position')
            message = self.arm._construct_joint_command_msg(self.arm._target_command)

            async def replace_during_transport(api: Any, packet: Any) -> None:
                await self.capture(api, packet)
                self.adapter._target = second
                updated = self.arm._construct_custom_motor_msg(
                    self.adapter._command_types.POSITION, list(second))
                self.adapter._send(
                    self.arm._construct_custom_joint_command_msg(updated), 'position')
                self.arm._project_config.set_last_positions(25, [.0003] * 6)

            try:
                with patch.object(self.sdk, '_send_down_message', replace_during_transport):
                    await self.api._send_down_message(message)
                self.assertEqual(native.last_position, first)
                self.assertEqual(native.pending['position'].target, second)
            finally:
                await self.finish(task)
        asyncio.run(run())

    def test_actual_delivery_uses_packet_snapshot_during_encoding_update(self) -> None:
        """发送await期间同一缓存被再次编码，历史和确认仍属于已经交给传输的包。"""
        async def run() -> None:
            from archer_hardware.sdk_cycle import PositionSample

            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            native = self.adapter._native
            assert native is not None
            first, second = (.0001,) * 6, (.0002,) * 6
            acknowledgments = []
            self.adapter._target = first
            targets = self.arm._construct_custom_motor_msg(
                self.adapter._command_types.POSITION, list(first))
            self.adapter._send(self.arm._construct_custom_joint_command_msg(targets), 'position')
            native.sample_position = lambda: PositionSample(
                first, lambda: acknowledgments.append('first'))
            message = self.arm._construct_joint_command_msg(self.arm._target_command)

            async def reencode_during_transport(api: Any, packet: Any) -> None:
                await self.capture(api, packet)
                native.sample_position = lambda: PositionSample(
                    second, lambda: acknowledgments.append('second'))
                self.arm._construct_joint_command_msg(self.arm._target_command)

            try:
                with patch.object(self.sdk, '_send_down_message', reencode_during_transport):
                    await self.api._send_down_message(message)
                self.assertEqual(native.last_position, first)
                self.assertEqual(native.pending['position'].target, second)
                self.assertEqual(acknowledgments, ['first'])
            finally:
                await self.finish(task)
        asyncio.run(run())

    def test_sdk_sampling_prevents_stale_cache_then_jump(self) -> None:
        """缓存每12ms才更新时，SDK每4ms仍按路径取样；过期后不能自行续跑."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                plan = ContinuousPositionTrajectory.prepare(
                    (0, 1000000000, 2000000000, 3000000000),
                    ((0.,) * 6, (.04,) * 6, (.12,) * 6, (.16,) * 6),
                    ((0.,) * 6, (.08,) * 6, (.08,) * 6, (0.,) * 6),
                    ((0.,) * 6,) * 4, self.limits, .2)
                self.adapter._move = replace(plan, preserve_path=True, joint_error_rad=.05)
                self.adapter._trajectory_side = 'left'
                self.adapter.state = 'RUNNING'
                native = self.adapter._native
                assert native is not None
                with patch('archer_hardware.position_plan.time.monotonic') as clock, \
                        patch('archer_hardware.sdk_cycle.time.perf_counter') as send_clock:
                    clock.return_value = 100.
                    send_clock.return_value = 100.2
                    gate = PairedTrajectoryStart(reference_step_s=.0042, duration_s=3.,
                                                 stream_period_s=.004)
                    gate.ready('left')
                    gate.ready('right')
                    self.adapter._trajectory_gate = gate
                    native.last_position = (0.,) * 6
                    native.last_position_at = 100.2

                    def publish_old_cache() -> None:
                        self.adapter._target = (0.,) * 6
                        targets = self.arm._construct_custom_motor_msg(
                            self.adapter._command_types.POSITION, [0.] * 6)
                        self.adapter._send(
                            self.arm._construct_custom_joint_command_msg(targets), 'position')

                    publish_old_cache()
                    generation = 0
                    for tick in range(1, 756):
                        clock.return_value = 100.2 + tick * .004
                        send_clock.return_value = clock.return_value
                        self.adapter._feedback = replace(
                            self.frame(), positions=native.last_position)
                        # 工作线程可能只看见下一代缓存事件；发送健康必须由实际SDK发送确认。
                        gate.playback('left', None)
                        _, generation, _ = gate.playback('right', generation)
                        if tick % 3 == 0:
                            publish_old_cache()
                        message = self.arm._construct_joint_command_msg(self.arm._target_command)
                        await self.api._send_down_message(message)
                        assert native.last_position is not None
                        self.assertAlmostEqual(
                            native.last_position[0], plan.sample(tick * .004)[0], places=11)
                        self.assertEqual(self.adapter.state, 'RUNNING')
                    send_clock.return_value += .041
                    with self.assertRaisesRegex(RuntimeError, '40ms'):
                        self.arm._construct_joint_command_msg(self.arm._target_command)
                    self.assertEqual(self.adapter.state, 'FAULT')
                    self.assertTrue(gate.cancelled)
                await self.finish(task)
        asyncio.run(run())

    def test_identical_position_refresh_keeps_transport_confirmation(self) -> None:
        """SDK未交付时重复发布相同目标，不得替换原发送完成事件造成永久等待."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            native = self.adapter._native
            assert native is not None
            self.adapter._target = (.0001,) * 6
            targets = self.arm._construct_custom_motor_msg(
                self.adapter._command_types.POSITION, list(self.adapter._target))
            message = self.arm._construct_custom_joint_command_msg(targets)
            first = self.adapter._send(message, 'position')
            second = self.adapter._send(message, 'position')
            self.assertIs(first, second)
            self.assertFalse(first.is_set())
            packet = self.arm._construct_joint_command_msg(self.arm._target_command)
            try:
                with patch.object(self.sdk, '_send_down_message', self.capture):
                    await self.api._send_down_message(packet)
                self.assertTrue(first.is_set())
                self.assertEqual(native.last_position, (.0001,) * 6)
                renewed = self.adapter._send(message, 'position')
                self.assertIsNot(renewed, first)
                self.assertFalse(renewed.is_set())
                self.adapter._target = (.00010000001,) * 6
                updated = self.arm._construct_custom_motor_msg(
                    self.adapter._command_types.POSITION, list(self.adapter._target))
                fresh = self.adapter._send(
                    self.arm._construct_custom_joint_command_msg(updated), 'position')
                self.assertIsNot(fresh, first)
                self.assertEqual(native.pending['position'].target, (.00010000001,) * 6)
            finally:
                await self.finish(task)
        asyncio.run(run())

    def test_old_ack_cannot_restore_position_after_cancel(self) -> None:
        """传输期间取消后，旧包完成事件不能恢复已清空的位置或旧代次."""
        async def run() -> None:
            with patch.object(self.sdk, '_send_down_message', self.capture):
                task = await self.acquire()
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            native = self.adapter._native
            assert native is not None
            message = self.arm._construct_joint_command_msg(self.arm._target_command)

            async def cancel_during_transport(api: Any, packet: Any) -> None:
                await self.capture(api, packet)
                self.adapter.stop(fault=True, wait=False)

            try:
                with patch.object(self.sdk, '_send_down_message', cancel_during_transport):
                    await self.api._send_down_message(message)
                self.assertIsNone(native.last_position)
                self.assertEqual(self.adapter.state, 'FAULT')
                self.assertIn('brake', native.pending)
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

    def test_paired_file_clock_with_real_sdk_cycles_and_delayed_peer(self) -> None:
        """两个真实SDK周期共用参考时钟，单侧传输延迟后仍完成原曲线及末点保持."""
        async def run() -> None:
            limits = replace(self.limits, feedback_tolerance=.005)
            self.adapter.limits = limits
            peer = HexPositionAdapter('ws://127.0.0.1:9439', limits)
            arm_type = isolated_arm_class()
            with patch('archer_hardware.position_adapter.isolated_arm_class',
                       return_value=arm_type):
                with patch.object(self.sdk, '__init__', return_value=None):
                    peer.connect()
            api = peer._api
            # 两侧SDK周期均由本测试手动调度，不创建网络后台任务。
            api._project_start_control_periodic = lambda: None
            api._HexDeviceApi__is_closing = Event()
            arm = arm_type(robot_type=25, motor_count=6, proto_version=(1, 6), control_hz=250,
                           send_message_callback=api._send_down_message)
            arm._my_session_id = 7
            arm._error_codes = [None] * 6
            arm.get_motor_error_codes = lambda: arm._error_codes
            arm._pulse_per_rotation = np.array([100000.] * 6)
            arm._project_position_error = peer._native_failure
            api._internal_device_list = [arm]
            api._optional_device_list = []
            api._device_id_map = {1: arm}
            delayed = False
            enable_delay = False
            packet_counts = {'left': 0, 'right': 0}
            right_target = (0.,) * 6
            lag_until = 0.

            def right_feedback() -> JointFeedback:
                """在内存中模拟右侧短暂跟踪滞后，不接触真实设备."""
                offset = ((.00095 if time.monotonic() < lag_until else .000835)
                          if enable_delay else 0.)
                frame = JointFeedback(tuple(q + offset for q in right_target),
                                      (0.,) * 6, (0.,) * 6, time.perf_counter_ns())
                peer._feedback = frame
                return frame

            async def capture_pair(device_api: Any, packet: Any) -> None:
                nonlocal delayed, right_target
                device = self.arm if device_api is self.api else arm
                side = 'left' if device_api is self.api else 'right'
                command = packet.arm_command.arm_exclusive_command
                field = command.WhichOneof('exclusive_command')
                if field == 'api_control_initialize':
                    device._session_holder = 7
                    device._api_control_initialized = True
                elif field == 'calibrate':
                    device._calibrated = True
                elif (field == 'arm_api_control_command'
                      and command.arm_api_control_command.motor_targets.targets):
                    packet_counts[side] += 1
                    targets = command.arm_api_control_command.motor_targets.targets
                    if side == 'right' and all(t.HasField('position') for t in targets):
                        right_target = tuple(float(q) for q in arm.convert_positions_to_rad(
                            np.asarray([t.position for t in targets]),
                            arm.get_motor_pulse_per_rotations()))
                    if enable_delay and side == 'right' and not delayed:
                        delayed = True
                        await asyncio.sleep(.012)

            with patch.object(self.sdk, '_send_down_message', capture_pair), \
                    patch.object(peer, 'read_feedback', side_effect=right_feedback):
                tasks = [asyncio.create_task(device._periodic()) for device in (self.arm, arm)]
                try:
                    self.adapter.request_control()
                    peer.request_control()
                    for _ in range(50):
                        await asyncio.sleep(.003)
                        self.adapter.tick()
                        peer.tick()
                        if self.adapter.state == peer.state == 'HOLDING':
                            break
                    self.assertEqual(self.adapter.state, 'HOLDING')
                    self.assertEqual(peer.state, 'HOLDING')
                    plan = replace(ContinuousPositionTrajectory.prepare(
                        (0, 60_000_000, 120_000_000),
                        ((0.,) * 6, (.0001,) * 6, (0.,) * 6),
                        ((0.,) * 6,) * 3, ((0.,) * 6,) * 3, limits, .05),
                        preserve_path=True, joint_error_rad=.001)
                    original = tuple(c.tobytes() for c in plan.polynomials)
                    gate = PairedTrajectoryStart(reference_step_s=.01, duration_s=plan.duration,
                                                 stream_period_s=.004)
                    self.adapter.arm_continuous(plan, 'left', gate)
                    peer.arm_continuous(plan, 'right', gate)
                    enable_delay = True
                    start = gate.start_time()
                    assert start is not None
                    lag_until = start + .08
                    deadline = time.monotonic() + 2.
                    while time.monotonic() < deadline:
                        await asyncio.sleep(.003)
                        self.adapter.tick()
                        peer.tick()
                        if self.adapter.state == peer.state == 'HOLDING':
                            break
                    self.assertTrue(delayed)
                    self.assertEqual(self.adapter.state, 'HOLDING')
                    self.assertEqual(peer.state, 'HOLDING')
                    self.assertTrue(gate.playback_status()['end_transport_delivered'])
                    self.assertEqual(gate.playback_status()['reference_elapsed_s'], plan.duration)
                    self.assertEqual(original, tuple(c.tobytes() for c in plan.polynomials))
                    self.assertTrue(all(n > 10 for n in packet_counts.values()))
                finally:
                    for adapter in (self.adapter, peer):
                        assert adapter._native is not None
                        adapter._native.close()
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
        asyncio.run(run())


if __name__ == '__main__':
    unittest.main()
