"""离线验证位置控制，所有发送均被替身接收，不连接实际设备."""

import asyncio
from importlib import import_module
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from archer_hardware.hex_adapter import HexFeedbackAdapter, JointFeedback
from archer_hardware.position_adapter import HexPositionAdapter
from archer_hardware.position_plan import JointPositionMove, PositionLimits, StopAndReturnMove


def fixture_limits() -> PositionLimits:
    """仅供测试使用的虚构限位，不是 Archer 实机参数."""
    return PositionLimits((-2.0,) * 6, (2.0,) * 6, (0.1,) * 6, (0.2,) * 6)


class PositionPlanTests(unittest.TestCase):
    """检查全段轨迹的边界和解析运动约束."""

    def test_endpoints_and_monotonicity(self) -> None:
        """五次插值准确到达起终点，位置不会超出端点范围."""
        plan = JointPositionMove.prepare(
            [0.0] * 6, [0.02] * 6, 2.0, fixture_limits(), 0.05,
        )
        self.assertEqual(plan.sample(-1), (0.0,) * 6)
        self.assertEqual(plan.sample(2.0), (0.02,) * 6)
        self.assertEqual(plan.sample(3.0), (0.02,) * 6)
        values = [plan.sample(i / 100.0)[0] for i in range(201)]
        self.assertEqual(values, sorted(values))
        self.assertAlmostEqual(plan.sample(1.0)[0], 0.01)

    def test_reject_limits_and_excursion(self) -> None:
        """拒绝越界、过大试运行位移和无效数据，不裁剪目标."""
        for goal in ([3.0] * 6, [0.1] * 6, [float('nan')] * 6, [0.0] * 5):
            with self.subTest(goal=goal), self.assertRaises(ValueError):
                JointPositionMove.prepare([0.0] * 6, goal, 2.0, fixture_limits(), 0.05)

    def test_speed_and_acceleration(self) -> None:
        """分别检查速度和加速度约束，避免只检离散目标点."""
        with self.assertRaisesRegex(ValueError, '速度'):
            JointPositionMove.prepare([0.0] * 6, [0.02] * 6, 0.1, fixture_limits(), 0.05)
        limits = PositionLimits((-2.0,) * 6, (2.0,) * 6, (10.0,) * 6, (0.001,) * 6)
        with self.assertRaisesRegex(ValueError, '加速度'):
            JointPositionMove.prepare([0.0] * 6, [0.02] * 6, 1.0, limits, 0.05)

    def test_invalid_duration_and_limits(self) -> None:
        """禁止空限位、非正约束和无效时长."""
        for duration in (0.0, -1.0, float('inf'), True):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                JointPositionMove.prepare(
                    [0.0] * 6, [0.02] * 6, duration, fixture_limits(), 0.05,
                )
        with self.assertRaises(ValueError):
            PositionLimits((0.0,) * 6, (0.0,) * 6, (0.1,) * 6, (0.2,) * 6)

    def test_small_feedback_overrun_is_preserved_but_cannot_grow(self) -> None:
        """容差接受真实读数，但目标必须回到名义范围内."""
        limits = PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.1,) * 6,
                                (0.2,) * 6, feedback_tolerance=0.002)
        first = (-1.001,) * 6
        self.assertEqual(limits.check_feedback(first), first)
        inward = JointPositionMove.prepare(first, [-0.995] * 6, 2.0, limits, 0.05)
        self.assertEqual(inward.start, first)
        with self.assertRaises(ValueError):
            JointPositionMove.prepare(first, [-1.0015] * 6, 2.0, limits, 0.05)
        with self.assertRaises(ValueError):
            limits.check_feedback([-1.003] * 6)
        with self.assertRaises(ValueError):
            JointPositionMove.prepare([-0.99] * 6, [-1.001] * 6, 2.0, limits, 0.05)

    def test_return_targets_boundary_but_preserves_recorded_overrun(self) -> None:
        """实测初值保留原值，归位目标单独限制到名义边界."""
        limits = PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.1,) * 6,
                                (0.2,) * 6, feedback_tolerance=0.002)
        home = (-1.001,) * 6
        plan = StopAndReturnMove.prepare([0.0] * 6, [0.0] * 6, home, 1.0, limits)
        self.assertEqual(plan.sample(plan.duration), (-1.0,) * 6)
        self.assertEqual(home, (-1.001,) * 6)
        self.assertGreater(plan.duration, 1.0)

    def test_recovery_only_moves_outside_joints(self) -> None:
        """越界起步先收回，不夹带其他关节的任务运动."""
        limits = PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.1,) * 6,
                                (0.2,) * 6, feedback_tolerance=0.005)
        start = (-1.0038, 1.001) + (0.0,) * 4
        goal = limits.recovery_goal(start)
        self.assertEqual(goal, (-0.995, 0.995) + (0.0,) * 4)
        plan = JointPositionMove.prepare(start, goal, 2.0, limits, 0.05)
        for step in range(101):
            q = plan.sample(step / 50.0)
            self.assertGreaterEqual(q[0], start[0])
            self.assertLessEqual(q[1], start[1])
        invalid = list(goal)
        invalid[2] = 0.01
        with self.assertRaisesRegex(ValueError, '准备阶段'):
            JointPositionMove.prepare(start, invalid, 2.0, limits, 0.05)

    def test_normal_segment_stays_inside_nominal_bounds(self) -> None:
        """名义限位内起终点的单调插值整段均在范围内."""
        limits = fixture_limits()
        plan = JointPositionMove.prepare((-1.99,) * 6, (-1.96,) * 6, 2.0, limits, 0.05)
        for step in range(101):
            limits.check_positions(plan.sample(step / 50.0))

    def test_deceleration_uses_actual_velocity(self) -> None:
        """减速段从实际速度起步并在连接归位段时降为零."""
        limits = fixture_limits()
        plan = StopAndReturnMove.prepare([0.2] * 6, [0.05] * 6,
                                         (0.0,) * 6, 1.0, limits)
        self.assertAlmostEqual(plan.deceleration_time, 0.25)
        self.assertAlmostEqual(plan.sample(plan.deceleration_time)[0], 0.20625)
        step = 1e-6
        first_speed = (plan.sample(step)[0] - plan.sample(0.0)[0]) / step
        self.assertAlmostEqual(first_speed, 0.05, places=5)
        boundary_speed = (
            plan.sample(plan.deceleration_time + step)[0]
            - plan.sample(plan.deceleration_time - step)[0]
        ) / (2 * step)
        self.assertAlmostEqual(boundary_speed, 0.0, places=5)
        self.assertEqual(plan.sample(plan.duration), (0.0,) * 6)

    def test_unsafe_stopping_distance_is_rejected(self) -> None:
        """预计减速距离超出容差包络时，普通归位不继续执行."""
        limits = PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.1,) * 6,
                                (0.2,) * 6, feedback_tolerance=0.002)
        with self.assertRaises(ValueError):
            StopAndReturnMove.prepare([0.999] * 6, [0.1] * 6,
                                      (0.0,) * 6, 2.0, limits)


class PositionAdapterTests(unittest.TestCase):
    """使用官方 Arm 消息构造器，替换所有反馈与网络发送."""

    def setUp(self) -> None:
        """构造离线 Arm，设定测试编码器分辨率和会话状态."""
        arm_type = import_module('hex_device.arm').Arm
        self.arm = arm_type(robot_type=16, motor_count=6, proto_version=(1, 4))
        self.arm._pulse_per_rotation = np.asarray([65536.0] * 6)
        self.arm._error_codes = [None] * 6
        self.arm._calibrated = True
        self.arm._api_control_initialized = False
        self.arm._session_holder = 0
        self.arm._my_session_id = 7
        self.adapter = HexPositionAdapter('ws://127.0.0.1:8439', fixture_limits())
        self.adapter._api = SimpleNamespace(device_list=[self.arm])
        self.adapter._arm_type = arm_type
        self.adapter._command_types = import_module('hex_device.motor_base').CommandType
        self.adapter.state = 'CONNECTED'
        self.positions = (0.0,) * 6
        self.velocity = (0.0,) * 6
        self.reader = patch.object(HexFeedbackAdapter, 'read', side_effect=self.frame)
        self.reader.start()
        self.addCleanup(self.reader.stop)
        self.sender = patch.object(self.adapter, '_send')
        self.send = self.sender.start()
        self.addCleanup(self.sender.stop)

    def frame(self) -> JointFeedback:
        """返回当前测试角度，源时间来自本机单调时钟."""
        return JointFeedback(self.positions, self.velocity, (0.0,) * 6, time.perf_counter_ns())

    def acquire(self) -> None:
        """模拟显式申请和设备确认，未自动连接任何设备."""
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        self.adapter.tick()

    def test_no_position_before_acknowledgement(self) -> None:
        """申请后等待设备确认，尚未确认时不发送位置指令."""
        self.adapter.request_control()
        self.assertEqual(self.adapter.tick(), 'REQUESTING')
        self.assertEqual([c.args[1] for c in self.send.call_args_list], ['initialize'])
        with self.assertRaises(RuntimeError):
            self.adapter.move_to([0.01] * 6, 2.0)

    def test_snapshot_is_actual_acquisition_pose(self) -> None:
        """取得控制权时记录实际反馈，不使用仿真初值."""
        self.positions = (0.001,) * 6
        self.acquire()
        self.assertEqual(self.adapter.initial_positions, self.positions)
        self.assertEqual(self.adapter.state, 'HOLDING')
        message = self.send.call_args.args[0]
        targets = message.arm_command.arm_exclusive_command.arm_api_control_command
        self.assertEqual(len(targets.motor_targets.targets), 6)
        self.assertTrue(all(t.HasField('position') for t in targets.motor_targets.targets))

    def test_official_encoder_conversion(self) -> None:
        """旧型号零弧度按官方偏置换算，发送包不含 MIT 字段."""
        self.acquire()
        message = self.send.call_args.args[0]
        targets = message.arm_command.arm_exclusive_command.arm_api_control_command
        self.assertEqual([t.position for t in targets.motor_targets.targets], [32767] * 6)
        for target in targets.motor_targets.targets:
            self.assertEqual([f.name for f, _ in target.ListFields()], ['position'])

    def test_faults_prevent_request(self) -> None:
        """未校准、其他会话占有和机械臂运动均禁止申请控制."""
        for field, value in (('_calibrated', False), ('_session_holder', 9)):
            with self.subTest(field=field):
                original = getattr(self.arm, field)
                setattr(self.arm, field, value)
                with self.assertRaises(RuntimeError):
                    self.adapter.request_control()
                setattr(self.arm, field, original)
        self.velocity = (0.1,) * 6
        with self.assertRaises(RuntimeError):
            self.adapter.request_control()
        self.send.assert_not_called()

    def test_feedback_stale(self) -> None:
        """过期反馈不允许进入控制流程."""
        stale = JointFeedback((0.0,) * 6, (0.0,) * 6, (0.0,) * 6,
                              time.perf_counter_ns() - 1_000_000_000)
        with patch.object(HexFeedbackAdapter, 'read', return_value=stale):
            with self.assertRaisesRegex(RuntimeError, '过期'):
                self.adapter.request_control()
        self.send.assert_not_called()

    def test_motor_fault_prevents_control(self) -> None:
        """缺少故障反馈或任一电机报错时，不申请控制权."""
        for codes in (None, [None] * 5, [None, 3, None, None, None, None]):
            with self.subTest(codes=codes):
                self.arm._error_codes = codes
                with self.assertRaisesRegex(RuntimeError, '故障'):
                    self.adapter.request_control()
        self.send.assert_not_called()

    def test_tick_gap_latches_fault_and_brakes(self) -> None:
        """调度中断后锁存故障，不用过期轨迹继续发送."""
        self.acquire()
        self.adapter._last_tick -= 0.1
        with self.assertRaisesRegex(RuntimeError, '40 ms'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        self.assertEqual(self.send.call_args.args[1], 'brake')
        calls = self.send.call_count
        self.adapter.tick()
        self.assertEqual(self.send.call_count, calls)

    def test_lost_control_never_commands_other_holder(self) -> None:
        """控制权转移后停止发送，不向其他会话发送位置或制动."""
        self.acquire()
        calls = self.send.call_count
        self.arm._session_holder = 8
        with self.assertRaisesRegex(RuntimeError, '其他会话'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        self.assertEqual(self.send.call_count, calls)

    def test_tracking_error_latches_fault(self) -> None:
        """位置偏差超限后请求制动，不继续追赶目标."""
        self.acquire()
        self.positions = (0.2,) * 6
        with self.assertRaisesRegex(RuntimeError, '偏差超限'):
            self.adapter.tick()
        self.assertEqual(self.send.call_args.args[1], 'brake')

    def test_task_feedback_outside_nominal_latches_fault(self) -> None:
        """正式运动不使用初值容差掩盖实际越界."""
        self.adapter.limits = PositionLimits(
            (-2.0,) * 6, (2.0,) * 6, (0.1,) * 6, (0.2,) * 6, feedback_tolerance=0.005,
        )
        self.positions = (-1.99,) * 6
        self.acquire()
        self.adapter.move_to([-1.97] * 6, 2.0)
        self.positions = (-2.001,) * 6
        with self.assertRaisesRegex(ValueError, '角度'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        self.assertEqual(self.send.call_args.args[1], 'brake')

    def test_recovery_must_enter_nominal_before_success(self) -> None:
        """准备动作不得因一百分之一弧度到位误差而在限位外报告成功."""
        self.adapter.limits = PositionLimits(
            (-2.0,) * 6, (2.0,) * 6, (0.1,) * 6, (0.2,) * 6, feedback_tolerance=0.005,
        )
        self.positions = (-2.003,) * 6
        self.acquire()
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.0
            self.adapter._last_tick = 100.0
            self.adapter.move_to([-1.995] * 6, 2.0)
            for step in range(1, 231):
                clock.return_value = 100.0 + step * 0.01
                self.adapter.tick()
            self.assertEqual(self.adapter.state, 'RUNNING')
            self.positions = (-1.995,) * 6
            for step in range(231, 255):
                clock.return_value = 100.0 + step * 0.01
                self.adapter.tick()
            self.assertEqual(self.adapter.state, 'HOLDING')
            self.assertTrue(self.adapter._nominal_required)
            self.assertEqual(self.adapter.initial_positions, (-2.003,) * 6)
        self.adapter.return_to_initial(2.0)
        assert self.adapter._move is not None
        self.assertEqual(self.adapter._move.goal, (-2.0,) * 6)
        self.assertEqual(self.adapter.initial_positions, (-2.003,) * 6)

    def test_stop_packet_contains_only_true_brakes(self) -> None:
        """停止只编码六个制动字段，不混入位置、速度、力矩或 MIT."""
        self.acquire()
        self.adapter.move_to([0.02] * 6, 2.0)
        api = self.adapter._api
        home = self.adapter.initial_positions
        self.send.reset_mock()
        with patch.object(HexFeedbackAdapter, 'close') as close:
            self.adapter.stop()
            close.assert_not_called()
        self.assertIs(self.adapter._api, api)
        self.assertEqual(self.adapter.initial_positions, home)
        message, kind = self.send.call_args.args
        self.assertEqual(kind, 'brake')
        command = message.arm_command.arm_exclusive_command.arm_api_control_command
        self.assertEqual(len(command.motor_targets.targets), 6)
        for target in command.motor_targets.targets:
            self.assertEqual([(f.name, value) for f, value in target.ListFields()],
                             [('brake', True)])
        calls = self.send.call_count
        self.assertEqual(self.adapter.tick(), 'STOPPED')
        self.assertEqual(self.send.call_count, calls)

    def test_brake_failure_has_no_torque_or_disconnect_fallback(self) -> None:
        """制动发送失败保留错误，不降级为零力矩或关闭连接."""
        self.acquire()
        api = self.adapter._api
        self.send.reset_mock()
        self.send.side_effect = RuntimeError('offline transport failure')
        with patch.object(HexFeedbackAdapter, 'close') as close:
            self.adapter.stop()
            close.assert_not_called()
        self.assertEqual(self.adapter.state, 'STOPPED')
        self.assertIs(self.adapter._api, api)
        self.assertIn('未确认', self.adapter.last_error)
        self.assertEqual(self.send.call_count, 1)
        self.assertEqual(self.send.call_args.args[1], 'brake')

    def test_move_and_explicit_return(self) -> None:
        """归位目标来自记录的真实起点，停止后不自动执行归位."""
        self.acquire()
        self.adapter.move_to([0.02] * 6, 2.0)
        self.assertEqual(self.adapter.state, 'RUNNING')
        self.adapter.stop()
        self.assertEqual(self.adapter.state, 'STOPPED')
        self.assertIsNone(self.adapter._move)
        with self.assertRaises(RuntimeError):
            self.adapter.return_to_initial(2.0)

    def test_complete_move_then_return_actual_initial(self) -> None:
        """到位需依据实际反馈保持稳定，之后显式归位到本次起点."""
        self.positions = (0.005,) * 6
        self.acquire()
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.0
            self.adapter._last_tick = 100.0
            self.adapter.move_to([0.025] * 6, 2.0)
            for step in range(1, 231):
                clock.return_value = 100.0 + step * 0.01
                target = self.adapter._target
                assert target is not None
                self.positions = target
                self.adapter.tick()
            self.assertEqual(self.adapter.state, 'HOLDING')
            self.adapter.return_to_initial(2.0)
            assert self.adapter._move is not None
            self.assertEqual(self.adapter._move.goal, (0.005,) * 6)

    def test_actual_send_boundary_checks_state(self) -> None:
        """执行真实发送协程入口，但只向内存替身发送，不建立 socket."""
        self.sender.stop()
        loop = asyncio.new_event_loop()
        ready = Event()
        loop.call_soon(ready.set)
        thread = Thread(target=loop.run_forever, daemon=True)
        thread.start()
        self.assertTrue(ready.wait(timeout=2.0))
        sent = []

        async def capture(api, message):
            sent.append(message)

        self.adapter._base = SimpleNamespace(_send_down_message=capture)
        self.adapter._api._HexDeviceApi__loop = loop
        try:
            with self.assertRaisesRegex(RuntimeError, '不允许'):
                self.adapter._send(object(), 'position')
            self.assertEqual(sent, [])
            self.adapter.state = 'HOLDING'
            packet = object()
            self.adapter._send(packet, 'position')
            self.assertEqual(sent, [packet])
        finally:
            loop.call_soon_threadsafe(loop.stop)
            thread.join(timeout=2.0)
            loop.close()

    def test_return_interrupts_running_move_using_actual_velocity(self) -> None:
        """显式归位取消原轨迹，保留本轮初始角度和实际减速起点."""
        self.acquire()
        self.adapter.move_to([0.02] * 6, 2.0)
        self.positions = (0.01,) * 6
        self.velocity = (0.03,) * 6
        self.adapter.return_to_initial(2.0)
        self.assertEqual(self.adapter.state, 'RETURNING')
        plan = self.adapter._move
        assert isinstance(plan, StopAndReturnMove)
        self.assertEqual(plan.sample(0.0), self.positions)
        self.assertEqual(plan.velocities, self.velocity)
        self.assertEqual(plan.goal, (0.0,) * 6)

    def test_controlled_return_finishes_in_holding_at_recorded_initial(self) -> None:
        """完整推进 RETURNING 状态，实际到位后保持初始位置而不重复归位."""
        self.acquire()
        self.positions = (0.01,) * 6
        self.velocity = (0.03,) * 6
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.0
            self.adapter._last_tick = 100.0
            self.adapter.return_to_initial(1.0)
            plan = self.adapter._move
            assert isinstance(plan, StopAndReturnMove)
            steps = int((plan.duration + 0.3) / 0.005) + 1
            for step in range(1, steps + 1):
                elapsed = step * 0.005
                clock.return_value = 100.0 + elapsed
                self.positions = plan.sample(elapsed)
                self.velocity = (0.0,) * 6
                self.adapter.tick()
        self.assertEqual(self.adapter.state, 'HOLDING')
        self.assertEqual(self.adapter.initial_positions, (0.0,) * 6)
        self.assertEqual(self.adapter._target, (0.0,) * 6)

    def test_invalid_frequency_and_limit_isolation(self) -> None:
        """频率不接受布尔值，左右实例不共享初始角度或轨迹."""
        with self.assertRaises(ValueError):
            HexPositionAdapter('ws://127.0.0.1:8439', fixture_limits(), control_hz=True)
        other = HexPositionAdapter('ws://127.0.0.1:9439', fixture_limits())
        self.acquire()
        self.assertIsNone(other.initial_positions)
        self.assertIsNone(other._move)

    def test_connect_never_starts_vendor_control_loop(self) -> None:
        """替换网络构造后，验证连接不会放行厂商自动控制消息."""
        sdk = import_module('hex_device.hex_device_api').HexDeviceApi
        message_type = import_module('hex_device.generated.public_api_down_pb2').APIDown
        adapter = HexPositionAdapter('ws://127.0.0.1:9439', fixture_limits())
        sent = []

        async def capture(api, message):
            sent.append(message)

        with patch.object(sdk, '__init__', return_value=None):
            adapter.connect()
        with patch.object(sdk, '_send_down_message', capture):
            command = message_type()
            command.arm_command.arm_exclusive_command.api_control_initialize = True
            asyncio.run(adapter._api._send_down_message(command))
            self.assertEqual(sent, [])
            report = message_type()
            report.set_report_frequency = 1
            asyncio.run(adapter._api._send_down_message(report))
            self.assertEqual(sent, [report])
        with patch.object(sdk, '_start_device_periodic_task') as start:
            adapter._api._start_device_periodic_task(1)
            start.assert_not_called()
        self.assertEqual(adapter.state, 'CONNECTED')
        with patch.object(sdk, 'close'):
            adapter.close()


if __name__ == '__main__':
    unittest.main()
