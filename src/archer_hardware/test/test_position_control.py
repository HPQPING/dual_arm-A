"""离线验证位置控制，所有发送均被替身接收，不连接实际设备."""

import asyncio
from dataclasses import replace
from copy import deepcopy
from importlib import import_module
import time
from types import SimpleNamespace
from typing import Any, cast
import unittest
from unittest.mock import patch

import numpy as np

from archer_hardware.hex_adapter import HexFeedbackAdapter, JointFeedback
from archer_hardware.position_adapter import HexPositionAdapter
from archer_hardware.position_plan import (
    JointPositionMove, JointPositionTrajectory, PairedTrajectoryStart, PositionLimits,
    prepare_continuous_pair,
)


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
        self.assertEqual(limits.return_goal(home), (-1.0,) * 6)
        self.assertEqual(home, (-1.001,) * 6)

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


class FullTrajectoryTests(unittest.TestCase):
    """验证多段执行使用的解析约束、真实起点对齐和连接处连续性."""

    def prepare(self, **overrides) -> JointPositionTrajectory:
        """生成只用于测试的轨迹，不代表实机目标."""
        parameters: dict[str, Any] = {
            'times_ns': (0, 2_000_000_000, 4_000_000_000),
            'positions': ((0.0,) * 6, (0.01,) * 6, (0.02,) * 6),
            'limits': fixture_limits(), 'max_excursion': 0.05,
            'actual_start': (0.0,) * 6,
        }
        parameters.update(overrides)
        return JointPositionTrajectory.prepare(**parameters)

    def test_sampling_and_connections(self) -> None:
        """连接点精确到达且两边速度和加速度趋近零，末端保持."""
        plan = self.prepare()
        self.assertEqual(plan.sample(-1.0), (0.0,) * 6)
        self.assertEqual(plan.sample(2.0), (0.01,) * 6)
        self.assertEqual(plan.sample(100.0), (0.02,) * 6)
        self.assertEqual(plan.duration, 4.0)
        h = 1e-3
        before, center, after = (plan.sample(t)[0] for t in (2 - h, 2.0, 2 + h))
        self.assertLess(abs((after - before) / (2 * h)), 1e-6)
        self.assertLess(abs((after - 2 * center + before) / h ** 2), 1e-4)

    def test_actual_start_alignment_and_mismatch(self) -> None:
        """容差内从实际角度起步，超出容差则拒绝而非自动跳到旧起点."""
        plan = self.prepare(actual_start=(0.003,) * 6)
        self.assertEqual(plan.sample(0.0), (0.003,) * 6)
        with self.assertRaisesRegex(ValueError, '初始角度'):
            self.prepare(actual_start=(0.006,) * 6)

    def test_alignment_rechecks_first_segment_acceleration(self) -> None:
        """调整首点增加首段位移后，必须重新验证峰值而非沿用旧报告."""
        limits = PositionLimits((-1.0,) * 6, (1.0,) * 6, (1.0,) * 6, (0.02,) * 6)
        with self.assertRaisesRegex(ValueError, '加速度超限'):
            self.prepare(times_ns=(0, 1_000_000_000),
                         positions=((0.0,) * 6, (0.003,) * 6),
                         actual_start=(-0.003,) * 6, limits=limits)

    def test_global_excursion_cannot_be_bypassed_by_small_segments(self) -> None:
        """每段很小但累计偏离实际起点超限时，完整轨迹仍拒绝."""
        with self.assertRaisesRegex(ValueError, '完整轨迹超出'):
            self.prepare(times_ns=(0, 2_000_000_000, 4_000_000_000, 6_000_000_000),
                         positions=((0.0,) * 6, (0.02,) * 6, (0.04,) * 6, (0.06,) * 6))

    def test_nominal_limits_required_after_initial_preparation(self) -> None:
        """完整正式计划不沿用初值轻微越界容差，须先完成准备."""
        limits = PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.1,) * 6, (0.2,) * 6, 0.005)
        with self.assertRaisesRegex(ValueError, '超出'):
            self.prepare(limits=limits, actual_start=(-1.001,) * 6)

    def test_invalid_times_and_elapsed(self) -> None:
        """拒绝浮点纳秒、重复时间、超长计划与非有限采样时间."""
        for times in ((0, True), (0, 2.0), (0, 0), (1, 2), (0, 3601 * 10 ** 9)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.prepare(times_ns=times, positions=((0.0,) * 6, (0.01,) * 6))
        plan = self.prepare()
        for elapsed in (True, float('nan'), float('inf')):
            with self.subTest(elapsed=elapsed), self.assertRaises(ValueError):
                plan.sample(elapsed)


class ContinuousExecutionTests(unittest.TestCase):
    """版本二文件重新解析并验证，计算过程中没有SDK连接或ROS初始化."""

    def setUp(self) -> None:
        """创建含非零内部速度的合成闭合轨迹，只用于测试."""
        from archer_control.continuous_trajectory import retime_recorded_pair

        self.bound = fixture_limits()
        self.bounds = {'left': self.bound, 'right': self.bound}
        q = np.repeat(np.array([0., .005, .012, .02, .018, 0.])[:, None], 6, axis=1)
        self.reference, _ = retime_recorded_pair(
            (0, 10 ** 9, 2 * 10 ** 9, 3 * 10 ** 9, 4 * 10 ** 9, 5 * 10 ** 9),
            q, q, [self.bound.lower] * 2, [self.bound.upper] * 2,
            [self.bound.max_velocity] * 2, [self.bound.max_acceleration] * 2,
        )
        self.payload = self.reference.payload()

    def test_payload_reconstructs_same_continuous_curve(self) -> None:
        """读取文件后的取样与准备阶段一致，内部点不被改成逐点停车."""
        plans = prepare_continuous_pair(self.payload, self.bounds, .05)
        self.assertGreater(abs(plans['left'].sample(float(self.reference.times[1]), 1)[0]), 0.)
        for side in ('left', 'right'):
            for elapsed in np.linspace(0., self.reference.times[-1], 101):
                for order in (0, 1, 2):
                    np.testing.assert_allclose(
                        plans[side].sample(float(elapsed), order),
                        self.reference.sample(side, float(elapsed), order), atol=1e-12, rtol=0.,
                    )
            np.testing.assert_array_equal(plans[side].sample(plans[side].duration + 1.), [0.] * 6)
            self.assertFalse(plans[side].polynomials[0].flags.writeable)

    def test_actual_start_alignment_only_rebuilds_first_segment(self) -> None:
        """允许容差内的实际起点并重检首段，保留其余多项式及共同时间轴."""
        plan = prepare_continuous_pair(self.payload, self.bounds, .05)['left']
        aligned = plan.aligned_start((.001,) * 6, self.bound, .05)
        self.assertEqual(aligned.sample(0.), (.001,) * 6)
        self.assertEqual(aligned.times_ns, plan.times_ns)
        self.assertEqual(plan.initial, (0.,) * 6)
        for old, new in zip(plan.polynomials[1:], aligned.polynomials[1:]):
            self.assertIs(old, new)
        with self.assertRaisesRegex(ValueError, '实际初始角度'):
            plan.aligned_start((.006,) * 6, self.bound, .05)
        changed = PositionLimits(self.bound.lower, self.bound.upper,
                                 (.09,) * 6, self.bound.max_acceleration)
        with self.assertRaisesRegex(ValueError, '限制'):
            plan.aligned_start((0.,) * 6, changed, .05)
        with self.assertRaisesRegex(ValueError, '允许位移'):
            plan.aligned_start((0.,) * 6, self.bound, .001)

    def test_forged_derivatives_times_and_stops_are_rejected(self) -> None:
        """拒绝错误版本、不共用时间、缺失导数、假停点及段内超速."""
        cases = []
        for variant in ('version', 'time', 'field', 'stop', 'velocity', 'initial_speed'):
            data = deepcopy(self.payload)
            if variant == 'version':
                data['version'] = True
            elif variant == 'time':
                data['right']['points'][1]['time_from_start']['nanosec'] += 1
            elif variant == 'field':
                del data['left']['points'][1]['accelerations']
            elif variant == 'stop':
                data['stop_indices'] = [0, 1, 5]
            elif variant == 'velocity':
                data['left']['points'][1]['velocities'] = [2.] * 6
            else:
                data['left']['points'][0]['velocities'] = [.001] * 6
            cases.append(data)
        for data in cases:
            with self.subTest(data=str(data)[:50]), self.assertRaises(ValueError):
                prepare_continuous_pair(data, self.bounds, .05)

    def test_interior_overshoot_is_checked_not_only_waypoints(self) -> None:
        """端点都在限制内，但导数造成段内过冲时仍须拒绝."""
        data = deepcopy(self.payload)
        for side in ('left', 'right'):
            points = data[side]['points']
            points[1]['positions'] = points[2]['positions'] = [.99] * 6
            points[1]['velocities'] = [.1] * 6
            points[2]['velocities'] = [-.1] * 6
        wide = PositionLimits((-1.,) * 6, (1.,) * 6, (10.,) * 6, (100.,) * 6)
        with self.assertRaisesRegex(ValueError, '角度'):
            prepare_continuous_pair(data, {'left': wide, 'right': wide}, 2.)

    def test_alignment_rechecks_first_segment_speed(self) -> None:
        """即使初值差在容差内，首段太短也不能通过瞬间跳转对齐."""
        data = deepcopy(self.payload)
        for side in ('left', 'right'):
            for i, point in enumerate(data[side]['points']):
                point['positions'] = point['velocities'] = point['accelerations'] = [0.] * 6
                point['time_from_start'] = {'sec': 0, 'nanosec': i * 20000000}
        plan = prepare_continuous_pair(data, self.bounds, .05)['left']
        with self.assertRaisesRegex(ValueError, '速度或加速度'):
            plan.aligned_start((.004,) * 6, self.bound, .05)

    def test_sampling_does_not_repeat_peak_preflight(self) -> None:
        """发送周期不重复调用多项式实根预检，避免阻塞位置目标发送."""
        plan = prepare_continuous_pair(self.payload, self.bounds, .05)['left']
        with patch('archer_control.continuous_trajectory._extrema',
                   side_effect=AssertionError('cycle repeated preflight')):
            for elapsed in np.linspace(0., plan.duration, 100):
                plan.sample(float(elapsed))

    def test_cancelled_preflight_stops_during_segment_work(self) -> None:
        """取消检查覆盖段内实根计算循环，不必把整条轨迹计算完再丢弃."""
        calls = 0

        def cancelled():
            nonlocal calls
            calls += 1
            return calls >= 8

        with self.assertRaisesRegex(ValueError, '已取消'):
            prepare_continuous_pair(self.payload, self.bounds, .05, cancelled=cancelled)
        self.assertEqual(calls, 8)


class PairedStartTests(unittest.TestCase):
    """共享门控不访问设备，准备不完整或取消后不能释放启动."""

    def test_both_distinct_sides_required(self) -> None:
        """两个不同侧准备后才给出共同未来时刻."""
        with patch('archer_hardware.position_plan.time.monotonic', return_value=100.0):
            gate = PairedTrajectoryStart()
            gate.ready('left')
            self.assertIsNone(gate.start_time())
            with self.assertRaises(ValueError):
                gate.ready('left')
            gate.ready('right')
            start = gate.start_time()
            assert start is not None
            self.assertAlmostEqual(start, 100.2)
            gate.cancel()
            with self.assertRaises(RuntimeError):
                gate.start_time()

    def test_preparation_timeout_cannot_be_reused(self) -> None:
        """迟到的第二侧不能启动已超时的一次性计划."""
        with patch('archer_hardware.position_plan.time.monotonic') as clock:
            clock.return_value = 100.0
            gate = PairedTrajectoryStart()
            gate.ready('left')
            clock.return_value = 100.6
            with self.assertRaises(RuntimeError):
                gate.start_time()
            with self.assertRaises(RuntimeError):
                gate.ready('right')
            self.assertTrue(gate.cancelled)


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
        self.adapter = HexPositionAdapter('ws://127.0.0.1:8439', fixture_limits(),
                                          allow_api_calibration=False)
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

    def test_local_receipt_allows_hold_but_never_refreshes_old_feedback(self) -> None:
        """走正式读取及保持入口，SDK时间略领先不会误停，真正超时仍拒绝."""
        self.reader.stop()
        received = time.perf_counter_ns()
        sdk_stamp = received + 2000000
        status = {
            'pos': self.positions, 'vel': self.velocity, 'eff': (0.,) * 6,
            'ts': {'s': sdk_stamp // 1000000000, 'ns': sdk_stamp % 1000000000},
        }
        self.adapter._api.is_api_exit = lambda: False
        self.adapter._api.is_websocket_recv_timeout = lambda: False
        self.adapter._api._project_feedback_frames = {16: (self.arm, received, status)}
        with patch('archer_hardware.hex_adapter.time.perf_counter_ns',
                   return_value=received + 1000000):
            self.acquire()
            self.assertEqual(self.adapter.state, 'HOLDING')
            self.assertEqual(self.adapter._target, self.positions)
            self.assertTrue(any(call.args[1] == 'position'
                                for call in self.send.call_args_list))
        self.send.reset_mock()
        with patch('archer_hardware.hex_adapter.time.perf_counter_ns',
                   return_value=received + 100000001):
            with self.assertRaisesRegex(RuntimeError, '过期'):
                self.adapter.tick()
        self.assertFalse(any(call.args[1] == 'position'
                             for call in self.send.call_args_list))

    def continuous_plan(self):
        """准备小幅合成连续轨迹，完整预检先于模拟申请控制."""
        from archer_control.continuous_trajectory import retime_recorded_pair

        limits = self.adapter.limits
        q = np.repeat(np.array([0., .005, .012, .02, .018, 0.])[:, None], 6, axis=1)
        plan, _ = retime_recorded_pair(
            (0, 10 ** 9, 2 * 10 ** 9, 3 * 10 ** 9, 4 * 10 ** 9, 5 * 10 ** 9),
            q, q, [limits.lower] * 2, [limits.upper] * 2,
            [limits.max_velocity] * 2, [limits.max_acceleration] * 2,
        )
        return prepare_continuous_pair(plan.payload(), {'left': limits, 'right': limits}, .05)

    def set_api_timeout(self, **changes) -> None:
        """设置可远程清除的通信超时或其变种，不连接实际控制器."""
        types = import_module('hex_device.generated.public_api_types_pb2')
        fields = {'reason': 'API communication timeout', 'is_remotely_clearable': True,
                  'category': types.ParkingStopCategory.PscAPICommunicationTimeout}
        fields.update(changes)
        self.arm._parking_stop_detail.CopyFrom(types.ParkingStopDetail(**fields))

    def test_startup_clear_waits_for_new_feedback_before_initialization(self) -> None:
        """单次清除后等更新实测帧确认，不能同周期申请控制或进入保持."""
        self.set_api_timeout()
        first = self.frame()
        with patch.object(HexFeedbackAdapter, 'read', return_value=first):
            self.adapter.request_control()
            self.assertEqual(self.adapter.readiness_stage, 'CLEARING_API_TIMEOUT')
            for _ in range(3):
                self.adapter.tick()
            self.arm._parking_stop_detail.Clear()
            self.adapter.tick()
        self.assertEqual([c.args[1] for c in self.send.call_args_list], ['clear_api_timeout'])
        self.adapter.tick()
        self.assertEqual([c.args[1] for c in self.send.call_args_list],
                         ['clear_api_timeout', 'initialize'])
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        self.adapter.tick()
        self.assertEqual(self.adapter.state, 'HOLDING')
        self.assertEqual(self.adapter.initial_positions, self.positions)

    def test_unknown_parking_and_unsafe_clear_conditions_are_rejected(self) -> None:
        """不可远程清除、原因不符、电机故障、其他会话和运动均不清除."""
        for changes in ({'is_remotely_clearable': False}, {'reason': 'Other stop'},
                        {'category': 0}):
            self.set_api_timeout(**changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError, '停车'):
                self.adapter.request_control()
        self.set_api_timeout()
        for name, value in (('_session_holder', 9), ('_error_codes', [3] * 6)):
            original = getattr(self.arm, name)
            setattr(self.arm, name, value)
            with self.subTest(field=name), self.assertRaises(RuntimeError):
                self.adapter.request_control()
            setattr(self.arm, name, original)
        self.velocity = (.03,) * 6
        with self.assertRaisesRegex(RuntimeError, '运动'):
            self.adapter.request_control()
        self.send.assert_not_called()

    def test_startup_clear_timeout_and_halt_never_retry(self) -> None:
        """清除未确认超时后不重复清除，迟到确认不能恢复初始化或位置发送."""
        self.set_api_timeout()
        self.adapter.request_control()
        self.adapter._requested_at -= self.adapter.readiness_timeout + 1.
        with self.assertRaisesRegex(RuntimeError, '就绪超时.*API通信超时'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        self.arm._parking_stop_detail.Clear()
        self.adapter.tick()
        self.assertEqual([c.args[1] for c in self.send.call_args_list], ['clear_api_timeout'])

    def test_clear_dispatch_rechecks_reason_payload_and_feedback(self) -> None:
        """真实发送前再次核对清除原因、消息纯度及反馈有效期."""
        self.set_api_timeout()
        self.adapter.request_control()
        message = self.send.call_args.args[0]
        self.adapter._check_dispatch(message, 'clear_api_timeout')
        mixed = type(message)()
        mixed.CopyFrom(message)
        mixed.arm_command.arm_exclusive_command.api_control_initialize = True
        with self.assertRaisesRegex(RuntimeError, '单独'):
            self.adapter._check_dispatch(mixed, 'clear_api_timeout')
        self.set_api_timeout(reason='Other stop')
        with self.assertRaisesRegex(RuntimeError, '停车'):
            self.adapter._check_dispatch(message, 'clear_api_timeout')
        self.set_api_timeout()
        self.adapter._feedback = replace(self.frame(), source_ns=time.perf_counter_ns()-10**9)
        with self.assertRaisesRegex(RuntimeError, '新鲜'):
            self.adapter._check_dispatch(message, 'clear_api_timeout')
        self.adapter.stop()
        with self.assertRaisesRegex(RuntimeError, '不允许'):
            self.adapter._check_dispatch(message, 'clear_api_timeout')

    def test_checked_path_rejects_drift_before_arm(self) -> None:
        """已检文件不能在SDK执行入口悄悄重建首段."""
        plan = replace(self.continuous_plan()['left'], preserve_path=True)
        self.acquire()
        self.positions = (.000001,) * 6
        self.send.reset_mock()
        with self.assertRaisesRegex(ValueError, '禁止重建首段'):
            self.adapter.arm_continuous(plan, 'left', PairedTrajectoryStart())
        self.send.assert_not_called()
        self.assertEqual(self.adapter.state, 'HOLDING')

    def test_checked_path_waiting_drift_cancels_gate(self) -> None:
        """双方准备期间实际起点再次变化，不能继续沿旧几何路径启动."""
        plan = replace(self.continuous_plan()['left'], preserve_path=True)
        self.acquire()
        gate = PairedTrajectoryStart()
        self.adapter.arm_continuous(plan, 'left', gate)
        self.positions = (.000001,) * 6
        with self.assertRaisesRegex(RuntimeError, '等待启动时起点变化'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        self.assertTrue(gate.cancelled)

    def test_guarded_feedback_keeps_command_anchor_and_original_curve(self) -> None:
        """原始反馈有微小偏差时，不把目标改成反馈，不重建多项式."""
        self.adapter.limits = replace(self.adapter.limits, feedback_tolerance=.005)
        plan = replace(self.continuous_plan()['left'], preserve_path=True, joint_error_rad=.0001)
        self.acquire()
        self.positions = (.00005,) * 6
        self.assertEqual(self.adapter.commanded_positions, (0.,) * 6)
        gate = PairedTrajectoryStart()
        self.adapter.arm_continuous(plan, 'left', gate)
        self.assertIs(self.adapter._move, plan)
        self.assertEqual(self.adapter._target, (0.,) * 6)
        self.adapter.tick()
        self.assertEqual(self.adapter._target, (0.,) * 6)
        self.assertIsNone(self.adapter.commanded_positions)
        self.positions = (.000101,) * 6
        with self.assertRaisesRegex(RuntimeError, '等待启动时起点变化'):
            self.adapter.tick()
        self.assertTrue(gate.cancelled)

    def test_guarded_path_rejects_changed_command_even_when_feedback_matches(self) -> None:
        """反馈看似符合不能掩盖保持目标已被其他动作更改."""
        self.adapter.limits = replace(self.adapter.limits, feedback_tolerance=.005)
        plan = replace(self.continuous_plan()['left'], preserve_path=True, joint_error_rad=.0001)
        self.acquire()
        self.adapter._target = (.00005,) * 6
        with self.assertRaisesRegex(ValueError, '保持指令目标'):
            self.adapter.arm_continuous(plan, 'left', PairedTrajectoryStart())

    def test_guarded_curve_error_validation_and_nominal_limits(self) -> None:
        """不允许伪造宽容差、非有限数或用读取容差放宽正式限位."""
        self.adapter.limits = replace(self.adapter.limits, feedback_tolerance=.005)
        plan = self.continuous_plan()['left']
        guarded = replace(plan, preserve_path=True, joint_error_rad=.0001)
        self.assertIs(guarded.aligned_start((.00005,) * 6, plan.limits, .05), guarded)
        for error in (-.0001, .00101, float('nan'), float('inf'), True):
            with self.subTest(error=error), self.assertRaises(ValueError):
                replace(guarded, joint_error_rad=error).aligned_start(
                    plan.initial, plan.limits, .05,
                )
        with self.assertRaisesRegex(ValueError, '已检完整路径'):
            replace(plan, joint_error_rad=.0001).aligned_start(plan.initial, plan.limits, .05)
        with self.assertRaises(ValueError):
            guarded.aligned_start((2.00001,) * 6, plan.limits, .05)

    def test_guarded_running_error_cancels_before_next_position_command(self) -> None:
        """执行中不能只用旧的宽跟随容差冒充几何误差界限."""
        self.adapter.limits = replace(self.adapter.limits, feedback_tolerance=.005)
        plan = replace(self.continuous_plan()['left'], preserve_path=True, joint_error_rad=.0001)
        self.acquire()
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.
            self.adapter._last_tick = 100.
            gate = PairedTrajectoryStart()
            self.adapter.arm_continuous(plan, 'left', gate)
            gate.ready('right')
            start = gate.start_time()
            assert start is not None
            for step in range(1, 61):
                clock.return_value = 100. + step * .005
                self.positions = plan.sample(max(0., clock.return_value - start))
                self.adapter.tick()
            self.assertEqual(self.adapter.state, 'RUNNING')
            clock.return_value += .005
            self.positions = tuple(q + .0002 for q in plan.sample(clock.return_value - start))
            self.send.reset_mock()
            with self.assertRaisesRegex(RuntimeError, '几何误差界限'):
                self.adapter.tick()
            self.assertFalse(any(call.args[1] == 'position' for call in self.send.call_args_list))
            self.assertTrue(gate.cancelled)

    def test_continuous_cycle_shared_start_and_home_hold(self) -> None:
        """连续轨迹按共同时刻启动，返回记录起点并保持，仅发送位置模式字段."""
        plan = self.continuous_plan()['left']
        self.acquire()
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.
            self.adapter._last_tick = 100.
            gate = PairedTrajectoryStart()
            self.adapter.arm_continuous(plan, 'left', gate)
            self.assertEqual(self.adapter.state, 'ARMED')
            self.adapter.tick()
            self.assertIsNone(self.adapter.trajectory_start_at)
            gate.ready('right')
            start = gate.start_time()
            assert start is not None
            for step in range(1, int((plan.duration + .6) / .005) + 1):
                clock.return_value = 100. + step * .005
                self.positions = plan.sample(max(0., clock.return_value - start))
                self.adapter.tick()
            self.assertEqual(self.adapter.trajectory_start_at, start)
        self.assertEqual(self.adapter.state, 'HOLDING')
        self.assertEqual(self.adapter._target, (0.,) * 6)
        self.assertEqual(self.adapter.initial_positions, (0.,) * 6)
        self.assertTrue(all(call.args[1] in ('initialize', 'position')
                            for call in self.send.call_args_list))
        packet = self.send.call_args.args[0]
        targets = packet.arm_command.arm_exclusive_command.arm_api_control_command
        for target in targets.motor_targets.targets:
            self.assertEqual([field.name for field, _ in target.ListFields()], ['position'])

    def test_continuous_halt_prevents_further_positions(self) -> None:
        """连续运动中停止取消共同门控，后续周期不发送位置或自动归位."""
        plan = self.continuous_plan()['left']
        self.acquire()
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.
            self.adapter._last_tick = 100.
            gate = PairedTrajectoryStart()
            self.adapter.arm_continuous(plan, 'left', gate)
            gate.ready('right')
            for i in range(1, 81):
                clock.return_value = 100. + i * .005
                self.positions = self.adapter._target or (0.,) * 6
                self.adapter.tick()
            self.assertEqual(self.adapter.state, 'RUNNING')
            self.adapter.stop()
            self.assertTrue(gate.cancelled)
            self.send.reset_mock()
            for _ in range(5):
                self.assertEqual(self.adapter.tick(), 'STOPPED')
            self.send.assert_not_called()
            self.assertIsNone(self.adapter._move)

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
        """其他会话占有和机械臂运动均禁止申请控制."""
        for field, value in (('_session_holder', 9), ('_my_session_id', 0)):
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

    def test_readonly_and_unknown_models_never_request_control(self) -> None:
        """即使反馈和会话正常，只读 Hello 或未知型号也不发送初始化."""
        for robot_type in (26, 28, 999):
            with self.subTest(robot_type=robot_type):
                self.arm.robot_type = robot_type
                with self.assertRaisesRegex(RuntimeError, '位置控制型号'):
                    self.adapter.request_control()
                self.assertEqual(self.adapter.state, 'CONNECTED')
        self.send.assert_not_called()

    def test_initialization_is_separate_from_calibrated(self) -> None:
        """type25可先申请初始化，校准状态未就绪前不发送位置或校准命令."""
        self.arm.robot_type = 25
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        for _ in range(3):
            self.assertEqual(self.adapter.tick(), 'REQUESTING')
        self.assertIsNone(self.adapter.initial_positions)
        self.assertEqual([c.args[1] for c in self.send.call_args_list],
                         ['initialize'] + ['heartbeat'] * 3)
        self.arm._calibrated = True
        self.positions = (0.002,) * 6
        self.assertEqual(self.adapter.tick(), 'HOLDING')
        self.assertEqual(self.adapter.initial_positions, self.positions)
        self.assertEqual([c.args[1] for c in self.send.call_args_list],
                         ['initialize'] + ['heartbeat'] * 3 + ['position'])

    def test_calibrated_does_not_replace_initialization(self) -> None:
        """校准状态已就绪仍须等会话和API确认，不能提前发送位置."""
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.assertEqual(self.adapter.tick(), 'REQUESTING')
        self.assertIsNone(self.adapter.initial_positions)
        self.assertEqual(self.adapter.state, 'REQUESTING')
        self.assertEqual([c.args[1] for c in self.send.call_args_list],
                         ['initialize', 'heartbeat'])

    def test_calibration_wait_timeout_reports_reason(self) -> None:
        """等待校准状态超时明确说明原因，不发送位置或自动校准."""
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        self.adapter._requested_at -= self.adapter.readiness_timeout + 1.0
        with self.assertRaisesRegex(RuntimeError, '就绪超时.*calibrated=true'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        self.assertNotIn('position', [c.args[1] for c in self.send.call_args_list])
        self.assertIsNone(self.adapter.initial_positions)

    def test_calibrated_loss_blocks_position_after_acquisition(self) -> None:
        """取得控制后状态失效立即取消位置发送，沿用已有制动处理."""
        self.acquire()
        self.send.reset_mock()
        self.arm._calibrated = False
        with self.assertRaisesRegex(RuntimeError, 'calibrated 状态失效'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        self.assertEqual([c.args[1] for c in self.send.call_args_list], ['brake'])

    def test_one_calibration_after_session_without_initialization_deadlock(self) -> None:
        """显式允许且取得控制会话后校准一次，不以初始化未完成阻塞校准."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.adapter.tick()
        self.assertEqual(self.adapter.readiness_stage, 'WAITING_SESSION')
        self.arm._session_holder = 7
        self.adapter.tick()
        self.assertEqual(self.adapter.readiness_stage, 'WAITING_CALIBRATION')
        self.assertFalse(self.arm._api_control_initialized)
        self.assertEqual(self.send.call_count, 2)
        packet = self.send.call_args.args[0]
        exclusive = packet.arm_command.arm_exclusive_command
        self.assertEqual(exclusive.WhichOneof('exclusive_command'), 'calibrate')
        self.assertTrue(exclusive.calibrate)
        for _ in range(5):
            self.adapter.tick()
        self.assertEqual(self.adapter.readiness_stage, 'WAITING_CALIBRATION')
        self.assertEqual([c.args[1] for c in self.send.call_args_list],
                         ['initialize', 'calibrate'] + ['heartbeat'] * 5)
        self.assertIsNone(self.adapter.initial_positions)
        self.arm._calibrated = True
        self.adapter.tick()
        self.assertEqual(self.adapter.readiness_stage, 'WAITING_INITIALIZATION')
        self.assertEqual(self.send.call_count, 8)
        self.assertIsNone(self.adapter.initial_positions)
        self.positions = (.00005,) * 6
        self.arm._api_control_initialized = True
        self.adapter.tick()
        self.assertEqual(self.adapter.readiness_stage, 'READY')
        self.assertEqual(self.adapter.initial_positions, self.positions)
        self.assertEqual(self.adapter.commanded_positions, self.positions)
        self.assertEqual(self.send.call_args.args[1], 'position')

    def test_uncalibrated_without_initialization_still_requires_permission(self) -> None:
        """状态顺序调整不能使默认配置自动发送校准或位置消息."""
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        for _ in range(3):
            self.adapter.tick()
        self.assertEqual(self.adapter.readiness_stage, 'CALIBRATION_PERMISSION_REQUIRED')
        self.assertEqual([c.args[1] for c in self.send.call_args_list],
                         ['initialize'] + ['heartbeat'] * 3)
        self.assertIsNone(self.adapter.initial_positions)

    def test_calibration_before_initialization_cannot_recover_lost_session(self) -> None:
        """校准请求后会话丢失立即作废，不因迟到确认恢复."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.adapter.tick()
        self.arm._session_holder = 0
        with self.assertRaisesRegex(RuntimeError, '校准请求后'):
            self.adapter.tick()
        self.arm._session_holder = 7
        self.arm._calibrated = True
        self.arm._api_control_initialized = True
        count = self.send.call_count
        self.adapter.tick()
        self.assertEqual(self.send.call_count, count)
        self.assertNotIn('position', [c.args[1] for c in self.send.call_args_list])

    def test_calibration_before_initialization_timeout_is_not_retried(self) -> None:
        """初始化和校准都未确认时，超时退出且校准不重发."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.adapter.tick()
        self.adapter._requested_at -= self.adapter.readiness_timeout + 1.
        with self.assertRaisesRegex(RuntimeError, '就绪超时.*初始化.*calibrated=true'):
            self.adapter.tick()
        self.adapter.tick()
        kinds = [c.args[1] for c in self.send.call_args_list]
        self.assertEqual(kinds.count('calibrate'), 1)
        self.assertNotIn('position', kinds)
        self.assertEqual(self.adapter.state, 'FAULT')

    def test_halt_before_initialization_blocks_late_ready_confirmation(self) -> None:
        """尚未初始化时停止，后来的两个确认都不能触发位置保持."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.adapter.tick()
        self.adapter.stop()
        count = self.send.call_count
        self.arm._api_control_initialized = True
        self.arm._calibrated = True
        self.adapter.tick()
        self.assertEqual(self.send.call_count, count)
        self.assertEqual(self.adapter.state, 'STOPPED')
        self.assertIsNone(self.adapter.initial_positions)

    def test_explicitly_disabled_calibration_never_sends_calibration(self) -> None:
        """显式禁用校准时等待许可，不绕过设置进入保持."""
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        self.adapter.tick()
        self.assertEqual(self.adapter.readiness_stage, 'CALIBRATION_PERMISSION_REQUIRED')
        self.assertEqual([c.args[1] for c in self.send.call_args_list],
                         ['initialize', 'heartbeat'])

    def test_waiting_heartbeat_contains_no_motor_targets_and_stops_on_halt(self) -> None:
        """等待阶段持续空目标通信，不绕过校准许可，也不在停止后续发."""
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        for _ in range(20):
            self.adapter.tick()
            message, kind = self.send.call_args.args
            self.assertEqual(kind, 'heartbeat')
            command = message.arm_command.arm_exclusive_command
            self.assertEqual(command.WhichOneof('exclusive_command'), 'arm_api_control_command')
            control = command.arm_api_control_command
            self.assertEqual([f.name for f, _ in control.ListFields()], ['motor_targets'])
            self.assertEqual(len(control.motor_targets.targets), 0)
        self.assertEqual(self.adapter.state, 'REQUESTING')
        self.assertFalse(self.adapter._calibration_sent)
        self.adapter.stop()
        count = self.send.call_count
        self.adapter.tick()
        self.assertEqual(self.send.call_count, count)
        self.assertIsNone(self.adapter.initial_positions)

    def test_already_ready_skips_calibration_even_when_allowed(self) -> None:
        """设备已有校准确认时不重复校准."""
        self.adapter.allow_api_calibration = True
        self.acquire()
        self.assertFalse(self.adapter._calibration_sent)
        self.assertNotIn('calibrate', [c.args[1] for c in self.send.call_args_list])

    def test_calibration_send_failure_is_not_retried(self) -> None:
        """校准传输结果不明时结束当前运行，不重发潜在动作."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True

        def fail_calibration(message, kind):
            if kind == 'calibrate':
                raise RuntimeError('calibration send unconfirmed')

        self.send.side_effect = fail_calibration
        with self.assertRaisesRegex(RuntimeError, 'send unconfirmed'):
            self.adapter.tick()
        self.assertTrue(self.adapter._calibration_sent)
        self.assertEqual(self.adapter.state, 'FAULT')
        self.adapter.tick()
        self.assertEqual([c.args[1] for c in self.send.call_args_list].count('calibrate'), 1)
        self.assertNotIn('position', [c.args[1] for c in self.send.call_args_list])

    def check_waiting_motion(self, positions, velocity) -> None:
        """构造等待中异常反馈，验证先取消再沿用停止处理."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        self.positions, self.velocity = positions, velocity
        with self.assertRaisesRegex(RuntimeError, '出现运动|漂移'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        kinds = [c.args[1] for c in self.send.call_args_list]
        self.assertNotIn('calibrate', kinds)
        self.assertNotIn('position', kinds)

    def test_waiting_motion_blocks_calibration(self) -> None:
        """等待时速度异常先取消，不发送校准和位置目标."""
        self.check_waiting_motion((0.0,) * 6, (0.03,) * 6)

    def test_waiting_drift_blocks_calibration(self) -> None:
        """速度为零但累计关节位置漂移超限时仍拒绝继续."""
        self.check_waiting_motion((0.011,) * 6, (0.0,) * 6)

    def test_calibration_wait_cannot_resume_after_halt(self) -> None:
        """停止使排队校准失效；随后确认到来不能恢复当前位置发送."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        self.adapter.tick()
        self.adapter.stop()
        calls = self.send.call_count
        self.arm._calibrated = True
        self.adapter.tick()
        self.assertEqual(self.adapter.state, 'STOPPED')
        self.assertEqual(self.send.call_count, calls)
        self.assertIsNone(self.adapter.initial_positions)

    def test_calibration_confirmation_timeout_has_no_retry(self) -> None:
        """校准消息已发送但未确认时，超时后结束且不重复发送."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        self.adapter.tick()
        self.adapter._requested_at -= self.adapter.readiness_timeout + 1.0
        with self.assertRaisesRegex(RuntimeError, '就绪超时.*calibrated=true'):
            self.adapter.tick()
        self.adapter.tick()
        kinds = [c.args[1] for c in self.send.call_args_list]
        self.assertEqual(kinds.count('calibrate'), 1)
        self.assertNotIn('position', kinds)
        self.assertEqual(self.adapter.state, 'FAULT')

    def test_lost_initialization_after_calibration_ends_request(self) -> None:
        """校准请求后初始化状态丢失不重新申请或校准."""
        self.adapter.allow_api_calibration = True
        self.arm._calibrated = False
        self.adapter.request_control()
        self.arm._session_holder = 7
        self.arm._api_control_initialized = True
        self.adapter.tick()
        self.arm._api_control_initialized = False
        with self.assertRaisesRegex(RuntimeError, '校准请求后'):
            self.adapter.tick()
        kinds = [c.args[1] for c in self.send.call_args_list]
        self.assertEqual(kinds.count('initialize'), 1)
        self.assertEqual(kinds.count('calibrate'), 1)
        self.assertNotIn('position', kinds)

    def test_calibration_parameters_are_validated(self) -> None:
        """校准许可严格为布尔值，等待时间有界且拒绝非有限数值."""
        for value in (None, 0, 1, 'true'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HexPositionAdapter('ws://127.0.0.1:8439', fixture_limits(),
                                   allow_api_calibration=cast(Any, value))
        for value in (True, 0, -1, float('nan'), float('inf'), 30.1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                HexPositionAdapter('ws://127.0.0.1:8439', fixture_limits(),
                                   readiness_timeout=value)

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
        initial = self.adapter.initial_positions
        assert initial is not None
        self.assertEqual(self.adapter.limits.return_goal(initial), (-2.0,) * 6)
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

    def test_stop_cannot_start_another_move(self) -> None:
        """停止清空轨迹；不提供独立归位入口，也不能继续发送目标."""
        self.acquire()
        self.adapter.move_to([0.02] * 6, 2.0)
        self.adapter.stop()
        self.assertEqual(self.adapter.state, 'STOPPED')
        self.assertIsNone(self.adapter._move)
        self.assertFalse(hasattr(self.adapter, 'return_to_initial'))
        with self.assertRaises(RuntimeError):
            self.adapter.move_to([0.0] * 6, 2.0)

    def test_full_trajectory_completes_and_holds_without_auto_return(self) -> None:
        """完整接口只执行提交的多段轨迹，到位保持，不追加其他动作."""
        self.acquire()
        self.positions = (0.001,) * 6
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.0
            self.adapter._last_tick = 100.0
            self.adapter.follow_trajectory(
                (0, 2_000_000_000, 4_000_000_000),
                ((0.0,) * 6, (0.01,) * 6, (0.02,) * 6),
            )
            plan = self.adapter._move
            assert isinstance(plan, JointPositionTrajectory)
            self.assertEqual(plan.sample(0), self.positions)
            for step in range(1, 431):
                elapsed = step / 100
                clock.return_value = 100 + elapsed
                self.positions = plan.sample(elapsed)
                self.adapter.tick()
        self.assertEqual(self.adapter.state, 'HOLDING')
        self.assertEqual(self.adapter._target, (0.02,) * 6)
        self.assertEqual(self.adapter.initial_positions, (0.0,) * 6)
        self.assertTrue(all(c.args[1] in ('initialize', 'position')
                            for c in self.send.call_args_list))

    def test_full_trajectory_rejects_initial_mismatch_before_sending(self) -> None:
        """从当前真实反馈重新预检，起点已变化时不执行旧轨迹."""
        self.acquire()
        self.send.reset_mock()
        self.positions = (0.01,) * 6
        with self.assertRaisesRegex(ValueError, '初始角度'):
            self.adapter.follow_trajectory(
                (0, 2_000_000_000), ((0.0,) * 6, (0.02,) * 6),
            )
        self.assertEqual(self.adapter.state, 'HOLDING')
        self.send.assert_not_called()

    def test_full_trajectory_halt_prevents_remaining_points(self) -> None:
        """执行中取消整条轨迹，后续tick不再恢复或补发剩余点."""
        self.acquire()
        self.adapter.follow_trajectory(
            (0, 2_000_000_000, 4_000_000_000),
            ((0.0,) * 6, (0.01,) * 6, (0.02,) * 6),
        )
        self.adapter.tick()
        self.adapter.stop()
        calls = self.send.call_count
        self.adapter.tick()
        self.assertEqual(self.send.call_count, calls)
        self.assertIsNone(self.adapter._move)
        with self.assertRaises(RuntimeError):
            self.adapter.follow_trajectory(
                (0, 2_000_000_000), ((0.0,) * 6, (0.02,) * 6),
            )

    def test_armed_trajectory_holds_until_shared_time(self) -> None:
        """实际适配器准备后先保持位置，使用共同起点而非各自启动时刻."""
        self.acquire()
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.0
            self.adapter._last_tick = 100.0
            gate = PairedTrajectoryStart()
            self.adapter.arm_trajectory((0, 2_000_000_000),
                                        ((0.0,) * 6, (0.01,) * 6), 'left', gate)
            gate.ready('right')
            start = gate.start_time()
            assert start is not None
            clock.return_value = 100.01
            self.adapter.tick()
            self.assertEqual(self.adapter.state, 'ARMED')
            self.assertEqual(self.adapter._target, (0.0,) * 6)
            self.adapter._last_tick = start - 0.005
            clock.return_value = start + 0.005
            self.adapter.tick()
            self.assertEqual(self.adapter.state, 'RUNNING')
            self.assertEqual(self.adapter.trajectory_start_at, start)
            self.assertEqual(self.adapter.trajectory_observed_start_at, start + 0.005)
            assert self.adapter._target is not None
            self.assertGreater(self.adapter._target[0], 0.0)

    def test_armed_peer_cancellation_prevents_position_messages(self) -> None:
        """另一臂取消门控时，本臂下一周期先停止，不能继续发位置."""
        self.acquire()
        gate = PairedTrajectoryStart()
        self.adapter.arm_trajectory((0, 2_000_000_000),
                                    ((0.0,) * 6, (0.01,) * 6), 'left', gate)
        gate.cancel()
        self.send.reset_mock()
        with self.assertRaisesRegex(RuntimeError, '双臂轨迹已取消'):
            self.adapter.tick()
        self.assertEqual(self.adapter.state, 'FAULT')
        self.assertEqual([c.args[1] for c in self.send.call_args_list], ['brake'])

    def test_armed_late_start_is_cancelled(self) -> None:
        """软件启动迟到超过工程门槛时取消，不追赶发点."""
        self.acquire()
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.0
            gate = PairedTrajectoryStart()
            self.adapter.arm_trajectory((0, 2_000_000_000),
                                        ((0.0,) * 6, (0.01,) * 6), 'left', gate)
            gate.ready('right')
            start = gate.start_time()
            assert start is not None
            self.adapter._last_tick = start
            clock.return_value = start + 0.025
            with self.assertRaisesRegex(RuntimeError, '启动迟到'):
                self.adapter.tick()
            self.assertTrue(gate.cancelled)
            self.assertEqual(self.adapter.state, 'FAULT')

    def test_armed_drift_cancels_shared_gate(self) -> None:
        """等待启动期间实际角度变化不能继续使用已准备的旧起点."""
        self.acquire()
        gate = PairedTrajectoryStart()
        self.adapter.arm_trajectory((0, 2_000_000_000),
                                    ((0.0,) * 6, (0.01,) * 6), 'left', gate)
        self.positions = (0.006,) * 6
        with self.assertRaisesRegex(RuntimeError, '实际位置或速度'):
            self.adapter.tick()
        self.assertTrue(gate.cancelled)

    def test_full_cycle_holds_recorded_initial_without_extra_return(self) -> None:
        """归位是完整轨迹的最后一段，按实际初值结束并保持位置模式."""
        self.positions = (0.005,) * 6
        self.acquire()
        with patch('archer_hardware.position_adapter.time.monotonic') as clock:
            clock.return_value = 100.0
            self.adapter._last_tick = 100.0
            self.adapter.follow_trajectory(
                (0, 2_000_000_000, 4_000_000_000),
                ((0.005,) * 6, (0.025,) * 6, (0.005,) * 6),
            )
            plan = self.adapter._move
            assert isinstance(plan, JointPositionTrajectory)
            for step in range(1, 431):
                elapsed = step / 100.
                clock.return_value = 100.0 + elapsed
                self.positions = plan.sample(elapsed)
                self.adapter.tick()
        self.assertEqual(self.adapter.state, 'HOLDING')
        self.assertEqual(self.adapter._target, (0.005,) * 6)
        self.assertEqual(self.adapter.initial_positions, (0.005,) * 6)
        self.assertTrue(all(call.args[1] in ('initialize', 'position')
                            for call in self.send.call_args_list))

    def test_native_boundary_rechecks_state_and_feedback(self) -> None:
        """原生周期发送前重新核对，而非信任此前发布缓存时的状态."""
        self.acquire()
        self.adapter._feedback = self.frame()
        self.adapter._check_dispatch(object(), 'position')
        for field, value in (
            ('_session_holder', 8), ('_api_control_initialized', False),
            ('_calibrated', False), ('_error_codes', [3] * 6),
        ):
            original = getattr(self.arm, field)
            setattr(self.arm, field, value)
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                self.adapter._check_dispatch(object(), 'position')
            setattr(self.arm, field, original)
        self.adapter._feedback = None
        with self.assertRaisesRegex(RuntimeError, '新鲜'):
            self.adapter._check_dispatch(object(), 'position')
        self.adapter.state = 'FAULT'
        self.arm._calibrated = False
        self.arm._api_control_initialized = False
        self.arm._error_codes = [3] * 6
        self.adapter._check_dispatch(object(), 'brake')
        self.arm._session_holder = 8
        with self.assertRaises(RuntimeError):
            self.adapter._check_dispatch(object(), 'brake')

    def test_native_boundary_calibration_and_empty_only(self) -> None:
        """准备许可、控制权、空心跳及双臂取消仍在原生发送边界生效."""
        self.adapter.state = 'REQUESTING'
        self.arm._session_holder = 7
        self.arm._calibrated = False
        self.adapter._feedback = self.frame()
        for allowed, latched in ((False, False), (False, True), (True, False)):
            self.adapter.allow_api_calibration = allowed
            self.adapter._calibration_sent = latched
            with self.assertRaisesRegex(RuntimeError, '不允许'):
                self.adapter._check_dispatch(object(), 'calibrate')
        self.adapter.allow_api_calibration = True
        self.adapter._calibration_sent = True
        self.adapter._check_dispatch(object(), 'calibrate')
        self.arm._session_holder = 0
        with self.assertRaises(RuntimeError):
            self.adapter._check_dispatch(object(), 'calibrate')
        self.adapter._check_dispatch(object(), 'initialize')
        self.arm._session_holder = 7
        self.adapter._check_dispatch(self.arm._construct_empty_message(), 'heartbeat')
        with self.assertRaisesRegex(RuntimeError, '空电机目标'):
            self.adapter._check_dispatch(self.arm._construct_init_message(), 'heartbeat')
        gate = PairedTrajectoryStart()
        self.adapter._trajectory_gate = gate
        gate.cancel()
        with self.assertRaisesRegex(RuntimeError, '双臂轨迹已取消'):
            self.adapter._check_dispatch(object(), 'position')

    def test_invalid_frequency_and_limit_isolation(self) -> None:
        """频率不接受布尔值，左右实例不共享初始角度或轨迹."""
        with self.assertRaises(ValueError):
            HexPositionAdapter('ws://127.0.0.1:8439', fixture_limits(), control_hz=True)
        other = HexPositionAdapter('ws://127.0.0.1:9439', fixture_limits())
        self.acquire()
        self.assertIsNone(other.initial_positions)
        self.assertIsNone(other._move)

    def test_connect_filters_unrequested_vendor_control(self) -> None:
        """替换网络构造后，验证连接不会放行厂商自动控制消息."""
        sdk = import_module('hex_device.hex_device_api').HexDeviceApi
        message_type = import_module('hex_device.generated.public_api_down_pb2').APIDown
        adapter = HexPositionAdapter('ws://127.0.0.1:9439', fixture_limits())
        sent = []

        async def capture(api, message):
            sent.append(message)

        with patch.object(sdk, '__init__', return_value=None):
            adapter.connect()
        from threading import Event
        adapter._api._HexDeviceApi__is_closing = Event()
        adapter._api._internal_device_list = []
        adapter._api._device_id_map = {}
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
