"""验证192仅影响归位及结束交接，归位前与冻结189逐周期一致；不连接硬件。"""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase
from unittest.mock import Mock, patch

from archer_hardware.gripper_adapter import CooperativePositionAdapter as NominalAdapter
from archer_hardware.position_plan import (
    ContinuousPositionTrajectory,
    PositionLimits,
    TrajectoryCancelled,
)
from archer_hardware.sdk_cycle import PositionSample
from archer_hardware.cooperative_feedback import FeedbackCooperativeAdapter as BaselineAdapter
from archer_hardware.cooperative_feedback import ReturnFeedbackAdapter


def make(adapter_type: type[BaselineAdapter] = ReturnFeedbackAdapter):
    limits = PositionLimits((-2.0,) * 6, (2.0,) * 6, (0.5,) * 6, (0.3,) * 6)
    adapter = adapter_type('ws://localhost:9439', limits, tracking_tolerance=0.045)
    value = cast(Any, adapter)
    q = (0.0, 0.44, 0.0, 0.0, 0.0, 0.0)
    value.state = 'RUNNING'
    value._trajectory_side = 'right'
    value._feedback = SimpleNamespace(positions=(0.0, 0.466, 0.0, 0.0, 0.0, 0.0), source_ns=10**9)
    value._move = replace(
        ContinuousPositionTrajectory.prepare(
            (0, 10 * 10**9), (q, q), ((0.0,) * 6,) * 2, ((0.0,) * 6,) * 2, limits, 0.2
        ),
        preserve_path=True,
        joint_error_rad=0.045,
    )
    value._trajectory_gate = SimpleNamespace(
        cancelled=False,
        elapsed=0.0,
        delivered=False,
        playback_status=lambda: {
            'reference_elapsed_s': value._trajectory_gate.elapsed,
            'end_transport_delivered': value._trajectory_gate.delivered,
        },
    )
    value.gripper.program = {
        'phase_start_ns': {
            'OPEN_FOR_GRASP': 10**9,
            'CLOSE_BEFORE_HOME': 3 * 10**9,
            'RETURN_HOME': 4 * 10**9,
        }
    }
    return adapter, q


class ReturnTests(TestCase):
    def test_before_return_delegates_exact_original_sample_and_ack(self):
        adapter, q = make()
        sample = PositionSample(q, Mock())
        with patch.object(BaselineAdapter, '_stream_position_sample', return_value=sample) as old:
            self.assertIs(adapter._stream_position_sample(), sample)
        old.assert_called_once_with()

    def test_both_prefixes_are_numerically_identical_including_closing(self):
        for side in ('left', 'right'):
            old, q = make(BaselineAdapter)
            new, _ = make()
            for adapter in (old, new):
                cast(Any, adapter)._trajectory_side = side
            sample = PositionSample(q, lambda: None)
            with patch.object(NominalAdapter, '_stream_position_sample', return_value=sample):
                for i in range(1000):
                    elapsed = i * 0.004
                    for adapter in (old, new):
                        cast(Any, adapter)._trajectory_gate.elapsed = elapsed
                    with patch(
                        'archer_hardware.cooperative_feedback.time.perf_counter',
                        return_value=100.0 + elapsed,
                    ):
                        first = old._stream_position_sample()
                        second = new._stream_position_sample()
                    assert first is not None and second is not None
                    self.assertEqual(first.target, second.target)
                    self.assertIs(first.acknowledge, second.acknowledge)

    def test_return_feedback_is_enabled_for_both_j2_and_only_these_axes(self):
        for side in ('left', 'right'):
            adapter, q = make()
            value = cast(Any, adapter)
            value._trajectory_side = side
            value._trajectory_gate.elapsed = 6.0
            with patch.object(
                NominalAdapter,
                '_stream_position_sample',
                return_value=PositionSample(q, lambda: None),
            ):
                for i in range(1000):
                    with patch(
                        'archer_hardware.cooperative_feedback.time.perf_counter',
                        return_value=100.0 + i * 0.004,
                    ):
                        result = adapter._stream_position_sample()
                        assert result is not None
                        self.assertEqual(result.target[:1] + result.target[2:], q[:1] + q[2:])
                        self.assertLessEqual(abs(result.target[1] - q[1]), 0.02)
            self.assertLess(result.target[1], q[1] - 0.005)

    def test_return_reuses_existing_filter_state_without_a_reset_jump(self):
        adapter, q = make()
        value = cast(Any, adapter)
        value._trajectory_gate.elapsed = 4.0
        value._feedback_gate = value._trajectory_gate
        value._feedback_sample_at = 99.996
        value._feedback_correction.offset[1] = -0.005
        previous = q[1] - 0.005
        with patch.object(
            NominalAdapter, '_stream_position_sample', return_value=PositionSample(q, lambda: None)
        ), patch('archer_hardware.cooperative_feedback.time.perf_counter', return_value=100.0):
            result = adapter._stream_position_sample()
        assert result is not None
        self.assertLessEqual(abs(result.target[1] - previous), 0.01 * 0.004 + 1e-12)
        self.assertLess(result.target[1], q[1] - 0.004)

    def test_original_path_error_is_still_rejected(self):
        adapter, _ = make()
        cast(Any, adapter)._trajectory_gate.elapsed = 6.0
        with patch.object(
            NominalAdapter, '_stream_position_sample', side_effect=RuntimeError('原0.045几何保护')
        ):
            with self.assertRaisesRegex(RuntimeError, '原0.045几何保护'):
                adapter._stream_position_sample()

    def test_terminal_hold_preserves_last_command_and_rejects_cancellation(self):
        adapter, q = make()
        value = cast(Any, adapter)
        value._trajectory_gate.elapsed = 10.0
        with patch.object(
            NominalAdapter, '_stream_position_sample', return_value=PositionSample(q, lambda: None)
        ), patch('archer_hardware.cooperative_feedback.time.perf_counter', return_value=100.0):
            last = adapter._stream_position_sample()
        assert last is not None
        value.state = 'HOLDING'
        value._trajectory_gate.delivered = True
        with patch.object(NominalAdapter, '_stream_position_sample', return_value=None), patch(
            'archer_hardware.cooperative_feedback.time.perf_counter_ns', return_value=10**9 + 10**6
        ):
            held = adapter._stream_position_sample()
            assert held is not None
            self.assertEqual(held.target, last.target)
            value._trajectory_gate.cancelled = True
            with self.assertRaises(TrajectoryCancelled):
                adapter._stream_position_sample()

    def test_terminal_hold_rejects_stale_feedback(self):
        adapter, q = make()
        value = cast(Any, adapter)
        value._trajectory_gate.elapsed = 10.0
        with patch.object(
            NominalAdapter, '_stream_position_sample', return_value=PositionSample(q, lambda: None)
        ), patch('archer_hardware.cooperative_feedback.time.perf_counter', return_value=100.0):
            adapter._stream_position_sample()
        value.state = 'HOLDING'
        value._trajectory_gate.delivered = True
        with patch.object(NominalAdapter, '_stream_position_sample', return_value=None), patch(
            'archer_hardware.cooperative_feedback.time.perf_counter_ns', return_value=2 * 10**9
        ):
            with self.assertRaisesRegex(RuntimeError, '新鲜反馈'):
                adapter._stream_position_sample()

    def test_new_gate_cannot_reuse_an_old_terminal_command(self):
        adapter, q = make()
        value = cast(Any, adapter)
        value.state = 'HOLDING'
        value._trajectory_gate.elapsed = 10.0
        value._trajectory_gate.delivered = True
        value._return_gate = object()
        value._return_last = (0.0, 0.42, 0.0, 0.0, 0.0, 0.0)
        with patch.object(NominalAdapter, '_stream_position_sample', return_value=None):
            self.assertIsNone(adapter._stream_position_sample())

    def test_cancel_during_return_sampling_never_produces_a_corrected_command(self):
        adapter, q = make()
        value = cast(Any, adapter)
        value._trajectory_gate.elapsed = 6.0

        def cancel(_self):
            value._trajectory_gate.cancelled = True
            return PositionSample(q, lambda: None)

        with patch.object(NominalAdapter, '_stream_position_sample', side_effect=cancel):
            with self.assertRaises(TrajectoryCancelled):
                adapter._stream_position_sample()

    def test_terminal_hold_still_rejects_original_path_error(self):
        adapter, q = make()
        value = cast(Any, adapter)
        value.state = 'HOLDING'
        value._trajectory_gate.elapsed = 10.0
        value._trajectory_gate.delivered = True
        value._return_gate = value._trajectory_gate
        value._return_last = q
        value._feedback.positions = (0.0, 0.50, 0.0, 0.0, 0.0, 0.0)
        with patch(
            'archer_hardware.cooperative_feedback.time.perf_counter_ns', return_value=10**9 + 10**6
        ):
            with self.assertRaisesRegex(RuntimeError, '原路径误差界限'):
                adapter._stream_position_sample()

    def test_near_limit_return_does_not_hard_clip_its_correction(self):
        """归位接近下限时保持位置合法，避免修正硬截断产生大加速度。"""
        adapter, _ = make()
        value = cast(Any, adapter)
        q0, q1 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), (0.0, -1.999, 0.0, 0.0, 0.0, 0.0)
        value._move = replace(
            ContinuousPositionTrajectory.prepare(
                (0, 4 * 10**9, 34 * 10**9),
                (q0, q0, q1),
                ((0.0,) * 6,) * 3,
                ((0.0,) * 6,) * 3,
                adapter.limits,
                2.0,
            ),
            preserve_path=True,
            joint_error_rad=0.045,
        )
        value._feedback_gate = value._trajectory_gate
        value._feedback_correction.offset[1] = -0.013
        value._feedback_sample_at = 103.996
        previous, speed, peak = q0[1] - 0.013, 0.0, 0.0
        for i in range(7501):
            elapsed = 4.0 + i * 0.004
            value._trajectory_gate.elapsed = elapsed
            q = value._move.sample(elapsed)
            value._feedback.positions = (0.0, q[1] + 0.026, 0.0, 0.0, 0.0, 0.0)
            with patch.object(
                NominalAdapter,
                '_stream_position_sample',
                return_value=PositionSample(q, lambda: None),
            ), patch(
                'archer_hardware.cooperative_feedback.time.perf_counter',
                return_value=100.0 + elapsed,
            ):
                result = adapter._stream_position_sample()
            assert result is not None
            self.assertGreaterEqual(result.target[1], -2.0)
            current_speed = (result.target[1] - previous) / 0.004
            if i:
                peak = max(peak, abs(current_speed - speed) / 0.004)
            previous, speed = result.target[1], current_speed
        self.assertLessEqual(peak, 0.3 + 1e-4)

    def test_signed_error_reversal_and_inherited_velocity_remain_bounded(self):
        adapter, q = make()
        value = cast(Any, adapter)
        value._feedback_gate = value._trajectory_gate
        value._feedback_correction.offset[1] = -0.006
        value._feedback_correction.velocity[1] = 0.01
        value._feedback_sample_at = 103.996
        previous, speed, max_speed, max_accel = q[1] - 0.006, 0.01, 0.0, 0.0
        for i in range(2001):
            elapsed = 4.0 + i * 0.004
            value._trajectory_gate.elapsed = elapsed
            sign = 1.0 if i < 500 or i >= 1500 else -1.0
            value._feedback.positions = (0.0, q[1] + sign * 0.044, 0.0, 0.0, 0.0, 0.0)
            with patch.object(
                NominalAdapter,
                '_stream_position_sample',
                return_value=PositionSample(q, lambda: None),
            ), patch(
                'archer_hardware.cooperative_feedback.time.perf_counter',
                return_value=100.0 + elapsed,
            ):
                result = adapter._stream_position_sample()
            assert result is not None
            current = (result.target[1] - previous) / 0.004
            max_speed = max(max_speed, abs(current))
            max_accel = max(max_accel, abs(current - speed) / 0.004)
            previous, speed = result.target[1], current
        self.assertLessEqual(max_speed, 0.01 + 1e-8)
        self.assertLessEqual(max_accel, 0.04 + 1e-4)
