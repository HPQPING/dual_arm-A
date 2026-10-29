"""离线验证搬运J3修正与成功204其他阶段隔离，不建立设备连接。"""

from dataclasses import replace
import math
from types import SimpleNamespace
from typing import Any, cast
import unittest
from unittest.mock import patch

import numpy as np

from archer_hardware.gripper_adapter import CooperativePositionAdapter as NominalAdapter
from archer_hardware.position_plan import (
    ContinuousPositionTrajectory,
    PositionLimits,
    TrajectoryCancelled,
)
from archer_hardware.sdk_cycle import PositionSample
from archer_hardware.cooperative_feedback import ReturnFeedbackAdapter, CarryFeedbackAdapter


def make(kind):
    limits = PositionLimits((-4.0,) * 6, (4.0,) * 6, (0.5,) * 6, (0.3,) * 6)
    adapter = cast(Any, kind('ws://localhost:8439', limits, tracking_tolerance=0.045))
    q = (0.0, 0.44, 1.0, 0.0, 0.0, 0.0)
    adapter.state = 'RUNNING'
    adapter._trajectory_side = 'left'
    adapter._feedback = SimpleNamespace(positions=(0.0, 0.466, 1.03, 0.0, 0.0, 0.0), source_ns=0)
    adapter._move = replace(
        ContinuousPositionTrajectory.prepare(
            (0, 40 * 10**9), (q, q), ((0.0,) * 6,) * 2, ((0.0,) * 6,) * 2, limits, 0.2
        ),
        preserve_path=True,
        joint_error_rad=0.045,
    )
    gate = SimpleNamespace(cancelled=False, elapsed=0.0)
    gate.playback_status = lambda: {
        'reference_elapsed_s': gate.elapsed,
        'end_transport_delivered': False,
    }
    adapter._trajectory_gate = gate
    adapter.gripper.program = {
        'phase_start_ns': {
            'OPEN_FOR_GRASP': 10**9,
            'LIFT': 5 * 10**9,
            'PLACE_SETTLE': 25 * 10**9,
            'CLOSE_BEFORE_HOME': 29 * 10**9,
            'RETURN_HOME': 30 * 10**9,
        }
    }
    return adapter, q


def test_carrying_j3_bias_gets_a_small_slow_correction_without_touching_j2():
    """旧204对J3无修正；新候选只在完整搬运阶段小幅反向跟踪偏置。"""
    adapter, q = make(CarryFeedbackAdapter)
    baseline, _ = make(ReturnFeedbackAdapter)
    ack = lambda: None
    commands = []
    with patch.object(
        NominalAdapter, '_stream_position_sample', return_value=PositionSample(q, ack)
    ):
        for elapsed in np.arange(0.0, 29.0, 0.004):
            for current in (adapter, baseline):
                current._trajectory_gate.elapsed = float(elapsed)
            with patch('time.perf_counter', return_value=100.0 + float(elapsed)):
                sample = adapter._stream_position_sample()
                original = baseline._stream_position_sample()
            assert sample is not None and original is not None
            assert sample.acknowledge is original.acknowledge
            assert (
                sample.target[:2] + sample.target[3:] == original.target[:2] + original.target[3:]
            )
            if elapsed < 5.0 or elapsed >= 25.0:
                assert sample.target == original.target
            commands.append(sample.target[2] - original.target[2])
    assert -0.012 <= min(commands) < -0.003
    assert max(commands) <= 1e-12


def test_seventh_order_window_has_zero_first_three_jets_at_both_edges():
    from archer_hardware.cooperative_feedback import stage_weight

    p = np.polynomial.Polynomial([0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0])
    for order in (1, 2, 3):
        assert p.deriv(order)(0.0) == p.deriv(order)(1.0) == 0.0
    for stamp in (-1.0, 5.0, 25.0, 30.0):
        assert stage_weight(stamp, 5.0, 25.0) == 0.0
    assert stage_weight(7.0, 5.0, 25.0) == stage_weight(23.0, 5.0, 25.0) == 1.0
    assert stage_weight(6.0, 5.0, 25.0) == stage_weight(24.0, 5.0, 25.0) == 0.5


def test_exact_filter_discretization_does_not_depend_on_subdividing_a_constant_input():
    from archer_hardware.cooperative_feedback import CarryJ3Filter

    a, b = CarryJ3Filter(), CarryJ3Filter()
    for _ in range(250):
        a.step(0.03, 0.004, 10.0, 0.0, 100.0)
    for _ in range(25):
        b.step(0.03, 0.04, 10.0, 0.0, 100.0)
    np.testing.assert_allclose([a.z1, a.z2, a.z3], [b.z1, b.z2, b.z3], atol=2e-15)


def test_filter_rejects_three_hz_feedback_fluctuation_instead_of_copying_it():
    from archer_hardware.cooperative_feedback import CarryJ3Filter, GAIN, MAX_OFFSET

    filt = CarryJ3Filter()
    input_values, output_values = [], []
    for stamp in np.arange(0.0, 60.0, 0.004):
        error = 0.03 + 0.002 * math.sin(2.0 * math.pi * 3.0 * stamp)
        value = filt.step(error, 0.004, 100.0, 0.0, 1000.0)
        if stamp >= 40.0:
            input_values.append(-MAX_OFFSET * math.tanh(GAIN * error / MAX_OFFSET))
            output_values.append(value)
    assert np.std(output_values) / np.std(input_values) < 0.001


def test_filter_states_and_output_remain_bounded_for_alternating_errors_and_intervals():
    from archer_hardware.cooperative_feedback import CarryJ3Filter, MAX_OFFSET

    filt = CarryJ3Filter()
    for index in range(10000):
        value = filt.step(
            0.04 * (-1.0) ** (index // 100),
            (0.001, 0.004, 0.02, 0.04)[index % 4],
            100.0,
            0.0,
            1000.0,
        )
        assert max(abs(v) for v in (filt.z1, filt.z2, filt.z3, value)) <= MAX_OFFSET


def test_invalid_filter_inputs_cannot_produce_a_position_command():
    from archer_hardware.cooperative_feedback import CarryJ3Filter, stage_weight

    filt = CarryJ3Filter()
    for dt in (0.0, -0.004, 0.04001, float('nan'), float('inf'), True):
        with np.testing.assert_raises(ValueError):
            filt.step(0.03, dt, 10.0, 0.0, 100.0)
    for error in (float('nan'), float('inf'), True):
        with np.testing.assert_raises(ValueError):
            filt.step(error, 0.004, 10.0, 0.0, 100.0)
    for values in ((float('nan'), 0.0, 100.0), (1.0, 0.0, 3.0), (1.0, 4.0, 0.0)):
        with np.testing.assert_raises(ValueError):
            stage_weight(*values)


def test_parent_cancellation_is_propagated_without_reclassifying_the_first_cause():
    from archer_hardware.cooperative_feedback import CarryFeedbackAdapter

    adapter, _ = make(CarryFeedbackAdapter)
    with patch.object(
        ReturnFeedbackAdapter,
        '_stream_position_sample',
        side_effect=TrajectoryCancelled('另一侧已取消'),
    ):
        with np.testing.assert_raises_regex(TrajectoryCancelled, '另一侧已取消'):
            adapter._stream_position_sample()


def test_cancellation_during_correction_rejects_the_old_sample_without_ack():
    from archer_hardware.cooperative_feedback import CarryFeedbackAdapter, CarryJ3Filter

    adapter, q = make(CarryFeedbackAdapter)
    adapter._trajectory_gate.elapsed = 10.0
    acknowledged = []

    def cancel(*_args):
        adapter._trajectory_gate.cancelled = True
        return -0.001

    with patch.object(
        ReturnFeedbackAdapter,
        '_stream_position_sample',
        return_value=PositionSample(q, lambda: acknowledged.append(True)),
    ), patch.object(CarryJ3Filter, 'step', side_effect=cancel):
        with np.testing.assert_raises(TrajectoryCancelled):
            adapter._stream_position_sample()
    assert acknowledged == []


def test_outside_carrying_returns_identical_sample_even_with_old_filter_state():
    from archer_hardware.cooperative_feedback import CarryFeedbackAdapter, CarryJ3Filter

    adapter, q = make(CarryFeedbackAdapter)
    adapter._carry_filter.z3 = -0.008
    original = PositionSample(q, lambda: None)
    with patch.object(
        ReturnFeedbackAdapter, '_stream_position_sample', return_value=original
    ), patch.object(CarryJ3Filter, 'step', side_effect=AssertionError('外部不得调用')):
        for stamp in (0.0, 4.9, 5.0, 25.0, 26.0, 30.0, 40.0):
            adapter._trajectory_gate.elapsed = stamp
            assert adapter._stream_position_sample() is original


def test_new_shared_gate_discards_previous_trial_correction():
    from archer_hardware.cooperative_feedback import CarryFeedbackAdapter

    adapter, q = make(CarryFeedbackAdapter)
    adapter._carry_filter.z1 = adapter._carry_filter.z2 = adapter._carry_filter.z3 = -0.01
    adapter._carry_gate = object()
    adapter._trajectory_gate.elapsed = 10.0
    with patch.object(
        ReturnFeedbackAdapter,
        '_stream_position_sample',
        return_value=PositionSample(q, lambda: None),
    ):
        sample = adapter._stream_position_sample()
    assert sample is not None and abs(sample.target[2] - q[2]) < 1e-8


def test_original_path_guard_rejects_before_j3_correction_runs():
    from archer_hardware.cooperative_feedback import CarryFeedbackAdapter, CarryJ3Filter

    adapter, _ = make(CarryFeedbackAdapter)
    with patch.object(
        ReturnFeedbackAdapter,
        '_stream_position_sample',
        side_effect=RuntimeError('原路径误差超限'),
    ), patch.object(CarryJ3Filter, 'step', side_effect=AssertionError('不得绕过')):
        with np.testing.assert_raises_regex(RuntimeError, '原路径误差超限'):
            adapter._stream_position_sample()


def test_late_update_keeps_existing_forty_ms_rejection():
    from archer_hardware.cooperative_feedback import CarryFeedbackAdapter

    adapter, q = make(CarryFeedbackAdapter)
    adapter._carry_gate = adapter._trajectory_gate
    adapter._carry_sample_at = 100.0
    adapter._trajectory_gate.elapsed = 10.0
    with patch.object(
        ReturnFeedbackAdapter,
        '_stream_position_sample',
        return_value=PositionSample(q, lambda: None),
    ), patch('time.perf_counter', return_value=100.041):
        with np.testing.assert_raises_regex(ValueError, '40ms'):
            adapter._stream_position_sample()


def load_tests(_loader, _tests, _pattern):
    """Windows仅需标准库unittest，Ubuntu也可按pytest函数发现运行。"""
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in globals().items()
        if name.startswith('test_') and callable(value)
    )
