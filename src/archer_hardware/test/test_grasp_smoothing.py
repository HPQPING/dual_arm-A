"""检查协同样条的共同时间、位置边界、等待时长和故障说明，不连接真实设备。"""

from copy import deepcopy
import time
from types import SimpleNamespace
from typing import Any, cast
from unittest import TestCase
from unittest.mock import patch

import numpy as np

from archer_hardware.grasp_smoothing import smooth_grasp_samples
from archer_hardware.grasp_events import settle_blend_info
from archer_hardware.gripper_adapter import CooperativePositionAdapter
from archer_hardware.position_adapter import HexPositionAdapter
from archer_hardware.position_plan import ContinuousPositionTrajectory, PositionLimits
from archer_mujoco.joint_interpolation import coefficients, _extrema


def samples():
    """包含带微噪声的接近、五秒静止等待和J6对齐。"""
    times = np.r_[np.linspace(0., 2., 21), np.linspace(2.1, 7.1, 51),
                  np.linspace(7.2, 11.2, 41)]
    positions = np.zeros((len(times), 6))
    positions[:21, 0] = np.linspace(0., .2, 21)
    positions[1:20, 1] = 1e-5 * np.sin(np.arange(1, 20) * 2.3)
    positions[21:, 0] = .2
    positions[72:, 5] = np.linspace(0., .4, 41)
    labels = ['APPROACH']*21 + ['GRASP_WAIT']*51 + ['ALIGN_JOINT6']*41
    result = {'version': 3, 'kind': 'recorded_joint_samples', 'interpolation': 'unassigned',
              'source_time_origin_ns': 0, 'reference_events': []}
    for side in ('left', 'right'):
        result[side] = {'joint_names': [f'{side}_joint_{i}' for i in range(1, 7)], 'points': [
            {'positions': q.tolist(), 'source_stage': label,
             'time_from_start': {'sec': int(round(float(t)*1e9))//10**9,
                                 'nanosec': int(round(float(t)*1e9)) % 10**9}}
            for q, t, label in zip(positions, times, labels)]}
    return result


def test_shared_bounded_spline_keeps_boundary_wait_and_input():
    limits = {s: PositionLimits((-3.,)*6, (3.,)*6, (.5,)*6, (.3,)*6)
              for s in ('left', 'right')}
    raw = samples()
    original = deepcopy(raw)
    result, report = smooth_grasp_samples(raw, limits, execution_hz=250, margin=1.05)
    assert raw == original
    assert result['left'] == {**result['right'], 'joint_names': raw['left']['joint_names']}
    assert report['position_smoothing_tolerance_rad'] == .004
    assert max(report['max_sample_position_change_rad'].values()) <= .004
    prepared_ns = report['prepared_times_ns']
    assert prepared_ns[71] - prepared_ns[21] >= 5*10**9
    for side in ('left', 'right'):
        for i in result['stop_indices']:
            assert result[side]['points'][i]['positions'] == raw[side]['points'][i]['positions']
            assert result[side]['points'][i]['velocities'] == [0.]*6
            assert result[side]['points'][i]['accelerations'] == [0.]*6


def test_desynchronized_phases_are_rejected():
    raw = samples()
    raw['right']['points'][10]['source_stage'] = 'WRONG'
    limits = {s: PositionLimits((-3.,)*6, (3.,)*6, (.5,)*6, (.3,)*6)
              for s in ('left', 'right')}
    with TestCase().assertRaisesRegex(ValueError, '完全相同'):
        smooth_grasp_samples(raw, limits, execution_hz=250, margin=1.05)


def test_approach_joint3_has_independent_conservative_speed_limit():
    """解析检查J3主导的接近曲线，不通过放宽0.04rad保护掩盖跟踪问题。"""
    limits = {s: PositionLimits((-3.,)*6, (3.,)*6, (.5,)*6, (.3,)*6)
              for s in ('left', 'right')}
    raw = samples()
    for side in ('left', 'right'):
        for i, point in enumerate(raw[side]['points']):
            if i <= 20:
                u = i / 20.
                point['positions'][0] = .2*(10.*u**3-15.*u**4+6.*u**5)
            point['positions'][2] = 4.*point['positions'][0]
    result, report = smooth_grasp_samples(raw, limits, execution_hz=250, margin=1.05)
    for side in ('left', 'right'):
        points = result[side]['points']
        times = np.asarray(report['prepared_times_ns']) / 1e9
        peak = 0.
        for i in range(20):
            first, last = points[i:i+2]
            h = float(times[i+1]-times[i])
            values = [np.asarray(point[key]) for key in
                      ('positions', 'velocities', 'accelerations') for point in (first, last)]
            c = coefficients(values[0], values[1], values[2], values[3], values[4], values[5], h)
            peak = max(peak, float(np.max(abs(_extrema(c, h, 1)[2]))))
        assert peak <= .12 + 1e-8, f'{side}接近J3峰值{peak:.6f}rad/s超过0.12'


def test_tracking_guard_retains_failure_and_names_the_joint():
    adapter = object.__new__(CooperativePositionAdapter)
    backend = cast(Any, adapter)
    backend._feedback = SimpleNamespace(positions=(0., 0., .041, 0., 0., 0.),
                                        source_ns=time.perf_counter_ns())
    backend._target = (0.,)*6
    backend._move = None
    backend._trajectory_gate = None
    backend.tracking_tolerance = .04
    message = '实际反馈超出完整路径几何误差界限，取消后续位置发送'
    with patch.object(HexPositionAdapter, '_tick', side_effect=RuntimeError(message)):
        with TestCase().assertRaisesRegex(RuntimeError, 'J3.*偏差=0.041000rad.*上限=0.040000rad'):
            adapter._tick()
    with patch.object(HexPositionAdapter, '_tick', side_effect=RuntimeError('反馈超时')):
        with TestCase().assertRaisesRegex(RuntimeError, '^反馈超时$'):
            adapter._tick()


def test_cooperative_trace_uses_reported_targets_is_bounded_and_read_only():
    """通过原反馈观察近五秒数据，区分本地缓存与设备上报，且不发任何指令。"""
    limits = PositionLimits((-3.,)*6, (3.,)*6, (.5,)*6, (.3,)*6)
    adapter = CooperativePositionAdapter('ws://localhost:8439', limits)
    backend = cast(Any, adapter)
    backend.state = 'RUNNING'
    backend._move = ContinuousPositionTrajectory.prepare(
        (0, 10_000_000_000), ((0.,)*6, (.1,)*6), ((0.,)*6,)*2, ((0.,)*6,)*2,
        limits, .2)
    backend._trajectory_gate = SimpleNamespace(
        playback_status=lambda: {'reference_elapsed_s': 2.})
    sender = SimpleNamespace(position_timing=lambda: ((.02,)*6, .004))
    backend._native = sender
    status = {'pos': [.05]*6, 'vel': [-.003]*6, 'eff': [1.]*6}
    backend._api = SimpleNamespace(_project_feedback_frames={25: (None, 0, status)})
    targets = [SimpleNamespace(
        pulse_per_rotation=100000., current_target=SimpleNamespace(
            HasField=lambda name: name == 'position', position=1000)) for _ in range(6)]
    message = SimpleNamespace(robot_type=25, arm_status=SimpleNamespace(motor_status=targets))
    for i in range(400):
        now = 1_000_000_000+i*21_000_000
        backend._api._project_feedback_frames[25] = (None, now, status)
        with patch('archer_hardware.gripper_adapter.time.perf_counter_ns', return_value=now):
            adapter.observe_tracking(message)
    rows = adapter.tracking_trace()
    assert 1 < len(rows) <= 251
    assert rows[-1]['actual_rad'] == tuple(status['pos'])
    assert rows[-1]['velocity_rad_s'] == tuple(status['vel'])
    assert rows[-1]['sent_rad'] == (.02,)*6
    assert abs(rows[-1]['device_rad'][0] - .02*np.pi) < 1e-12
    assert backend._native is sender and backend.state == 'RUNNING'
    targets[0].current_target.HasField = lambda name: False
    with patch('archer_hardware.gripper_adapter.time.perf_counter_ns',
               return_value=12_000_000_000):
        adapter.observe_tracking(message)
    assert adapter.tracking_trace()[-1]['device_rad'][0] is None


def test_only_settle_markers_without_actuation_can_be_continuous():
    raw = samples()
    for side in ('left', 'right'):
        raw[side]['points'][20]['source_stage'] = 'APPROACH_SETTLE'
    raw['reference_events'] = [
        {'stage': 'APPROACH', 'gripper_command': 5.65, 'virtual_attached': 0},
        {'stage': 'APPROACH_SETTLE', 'gripper_command': 5.65, 'virtual_attached': 0},
    ]
    assert settle_blend_info(raw)['boundaries'] == [20]
    assert settle_blend_info(raw)['continuous_reference_event_indices'] == [1]
    raw['reference_events'][1]['gripper_command'] = 5.5
    with TestCase().assertRaisesRegex(ValueError, '不允许合并'):
        settle_blend_info(raw)
