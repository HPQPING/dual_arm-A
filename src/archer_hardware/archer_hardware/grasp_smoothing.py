"""按协同阶段拟合有界五次样条，抑制仿真采样微波动，保留事件停顿与双臂共同时间。"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import numpy as np

from .circle_import import prepare_samples
from .position_plan import PositionLimits
from .grasp_events import SETTLE_PAIRS, settle_blend_info


MOTION_STAGES = frozenset((
    'APPROACH', 'ALIGN_JOINT6', 'INSERT', 'LIFT', 'MOVE', 'RETURN', 'LOWER',
    'RETREAT',
))
FIT_TOLERANCE_RAD = .0005
APPROACH_FIT_TOLERANCE_RAD = .004
APPROACH_JOINT3_SPEED_RAD_S = .12


def smooth_grasp_samples(
    samples: dict[str, Any], limits: dict[str, PositionLimits], *,
    execution_hz: int, margin: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """保留采样数量与边界角度；只替换同阶段运动内部，解析复核q/v/a/jerk。"""
    from scipy.interpolate import BSpline, make_interp_spline
    from scipy.optimize import lsq_linear

    from archer_control.continuous_trajectory import (
        ContinuousPair, RecordedTimeMap, trajectory_jerk_metrics, _path_progress,
    )
    from archer_mujoco.joint_interpolation import coefficients, _extrema

    blend_info = settle_blend_info(samples)
    grouped = deepcopy(samples)
    for side in ('left', 'right'):
        for point in grouped[side]['points']:
            point['source_stage'] = SETTLE_PAIRS.get(
                point['source_stage'], point['source_stage'])
    grouped['reference_events'] = [e for i, e in enumerate(samples['reference_events'])
                                   if i not in blend_info['continuous_reference_event_indices']]
    baseline, report = prepare_samples(
        grouped, limits, smooth=True, execution_hz=execution_hz, margin=margin)
    count = len(baseline['left']['points'])
    source_ns = tuple(report['source_times_ns'])
    labels = report['source_stages']['left']
    if report['source_stages']['right'] != labels:
        raise ValueError('协同样条需要双侧完全相同的阶段')
    stops = tuple(baseline['stop_indices'])
    raw = np.asarray([[p['positions'] for p in samples[s]['points']]
                      for s in ('left', 'right')], dtype=np.float64)
    q = raw.copy()
    v = np.zeros_like(q)
    a = np.zeros_like(q)
    old_ns = np.asarray(report['prepared_times_ns'], dtype=np.int64)
    time_ns = np.zeros(count, dtype=np.int64)
    low, high = [np.asarray([getattr(limits[s], attr) for s in ('left', 'right')])
                 for attr in ('lower', 'upper')]
    blocks = []
    polynomials: dict[str, list[Any]] = {'left': [], 'right': []}
    for first, last in zip(stops, stops[1:]):
        n = last - first + 1
        phase = labels[first]
        source_span_ns = source_ns[last] - source_ns[first]
        selected = n >= 6 and phase in MOTION_STAGES and labels[last] == phase
        controls = 0
        if selected:
            # 同一时间参数和节点用于两臂。端点角度固定；共同五次进度给出零端速度与加速度。
            # 所有控制点在关节范围内，利用B样条凸包性质约束拟合曲线。
            original = np.concatenate((raw[0, first:last + 1], raw[1, first:last + 1]), axis=1)
            length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))]
            if np.any(np.diff(length) <= 0.):
                raise ValueError('运动块内部含重复点，需要显式停车分块')
            spatial = phase != 'APPROACH'
            x = (length / length[-1] if spatial else
                 (np.asarray(source_ns[first:last + 1])-source_ns[first])/source_span_ns)
            for requested in (6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, n):
                if not spatial:
                    requested += 4
                controls = min(requested, n if spatial else n + 4)
                zeros = np.zeros(12)
                boundary = None if spatial else (
                    [(1, zeros), (2, zeros)], [(1, zeros), (2, zeros)])
                knots = (make_interp_spline(x, original, k=5, bc_type=boundary).t
                         if controls == (n if spatial else n + 4) else
                         np.r_[np.zeros(6),
                               x[np.rint(np.linspace(1, n - 2, controls - 6)).astype(int)],
                               np.ones(6)])
                matrix = BSpline.design_matrix(x, knots, 5).toarray()
                c = np.empty((controls, 12), dtype=np.float64)
                end = 1 if spatial else 3
                c[:end], c[-end:] = original[0], original[-1]
                penalty = .1 * np.diff(np.eye(controls), n=2, axis=0)
                # P样条控制点二阶差分惩罚，避免为贴合细小反馈波动产生尖锐导数。
                design_full = np.vstack((matrix, penalty))
                observed = np.vstack((original, np.zeros((controls - 2, 12))))
                rhs = observed - design_full[:, :end] @ c[:end] - design_full[:, -end:] @ c[-end:]
                design = design_full[:, end:-end]
                for joint in range(12):
                    result = lsq_linear(
                        design, rhs[:, joint],
                        bounds=(low.reshape(-1)[joint], high.reshape(-1)[joint]),
                        method='bvls', tol=1e-10, max_iter=500)
                    if not result.success:
                        raise ValueError(f'协同有界样条拟合未收敛：{phase}/{controls}/J{joint + 1}')
                    c[end:-end, joint] = result.x
                spline = BSpline(knots, c, 5)
                fitted = np.asarray(spline(x), dtype=np.float64)
                fitted[[0, -1]] = original[[0, -1]]
                if np.max(abs(fitted - original)) <= (
                    APPROACH_FIT_TOLERANCE_RAD if phase == 'APPROACH' else FIT_TOLERANCE_RAD
                ):
                    dx, dxx = spline(x, 1), spline(x, 2)
                    lo, hi = np.zeros(n), np.ones(n)
                    for _ in range(55):
                        u = (lo + hi) / 2.
                        progress = _path_progress(u, .25)[0]
                        lo, hi = np.where(progress < x, u, lo), np.where(progress >= x, u, hi)
                    u = (lo + hi) / 2.
                    u[0], u[-1] = 0., 1.
                    _, progress_v, progress_a = _path_progress(u, .25)
                    unit_v = dx * progress_v[:, None]
                    unit_a = dxx * progress_v[:, None]**2 + dx * progress_a[:, None]
                    if not spatial:
                        u, unit_v, unit_a = x, dx, dxx
                    unit_v[[0, -1]], unit_a[[0, -1]] = 0., 0.
                    break
            else:
                raise ValueError('协同阶段不能在限定误差内平滑：' + phase)
            # J6独立对齐使用较高的平滑规划速度；夹持阶段降低动态需求。
            if phase == 'ALIGN_JOINT6':
                speed, accel, jerk = .30, .18, .8
            elif phase in ('INSERT', 'LIFT', 'MOVE', 'RETURN', 'LOWER', 'RETREAT'):
                speed, accel, jerk = .10, .08, .4
            else:
                speed, accel, jerk = .25, .18, 8.
            cap_v = np.minimum(speed, [limits[s].max_velocity for s in ('left', 'right')])
            if phase == 'APPROACH':
                # 实机接近阶段J3出现跟踪滞后，先降低该轴动态需求，不改硬件或误差界限。
                cap_v[:, 2] = np.minimum(cap_v[:, 2], APPROACH_JOINT3_SPEED_RAD_S)
            cap_a = np.minimum(accel, [limits[s].max_acceleration for s in ('left', 'right')])
            duration = source_span_ns / 1e9
            for arm in range(2):
                section = slice(arm * 6, (arm + 1) * 6)
                for i, h in enumerate(np.diff(u)):
                    c_unit = coefficients(
                        fitted[i, section], fitted[i + 1, section], unit_v[i, section],
                        unit_v[i + 1, section], unit_a[i, section], unit_a[i + 1, section],
                        float(h))
                    for order, cap in ((1, cap_v[arm]), (2, cap_a[arm]),
                                       (3, np.full(6, jerk))):
                        peak = np.max(abs(_extrema(c_unit, float(h), order)), axis=1)
                        duration = max(duration, margin * float(np.max(peak / cap))**(1./order))
            duration_ns = max(source_span_ns, math.ceil(duration * 1e9))
            offsets = np.rint(u * duration_ns).astype(np.int64)
            if np.any(np.diff(offsets) <= 0):
                raise ValueError('协同拟合后的纳秒节点重合')
            time_ns[first:last + 1] = time_ns[first] + offsets
            duration = duration_ns / 1e9
            for arm in range(2):
                section = slice(arm * 6, (arm + 1) * 6)
                q[arm, first:last + 1] = fitted[:, section]
                v[arm, first:last + 1] = unit_v[:, section] / duration
                a[arm, first:last + 1] = unit_a[:, section] / duration**2
        else:
            offsets = old_ns[first:last + 1] - old_ns[first]
            time_ns[first:last + 1] = time_ns[first] + offsets
            for arm, side in enumerate(('left', 'right')):
                pts = baseline[side]['points'][first:last + 1]
                v[arm, first:last + 1] = [p['velocities'] for p in pts]
                a[arm, first:last + 1] = [p['accelerations'] for p in pts]
        for arm, side in enumerate(('left', 'right')):
            for i in range(first, last):
                h = int(time_ns[i + 1] - time_ns[i]) / 1e9
                c_final = coefficients(q[arm, i], q[arm, i + 1], v[arm, i], v[arm, i + 1],
                                       a[arm, i], a[arm, i + 1], h)
                extent = _extrema(c_final, h, 0)
                if (np.any(extent[:, 0] < low[arm] - 1e-12)
                        or np.any(extent[:, 1] > high[arm] + 1e-12)):
                    raise ValueError(side + '协同样条段内角度超限：' + phase)
                for order, cap in ((1, limits[side].max_velocity),
                                   (2, limits[side].max_acceleration)):
                    if np.any(np.max(abs(_extrema(c_final, h, order)), axis=1) > (
                        np.asarray(cap) * (1. + 1e-8)
                    )):
                        raise ValueError(side + '协同样条段内导数超限：' + phase)
                if selected and np.max(abs(_extrema(c_final, h, 3))) > jerk * (1. + 1e-6):
                    raise ValueError(side + '协同样条加加速度超限：' + phase)
                polynomials[side].append(c_final)
        blocks.append({'first': first, 'last': last, 'phase': phase, 'fitted': selected,
                       'spline_controls': controls,
                       'duration_s': int(time_ns[last] - time_ns[first]) / 1e9,
                       **({'planning_velocity_rad_s': speed,
                           'planning_joint3_velocity_rad_s': float(np.max(cap_v[:, 2])),
                           'planning_acceleration_rad_s2': accel,
                           'planning_jerk_rad_s3': jerk} if selected else {})})
    times = time_ns / 1e9
    plan = ContinuousPair(times, {s: q[i] for i, s in enumerate(('left', 'right'))},
                          {s: v[i] for i, s in enumerate(('left', 'right'))},
                          {s: a[i] for i, s in enumerate(('left', 'right'))},
                          {s: np.asarray(polynomials[s]) for s in polynomials}, stops)
    changes = {s: float(np.max(abs(q[i] - raw[i]))) for i, s in enumerate(('left', 'right'))}
    report.update({
        'position_filter': 'bounded_spatial_spline',
        'position_smoothing_tolerance_rad': APPROACH_FIT_TOLERANCE_RAD,
        'max_sample_position_change_rad': changes,
        'positions_preserved': False, 'smoothing_policy': 'bounded_phase_quintic_bspline',
        'cooperative_smoothing_blocks': blocks,
        'duration_s': float(times[-1]), 'prepared_times_ns': time_ns.tolist(),
        'effective_time_scale': float(time_ns[-1] / source_ns[-1]),
        'jerk_metrics': trajectory_jerk_metrics(plan),
        'baseline_before_phase_smoothing_duration_s': baseline['left']['points'][-1][
            'time_from_start']['sec'] + baseline['left']['points'][-1]['time_from_start'][
                'nanosec'] / 1e9,
    })
    for field in ('smoothing_time_scale', 'smoothing_max_segment_change_rad',
                  'smoothing_jerk_jump_after_rad_s3', 'max_segment_shape_change_rad', 'arms'):
        report.pop(field, None)
    report['source_stages'] = {
        s: [p['source_stage'] for p in samples[s]['points']] for s in ('left', 'right')}
    report['cooperative_settle_blend'] = blend_info
    mapping = RecordedTimeMap.from_report(
        report, source_ns, tuple(int(t) for t in time_ns), held_intervals=tuple(
            bool(np.all(raw[:, i] == raw[:, i + 1])) for i in range(count - 1)))
    report['reference_events'] = [
        {**event, 'prepared_time_ns': mapping.map_ns(
            event['source_time_ns'] - samples['source_time_origin_ns'])}
        for event in samples['reference_events']]
    return plan.payload(), report
