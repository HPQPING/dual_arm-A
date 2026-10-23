"""仅离线压缩画圆接近归位内部时轴，保留圆弧与两个成功衔接窗口."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from archer_control.continuous_trajectory import ContinuousPair, circle_blend_windows
from archer_mujoco.joint_interpolation import coefficients, _extrema


def optimize_transit_timing(
    plan: ContinuousPair, stages: dict[str, list[str]], lower: ArrayLike, upper: ArrayLike,
    max_velocity: ArrayLike, max_acceleration: ArrayLike, *, max_jerk: float,
    source: Mapping[str, ArrayLike],
) -> tuple[ContinuousPair, dict[str, Any]]:
    """共享时间变换两端导数回到原值，逐段解析核对V/A/J和路径变化后选择更快方案."""
    try:
        from scipy.interpolate import BSpline
        from scipy.linalg import null_space
    except (ImportError, ValueError) as error:
        raise ValueError('局部时间优化需要兼容的SciPy转换环境') from error

    bounds = [np.asarray(x, dtype=float) for x in
              (lower, upper, max_velocity, max_acceleration)]
    if (any(x.shape != (2, 6) or not np.all(np.isfinite(x)) for x in bounds)
            or np.any(bounds[0] >= bounds[1]) or np.any(bounds[2] <= 0.)
            or np.any(bounds[3] <= 0.) or type(max_jerk) not in (int, float)
            or not np.isfinite(max_jerk) or max_jerk <= 0.
            or plan.stops != (0, len(plan.times) - 1)):
        raise ValueError('局部时间优化需要有效限制及已完成融合的画圆路径')
    raw = {s: np.asarray(x, dtype=float) for s, x in source.items()}
    if (set(raw) != {'left', 'right'}
            or any(x.shape != (len(plan.times), 6) or not np.all(np.isfinite(x))
                   for x in raw.values())):
        raise ValueError('局部时间优化需要完整双臂原始节点以核对拟合误差')
    required = {0, len(plan.times) - 1}
    for labels in stages.values():
        required.update(k for i in range(1, len(labels)) if labels[i] != labels[i - 1]
                        for k in (i - 1, i))
    windows = circle_blend_windows(plan.times, stages, tuple(sorted(required)))
    spans = ((0, windows[0]['first'], 'APPROACH'),
             (windows[1]['last'], len(plan.times) - 1, 'RETURN_HOME'))
    ns = np.rint(plan.times * 1e9).astype(np.int64)
    gaps = np.diff(ns).copy()
    velocities = {s: x.copy() for s, x in plan.velocities.items()}
    accelerations = {s: x.copy() for s, x in plan.accelerations.items()}
    polynomials = {s: x.copy() for s, x in plan.polynomials.items()}
    positions = {s: x.copy() for s, x in plan.positions.items()}
    accepted = []
    rejected = []
    for first, last, name in spans:
        if last - first < 32:
            continue
        if any(all(np.array_equal(raw[s][k], raw[s][k + 1])
                   for s in ('left', 'right')) for k in range(first, last)):
            raise ValueError('接近归位含实际保持区间，不压缩其时间')
        duration = (ns[last] - ns[first]) / 1e9
        u = (ns[first:last + 1] - ns[first]) / (ns[last] - ns[first])
        old_gaps = np.diff(ns[first:last + 1]) / 1e9
        # dt_new/dt_old = 1 - strength * 64u³(1-u)³.
        # 端点处rho=1、rho'=rho''=0，保留q/v/a/j边界而非改变成功融合窗口.
        for strength in (.35, .3, .25, .2, .15, .1, .075, .05, .025):
            integral = 16. * u**4 - 38.4 * u**5 + 32. * u**6 - (64. / 7.) * u**7
            local_ns = np.rint(duration * (u - strength * integral) * 1e9).astype(np.int64)
            local_gaps = np.diff(local_ns) / 1e9
            if np.any(local_gaps <= 0.):
                continue
            candidate = {}
            fields = {}
            changes, peaks, jumps = {}, {}, {}
            valid = True
            for axis, side in enumerate(('left', 'right')):
                q = plan.positions[side][first:last + 1]
                v0 = plan.velocities[side][first:last + 1]
                a0 = plan.accelerations[side][first:last + 1]
                old = plan.polynomials[side][first:last]
                j0 = 6. * old[0, :, 3] / old_gaps[0]**3
                j1 = (6. * old[-1, :, 3] + 24. * old[-1, :, 4]
                      + 60. * old[-1, :, 5]) / old_gaps[-1]**3
                x = local_ns / local_ns[-1]
                span = local_ns[-1] / 1e9
                original_q = q.copy()
                boundary = np.array([q[0], v0[0] * span, a0[0] * span**2, j0 * span**3,
                                     q[-1], v0[-1] * span, a0[-1] * span**2, j1 * span**3])
                for controls in (32, 48, 64, 96, 128, 192, 256, 384, 512):
                    if controls > len(q):
                        continue
                    ids = np.unique(np.r_[
                        np.linspace(1, len(q) - 2, controls - 6).astype(int),
                        np.searchsorted(x, np.linspace(0., 1., controls - 4)[1:-1])])
                    ids = ids[(ids > 0) & (ids < len(q) - 1)]
                    controls = len(ids) + 6
                    knots = np.r_[np.zeros(6), x[ids], np.ones(6)]
                    basis = BSpline(knots, np.eye(controls), 5)
                    constraint = np.array([basis(t, order) for t in (0., 1.)
                                           for order in range(4)])
                    norm = np.linalg.norm(constraint, axis=1)
                    fixed = constraint / norm[:, None]
                    c0 = np.linalg.lstsq(fixed, boundary / norm[:, None], rcond=None)[0]
                    free = null_space(fixed)
                    matrix = basis(x)
                    c = c0 + free @ np.linalg.lstsq(
                        matrix @ free, original_q - matrix @ c0, rcond=None)[0]
                    spline = BSpline(knots, c, 5)
                    fitted = spline(x)
                    if np.max(abs(fitted - original_q)) <= .00002:
                        break
                else:
                    valid = False
                    break
                q = fitted
                q[[0, -1]] = original_q[[0, -1]]
                v, a = spline(x, 1) / span, spline(x, 2) / span**2
                v[[0, -1]], a[[0, -1]] = v0[[0, -1]], a0[[0, -1]]
                pc = np.stack([coefficients(q[k], q[k + 1], v[k], v[k + 1], a[k], a[k + 1],
                                            float(h)) for k, h in enumerate(local_gaps)])
                ext = [np.stack([_extrema(c, float(h), order)
                                 for c, h in zip(pc, local_gaps)]) for order in (0, 1, 2, 3)]
                change = max(float(np.max(abs(_extrema(c - b, 1., 0))))
                             for c, b in zip(pc, plan.polynomials[side][first:last]))
                jerk_start = 6. * pc[:, :, 3] / local_gaps[:, None]**3
                jerk_end = (6. * pc[:, :, 3] + 24. * pc[:, :, 4] + 60. * pc[:, :, 5]
                            ) / local_gaps[:, None]**3
                old = plan.polynomials[side][first:last]
                old_start = 6. * old[:, :, 3] / old_gaps[:, None]**3
                old_end = (6. * old[:, :, 3] + 24. * old[:, :, 4] + 60. * old[:, :, 5]
                           ) / old_gaps[:, None]**3
                old_jump = float(np.max(abs(old_start[1:] - old_end[:-1])))
                jump = float(np.max(abs(jerk_start[1:] - jerk_end[:-1])))
                edge = max(float(np.max(abs(jerk_start[0] - old_start[0]))),
                           float(np.max(abs(jerk_end[-1] - old_end[-1]))))
                valid &= (change <= .00005 and edge <= 1e-5
                          and np.max(abs(q - raw[side][first:last + 1])) <= .0005
                          and jump <= max(1e-5, old_jump * 1.1)
                          and np.all(ext[0] >= bounds[0][axis, :, None] - 1e-12)
                          and np.all(ext[0] <= bounds[1][axis, :, None] + 1e-12)
                          and np.all(abs(ext[1]) <= bounds[2][axis, :, None])
                          and np.all(abs(ext[2]) <= bounds[3][axis, :, None])
                          and np.all(abs(ext[3]) <= max_jerk))
                candidate[side], fields[side] = pc, (q, v, a)
                changes[side] = change
                peaks[side] = [float(np.max(abs(x))) for x in ext[1:]]
                jumps[side] = dict(before=old_jump, after=jump, edge_change=edge)
            if not valid:
                rejected.append(dict(stage=name, strength=strength, peaks=peaks, jumps=jumps,
                                     changes=changes))
                continue
            gaps[first:last] = np.diff(local_ns)
            for side in ('left', 'right'):
                q_next, v_next, a_next = fields[side]
                positions[side][first:last + 1] = q_next
                velocities[side][first:last + 1] = v_next
                accelerations[side][first:last + 1] = a_next
                polynomials[side][first:last] = candidate[side]
            accepted.append(dict(stage=name, first=first, last=last, strength=strength,
                                 before_s=duration, after_s=local_ns[-1] / 1e9,
                                 max_segment_change_rad=changes, peaks=peaks, jerk_jumps=jumps))
            break
    if not accepted:
        raise ValueError('没有同时满足原路径和导数约束的更快接近归位方案')
    times = np.r_[0, np.cumsum(gaps)] / 1e9
    final_windows = circle_blend_windows(times, stages, tuple(sorted(required)))
    if final_windows != windows:
        raise ValueError('时间变换改变了成功衔接窗口定位，保留原候选')
    for array in (times, *positions.values(), *velocities.values(),
                  *accelerations.values(), *polynomials.values()):
        array.setflags(write=False)
    result = ContinuousPair(times, positions, velocities, accelerations, polynomials, plan.stops)
    return result, {
        'policy': 'bounded_transit_time_warp_v1', 'blocks': accepted,
        'rejected_candidates': len(rejected),
        'arc_and_blend_segments_preserved': True, 'node_change_limit_rad': .00002,
        'velocity_limit_rad_s': bounds[2].tolist(),
        'acceleration_limit_rad_s2': bounds[3].tolist(),
        'jerk_limit_rad_s3': max_jerk,
    }
