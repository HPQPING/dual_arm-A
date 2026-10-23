"""离线优化画圆过渡时轴和同步归位，保留画圆及衔接窗口，不连接设备."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from archer_control.continuous_trajectory import ContinuousPair, circle_blend_windows
from archer_mujoco.joint_interpolation import coefficients, _extrema


def _jerk_bounds(
    polynomials: NDArray[np.float64], gaps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """批量求五次段的二次jerk极值，精确检查端点和区间内顶点。"""
    h3 = gaps[:, None] ** 3
    a = 6. * polynomials[:, :, 3] / h3
    b = 24. * polynomials[:, :, 4] / h3
    c = 60. * polynomials[:, :, 5] / h3
    end = a + b + c
    vertex = np.divide(-b, 2. * c, out=np.zeros_like(a), where=c != 0.)
    inside = (vertex > 0.) & (vertex < 1.)
    vertex = np.clip(vertex, 0., 1.)
    middle = a + vertex * (b + vertex * c)
    low = np.minimum(a, end)
    high = np.maximum(a, end)
    return np.stack((np.where(inside, np.minimum(low, middle), low),
                     np.where(inside, np.maximum(high, middle), high)), axis=-1)


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
                # jerk为二次多项式，可先批量精确检查；失败候选不继续昂贵实根求解。
                jerk_extent = _jerk_bounds(pc, local_gaps)
                if np.any(abs(jerk_extent) > max_jerk):
                    valid = False
                    break
                ext = [np.stack([_extrema(c, float(h), order)
                                 for c, h in zip(pc, local_gaps)]) for order in (0, 1, 2)]
                ext.append(jerk_extent)
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


RETURN_PROFILE_TOLERANCE_RAD = .004


def _septic_jet(first: list[NDArray[np.float64]], last: list[NDArray[np.float64]],
                duration: float) -> NDArray[np.float64]:
    """在归一化时间上匹配q/v/a/j，生成局部七次修正的八个系数."""
    matrix = np.array([[1, 1, 1, 1], [4, 5, 6, 7],
                       [12, 20, 30, 42], [24, 60, 120, 210]], dtype=float)
    c = np.zeros((6, 8))
    c[:, :4] = np.array([first[0], first[1] * duration,
                         first[2] * duration**2 / 2., first[3] * duration**3 / 6.]).T
    rhs = np.array([last[0] - np.sum(c[:, :4], axis=1),
                    last[1] * duration - c[:, 1] - 2*c[:, 2] - 3*c[:, 3],
                    last[2] * duration**2 - 2*c[:, 2] - 6*c[:, 3],
                    last[3] * duration**3 - 6*c[:, 3]])
    c[:, 4:] = np.linalg.solve(matrix, rhs).T
    return c


def _polynomial_values(c: NDArray[np.float64], unit: NDArray[np.float64], duration: float,
                       order: int) -> NDArray[np.float64]:
    """批量求多项式物理时间导数，供离线构造节点使用."""
    dc = np.array([c[:, k] * math.factorial(k) / math.factorial(k - order)
                   for k in range(order, c.shape[1])])
    return np.polynomial.polynomial.polyval(unit, dc).T / duration**order


def replan_circle_return(
    plan: ContinuousPair, stages: dict[str, list[str]], lower: ArrayLike, upper: ArrayLike,
    max_velocity: ArrayLike, max_acceleration: ArrayLike, *, max_jerk: float,
    source: Mapping[str, ArrayLike],
) -> tuple[ContinuousPair, dict[str, Any]]:
    """只在画圆出口融合后生成共享五次归位及局部C3修正，显式核对路径变化."""
    sides = ('left', 'right')
    bounds = [np.asarray(x, dtype=float) for x in
              (lower, upper, max_velocity, max_acceleration)]
    raw = {side: np.asarray(q, dtype=float) for side, q in source.items()}
    if (any(x.shape != (2, 6) or not np.all(np.isfinite(x)) for x in bounds)
            or np.any(bounds[0] >= bounds[1]) or np.any(bounds[2] <= 0.)
            or np.any(bounds[3] <= 0.) or type(max_jerk) not in (int, float)
            or not np.isfinite(max_jerk) or not 0. < max_jerk <= .8
            or plan.stops != (0, len(plan.times) - 1) or set(raw) != set(sides)
            or any(q.shape != (len(plan.times), 6) or not np.all(np.isfinite(q))
                   for q in raw.values())):
        raise ValueError('同步归位需要完整画圆、有效双臂限制和原始节点')
    required = {0, len(plan.times) - 1}
    for labels in stages.values():
        required.update(k for i in range(1, len(labels)) if labels[i] != labels[i-1]
                        for k in (i-1, i))
    windows = circle_blend_windows(plan.times, stages, tuple(sorted(required)))
    first, last = windows[1]['last'], len(plan.times) - 1
    moving_end = any(
        np.any(plan.velocities[s][-1] != 0.) or np.any(plan.accelerations[s][-1] != 0.)
        for s in sides)
    if (last-first < 32 or moving_end
            or any(all(np.array_equal(raw[s][i], raw[s][i+1]) for s in sides)
                   for i in range(first, last))):
        raise ValueError('归位节点不足、终点非零导数或存在实际保持，禁止重新规划')
    old_ns = np.rint(plan.times * 1e9).astype(np.int64)
    before = (old_ns[last] - old_ns[first]) / 1e9
    start = np.concatenate([plan.positions[s][first] for s in sides])
    delta = np.concatenate([plan.positions[s][-1] - plan.positions[s][first] for s in sides])
    if float(delta @ delta) <= 1e-12:
        raise ValueError('归位已无有效位移，不生成更快曲线')
    path = np.concatenate([plan.positions[s][first:] for s in sides], axis=1)
    phase = np.clip((path - start) @ delta / (delta @ delta), 0., 1.)
    phase[[0, -1]] = (0., 1.)
    left, right = np.zeros_like(phase), np.ones_like(phase)
    for _ in range(48):
        mid = (left + right) / 2.
        progress = mid**3 * (10. - 15.*mid + 6.*mid**2)
        left = np.where(progress < phase, mid, left)
        right = np.where(progress >= phase, mid, right)
    unit = (left + right) / 2.
    count = len(unit)
    minimum = max(float(np.max(1.875 * abs(delta.reshape(2, 6)) / bounds[2])),
                  float(np.sqrt(np.max((10./np.sqrt(3.)) * abs(delta.reshape(2, 6))
                                       / bounds[3]))), count * .008) * 1.05
    initial_duration = math.ceil(minimum / .004) * .004
    for duration in (initial_duration + .5*k for k in range(12)):
        if duration >= before - .1 or duration > (count-1)*.05:
            break
        for blend in (2., 1.5, 1.):
            if duration <= 2.*blend:
                continue
            t = np.clip(unit * duration, np.arange(count)*.008,
                        duration - np.arange(count-1, -1, -1)*.008)
            t[[0, -1]] = (0., duration)
            for i in range(1, count):
                t[i] = max(t[i-1] + .008, min(t[i], t[i-1] + .05))
            t[-1] = duration
            for i in range(count-2, -1, -1):
                t[i] = max(t[i], t[i+1] - .05)
            valid = True
            for cut in (blend, duration-blend):
                k = int(np.searchsorted(t, cut))
                eligible = [i for i in (k-1, k) if 0 < i < count-1
                            and cut - t[i-1] >= .004 and t[i+1] - cut >= .004]
                if not eligible:
                    valid = False
                    break
                t[min(eligible, key=lambda i: abs(t[i]-cut))] = cut
            if not valid:
                continue
            ns = np.rint(t * 1e9).astype(np.int64)
            t, gaps = ns / 1e9, np.diff(ns) / 1e9
            proposals, metrics = {}, {}
            for axis, side in enumerate(sides):
                zeros = [np.zeros(6) for _ in range(4)]
                base = np.pad(coefficients(plan.positions[side][first], plan.positions[side][-1],
                                           zeros[0], zeros[1], zeros[2], zeros[3], duration),
                              ((0, 0), (0, 2)))
                old = plan.polynomials[side]
                previous = old[first-1]
                h0 = (old_ns[first]-old_ns[first-1])/1e9
                h1 = (old_ns[-1]-old_ns[-2])/1e9
                # 与保留的出口融合末段接上，而不是复制旧归位第一段的jerk跳变。
                j0 = (6.*previous[:, 3]+24.*previous[:, 4]+60.*previous[:, 5])/h0**3
                j1 = (6.*old[-1, :, 3] + 24.*old[-1, :, 4] + 60.*old[-1, :, 5]) / h1**3
                base_j0 = 6.*base[:, 3] / duration**3
                base_j1 = (6.*base[:, 3] + 24.*base[:, 4] + 60.*base[:, 5]) / duration**3
                begin = _septic_jet([zeros[0], plan.velocities[side][first],
                                     plan.accelerations[side][first], j0-base_j0], zeros, blend)
                finish = _septic_jet(zeros, [zeros[0], zeros[1], zeros[2], j1-base_j1], blend)
                fields = []
                for order in range(3):
                    arr = _polynomial_values(base, t/duration, duration, order)
                    mask = t <= blend
                    arr[mask] += _polynomial_values(begin, t[mask]/blend, blend, order)
                    mask = t >= duration-blend
                    arr[mask] += _polynomial_values(finish, (t[mask]-duration+blend)/blend,
                                                    blend, order)
                    fields.append(arr)
                q, v, a = fields
                for arr, original in zip(fields, (plan.positions[side], plan.velocities[side],
                                                  plan.accelerations[side])):
                    arr[[0, -1]] = original[[first, -1]]
                change = float(np.max(abs(q - raw[side][first:])))
                if change > RETURN_PROFILE_TOLERANCE_RAD:
                    valid = False
                    break
                polys = np.stack([
                    coefficients(q[i], q[i+1], v[i], v[i+1], a[i], a[i+1], float(h))
                    for i, h in enumerate(gaps)])
                jerk = _jerk_bounds(polys, gaps)
                starts = 6.*polys[:, :, 3] / gaps[:, None]**3
                ends = (6.*polys[:, :, 3] + 24.*polys[:, :, 4] + 60.*polys[:, :, 5]
                        ) / gaps[:, None]**3
                jump = float(np.max(abs(starts[1:] - ends[:-1])))
                edge = max(float(np.max(abs(starts[0]-j0))), float(np.max(abs(ends[-1]-j1))))
                if np.max(abs(jerk)) > max_jerk or jump > 1e-5 or edge > 1e-5:
                    valid = False
                    break
                ext = [np.stack([_extrema(c, float(h), o) for c, h in zip(polys, gaps)])
                       for o in (0, 1, 2)]
                if (np.any(ext[0] < bounds[0][axis, :, None] - 1e-12)
                        or np.any(ext[0] > bounds[1][axis, :, None] + 1e-12)
                        or np.any(abs(ext[1]) > bounds[2][axis, :, None])
                        or np.any(abs(ext[2]) > bounds[3][axis, :, None])):
                    valid = False
                    break
                proposals[side] = q, v, a, polys
                metrics[side] = dict(raw_node_change_rad=change,
                                     peak_v_a_j=[float(np.max(abs(e))) for e in ext[1:]]
                                     + [float(np.max(abs(jerk)))], jerk_jump_rad_s3=jump,
                                     edge_jerk_change_rad_s3=edge)
            if not valid:
                continue
            result_fields = [{} for _ in range(4)]
            for side in sides:
                original_fields = (plan.positions, plan.velocities,
                                   plan.accelerations, plan.polynomials)
                entries = zip(proposals[side], original_fields, result_fields)
                for proposed, original, out in entries:
                    out[side] = np.concatenate((original[side][:first], proposed), axis=0)
                    out[side].setflags(write=False)
            times = np.r_[old_ns[:first], old_ns[first] + ns] / 1e9
            if circle_blend_windows(times, stages, tuple(sorted(required))) != windows:
                raise ValueError('新归位改变了原画圆出口融合窗口，保留成功版本')
            times.setflags(write=False)
            return ContinuousPair(times, result_fields[0], result_fields[1], result_fields[2],
                                  result_fields[3], plan.stops), {
                'policy': 'circle_return_joint_profile_v1', 'first': first, 'last': last,
                'before_s': before, 'after_s': duration, 'correction_s': blend,
                'source_node_tolerance_rad': RETURN_PROFILE_TOLERANCE_RAD, 'arms': metrics,
            }
    raise ValueError('没有同时满足路径变化和导数约束的更快归位候选')
