"""离线生成J6调姿与归位分臂共同时轴优化；固定空间节点，不连接机械臂。"""

from __future__ import annotations
from copy import deepcopy
from typing import Any
import numpy as np
from numpy.typing import NDArray
from .position_plan import prepare_continuous_pair
from archer_mujoco.joint_interpolation import coefficients, _extrema


def _constrained_fit(
    design: NDArray[np.float64],
    observed: NDArray[np.float64],
    constraint: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> NDArray[np.float64]:
    """在归一化端点约束的零空间中求同一个实数最小二乘问题。"""
    from scipy.linalg import null_space

    norms = np.linalg.norm(constraint, axis=1)
    fixed = constraint / norms[:, None]
    base = np.linalg.lstsq(fixed, targets / norms[:, None], rcond=None)[0]
    free = null_space(fixed)
    delta = np.linalg.lstsq(design @ free, observed - design @ base, rcond=None)[0]
    return np.asarray(base + free @ delta, dtype=np.float64)


def _metrics(c: NDArray[np.float64], gaps: NDArray[np.float64]) -> dict[str, Any]:
    """对最终五次段求解析V/A/J、连接jerk跳变与积分平方jerk；不只检查拟合器。"""
    peaks = np.asarray(
        [
            np.max(
                abs(np.stack([_extrema(pc, float(h), order) for pc, h in zip(c, gaps)])),
                axis=(0, 2),
            )
            for order in (1, 2, 3)
        ]
    )
    start = 6.0 * c[:, :, 3] / gaps[:, None] ** 3
    end = (6.0 * c[:, :, 3] + 24.0 * c[:, :, 4] + 60.0 * c[:, :, 5]) / gaps[:, None] ** 3
    b = c[:, :, 3:] * np.asarray([6.0, 24.0, 60.0])
    gram = 1.0 / (np.arange(3)[:, None] + np.arange(3)[None, :] + 1.0)
    energy = float(np.sum(np.einsum('nji,ik,njk->nj', b, gram, b) / gaps[:, None] ** 5))
    d0, d1 = (24.0 * c[:, :, 4], 120.0 * c[:, :, 5])
    snap = float(np.sum((d0**2 + d0 * d1 + d1**2 / 3.0) / gaps[:, None] ** 7))
    return {
        'peak_v_a_j': peaks.tolist(),
        'internal_jerk_jump_rad_s3': float(np.max(abs(start[1:] - end[:-1]))),
        'integrated_squared_jerk': energy,
        'integrated_squared_snap': snap,
        'start_jerk': start[0].tolist(),
        'end_jerk': end[-1].tolist(),
    }


class TimeDensity:
    """正C3密度：端部8%七次渐变，中部定密度；解析积分得到共同时轴。"""

    width = 0.08
    step = np.polynomial.Polynomial([0.0, 0.0, 0.0, 0.0, 35.0, -84.0, 70.0, -20.0])
    bump = np.polynomial.Polynomial([0.0, 0.0, 0.0, 0.0, 256.0, -1024.0, 1536.0, -1024.0, 256.0])

    def __init__(
        self,
        factor: float,
        endpoint_density: float = 1.0,
        shoulder_height: float = 0.0,
        endpoint_plateau: float = 0.0,
        transition_width: float = 0.08,
    ) -> None:
        if isinstance(factor, bool) or not np.isfinite(factor) or (not 0.0 < factor <= 1.0):
            raise ValueError('时间密度系数必须在(0,1]')
        self.factor = factor
        if (
            isinstance(endpoint_density, bool)
            or not np.isfinite(endpoint_density)
            or (not 1.0 <= endpoint_density <= 2.0)
        ):
            raise ValueError('端部时间密度必须在[1,2]')
        self.endpoint_density = endpoint_density
        if (
            isinstance(shoulder_height, bool)
            or not np.isfinite(shoulder_height)
            or (not 0.0 <= shoulder_height <= 1.0)
        ):
            raise ValueError('段内时间肩区高度必须在[0,1]')
        self.shoulder_height = shoulder_height
        if (
            isinstance(endpoint_plateau, bool)
            or not np.isfinite(endpoint_plateau)
            or (not 0.0 <= endpoint_plateau <= 0.1)
        ):
            raise ValueError('端部等速时间窗比例必须在[0,.1]')
        self.endpoint_plateau = endpoint_plateau
        if (
            isinstance(transition_width, bool)
            or not np.isfinite(transition_width)
            or (not 0.02 <= transition_width <= 0.2)
        ):
            raise ValueError('时间密度过渡比例必须在[.02,.2]')
        self.width = transition_width

    def __call__(self, unit):
        left, right = (
            np.clip((x - self.endpoint_plateau) / self.width, 0.0, 1.0) for x in (unit, 1.0 - unit)
        )
        return (
            self.factor
            + (self.endpoint_density - self.factor) * (2.0 - self.step(left) - self.step(right))
            + self.shoulder_height * (self.bump(left) + self.bump(right))
        )

    def derivative(self, unit, order=1):
        values = []
        for x, sign in ((unit, -1.0), (1.0 - unit, (-1.0) ** (order + 1))):
            local = x - self.endpoint_plateau
            active = (local > 0.0) & (local < self.width)
            z = np.clip(local / self.width, 0.0, 1.0)
            values.append(sign * self.step.deriv(order)(z) * active / self.width**order)
        shoulder = []
        for x, sign in ((unit, 1.0), (1.0 - unit, (-1.0) ** order)):
            local = x - self.endpoint_plateau
            active = (local > 0.0) & (local < self.width)
            z = np.clip(local / self.width, 0.0, 1.0)
            shoulder.append(sign * self.bump.deriv(order)(z) * active / self.width**order)
        return (self.endpoint_density - self.factor) * (
            values[0] + values[1]
        ) + self.shoulder_height * (shoulder[0] + shoulder[1])

    def integral(self, unit):

        def boundary(x):
            z = np.clip((x - self.endpoint_plateau) / self.width, 0.0, 1.0)
            return np.clip(x, 0.0, self.endpoint_plateau) + self.width * (z - self.step.integ()(z))

        integral_bump = self.bump.integ()
        left, right = (
            np.clip((x - self.endpoint_plateau) / self.width, 0.0, 1.0) for x in (unit, 1.0 - unit)
        )
        return (
            self.factor * unit
            + (self.endpoint_density - self.factor)
            * (boundary(unit) + self.endpoint_plateau + self.width / 2.0 - boundary(1.0 - unit))
            + self.shoulder_height
            * self.width
            * (integral_bump(left) + integral_bump(1.0) - integral_bump(right))
        )


def _jets(c, h, end):
    jerk = (6.0 * c[:, 3] + (24.0 * c[:, 4] + 60.0 * c[:, 5] if end else 0.0)) / h**3
    snap = (24.0 * c[:, 4] + (120.0 * c[:, 5] if end else 0.0)) / h**4
    return np.asarray([jerk, snap], dtype=np.float64)


def refine_c3_derivatives(q, v, a, old, old_gaps, gaps, *, preserve_boundary_segments=False):
    """原位置节点等式约束、原边界v/a/jerk/snap约束，避免只缩放旧导数产生接缝跳变。"""
    from scipy.interpolate import BSpline

    if (
        len(q) < 10
        or np.shape(q) != np.shape(v)
        or np.shape(q) != np.shape(a)
        or (np.shape(q)[1] != 6)
        or (len(gaps) != len(q) - 1)
        or (not np.all(np.isfinite(q)))
        or np.any(np.asarray(gaps) <= 0.0)
    ):
        raise ValueError('C3重建需要有限六轴节点和正时间间隔')
    duration = float(np.sum(gaps))
    x = np.r_[0.0, np.cumsum(gaps)] / duration
    x[-1] = 1.0
    knots = np.r_[np.zeros(6), np.repeat(x[1:-1], 2), np.ones(6)]
    basis = BSpline(knots, np.eye(len(knots) - 6), 5)
    constraints = np.asarray(
        np.vstack([basis(x), *[basis(t, order) for t in (0.0, 1.0) for order in (1, 2, 3, 4)]]),
        dtype=np.float64,
    )
    start, end = (
        _jets(old[0], float(old_gaps[0]), False),
        _jets(old[-1], float(old_gaps[-1]), True),
    )
    targets = np.vstack(
        [
            q - q[0],
            v[0] * duration,
            a[0] * duration**2,
            start[0] * duration**3,
            start[1] * duration**4,
            v[-1] * duration,
            a[-1] * duration**2,
            end[0] * duration**3,
            end[1] * duration**4,
        ]
    )
    if preserve_boundary_segments:
        if not np.array_equal(
            np.rint(np.asarray(gaps)[[0, -1]] * 1000000000.0),
            np.rint(np.asarray(old_gaps)[[0, -1]] * 1000000000.0),
        ):
            raise ValueError('边界保留需要原首末区间时长不变')
        constraints = np.vstack(
            [constraints, basis(x[1], 1), basis(x[1], 2), basis(x[-2], 1), basis(x[-2], 2)]
        )
        targets = np.vstack(
            [targets, v[1] * duration, a[1] * duration**2, v[-2] * duration, a[-2] * duration**2]
        )
    mid, half = ((x[:-1] + x[1:]) / 2.0, np.diff(x) / 2.0)
    nodes, weights = np.polynomial.legendre.leggauss(3)
    points = (mid[:, None] + half[:, None] * nodes).ravel()
    w = np.sqrt((half[:, None] * weights).ravel())[:, None]
    before = _metrics(old, old_gaps)
    design = np.vstack(
        [
            basis(points, 3)
            * w
            / duration**2.5
            / np.sqrt(max(before['integrated_squared_jerk'], 1e-30)),
            basis(points, 4)
            * w
            / duration**3.5
            / np.sqrt(max(before['integrated_squared_snap'], 1e-30)),
        ]
    )
    controls = _constrained_fit(design, np.zeros((len(design), 6)), constraints, targets)
    norms = np.linalg.norm(constraints, axis=1)
    for _ in range(2):
        residual = targets - constraints @ controls
        controls += np.linalg.lstsq(
            constraints / norms[:, None], residual / norms[:, None], rcond=None
        )[0]
    spline = BSpline(knots, controls, 5)
    if np.max(abs(spline(x) - (q - q[0]))) > 1e-10:
        raise ValueError('C3固定节点等式残差过大')
    new_v, new_a = (spline(x, 1) / duration, spline(x, 2) / duration**2)
    new_v[[0, -1]], new_a[[0, -1]] = (v[[0, -1]], a[[0, -1]])
    if preserve_boundary_segments:
        new_v[[1, -2]], new_a[[1, -2]] = (v[[1, -2]], a[[1, -2]])
    if not np.all(np.isfinite(new_v)) or not np.all(np.isfinite(new_a)):
        raise ValueError('C3重建导数不是有限值')
    return (new_v, new_a)


def choose_ranges(report):
    """唯一J6主块、唯一归位主块的前35%时间，不压缩夹爪保持或搬运。"""
    ns = report['prepared_times_ns']
    ranges = []
    for phase in ('ALIGN_JOINT6', 'RETURN_HOME'):
        blocks = [
            b
            for b in report['cooperative_smoothing_blocks']
            if b['phase'] == phase and b['last'] - b['first'] >= 10
        ]
        if len(blocks) != 1:
            raise ValueError('需要唯一主块：' + phase)
        first, last = (blocks[0]['first'], blocks[0]['last'])
        if phase == 'RETURN_HOME':
            last = int(np.searchsorted(ns, ns[first] + 0.35 * (ns[last] - ns[first])))
        ranges.append((phase if phase != 'RETURN_HOME' else 'RETURN_HOME_FRONT', first, last))
    return ranges


def remap(stamp, old, new):
    """分段线性纳秒映射，与原节点关联的阶段/事件保持关联，拒绝外推。"""
    if (
        len(old) != len(new)
        or len(old) < 2
        or np.any(np.diff(old) <= 0)
        or np.any(np.diff(new) <= 0)
        or (not old[0] <= stamp <= old[-1])
    ):
        raise ValueError('时间映射需要同长递增网格及范围内事件')
    i = min(int(np.searchsorted(old, stamp, side='right')) - 1, len(old) - 2)
    return int(new[i]) + round(
        (stamp - int(old[i])) * (int(new[i + 1]) - int(new[i])) / (int(old[i + 1]) - int(old[i]))
    )


def transit(payload, report, limits):
    """各窗口独立搜索；任一无合格曲线即拒绝，不发布部分优化。"""
    result, info = (deepcopy(payload), deepcopy(report))
    evidence = []
    for phase, first, last in choose_ranges(report):
        plans = prepare_continuous_pair(result, limits, 3.5)
        ns = np.asarray(info['prepared_times_ns'], dtype=np.int64)
        if not 0 < first < last < len(ns) - 1 or any(
            (first < i < last for i in payload['stop_indices'])
        ):
            raise ValueError('窗口跨过停止节点')
        for event in info['gripper_event_schedule']['events']:
            for key in ('time_ns', 'end_time_ns'):
                if key in event and ns[first] < event[key] < ns[last]:
                    raise ValueError('窗口包含夹爪动作事件')
        old_gaps = np.diff(ns[first : last + 1]) / 1000000000.0
        duration = (ns[last] - ns[first]) / 1000000000.0
        unit = (ns[first : last + 1] - ns[first]) / (ns[last] - ns[first])
        failures = []
        profiles = [
            (factor, width)
            for factor in (0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)
            for width in ((0.08, 0.12, 0.16, 0.2) if phase == 'ALIGN_JOINT6' else (0.08,))
        ]
        for factor, width in profiles:
            clock = TimeDensity(factor, endpoint_plateau=0.04, transition_width=width)
            local = np.rint(duration * clock.integral(unit) * 1000000000.0).astype(np.int64)
            gaps = np.diff(local) / 1000000000.0
            if np.any(gaps <= 0) or not np.array_equal(
                np.diff(local)[[0, -1]], np.diff(ns[first : last + 1])[[0, -1]]
            ):
                failures.append({'factor': factor, 'reason': 'boundary_grid'})
                continue
            updates, metrics = ({}, {})
            for side in limits:
                points = result[side]['points'][first : last + 1]
                q, v, a = (
                    np.asarray([p[k] for p in points], dtype=np.float64)
                    for k in ('positions', 'velocities', 'accelerations')
                )
                old = np.asarray(plans[side].polynomials[first:last], dtype=np.float64)
                try:
                    nv, na = refine_c3_derivatives(
                        q, v, a, old, old_gaps, gaps, preserve_boundary_segments=True
                    )
                except ValueError as error:
                    failures.append({'factor': factor, 'side': side, 'reason': str(error)})
                    break
                c = np.asarray(
                    [
                        coefficients(q[i], q[i + 1], nv[i], nv[i + 1], na[i], na[i + 1], float(h))
                        for i, h in enumerate(gaps)
                    ]
                )
                before, after = (_metrics(old, old_gaps), _metrics(c, gaps))
                peaks = np.asarray(after['peak_v_a_j'])
                shape = max((float(np.max(abs(_extrema(x - y, 1.0, 0)))) for x, y in zip(c, old)))
                bounds = np.asarray([_extrema(x, float(h), 0) for x, h in zip(c, gaps)])
                ratios = [
                    after[k] / max(before[k], 1e-30) * (sum(gaps) / duration) ** power
                    for k, power in [
                        ('integrated_squared_jerk', 5),
                        ('integrated_squared_snap', 7),
                    ]
                ]
                boundary_jets_ok = all(
                    (
                        np.max(abs(_jets(c[i], gaps[i], end) - _jets(old[i], old_gaps[i], end)))
                        <= 1e-05
                        for i, end in ((0, False), (-1, True))
                    )
                )
                vcap = np.minimum(limits[side].max_velocity, [0.45, 0.16, 0.35, 0.45, 0.45, 0.45])
                acap = np.minimum(
                    limits[side].max_acceleration, [0.27, 0.22, 0.27, 0.27, 0.27, 0.27]
                )
                if (
                    shape > 0.00025
                    or np.any(peaks[0] > vcap)
                    or np.any(peaks[1] > acap)
                    or np.any(peaks[2] > 2.0)
                    or (max(ratios) > 1.1)
                    or (
                        after['internal_jerk_jump_rad_s3']
                        > max(0.0001, before['internal_jerk_jump_rad_s3'] + 1e-05)
                    )
                    or np.any(bounds[:, :, 0] < np.asarray(limits[side].lower) - 1e-12)
                    or np.any(bounds[:, :, 1] > np.asarray(limits[side].upper) + 1e-12)
                    or (not boundary_jets_ok)
                    or (not np.array_equal(c[[0, -1]], old[[0, -1]]))
                ):
                    failures.append(
                        {
                            'factor': factor,
                            'side': side,
                            'shape': shape,
                            'ratios': ratios,
                            'peaks': np.max(peaks, axis=1).tolist(),
                            'jump': after['internal_jerk_jump_rad_s3'],
                        }
                    )
                    break
                updates[side] = (nv, na)
                metrics[side] = {
                    'before': before,
                    'after': after,
                    'shape_change_rad': shape,
                    'normalized_jerk_snap_ratios': ratios,
                }
            if len(updates) != 2:
                continue
            prepared = ns.copy()
            prepared[first : last + 1] = ns[first] + local
            prepared[last + 1 :] += local[-1] - (ns[last] - ns[first])
            for side in limits:
                nv, na = updates[side]
                for i, p in enumerate(result[side]['points']):
                    t = int(prepared[i])
                    p['time_from_start'] = {'sec': t // 10**9, 'nanosec': t % 10**9}
                    if first < i < last:
                        p['velocities'] = nv[i - first].tolist()
                        p['accelerations'] = na[i - first].tolist()
            for event in info['reference_events']:
                event['prepared_time_ns'] = remap(event['prepared_time_ns'], ns, prepared)
            schedule = info['gripper_event_schedule']
            schedule['phase_start_ns'] = {
                k: remap(t, ns, prepared) for k, t in schedule['phase_start_ns'].items()
            }
            for event in schedule['events']:
                for key in ('time_ns', 'end_time_ns'):
                    if key in event:
                        event[key] = remap(event[key], ns, prepared)
            for b in info['cooperative_smoothing_blocks']:
                b['duration_s'] = (
                    int(prepared[b['last']]) - int(prepared[b['first']])
                ) / 1000000000.0
            info['prepared_times_ns'] = prepared.tolist()
            info['duration_s'] = int(prepared[-1]) / 1000000000.0
            info['effective_time_scale'] = int(prepared[-1]) / info['source_times_ns'][-1]
            evidence.append(
                {
                    'phase': phase,
                    'first': first,
                    'last': last,
                    'factor': factor,
                    'transition_width': width,
                    'old_duration_s': duration,
                    'new_duration_s': int(local[-1]) / 1000000000.0,
                    'arms': metrics,
                    'rejected': failures,
                }
            )
            break
        else:
            raise ValueError('窗口没有合格提速曲线：' + phase + ' ' + str(failures))
    info['transit210'] = {
        'policy': '206_fixed_nodes_c3_positive_shared_time',
        'windows': evidence,
        'physical_validation_pending': True,
    }
    baseline_meta = info.pop('return_efficiency')
    info['baseline_return_efficiency_before210'] = baseline_meta
    block = next(
        (
            b
            for b in report['cooperative_smoothing_blocks']
            if b['phase'] == 'RETURN_HOME' and b['last'] - b['first'] >= 10
        )
    )
    f, l = (block['first'], block['last'])
    final_plans = prepare_continuous_pair(result, limits, 3.5)
    gaps = np.diff(info['prepared_times_ns'][f : l + 1]) / 1000000000.0
    info['return_efficiency'] = {
        'policy': '206_return_front_only_210',
        'first': f,
        'last': l,
        'old_duration_s': (report['prepared_times_ns'][l] - report['prepared_times_ns'][f])
        / 1000000000.0,
        'new_duration_s': sum(gaps),
        'all_q_nodes_identical': True,
        'changed_window': evidence[-1],
        'physical_validation_pending': True,
        'arms': {s: _metrics(np.asarray(final_plans[s].polynomials[f:l]), gaps) for s in limits},
    }
    return (result, info)


def optimize_transitions(payload, report, limits):
    result, info = transit(payload, report, limits)
    plans = prepare_continuous_pair(result, limits, 3.5)
    f, l = next(
        (
            (w['first'], w['last'])
            for w in info['transit210']['windows']
            if w['phase'] == 'ALIGN_JOINT6'
        )
    )
    ns = np.asarray(info['prepared_times_ns'], dtype=np.int64)
    old_h = np.diff(ns[f : l + 1]) / 1000000000.0
    rejected = []
    for scale in (0.72, 0.73, 0.74, 0.745, 0.75, 0.77, 0.8, *[i / 1000 for i in range(801, 851)]):
        local = np.rint((ns[f : l + 1] - ns[f]) * scale).astype(np.int64)
        gaps = np.diff(local) / 1000000000.0
        if np.any(gaps <= 0):
            raise ValueError('J6节点时间重复')
        accepted = {}
        for side in limits:
            points = result[side]['points'][f : l + 1]
            q, v, a = (
                np.asarray([p[k] for p in points], dtype=np.float64)
                for k in ('positions', 'velocities', 'accelerations')
            )
            old = np.asarray(plans[side].polynomials[f:l], dtype=np.float64)
            nv, na = refine_c3_derivatives(q, v / scale, a / scale**2, old, gaps, gaps)
            c = np.asarray(
                [
                    coefficients(q[i], q[i + 1], nv[i], nv[i + 1], na[i], na[i + 1], float(h))
                    for i, h in enumerate(gaps)
                ]
            )
            m = _metrics(c, gaps)
            peaks = np.asarray(m['peak_v_a_j'])
            extent = np.asarray([_extrema(x, float(h), 0) for x, h in zip(c, gaps)])
            snap = max((float(np.max(abs(_extrema(x, float(h), 4)))) for x, h in zip(c, gaps)))
            shape = max((float(np.max(abs(_extrema(x - y, 1.0, 0)))) for x, y in zip(c, old)))
            edge = np.max(
                np.asarray(
                    [
                        abs(_jets(c[i], gaps[i], end) - _jets(old[i], old_h[i], end))
                        for i, end in ((0, False), (-1, True))
                    ]
                ),
                axis=(0, 2),
            )
            if (
                np.any(peaks[0] > np.minimum(limits[side].max_velocity, 0.495))
                or np.any(peaks[1] > np.minimum(limits[side].max_acceleration, 0.295))
                or np.any(peaks[2] > 2.0)
                or (snap > 50.0)
                or (edge[0] > 0.001)
                or (edge[1] > 5.0)
                or (m['internal_jerk_jump_rad_s3'] > 0.0001)
                or (shape > 0.00025)
                or np.any(extent[:, :, 0] < np.asarray(limits[side].lower) - 1e-12)
                or np.any(extent[:, :, 1] > np.asarray(limits[side].upper) + 1e-12)
            ):
                rejected.append(
                    {
                        'side': side,
                        'scale': scale,
                        'peaks': np.max(peaks, axis=1).tolist(),
                        'snap': snap,
                        'boundary_jerk_snap_change': edge.tolist(),
                        'jerk_jump': m['internal_jerk_jump_rad_s3'],
                    }
                )
                break
            accepted[side] = (q, nv, na, m, snap, edge, shape)
        if len(accepted) != 2:
            continue
        prepared = ns.copy()
        prepared[f : l + 1] = ns[f] + local
        prepared[l + 1 :] += local[-1] - (ns[l] - ns[f])
        for side in limits:
            q, nv, na, *_ = accepted[side]
            for i, p in enumerate(result[side]['points']):
                stamp = int(prepared[i])
                p['time_from_start'] = {'sec': stamp // 10**9, 'nanosec': stamp % 10**9}
                if f <= i <= l:
                    p['velocities'], p['accelerations'] = (nv[i - f].tolist(), na[i - f].tolist())
        for e in info['reference_events']:
            e['prepared_time_ns'] = remap(e['prepared_time_ns'], ns, prepared)
        schedule = info['gripper_event_schedule']
        schedule['phase_start_ns'] = {
            k: remap(t, ns, prepared) for k, t in schedule['phase_start_ns'].items()
        }
        for e in schedule['events']:
            for k in ('time_ns', 'end_time_ns'):
                if k in e:
                    e[k] = remap(e[k], ns, prepared)
        for b in info['cooperative_smoothing_blocks']:
            b['duration_s'] = (int(prepared[b['last']]) - int(prepared[b['first']])) / 1000000000.0
        info['prepared_times_ns'] = prepared.tolist()
        info['duration_s'] = int(prepared[-1]) / 1000000000.0
        info['effective_time_scale'] = int(prepared[-1]) / info['source_times_ns'][-1]
        info['baseline_transit210_before211'] = info.pop('transit210')
        windows = deepcopy(info['baseline_transit210_before211']['windows'])
        windows[0] = {
            'phase': 'ALIGN_JOINT6',
            'first': f,
            'last': l,
            'old_duration_s': (report['prepared_times_ns'][l] - report['prepared_times_ns'][f])
            / 1000000000.0,
            'new_duration_s': int(local[-1]) / 1000000000.0,
            'additional_uniform_scale': scale,
            'arms': {
                s: {
                    'after': accepted[s][3],
                    'peak_snap_rad_s4': accepted[s][4],
                    'boundary_jerk_snap_change': accepted[s][5].tolist(),
                    'shape_change_rad': accepted[s][6],
                }
                for s in limits
            },
        }
        info['transit211'] = {
            'policy': '206_plus_J6_fixed_nodes_C3_and_home_front',
            'windows': windows,
            'rejected': rejected,
            'physical_validation_pending': True,
        }
        return (result, info)
    raise ValueError('J6原节点缩时全部未通过：' + str(rejected))
