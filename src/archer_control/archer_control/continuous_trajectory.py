"""离线生成速度和加速度连续的双臂候选轨迹，不连接设备或发送指令。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from .trajectory import FloatArray, joint_names


def coefficients(q0: FloatArray, q1: FloatArray, v0: FloatArray, v1: FloatArray,
                 a0: FloatArray, a1: FloatArray, duration: float) -> FloatArray:
    """构造归一化段时间上的五次多项式，约束两端位置、速度和加速度。"""
    c = np.zeros((6, 6), dtype=np.float64)
    c[:, 0], c[:, 1], c[:, 2] = q0, v0 * duration, a0 * duration ** 2 / 2.
    delta = q1 - c[:, 0] - c[:, 1] - c[:, 2]
    speed = v1 * duration - c[:, 1] - 2. * c[:, 2]
    accel = a1 * duration ** 2 - 2. * c[:, 2]
    c[:, 3] = 10. * delta - 4. * speed + .5 * accel
    c[:, 4] = -15. * delta + 7. * speed - accel
    c[:, 5] = 6. * delta - 3. * speed + .5 * accel
    return c


def evaluate(c: FloatArray, u: float, duration: float, derivative: int = 0) -> FloatArray:
    """计算一段的关节位置或时间导数；输入进度范围为零到一。"""
    if derivative not in (0, 1, 2) or not math.isfinite(u) or not 0. <= u <= 1.:
        raise ValueError('进度或导数阶数无效')
    result = np.zeros(6, dtype=np.float64)
    for power in range(derivative, 6):
        multiplier = math.factorial(power) / math.factorial(power - derivative)
        result += c[:, power] * multiplier * u ** (power - derivative)
    return np.asarray(result / duration ** derivative, dtype=np.float64)


def _extrema(c: FloatArray, duration: float, derivative: int) -> FloatArray:
    """检查端点及导数实根，不用离散采样替代关节约束峰值检查。"""
    result = np.zeros((6, 2), dtype=np.float64)
    for joint in range(6):
        polynomial = np.asarray([
            c[joint, power] * math.factorial(power) / math.factorial(power - derivative)
            for power in range(derivative, 6)
        ], dtype=np.float64)
        slope = np.asarray([i * polynomial[i] for i in range(1, len(polynomial))],
                           dtype=np.float64)
        roots = (np.polynomial.polynomial.polyroots(np.trim_zeros(slope, 'b'))
                 if np.any(slope) else [])
        candidates = [0., 1.] + [min(1., max(0., float(root.real))) for root in roots
                                 if abs(root.imag) <= 1e-8 and -1e-10 <= root.real <= 1. + 1e-10]
        values = [float(np.polynomial.polynomial.polyval(u, polynomial))
                  / duration ** derivative for u in candidates]
        result[joint] = min(values), max(values)
    return result


def _tangents(q: FloatArray) -> tuple[FloatArray, FloatArray]:
    """以夹持端点三次样条估计内部导数，段组起止导数强制为零。"""
    count = len(q)
    if count == 2:
        return np.zeros_like(q), np.zeros_like(q)
    matrix = np.zeros((count, count), dtype=np.float64)
    rhs = np.zeros_like(q)
    matrix[0, :2] = [2., 1.]
    matrix[-1, -2:] = [1., 2.]
    rhs[0], rhs[-1] = 6. * (q[1] - q[0]), -6. * (q[-1] - q[-2])
    for index in range(1, count - 1):
        matrix[index, index - 1:index + 2] = [1., 4., 1.]
        rhs[index] = 6. * (q[index + 1] - 2. * q[index] + q[index - 1])
    acceleration = np.asarray(np.linalg.solve(matrix, rhs), dtype=np.float64)
    velocity = np.zeros_like(q)
    velocity[1:-1] = (q[1:-1] - q[:-2]
                      + (acceleration[:-2] + 2. * acceleration[1:-1]) / 6.)
    acceleration[[0, -1]] = 0.
    return velocity, acceleration


@dataclass(frozen=True)
class ContinuousPair:
    """两臂共用时间轴；数组在构造后只读，边界处保持最终姿态。"""

    times: FloatArray
    positions: dict[str, FloatArray]
    velocities: dict[str, FloatArray]
    accelerations: dict[str, FloatArray]
    polynomials: dict[str, FloatArray]
    stops: tuple[int, ...]

    def sample(self, side: str, elapsed: float, derivative: int = 0) -> FloatArray:
        """按当前段的绝对已用时间取值，不逐点补发积压命令。"""
        if side not in ('left', 'right') or not math.isfinite(elapsed):
            raise ValueError('侧别或时间无效')
        if derivative not in (0, 1, 2):
            raise ValueError('导数阶数必须为零、一或二')
        if elapsed <= 0. or elapsed >= self.times[-1]:
            if derivative:
                return np.zeros(6, dtype=np.float64)
            return self.positions[side][0 if elapsed <= 0. else -1].copy()
        index = int(np.searchsorted(self.times, elapsed, side='right')) - 1
        duration = float(self.times[index + 1] - self.times[index])
        u = min(1., max(0., (elapsed - float(self.times[index])) / duration))
        return evaluate(self.polynomials[side][index], u, duration, derivative)

    def payload(self) -> dict[str, Any]:
        """候选格式使用版本二和显式导数，旧执行器必须拒绝，不能静默降级。"""
        result: dict[str, Any] = {
            'version': 2, 'interpolation': 'quintic_c2', 'stop_indices': list(self.stops),
        }
        for side in ('left', 'right'):
            points = []
            for index, elapsed in enumerate(self.times):
                stamp = int(round(float(elapsed) * 1e9))
                points.append({
                    'positions': self.positions[side][index].tolist(),
                    'velocities': self.velocities[side][index].tolist(),
                    'accelerations': self.accelerations[side][index].tolist(),
                    'time_from_start': {'sec': stamp // 10 ** 9, 'nanosec': stamp % 10 ** 9},
                })
            result[side] = {'joint_names': list(joint_names(side)), 'points': points}
        return result


def plan_continuous_pair(
    left: ArrayLike, right: ArrayLike, lower: ArrayLike, upper: ArrayLike,
    max_velocity: ArrayLike, max_acceleration: ArrayLike,
    stops: tuple[int, ...] | None = None, margin: float = 1.05,
) -> tuple[ContinuousPair, dict[str, Any]]:
    """各停顿段组独立定时；内部连续通过路径点，解析检查整段角度和导数。"""
    q = {'left': np.asarray(left, dtype=np.float64).copy(),
         'right': np.asarray(right, dtype=np.float64).copy()}
    if any(values.ndim != 2 for values in q.values()):
        raise ValueError('关节路径必须是二维数组')
    count = len(q['left'])
    stops = (0, count - 1) if stops is None else stops
    if any(values.shape != (count, 6) or not np.all(np.isfinite(values)) for values in q.values()):
        raise ValueError('双臂必须提供相同数量的有限六关节路径点')
    if (not 2 <= count <= 1000 or any(type(index) is not int for index in stops)
            or len(stops) < 2 or stops[0] != 0 or stops[-1] != count - 1
            or any(b <= a for a, b in zip(stops, stops[1:]))):
        raise ValueError('停止点索引必须递增且包含首尾路径点')
    if isinstance(margin, bool) or not math.isfinite(margin) or not 1.001 <= margin <= 2.:
        raise ValueError('时间余量必须在1.001到2之间')
    bounds = [np.asarray(values, dtype=np.float64)
              for values in (lower, upper, max_velocity, max_acceleration)]
    low, high, speed, accel = bounds
    if (any(values.shape != (2, 6) or not np.all(np.isfinite(values))
            for values in (low, high, speed, accel)) or np.any(low >= high)
            or np.any(speed <= 0.) or np.any(accel <= 0.)):
        raise ValueError('限制必须是左右两臂各六维，速度和加速度必须为正')
    velocities = {side: np.zeros_like(values) for side, values in q.items()}
    accelerations = {side: np.zeros_like(values) for side, values in q.items()}
    times = np.zeros(count, dtype=np.float64)
    blocks = []
    for first, last in zip(stops, stops[1:]):
        required = .02
        for arm, side in enumerate(('left', 'right')):
            v, a = _tangents(q[side][first:last + 1])
            velocities[side][first:last + 1], accelerations[side][first:last + 1] = v, a
            for index in range(first, last):
                local = index - first
                c = coefficients(q[side][index], q[side][index + 1],
                                 v[local], v[local + 1], a[local], a[local + 1], 1.)
                extrema = _extrema(c, 1., 0)
                if (np.any(extrema[:, 0] < low[arm] - 1e-12)
                        or np.any(extrema[:, 1] > high[arm] + 1e-12)):
                    raise ValueError(f'{side} 第{index + 1}段插值角度超限，不能靠放慢修复')
                velocity_ratio = float(np.max(np.abs(_extrema(c, 1., 1))
                                              / speed[arm, :, None]))
                acceleration_ratio = float(np.max(np.abs(_extrema(c, 1., 2))
                                                  / accel[arm, :, None]))
                required = max(required, velocity_ratio, math.sqrt(acceleration_ratio))
        duration = math.ceil(required * margin * 1e9) / 1e9
        for index in range(first + 1, last + 1):
            times[index] = times[first] + (index - first) * duration
        for side in ('left', 'right'):
            velocities[side][first:last + 1] /= duration
            accelerations[side][first:last + 1] /= duration ** 2
        blocks.append({'first': first, 'last': last, 'duration_s': (last - first) * duration})
    if times[-1] > 3600.:
        raise ValueError('连续候选轨迹超过一小时，未生成')
    # 以导出所用的整数纳秒重建多项式，再次解析检查，避免导出后时间舍入差异。
    times = np.asarray(np.rint(times * 1e9) / 1e9, dtype=np.float64)
    polys = {side: np.stack([
        coefficients(q[side][i], q[side][i + 1], velocities[side][i], velocities[side][i + 1],
                     accelerations[side][i], accelerations[side][i + 1],
                     float(times[i + 1] - times[i]))
        for i in range(count - 1)
    ]) for side in ('left', 'right')}
    report: dict[str, Any] = {'duration_s': float(times[-1]), 'blocks': blocks, 'arms': {}}
    for arm, side in enumerate(('left', 'right')):
        peak_v, peak_a = np.zeros(6), np.zeros(6)
        for i, c in enumerate(polys[side]):
            duration = float(times[i + 1] - times[i])
            positions = _extrema(c, duration, 0)
            peak_v = np.maximum(peak_v, np.max(np.abs(_extrema(c, duration, 1)), axis=1))
            peak_a = np.maximum(peak_a, np.max(np.abs(_extrema(c, duration, 2)), axis=1))
            if (np.any(positions[:, 0] < low[arm] - 1e-12)
                    or np.any(positions[:, 1] > high[arm] + 1e-12)):
                raise ValueError('连续插值角度超出名义限制')
        if (np.any(peak_v > speed[arm] * (1. + 1e-10))
                or np.any(peak_a > accel[arm] * (1. + 1e-10))):
            raise ValueError('连续插值峰值速度或加速度超限')
        report['arms'][side] = {'peak_velocity_rad_s': peak_v.tolist(),
                                'peak_acceleration_rad_s2': peak_a.tolist()}
    arrays = [times, *q.values(), *velocities.values(),
              *accelerations.values(), *polys.values()]
    for values in arrays:
        values.setflags(write=False)
    return ContinuousPair(times, q, velocities, accelerations, polys, stops), report


def retime_recorded_pair(
    source_times_ns: tuple[int, ...], left: ArrayLike, right: ArrayLike,
    lower: ArrayLike, upper: ArrayLike, max_velocity: ArrayLike,
    max_acceleration: ArrayLike, *, stops: tuple[int, ...] = (), margin: float = 1.05,
) -> tuple[ContinuousPair, dict[str, Any]]:
    """保留非均匀采样位置及相对时序，统一放慢并解析检查连续五次插值。"""
    count = len(source_times_ns)
    if (not 2 <= count <= 20000 or any(type(t) is not int for t in source_times_ns)
            or source_times_ns[0] != 0 or source_times_ns[-1] > 3600 * 10 ** 9
            or any(b <= a for a, b in zip(source_times_ns, source_times_ns[1:]))):
        raise ValueError('需要2到20000个从零开始递增的整数纳秒采样，最长一小时')
    if (isinstance(margin, bool) or not math.isfinite(margin)
            or not 1.001 <= margin <= 2.):
        raise ValueError('时间余量必须在1.001到2之间')
    if (any(type(i) is not int or not 0 <= i < count for i in stops)
            or any(b <= a for a, b in zip(stops, stops[1:]))):
        raise ValueError('停止点必须为范围内严格递增的整数索引')
    stops = tuple(sorted({0, count - 1, *stops}))
    q = {'left': np.asarray(left, dtype=np.float64).copy(),
         'right': np.asarray(right, dtype=np.float64).copy()}
    if any(p.shape != (count, 6) or not np.all(np.isfinite(p)) for p in q.values()):
        raise ValueError('两臂采样必须是相同数量的有限六关节角')
    low, high, speed, accel = [np.asarray(p, dtype=np.float64)
                              for p in (lower, upper, max_velocity, max_acceleration)]
    if (any(p.shape != (2, 6) or not np.all(np.isfinite(p))
            for p in (low, high, speed, accel)) or np.any(low >= high)
            or np.any(speed <= 0.) or np.any(accel <= 0.)):
        raise ValueError('限制需要两臂各六维有限数，速度和加速度为正')
    source_times = np.asarray(source_times_ns, dtype=np.float64) / 1e9
    gaps = np.diff(source_times)
    velocities = {side: np.zeros_like(p) for side, p in q.items()}
    accelerations = {side: np.zeros_like(p) for side, p in q.items()}
    for arm, side in enumerate(('left', 'right')):
        if np.any(q[side] < low[arm]) or np.any(q[side] > high[arm]):
            raise ValueError(f'{side} 原始角度超出名义限制')
        slopes = np.diff(q[side], axis=0) / gaps[:, None]
        before, after = slopes[:-1], slopes[1:]
        # 同向时取较小斜率；反向、停顿或显式阶段边界处速度为零。
        # 两端加速度为零，且端速度不超过本段割线，避免段内位置过冲。
        velocities[side][1:-1] = np.where(
            np.sign(before) == np.sign(after),
            np.sign(before) * np.minimum(np.abs(before), np.abs(after)), 0.,
        )
        velocities[side][list(stops)] = 0.

    def build(times: FloatArray) -> dict[str, FloatArray]:
        return {side: np.stack([
            coefficients(q[side][i], q[side][i + 1], velocities[side][i],
                         velocities[side][i + 1], accelerations[side][i],
                         accelerations[side][i + 1], float(times[i + 1] - times[i]))
            for i in range(count - 1)
        ]) for side in ('left', 'right')}

    def measure(polys: dict[str, FloatArray], times: FloatArray) -> dict[str, Any]:
        measured: dict[str, Any] = {}
        for arm, side in enumerate(('left', 'right')):
            peak_v, peak_a = np.zeros(6), np.zeros(6)
            for i, c in enumerate(polys[side]):
                duration = float(times[i + 1] - times[i])
                positions = _extrema(c, duration, 0)
                if (np.any(positions[:, 0] < low[arm] - 1e-12)
                        or np.any(positions[:, 1] > high[arm] + 1e-12)):
                    raise ValueError(f'{side} 第{i + 1}段连续插值角度超限')
                peak_v = np.maximum(peak_v, np.max(np.abs(_extrema(c, duration, 1)), axis=1))
                peak_a = np.maximum(peak_a, np.max(np.abs(_extrema(c, duration, 2)), axis=1))
            measured[side] = {'peak_velocity_rad_s': peak_v.tolist(),
                              'peak_acceleration_rad_s2': peak_a.tolist()}
        return measured

    original_peaks = measure(build(source_times), source_times)
    required = 0.
    for arm, side in enumerate(('left', 'right')):
        required = max(required,
                       float(np.max(np.asarray(original_peaks[side]['peak_velocity_rad_s'])
                                    / speed[arm])),
                       math.sqrt(float(np.max(np.asarray(
                           original_peaks[side]['peak_acceleration_rad_s2']) / accel[arm]))))
    scale = max(1., required * margin)
    if not math.isfinite(scale) or source_times[-1] * scale > 3600.:
        raise ValueError('满足限速后的轨迹超过一小时，未生成')
    times = np.asarray([math.ceil(t * scale) / 1e9 for t in source_times_ns], dtype=np.float64)
    for side in ('left', 'right'):
        velocities[side] /= scale
        accelerations[side] /= scale ** 2
    polys = build(times)
    peaks = measure(polys, times)
    for arm, side in enumerate(('left', 'right')):
        if (np.any(np.asarray(peaks[side]['peak_velocity_rad_s']) > speed[arm] * (1. + 1e-10))
                or np.any(np.asarray(peaks[side]['peak_acceleration_rad_s2'])
                          > accel[arm] * (1. + 1e-10))):
            raise ValueError('纳秒时间舍入后峰值超限，未生成轨迹')
    for array in (times, *q.values(), *velocities.values(),
                  *accelerations.values(), *polys.values()):
        array.setflags(write=False)
    return ContinuousPair(times, q, velocities, accelerations, polys, stops), {
        'continuous_constraints_passed': True, 'samples_preserved': count,
        'source_duration_s': float(source_times[-1]), 'duration_s': float(times[-1]),
        'uniform_time_scale': scale, 'stop_indices': list(stops), 'arms': peaks,
    }
