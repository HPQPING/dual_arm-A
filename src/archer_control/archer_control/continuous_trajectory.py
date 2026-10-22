"""离线生成速度和加速度连续的双臂候选轨迹，不连接设备或发送指令."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .trajectory import FloatArray, joint_names
from archer_mujoco.joint_interpolation import coefficients, evaluate, _extrema


def _tangents(q: FloatArray) -> tuple[FloatArray, FloatArray]:
    """以夹持端点三次样条估计内部导数，段组起止导数强制为零."""
    count = len(q)
    if count == 2:
        return np.zeros_like(q), np.zeros_like(q)
    matrix = np.zeros((count, count), dtype=np.float64)
    rhs = np.zeros_like(q)
    matrix[0, :2] = [2.0, 1.0]
    matrix[-1, -2:] = [1.0, 2.0]
    rhs[0], rhs[-1] = 6.0 * (q[1] - q[0]), -6.0 * (q[-1] - q[-2])
    for index in range(1, count - 1):
        matrix[index, index - 1: index + 2] = [1.0, 4.0, 1.0]
        rhs[index] = 6.0 * (q[index + 1] - 2.0 * q[index] + q[index - 1])
    acceleration = np.asarray(np.linalg.solve(matrix, rhs), dtype=np.float64)
    velocity = np.zeros_like(q)
    velocity[1:-1] = q[1:-1] - q[:-2] + (acceleration[:-2] + 2.0 * acceleration[1:-1]) / 6.0
    acceleration[[0, -1]] = 0.0
    return velocity, acceleration


@dataclass(frozen=True)
class ContinuousPair:
    """两臂共用时间轴；数组在构造后只读，边界处保持最终姿态."""

    times: FloatArray
    positions: dict[str, FloatArray]
    velocities: dict[str, FloatArray]
    accelerations: dict[str, FloatArray]
    polynomials: dict[str, FloatArray]
    stops: tuple[int, ...]

    def sample(self, side: str, elapsed: float, derivative: int = 0) -> FloatArray:
        """按当前段的绝对已用时间取值，不逐点补发积压命令."""
        if side not in ("left", "right") or not math.isfinite(elapsed):
            raise ValueError("侧别或时间无效")
        if derivative not in (0, 1, 2):
            raise ValueError("导数阶数必须为零、一或二")
        if elapsed <= 0.0 or elapsed >= self.times[-1]:
            if derivative:
                return np.zeros(6, dtype=np.float64)
            return self.positions[side][0 if elapsed <= 0.0 else -1].copy()
        index = int(np.searchsorted(self.times, elapsed, side="right")) - 1
        duration = float(self.times[index + 1] - self.times[index])
        u = min(1.0, max(0.0, (elapsed - float(self.times[index])) / duration))
        return evaluate(self.polynomials[side][index], u, duration, derivative)

    def payload(self) -> dict[str, Any]:
        """候选格式使用版本二和显式导数，旧执行器必须拒绝，不能静默降级."""
        result: dict[str, Any] = {
            "version": 2,
            "interpolation": "quintic_c2",
            "stop_indices": list(self.stops),
        }
        for side in ("left", "right"):
            points = []
            for index, elapsed in enumerate(self.times):
                stamp = int(round(float(elapsed) * 1e9))
                points.append(
                    {
                        "positions": self.positions[side][index].tolist(),
                        "velocities": self.velocities[side][index].tolist(),
                        "accelerations": self.accelerations[side][index].tolist(),
                        "time_from_start": {"sec": stamp // 10**9, "nanosec": stamp % 10**9},
                    }
                )
            result[side] = {"joint_names": list(joint_names(side)), "points": points}
        return result


def plan_continuous_pair(
    left: ArrayLike,
    right: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    max_velocity: ArrayLike,
    max_acceleration: ArrayLike,
    stops: tuple[int, ...] | None = None,
    margin: float = 1.05,
) -> tuple[ContinuousPair, dict[str, Any]]:
    """各停顿段组独立定时；内部连续通过路径点，解析检查整段角度和导数."""
    q = {
        "left": np.asarray(left, dtype=np.float64).copy(),
        "right": np.asarray(right, dtype=np.float64).copy(),
    }
    if any(values.ndim != 2 for values in q.values()):
        raise ValueError("关节路径必须是二维数组")
    count = len(q["left"])
    stops = (0, count - 1) if stops is None else stops
    if any(values.shape != (count, 6) or not np.all(np.isfinite(values)) for values in q.values()):
        raise ValueError("双臂必须提供相同数量的有限六关节路径点")
    if (
        not 2 <= count <= 1000
        or any(type(index) is not int for index in stops)
        or len(stops) < 2
        or stops[0] != 0
        or stops[-1] != count - 1
        or any(b <= a for a, b in zip(stops, stops[1:]))
    ):
        raise ValueError("停止点索引必须递增且包含首尾路径点")
    if isinstance(margin, bool) or not math.isfinite(margin) or not 1.001 <= margin <= 2.0:
        raise ValueError("时间余量必须在1.001到2之间")
    bounds = [
        np.asarray(values, dtype=np.float64)
        for values in (lower, upper, max_velocity, max_acceleration)
    ]
    low, high, speed, accel = bounds
    if (
        any(
            values.shape != (2, 6) or not np.all(np.isfinite(values))
            for values in (low, high, speed, accel)
        )
        or np.any(low >= high)
        or np.any(speed <= 0.0)
        or np.any(accel <= 0.0)
    ):
        raise ValueError("限制必须是左右两臂各六维，速度和加速度必须为正")
    velocities = {side: np.zeros_like(values) for side, values in q.items()}
    accelerations = {side: np.zeros_like(values) for side, values in q.items()}
    times = np.zeros(count, dtype=np.float64)
    blocks = []
    for first, last in zip(stops, stops[1:]):
        required = 0.02
        for arm, side in enumerate(("left", "right")):
            v, a = _tangents(q[side][first: last + 1])
            velocities[side][first: last + 1], accelerations[side][first: last + 1] = v, a
            for index in range(first, last):
                local = index - first
                c = coefficients(
                    q[side][index],
                    q[side][index + 1],
                    v[local],
                    v[local + 1],
                    a[local],
                    a[local + 1],
                    1.0,
                )
                extrema = _extrema(c, 1.0, 0)
                if np.any(extrema[:, 0] < low[arm] - 1e-12) or np.any(
                    extrema[:, 1] > high[arm] + 1e-12
                ):
                    raise ValueError(f"{side} 第{index + 1}段插值角度超限，不能靠放慢修复")
                velocity_ratio = float(np.max(np.abs(_extrema(c, 1.0, 1)) / speed[arm, :, None]))
                acceleration_ratio = float(
                    np.max(np.abs(_extrema(c, 1.0, 2)) / accel[arm, :, None])
                )
                required = max(required, velocity_ratio, math.sqrt(acceleration_ratio))
        duration = math.ceil(required * margin * 1e9) / 1e9
        for index in range(first + 1, last + 1):
            times[index] = times[first] + (index - first) * duration
        for side in ("left", "right"):
            velocities[side][first: last + 1] /= duration
            accelerations[side][first: last + 1] /= duration**2
        blocks.append({"first": first, "last": last, "duration_s": (last - first) * duration})
    if times[-1] > 3600.0:
        raise ValueError("连续候选轨迹超过一小时，未生成")
    # 以导出所用的整数纳秒重建多项式，再次解析检查，避免导出后时间舍入差异。
    times = np.asarray(np.rint(times * 1e9) / 1e9, dtype=np.float64)
    polys = {
        side: np.stack(
            [
                coefficients(
                    q[side][i],
                    q[side][i + 1],
                    velocities[side][i],
                    velocities[side][i + 1],
                    accelerations[side][i],
                    accelerations[side][i + 1],
                    float(times[i + 1] - times[i]),
                )
                for i in range(count - 1)
            ]
        )
        for side in ("left", "right")
    }
    report: dict[str, Any] = {"duration_s": float(times[-1]), "blocks": blocks, "arms": {}}
    for arm, side in enumerate(("left", "right")):
        peak_v, peak_a = np.zeros(6), np.zeros(6)
        for i, c in enumerate(polys[side]):
            duration = float(times[i + 1] - times[i])
            positions = _extrema(c, duration, 0)
            peak_v = np.maximum(peak_v, np.max(np.abs(_extrema(c, duration, 1)), axis=1))
            peak_a = np.maximum(peak_a, np.max(np.abs(_extrema(c, duration, 2)), axis=1))
            if np.any(positions[:, 0] < low[arm] - 1e-12) or np.any(
                positions[:, 1] > high[arm] + 1e-12
            ):
                raise ValueError("连续插值角度超出名义限制")
        if np.any(peak_v > speed[arm] * (1.0 + 1e-10)) or np.any(
            peak_a > accel[arm] * (1.0 + 1e-10)
        ):
            raise ValueError("连续插值峰值速度或加速度超限")
        report["arms"][side] = {
            "peak_velocity_rad_s": peak_v.tolist(),
            "peak_acceleration_rad_s2": peak_a.tolist(),
        }
    arrays = [times, *q.values(), *velocities.values(), *accelerations.values(), *polys.values()]
    for values in arrays:
        values.setflags(write=False)
    return ContinuousPair(times, q, velocities, accelerations, polys, stops), report


@dataclass(frozen=True)
class RecordedTimeMap:
    """核对原始与准备时间轴，按同一映射转换参考事件，兼容旧统一放慢报告."""

    source_ns: tuple[int, ...]
    prepared_ns: tuple[int, ...]
    uniform_scale: float | None = None

    @classmethod
    def from_report(
        cls, report: dict[str, Any], source_ns: tuple[int, ...], prepared_ns: tuple[int, ...],
        *, held_intervals: tuple[bool, ...] | None = None,
    ) -> RecordedTimeMap:
        """使用外部已读原始时间和候选时间核对报告，不相信报告自行提供的来源."""
        if (not 2 <= len(source_ns) <= 20000 or len(source_ns) != len(prepared_ns)
                or any(type(t) is not int for axis in (source_ns, prepared_ns) for t in axis)
                or source_ns[0] != 0 or prepared_ns[0] != 0
                or any(b <= a for axis in (source_ns, prepared_ns)
                       for a, b in zip(axis, axis[1:]))
                or max(source_ns[-1], prepared_ns[-1]) > 3600 * 10 ** 9):
            raise ValueError("采样时间映射必须包含两条从零开始递增的有效整数纳秒轴")
        policy = report.get("timing_policy", "uniform")
        scale = report.get("uniform_time_scale")
        if policy == "uniform":
            if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                    or not math.isfinite(scale) or scale < 1.):
                raise ValueError("统一时间伸长系数必须是大于等于一的有限数")
            if any(abs(b - math.ceil(a * scale)) > 2
                   for a, b in zip(source_ns, prepared_ns)):
                raise ValueError("准备时间与统一时间伸长不一致")
            return cls(source_ns, prepared_ns, float(scale))
        if policy != "local_segment_c2" or scale is not None:
            raise ValueError("时间分配策略或统一伸长字段无效")
        for key, actual in (("source_times_ns", source_ns), ("prepared_times_ns", prepared_ns)):
            values = report.get(key)
            if (not isinstance(values, list) or any(type(t) is not int for t in values)
                    or tuple(values) != actual):
                raise ValueError("时间映射与原始或准备时间轴不一致：" + key)
        held = (True,) * (len(source_ns) - 1) if held_intervals is None else held_intervals
        if len(held) != len(source_ns) - 1 or any(type(v) is not bool for v in held):
            raise ValueError("静止区间标记必须由原始双臂角度逐段产生")
        if any(stop and b - a < d - c for stop, a, b, c, d in zip(
            held, prepared_ns, prepared_ns[1:], source_ns, source_ns[1:],
        )):
            raise ValueError("分段准备不允许缩短原始静止等待")
        stops = report.get("stop_indices")
        if (not isinstance(stops, list) or len(stops) < 2 or stops[0] != 0
                or stops[-1] != len(source_ns) - 1
                or any(type(i) is not int or not 0 <= i < len(source_ns) for i in stops)
                or any(b <= a for a, b in zip(stops, stops[1:]))):
            raise ValueError("时间映射需要完整递增的停止点")
        if any(prepared_ns[b] - prepared_ns[a] < source_ns[b] - source_ns[a]
               for a, b in zip(stops, stops[1:])):
            raise ValueError("不允许缩短阶段或事件停点之间的原始时长")
        effective = report.get("effective_time_scale")
        if (isinstance(effective, bool) or not isinstance(effective, (int, float))
                or not math.isfinite(effective)
                or not math.isclose(effective, prepared_ns[-1] / source_ns[-1],
                                    rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError("整轮时间比例与时间映射不一致")
        return cls(source_ns, prepared_ns)

    def map_ns(self, source_stamp: int) -> int:
        """阶段边界精确落在对应点，区间内事件用整数比例向上取整映射."""
        if type(source_stamp) is not int or not 0 <= source_stamp <= self.source_ns[-1]:
            raise ValueError("参考事件超出原始时间轴")
        if self.uniform_scale is not None:
            return math.ceil(source_stamp * self.uniform_scale)
        i = min(len(self.source_ns) - 2, bisect_right(self.source_ns, source_stamp) - 1)
        gap = self.source_ns[i + 1] - self.source_ns[i]
        numerator = (source_stamp - self.source_ns[i]) * (
            self.prepared_ns[i + 1] - self.prepared_ns[i])
        return self.prepared_ns[i] + (numerator + gap - 1) // gap


def _recorded_derivative_peaks(
    slopes: FloatArray, v0: FloatArray, v1: FloatArray, durations: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """批量求零端加速度五次段的导数实根峰值，不用采样或宽松控制点界限."""
    w3 = 10. * slopes - 6. * v0 - 4. * v1
    w4 = -15. * slopes + 8. * v0 + 7. * v1
    w5 = 6. * slopes - 3. * (v0 + v1)

    def roots(a: FloatArray, b: FloatArray, c: FloatArray) -> tuple[FloatArray, FloatArray]:
        """稳定二次求根，兼容退化直线；区间外实根只替换成已检查端点."""
        disc = b * b - 4. * a * c
        numerator = -.5 * (b + np.copysign(np.sqrt(np.maximum(disc, 0.)), b))
        valid = (disc >= 0.) & (a != 0.)
        first = np.divide(numerator, a, out=np.full_like(a, np.nan), where=valid)
        second = np.divide(c, numerator, out=np.full_like(a, np.nan),
                           where=valid & (numerator != 0.))
        linear = (a == 0.) & (b != 0.)
        np.divide(-c, b, out=first, where=linear)
        clipped = [np.asarray(np.where(np.isfinite(r) & (r >= 0.) & (r <= 1.), r, 0.),
                              dtype=np.float64) for r in (first, second)]
        return clipped[0], clipped[1]

    velocity = np.maximum(np.abs(v0), np.abs(v1))
    for u in roots(20. * w5, 12. * w4, 6. * w3):
        velocity = np.maximum(velocity, np.abs(v0 + u ** 2 * (
            3. * w3 + u * (4. * w4 + 5. * w5 * u))))
    acceleration = np.zeros_like(velocity)
    for u in roots(60. * w5, 24. * w4, 6. * w3):
        acceleration = np.maximum(acceleration, np.abs(u * (
            6. * w3 + u * (12. * w4 + 20. * w5 * u))) / durations[:, None])
    return velocity, acceleration


def retime_recorded_pair(
    source_times_ns: tuple[int, ...],
    left: ArrayLike,
    right: ArrayLike,
    lower: ArrayLike,
    upper: ArrayLike,
    max_velocity: ArrayLike,
    max_acceleration: ArrayLike,
    *,
    stops: tuple[int, ...] = (),
    margin: float = 1.05,
) -> tuple[ContinuousPair, dict[str, Any]]:
    """保留全部采样点，局部分配双臂共同时间并重新检查连续五次插值."""
    count = len(source_times_ns)
    if (
        not 2 <= count <= 20000
        or any(type(t) is not int for t in source_times_ns)
        or source_times_ns[0] != 0
        or source_times_ns[-1] > 3600 * 10**9
        or any(b <= a for a, b in zip(source_times_ns, source_times_ns[1:]))
    ):
        raise ValueError("需要2到20000个从零开始递增的整数纳秒采样，最长一小时")
    if isinstance(margin, bool) or not math.isfinite(margin) or not 1.001 <= margin <= 2.0:
        raise ValueError("时间余量必须在1.001到2之间")
    if any(type(i) is not int or not 0 <= i < count for i in stops) or any(
        b <= a for a, b in zip(stops, stops[1:])
    ):
        raise ValueError("停止点必须为范围内严格递增的整数索引")
    stops = tuple(sorted({0, count - 1, *stops}))
    q = {
        "left": np.asarray(left, dtype=np.float64).copy(),
        "right": np.asarray(right, dtype=np.float64).copy(),
    }
    if any(p.shape != (count, 6) or not np.all(np.isfinite(p)) for p in q.values()):
        raise ValueError("两臂采样必须是相同数量的有限六关节角")
    low, high, speed, accel = [
        np.asarray(p, dtype=np.float64) for p in (lower, upper, max_velocity, max_acceleration)
    ]
    if (
        any(p.shape != (2, 6) or not np.all(np.isfinite(p)) for p in (low, high, speed, accel))
        or np.any(low >= high)
        or np.any(speed <= 0.0)
        or np.any(accel <= 0.0)
    ):
        raise ValueError("限制需要两臂各六维有限数，速度和加速度为正")
    source_times = np.asarray(source_times_ns, dtype=np.float64) / 1e9
    gaps = np.diff(source_times)
    velocities = {side: np.zeros_like(p) for side, p in q.items()}
    accelerations = {side: np.zeros_like(p) for side, p in q.items()}
    for arm, side in enumerate(("left", "right")):
        if np.any(q[side] < low[arm]) or np.any(q[side] > high[arm]):
            raise ValueError(f"{side} 原始角度超出名义限制")
        slopes = np.diff(q[side], axis=0) / gaps[:, None]
        before, after = slopes[:-1], slopes[1:]
        # 同向时取较小斜率；反向、停顿或显式阶段边界处速度为零。
        # 两端加速度为零，且端速度不超过本段割线，避免段内位置过冲。
        velocities[side][1:-1] = np.where(
            np.sign(before) == np.sign(after),
            np.sign(before) * np.minimum(np.abs(before), np.abs(after)),
            0.0,
        )
        velocities[side][list(stops)] = 0.0

    def build(times: FloatArray) -> dict[str, FloatArray]:
        return {
            side: np.stack(
                [
                    coefficients(
                        q[side][i],
                        q[side][i + 1],
                        velocities[side][i],
                        velocities[side][i + 1],
                        accelerations[side][i],
                        accelerations[side][i + 1],
                        float(times[i + 1] - times[i]),
                    )
                    for i in range(count - 1)
                ]
            )
            for side in ("left", "right")
        }

    def measure(polys: dict[str, FloatArray], times: FloatArray) -> dict[str, Any]:
        measured: dict[str, Any] = {}
        for arm, side in enumerate(("left", "right")):
            peak_v, peak_a = np.zeros(6), np.zeros(6)
            for i, c in enumerate(polys[side]):
                duration = float(times[i + 1] - times[i])
                positions = _extrema(c, duration, 0)
                if np.any(positions[:, 0] < low[arm] - 1e-12) or np.any(
                    positions[:, 1] > high[arm] + 1e-12
                ):
                    raise ValueError(f"{side} 第{i + 1}段连续插值角度超限")
                peak_v = np.maximum(peak_v, np.max(np.abs(_extrema(c, duration, 1)), axis=1))
                peak_a = np.maximum(peak_a, np.max(np.abs(_extrema(c, duration, 2)), axis=1))
            measured[side] = {
                "peak_velocity_rad_s": peak_v.tolist(),
                "peak_acceleration_rad_s2": peak_a.tolist(),
            }
        return measured

    original_polys = build(source_times)
    original_peaks = measure(original_polys, source_times)
    required = 0.0
    for arm, side in enumerate(("left", "right")):
        required = max(
            required,
            float(np.max(np.asarray(original_peaks[side]["peak_velocity_rad_s"]) / speed[arm])),
            math.sqrt(
                float(
                    np.max(
                        np.asarray(original_peaks[side]["peak_acceleration_rad_s2"]) / accel[arm]
                    )
                )
            ),
        )
    scale = max(1.0, required * margin)
    if not math.isfinite(scale):
        raise ValueError("时间伸长系数无效，未生成轨迹")
    original_velocities = {s: v.copy() for s, v in velocities.items()}
    gap_ns = np.diff(np.asarray(source_times_ns, dtype=np.int64))
    distance_time = np.maximum.reduce([
        np.max(np.abs(np.diff(q[s], axis=0)) / speed[i], axis=1)
        for i, s in enumerate(("left", "right"))])
    # 以关节位移初始化局部时间，避免把采样时钟抖动原样当作速度波动。
    # 真实静止与阶段等待作为下界；运动日志的短间隔不是必须执行的等待。
    total_distance_time = float(np.sum(distance_time))
    reference = max(1., source_times[-1] / total_distance_time) if total_distance_time else 1.
    held = np.all(np.diff(q['left'], axis=0) == 0., axis=1) & np.all(
        np.diff(q['right'], axis=0) == 0., axis=1)
    seeded = np.ceil(np.maximum(np.where(held, gap_ns, 1), distance_time * reference * 1e9))
    for first, last in zip(stops, stops[1:]):
        original_span = source_times_ns[last] - source_times_ns[first]
        block = seeded[first:last]
        if np.sum(block) < original_span:
            seeded[first:last] = np.ceil(block * (original_span / float(np.sum(block))))
    if np.sum(seeded) < min(3600. * 1e9, source_times_ns[-1] * scale):
        gap_ns = seeded.astype(np.int64)
    initial_gaps = gap_ns.copy()
    best_gaps: NDArray[np.int64] | None = None
    best_velocities: dict[str, FloatArray] = {}
    best_iterations, best_method = 0, ''
    # 同一结点两边共用重算速度，保持C2连续和显式停止点。
    # 同时尝试直接和阻尼伸长，避免一次过度伸长迫使整轮回退；只选择已满足约束的较短结果。
    for exponent, maximum in ((1., 64), (.5, 256)):
        gap_ns = initial_gaps.copy()
        for iteration in range(maximum):
            durations = gap_ns.astype(np.float64) / 1e9
            factors = np.ones(count - 1, dtype=np.float64)
            for arm, side in enumerate(("left", "right")):
                slopes = np.diff(q[side], axis=0) / durations[:, None]
                velocities[side].fill(0.)
                before, after = slopes[:-1], slopes[1:]
                velocities[side][1:-1] = np.where(
                    np.sign(before) == np.sign(after),
                    np.sign(before) * np.minimum(np.abs(before), np.abs(after)), 0.)
                velocities[side][list(stops)] = 0.
                v0, v1 = velocities[side][:-1], velocities[side][1:]
                speed_bound, accel_bound = _recorded_derivative_peaks(slopes, v0, v1, durations)
                ratio = np.maximum(np.max(speed_bound / speed[arm], axis=1),
                                   np.sqrt(np.max(accel_bound / accel[arm], axis=1)))
                factors = np.maximum(factors, ratio * margin)
            if np.max(factors) <= 1. + 1e-8:
                if best_gaps is None or np.sum(gap_ns) < np.sum(best_gaps):
                    best_gaps = gap_ns.copy()
                    best_velocities = {s: v.copy() for s, v in velocities.items()}
                    best_iterations = iteration + 1
                    best_method = 'direct' if exponent == 1. else 'damped'
                break
            proposed = np.ceil(gap_ns * np.power(factors, exponent))
            if (not np.all(np.isfinite(proposed)) or np.sum(proposed) > 3600. * 1e9
                    or np.sum(proposed) >= source_times_ns[-1] * scale):
                break
            gap_ns = proposed.astype(np.int64)
    if best_gaps is not None and int(np.sum(best_gaps)) < source_times_ns[-1] * scale:
        gap_ns = best_gaps
        for side in ('left', 'right'):
            velocities[side][:] = best_velocities[side]
        stamps = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(gap_ns)))
        times = np.asarray(stamps / 1e9, dtype=np.float64)
        timing: dict[str, Any] = {
            "timing_policy": "local_segment_c2", "uniform_time_scale": None,
            "effective_time_scale": float(times[-1] / source_times[-1]),
            "source_times_ns": list(source_times_ns),
            "prepared_times_ns": [int(t) for t in stamps],
            "local_retime_iterations": best_iterations,
            "local_retime_method": best_method,
        }
    else:
        if source_times[-1] * scale > 3600.:
            raise ValueError("满足限速后的轨迹超过一小时，未生成")
        times = np.asarray([math.ceil(t * scale) / 1e9 for t in source_times_ns], dtype=np.float64)
        for side in ("left", "right"):
            velocities[side][:] = original_velocities[side] / scale
        timing = {"timing_policy": "uniform", "uniform_time_scale": scale}
    polys = build(times)
    peaks = measure(polys, times)
    shape_change = {
        side: max(float(np.max(np.abs(_extrema(a - b, 1., 0))))
                  for a, b in zip(polys[side], original_polys[side]))
        for side in ("left", "right")
    }
    for arm, side in enumerate(("left", "right")):
        if np.any(
            np.asarray(peaks[side]["peak_velocity_rad_s"]) > speed[arm] * (1.0 + 1e-10)
        ) or np.any(
            np.asarray(peaks[side]["peak_acceleration_rad_s2"]) > accel[arm] * (1.0 + 1e-10)
        ):
            raise ValueError("纳秒时间舍入后峰值超限，未生成轨迹")
    for array in (
        times,
        *q.values(),
        *velocities.values(),
        *accelerations.values(),
        *polys.values(),
    ):
        array.setflags(write=False)
    return ContinuousPair(times, q, velocities, accelerations, polys, stops), {
        "continuous_constraints_passed": True,
        "samples_preserved": count,
        "source_duration_s": float(source_times[-1]),
        "duration_s": float(times[-1]),
        "uniform_baseline_duration_s": float(source_times[-1] * scale),
        **timing,
        "max_segment_shape_change_rad": shape_change,
        "stop_indices": list(stops),
        "arms": peaks,
    }


def smooth_recorded_pair(
    plan: ContinuousPair,
    lower: ArrayLike,
    upper: ArrayLike,
    max_velocity: ArrayLike,
    max_acceleration: ArrayLike,
    *,
    margin: float = 1.05,
) -> tuple[ContinuousPair, dict[str, Any]]:
    """离线用五次样条重算节点导数，保留位置和停点；执行器不依赖SciPy。"""
    try:
        from scipy.interpolate import make_interp_spline
    except (ImportError, ValueError) as error:
        raise ValueError('平滑转换需要与NumPy兼容的SciPy；请在隔离转换环境安装') from error
    bounds = [np.asarray(p, dtype=np.float64) for p in
              (lower, upper, max_velocity, max_acceleration)]
    low, high, speed, accel = bounds
    if (any(p.shape != (2, 6) or not np.all(np.isfinite(p)) for p in bounds)
            or np.any(low >= high) or np.any(speed <= 0.) or np.any(accel <= 0.)
            or isinstance(margin, bool) or not math.isfinite(margin)
            or not 1.001 <= margin <= 2.):
        raise ValueError('平滑需要有效的双臂关节约束及时间余量')
    original_ns = np.rint(plan.times * 1e9).astype(np.int64)
    zeros = np.zeros(6, dtype=np.float64)
    held = np.all(np.diff(plan.positions['left'], axis=0) == 0., axis=1) & np.all(
        np.diff(plan.positions['right'], axis=0) == 0., axis=1)
    stops = sorted(set(plan.stops) | {
        j for i in np.flatnonzero(held) for j in (int(i), int(i) + 1)})
    side_stops = {
        side: sorted(set(stops) | {
            j for i in np.flatnonzero(np.all(np.diff(plan.positions[side], axis=0) == 0., axis=1))
            for j in (int(i), int(i) + 1)}) for side in ('left', 'right')}

    def jerk_jump(polys: FloatArray, times: FloatArray, boundaries: list[int]) -> float:
        """比较普通节点两侧三阶时间导数；阶段停止允许重新起步。"""
        gaps = np.diff(times)
        end = (6. * polys[:-1, :, 3] + 24. * polys[:-1, :, 4]
               + 60. * polys[:-1, :, 5]) / gaps[:-1, None] ** 3
        start = 6. * polys[1:, :, 3] / gaps[1:, None] ** 3
        ordinary = np.ones(len(times) - 2, dtype=bool)
        ordinary[np.asarray(boundaries[1:-1], dtype=int) - 1] = False
        return float(np.max(np.abs(end[ordinary] - start[ordinary]), initial=0.))

    scale = 1.
    for _ in range(4):
        times = np.asarray(np.ceil(original_ns * scale) / 1e9, dtype=np.float64)
        if times[-1] > 3600.:
            raise ValueError('平滑后满足约束的时长超过一小时，未生成')
        velocities, accelerations, polynomials, peaks = {}, {}, {}, {}
        ratio = 1.
        for arm, side in enumerate(('left', 'right')):
            q = plan.positions[side]
            v, a = np.zeros_like(q), np.zeros_like(q)
            boundaries = side_stops[side]
            for first, last in zip(boundaries, boundaries[1:]):
                x = times[first:last + 1] - times[first]
                # 普通节点共用全局样条导数；只有明确停点的速度和加速度为零。
                spline = make_interp_spline(
                    x, q[first:last + 1], k=5,
                    bc_type=([(1, zeros), (2, zeros)], [(1, zeros), (2, zeros)]),
                )
                v[first:last + 1] = spline(x, 1)
                a[first:last + 1] = spline(x, 2)
            v[boundaries], a[boundaries] = 0., 0.
            polys = np.stack([
                coefficients(q[i], q[i + 1], v[i], v[i + 1], a[i], a[i + 1], float(h))
                for i, h in enumerate(np.diff(times))
            ])
            peak_v, peak_a = zeros.copy(), zeros.copy()
            for i, (c, h) in enumerate(zip(polys, np.diff(times))):
                extent = _extrema(c, float(h), 0)
                if np.any(extent[:, 0] < low[arm] - 1e-12) or np.any(
                        extent[:, 1] > high[arm] + 1e-12):
                    raise ValueError(f'{side} 平滑后第{i + 1}段角度超限，未生成')
                peak_v = np.maximum(peak_v, np.max(np.abs(_extrema(c, float(h), 1)), axis=1))
                peak_a = np.maximum(peak_a, np.max(np.abs(_extrema(c, float(h), 2)), axis=1))
            ratio = max(ratio, float(np.max(peak_v / speed[arm])),
                        math.sqrt(float(np.max(peak_a / accel[arm]))))
            velocities[side], accelerations[side], polynomials[side] = v, a, polys
            peaks[side] = {'peak_velocity_rad_s': peak_v.tolist(),
                           'peak_acceleration_rad_s2': peak_a.tolist()}
        if ratio <= 1.:
            break
        # 只在峰值超限时统一伸长，随后在最终纳秒时间上重新拟合，不提高配置上限。
        scale *= ratio * margin
    else:
        raise ValueError('平滑后的连续速度或加速度未满足限制，未生成')
    changes = {
        side: max(float(np.max(np.abs(_extrema(a - b, 1., 0))))
                  for a, b in zip(polynomials[side], plan.polynomials[side]))
        for side in ('left', 'right')
    }
    report = {
        'smoothing_policy': 'clamped_quintic_spline',
        'smoothing_time_scale': scale,
        'smoothing_max_segment_change_rad': changes,
        'smoothing_jerk_jump_before_rad_s3': {
            s: jerk_jump(plan.polynomials[s], plan.times, side_stops[s])
            for s in ('left', 'right')},
        'smoothing_jerk_jump_after_rad_s3': {
            s: jerk_jump(polynomials[s], times, side_stops[s]) for s in ('left', 'right')},
        'arms': peaks, 'duration_s': float(times[-1]),
    }
    for array in (times, *velocities.values(), *accelerations.values(), *polynomials.values()):
        array.setflags(write=False)
    return ContinuousPair(times, plan.positions, velocities, accelerations,
                          polynomials, tuple(stops)), report


def retime_spatial_pair(
    source_times_ns: tuple[int, ...], left: ArrayLike, right: ArrayLike,
    lower: ArrayLike, upper: ArrayLike, max_velocity: ArrayLike, max_acceleration: ArrayLike,
    *, stops: tuple[int, ...], margin: float = 1.05, tolerance: float = .0005,
) -> tuple[ContinuousPair, dict[str, Any]]:
    """按共同关节弧长拟合有界路径，再分阶段定时；不连接设备或改写源记录。"""
    try:
        from scipy.interpolate import BSpline
    except (ImportError, ValueError) as error:
        raise ValueError('路径重定时需要与NumPy兼容的SciPy转换环境') from error
    count = len(source_times_ns)
    source = np.asarray(source_times_ns, dtype=np.float64) / 1e9
    raw = np.asarray([left, right], dtype=np.float64)
    low, high, speed, accel = [np.asarray(x, dtype=np.float64) for x in
                               (lower, upper, max_velocity, max_acceleration)]
    if (not 2 <= count <= 20000 or raw.shape != (2, count, 6)
            or not np.all(np.isfinite(raw)) or source_times_ns[0] != 0
            or any(type(t) is not int for t in source_times_ns)
            or np.any(np.diff(source) <= 0.) or source[-1] > 3600.
            or any(b.shape != (2, 6) or not np.all(np.isfinite(b))
                   for b in (low, high, speed, accel))
            or np.any(low >= high) or np.any(speed <= 0.) or np.any(accel <= 0.)
            or np.any(raw < low[:, None, :]) or np.any(raw > high[:, None, :])
            or isinstance(tolerance, bool) or not math.isfinite(tolerance)
            or not 0. < tolerance <= .0005
            or isinstance(margin, bool) or not math.isfinite(margin) or not 1.001 <= margin <= 2.
            or not stops or stops[0] != 0 or stops[-1] != count - 1
            or any(type(i) is not int or not 0 <= i < count for i in stops)
            or any(b <= a for a, b in zip(stops, stops[1:]))):
        raise ValueError('路径重定时的采样、限制、停点或误差上限无效')
    held = np.all(np.diff(raw, axis=1) == 0., axis=(0, 2))
    stops = tuple(sorted(set(stops) | {
        j for i in np.flatnonzero(held) for j in (int(i), int(i) + 1)}))
    q, velocity, acceleration = raw.copy(), np.zeros_like(raw), np.zeros_like(raw)
    times_ns = np.zeros(count, dtype=np.int64)
    polys = {s: [] for s in ('left', 'right')}
    peaks = {s: {'peak_velocity_rad_s': np.zeros(6), 'peak_acceleration_rad_s2': np.zeros(6)}
             for s in polys}
    blocks = []
    for first, last in zip(stops, stops[1:]):
        original = np.concatenate((raw[0, first:last + 1], raw[1, first:last + 1]), axis=1)
        length = np.r_[0., np.cumsum(np.linalg.norm(np.diff(original, axis=0), axis=1))]
        span = float(source[last] - source[first])
        n = len(original)
        controls = 0
        if length[-1] == 0.:
            x = (source[first:last + 1] - source[first]) / span
            fitted, dx, dxx = original.copy(), np.zeros_like(original), np.zeros_like(original)
            u = x.copy()
        else:
            x = length / length[-1]
            if n == 2:
                fitted = original.copy()
                dx = np.repeat((original[-1] - original[0])[None, :], n, axis=0)
                dxx = np.zeros_like(original)
            else:
                degree = min(5, n - 1)
                for requested in (8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512):
                    controls = min(requested, n)
                    knots = np.r_[np.zeros(degree + 1),
                                  np.linspace(0., 1., controls - degree + 1)[1:-1],
                                  np.ones(degree + 1)]
                    matrix = BSpline.design_matrix(x, knots, degree).toarray()
                    c = np.empty((controls, 12), dtype=np.float64)
                    c[0], c[-1] = original[0], original[-1]
                    rhs = original - matrix[:, 0, None] * c[0] - matrix[:, -1, None] * c[-1]
                    c[1:-1] = np.linalg.lstsq(matrix[:, 1:-1], rhs, rcond=None)[0]
                    spline = BSpline(knots, c, degree)
                    fitted = np.asarray(spline(x), dtype=np.float64)
                    fitted[[0, -1]] = original[[0, -1]]
                    if float(np.max(np.abs(fitted - original))) <= tolerance:
                        dx = np.asarray(spline(x, 1), dtype=np.float64)
                        dxx = np.asarray(spline(x, 2), dtype=np.float64)
                        break
                else:
                    raise ValueError('有限控制点内无法达到位置误差上限，保留旧转换文件')
            # 用五次进度代替加速度突变的梯形；对共同弧长做单调反求。
            lo, hi = np.zeros(n), np.ones(n)
            for _ in range(55):
                u = (lo + hi) / 2.
                progress = u**3 * (10. + u * (-15. + 6. * u))
                lo = np.where(progress < x, u, lo)
                hi = np.where(progress >= x, u, hi)
            u = (lo + hi) / 2.
            u[0], u[-1] = 0., 1.
        progress_v = 30. * u**2 * (1. - u)**2
        progress_a = 60. * u * (1. - u) * (1. - 2. * u)
        unit_v = dx * progress_v[:, None]
        # 同时计入路径曲率，不能只限制沿路径的加速度。
        unit_a = dxx * progress_v[:, None]**2 + dx * progress_a[:, None]
        unit_v[[0, -1]], unit_a[[0, -1]] = 0., 0.
        duration = span
        for arm in range(2):
            section = slice(arm * 6, (arm + 1) * 6)
            for i, h in enumerate(np.diff(u)):
                c = coefficients(fitted[i, section], fitted[i + 1, section],
                                 unit_v[i, section], unit_v[i + 1, section],
                                 unit_a[i, section], unit_a[i + 1, section], float(h))
                duration = max(duration,
                               margin * float(np.max(np.abs(_extrema(c, float(h), 1))
                                                     / speed[arm, :, None])),
                               margin * math.sqrt(float(np.max(
                                   np.abs(_extrema(c, float(h), 2)) / accel[arm, :, None]))))
        if times_ns[first] / 1e9 + duration > 3600.:
            raise ValueError('路径重定时超过一小时，未生成')
        duration_ns = max(math.ceil(duration * 1e9),
                          source_times_ns[last] - source_times_ns[first])
        stamps = np.rint(u * duration_ns).astype(np.int64)
        if np.any(np.diff(stamps) <= 0):
            raise ValueError('路径节点在纳秒时间上重合，未生成')
        duration = int(stamps[-1]) / 1e9
        times_ns[first:last + 1] = times_ns[first] + stamps
        if times_ns[-1] > 3600 * 10**9:
            raise ValueError('路径重定时超过一小时，未生成')
        for arm, side in enumerate(('left', 'right')):
            section = slice(arm * 6, (arm + 1) * 6)
            q[arm, first:last + 1] = fitted[:, section]
            velocity[arm, first:last + 1] = unit_v[:, section] / duration
            acceleration[arm, first:last + 1] = unit_a[:, section] / duration**2
            for i, h in enumerate(np.diff(stamps) / 1e9):
                j = first + i
                c = coefficients(q[arm, j], q[arm, j + 1],
                                 velocity[arm, j], velocity[arm, j + 1],
                                 acceleration[arm, j], acceleration[arm, j + 1], float(h))
                extent = _extrema(c, float(h), 0)
                pv = np.max(np.abs(_extrema(c, float(h), 1)), axis=1)
                pa = np.max(np.abs(_extrema(c, float(h), 2)), axis=1)
                if (np.any(extent[:, 0] < low[arm] - 1e-12)
                        or np.any(extent[:, 1] > high[arm] + 1e-12)
                        or np.any(pv > speed[arm] * (1. + 1e-8))
                        or np.any(pa > accel[arm] * (1. + 1e-8))):
                    raise ValueError(f'{side} 重定时后段内位置、速度或加速度超限，未生成')
                polys[side].append(c)
                peaks[side]['peak_velocity_rad_s'] = np.maximum(
                    peaks[side]['peak_velocity_rad_s'], pv)
                peaks[side]['peak_acceleration_rad_s2'] = np.maximum(
                    peaks[side]['peak_acceleration_rad_s2'], pa)
        blocks.append({'first': first, 'last': last, 'duration_s': duration,
                       'spline_controls': controls})
    times = np.asarray(times_ns / 1e9, dtype=np.float64)
    arrays = [{s: values[i].copy() for i, s in enumerate(('left', 'right'))}
              for values in (q, velocity, acceleration)]
    polynomials = {s: np.stack(values) for s, values in polys.items()}
    for values in (times, *(a for group in arrays for a in group.values()), *polynomials.values()):
        values.setflags(write=False)
    changes = {s: float(np.max(np.abs(q[i] - raw[i])))
               for i, s in enumerate(('left', 'right'))}
    return ContinuousPair(times, *arrays, polynomials, stops), {
        'continuous_constraints_passed': True, 'samples_preserved': count,
        'positions_preserved': False, 'position_filter': 'bounded_spatial_spline',
        'position_smoothing_tolerance_rad': tolerance, 'max_sample_position_change_rad': changes,
        'source_duration_s': float(source[-1]), 'duration_s': float(times[-1]),
        'timing_policy': 'local_segment_c2', 'uniform_time_scale': None,
        'effective_time_scale': float(times[-1] / source[-1]),
        'source_times_ns': list(source_times_ns), 'prepared_times_ns': times_ns.tolist(),
        'stop_indices': list(stops), 'spatial_blocks': blocks,
        'arms': {s: {name: values.tolist() for name, values in arm.items()}
                 for s, arm in peaks.items()},
    }
