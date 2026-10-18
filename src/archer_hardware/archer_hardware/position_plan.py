"""检查关节限制并生成位置轨迹与双臂启动屏障，不连接设备."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
import math
from threading import Lock
import time
from typing import TYPE_CHECKING, Callable, Iterable

from .hex_adapter import six_values

if TYPE_CHECKING:
    from archer_control.trajectory import FloatArray


def positive(value: object, name: str) -> float:
    """校验有限正数，不接受布尔值."""
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f'{name} 必须是有限正数')
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f'{name} 超出浮点数范围') from error
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f'{name} 必须是有限正数')
    return result


@dataclass(frozen=True)
class PositionLimits:
    """独立保存每只臂的限位，调用方必须提供已确认的实机数据."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]
    max_velocity: tuple[float, ...]
    max_acceleration: tuple[float, ...]
    feedback_tolerance: float = 0.0

    def __post_init__(self) -> None:
        """固定六维有限数值，拒绝空限位和无效上下界."""
        for name in ('lower', 'upper', 'max_velocity', 'max_acceleration'):
            object.__setattr__(self, name, six_values(getattr(self, name)))
        if any(a >= b for a, b in zip(self.lower, self.upper)):
            raise ValueError('每个关节的下限必须小于上限')
        if any(v <= 0 for v in self.max_velocity + self.max_acceleration):
            raise ValueError('速度和加速度限制必须为正数')
        tolerance = self.feedback_tolerance
        if (isinstance(tolerance, bool) or not isinstance(tolerance, (int, float))
                or not math.isfinite(tolerance) or not 0 <= tolerance <= 0.01):
            raise ValueError('反馈角度容差必须在 [0, 0.01] rad 内')

    def check_positions(self, values: Iterable[object]) -> tuple[float, ...]:
        """越界立即拒绝，不裁剪实际角度或修改目标位置."""
        positions = six_values(values)
        for index, (q, low, high) in enumerate(zip(positions, self.lower, self.upper)):
            if not low <= q <= high:
                raise ValueError(f'joint_{index + 1} 角度 {q} 超出 [{low}, {high}] rad')
        return positions

    def check_feedback(self, values: Iterable[object]) -> tuple[float, ...]:
        """允许明确容差内的真实读数，不裁剪、不改写编码器反馈."""
        positions = six_values(values)
        for index, (q, low, high) in enumerate(zip(positions, self.lower, self.upper)):
            if not low - self.feedback_tolerance <= q <= high + self.feedback_tolerance:
                raise ValueError(f'joint_{index + 1} 真实角度超出限位及反馈容差')
        return positions

    def check_goal(
        self, start: tuple[float, ...], goal: Iterable[object],
    ) -> tuple[float, ...]:
        """目标必须在名义限位内；越界起步只允许收回越界关节."""
        first = self.check_feedback(start)
        last = self.check_positions(goal)
        if not self.within_nominal(first):
            for index, (q, a, low, high) in enumerate(zip(last, first, self.lower, self.upper)):
                if low <= a <= high and q != a:
                    raise ValueError(f'准备阶段只能收回越界关节，joint_{index + 1} 应保持原位')
        return last

    def within_nominal(self, values: Iterable[object]) -> bool:
        """判断实测值是否已回到名义限位内，不应用初始读取容差."""
        try:
            self.check_positions(values)
            return True
        except (ValueError, TypeError, OverflowError):
            return False

    def return_goal(self, initial: Iterable[object]) -> tuple[float, ...]:
        """保留原始初值，另生成不越过名义边界的归位指令目标."""
        actual = self.check_feedback(initial)
        return tuple(min(high, max(low, q)) for q, low, high in zip(
            actual, self.lower, self.upper,
        ))

    def recovery_goal(self, actual: Iterable[object]) -> tuple[float, ...]:
        """仅收回越界关节，并在对应边界内留出五毫弧度余量."""
        positions = self.check_feedback(actual)
        return tuple(
            low + min(0.005, (high - low) / 2) if q < low else
            high - min(0.005, (high - low) / 2) if q > high else q
            for q, low, high in zip(positions, self.lower, self.upper)
        )


@dataclass(frozen=True)
class JointPositionMove:
    """一段五次时间插值，起终点速度及加速度为零."""

    start: tuple[float, ...]
    goal: tuple[float, ...]
    duration: float

    @classmethod
    def prepare(
        cls, start: Iterable[object], goal: Iterable[object], duration: float,
        limits: PositionLimits, max_excursion: float,
    ) -> JointPositionMove:
        """用解析峰值检查整段轨迹，拒绝过短时长或过大试运行幅度."""
        first = limits.check_feedback(start)
        last = limits.check_goal(first, goal)
        seconds = positive(duration, 'duration')
        excursion = positive(max_excursion, 'max_excursion')
        for index, (a, b, vmax, amax) in enumerate(zip(
            first, last, limits.max_velocity, limits.max_acceleration,
        )):
            distance = abs(b - a)
            if distance > excursion:
                raise ValueError(f'joint_{index + 1} 超出本次允许的最大位移')
            # 五次插值的最大一阶导数为 15/8，最大二阶导数为 10/√3。
            if 1.875 * distance / seconds > vmax:
                raise ValueError(f'joint_{index + 1} 轨迹峰值速度超限，请增加时长')
            if (10.0 / math.sqrt(3.0)) * distance / seconds ** 2 > amax:
                raise ValueError(f'joint_{index + 1} 轨迹峰值加速度超限，请增加时长')
        return cls(first, last, seconds)

    def sample(self, elapsed: float) -> tuple[float, ...]:
        """按已用时间取目标点，超过末端后保持最终目标."""
        if isinstance(elapsed, bool) or not math.isfinite(elapsed):
            raise ValueError('elapsed 必须为有限秒数')
        u = min(1.0, max(0.0, elapsed / self.duration))
        blend = u ** 3 * (10.0 - 15.0 * u + 6.0 * u ** 2)
        return tuple(a + blend * (b - a) for a, b in zip(self.start, self.goal))


@dataclass(frozen=True)
class JointPositionTrajectory:
    """共用纳秒时间轴的多段五次轨迹，每个连接点速度和加速度为零."""

    times_ns: tuple[int, ...]
    segments: tuple[JointPositionMove, ...]

    @property
    def duration(self) -> float:
        """从整数纳秒时间轴换算完整计划秒数."""
        return self.times_ns[-1] / 1e9

    @property
    def goal(self) -> tuple[float, ...]:
        """最后一段目标，用于实际到位检查及完成后的保持."""
        return self.segments[-1].goal

    @classmethod
    def prepare(
        cls, times_ns: Iterable[int], positions: Iterable[Iterable[object]],
        limits: PositionLimits, max_excursion: float, *,
        actual_start: Iterable[object], start_tolerance: float = 0.005,
    ) -> JointPositionTrajectory:
        """比对真实起点后从实际角度起步，重新预检首段及全程约束."""
        times = tuple(times_ns)
        points = tuple(limits.check_positions(row) for row in positions)
        if len(times) < 2 or len(times) != len(points):
            raise ValueError('轨迹时间和位置数量必须相同且至少有两个点')
        if any(type(value) is not int or value < 0 for value in times):
            raise ValueError('轨迹时间必须为非负整数纳秒')
        if times[0] != 0 or any(b <= a for a, b in zip(times, times[1:])):
            raise ValueError('轨迹时间必须从零开始并严格递增')
        if times[-1] > 3600 * 10 ** 9:
            raise ValueError('轨迹总时长不能超过一小时')
        tolerance = positive(start_tolerance, 'start_tolerance')
        if tolerance > 0.01:
            raise ValueError('轨迹起点容差不能超过 0.01 rad')
        excursion = positive(max_excursion, 'max_excursion')
        actual = limits.check_positions(actual_start)
        if max(abs(a - b) for a, b in zip(actual, points[0])) > tolerance:
            raise ValueError('实际初始角度与轨迹起点不符，需要重新规划接近段')
        # 只在起点容差内替换起点，避免执行第一刻跳向记录的旧角度。
        aligned = (actual,) + points[1:]
        for row in aligned:
            if any(abs(q - start) > excursion for q, start in zip(row, actual)):
                raise ValueError('完整轨迹超出相对实际起点的允许位移')
        segments = []
        for index, (first, last, begin, end) in enumerate(zip(
            aligned, aligned[1:], times, times[1:],
        ), start=1):
            try:
                segments.append(JointPositionMove.prepare(
                    first, last, (end - begin) / 1e9, limits, excursion,
                ))
            except ValueError as error:
                raise ValueError(f'第 {index} 段：{error}') from error
        return cls(times, tuple(segments))

    def sample(self, elapsed: float) -> tuple[float, ...]:
        """按绝对已用时间查找分段，落后时不逐点补发积压命令."""
        if isinstance(elapsed, bool) or not math.isfinite(elapsed):
            raise ValueError('elapsed 必须为有限秒数')
        if elapsed <= 0:
            return self.segments[0].start
        if elapsed >= self.duration:
            return self.goal
        index = bisect_right(self.times_ns, int(elapsed * 1e9)) - 1
        return self.segments[index].sample(elapsed - self.times_ns[index] / 1e9)


@dataclass(frozen=True)
class ContinuousPositionTrajectory:
    """已完成段内解析检查的连续轨迹，发送循环仅查段并计算位置."""

    times_ns: tuple[int, ...]
    polynomials: tuple[FloatArray, ...]
    initial: tuple[float, ...]
    goal: tuple[float, ...]
    position_bounds: tuple[tuple[float, float], ...]
    limits: PositionLimits

    @property
    def duration(self) -> float:
        """返回共用时间轴的总秒数."""
        return self.times_ns[-1] / 1e9

    @staticmethod
    def _check_segment(c: FloatArray, duration: float, limits: PositionLimits) -> FloatArray:
        """复用连续规划器的导数实根检查，不用采样代替段内峰值证明."""
        import numpy as np
        from archer_control.continuous_trajectory import _extrema

        if not np.all(np.isfinite(c)):
            raise ValueError('连续多项式包含非有限数')
        bounds = _extrema(c, duration, 0)
        if (np.any(bounds[:, 0] < np.asarray(limits.lower) - 1e-12)
                or np.any(bounds[:, 1] > np.asarray(limits.upper) + 1e-12)):
            raise ValueError('连续段内角度超出名义限位')
        for order, maximum in ((1, limits.max_velocity), (2, limits.max_acceleration)):
            peaks = np.max(np.abs(_extrema(c, duration, order)), axis=1)
            if np.any(peaks > np.asarray(maximum) * (1. + 1e-10)):
                raise ValueError('连续段内速度或加速度超限')
        return bounds

    @classmethod
    def prepare(
        cls, times_ns: Iterable[int], positions: Iterable[Iterable[object]],
        velocities: Iterable[Iterable[object]], accelerations: Iterable[Iterable[object]],
        limits: PositionLimits, max_excursion: float, *,
        cancelled: Callable[[], bool] | None = None,
    ) -> ContinuousPositionTrajectory:
        """离线准备完整多项式并冻结数组，不在周期发送线程中调用本方法."""
        import numpy as np
        from archer_control.continuous_trajectory import coefficients

        times = tuple(times_ns)
        q = tuple(limits.check_positions(p) for p in positions)
        v = tuple(six_values(p) for p in velocities)
        a = tuple(six_values(p) for p in accelerations)
        if (not 2 <= len(times) <= 20000 or any(len(p) != len(times) for p in (q, v, a))
                or any(type(t) is not int for t in times) or times[0] != 0
                or times[-1] > 3600 * 10 ** 9
                or any(y <= x for x, y in zip(times, times[1:]))):
            raise ValueError('连续轨迹需要2到20000个从零开始递增的纳秒时间点，最长一小时')
        if any(abs(x) > 1e-12 for row in (v[0], v[-1], a[0], a[-1]) for x in row):
            raise ValueError('连续轨迹起终点的速度和加速度必须为零')
        excursion = positive(max_excursion, 'max_excursion')
        polys = []
        bounds = np.column_stack((np.full(6, np.inf), np.full(6, -np.inf)))
        for i in range(len(times) - 1):
            if cancelled is not None and cancelled():
                raise ValueError('连续轨迹预检已取消')
            duration = (times[i + 1] - times[i]) / 1e9
            q0, q1, v0, v1, a0, a1 = (
                np.asarray(p, dtype=np.float64)
                for p in (q[i], q[i + 1], v[i], v[i + 1], a[i], a[i + 1])
            )
            c = coefficients(q0, q1, v0, v1, a0, a1, duration)
            local = cls._check_segment(c, duration, limits)
            bounds[:, 0] = np.minimum(bounds[:, 0], local[:, 0])
            bounds[:, 1] = np.maximum(bounds[:, 1], local[:, 1])
            c.setflags(write=False)
            polys.append(c)
        if np.any(np.abs(bounds - np.asarray(q[0])[:, None]) > excursion + 1e-12):
            raise ValueError('连续轨迹段内超出相对起点的允许位移')
        return cls(times, tuple(polys), q[0], q[-1],
                   tuple((float(p[0]), float(p[1])) for p in bounds), limits)

    def aligned_start(
        self, actual_start: Iterable[object], limits: PositionLimits,
        max_excursion: float, tolerance: float = .005,
    ) -> ContinuousPositionTrajectory:
        """核对新鲜实际起点；容差内只重建首段，禁止改写其余路径与时间."""
        import numpy as np
        from archer_control.continuous_trajectory import coefficients, evaluate

        if limits != self.limits:
            raise ValueError('预检限制与当前执行限制不一致')
        actual = limits.check_positions(actual_start)
        tolerance = positive(tolerance, '起点容差')
        if tolerance > .005 or max(abs(x - y) for x, y in zip(actual, self.initial)) > tolerance:
            raise ValueError('实际初始角度不符，需要另行规划接近段')
        excursion = positive(max_excursion, 'max_excursion')
        if any(abs(q - first) > excursion + 1e-12
               for pair, first in zip(self.position_bounds, actual) for q in pair):
            raise ValueError('实际起点对齐后超出允许位移')
        if actual == self.initial:
            return self
        duration = (self.times_ns[1] - self.times_ns[0]) / 1e9
        old = self.polynomials[0]
        c = coefficients(np.asarray(actual, dtype=np.float64), evaluate(old, 1., duration),
                         np.zeros(6), evaluate(old, 1., duration, 1),
                         np.zeros(6), evaluate(old, 1., duration, 2), duration)
        local = self._check_segment(c, duration, limits)
        if np.any(np.abs(local - np.asarray(actual)[:, None]) > excursion + 1e-12):
            raise ValueError('首段对齐后超出允许位移')
        c.setflags(write=False)
        bounds = tuple((min(pair[0], float(row[0])), max(pair[1], float(row[1])))
                       for pair, row in zip(self.position_bounds, local))
        return replace(self, initial=actual, polynomials=(c,) + self.polynomials[1:],
                       position_bounds=bounds)

    def sample(self, elapsed: float, derivative: int = 0) -> tuple[float, ...]:
        """按绝对时间取连续目标，不逐个补发过期采样点，末端保持."""
        from archer_control.continuous_trajectory import evaluate

        if (isinstance(elapsed, bool) or not math.isfinite(elapsed)
                or type(derivative) is not int or derivative not in (0, 1, 2)):
            raise ValueError('连续轨迹时间或导数阶数无效')
        if elapsed <= 0. or elapsed >= self.duration:
            return (0.,) * 6 if derivative else (self.initial if elapsed <= 0. else self.goal)
        i = min(len(self.polynomials) - 1,
                bisect_right(self.times_ns, int(elapsed * 1e9)) - 1)
        duration = (self.times_ns[i + 1] - self.times_ns[i]) / 1e9
        u = min(1., max(0., (elapsed - self.times_ns[i] / 1e9) / duration))
        return tuple(float(x) for x in evaluate(self.polynomials[i], u, duration, derivative))


def prepare_continuous_pair(
    data: object, limits: dict[str, PositionLimits], max_excursion: float, *,
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, ContinuousPositionTrajectory]:
    """严格读取版本二并检查双臂共享时间轴；不创建ROS或SDK连接."""
    if (not isinstance(data, dict) or set(data) != {
        'version', 'interpolation', 'stop_indices', 'left', 'right',
    } or type(data['version']) is not int or data['version'] != 2
            or data['interpolation'] != 'quintic_c2'):
        raise ValueError('需要完整的版本二quintic_c2轨迹')
    result = {}
    for side in ('left', 'right'):
        arm = data[side]
        if (not isinstance(arm, dict) or set(arm) != {'joint_names', 'points'}
                or arm['joint_names'] != [f'{side}_joint_{i}' for i in range(1, 7)]
                or not isinstance(arm['points'], list) or not 2 <= len(arm['points']) <= 20000):
            raise ValueError('连续轨迹关节顺序或点数错误')
        times, positions, velocities, accelerations = [], [], [], []
        for p in arm['points']:
            if cancelled is not None and cancelled():
                raise ValueError('连续轨迹预检已取消')
            if not isinstance(p, dict) or set(p) != {
                'positions', 'velocities', 'accelerations', 'time_from_start',
            }:
                raise ValueError('连续轨迹点需要位置、速度、加速度及时间')
            t = p['time_from_start']
            if (not isinstance(t, dict) or set(t) != {'sec', 'nanosec'}
                    or type(t['sec']) is not int or type(t['nanosec']) is not int
                    or t['sec'] < 0 or not 0 <= t['nanosec'] < 10 ** 9):
                raise ValueError('连续轨迹时间字段无效')
            times.append(t['sec'] * 10 ** 9 + t['nanosec'])
            positions.append(p['positions'])
            velocities.append(p['velocities'])
            accelerations.append(p['accelerations'])
        stops = data['stop_indices']
        if (not isinstance(stops, list) or len(stops) < 2
                or any(type(i) is not int or not 0 <= i < len(times) for i in stops)
                or stops[0] != 0 or stops[-1] != len(times) - 1
                or any(b <= a for a, b in zip(stops, stops[1:]))):
            raise ValueError('连续轨迹停止点索引无效')
        for i in stops:
            if any(abs(x) > 1e-12 for row in (velocities[i], accelerations[i])
                   for x in six_values(row)):
                raise ValueError('停止点速度和加速度必须为零')
        result[side] = ContinuousPositionTrajectory.prepare(
            times, positions, velocities, accelerations, limits[side], max_excursion,
            cancelled=cancelled,
        )
    if result['left'].times_ns != result['right'].times_ns:
        raise ValueError('连续双臂轨迹必须共用纳秒时间轴')
    return result


class PairedTrajectoryStart:
    """两侧准备确认后给出共同单调时刻，取消状态不可恢复."""

    def __init__(self) -> None:
        """准备最多等待0.5秒，两侧就绪后预留0.2秒软件启动窗口."""
        self._lock = Lock()
        self._ready: set[str] = set()
        self._start: float | None = None
        self._cancelled = False
        self._expires = time.monotonic() + 0.5

    def ready(self, side: str) -> None:
        """必须是两个不同侧的确认，不能重复一侧来提前释放."""
        with self._lock:
            if side not in ('left', 'right') or side in self._ready:
                raise ValueError('双臂准备侧别无效或重复')
            if self._cancelled or time.monotonic() > self._expires:
                self._cancelled = True
                raise RuntimeError('双臂准备已取消或超时')
            self._ready.add(side)
            if self._ready == {'left', 'right'}:
                self._start = time.monotonic() + 0.2

    def start_time(self) -> float | None:
        """等待双方就绪，准备超时或已取消时拒绝继续."""
        with self._lock:
            if self._start is None and time.monotonic() > self._expires:
                self._cancelled = True
            if self._cancelled:
                raise RuntimeError('双臂轨迹启动已取消或准备超时')
            return self._start

    def cancel(self) -> None:
        """任何一侧均可取消，已经确定启动时刻也不得恢复."""
        with self._lock:
            self._cancelled = True

    @property
    def cancelled(self) -> bool:
        """供发送出口和状态监视检查，不隐式解除取消."""
        with self._lock:
            return self._cancelled
