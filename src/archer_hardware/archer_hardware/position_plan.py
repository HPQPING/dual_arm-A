"""检查关节限制并生成位置轨迹与双臂启动屏障，不连接设备."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, replace
import math
from threading import Lock
import time
from typing import TYPE_CHECKING, Callable, Iterable

from .hex_adapter import six_values


MAX_TASK_JOINT_ERROR_RAD = .05  # 完整文件观察运行的最大偏差范围，不改变关节限位。

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


def limit_preparation_goal(
    limits: PositionLimits, actual: Iterable[object],
) -> tuple[float, ...]:
    """先仅收回超限轴；全部入限后，仅为距边界不足三毫弧度的轴留余量."""
    start = limits.check_feedback(actual)
    outside = not limits.within_nominal(start)
    result = []
    for q, low, high in zip(start, limits.lower, limits.upper):
        reserve = min(.01, (high - low) / 4.)
        required = 0. if outside else min(.003, (high - low) / 8.)
        result.append(low + reserve if q < low + required else
                      high - reserve if q > high - required else q)
    return limits.check_goal(start, result)


def limit_preparation_complete(
    limits: PositionLimits, start: Iterable[object], goal: Iterable[object],
    actual: Iterable[object],
) -> bool:
    """入限准入独立于末点精度：实际不越界、变化轴有余量且目标偏差不超五毫弧度."""
    first = limits.check_feedback(start)
    target = limits.check_goal(first, goal)
    try:
        observed = limits.check_positions(actual)
    except ValueError:
        return False
    if any(abs(q - g) > .005 for q, g in zip(observed, target)):
        return False
    return all(
        min(q - low, high - q) >= min(.003, (high - low) / 8.)
        for q, old, g, low, high in zip(
            observed, first, target, limits.lower, limits.upper)
        if old != g
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


def checked_joint_error(value: object, feedback_tolerance: float, *,
                        nominal_feedback_guarded: bool = False) -> float:
    """完整文件逐帧检查名义限位时，可显式采用最多0.05rad的跟踪误差界限。"""
    maximum = MAX_TASK_JOINT_ERROR_RAD if nominal_feedback_guarded else min(.01, feedback_tolerance)
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0. <= value <= maximum):
        raise ValueError('joint_error_rad需有限非负且不超过0.05；未启用名义限位保护时不能超过读取容差')
    return float(value)


def checked_startup_error(value: object, body_error: float) -> float:
    """只有启用了主体误差保护，才接受不大于0.05rad的独立接近段容差。"""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0. <= value <= MAX_TASK_JOINT_ERROR_RAD
            or value and (not body_error or value < body_error)):
        raise ValueError('起始容差需为0或主体误差到0.05rad之间的有限数')
    return float(value)


@dataclass(frozen=True)
class ContinuousPositionTrajectory:
    """已完成段内解析检查的连续轨迹，发送循环仅查段并计算位置."""

    times_ns: tuple[int, ...]
    polynomials: tuple[FloatArray, ...]
    initial: tuple[float, ...]
    goal: tuple[float, ...]
    position_bounds: tuple[tuple[float, float], ...]
    limits: PositionLimits
    preserve_path: bool = False
    joint_error_rad: float = 0.
    startup_joint_error_rad: float = 0.
    startup_duration_s: float = 0.

    def tracking_error_at(self, elapsed: float) -> float:
        """接近段可使用独立已检容差，跨入文件主体立即恢复主体标准。"""
        return (self.startup_joint_error_rad if self.startup_duration_s > 0.
                and elapsed < self.startup_duration_s else self.joint_error_rad)

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
        """完整路径仅核对有界反馈且保留原多项式；普通路径沿用首段对齐."""
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
        error = checked_joint_error(self.joint_error_rad, limits.feedback_tolerance,
                                    nominal_feedback_guarded=self.preserve_path)
        if self.startup_joint_error_rad or self.startup_duration_s:
            if (not self.preserve_path or not error
                    or isinstance(self.startup_joint_error_rad, bool)
                    or not error <= self.startup_joint_error_rad <= MAX_TASK_JOINT_ERROR_RAD
                    or isinstance(self.startup_duration_s, bool)
                    or self.startup_duration_s != self.times_ns[1] / 1e9):
                raise ValueError('起始容差只能覆盖完整路径的首个接近段且不超过0.05rad')
        if error and not self.preserve_path:
            raise ValueError('关节误差保护只适用于已检完整路径')
        if self.preserve_path:
            if max(abs(a - b) for a, b in zip(actual, self.initial)) > error:
                raise ValueError('已做几何预检的完整路径起点发生变化，需要重新规划和预检，禁止重建首段')
            return self
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


def delivery_reference_step(plans: dict[str, ContinuousPositionTrajectory]) -> float:
    """离线按全路径解析速度峰值计算单次参考推进量，不扩大SDK限速或重拟合路径."""
    import numpy as np
    from archer_control.continuous_trajectory import _extrema

    if set(plans) != {'left', 'right'}:
        raise ValueError('发送节拍需要完整双臂计划')
    ratio = 0.
    for plan in plans.values():
        for i, polynomial in enumerate(plan.polynomials):
            duration = (plan.times_ns[i + 1] - plan.times_ns[i]) / 1e9
            peaks = np.max(np.abs(_extrema(polynomial, duration, 1)), axis=1)
            ratio = max(ratio, float(np.max(peaks / np.asarray(plan.limits.max_velocity))))
    # 对250/500Hz均按更短的2ms校验周期预留20%编码余量。
    # 实际推进同时受单调时间约束，不能因为峰值很小而提前播放。
    return min(.02, .8 / 500. / ratio) if ratio else .02


class PairedTrajectoryStart:
    """两侧准备确认后给出共同单调时刻，取消状态不可恢复."""

    def __init__(self, *, reference_step_s: float | None = None,
                 duration_s: float | None = None, following_feedback: bool = False,
                 stream_period_s: float | None = None) -> None:
        """准备最多等待0.5秒，两侧就绪后预留0.2秒软件启动窗口."""
        self._lock = Lock()
        self._ready: set[str] = set()
        self._start: float | None = None
        self._cancelled = False
        self._expires = time.monotonic() + 0.5
        if (reference_step_s is None) != (duration_s is None):
            raise ValueError('发送确认时钟需要同时提供推进上限与总时长')
        self._step = (None if reference_step_s is None else
                      positive(reference_step_s, '参考时间推进上限'))
        self._duration = 0. if duration_s is None else positive(duration_s, '参考路径时长')
        if self._step is not None and self._step > .02:
            raise ValueError('参考时间单次推进不能大于20ms')
        self._reference = 0.
        self._generation = 0
        self._acknowledged: set[str] = set()
        self._advanced_at: float | None = None
        self._end_delivered_at: float | None = None
        self._delivered_reference = 0.
        if type(following_feedback) is not bool or following_feedback and self._step is None:
            raise ValueError('实际跟踪门控只能用于完整文件参考时钟')
        self.following_feedback = following_feedback
        self._stream_period = (None if stream_period_s is None else
                               positive(stream_period_s, '共同目标更新周期'))
        if self._stream_period is not None and (
                self._step is None or following_feedback or self._stream_period > .02):
            raise ValueError('周期播放需要完整轨迹时钟且不启用逐点跟踪等待，周期最多20ms')
        self._stream_seen = {'left': -1, 'right': -1}
        self._stream_sent_reference = {'left': 0., 'right': 0.}
        self._stream_references = {0: 0.}
        self._stream_tick = 0
        self._feedback_seen: dict[str, float] = {}
        self._delivery_seen: dict[str, float] = {}
        self._tracking_ratios: dict[str, float] = {}
        self._tracking_generation: dict[str, int] = {}
        self._tracking_paused = False
        self._tracking_since: float | None = None
        self._tracking_waits = 0

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
                self._advanced_at = self._start
                self._delivery_seen = {'left': self._start, 'right': self._start}

    @property
    def delivery_paced(self) -> bool:
        """只有完整文件显式安装参考时钟，原单点和逐点接口保持原行为."""
        return self._step is not None

    def playback(self, side: str, delivered_generation: int | None, *,
                 tracking_ratio: Callable[[float], float] | None = None,
                 ) -> tuple[float, int, float | None]:
        """完整文件按共同周期推进；旧接口仍可显式选择发送确认时钟。"""
        now = time.monotonic()
        with self._lock:
            if (not self.delivery_paced or side not in ('left', 'right')
                    or self._start is None or now < self._start or self._cancelled):
                raise RuntimeError('共用参考时钟未启动或已经取消')
            assert self._advanced_at is not None and self._step is not None
            if self._stream_period is not None:
                return self._stream_playback(side, delivered_generation, now)
            if (self._end_delivered_at is None and now - self._advanced_at > .04
                    and (not self.following_feedback
                         or self._acknowledged != {'left', 'right'})):
                self._cancelled = True
                raise RuntimeError('双臂参考目标超过40ms未共同确认发送')
            if self.following_feedback and self._end_delivered_at is None:
                if any(now - seen > .04 for seen in
                       (*self._feedback_seen.values(), *self._delivery_seen.values())):
                    self._cancelled = True
                    raise RuntimeError('跟踪等待期间反馈调度或发送确认超过40ms未更新')
                if self._tracking_since is not None and now - self._tracking_since > 10.:
                    self._cancelled = True
                    raise RuntimeError('实际跟踪单次等待超过10秒，取消双臂路径')
                try:
                    if tracking_ratio is None:
                        raise ValueError('完整路径缺少实际跟踪检查')
                    ratio = tracking_ratio(self._reference)
                    if (isinstance(ratio, bool) or not isinstance(ratio, (int, float))
                            or not math.isfinite(ratio) or not 0. <= ratio <= 1.):
                        raise RuntimeError('实际跟踪偏差超过已检界限或反馈比例无效')
                except Exception:
                    self._cancelled = True
                    raise
                self._feedback_seen[side] = now
                self._tracking_ratios[side] = ratio
                self._tracking_generation[side] = self._generation
            if delivered_generation is not None:
                if (type(delivered_generation) is not int or delivered_generation < 0
                        or delivered_generation > self._generation):
                    raise ValueError('发送确认代次无效')
                if delivered_generation == self._generation:
                    self._acknowledged.add(side)
                    self._delivery_seen[side] = now
            if self._acknowledged == {'left', 'right'}:
                self._delivered_reference = self._reference
                if self._reference >= self._duration:
                    if self._end_delivered_at is None:
                        self._end_delivered_at = now
                elif now > self._advanced_at:
                    tracking_scale = 1.
                    if self.following_feedback:
                        if any(self._tracking_generation.get(s) != self._generation
                               for s in ('left', 'right')):
                            return self._reference, self._generation, None
                        worst = max(self._tracking_ratios.values())
                        # 容差80%以内按原节奏，80%至95%逐步放慢；不要求逐点精确到位。
                        tracking_scale = min(1., max(0., (.95 - worst) / .15))
                        if not self._tracking_paused and worst >= .95:
                            self._tracking_paused = True
                            self._tracking_since = now
                            self._tracking_waits += 1
                        if self._tracking_paused:
                            if worst >= .95:
                                return self._reference, self._generation, None
                            self._tracking_paused = False
                            self._tracking_since = None
                            # 等待时间作废；恢复这一拍不把等待欠账变成大目标增量。
                            self._advanced_at = now
                            return self._reference, self._generation, None
                    self._reference = min(self._duration, self._reference + min(
                        self._step, now - self._advanced_at) * tracking_scale)
                    self._advanced_at = now
                    self._generation += 1
                    self._acknowledged.clear()
                    self._tracking_generation.clear()
            age = None if self._end_delivered_at is None else now - self._end_delivered_at
            return self._reference, self._generation, age

    def _stream_playback(
        self, side: str, delivered_generation: int | None, now: float,
    ) -> tuple[float, int, float | None]:
        """锁内推进共同周期；发送确认只监视失联及末包，不逐点阻塞进度。"""
        assert self._stream_period is not None and self._advanced_at is not None
        assert self._step is not None
        if self._end_delivered_at is None and (
                now - self._advanced_at > .04
                or any(now - seen > .04 for seen in self._delivery_seen.values())):
            self._cancelled = True
            raise RuntimeError('双臂周期更新或实际发送确认超过40ms，取消后续目标')
        if delivered_generation is not None:
            if (type(delivered_generation) is not int or delivered_generation < 0
                    or delivered_generation > self._generation):
                raise ValueError('发送确认代次无效')
            if delivered_generation > self._stream_seen[side]:
                self._stream_seen[side] = delivered_generation
                self._delivery_seen[side] = now
                self._stream_sent_reference[side] = self._stream_references[delivered_generation]
        self._delivered_reference = min(self._stream_sent_reference.values())
        if self._reference >= self._duration:
            if (self._end_delivered_at is None
                    and all(g == self._generation for g in self._stream_seen.values())):
                self._end_delivered_at = now
        else:
            assert self._start is not None
            # 从共同起点计算整数周期，避免长期累加浮点周期后把准时一拍
            # 错判成早到，随后又丢弃一拍时间，造成持续减速及零增量采样。
            tick = math.floor((now - self._start + 1e-9) / self._stream_period)
            periods = tick - self._stream_tick
            if periods > 0:
                # 两线程落在同一周期时只推进一次。偶发迟到最多推进已检步长，
                # 作废剩余欠账，不能在下一拍集中补发积压目标。
                # 按共同时间轴采样；发送确认只用于失联和末包判断。
                # 不把未用满的一小段发送预算切成额外的微小目标或停顿。
                # SDK另按实际包间隔检查速度，不以漏过缓存的数量拉慢路径。
                next_reference = min(
                    self._duration, self._reference + min(
                        self._step, periods * self._stream_period),
                )
                self._stream_tick = tick
                self._advanced_at = self._start + tick * self._stream_period
                if next_reference > self._reference:
                    self._reference = next_reference
                    self._generation += 1
                    self._stream_references[self._generation] = self._reference
        minimum = min(self._stream_seen.values())
        self._stream_references = {g: t for g, t in self._stream_references.items()
                                   if g >= minimum}
        age = None if self._end_delivered_at is None else now - self._end_delivered_at
        return self._reference, self._generation, age

    def playback_status(self) -> dict[str, object]:
        """状态只报告软件发送确认，不宣称控制器确认或机械臂实际到位."""
        with self._lock:
            return {'enabled': self.delivery_paced, 'reference_elapsed_s': self._reference,
                    'delivered_reference_elapsed_s': self._delivered_reference,
                    'reference_step_cap_s': self._step, 'generation': self._generation,
                    'end_transport_delivered': self._end_delivered_at is not None,
                    'following_feedback': self.following_feedback,
                    'stream_period_s': self._stream_period,
                    'tracking_paused': self._tracking_paused,
                    'tracking_waits': self._tracking_waits,
                    'tracking_error_ratio': dict(self._tracking_ratios),
                    'cancelled': self._cancelled}

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
