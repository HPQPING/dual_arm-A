"""检查关节限制并生成位置轨迹、双臂启动屏障和停止归位曲线，不连接设备."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from threading import Lock
import time
from typing import Iterable

from .hex_adapter import six_values


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
        reference: tuple[float, ...] | None = None,
    ) -> tuple[float, ...]:
        """目标必须在名义限位内；越界起步只允许收回越界关节."""
        first = self.check_feedback(start)
        last = self.check_positions(goal)
        if reference is not None:
            self.check_feedback(reference)
        elif not self.within_nominal(first):
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
        reference: tuple[float, ...] | None = None,
    ) -> JointPositionMove:
        """用解析峰值检查整段轨迹，拒绝过短时长或过大试运行幅度."""
        first = limits.check_feedback(start)
        last = limits.check_goal(first, goal, reference)
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


@dataclass(frozen=True)
class StopAndReturnMove:
    """先按实际速度减速，再返回记录的初始位置，目标速度连续."""

    start: tuple[float, ...]
    velocities: tuple[float, ...]
    deceleration_time: float
    return_move: JointPositionMove

    @property
    def duration(self) -> float:
        """完整减速和归位的计划时长."""
        return self.deceleration_time + self.return_move.duration

    @property
    def goal(self) -> tuple[float, ...]:
        """本轮初始角度对应的名义限位内归位目标."""
        return self.return_move.goal

    @classmethod
    def prepare(
        cls, positions: Iterable[object], velocities: Iterable[object],
        initial: tuple[float, ...], minimum_duration: float, limits: PositionLimits,
    ) -> StopAndReturnMove:
        """检查减速距离及整段约束，必要时延长归位时长."""
        first = limits.check_feedback(positions)
        speeds = six_values(velocities)
        home = limits.return_goal(initial)
        minimum = positive(minimum_duration, '归位最小时长')
        if any(abs(v) > cap for v, cap in zip(speeds, limits.max_velocity)):
            raise ValueError('实际速度超过配置上限，不能按普通停止归位')
        stop_time = max(abs(v) / cap for v, cap in zip(speeds, limits.max_acceleration))
        stop_position = tuple(q + 0.5 * v * stop_time for q, v in zip(first, speeds))
        # 减速的必要停止距离只能落在名义范围加反馈容差以内。
        limits.check_feedback(stop_position)
        duration = max(minimum - stop_time, 1e-6)
        for q, h, vmax, amax in zip(
            stop_position, home, limits.max_velocity, limits.max_acceleration,
        ):
            distance = abs(h - q)
            duration = max(
                duration, 1.875 * distance / vmax,
                math.sqrt((10.0 / math.sqrt(3.0)) * distance / amax),
            )
        # 避免峰值公式在浮点临界点因舍入被判超限。
        duration *= 1.0 + 1e-9
        excursion = max(1e-9, max(abs(h - q) for q, h in zip(stop_position, home)))
        returning = JointPositionMove.prepare(
            stop_position, home, duration, limits, excursion, reference=home,
        )
        return cls(first, speeds, stop_time, returning)

    def sample(self, elapsed: float) -> tuple[float, ...]:
        """减速段采用有界恒加速度，归位段采用五次时间插值."""
        if isinstance(elapsed, bool) or not math.isfinite(elapsed):
            raise ValueError('elapsed 必须为有限秒数')
        t = max(0.0, elapsed)
        if self.deceleration_time > 0 and t < self.deceleration_time:
            factor = t - 0.5 * t * t / self.deceleration_time
            return tuple(q + v * factor for q, v in zip(self.start, self.velocities))
        return self.return_move.sample(t - self.deceleration_time)
