"""协同双J2有界平滑反馈滤波；不导入ROS/SDK或发送指令。"""

from __future__ import annotations

import math


def _six(values: tuple[float, ...]) -> tuple[float, ...]:
    if len(values) != 6 or any(isinstance(x, bool) or not math.isfinite(x) for x in values):
        raise ValueError('需要六个有限数值')
    return tuple(float(x) for x in values)


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


class BoundedPositionFeedback:
    """参考+过滤后的比例误差修正；不积分累积，候选参数不是厂商推荐值。"""

    def __init__(
        self,
        lower: tuple[float, ...],
        upper: tuple[float, ...],
        *,
        enabled_axes: tuple[bool, ...] = (True,) * 6,
        max_velocity: tuple[float, ...] = (0.5,) * 6,
        max_acceleration: tuple[float, ...] = (0.3,) * 6,
    ) -> None:
        self.lower, self.upper = _six(lower), _six(upper)
        self.velocity_limits, self.acceleration_limits = _six(max_velocity), _six(max_acceleration)
        if (
            any(a >= b for a, b in zip(self.lower, self.upper))
            or len(enabled_axes) != 6
            or any(type(x) is not bool for x in enabled_axes)
        ):
            raise ValueError('限位或使能轴无效')
        self.axes = enabled_axes
        self.gain = 0.5
        self.max_offset = 0.02
        self.max_rate = 0.01
        self.max_acceleration = 0.04
        self.max_jerk = 0.2
        self.omega = 4.0
        self.reset()

    def reset(self) -> None:
        self.offset = [0.0] * 6
        self.velocity = [0.0] * 6
        self.acceleration = [0.0] * 6

    def step(
        self,
        reference: tuple[float, ...],
        actual: tuple[float, ...],
        dt: float,
        *,
        enabled: bool = True,
        reference_velocity: tuple[float, ...] = (0.0,) * 6,
        reference_acceleration: tuple[float, ...] = (0.0,) * 6,
    ) -> tuple[float, ...]:
        """仅计算下一指令；调用方仍须核对完整目标V/A、反馈几何、控制权及取消。"""
        reference, actual = _six(reference), _six(actual)
        reference_velocity = _six(reference_velocity)
        reference_acceleration = _six(reference_acceleration)
        if isinstance(dt, bool) or not math.isfinite(dt) or not 0.0 < dt <= 0.04:
            raise ValueError('原型更新间隔需在0到40ms内')
        if any(not low <= q <= high for low, q, high in zip(self.lower, reference, self.upper)):
            raise ValueError('参考角度越界')
        result = []
        for i, q in enumerate(reference):
            rate_limit = min(self.max_rate, self.velocity_limits[i] - abs(reference_velocity[i]))
            accel_limit = min(
                self.max_acceleration, self.acceleration_limits[i] - abs(reference_acceleration[i])
            )
            if rate_limit < 0.0 or accel_limit < 0.0:
                raise ValueError('参考导数已超硬件预算，不能叠加反馈修正')
            low = max(-self.max_offset, self.lower[i] - q)
            high = min(self.max_offset, self.upper[i] - q)
            goal = self.gain * (q - actual[i]) if self.axes[i] and enabled else 0.0
            goal = _clip(goal, low, high)
            requested = (
                self.omega**2 * (goal - self.offset[i]) - 2.0 * self.omega * self.velocity[i]
            )
            requested = _clip(requested, -accel_limit, accel_limit)
            acceleration = _clip(
                requested,
                self.acceleration[i] - self.max_jerk * dt,
                self.acceleration[i] + self.max_jerk * dt,
            )
            acceleration = _clip(acceleration, -accel_limit, accel_limit)
            velocity = _clip(self.velocity[i] + acceleration * dt, -rate_limit, rate_limit)
            proposed = self.offset[i] + velocity * dt
            offset = _clip(proposed, low, high)
            if offset != proposed:
                # 碰到名义范围/输出饱和时不保留朝范围外累积的状态。
                velocity = acceleration = 0.0
            self.offset[i], self.velocity[i], self.acceleration[i] = offset, velocity, acceleration
            result.append(q + offset)
        return tuple(result)
