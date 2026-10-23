"""公共关节五次插值与解析峰值计算，不导入ROS或连接设备."""

from __future__ import annotations

import math
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def coefficients(
    q0: FloatArray,
    q1: FloatArray,
    v0: FloatArray,
    v1: FloatArray,
    a0: FloatArray,
    a1: FloatArray,
    duration: float,
) -> FloatArray:
    """构造归一化段时间上的五次多项式，约束两端位置、速度和加速度."""
    c = np.zeros((6, 6), dtype=np.float64)
    c[:, 0], c[:, 1], c[:, 2] = q0, v0 * duration, a0 * duration**2 / 2.0
    delta = q1 - c[:, 0] - c[:, 1] - c[:, 2]
    speed = v1 * duration - c[:, 1] - 2.0 * c[:, 2]
    accel = a1 * duration**2 - 2.0 * c[:, 2]
    c[:, 3] = 10.0 * delta - 4.0 * speed + 0.5 * accel
    c[:, 4] = -15.0 * delta + 7.0 * speed - accel
    c[:, 5] = 6.0 * delta - 3.0 * speed + 0.5 * accel
    return c


def evaluate(c: FloatArray, u: float, duration: float, derivative: int = 0) -> FloatArray:
    """计算一段的关节位置或时间导数；输入进度范围为零到一."""
    if derivative not in (0, 1, 2) or not math.isfinite(u) or not 0.0 <= u <= 1.0:
        raise ValueError("进度或导数阶数无效")
    # Horner形式与原五次多项式完全相同，减少250/500Hz取样中的临时数组与幂运算。
    if derivative == 0:
        result = ((((c[:, 5] * u + c[:, 4]) * u + c[:, 3]) * u
                   + c[:, 2]) * u + c[:, 1]) * u + c[:, 0]
    elif derivative == 1:
        result = (((5. * c[:, 5] * u + 4. * c[:, 4]) * u
                   + 3. * c[:, 3]) * u + 2. * c[:, 2]) * u + c[:, 1]
    else:
        result = ((20. * c[:, 5] * u + 12. * c[:, 4]) * u
                  + 6. * c[:, 3]) * u + 2. * c[:, 2]
    return np.asarray(result / duration**derivative, dtype=np.float64)


def _extrema(c: FloatArray, duration: float, derivative: int) -> FloatArray:
    """检查端点及导数实根，不用离散采样替代关节约束峰值检查."""
    result = np.zeros((6, 2), dtype=np.float64)
    for joint in range(6):
        polynomial = np.asarray(
            [
                c[joint, power] * math.factorial(power) / math.factorial(power - derivative)
                for power in range(derivative, 6)
            ],
            dtype=np.float64,
        )
        slope = np.asarray(
            [i * polynomial[i] for i in range(1, len(polynomial))], dtype=np.float64
        )
        roots = (
            np.polynomial.polynomial.polyroots(np.trim_zeros(slope, "b")) if np.any(slope) else []
        )
        candidates = [0.0, 1.0] + [
            min(1.0, max(0.0, float(root.real)))
            for root in roots
            if abs(root.imag) <= 1e-8 and -1e-10 <= root.real <= 1.0 + 1e-10
        ]
        values = [
            float(np.polynomial.polynomial.polyval(u, polynomial)) / duration**derivative
            for u in candidates
        ]
        result[joint] = min(values), max(values)
    return result
