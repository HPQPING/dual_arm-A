"""核对高频五次取样优化的数值等价性，不启动仿真或连接设备。"""

import math

import numpy as np
import pytest

from archer_mujoco.joint_interpolation import coefficients, evaluate


@pytest.mark.parametrize('order', (0, 1, 2))
def test_horner_matches_power_sum(order):
    """随机端点及极短段中，位置、速度和加速度保持原多项式定义。"""
    rng = np.random.default_rng(147)
    for duration in (.001, .004, .1, 3., 120.):
        for _ in range(20):
            c = coefficients(*(rng.normal(size=(6, 6))), duration)
            for u in (0., .001, .25, .5, .999, 1.):
                reference = sum(
                    c[:, power] * math.factorial(power) / math.factorial(power - order)
                    * u ** (power - order) for power in range(order, 6)
                ) / duration ** order
                np.testing.assert_allclose(evaluate(c, u, duration, order), reference,
                                           rtol=1e-10, atol=1e-7 if order == 2 else 1e-10)


def test_horner_preserves_endpoint_conditions():
    """完整任务边界的零速零加速度仍满足执行器精确衔接要求。"""
    q0, q1 = np.arange(6.) / 10., np.arange(6.) / 20.
    zero = np.zeros(6)
    c = coefficients(q0, q1, zero, zero, zero, zero, 3.)
    for u, expected in ((0., q0), (1., q1)):
        np.testing.assert_allclose(evaluate(c, u, 3.), expected, atol=1e-14)
        for order in (1, 2):
            np.testing.assert_allclose(evaluate(c, u, 3., order), zero, atol=1e-14)
