"""离线核对接近归位提速、边界连续和圆弧保留，不连接设备."""

from collections.abc import Mapping
from dataclasses import replace
import unittest

import numpy as np
from numpy.typing import ArrayLike

from archer_control.continuous_trajectory import ContinuousPair, circle_blend_windows
from archer_control.transit_timing import optimize_transit_timing
from archer_mujoco.joint_interpolation import coefficients, _extrema


class TransitTimingTests(unittest.TestCase):
    """用可解析的合成双臂路径检查局部时轴处理，不把测试参数用于真机."""

    def setUp(self) -> None:
        """生成二十秒往返；阶段边界已连续通过，只有整轮起止停车."""
        self.t = np.linspace(0., 20., 201)
        u = np.where(self.t <= 10., self.t / 10., (20. - self.t) / 10.)
        q = .1 * u**3 * (10. - 15. * u + 6. * u**2)
        v = .01 * 30. * u**2 * (1. - u)**2 * np.where(self.t <= 10., 1., -1.)
        a = .001 * 60. * u * (1. - u) * (1. - 2. * u)
        arrays = [{s: np.repeat((values * factor)[:, None], 6, axis=1)
                   for s, factor in (('left', 1.), ('right', .8))} for values in (q, v, a)]
        positions, velocities, accelerations = arrays
        polys = {s: np.stack([
            coefficients(positions[s][i], positions[s][i + 1], velocities[s][i],
                         velocities[s][i + 1], accelerations[s][i], accelerations[s][i + 1],
                         float(h)) for i, h in enumerate(np.diff(self.t))]) for s in positions}
        self.plan = ContinuousPair(self.t, *arrays, polys, (0, 200))
        self.stages = {s: ['APPROACH'] * 70 + ['ARC'] * 60 + ['RETURN_HOME'] * 71
                       for s in positions}
        self.lower, self.upper = np.full((2, 6), -1.), np.ones((2, 6))
        self.speed, self.accel = np.full((2, 6), .1), np.full((2, 6), .1)

    def optimize(
        self, plan: ContinuousPair | None = None, *, max_jerk: float = 2.,
        source: Mapping[str, ArrayLike] | None = None,
    ):
        """固定测试约束，可显式覆盖以检查拒绝行为."""
        return optimize_transit_timing(
            self.plan if plan is None else plan, self.stages, self.lower, self.upper,
            self.speed, self.accel, max_jerk=max_jerk,
            source=self.plan.positions if source is None else source)

    def test_faster_transit_preserves_arc_and_blends(self) -> None:
        """两段更快，圆弧及融合窗口的节点状态、系数和纳秒段时长完全保留."""
        original = {s: self.plan.positions[s].copy() for s in ('left', 'right')}
        new, report = self.optimize()
        self.assertEqual({b['stage'] for b in report['blocks']}, {'APPROACH', 'RETURN_HOME'})
        self.assertLess(new.times[-1], self.plan.times[-1])
        windows = circle_blend_windows(self.t, self.stages, (0, 69, 70, 129, 130, 200))
        first, last = windows[0]['first'], windows[1]['last']
        old_ns = np.rint(self.plan.times * 1e9).astype(np.int64)
        new_ns = np.rint(new.times * 1e9).astype(np.int64)
        np.testing.assert_array_equal(np.diff(old_ns)[first:last], np.diff(new_ns)[first:last])
        for side in ('left', 'right'):
            for field in ('positions', 'velocities', 'accelerations'):
                np.testing.assert_array_equal(getattr(new, field)[side][first:last + 1],
                                              getattr(self.plan, field)[side][first:last + 1])
            np.testing.assert_array_equal(new.polynomials[side][first:last],
                                          self.plan.polynomials[side][first:last])
            np.testing.assert_array_equal(self.plan.positions[side], original[side])
            self.assertFalse(new.positions[side].flags.writeable)
            for c, h in zip(new.polynomials[side], np.diff(new.times)):
                self.assertLessEqual(np.max(abs(_extrema(c, float(h), 1))), .1 + 1e-10)
                self.assertLessEqual(np.max(abs(_extrema(c, float(h), 2))), .1 + 1e-10)
                self.assertLessEqual(np.max(abs(_extrema(c, float(h), 3))), 2. + 1e-8)
        for block in report['blocks']:
            self.assertLess(block['after_s'], block['before_s'])
            for values in block['jerk_jumps'].values():
                self.assertLessEqual(values['edge_change'], 1e-5)
                self.assertLessEqual(values['after'], max(1e-5, values['before'] * 1.1))

    def test_extra_stops_and_source_errors_are_rejected(self) -> None:
        """不删除其他停点，不接受原始节点缺失或非有限读数."""
        with self.assertRaisesRegex(ValueError, '融合'):
            self.optimize(replace(self.plan, stops=(0, 50, 200)))
        with self.assertRaisesRegex(ValueError, '原始节点'):
            self.optimize(source={'left': self.plan.positions['left']})
        bad = {s: q.copy() for s, q in self.plan.positions.items()}
        bad['left'][2, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, '原始节点'):
            self.optimize(source=bad)

    def test_actual_holds_are_not_compressed(self) -> None:
        """源CSV的静止区间不能因拟合曲线略动而被误当成可压缩运动."""
        raw = {s: q.copy() for s, q in self.plan.positions.items()}
        for q in raw.values():
            q[5] = q[4]
        with self.assertRaisesRegex(ValueError, '保持'):
            self.optimize(source=raw)

    def test_infeasible_derivative_limit_keeps_input(self) -> None:
        """约束太紧时拒绝，不放宽上限或修改原始时间轴."""
        saved = self.plan.times.copy()
        with self.assertRaisesRegex(ValueError, '更快'):
            self.optimize(max_jerk=1e-9)
        np.testing.assert_array_equal(self.plan.times, saved)

    def test_invalid_jerk_is_rejected(self) -> None:
        """布尔、非有限和非正jerk不能作为规划参数."""
        for value in (True, float('nan'), 0., -1.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.optimize(max_jerk=value)
