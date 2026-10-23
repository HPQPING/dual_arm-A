"""离线核对接近归位提速、边界连续和圆弧保留，不连接设备."""

from collections.abc import Mapping
from dataclasses import replace
import unittest

import numpy as np
from numpy.typing import ArrayLike

from archer_control.continuous_trajectory import ContinuousPair, circle_blend_windows
from archer_control.transit_timing import (
    optimize_transit_timing, _jerk_bounds, replan_circle_return,
)
from archer_mujoco.joint_interpolation import coefficients, _extrema


class TransitTimingTests(unittest.TestCase):
    """用可解析的合成双臂路径检查局部时轴处理，不把测试参数用于真机."""

    def setUp(self) -> None:
        """生成二十点三秒往返；阶段边界已连续通过，只有整轮起止停车."""
        self.t = np.linspace(0., 20.3, 201)
        u = np.where(self.t <= 10.15, self.t / 10.15, (20.3 - self.t) / 10.15)
        q = .1 * u**3 * (10. - 15. * u + 6. * u**2)
        v = (.1 / 10.15) * 30. * u**2 * (1. - u)**2 * np.where(self.t <= 10.15, 1., -1.)
        a = (.1 / 10.15**2) * 60. * u * (1. - u) * (1. - 2. * u)
        arrays = [{s: np.repeat((values * factor)[:, None], 6, axis=1)
                   for s, factor in (('left', 1.), ('right', .8))} for values in (q, v, a)]
        positions, velocities, accelerations = arrays
        polys = {s: np.stack([
            coefficients(positions[s][i], positions[s][i + 1], velocities[s][i],
                         velocities[s][i + 1], accelerations[s][i], accelerations[s][i + 1],
                         float(h)) for i, h in enumerate(np.diff(self.t))]) for s in positions}
        self.plan = ContinuousPair(self.t, positions, velocities, accelerations, polys, (0, 200))
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

    def test_batched_jerk_extrema_matches_polynomial_root_reference(self) -> None:
        """批量提前筛选与原实根检查一致，覆盖常数、线性和区间外顶点。"""
        rng = np.random.default_rng(163)
        pc = rng.normal(size=(24, 6, 6))
        pc[:6, :, 5] = 0.
        pc[:3, :, 4] = 0.
        gaps = np.geomspace(.004, 4., len(pc))
        expected = np.stack([_extrema(c, float(h), 3) for c, h in zip(pc, gaps)])
        np.testing.assert_allclose(_jerk_bounds(pc, gaps), expected, rtol=1e-12, atol=1e-8)

    def test_return_profile_preserves_approach_arc_and_boundary_states(self) -> None:
        """只调整出口窗口之后的归位，保留完整前缀和起止状态，并解析检查jerk."""
        # 合成阶段与动作边界对齐，归位保留足够多的原始节点。
        times = np.arange(1007, dtype=np.int64)*50_000_000/1e9
        approach = np.clip(times/10., 0., 1.)
        home = np.clip((times-20.)/30.3, 0., 1.)
        home[-1] = 1.  # 合成终点明确为零速度和零加速度。

        def blend(u):
            return u**3*(10.-15.*u+6.*u**2)

        q1 = np.where(times < 10., .1*blend(approach), .1*(1.-blend(home)))
        v1 = np.where(times < 10., .1/10.*30.*approach**2*(1.-approach)**2,
                      -.1/30.3*30.*home**2*(1.-home)**2)
        a1 = np.where(times < 10., .1/10.**2*60.*approach*(1.-approach)*(1.-2.*approach),
                      -.1/30.3**2*60.*home*(1.-home)*(1.-2.*home))
        arrays = [{side: np.repeat((factor*x)[:, None], 6, axis=1)
                   for side, factor in (('left', 1.), ('right', .8))} for x in (q1, v1, a1)]
        q, v, a = arrays
        polynomials = {side: np.stack([
            coefficients(q[side][i], q[side][i+1], v[side][i], v[side][i+1],
                         a[side][i], a[side][i+1], float(h))
            for i, h in enumerate(np.diff(times))]) for side in q}
        plan = ContinuousPair(times, q, v, a, polynomials, (0, len(times)-1))
        stages = {side: ['APPROACH' if t < 10. else 'ARC' if t < 20. else 'RETURN_HOME'
                         for t in times] for side in q}
        saved = plan.positions['left'].copy()
        result, report = replan_circle_return(
            plan, stages, self.lower, self.upper, self.speed, self.accel,
            max_jerk=.8, source=plan.positions)
        first = report['first']
        self.assertLess(report['after_s'], report['before_s'])
        self.assertLess(result.times[-1], plan.times[-1])
        np.testing.assert_array_equal(result.times[:first+1], plan.times[:first+1])
        for side in ('left', 'right'):
            for field in ('positions', 'velocities', 'accelerations'):
                np.testing.assert_array_equal(getattr(result, field)[side][:first+1],
                                              getattr(plan, field)[side][:first+1])
                np.testing.assert_array_equal(getattr(result, field)[side][-1],
                                              getattr(plan, field)[side][-1])
            np.testing.assert_array_equal(result.polynomials[side][:first],
                                          plan.polynomials[side][:first])
            self.assertLessEqual(np.max(abs(result.positions[side]-plan.positions[side])),
                                 .004)
            self.assertLessEqual(np.max(abs(_jerk_bounds(result.polynomials[side][first:],
                                                         np.diff(result.times)[first:]))), .8)
            self.assertFalse(result.positions[side].flags.writeable)
        np.testing.assert_array_equal(plan.positions['left'], saved)

    def test_return_profile_rejects_extra_stops_and_actual_waits(self) -> None:
        """新的同步归位不能删除停车或把真实等待重新规划成动作."""
        with self.assertRaisesRegex(ValueError, '完整画圆'):
            replan_circle_return(replace(self.plan, stops=(0, 50, 200)), self.stages,
                                 self.lower, self.upper, self.speed, self.accel,
                                 max_jerk=.8, source=self.plan.positions)
        raw = {s: q.copy() for s, q in self.plan.positions.items()}
        for q in raw.values():
            q[-4] = q[-5]
        with self.assertRaisesRegex(ValueError, '实际保持'):
            replan_circle_return(self.plan, self.stages, self.lower, self.upper,
                                 self.speed, self.accel, max_jerk=.8, source=raw)

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
