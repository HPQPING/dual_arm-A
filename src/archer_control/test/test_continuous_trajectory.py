"""离线验证连续插值、整段约束及候选格式，不连接设备."""

import unittest

import numpy as np

from archer_control.continuous_trajectory import (
    coefficients, evaluate, plan_continuous_pair, retime_recorded_pair, smooth_recorded_pair,
)


class ContinuousTrajectoryTests(unittest.TestCase):
    """使用虚构数据验证数学与格式边界，不作为实机任务数据."""

    def setUp(self) -> None:
        """构造接近段和单调圆弧示意关节路径."""
        self.q = np.repeat(np.array([0.0, 0.1, 0.12, 0.15, 0.18, 0.2])[:, None], 6, axis=1)
        self.low, self.high = -np.ones((2, 6)), np.ones((2, 6))
        self.speed, self.accel = np.full((2, 6), 0.05), np.full((2, 6), 0.1)

    def plan(self):
        """使用统一虚构约束生成候选."""
        return plan_continuous_pair(
            self.q, self.q, self.low, self.high, self.speed, self.accel, stops=(0, 1, 5)
        )

    def test_internal_velocity_and_c2_continuity(self) -> None:
        """内部路径点不停顿，全部连接处位置、速度和加速度连续."""
        plan, _ = self.plan()
        self.assertTrue(np.any(np.abs(plan.velocities["left"][2:-1]) > 1e-5))
        for side in ("left", "right"):
            for index in range(1, len(plan.times) - 1):
                before = float(plan.times[index] - plan.times[index - 1])
                after = float(plan.times[index + 1] - plan.times[index])
                for derivative in (0, 1, 2):
                    np.testing.assert_allclose(
                        evaluate(plan.polynomials[side][index - 1], 1.0, before, derivative),
                        evaluate(plan.polynomials[side][index], 0.0, after, derivative),
                        atol=1e-12,
                        rtol=0.0,
                    )
            for index in plan.stops:
                np.testing.assert_array_equal(plan.velocities[side][index], np.zeros(6))
                np.testing.assert_array_equal(plan.accelerations[side][index], np.zeros(6))

    def test_dense_derivatives_stay_within_limits(self) -> None:
        """独立密集取样核对解析峰值保护，不能只检查各路径点."""
        plan, report = self.plan()
        for elapsed in np.linspace(0.0, plan.times[-1], 1001):
            self.assertLessEqual(
                float(np.max(np.abs(plan.sample("left", float(elapsed), 1)))), 0.05
            )
            self.assertLessEqual(
                float(np.max(np.abs(plan.sample("left", float(elapsed), 2)))), 0.1
            )
        self.assertLessEqual(max(report["arms"]["right"]["peak_velocity_rad_s"]), 0.05)
        self.assertLessEqual(max(report["arms"]["right"]["peak_acceleration_rad_s2"]), 0.1)

    def test_tighter_right_arm_shared_time_and_unchanged_input(self) -> None:
        """右臂更严格的限制控制共享时长，不修改输入关节角."""
        before = self.q.copy()
        normal, _ = self.plan()
        self.speed[1] *= 0.2
        slower, _ = self.plan()
        self.assertGreater(slower.times[-1], normal.times[-1])
        np.testing.assert_array_equal(before, self.q)
        np.testing.assert_array_equal(slower.positions["right"], self.q)
        self.assertFalse(slower.times.flags.writeable)

    def test_payload_rebuild_and_hold_endpoint(self) -> None:
        """导出纳秒时间和显式导数能重建同一多项式，完成后保持最终位置."""
        plan, _ = self.plan()
        payload = plan.payload()
        self.assertEqual(payload["version"], 2)
        self.assertEqual(payload["interpolation"], "quintic_c2")
        self.assertEqual(payload["stop_indices"], [0, 1, 5])
        for side in ("left", "right"):
            points = payload[side]["points"]
            for index, (a, b) in enumerate(zip(points, points[1:])):
                stamps = [
                    p["time_from_start"]["sec"] + p["time_from_start"]["nanosec"] / 1e9
                    for p in (a, b)
                ]
                q0, q1, v0, v1, a0, a1 = [
                    np.asarray(p[key], dtype=np.float64)
                    for key in ("positions", "velocities", "accelerations")
                    for p in (a, b)
                ]
                c = coefficients(q0, q1, v0, v1, a0, a1, stamps[1] - stamps[0])
                np.testing.assert_allclose(c, plan.polynomials[side][index], atol=1e-12, rtol=0.0)
            np.testing.assert_array_equal(
                plan.sample(side, float(plan.times[-1]) + 5.0), self.q[-1]
            )
            np.testing.assert_array_equal(plan.sample(side, -1.0), self.q[0])
            np.testing.assert_array_equal(plan.sample(side, float(plan.times[-1]), 1), np.zeros(6))

    def test_invalid_data_and_interior_overshoot(self) -> None:
        """非有限数据、停止索引和插值越界必须拒绝，不能靠放慢隐藏."""
        for stops in ((1, 5), (0, 1, 1, 5), (0, True, 5), (0, 6)):
            with self.subTest(stops=stops), self.assertRaises(ValueError):
                plan_continuous_pair(
                    self.q, self.q, self.low, self.high, self.speed, self.accel, stops=stops
                )
        broken = self.q.copy()
        broken[2, 0] = np.nan
        with self.assertRaises(ValueError):
            plan_continuous_pair(broken, self.q, self.low, self.high, self.speed, self.accel)
        # 全部路径点都在边界内，但三次导数估计导致段内过冲，应被解析检查拒绝。
        overshoot = np.repeat(np.array([0.0, 0.99, 1.0, 0.99, 0.0])[:, None], 6, axis=1)
        with self.assertRaisesRegex(ValueError, "插值角度超限"):
            plan_continuous_pair(overshoot, overshoot, self.low, self.high, self.speed, self.accel)


class RecordedSmoothingTests(unittest.TestCase):
    """检查样条导数、阶段停顿和真实静止段，不连接机械臂."""

    def setUp(self) -> None:
        """使用隔离环境的SciPy；旧转换和驱动不需要它."""
        try:
            from scipy.interpolate import make_interp_spline
            self.assertTrue(callable(make_interp_spline))
        except (ImportError, ValueError):
            self.skipTest('需要兼容NumPy的SciPy转换环境')
        self.low, self.high = -np.ones((2, 6)), np.ones((2, 6))
        self.speed, self.accel = np.full((2, 6), .5), np.full((2, 6), .3)

    def prepare(self, values, *, stops=()):
        """先用已有转换器得到受限基线."""
        q = np.repeat(np.asarray(values, dtype=np.float64)[:, None], 6, axis=1)
        plan, _ = retime_recorded_pair(
            tuple(i * 1000000000 for i in range(len(q))), q, q,
            self.low, self.high, self.speed, self.accel, stops=stops,
        )
        return plan

    def test_samples_stops_and_jerk_continuity(self) -> None:
        """普通点连续三阶导数，保留所有位置采样和指定阶段停点."""
        old = self.prepare([0., .01, .03, .06, .08, .09, .1], stops=(3,))
        new, report = smooth_recorded_pair(
            old, self.low, self.high, self.speed, self.accel)
        for side in ('left', 'right'):
            np.testing.assert_array_equal(old.positions[side], new.positions[side])
            np.testing.assert_array_equal(new.velocities[side][list(new.stops)], 0.)
            np.testing.assert_array_equal(new.accelerations[side][list(new.stops)], 0.)
            self.assertGreater(np.max(np.abs(new.accelerations[side])), 0.)
            self.assertLess(report['smoothing_jerk_jump_after_rad_s3'][side], 1e-8)
            self.assertLess(report['smoothing_jerk_jump_after_rad_s3'][side],
                            report['smoothing_jerk_jump_before_rad_s3'][side])
            self.assertFalse(new.velocities[side].flags.writeable)
        np.testing.assert_array_equal(old.times, new.times)

    def test_tighter_acceleration_retimes_both_arms(self) -> None:
        """导数超限只放慢时间，最终导出纳秒时间可重建同一曲线."""
        old = self.prepare([0., .01, .03, .06, .08])
        limit = np.full((2, 6), .001)
        new, report = smooth_recorded_pair(old, self.low, self.high, self.speed, limit)
        self.assertGreater(new.times[-1], old.times[-1])
        for side in ('left', 'right'):
            self.assertLessEqual(max(report['arms'][side]['peak_acceleration_rad_s2']), .001)
            for i, (p, following) in enumerate(zip(new.payload()[side]['points'],
                                                   new.payload()[side]['points'][1:])):
                duration = float(new.times[i + 1] - new.times[i])
                rebuilt = coefficients(*[np.asarray(x) for x in (
                    p['positions'], following['positions'], p['velocities'],
                    following['velocities'], p['accelerations'], following['accelerations'])],
                    duration)
                np.testing.assert_allclose(rebuilt, new.polynomials[side][i], atol=1e-12)

    def test_stationary_interval_stays_stationary(self) -> None:
        """即使没有阶段标签变化，也不能让真实静止区间产生样条摆动."""
        old = self.prepare([0., .01, .01, .03, .04])
        new, _ = smooth_recorded_pair(old, self.low, self.high, self.speed, self.accel)
        for elapsed in np.linspace(new.times[1], new.times[2], 20):
            np.testing.assert_allclose(new.sample('left', float(elapsed)), .01, atol=1e-14)

    def test_segment_overshoot_is_rejected(self) -> None:
        """所有采样都合法仍须检查段内过冲，不能靠夹紧角度掩盖."""
        old = self.prepare([0., .9, 1., .9, .8])
        with self.assertRaisesRegex(ValueError, '角度超限'):
            smooth_recorded_pair(old, self.low, self.high, self.speed, self.accel)

    def test_one_arm_hold_does_not_stop_other_arm(self) -> None:
        """单臂静止段保持原位置，另一臂不额外增加停点."""
        left = np.repeat(np.asarray([0., .01, .01, .03, .04])[:, None], 6, axis=1)
        right = np.repeat(np.asarray([0., .01, .02, .03, .04])[:, None], 6, axis=1)
        old, _ = retime_recorded_pair(
            tuple(i * 1000000000 for i in range(5)),
            left, right, self.low, self.high, self.speed, self.accel,
        )
        new, _ = smooth_recorded_pair(old, self.low, self.high, self.speed, self.accel)
        self.assertEqual(new.stops, (0, 4))
        middle = float((new.times[1] + new.times[2]) / 2.)
        np.testing.assert_allclose(new.sample('left', middle), .01, atol=1e-14)
        self.assertGreater(float(np.min(new.sample('right', middle, 1))), 0.)


class CircleBoundaryTests(unittest.TestCase):
    """用三阶段合成数据核对局部融合及窗口外逐段不变."""

    def fixture(self):
        """构造两个有间隙的阶段边界；角度很小，仅用于算法测试."""
        from archer_control.continuous_trajectory import ContinuousPair
        times = np.arange(61) * .1
        q = np.repeat((np.arange(61) * .00001)[:, None], 6, axis=1)
        zero = np.zeros_like(q)
        velocity = np.full_like(q, .0001)
        velocity[[0, 20, 21, 41, 42, 60]] = 0.
        polys = np.array([coefficients(q[i], q[i + 1], velocity[i], velocity[i + 1],
                                       zero[i], zero[i + 1], float(h))
                          for i, h in enumerate(np.diff(times))])
        plan = ContinuousPair(times, {s: q.copy() for s in ('left', 'right')},
                              {s: velocity.copy() for s in ('left', 'right')},
                              {s: zero.copy() for s in ('left', 'right')},
                              {s: polys.copy() for s in ('left', 'right')},
                              (0, 20, 21, 41, 42, 60))
        labels = ['APPROACH'] * 21 + ['ARC'] * 21 + ['RETURN_HOME'] * 19
        return plan, {s: labels.copy() for s in ('left', 'right')}

    def blend(self, old, labels):
        """使用显式位置、速度、加速度与jerk上限."""
        from archer_control.continuous_trajectory import blend_circle_boundaries
        return blend_circle_boundaries(old, labels, old.positions, -np.ones((2, 6)),
                                       np.ones((2, 6)), np.full((2, 6), .25),
                                       np.full((2, 6), .2), max_jerk=2.)

    def test_local_both_boundaries_keep_time_and_outside_polynomials(self):
        """入口出口都处理，起终点停车与外部段严格不变，所有拼接仍C2."""
        old, labels = self.fixture()
        new, windows = self.blend(old, labels)
        np.testing.assert_array_equal(new.times, old.times)
        self.assertEqual(new.stops, (0, 60))
        self.assertEqual([w['transition'] for w in windows], [21, 42])
        for window in windows:
            for measured in window['arms'].values():
                before, after = measured['jerk_jump_before_after_rad_s3']
                self.assertLess(after, before)
        for s in ('left', 'right'):
            touched = {i for w in windows for i in range(w['first'], w['last'])}
            for i, h in enumerate(np.diff(old.times)):
                if i not in touched:
                    np.testing.assert_array_equal(new.polynomials[s][i], old.polynomials[s][i])
                for order in (0, 1, 2):
                    target = (new.positions, new.velocities, new.accelerations)[order][s][i + 1]
                    actual = evaluate(new.polynomials[s][i], 1., float(h), order)
                    np.testing.assert_allclose(actual, target, atol=1e-10)
            self.assertLessEqual(float(np.max(abs(new.positions[s] - old.positions[s]))), .0005)
            self.assertTrue(np.any(abs(new.velocities[s][[20, 21, 41, 42]]) > 0.))

    def test_other_stops_or_wrong_phase_are_rejected(self):
        """不得把抓取、额外停车或单侧阶段不一致当作画圆融合."""
        from dataclasses import replace
        old, labels = self.fixture()
        with self.assertRaises(ValueError):
            self.blend(replace(old, stops=tuple(sorted((*old.stops, 19)))), labels)
        labels['right'][21] = 'INSERT'
        with self.assertRaises(ValueError):
            self.blend(old, labels)

    def test_real_hold_boundary_is_not_removed(self):
        """双臂确实静止的边界必须拒绝连续通过."""
        old, labels = self.fixture()
        for s in old.positions:
            old.positions[s][21] = old.positions[s][20]
        with self.assertRaisesRegex(ValueError, '保持区间'):
            self.blend(old, labels)


class ArcInteriorTests(unittest.TestCase):
    """高频小扰动合成路径验证圆弧拟合，不能作为实机振动测量."""

    def fixture(self):
        """两端已衔接的三阶段示例，圆弧中部含很小的高频位置扰动."""
        from scipy.interpolate import make_interp_spline
        from archer_control.continuous_trajectory import ContinuousPair
        times = np.arange(801) * .01
        envelope = np.sin(np.pi * np.clip((times - 3.2) / 2., 0., 1.))**6
        data = .0005 * times + 1e-6 * envelope * np.sin(80. * times)
        q = np.repeat(data[:, None], 6, axis=1)
        zero = np.zeros(6)
        curve = make_interp_spline(times, q, k=5,
                                   bc_type=([(1, zero), (2, zero)], [(1, zero), (2, zero)]))
        v, a = curve(times, 1), curve(times, 2)
        v[[0, -1]], a[[0, -1]] = 0., 0.
        c = np.stack([coefficients(q[i], q[i + 1], v[i], v[i + 1], a[i], a[i + 1], float(h))
                      for i, h in enumerate(np.diff(times))])
        fields = [{s: x.copy() for s in ('left', 'right')} for x in (q, v, a, c)]
        labels = ['APPROACH'] * 201 + ['ARC'] * 400 + ['RETURN_HOME'] * 200
        return ContinuousPair(times, *fields, (0, 800)), {s: labels for s in ('left', 'right')}

    def smooth(self, plan, labels, speed=.25, source=None):
        """复用原时轴和显式导数上限，只计算离线候选."""
        from archer_control.continuous_trajectory import smooth_circle_interior
        return smooth_circle_interior(
            plan, labels, plan.positions if source is None else source,
            -np.ones((2, 6)), np.ones((2, 6)), np.full((2, 6), speed),
            np.full((2, 6), .2), max_jerk=2.)

    def test_bounded_roughness_reduction_keeps_both_blends_and_clock(self):
        """降低最终五次执行段的jerk/snap积分，窗口外、两端导数和源时轴不变."""
        old, labels = self.fixture()
        new, report = self.smooth(old, labels)
        first, last = report['first'], report['last']
        np.testing.assert_array_equal(new.times, old.times)
        self.assertEqual(new.stops, old.stops)
        for metric in ('jerk_energy_rad2_s5', 'snap_energy_rad2_s7'):
            self.assertLess(report['after'][metric], report['before'][metric])
        for side in ('left', 'right'):
            for field in ('positions', 'velocities', 'accelerations'):
                before, after = getattr(old, field)[side], getattr(new, field)[side]
                np.testing.assert_array_equal(after[:first + 1], before[:first + 1])
                np.testing.assert_array_equal(after[last:], before[last:])
                self.assertFalse(after.flags.writeable)
            np.testing.assert_array_equal(new.polynomials[side][:first],
                                          old.polynomials[side][:first])
            np.testing.assert_array_equal(new.polynomials[side][last:],
                                          old.polynomials[side][last:])
            self.assertLessEqual(report['arms'][side]['max_segment_change_rad'], .0001)
            for i in (first, (first + last) // 2, last - 1):
                h = float(new.times[i + 1] - new.times[i])
                for order, field in enumerate(('positions', 'velocities', 'accelerations')):
                    np.testing.assert_allclose(evaluate(new.polynomials[side][i], 1., h, order),
                                               getattr(new, field)[side][i + 1], atol=1e-9)

    def test_no_feasible_candidate_keeps_input(self):
        """更严限速使拟合不合格时拒绝，不原地修改成功输入."""
        old, labels = self.fixture()
        saved = {s: a.copy() for s, a in old.positions.items()}
        with self.assertRaisesRegex(ValueError, '保留成功候选'):
            self.smooth(old, labels, speed=1e-6)
        for side in saved:
            np.testing.assert_array_equal(old.positions[side], saved[side])

    def test_extra_stop_or_real_hold_is_rejected(self):
        """内部额外停车与实际保持不能被最小jerk拟合消除."""
        from dataclasses import replace
        old, labels = self.fixture()
        with self.assertRaises(ValueError):
            self.smooth(replace(old, stops=(0, 400, 800)), labels)
        source = {s: q.copy() for s, q in old.positions.items()}
        for q in source.values():
            q[401] = q[400]
        with self.assertRaisesRegex(ValueError, '保持区间'):
            self.smooth(old, labels, source=source)


if __name__ == "__main__":
    unittest.main()
