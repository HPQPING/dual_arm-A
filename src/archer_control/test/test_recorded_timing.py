"""检查采样轨迹的分段时间、连续插值及数据保留，不依赖ROS或机械臂."""

import unittest
from typing import Any

import numpy as np

from archer_control.continuous_trajectory import (
    evaluate, retime_recorded_pair, RecordedTimeMap, coefficients, _extrema,
    _recorded_derivative_peaks,
)


class RecordedTimingTests(unittest.TestCase):
    """用合成路径核对非均匀时间、停顿和解析约束."""

    def setUp(self):
        """创建虚构的双臂限制，与实机参数无关."""
        self.low, self.high = -np.ones((2, 6)), np.ones((2, 6))
        self.speed, self.accel = np.full((2, 6), 0.05), np.full((2, 6), 0.1)
        self.times = (0, 50000000, 120000000, 200000000, 300000000, 400000000)
        self.q = np.repeat(np.array([0.0, 0.01, 0.03, 0.02, 0.02, 0.01])[:, None], 6, axis=1)

    def prepare(self, **overrides):
        """仅准备内存轨迹，不导入硬件模块."""
        parameters: dict[str, Any] = dict(
            source_times_ns=self.times,
            left=self.q,
            right=self.q,
            lower=self.low,
            upper=self.high,
            max_velocity=self.speed,
            max_acceleration=self.accel,
        )
        parameters.update(overrides)
        return retime_recorded_pair(**parameters)

    def test_points_nonuniform_timing_and_input_preserved(self):
        """保留全部原角度、两臂共同轴和原有等待，输入数组不被改写."""
        before = self.q.copy()
        plan, report = self.prepare()
        self.assertEqual(report["samples_preserved"], len(self.times))
        for side in ("left", "right"):
            np.testing.assert_array_equal(plan.positions[side], before)
        np.testing.assert_array_equal(self.q, before)
        prepared = tuple(int(round(t * 1e9)) for t in plan.times)
        held = tuple(bool(np.array_equal(a, b)) for a, b in zip(self.q, self.q[1:]))
        mapping = RecordedTimeMap.from_report(report, self.times, prepared, held_intervals=held)
        self.assertTrue(all(abs(mapping.map_ns(a) - b) <= 2
                            for a, b in zip(self.times, prepared)))
        self.assertGreaterEqual(plan.times[-1], self.times[-1] / 1e9)
        self.assertTrue(all(not h or b - a >= (d - c) / 1e9 - 2e-9
                            for h, a, b, c, d in zip(
                                held, plan.times, plan.times[1:], self.times, self.times[1:])))
        self.assertFalse(plan.times.flags.writeable)
        self.assertFalse(plan.positions["left"].flags.writeable)

    def test_local_bottleneck_does_not_stretch_whole_record(self):
        """末段大位移只放慢必要区间，较小区间不承受整轮统一倍率."""
        q = np.repeat(np.array([0., .001, .002, .003, .004, .3])[:, None], 6, axis=1)
        plan, report = self.prepare(left=q, right=q)
        self.assertEqual(report['timing_policy'], 'local_segment_c2')
        self.assertIsNone(report['uniform_time_scale'])
        self.assertLess(plan.times[-1], report['uniform_baseline_duration_s'] * .8)
        self.assertGreater(len(set(np.diff(plan.times).round(6))), 1)
        for side in ('left', 'right'):
            np.testing.assert_array_equal(plan.positions[side], q)
            self.assertLessEqual(max(report['arms'][side]['peak_velocity_rad_s']), .05)
            self.assertLessEqual(max(report['arms'][side]['peak_acceleration_rad_s2']), .1)

    def test_bounded_updates_avoid_unnecessary_overstretch(self):
        """非均匀采样的大倍率更新不能把已可平滑执行的短路径过度拉长."""
        rng = np.random.default_rng(20261021)
        gaps = rng.integers(2000000, 45000000, 17)
        times = tuple(int(t) for t in np.concatenate(([0], np.cumsum(gaps))))
        values = np.cumsum(rng.uniform(.00001, .01, 18))
        values -= values[0]
        q = np.repeat(values[:, None], 6, axis=1)
        plan, report = self.prepare(source_times_ns=times, left=q, right=q,
                                    max_velocity=np.full((2, 6), .5),
                                    max_acceleration=np.full((2, 6), .3))
        self.assertLess(plan.times[-1], 3.)
        self.assertTrue(report['continuous_constraints_passed'])
        for side in ('left', 'right'):
            np.testing.assert_array_equal(plan.positions[side], q)
            self.assertLessEqual(max(report['arms'][side]['peak_velocity_rad_s']), .5)
            self.assertLessEqual(max(report['arms'][side]['peak_acceleration_rad_s2']), .3)

    def test_time_mapping_events_and_legacy_uniform_report(self):
        """旧报告按原统一倍率映射，新报告按区间映射事件并精确保留边界."""
        source, prepared = (0, 10, 20), (0, 30, 50)
        report = {'timing_policy': 'local_segment_c2', 'uniform_time_scale': None,
                  'source_times_ns': list(source), 'prepared_times_ns': list(prepared),
                  'effective_time_scale': 2.5, 'stop_indices': [0, 2]}
        mapping = RecordedTimeMap.from_report(report, source, prepared)
        self.assertEqual(mapping.map_ns(3), 9)
        self.assertEqual(mapping.map_ns(10), 30)
        self.assertEqual(mapping.map_ns(15), 40)
        self.assertEqual(mapping.map_ns(20), 50)
        legacy = RecordedTimeMap.from_report({'uniform_time_scale': 2.}, source, (0, 20, 40))
        self.assertEqual(legacy.map_ns(7), 14)
        for changes in ({'source_times_ns': [0, 9, 20]},
                        {'prepared_times_ns': [0, True, 50]},
                        {'effective_time_scale': float('nan')},
                        {'timing_policy': 'unknown'}, {'uniform_time_scale': 2.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                RecordedTimeMap.from_report({**report, **changes}, source, prepared)
        with self.assertRaisesRegex(ValueError, '不允许缩短'):
            RecordedTimeMap.from_report({**report, 'prepared_times_ns': [0, 5, 50]},
                                        source, (0, 5, 50))
        for stamp in (True, -1, 21):
            with self.assertRaises(ValueError):
                mapping.map_ns(stamp)
        shifted = {**report, 'prepared_times_ns': [0, 5, 50]}
        moving = RecordedTimeMap.from_report(shifted, source, (0, 5, 50),
                                             held_intervals=(False, False))
        self.assertEqual(moving.map_ns(10), 5)
        with self.assertRaisesRegex(ValueError, '阶段或事件'):
            RecordedTimeMap.from_report({**shifted, 'stop_indices': [0, 1, 2]}, source,
                                        (0, 5, 50), held_intervals=(False, False))

    def test_vectorized_peaks_match_independent_polynomial_roots(self):
        """随机时长与正反向段的批量峰值，必须和独立通用多项式求根结果一致."""
        rng = np.random.default_rng(20261020)
        duration = 10. ** rng.uniform(-3., 1., 30)
        slopes = rng.uniform(-.5, .5, (30, 6))
        v0, v1 = (slopes * rng.uniform(0., 1., slopes.shape) for _ in range(2))
        slopes[0] = v0[0] = v1[0] = 0.
        v0[1] = v1[1] = slopes[1]
        speeds, accelerations = _recorded_derivative_peaks(slopes, v0, v1, duration)
        for i, seconds in enumerate(duration):
            c = coefficients(np.zeros(6), slopes[i] * seconds, v0[i], v1[i],
                             np.zeros(6), np.zeros(6), float(seconds))
            np.testing.assert_allclose(speeds[i], np.max(np.abs(_extrema(c, seconds, 1)), axis=1),
                                       rtol=1e-9, atol=1e-10)
            np.testing.assert_allclose(accelerations[i],
                                       np.max(np.abs(_extrema(c, seconds, 2)), axis=1),
                                       rtol=1e-9, atol=1e-10)

    def test_c2_continuity_and_stop_points(self):
        """各连接点的位置、速度及加速度连续；指定停点不被平滑穿过."""
        plan, _ = self.prepare(stops=(0, 3, 5))
        self.assertGreater(abs(float(plan.velocities["left"][1, 0])), 0.0)
        for side in ("left", "right"):
            for i in range(1, len(plan.times) - 1):
                for derivative in (0, 1, 2):
                    a = evaluate(
                        plan.polynomials[side][i - 1],
                        1.0,
                        float(plan.times[i] - plan.times[i - 1]),
                        derivative,
                    )
                    b = evaluate(
                        plan.polynomials[side][i],
                        0.0,
                        float(plan.times[i + 1] - plan.times[i]),
                        derivative,
                    )
                    np.testing.assert_allclose(a, b, atol=1e-11, rtol=0.0)
            for index in plan.stops:
                np.testing.assert_array_equal(plan.velocities[side][index], np.zeros(6))
                np.testing.assert_array_equal(plan.accelerations[side][index], np.zeros(6))

    def test_dense_derivatives_and_no_position_overshoot(self):
        """独立密集取样核对解析检查结果，段内位置不越过相邻采样角度."""
        plan, report = self.prepare()
        self.assertTrue(report["continuous_constraints_passed"])
        for elapsed in np.linspace(0.0, plan.times[-1], 501):
            self.assertLessEqual(
                float(np.max(np.abs(plan.sample("left", float(elapsed), 1)))), 0.05 + 1e-12
            )
            self.assertLessEqual(
                float(np.max(np.abs(plan.sample("left", float(elapsed), 2)))), 0.1 + 1e-12
            )
        for i in range(len(plan.times) - 1):
            for elapsed in np.linspace(plan.times[i], plan.times[i + 1], 31):
                q = plan.sample("left", float(elapsed))
                self.assertTrue(np.all(q >= np.minimum(self.q[i], self.q[i + 1]) - 1e-12))
                self.assertTrue(np.all(q <= np.maximum(self.q[i], self.q[i + 1]) + 1e-12))

    def test_recorded_hold_and_endpoint_hold(self):
        """记录里的静止区间保持原姿态，越过轨迹末端后保持最终位置."""
        plan, _ = self.prepare()
        for elapsed in np.linspace(plan.times[3], plan.times[4], 21):
            np.testing.assert_allclose(plan.sample("left", float(elapsed)), self.q[3], atol=1e-12)
        np.testing.assert_array_equal(
            plan.sample("right", float(plan.times[-1]) + 5.0), self.q[-1]
        )
        np.testing.assert_array_equal(
            plan.sample("right", float(plan.times[-1]) + 5.0, 1), np.zeros(6)
        )

    def test_right_constraints_determine_shared_timing(self):
        """较慢的右臂限制决定共同时间，禁止两臂分别播放."""
        normal, _ = self.prepare()
        self.speed[1] *= 0.1
        slower, report = self.prepare()
        self.assertGreater(slower.times[-1], normal.times[-1])
        self.assertLessEqual(max(report["arms"]["right"]["peak_velocity_rad_s"]), 0.005 + 1e-12)

    def test_invalid_input_does_not_generate_a_plan(self):
        """拒绝无效时间、角度、停点和超长限速结果."""
        for times in ((1, 2), (0, 0), (0, True), (0, -1)):
            with self.subTest(times=times), self.assertRaises(ValueError):
                self.prepare(source_times_ns=times)
        for stops in ((3, 1), (True,), (6,)):
            with self.subTest(stops=stops), self.assertRaises(ValueError):
                self.prepare(stops=stops)
        broken = self.q.copy()
        broken[1, 1] = 1.01
        with self.assertRaises(ValueError):
            self.prepare(left=broken)
        with self.assertRaises(ValueError):
            self.prepare(margin=True)
        self.speed[:] = 1e-8
        with self.assertRaisesRegex(ValueError, "一小时"):
            self.prepare()


class SpatialRecordedTimingTests(unittest.TestCase):
    """验证有界空间拟合、共同弧长定时及曲率加速度，不连接设备。"""

    def setUp(self):
        """准备含细小测量波动的双臂曲线路径。"""
        try:
            from scipy.interpolate import BSpline
            self.assertTrue(callable(BSpline))
        except (ImportError, ValueError):
            self.skipTest('需要兼容的SciPy离线转换环境')
        x = np.linspace(0., 1., 80)
        self.q = np.column_stack([
            .15 * x + .01 * np.sin(x * j) + .00001 * np.sin(x * 90.) for j in range(1, 7)])
        self.times = tuple(i * 20000000 for i in range(len(x)))
        self.low, self.high = -np.ones((2, 6)), np.ones((2, 6))
        self.speed, self.accel = np.full((2, 6), .2), np.full((2, 6), .3)

    def prepare(self, **kwargs):
        """使用同一配置生成待检查数学对象。"""
        from archer_control.continuous_trajectory import retime_spatial_pair

        return retime_spatial_pair(self.times, self.q, self.q, self.low, self.high,
                                   self.speed, self.accel, stops=(0, 40, 79), **kwargs)

    def test_bounded_positions_and_exact_phase_stops(self):
        """节点拟合变化受限，起止和阶段边界位置不变，内部不逐点停车。"""
        plan, report = self.prepare()
        for s in ('left', 'right'):
            self.assertLessEqual(float(np.max(abs(plan.positions[s] - self.q))), .0005)
            np.testing.assert_array_equal(plan.positions[s][list(plan.stops)],
                                          self.q[list(plan.stops)])
            np.testing.assert_array_equal(plan.velocities[s][list(plan.stops)], 0.)
            np.testing.assert_array_equal(plan.accelerations[s][list(plan.stops)], 0.)
            self.assertTrue(np.all(np.linalg.norm(plan.velocities[s][1:40], axis=1) > 0.))
        self.assertEqual(report['position_filter'], 'bounded_spatial_spline')
        mapping = RecordedTimeMap.from_report(report, self.times,
                                              tuple(report['prepared_times_ns']),
                                              held_intervals=(False,) * (len(self.times) - 1))
        self.assertEqual(mapping.map_ns(self.times[40]), report['prepared_times_ns'][40])

    def test_dense_derivatives_include_path_curvature(self):
        """独立密集取样确认每关节速度和总加速度，而非只检查标量进度。"""
        plan, report = self.prepare()
        for t in np.linspace(0., plan.times[-1], 1000):
            self.assertLessEqual(float(np.max(abs(plan.sample('left', float(t), 1)))), .2 + 1e-9)
            self.assertLessEqual(float(np.max(abs(plan.sample('left', float(t), 2)))), .3 + 1e-9)
        self.assertGreaterEqual(report['duration_s'], self.times[-1] / 1e9)

    def test_shared_stationary_interval_is_preserved(self):
        """两臂真实静止区间不被空间进度除零或缩短。"""
        from archer_control.continuous_trajectory import retime_spatial_pair

        q = np.repeat(np.asarray([0., .01, .01, .03])[:, None], 6, axis=1)
        plan, _ = retime_spatial_pair(
            (0, 10**9, 2 * 10**9, 3 * 10**9), q, q,
            self.low, self.high, self.speed, self.accel, stops=(0, 3))
        for t in np.linspace(plan.times[1], plan.times[2], 20):
            np.testing.assert_allclose(plan.sample('left', float(t)), .01, atol=1e-14)
        self.assertGreaterEqual(plan.times[2] - plan.times[1], 1.)

    def test_invalid_position_tolerance_is_rejected(self):
        """不能扩大固定节点拟合误差上限。"""
        for value in (0., .001, True, float('nan')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepare(tolerance=value)


if __name__ == "__main__":
    unittest.main()
