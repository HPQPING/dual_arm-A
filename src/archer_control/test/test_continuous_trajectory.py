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
    """检查样条导数、阶段停顿和真实静止段，不连接机械臂。"""

    def setUp(self) -> None:
        """使用隔离环境的SciPy；旧转换和驱动不需要它。"""
        try:
            from scipy.interpolate import make_interp_spline
            self.assertTrue(callable(make_interp_spline))
        except (ImportError, ValueError):
            self.skipTest('需要兼容NumPy的SciPy转换环境')
        self.low, self.high = -np.ones((2, 6)), np.ones((2, 6))
        self.speed, self.accel = np.full((2, 6), .5), np.full((2, 6), .3)

    def prepare(self, values, *, stops=()):
        """先用已有转换器得到受限基线。"""
        q = np.repeat(np.asarray(values, dtype=np.float64)[:, None], 6, axis=1)
        plan, _ = retime_recorded_pair(
            tuple(i * 1000000000 for i in range(len(q))), q, q,
            self.low, self.high, self.speed, self.accel, stops=stops,
        )
        return plan

    def test_samples_stops_and_jerk_continuity(self) -> None:
        """普通点连续三阶导数，保留所有位置采样和指定阶段停点。"""
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
        """导数超限只放慢时间，最终导出纳秒时间可重建同一曲线。"""
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
        """即使没有阶段标签变化，也不能让真实静止区间产生样条摆动。"""
        old = self.prepare([0., .01, .01, .03, .04])
        new, _ = smooth_recorded_pair(old, self.low, self.high, self.speed, self.accel)
        for elapsed in np.linspace(new.times[1], new.times[2], 20):
            np.testing.assert_allclose(new.sample('left', float(elapsed)), .01, atol=1e-14)

    def test_segment_overshoot_is_rejected(self) -> None:
        """所有采样都合法仍须检查段内过冲，不能靠夹紧角度掩盖。"""
        old = self.prepare([0., .9, 1., .9, .8])
        with self.assertRaisesRegex(ValueError, '角度超限'):
            smooth_recorded_pair(old, self.low, self.high, self.speed, self.accel)

    def test_one_arm_hold_does_not_stop_other_arm(self) -> None:
        """单臂静止段保持原位置，另一臂不额外增加停点。"""
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


if __name__ == "__main__":
    unittest.main()
