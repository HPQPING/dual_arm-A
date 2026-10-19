"""检查采样轨迹连续插值、统一限速及数据保留，不依赖ROS或机械臂."""

import unittest
from typing import Any

import numpy as np

from archer_control.continuous_trajectory import evaluate, retime_recorded_pair


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
        """保留全部原角度；共享时间轴仅统一放慢，输入数组不被改写."""
        before = self.q.copy()
        plan, report = self.prepare()
        self.assertGreater(report["uniform_time_scale"], 1.0)
        self.assertEqual(report["samples_preserved"], len(self.times))
        for side in ("left", "right"):
            np.testing.assert_array_equal(plan.positions[side], before)
        np.testing.assert_array_equal(self.q, before)
        source = np.array(self.times, dtype=np.float64) / 1e9
        error = plan.times - source * report["uniform_time_scale"]
        self.assertTrue(np.all(error >= -1e-12))
        self.assertLessEqual(float(np.max(error)), 1.001e-9)
        self.assertFalse(plan.times.flags.writeable)
        self.assertFalse(plan.positions["left"].flags.writeable)

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


if __name__ == "__main__":
    unittest.main()
