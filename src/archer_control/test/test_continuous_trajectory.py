"""离线验证连续插值、整段约束及候选格式，不连接设备。"""

import unittest

import numpy as np

from archer_control.continuous_trajectory import coefficients, evaluate, plan_continuous_pair


class ContinuousTrajectoryTests(unittest.TestCase):
    """使用虚构数据验证数学与格式边界，不作为实机任务数据。"""

    def setUp(self) -> None:
        """构造接近段和单调圆弧示意关节路径。"""
        self.q = np.repeat(np.array([0., .1, .12, .15, .18, .2])[:, None], 6, axis=1)
        self.low, self.high = -np.ones((2, 6)), np.ones((2, 6))
        self.speed, self.accel = np.full((2, 6), .05), np.full((2, 6), .1)

    def plan(self):
        """使用统一虚构约束生成候选。"""
        return plan_continuous_pair(self.q, self.q, self.low, self.high,
                                    self.speed, self.accel, stops=(0, 1, 5))

    def test_internal_velocity_and_c2_continuity(self) -> None:
        """内部路径点不停顿，全部连接处位置、速度和加速度连续。"""
        plan, _ = self.plan()
        self.assertTrue(np.any(np.abs(plan.velocities['left'][2:-1]) > 1e-5))
        for side in ('left', 'right'):
            for index in range(1, len(plan.times) - 1):
                before = float(plan.times[index] - plan.times[index - 1])
                after = float(plan.times[index + 1] - plan.times[index])
                for derivative in (0, 1, 2):
                    np.testing.assert_allclose(
                        evaluate(plan.polynomials[side][index - 1], 1., before, derivative),
                        evaluate(plan.polynomials[side][index], 0., after, derivative),
                        atol=1e-12, rtol=0.,
                    )
            for index in plan.stops:
                np.testing.assert_array_equal(plan.velocities[side][index], np.zeros(6))
                np.testing.assert_array_equal(plan.accelerations[side][index], np.zeros(6))

    def test_dense_derivatives_stay_within_limits(self) -> None:
        """独立密集取样核对解析峰值保护，不能只检查各路径点。"""
        plan, report = self.plan()
        for elapsed in np.linspace(0., plan.times[-1], 1001):
            self.assertLessEqual(
                float(np.max(np.abs(plan.sample('left', float(elapsed), 1)))), .05)
            self.assertLessEqual(float(np.max(np.abs(plan.sample('left', float(elapsed), 2)))), .1)
        self.assertLessEqual(max(report['arms']['right']['peak_velocity_rad_s']), .05)
        self.assertLessEqual(max(report['arms']['right']['peak_acceleration_rad_s2']), .1)

    def test_tighter_right_arm_shared_time_and_unchanged_input(self) -> None:
        """右臂更严格的限制控制共享时长，不修改输入关节角。"""
        before = self.q.copy()
        normal, _ = self.plan()
        self.speed[1] *= .2
        slower, _ = self.plan()
        self.assertGreater(slower.times[-1], normal.times[-1])
        np.testing.assert_array_equal(before, self.q)
        np.testing.assert_array_equal(slower.positions['right'], self.q)
        self.assertFalse(slower.times.flags.writeable)

    def test_payload_rebuild_and_hold_endpoint(self) -> None:
        """导出纳秒时间和显式导数能重建同一多项式，完成后保持最终位置。"""
        plan, _ = self.plan()
        payload = plan.payload()
        self.assertEqual(payload['version'], 2)
        self.assertEqual(payload['interpolation'], 'quintic_c2')
        self.assertEqual(payload['stop_indices'], [0, 1, 5])
        for side in ('left', 'right'):
            points = payload[side]['points']
            for index, (a, b) in enumerate(zip(points, points[1:])):
                stamps = [p['time_from_start']['sec'] + p['time_from_start']['nanosec'] / 1e9
                          for p in (a, b)]
                q0, q1, v0, v1, a0, a1 = [
                    np.asarray(p[key], dtype=np.float64)
                    for key in ('positions', 'velocities', 'accelerations') for p in (a, b)
                ]
                c = coefficients(q0, q1, v0, v1, a0, a1, stamps[1] - stamps[0])
                np.testing.assert_allclose(c, plan.polynomials[side][index], atol=1e-12, rtol=0.)
            np.testing.assert_array_equal(
                plan.sample(side, float(plan.times[-1]) + 5.), self.q[-1])
            np.testing.assert_array_equal(plan.sample(side, -1.), self.q[0])
            np.testing.assert_array_equal(plan.sample(side, float(plan.times[-1]), 1), np.zeros(6))

    def test_invalid_data_and_interior_overshoot(self) -> None:
        """非有限数据、停止索引和插值越界必须拒绝，不能靠放慢隐藏。"""
        for stops in ((1, 5), (0, 1, 1, 5), (0, True, 5), (0, 6)):
            with self.subTest(stops=stops), self.assertRaises(ValueError):
                plan_continuous_pair(self.q, self.q, self.low, self.high,
                                     self.speed, self.accel, stops=stops)
        broken = self.q.copy()
        broken[2, 0] = np.nan
        with self.assertRaises(ValueError):
            plan_continuous_pair(broken, self.q, self.low, self.high, self.speed, self.accel)
        # 全部路径点都在边界内，但三次导数估计导致段内过冲，应被解析检查拒绝。
        overshoot = np.repeat(np.array([0., .99, 1., .99, 0.])[:, None], 6, axis=1)
        with self.assertRaisesRegex(ValueError, '插值角度超限'):
            plan_continuous_pair(overshoot, overshoot, self.low, self.high,
                                 self.speed, self.accel)


if __name__ == '__main__':
    unittest.main()
