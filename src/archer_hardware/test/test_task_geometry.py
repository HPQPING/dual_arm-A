"""验证离线接近、连续路径加密采样及碰撞拒绝，不连接实机."""

from dataclasses import replace
from typing import Any
import unittest
from unittest.mock import patch

import numpy as np

from archer_hardware.position_plan import ContinuousPositionTrajectory, PositionLimits
from archer_hardware.task_geometry import (
    check_samples, check_task_geometry, geometry_schedule, prepare_approach, prepare_return,
)


class TaskGeometryTests(unittest.TestCase):
    """用虚构限位与几何判定替身验证预检流程，不宣称实物无碰撞."""

    def setUp(self) -> None:
        """建立两秒小幅连续路径和未连接任何设备的双臂映射."""
        self.limits = PositionLimits((-1.,) * 6, (1.,) * 6, (.05,) * 6, (.1,) * 6, .005)
        plan = ContinuousPositionTrajectory.prepare(
            (0, 2000000000), ((0.,) * 6, (.02,) * 6),
            ((0.,) * 6,) * 2, ((0.,) * 6,) * 2, self.limits, .05,
        )
        self.plans = {'left': plan, 'right': plan}

    def test_approach_keeps_task_and_exact_start(self) -> None:
        """接近单独保存，起止精确连接，原文件多项式保持不变."""
        original = self.plans['left']
        before = original.polynomials[0].tobytes()
        starts = {'left': [-.01] * 6, 'right': [-.02] * 6}
        approach = prepare_approach(self.plans, starts, max_excursion=.05)
        for side, p in approach.items():
            self.assertEqual(p.initial, tuple(starts[side]))
            self.assertEqual(p.goal, self.plans[side].initial)
            self.assertEqual(p.sample(0., 1), (0.,) * 6)
            self.assertEqual(p.sample(p.duration, 2), (0.,) * 6)
        self.assertEqual(approach['left'].times_ns, approach['right'].times_ns)
        self.assertIs(self.plans['left'], original)
        self.assertEqual(original.polynomials[0].tobytes(), before)

    def test_shared_duration_uses_slower_arm(self) -> None:
        """两臂使用更严格一侧的速度限值选择共同接近时长."""
        slower = replace(self.limits, max_velocity=(.01,) * 6)
        right = ContinuousPositionTrajectory.prepare(
            (0, 5000000000), ((0.,) * 6, (.02,) * 6),
            ((0.,) * 6,) * 2, ((0.,) * 6,) * 2, slower, .05,
        )
        left = ContinuousPositionTrajectory.prepare(
            right.times_ns, ((0.,) * 6, (.02,) * 6),
            ((0.,) * 6,) * 2, ((0.,) * 6,) * 2, self.limits, .05,
        )
        approach = prepare_approach({'left': left, 'right': right},
                                    {'left': [-.04] * 6, 'right': [-.04] * 6},
                                    max_excursion=.05)
        self.assertGreaterEqual(approach['left'].duration, 7.875)
        self.assertEqual(approach['left'].duration, approach['right'].duration)

    def test_return_goes_to_saved_start_instead_of_file_home(self) -> None:
        """归位目标是本轮保存位置，不能误停在文件原始起点或原末点."""
        saved = {'left': [-.01] * 6, 'right': [-.02] * 6}
        result = prepare_return(self.plans, saved, max_excursion=.05)
        for side, p in result.items():
            self.assertEqual(p.initial, self.plans[side].goal)
            self.assertEqual(p.goal, tuple(saved[side]))
            self.assertNotEqual(p.goal, self.plans[side].initial)
        self.assertEqual(result['left'].times_ns, result['right'].times_ns)

    def test_return_rechecks_limits_and_excursion(self) -> None:
        """归位也检查名义限位和幅度，不为回到起点跳过保护."""
        for saved in ({'left': [-1.001] * 6, 'right': [0.] * 6},
                      {'left': [-.2] * 6, 'right': [0.] * 6}):
            with self.assertRaises(ValueError):
                prepare_return(self.plans, saved, max_excursion=.05)

    def test_approach_rejects_missing_limit_and_excursion(self) -> None:
        """缺失起点、略超名义限位和过大接近均拒绝，不裁剪或替换反馈."""
        cases: list[Any] = [None, {'left': [0.] * 6},
                            {'left': [-1.001] * 6, 'right': [0.] * 6},
                            {'left': [-.1] * 6, 'right': [0.] * 6}]
        for start in cases:
            with self.subTest(start=start), self.assertRaises(ValueError):
                prepare_approach(self.plans, start, max_excursion=.05)

    def test_nonstationary_file_start_rejected(self) -> None:
        """文件初始导数非零时不能把静止接近直接拼在前面."""
        c = self.plans['left'].polynomials[0].copy()
        c[:, 1] = .001
        plan = replace(self.plans['left'], polynomials=(c,))
        with self.assertRaisesRegex(ValueError, '零速'):
            prepare_approach({'left': plan, 'right': plan},
                             {'left': [0.] * 6, 'right': [0.] * 6}, max_excursion=.05)

    def test_schedule_includes_boundaries_and_travel_bound(self) -> None:
        """从实际连续多项式取样，端点保留，间隔及行程受控."""
        count, iterator = geometry_schedule(self.plans)
        stamps = list(iterator)
        self.assertEqual(count, len(stamps))
        self.assertEqual(stamps[0], 0)
        self.assertEqual(stamps[-1], 2000000000)
        self.assertTrue(all(0 < b - a <= 100000000 for a, b in zip(stamps, stamps[1:])))
        q = [self.plans['left'].sample(t / 1e9) for t in stamps]
        self.assertLessEqual(float(np.max(np.abs(np.diff(q, axis=0)))), .005)

    def test_collision_between_recorded_endpoints_is_found(self) -> None:
        """仅端点安全仍可能中途碰撞，加密检查必须拒绝."""
        self.assertEqual(self.plans['left'].initial[0], 0.)
        self.assertEqual(self.plans['left'].goal[0], .02)

        def failure(left: tuple[float, ...], right: tuple[float, ...]) -> str | None:
            return '合成跨臂碰撞' if .009 <= left[0] <= .011 else None

        with self.assertRaisesRegex(ValueError, '合成跨臂碰撞'):
            check_samples(self.plans, failure)

    def test_budget_failure_checks_no_partial_path(self) -> None:
        """超过预算先拒绝，不能检查前几个点后返回成功."""
        with patch('archer_hardware.task_geometry.geometry_schedule',
                   wraps=geometry_schedule) as schedule:
            calls: list[int] = []
            with self.assertRaisesRegex(ValueError, '预算'):
                check_samples(self.plans, lambda a, b: calls.append(1) or None, max_samples=2)
            self.assertEqual(calls, [])
            schedule.assert_called_once()

    def test_invalid_or_unpaired_schedule_rejected(self) -> None:
        """拒绝单臂、不同时间轴和通过放大采样步长逃避检查."""
        with self.assertRaises(ValueError):
            geometry_schedule({'left': self.plans['left']})
        with self.assertRaises(ValueError):
            geometry_schedule({'left': self.plans['left'], 'right': replace(
                self.plans['right'], times_ns=(0, 3000000000))})
        invalid: list[dict[str, Any]] = [
            {'max_samples': True}, {'max_time_step': 1.},
            {'max_joint_step': .1}, {'max_time_step': 0.},
        ]
        for options in invalid:
            with self.subTest(options=options), self.assertRaises(ValueError):
                geometry_schedule(self.plans, **options)

    def test_success_checks_all_points(self) -> None:
        """成功时实际调用次数必须等于完整计划的采样数量."""
        calls: list[int] = []
        result = check_samples(self.plans, lambda a, b: calls.append(1) or None)
        self.assertEqual(len(calls), result['samples_checked'])
        self.assertGreater(len(calls), 2)

    def test_budget_and_boundary_failure_happen_before_model_load(self) -> None:
        """采样预算和段落衔接失败不能加载模型后继续执行部分检查."""
        with patch('archer_mujoco.dual_model_loader.load_dual_arm') as loader:
            with self.assertRaisesRegex(ValueError, '预算'):
                check_task_geometry(self.plans, max_samples=2)
            saved = {'left': [-.01] * 6, 'right': [-.02] * 6}
            approach = prepare_approach(self.plans, saved, max_excursion=.05)
            home = prepare_return(self.plans, saved, max_excursion=.05)
            home['left'] = replace(home['left'], goal=(.123,) * 6)
            with self.assertRaisesRegex(ValueError, '本轮保存起点'):
                check_task_geometry(self.plans, approach=approach, return_home=home)
            loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
