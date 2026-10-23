"""验证离线接近、连续路径加密采样及碰撞拒绝，不连接实机."""

from __future__ import annotations

from dataclasses import replace
from contextlib import redirect_stdout
import io
from importlib import import_module
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast
import unittest
from unittest.mock import patch

import numpy as np
import pytest

from archer_hardware.position_plan import ContinuousPositionTrajectory, PositionLimits
from archer_hardware.task_geometry import (
    assemble_task_cycle, check_samples, check_task_geometry, geometry_schedule,
    prepare_approach, prepare_return,
    joint_error_displacement_bounds, geometry_error_displacement_bounds,
    _pose_geometry_error_bounds,
)

if TYPE_CHECKING:
    from archer_mujoco.dual_model_loader import LoadedDualArm


def envelope_model(*, fixed_base: bool = False) -> LoadedDualArm:
    """构造两侧六转动关节与球形碰撞体，避免依赖设备或工程模型文件."""
    mj = import_module('mujoco')
    bodies = []
    for side in ('left', 'right'):
        chain = ''
        for index in range(6):
            axis = ('1 0 0', '0 1 0', '0 0 1')[index % 3]
            chain += (f'<body name="{side}_{index}" pos="0 0 .1">'
                      f'<joint type="hinge" axis="{axis}"/>'
                      '<geom type="sphere" size=".03" pos=".05 0 .06" mass=".01"/>')
        chain += '</body>' * 6
        if fixed_base:
            chain = (f'<body name="{side}_base">'
                     '<geom type="sphere" size=".03" mass=".01"/>' + chain + '</body>')
        bodies.append(chain)
    model = mj.MjModel.from_xml_string(
        '<mujoco><worldbody>' + ''.join(bodies) + '</worldbody></mujoco>')
    return cast('LoadedDualArm', SimpleNamespace(
        model=model, data=mj.MjData(model), arms={
            'left': SimpleNamespace(qpos_indices=np.arange(6)),
            'right': SimpleNamespace(qpos_indices=np.arange(6, 12)),
        }))


def test_error_displacement_bound_contains_perturbed_geometry():
    """随机非零姿态与全关节误差下，碰撞球外形点位移不超过解析保守上界."""
    mj = import_module('mujoco')
    loaded = envelope_model()
    bounds = joint_error_displacement_bounds(loaded, .0001)
    model, data = cast(Any, loaded.model), cast(Any, loaded.data)
    rng = np.random.default_rng(20261019)
    offsets = np.concatenate((np.eye(3), -np.eye(3))) * .03
    for _ in range(20):
        original = rng.uniform(-2., 2., 12)
        data.qpos[:] = original
        mj.mj_forward(model, data)
        before = np.asarray([data.geom_xpos[i] + offsets @ data.geom_xmat[i].reshape(3, 3).T
                             for i in range(model.ngeom)])
        data.qpos[:] = original + rng.uniform(-.0001, .0001, 12)
        mj.mj_forward(model, data)
        for i in range(model.ngeom):
            after = data.geom_xpos[i] + offsets @ data.geom_xmat[i].reshape(3, 3).T
            side = 'left' if i < 6 else 'right'
            assert float(np.max(np.linalg.norm(after - before[i], axis=1))) <= bounds[side]


def test_per_geometry_bounds_cover_large_errors_without_inflating_fixed_base():
    """固定底座不因末端误差膨胀；每个活动零件仍覆盖完整0.05rad误差盒."""
    loaded = envelope_model(fixed_base=True)
    mj = import_module('mujoco')
    bounds = geometry_error_displacement_bounds(loaded, .05)
    model, data = cast(Any, loaded.model), cast(Any, loaded.data)
    base_ids = [mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, side + '_base')
                for side in ('left', 'right')]
    for side, base in zip(('left', 'right'), base_ids):
        geom = int(np.flatnonzero(model.geom_bodyid == base)[0])
        self_bounds = bounds[side]
        assert self_bounds[geom] == 0.
        assert len(set(self_bounds.values())) > 2
    offsets = np.concatenate((np.eye(3), -np.eye(3))) * .03
    rng = np.random.default_rng(20261021)
    radii = {geom: float(model.geom_rbound[geom]) for arm in bounds.values() for geom in arm}
    chains = {}
    for geom in radii:
        body = int(model.geom_bodyid[geom])
        ancestors = []
        while body:
            ancestors.extend(int(j) for j in np.flatnonzero(model.jnt_bodyid == body))
            body = int(model.body_parentid[body])
        chains[geom] = tuple(reversed(ancestors))
    for trial in range(30):
        original = rng.uniform(-2., 2., 12)
        data.qpos[:] = original
        mj.mj_forward(model, data)
        posed = _pose_geometry_error_bounds(data, radii, chains, .05)
        normal = rng.normal(size=3)
        normal /= np.linalg.norm(normal)
        directions = (np.array([0., 0., 1.]), normal)
        projected = [_pose_geometry_error_bounds(data, radii, chains, .05, direction=n)
                     for n in directions]
        before = [data.geom_xpos[i] + offsets @ data.geom_xmat[i].reshape(3, 3).T
                  for i in range(model.ngeom)]
        perturbation = (np.full(12, .05 if trial == 0 else -.05) if trial < 2
                        else rng.uniform(-.05, .05, 12))
        data.qpos[:] = original + perturbation
        mj.mj_forward(model, data)
        for side_bounds in bounds.values():
            for geom, bound in side_bounds.items():
                after = data.geom_xpos[geom] + offsets @ data.geom_xmat[geom].reshape(3, 3).T
                assert np.max(np.linalg.norm(after - before[geom], axis=1)) <= (
                    min(bound, posed[geom]) + 1e-12)
                for normal, projection in zip(directions, projected):
                    assert np.max(np.abs((after - before[geom]) @ normal)) <= (
                        min(bound, posed[geom], projection[geom]) + 1e-12)


def test_error_bound_scales_and_rejects_unsupported_input():
    """误差增大会扩大包围，非法误差或非转动关节不返回通过结果."""
    loaded = envelope_model()
    assert joint_error_displacement_bounds(loaded, 0.) == {'left': 0., 'right': 0.}
    small = joint_error_displacement_bounds(loaded, .0001)
    large = joint_error_displacement_bounds(loaded, .0002)
    assert all(large[s] == 2 * small[s] > 0. for s in small)
    allowed = joint_error_displacement_bounds(loaded, .005)
    assert all(allowed[s] == pytest.approx(50 * small[s]) for s in small)
    observation = joint_error_displacement_bounds(loaded, .01)
    assert all(observation[s] == pytest.approx(100 * small[s]) for s in small)
    wider = joint_error_displacement_bounds(loaded, .02)
    assert all(wider[s] == pytest.approx(2 * observation[s]) for s in small)
    coarse = joint_error_displacement_bounds(loaded, .05)
    assert all(coarse[s] == pytest.approx(5 * observation[s]) for s in small)
    for value in (True, -.001, .05001, float('nan')):
        with pytest.raises(ValueError):
            joint_error_displacement_bounds(loaded, value)
    model = cast(Any, loaded.model)
    model.jnt_type[0] = 2
    with pytest.raises(ValueError, match='转动关节'):
        joint_error_displacement_bounds(loaded, .0001)


def test_invalid_collision_envelope_is_not_accepted():
    """缺少启用的碰撞体或损坏包围球时不能返回位移上界."""
    loaded = envelope_model()
    model = cast(Any, loaded.model)
    model.geom_rbound[0] = float('nan')
    with pytest.raises(ValueError, match='包围球'):
        joint_error_displacement_bounds(loaded, .0001)
    loaded = envelope_model()
    model = cast(Any, loaded.model)
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    with pytest.raises(ValueError, match='碰撞几何'):
        joint_error_displacement_bounds(loaded, .0001)


def test_unbounded_shape_and_multiple_joints_on_body_are_rejected():
    """无有限包围球或复合刚体关节不能套用单轴链的半径公式."""
    loaded = envelope_model()
    model = cast(Any, loaded.model)
    model.geom_rbound[0] = 0.
    with pytest.raises(ValueError, match='包围球'):
        joint_error_displacement_bounds(loaded, .0001)
    loaded = envelope_model()
    model = cast(Any, loaded.model)
    model.jnt_bodyid[1] = model.jnt_bodyid[0]
    with pytest.raises(ValueError, match='多个活动关节'):
        joint_error_displacement_bounds(loaded, .0001)


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

    def test_recovered_start_returns_to_clipped_saved_initial_with_real_geometry(self) -> None:
        """真实工程模型检查三段路径，恢复后出发点不同也须归回裁剪后的原始初值."""
        from archer_hardware.position_plan import load_limits

        limits = load_limits(Path(__file__).resolve().parents[1] / 'config/position_control.yaml')
        initial = (0., -1.55, 3.13, -1.55, 0., 1.5)
        goal = (.001, *initial[1:])
        task = {side: ContinuousPositionTrajectory.prepare(
            (0, 2000000000), (initial, goal), ((0.,) * 6,) * 2, ((0.,) * 6,) * 2,
            limits[side], .05,
        ) for side in ('left', 'right')}
        recovered = {side: initial for side in task}
        saved = {side: (0., -1.576, *initial[2:]) for side in task}
        targets = {side: limits[side].return_goal(saved[side]) for side in task}
        approach = prepare_approach(task, recovered, max_excursion=.05)
        home = prepare_return(task, targets, max_excursion=.05)
        with self.assertRaisesRegex(ValueError, '本轮保存起点'):
            check_task_geometry(task, approach=approach, return_home=home)
        result = check_task_geometry(task, approach=approach, return_home=home,
                                     return_targets=targets)
        self.assertTrue(result['sampled_cross_arm_table_passed'])
        self.assertEqual(set(result['phases']), {'APPROACH_FILE', 'TASK', 'RETURN_ACTUAL'})
        self.assertGreater(result['phases']['RETURN_ACTUAL']['samples_checked'], 2)
        self.assertIs(result['execution_ready'], False)
        core = check_task_geometry(task)
        with patch('archer_hardware.task_geometry.check_samples', wraps=check_samples) as samples:
            reused = check_task_geometry(task, approach=approach, return_home=home,
                                         return_targets=targets, verified_task_geometry=core)
        self.assertEqual(samples.call_count, 2)
        self.assertTrue(reused['task_geometry_reused'])
        self.assertEqual(reused['phases'], result['phases'])
        full = assemble_task_cycle(task, approach=approach, return_home=home,
                                   return_targets=targets, max_excursion=.1)
        fast = assemble_task_cycle(task, approach=approach, return_home=home,
                                   return_targets=targets, max_excursion=.1,
                                   verified_task_sha256=core['task_trajectory_sha256'])
        for side in task:
            self.assertEqual(full[side].times_ns, fast[side].times_ns)
            self.assertEqual(full[side].position_bounds, fast[side].position_bounds)
            for a, b in zip(full[side].polynomials, fast[side].polynomials):
                np.testing.assert_array_equal(a, b)
        for bad in ({**core, 'task_trajectory_sha256': 'changed'},
                    {**core, 'joint_error_rad': .001},
                    {**core, 'model_sha256': 'changed'}):
            with self.subTest(proof=bad['model_sha256']), self.assertRaises(ValueError):
                check_task_geometry(task, approach=approach, return_home=home,
                                    return_targets=targets, verified_task_geometry=bad)
        with self.assertRaisesRegex(ValueError, '主体轨迹已变化'):
            assemble_task_cycle(task, approach=approach, return_home=home,
                                return_targets=targets, max_excursion=.1,
                                verified_task_sha256='changed')
        for side in task:
            self.assertEqual(home[side].goal[1], -1.575)
            self.assertNotEqual(home[side].goal, approach[side].initial)

    def test_contact_candidates_match_all_geometry_bounds(self) -> None:
        """真实模型上，按碰撞候选计算与对所有零件计算得到相同的完整预检报告."""
        from archer_hardware.position_plan import load_limits
        from archer_mujoco.dual_model_loader import load_dual_arm

        limits = load_limits(Path(__file__).resolve().parents[1] / 'config/position_control.yaml')
        initial = (0., -1.55, 3.13, -1.55, 0., 1.5)
        goal = (.01, *initial[1:])
        task = {side: ContinuousPositionTrajectory.prepare(
            (0, 1000000000), (initial, goal), ((0.,) * 6,) * 2, ((0.,) * 6,) * 2,
            limits[side], .05,
        ) for side in ('left', 'right')}
        loaded = None

        def capture_model(**kwargs):
            nonlocal loaded
            loaded = load_dual_arm(**kwargs)
            return loaded

        def all_bounds(data, selected, chains, tolerance, *, direction=None):
            assert loaded is not None
            all_radii = {g: float(loaded.model.geom_rbound[g]) for g in chains}
            complete = _pose_geometry_error_bounds(
                data, all_radii, chains, tolerance, direction=direction)
            return {g: complete[g] for g in selected}

        for error in (.001, .05):
            expected = check_task_geometry(task, joint_error_rad=error,
                                           nominal_feedback_guarded=True)
            with patch('archer_mujoco.dual_model_loader.load_dual_arm', capture_model), \
                    patch('archer_hardware.task_geometry._pose_geometry_error_bounds', all_bounds):
                reference = check_task_geometry(task, joint_error_rad=error,
                                                nominal_feedback_guarded=True)
            self.assertEqual(expected, reference)

    def test_explicit_return_targets_cannot_hide_wrong_home(self) -> None:
        """明确目标仍需双臂、名义限位及归位段匹配，不放宽错误或缺失归位."""
        saved = {'left': (-.01,) * 6, 'right': (-.02,) * 6}
        approach = prepare_approach(self.plans, saved, max_excursion=.05)
        home = prepare_return(self.plans, saved, max_excursion=.05)
        with patch('archer_mujoco.dual_model_loader.load_dual_arm') as loader:
            for targets in ({'left': saved['left']},
                            {**saved, 'left': (1.001,) * 6},
                            {**saved, 'right': (.001,) * 6}):
                with self.subTest(targets=targets), self.assertRaises(ValueError):
                    check_task_geometry(self.plans, approach=approach, return_home=home,
                                        return_targets=targets)
            with self.assertRaisesRegex(ValueError, '完整归位段'):
                check_task_geometry(self.plans, return_targets=saved)
            loader.assert_not_called()

    def test_boundary_error_intersection_keeps_collision_clearance(self) -> None:
        """边界起点允许误差集合取名义范围交集，碰撞膨胀仍覆盖完整误差盒."""
        from archer_hardware.position_plan import load_limits
        limits = load_limits(Path(__file__).resolve().parents[1] / 'config/position_control.yaml')
        initial = (0., -1.5748, 3.13, -1.55, 0., 1.5)
        goal = (.001, *initial[1:])
        task = {side: ContinuousPositionTrajectory.prepare(
            (0, 2_000_000_000), (initial, goal), ((0.,) * 6,) * 2,
            ((0.,) * 6,) * 2, limits[side], .05) for side in ('left', 'right')}
        with self.assertRaisesRegex(ValueError, 'URDF限位'):
            check_task_geometry(task, joint_error_rad=.001)
        result = check_task_geometry(task, joint_error_rad=.001, nominal_feedback_guarded=True)
        self.assertTrue(result['sampled_cross_arm_table_passed'])
        self.assertTrue(result['nominal_feedback_required'])
        self.assertEqual(result['joint_error_rad'], .001)
        self.assertGreater(result['checked_cross_arm_clearance_m'], .01)
        self.assertGreater(result['checked_table_clearance_m'], .001)
        self.assertFalse(result['execution_ready'])
        approach = prepare_approach(task, {s: p.initial for s, p in task.items()},
                                    max_excursion=.05)
        startup = check_task_geometry(
            task, approach=approach, joint_error_rad=.001,
            startup_joint_error_rad=.01, nominal_feedback_guarded=True)
        self.assertEqual(startup['joint_error_rad'], .001)
        self.assertEqual(startup['startup_joint_error_rad'], .01)
        self.assertGreater(startup['startup_displacement_bounds_m']['left'],
                           startup['displacement_bounds_m']['left'])
        widened = {s: replace(p, limits=replace(p.limits, lower=(-3.,) * 6))
                   for s, p in task.items()}
        with self.assertRaisesRegex(ValueError, '反馈限位必须位于URDF'):
            check_task_geometry(widened, joint_error_rad=.001, nominal_feedback_guarded=True)
        bounds = ((-3., 0.),) + task['left'].position_bounds[1:]
        with self.assertRaisesRegex(ValueError, 'URDF限位'):
            check_task_geometry({**task, 'left': replace(task['left'], position_bounds=bounds)},
                                joint_error_rad=.001, nominal_feedback_guarded=True)


class TaskCycleTests(unittest.TestCase):
    """验证完整时间轴保留原多项式，并拒绝错误边界及整体超限."""

    def setUp(self) -> None:
        """建立文件路径及不同于文件起点的本轮起点."""
        self.limits = PositionLimits((-1.,) * 6, (1.,) * 6, (.05,) * 6, (.1,) * 6, .005)
        plan = ContinuousPositionTrajectory.prepare(
            (0, 2000000000), ((0.,) * 6, (.02,) * 6),
            ((0.,) * 6,) * 2, ((0.,) * 6,) * 2, self.limits, .05,
        )
        self.plans = {'left': plan, 'right': plan}
        self.saved = {'left': [-.01] * 6, 'right': [-.02] * 6}
        self.approach = prepare_approach(self.plans, self.saved, max_excursion=.05)
        self.home = prepare_return(self.plans, self.saved, max_excursion=.05)

    def cycle(self) -> dict[str, ContinuousPositionTrajectory]:
        """拼接只在内存中运行，不调用设备或ROS接口."""
        return assemble_task_cycle(self.plans, approach=self.approach,
                                   return_home=self.home, max_excursion=.05)

    def test_cycle_samples_match_each_original_phase(self) -> None:
        """两次衔接及各段内部的位置、速度、加速度均与原段相同."""
        cycle = self.cycle()
        for side, combined in cycle.items():
            self.assertEqual(combined.initial, tuple(self.saved[side]))
            self.assertEqual(combined.goal, tuple(self.saved[side]))
            offset = 0.
            for phase in (self.approach, self.plans, self.home):
                plan = phase[side]
                for fraction in (0., .1, .37, .8, 1.):
                    for order in (0, 1, 2):
                        np.testing.assert_allclose(
                            combined.sample(offset + fraction * plan.duration, order),
                            plan.sample(fraction * plan.duration, order), rtol=0., atol=1e-12,
                        )
                offset += plan.duration
            self.assertAlmostEqual(combined.duration, offset)
        self.assertEqual(cycle['left'].times_ns, cycle['right'].times_ns)

    def test_cycle_copies_and_freezes_exact_polynomials(self) -> None:
        """复制系数保持逐字节相同，后续修改输入数组不会改变拼接结果."""
        cycle = self.cycle()
        for side in ('left', 'right'):
            originals = tuple(c for p in (self.approach, self.plans, self.home)
                              for c in p[side].polynomials)
            for original, copied in zip(originals, cycle[side].polynomials):
                self.assertEqual(original.tobytes(), copied.tobytes())
                self.assertFalse(np.shares_memory(original, copied))
                self.assertFalse(copied.flags.writeable)

    def test_cycle_rechecks_excursion_from_saved_start(self) -> None:
        """文件自身幅度合格不代表相对本轮起点仍合格，必须整轮复核."""
        with self.assertRaisesRegex(ValueError, '本轮起点超过'):
            assemble_task_cycle(self.plans, approach=self.approach,
                                return_home=self.home, max_excursion=.025)

    def test_cycle_rejects_mismatched_home_limits_and_times(self) -> None:
        """拒绝错误归位目标、阶段限位不同和双臂时间轴不同."""
        for side in ('left', 'right'):
            cases = (
                replace(self.home[side], goal=(0.,) * 6),
                replace(self.home[side], limits=replace(self.limits, max_velocity=(.06,) * 6)),
                replace(self.home[side], times_ns=(0, self.home[side].times_ns[-1] + 1)),
            )
            for bad in cases:
                with self.subTest(side=side, bad=bad), self.assertRaises(ValueError):
                    assemble_task_cycle(self.plans, approach=self.approach,
                                        return_home={**self.home, side: bad}, max_excursion=.05)

    def test_cycle_rejects_metadata_hiding_wrong_polynomial_endpoint(self) -> None:
        """端点字段衔接不能掩盖系数中实际存在的位置跳变."""
        c = self.plans['left'].polynomials[0].copy()
        c[:, 0] += .001
        bad = replace(self.plans['left'], polynomials=(c,))
        with self.assertRaisesRegex(ValueError, '端点与实际'):
            assemble_task_cycle({**self.plans, 'left': bad}, approach=self.approach,
                                return_home=self.home, max_excursion=.05)

    def test_cycle_rejects_internal_derivative_jump(self) -> None:
        """逐段约束合格仍必须检查内部相邻段的速度连续性."""
        p = self.plans['left']
        c = p.polynomials[0].copy()
        c[:, 0] += .02
        c[:, 1] += .0001
        c[:, 2] -= .0003
        c[:, 3] += .0003
        c[:, 4] -= .0001
        bad = replace(p, times_ns=(0, 2000000000, 4000000000),
                      polynomials=(p.polynomials[0], c), goal=(.04,) * 6)
        plans = {'left': bad, 'right': bad}
        approach = prepare_approach(plans, self.saved, max_excursion=.1)
        home = prepare_return(plans, self.saved, max_excursion=.1)
        with self.assertRaisesRegex(ValueError, '不连续'):
            assemble_task_cycle(plans, approach=approach, return_home=home, max_excursion=.1)

    def test_cycle_rejects_total_duration_over_budget(self) -> None:
        """各段单独不超过一小时，拼接后超时仍须整轮拒绝."""
        home = {side: replace(p, times_ns=(0, 3600 * 10 ** 9))
                for side, p in self.home.items()}
        with self.assertRaisesRegex(ValueError, '一小时'):
            assemble_task_cycle(self.plans, approach=self.approach,
                                return_home=home, max_excursion=.05)

    def test_geometry_checked_cycle_refuses_start_rebuild(self) -> None:
        """几何已检路径即使仅有微小起点变化也拒绝，不沿用普通轨迹的首段重建."""
        plan = self.cycle()['left']
        self.assertTrue(plan.preserve_path)
        self.assertIs(plan.aligned_start(plan.initial, self.limits, .05), plan)
        original = plan.polynomials[0].tobytes()
        with self.assertRaisesRegex(ValueError, '禁止重建首段'):
            plan.aligned_start(tuple(x + 1e-7 for x in plan.initial), self.limits, .05)
        self.assertEqual(original, plan.polynomials[0].tobytes())

    def test_cycle_and_geometry_respond_to_cancellation(self) -> None:
        """取消后不继续生成完整对象，也不执行部分几何检查后返回通过."""
        with self.assertRaisesRegex(ValueError, '取消'):
            assemble_task_cycle(self.plans, approach=self.approach,
                                return_home=self.home, max_excursion=.05, cancelled=lambda: True)
        calls = []
        with self.assertRaisesRegex(ValueError, '取消'):
            check_samples(self.plans, lambda a, b: calls.append(1) or None,
                          cancelled=lambda: True)
        self.assertEqual(calls, [])

    def test_cycle_cli_report_keeps_execution_closed(self) -> None:
        """报告给出完整时间轴而不授权运动，拼接失败时不继续调用几何检查."""
        from archer_hardware.task_file import ReviewedTask, main

        reviewed = ReviewedTask(self.plans, {
            'task_files_verified': True, 'execution_ready': False, 'commands_sent': False,
            'actual_start_verified': False,
        })
        with TemporaryDirectory() as folder:
            start = Path(folder) / 'start.json'
            start.write_text(json.dumps(self.saved), encoding='utf-8')
            args = ['--run-dir', folder, '--limits-file', 'unused', '--prepared-file', 'unused',
                    '--report-file', 'unused', '--max-excursion', '.05', '--geometry',
                    '--start-file', str(start), '--approach-max-excursion', '.05']
            with patch('archer_hardware.task_file.load_reviewed_task', return_value=reviewed), \
                    patch('archer_hardware.task_geometry.check_task_geometry',
                          return_value={'sampled_cross_arm_table_passed': True}) as geometry:
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main(args), 0)
                report = json.loads(output.getvalue())
                self.assertFalse(report['execution_ready'])
                self.assertFalse(report['commands_sent'])
                self.assertFalse(report['actual_start_verified'])
                cycle = report['assembled_cycle']
                self.assertFalse(cycle['live_feedback_verified'])
                self.assertFalse(cycle['execution_ready'])
                self.assertEqual(cycle['duration_s'], self.cycle()['left'].duration)
                self.assertEqual(cycle['phase_start_ns']['TASK'],
                                 self.approach['left'].times_ns[-1])
                geometry.assert_called_once()
                geometry.reset_mock()
                args[args.index('--max-excursion') + 1] = '.025'
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(main(args), 1)
                geometry.assert_not_called()


if __name__ == '__main__':
    unittest.main()
