"""离线拼接接近、文件任务与归位轨迹，并对连续插值做离散几何预检."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import hashlib
import math
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .position_plan import ContinuousPositionTrajectory, positive


def _stationary_transition(
    task: Mapping[str, ContinuousPositionTrajectory], start: object,
    goals: Mapping[str, tuple[float, ...]], *, max_excursion: float, minimum_duration: float,
) -> dict[str, ContinuousPositionTrajectory]:
    """复用零速五次过渡计算，两臂共用更严格一侧的限速时长."""
    if set(task) != {'left', 'right'} or not isinstance(start, dict):
        raise ValueError('过渡段需要双臂任务和双侧起点映射')
    if set(start) != {'left', 'right'}:
        raise ValueError('起点必须明确包含left和right，不能用固定初值代替')
    if task['left'].times_ns != task['right'].times_ns:
        raise ValueError('过渡段需要原任务共用时间轴')
    excursion = positive(max_excursion, 'transition_max_excursion')
    required = positive(minimum_duration, 'minimum_duration')
    actual: dict[str, tuple[float, ...]] = {}
    targets: dict[str, tuple[float, ...]] = {}
    for side in ('left', 'right'):
        plan = task[side]
        feedback = plan.limits.check_feedback(start[side])
        if not plan.limits.within_nominal(feedback):
            raise ValueError(side + '初值略超名义限位，需要先单独进行限位恢复')
        actual[side] = feedback
        targets[side] = plan.limits.check_positions(goals[side])
        for a, b, vmax, amax in zip(feedback, targets[side], plan.limits.max_velocity,
                                    plan.limits.max_acceleration):
            distance = abs(b - a)
            if distance > excursion:
                raise ValueError(side + '过渡段超过明确允许的位移范围')
            required = max(required, 1.875 * distance / vmax,
                           math.sqrt((10. * math.sqrt(3.) / 3.) * distance / amax))
    duration_ns = math.ceil(required * 1.05 * 1e9)
    if duration_ns > 3600 * 10 ** 9:
        raise ValueError('过渡段限速后超过一小时')
    return {side: ContinuousPositionTrajectory.prepare(
        (0, duration_ns), (actual[side], targets[side]),
        ((0.,) * 6,) * 2, ((0.,) * 6,) * 2, task[side].limits, excursion,
    ) for side in ('left', 'right')}


def _check_stationary_boundary(
    task: Mapping[str, ContinuousPositionTrajectory], *, last: bool,
) -> None:
    """衔接处必须零速零加速度，不能忽略文件实际多项式导数."""
    from archer_control.continuous_trajectory import evaluate

    if set(task) != {'left', 'right'}:
        raise ValueError('衔接需要双臂任务')
    for plan in task.values():
        i = len(plan.polynomials) - 1 if last else 0
        seconds = (plan.times_ns[i + 1] - plan.times_ns[i]) / 1e9
        if any(np.max(np.abs(evaluate(plan.polynomials[i], float(last), seconds, order)))
               > 1e-12 for order in (1, 2)):
            raise ValueError('原文件衔接点必须是零速零加速度')


def prepare_approach(
    task: Mapping[str, ContinuousPositionTrajectory], start: object, *,
    max_excursion: float, minimum_duration: float = 1.,
) -> dict[str, ContinuousPositionTrajectory]:
    """从本轮提供起点到文件起点独立规划，不改写任务，也不表示反馈已实时验证."""
    _check_stationary_boundary(task, last=False)
    return _stationary_transition(task, start, {s: p.initial for s, p in task.items()},
                                  max_excursion=max_excursion, minimum_duration=minimum_duration)


def prepare_return(
    task: Mapping[str, ContinuousPositionTrajectory], saved_start: object, *,
    max_excursion: float, minimum_duration: float = 1.,
) -> dict[str, ContinuousPositionTrajectory]:
    """文件播放到底后返回本轮保存起点，独立检查归位段且不增加记录文件."""
    _check_stationary_boundary(task, last=True)
    if not isinstance(saved_start, dict) or set(saved_start) != {'left', 'right'}:
        raise ValueError('归位必须使用本轮保存的双侧实际起点')
    goals = {side: task[side].limits.check_positions(saved_start[side])
             for side in ('left', 'right')}
    return _stationary_transition(task, {s: p.goal for s, p in task.items()}, goals,
                                  max_excursion=max_excursion, minimum_duration=minimum_duration)


def assemble_task_cycle(
    task: Mapping[str, ContinuousPositionTrajectory], *,
    approach: Mapping[str, ContinuousPositionTrajectory],
    return_home: Mapping[str, ContinuousPositionTrajectory], max_excursion: float,
) -> dict[str, ContinuousPositionTrajectory]:
    """只平移段时间并复制原多项式；离线结果不是运动许可，起点仍须实时核对."""
    from archer_control.continuous_trajectory import evaluate

    excursion = positive(max_excursion, 'cycle_max_excursion')
    phases = (approach, task, return_home)
    for phase in phases:
        if set(phase) != {'left', 'right'}:
            raise ValueError('完整任务的每一阶段必须包含双臂')
        if phase['left'].times_ns != phase['right'].times_ns:
            raise ValueError('完整任务的每一阶段必须共用双臂时间轴')
        for plan in phase.values():
            times = plan.times_ns
            if (len(times) < 2 or any(type(t) is not int for t in times)
                    or times[0] != 0 or any(b <= a for a, b in zip(times, times[1:]))
                    or len(plan.polynomials) != len(times) - 1):
                raise ValueError('完整任务存在无效时间轴或多项式数量')
        _check_stationary_boundary(phase, last=False)
        _check_stationary_boundary(phase, last=True)
    times = [0]
    for phase in phases:
        offset = times[-1]
        times.extend(offset + t for t in phase['left'].times_ns[1:])
    if len(times) > 20000 or times[-1] > 3600 * 10 ** 9:
        raise ValueError('完整任务超过20000个时间点或一小时，不允许截断执行')
    result: dict[str, ContinuousPositionTrajectory] = {}
    for side in ('left', 'right'):
        first, body, home = (phase[side] for phase in phases)
        if first.limits != body.limits or home.limits != body.limits:
            raise ValueError(side + '完整任务各阶段限位配置不一致')
        if first.goal != body.initial or body.goal != home.initial:
            raise ValueError(side + '完整任务阶段位置没有精确衔接')
        if first.initial != home.goal:
            raise ValueError(side + '归位目标不是本轮保存起点')
        lower = np.full(6, np.inf)
        upper = np.full(6, -np.inf)
        polynomials: list[NDArray[np.float64]] = []
        previous: tuple[NDArray[np.float64], ...] | None = None
        for phase in phases:
            plan = phase[side]
            for i, original in enumerate(plan.polynomials):
                c = np.array(original, dtype=np.float64, copy=True)
                if c.shape != (6, 6):
                    raise ValueError(side + '完整任务多项式必须为六关节五次系数')
                seconds = (plan.times_ns[i + 1] - plan.times_ns[i]) / 1e9
                bounds = ContinuousPositionTrajectory._check_segment(c, seconds, body.limits)
                start = tuple(evaluate(c, 0., seconds, order) for order in (0, 1, 2))
                end = tuple(evaluate(c, 1., seconds, order) for order in (0, 1, 2))
                if (i == 0 and not np.allclose(start[0], plan.initial, rtol=0., atol=1e-12)
                        or i == len(plan.polynomials) - 1
                        and not np.allclose(end[0], plan.goal, rtol=0., atol=1e-12)):
                    raise ValueError(side + '阶段端点与实际多项式不一致')
                if previous is not None and any(
                    not np.allclose(a, b, rtol=0., atol=1e-10)
                    for a, b in zip(previous, start)
                ):
                    raise ValueError(side + '完整任务位置、速度或加速度不连续')
                previous = end
                lower = np.minimum(lower, bounds[:, 0])
                upper = np.maximum(upper, bounds[:, 1])
                c.setflags(write=False)
                polynomials.append(c)
        initial = np.asarray(first.initial, dtype=np.float64)
        if (np.any(np.abs(lower - initial) > excursion + 1e-12)
                or np.any(np.abs(upper - initial) > excursion + 1e-12)):
            raise ValueError(side + '完整任务相对本轮起点超过允许位移')
        result[side] = ContinuousPositionTrajectory(
            tuple(times), tuple(polynomials), first.initial, home.goal,
            tuple((float(a), float(b)) for a, b in zip(lower, upper)), body.limits,
        )
    return result


def geometry_schedule(
    plans: Mapping[str, ContinuousPositionTrajectory], *, max_samples: int = 100000,
    max_time_step: float = .1, max_joint_step: float = .005,
) -> tuple[int, Iterator[int]]:
    """按解析速度峰值加密，限制相邻点时间和关节行程；不是连续碰撞证明."""
    from archer_control.continuous_trajectory import _extrema

    if set(plans) != {'left', 'right'}:
        raise ValueError('几何检查必须同时提供两臂')
    left, right = plans['left'], plans['right']
    if left.times_ns != right.times_ns:
        raise ValueError('几何检查需要相同的双臂时间轴')
    dt = positive(max_time_step, 'max_time_step')
    dq = positive(max_joint_step, 'max_joint_step')
    if dt > .1 or dq > .01:
        raise ValueError('采样时间步长不能超过0.1秒，关节行程不能超过0.01rad')
    if type(max_samples) is not int or not 2 <= max_samples <= 250000:
        raise ValueError('几何预检点数上限必须在2到250000之间')
    divisions: list[int] = []
    count = 1
    for i, (a, b) in enumerate(zip(left.times_ns, left.times_ns[1:])):
        seconds = (b - a) / 1e9
        peak = max(float(np.max(np.abs(_extrema(p.polynomials[i], seconds, 1))))
                   for p in (left, right))
        steps = max(1, math.ceil(seconds / dt),
                    math.ceil(peak * seconds * (1. + 1e-9) / dq))
        count += steps
        if count > max_samples:
            raise ValueError('几何采样点超过预算，拒绝降低精度或只检查部分轨迹')
        divisions.append(steps)

    def stamps() -> Iterator[int]:
        yield left.times_ns[0]
        for a, b, steps in zip(left.times_ns, left.times_ns[1:], divisions):
            for k in range(1, steps + 1):
                yield a + (b - a) * k // steps

    return count, stamps()


def check_samples(
    plans: Mapping[str, ContinuousPositionTrajectory],
    failure: Callable[[tuple[float, ...], tuple[float, ...]], str | None], *,
    max_samples: int = 100000,
) -> dict[str, Any]:
    """检查全部加密采样，异常含时间定位；失败时不返回通过标记."""
    count, stamps = geometry_schedule(plans, max_samples=max_samples)
    for stamp in stamps:
        elapsed = stamp / 1e9
        problem = failure(plans['left'].sample(elapsed), plans['right'].sample(elapsed))
        if problem:
            raise ValueError(f'几何采样 t={elapsed:.9f}s：{problem}')
    return {'samples_checked': count, 'duration_s': plans['left'].duration,
            'max_time_step_s': .1, 'max_joint_travel_rad': .005}


class GeometryModel(Protocol):
    geom_margin: NDArray[np.float64]
    geom_pos: NDArray[np.float64]
    geom_bodyid: NDArray[np.int32]
    geom_contype: NDArray[np.int32]
    geom_conaffinity: NDArray[np.int32]


class NameKinds(Protocol):
    mjOBJ_GEOM: object
    mjOBJ_BODY: object


class GeometryAPI(Protocol):
    mjtObj: NameKinds
    def mj_name2id(self, model: object, kind: object, name: str) -> int: ...
    def mj_id2name(self, model: object, kind: object, index: int) -> str | None: ...


def check_task_geometry(
    task: Mapping[str, ContinuousPositionTrajectory], *,
    approach: Mapping[str, ContinuousPositionTrajectory] | None = None,
    return_home: Mapping[str, ContinuousPositionTrajectory] | None = None,
    max_samples: int = 100000,
) -> dict[str, Any]:
    """用当前工程安装模型检查跨臂和台面；不含工件、自碰撞或线缆."""
    from importlib import import_module
    from archer_control.kinematics import ArmKinematics
    from archer_control.model_checks import CollisionData, cross_arm_failure, validate_tool_chain
    from archer_mujoco.dual_model_loader import load_dual_arm
    from archer_mujoco.model_loader import mujoco

    phases = ([('TASK', task)] if approach is None else
              [('APPROACH_FILE', approach), ('TASK', task)])
    if return_home is not None:
        phases.append(('RETURN_ACTUAL', return_home))
    # 先估算整个会话的预算，再加载模型；不能只检查预算内的前半段。
    total = sum(geometry_schedule(p, max_samples=max_samples)[0] for _, p in phases)
    if total > max_samples:
        raise ValueError('接近段与任务合计超过几何采样预算')
    if approach is not None:
        for side in ('left', 'right'):
            if approach[side].goal != task[side].initial:
                raise ValueError('接近段末点与文件起点不一致')
    if return_home is not None:
        for side in ('left', 'right'):
            if return_home[side].initial != task[side].goal:
                raise ValueError('归位段起点与文件末点不一致')
            if approach is not None and return_home[side].goal != approach[side].initial:
                raise ValueError('归位末点与本轮保存起点不一致')
    for side in ('left', 'right'):
        validate_tool_chain(ArmKinematics.from_description(
            side, (0., 0., .147021696), (0., 0., 0.), (0., 0., 0.)))
    loaded = load_dual_arm(
        left_base_xyz=(-.21, 0., 0.), right_base_xyz=(.21, 0., 0.),
        left_base_rpy=(0., 0., np.pi / 2.), right_base_rpy=(0., 0., np.pi / 2.),
        tcp_xyz=(0., 0., .147021696), enable_contacts=True,
    )
    geometry = cast(GeometryModel, loaded.model)
    api = cast(GeometryAPI, import_module('mujoco'))
    floor = api.mj_name2id(loaded.model, api.mjtObj.mjOBJ_GEOM, 'floor')
    if floor < 0:
        raise ValueError('模型缺少台面')
    if not (geometry.geom_contype[floor] or geometry.geom_conaffinity[floor]):
        raise ValueError('台面未启用碰撞检查')
    geometry.geom_pos[floor, 2] = -.01
    clearance = .01
    geometry.geom_margin[:] = clearance + .001
    for side, arm in loaded.arms.items():
        for _, plans in phases:
            bounds = np.asarray(plans[side].position_bounds)
            if (np.any(bounds[:, 0] < arm.joint_limits[:, 0] - 1e-10)
                    or np.any(bounds[:, 1] > arm.joint_limits[:, 1] + 1e-10)):
                raise ValueError(side + '路径超出当前几何模型的URDF限位')
        active = [i for i, body in enumerate(geometry.geom_bodyid)
                  if (api.mj_id2name(loaded.model, api.mjtObj.mjOBJ_BODY, int(body))
                      or '').startswith(side + '_')
                  and (geometry.geom_contype[i] or geometry.geom_conaffinity[i])]
        if not active:
            raise ValueError(side + '模型没有启用碰撞几何')

    def failure(left: tuple[float, ...], right: tuple[float, ...]) -> str | None:
        for side, q in (('left', left), ('right', right)):
            loaded.data.qpos[loaded.arms[side].qpos_indices] = q
        mujoco.mj_forward(loaded.model, loaded.data)
        problem = cross_arm_failure(loaded, clearance)
        if problem:
            return '跨臂间隙不足10mm：' + problem
        data = cast(CollisionData, loaded.data)
        for i in range(data.ncon):
            contact = data.contact[i]
            if floor in (contact.geom1, contact.geom2) and contact.dist < .001:
                return '机器人与台面间隙不足1mm'
        return None

    measured = {}
    for phase, plans in phases:
        try:
            measured[phase] = check_samples(plans, failure, max_samples=max_samples)
        except ValueError as error:
            raise ValueError(phase + '：' + str(error)) from error
    fingerprint = hashlib.sha256()
    # 哈希覆盖加载后的网格和运动学几何，不使用临时文件路径作为模型身份。
    for name in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_bodyid',
                 'geom_contype', 'geom_conaffinity', 'body_pos', 'body_quat', 'body_parentid',
                 'jnt_pos', 'jnt_axis', 'mesh_vert', 'mesh_face'):
        value = np.asarray(getattr(loaded.model, name))
        fingerprint.update(name.encode('ascii'))
        fingerprint.update(str((value.shape, value.dtype)).encode('ascii'))
        fingerprint.update(value.tobytes())
    return {'sampled_cross_arm_table_passed': True, 'phases': measured,
            'samples_checked': total, 'model_sha256': fingerprint.hexdigest(),
            'cross_arm_clearance_m': clearance, 'table_clearance_m': .001,
            'table_z_m': -.01, 'interval_collision_proven': False,
            'self_collision_checked': False, 'object_collision_checked': False,
            'scope': '当前工程刚性GP80与双臂模型的离散跨臂和台面检查，不是连续碰撞证明'}
