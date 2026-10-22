"""离线拼接接近、文件任务与归位轨迹，并对连续插值做离散几何预检."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
import hashlib
import math
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .position_plan import ContinuousPositionTrajectory, MAX_TASK_JOINT_ERROR_RAD, positive

if TYPE_CHECKING:
    from archer_mujoco.dual_model_loader import LoadedDualArm


def trajectory_fingerprint(task: Mapping[str, ContinuousPositionTrajectory]) -> str:
    """绑定轨迹系数、时间轴、端点、范围和限位，用于本次启动内复用检查。"""
    if set(task) != {'left', 'right'}:
        raise ValueError('主体轨迹必须同时包含两臂')
    digest = hashlib.sha256(b'file_task_geometry_v1')
    for side in ('left', 'right'):
        plan = task[side]
        digest.update(side.encode('ascii'))
        digest.update(repr((plan.times_ns, plan.initial, plan.goal,
                            plan.position_bounds, plan.limits)).encode('utf-8'))
        for polynomial in plan.polynomials:
            digest.update(np.asarray(polynomial, dtype=np.float64).tobytes())
    return digest.hexdigest()


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
    cancelled: Callable[[], bool] | None = None,
    return_targets: Mapping[str, tuple[float, ...]] | None = None,
    verified_task_sha256: str | None = None,
) -> dict[str, ContinuousPositionTrajectory]:
    """只平移段时间并复制原多项式；离线结果不是运动许可，起点仍须实时核对."""
    from archer_control.continuous_trajectory import evaluate

    excursion = positive(max_excursion, 'cycle_max_excursion')
    if verified_task_sha256 is not None and verified_task_sha256 != trajectory_fingerprint(task):
        raise ValueError('主体轨迹已变化，禁止复用数值检查')
    if return_targets is not None and set(return_targets) != {'left', 'right'}:
        raise ValueError('完整任务必须明确双臂归位目标')
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
        expected_home = first.initial if return_targets is None else return_targets[side]
        body.limits.check_positions(expected_home)
        if expected_home != home.goal:
            raise ValueError(side + '归位目标不是本轮保存起点')
        lower = np.full(6, np.inf)
        upper = np.full(6, -np.inf)
        polynomials: list[NDArray[np.float64]] = []
        previous: tuple[NDArray[np.float64], ...] | None = None
        for phase in phases:
            plan = phase[side]
            if phase is task and verified_task_sha256 is not None:
                seconds = (plan.times_ns[1] - plan.times_ns[0]) / 1e9
                start = tuple(evaluate(plan.polynomials[0], 0., seconds, o) for o in (0, 1, 2))
                if previous is not None and any(
                    not np.allclose(a, b, rtol=0., atol=1e-10)
                    for a, b in zip(previous, start)
                ):
                    raise ValueError(side + '主体起点没有连续衔接')
                bounds = np.asarray(plan.position_bounds)
                lower = np.minimum(lower, bounds[:, 0])
                upper = np.maximum(upper, bounds[:, 1])
                for original in plan.polynomials:
                    c = np.array(original, dtype=np.float64, copy=True)
                    c.setflags(write=False)
                    polynomials.append(c)
                seconds = (plan.times_ns[-1] - plan.times_ns[-2]) / 1e9
                previous = tuple(evaluate(plan.polynomials[-1], 1., seconds, o)
                                 for o in (0, 1, 2))
                continue
            for i, original in enumerate(plan.polynomials):
                if cancelled is not None and cancelled():
                    raise ValueError('完整路径拼接已取消')
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
            preserve_path=True,
        )
    return result


def geometry_schedule(
    plans: Mapping[str, ContinuousPositionTrajectory], *, max_samples: int = 100000,
    max_time_step: float = .1, max_joint_step: float = .005,
    cancelled: Callable[[], bool] | None = None,
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
        if cancelled is not None and cancelled():
            raise ValueError('几何采样准备已取消')
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
    cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """检查全部加密采样，异常含时间定位；失败时不返回通过标记."""
    count, stamps = geometry_schedule(plans, max_samples=max_samples, cancelled=cancelled)
    for stamp in stamps:
        if cancelled is not None and cancelled():
            raise ValueError('几何检查已取消')
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


class EnvelopeModel(GeometryModel, Protocol):
    """只读取模型的包围球、关节与固定安装链，不修改限位或姿态."""

    geom_rbound: NDArray[np.float64]
    geom_type: NDArray[np.int32]
    body_parentid: NDArray[np.int32]
    body_pos: NDArray[np.float64]
    jnt_pos: NDArray[np.float64]
    jnt_type: NDArray[np.int32]
    jnt_bodyid: NDArray[np.int32]
    jnt_qposadr: NDArray[np.int32]


def _joint_error(value: float) -> float:
    """几何检查的明确误差上界，不是厂商精度或自动执行许可."""
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or not 0. <= value <= MAX_TASK_JOINT_ERROR_RAD):
        raise ValueError('关节误差上界必须在0到0.05rad之间')
    return float(value)


def geometry_error_displacement_bounds(
    loaded: LoadedDualArm, tolerance: float,
) -> dict[str, dict[int, float]]:
    """分别覆盖每个碰撞零件的位移，仅累加真正影响该零件的上游转动关节。"""
    from importlib import import_module

    delta = _joint_error(tolerance)
    geometry = cast(EnvelopeModel, loaded.model)
    api = cast(GeometryAPI, import_module('mujoco'))
    if set(loaded.arms) != {'left', 'right'}:
        raise ValueError('误差包围检查需要两侧六轴机械臂')
    result: dict[str, dict[int, float]] = {}
    for side, arm in loaded.arms.items():
        identifiers = []
        for qpos in arm.qpos_indices:
            matches = np.flatnonzero(geometry.jnt_qposadr == qpos)
            if len(matches) != 1 or int(geometry.jnt_type[int(matches[0])]) != 3:
                raise ValueError('误差包围仅支持已核对的单自由度转动关节')
            identifiers.append(int(matches[0]))
        if len(identifiers) != 6 or len(set(identifiers)) != 6:
            raise ValueError('误差包围需要六个唯一关节')
        bounds: dict[int, float] = {}
        for geom, body_value in enumerate(geometry.geom_bodyid):
            body = int(body_value)
            name = api.mj_id2name(loaded.model, api.mjtObj.mjOBJ_BODY, body) or ''
            if not name.startswith(side + '_') or not (
                geometry.geom_contype[geom] or geometry.geom_conaffinity[geom]
            ):
                continue
            radii = [0.] * 6
            radius = float(geometry.geom_rbound[geom])
            if not math.isfinite(radius) or radius <= 0.:
                raise ValueError('模型碰撞几何包围球无效')
            distance = float(np.linalg.norm(geometry.geom_pos[geom])) + radius
            visited = set()
            while body != 0:
                if body in visited or not 0 < body < len(geometry.body_parentid):
                    raise ValueError('模型父子链无效')
                visited.add(body)
                body_joints = np.flatnonzero(geometry.jnt_bodyid == body)
                if len(body_joints) > 1:
                    raise ValueError('误差包围不支持同一刚体上的多个活动关节')
                for joint in body_joints:
                    index = int(joint)
                    if index not in identifiers:
                        raise ValueError('碰撞链含未纳入误差上界的活动关节')
                    lever = distance + float(np.linalg.norm(geometry.jnt_pos[index]))
                    if not math.isfinite(lever):
                        raise ValueError('模型运动学长度包含非有限值')
                    slot = identifiers.index(index)
                    radii[slot] = max(radii[slot], lever)
                distance += float(np.linalg.norm(geometry.body_pos[body]))
                body = int(geometry.body_parentid[body])
            # 上游关节的半径使用固定安装长度与包围球构造，与当前姿态无关。
            # 依次改变各关节时，每一步位移不超过R*delta；三角不等式覆盖完整误差盒。
            bounds[geom] = sum(radii) * delta
        if not bounds:
            raise ValueError(side + '模型没有启用碰撞几何')
        result[side] = bounds
    return result


def joint_error_displacement_bounds(
    loaded: LoadedDualArm, tolerance: float,
) -> dict[str, float]:
    """返回各侧所有零件位移上界的最大值，供已有报告和检查调用。"""
    return {side: max(bounds.values()) for side, bounds in
            geometry_error_displacement_bounds(loaded, tolerance).items()}


class EnvelopeData(Protocol):
    """当前姿态下的碰撞零件中心及世界关节轴。"""

    geom_xpos: NDArray[np.float64]
    geom_xmat: NDArray[np.float64]
    xanchor: NDArray[np.float64]
    xaxis: NDArray[np.float64]


def _pose_geometry_error_bounds(
    data: EnvelopeData, radii: Mapping[int, float], joints: Mapping[int, tuple[int, ...]],
    tolerance: float, *, direction: NDArray[np.float64] | None = None,
) -> dict[int, float]:
    """按根到末端的关节顺序，覆盖零件整体位移或指定方向的位移。"""
    delta = _joint_error(tolerance)
    normal = None
    if direction is not None:
        length = float(np.linalg.norm(direction))
        if not math.isfinite(length) or length <= 0.:
            raise ValueError('位移检查方向必须是有限非零向量')
        normal = direction / length
    result = {}
    for geom, radius in radii.items():
        if not np.all(np.isfinite(data.geom_xpos[geom])):
            raise ValueError('当前姿态包含无效碰撞零件位置')
        total = 0.
        for depth, joint in enumerate(joints[geom]):
            axis = data.xaxis[joint]
            offset = data.geom_xpos[geom] - data.xanchor[joint]
            if (not np.all(np.isfinite(offset)) or not np.all(np.isfinite(axis))
                    or abs(float(np.linalg.norm(axis)) - 1.) > 1e-6):
                raise ValueError('当前姿态包含无效关节轴或碰撞零件位置')
            axis = axis / float(np.linalg.norm(axis))
            tangent = np.cross(axis, offset)
            radial = float(np.linalg.norm(tangent)) + radius
            if normal is None:
                total += radial * delta
            else:
                transverse = float(np.linalg.norm(np.cross(normal, axis)))
                first = abs(float(np.dot(normal, tangent))) + radius * transverse
                second = (abs(float(np.dot(normal, np.cross(axis, tangent))))
                          + radius * transverse)
                # Rodrigues公式：sin项不超过delta，1-cos项不超过delta²/2。
                # 前depth个祖先转动会改变世界投影方向，额外变化不超过
                # depth*delta；乘本步整体位移上界，完整计入二阶余量。
                total += first * delta + (.5 * second + depth * radial) * delta ** 2
        # 从根到末端依次扰动关节：此前的上游转动对当前关节轴与下游零件
        # 施加同一个刚体变换，因此本步半径仍等于名义姿态半径。
        # 每步转动位移不超过半径*误差角，三角不等式覆盖所有同时误差。
        result[geom] = total
    return result


def check_task_geometry(
    task: Mapping[str, ContinuousPositionTrajectory], *,
    approach: Mapping[str, ContinuousPositionTrajectory] | None = None,
    return_home: Mapping[str, ContinuousPositionTrajectory] | None = None,
    max_samples: int = 100000,
    cancelled: Callable[[], bool] | None = None,
    joint_error_rad: float = 0.,
    return_targets: Mapping[str, tuple[float, ...]] | None = None,
    nominal_feedback_guarded: bool = False,
    startup_joint_error_rad: float = 0.,
    verified_task_geometry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """用当前工程安装模型检查跨臂和台面；不含工件、自碰撞或线缆."""
    from importlib import import_module
    from archer_control.kinematics import ArmKinematics
    from archer_control.model_checks import CollisionData, validate_tool_chain
    from archer_mujoco.dual_model_loader import load_dual_arm
    from archer_mujoco.model_loader import mujoco

    error = _joint_error(joint_error_rad)
    if type(max_samples) is not int or not 2 <= max_samples <= 250000:
        raise ValueError('几何预检点数上限必须在2到250000之间')
    task_hash = trajectory_fingerprint(task)
    cached_task = None
    if verified_task_geometry is not None:
        proof = verified_task_geometry
        if (proof.get('sampled_cross_arm_table_passed') is not True
                or proof.get('task_trajectory_sha256') != task_hash
                or proof.get('joint_error_rad') != error
                or proof.get('nominal_feedback_required')
                != bool(error and nominal_feedback_guarded)
                or proof.get('table_z_m') != -.01
                or proof.get('cross_arm_clearance_m') != .01
                or proof.get('table_clearance_m') != .001):
            raise ValueError('主体几何证明与当前路径或检查标准不同')
        phase_proof = proof.get('phases')
        cached_task = phase_proof.get('TASK') if isinstance(phase_proof, Mapping) else None
        if (not isinstance(cached_task, dict)
                or type(cached_task.get('samples_checked')) is not int
                or cached_task['samples_checked'] < 2
                or cached_task.get('duration_s') != task['left'].duration
                or cached_task.get('max_time_step_s') != .1
                or cached_task.get('max_joint_travel_rad') != .005):
            raise ValueError('主体几何证明缺少完整采样标准')
    if (isinstance(startup_joint_error_rad, bool)
            or not isinstance(startup_joint_error_rad, (int, float))
            or not math.isfinite(startup_joint_error_rad)
            or not 0. <= startup_joint_error_rad <= MAX_TASK_JOINT_ERROR_RAD
            or startup_joint_error_rad and (approach is None or not error
                                            or startup_joint_error_rad < error
                                            or not nominal_feedback_guarded)):
        raise ValueError('独立起始误差需要接近段和主体误差，且不能超过0.05rad')
    if type(nominal_feedback_guarded) is not bool:
        raise ValueError('名义反馈限位保护标志必须为布尔值')
    if not nominal_feedback_guarded and any(
        error > p.limits.feedback_tolerance for p in task.values()
    ):
        raise ValueError('几何误差上界不能大于已配置的反馈读取容差')
    if return_targets is not None and (
        return_home is None or set(return_targets) != {'left', 'right'}
    ):
        raise ValueError('明确归位目标需要双臂目标映射及完整归位段')
    phases = ([('TASK', task)] if approach is None else
              [('APPROACH_FILE', approach), ('TASK', task)])
    if return_home is not None:
        phases.append(('RETURN_ACTUAL', return_home))
    # 先估算整个会话的预算，再加载模型；不能只检查预算内的前半段。
    total = sum(cached_task['samples_checked'] if phase == 'TASK' and cached_task is not None
                else geometry_schedule(p, max_samples=max_samples, cancelled=cancelled)[0]
                for phase, p in phases)
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
            # 入限后的出发位置可能不同于最初实测值；后者裁到名义边界才是归位目标。
            expected = (return_targets[side] if return_targets is not None else
                        approach[side].initial if approach is not None else None)
            if expected is not None and return_home[side].goal != (
                task[side].limits.check_positions(expected)
            ):
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
    per_geom = geometry_error_displacement_bounds(loaded, error)
    startup_per_geom = (geometry_error_displacement_bounds(loaded, startup_joint_error_rad)
                        if startup_joint_error_rad else per_geom)
    displacement = {side: max(bounds.values()) for side, bounds in per_geom.items()}
    startup_displacement = {side: max(bounds.values())
                            for side, bounds in startup_per_geom.items()}
    guarded_clearance = clearance + sum(displacement.values())
    guarded_table = .001 + max(displacement.values())
    # 宽阶段仍覆盖最大的可能间隙；实际判定使用接触两零件各自的上界。
    geometry.geom_margin[:] = max(guarded_clearance, guarded_table) + .001
    for side, arm in loaded.arms.items():
        if error and nominal_feedback_guarded and (
            np.any(np.asarray(task[side].limits.lower) < arm.joint_limits[:, 0])
            or np.any(np.asarray(task[side].limits.upper) > arm.joint_limits[:, 1])
        ):
            raise ValueError(side + '运行时反馈限位必须位于URDF范围内')
        for _, plans in phases:
            bounds = np.asarray(plans[side].position_bounds)
            # 完整文件执行逐帧强制实际反馈在名义范围内；允许误差集合取交集。
            # 碰撞膨胀仍使用完整误差盒的位移上界，绝不缩小几何保护距离。
            limit_error = 0. if nominal_feedback_guarded else error
            if (np.any(bounds[:, 0] - limit_error < arm.joint_limits[:, 0] - 1e-10)
                    or np.any(bounds[:, 1] + limit_error > arm.joint_limits[:, 1] + 1e-10)):
                raise ValueError(side + '路径超出当前几何模型的URDF限位')
            if error and (
                np.any(bounds[:, 0] - limit_error < np.asarray(plans[side].limits.lower))
                or np.any(bounds[:, 1] + limit_error > np.asarray(plans[side].limits.upper))
            ):
                raise ValueError(side + '误差包围超出名义限位')
        active = [i for i, body in enumerate(geometry.geom_bodyid)
                  if (api.mj_id2name(loaded.model, api.mjtObj.mjOBJ_BODY, int(body))
                      or '').startswith(side + '_')
                  and (geometry.geom_contype[i] or geometry.geom_conaffinity[i])]
        if not active:
            raise ValueError(side + '模型没有启用碰撞几何')

    envelope = cast(EnvelopeModel, loaded.model)
    radii = {geom: float(envelope.geom_rbound[geom])
             for bounds in per_geom.values() for geom in bounds}
    chains: dict[int, tuple[int, ...]] = {}
    for geom in radii:
        body = int(envelope.geom_bodyid[geom])
        ancestors = []
        while body != 0:
            ancestors.extend(int(j) for j in np.flatnonzero(envelope.jnt_bodyid == body))
            body = int(envelope.body_parentid[body])
        chains[geom] = tuple(reversed(ancestors))

    def failure(left: tuple[float, ...], right: tuple[float, ...]) -> str | None:
        for side, q in (('left', left), ('right', right)):
            loaded.data.qpos[loaded.arms[side].qpos_indices] = q
        mujoco.mj_forward(loaded.model, loaded.data)
        posed = _pose_geometry_error_bounds(
            cast(EnvelopeData, loaded.data), radii, chains, active_error)
        # 固定长度上界与姿态上界均成立，取两者较小值仍保守。
        contact_bounds = {geom: min(bound, flat_bounds[geom]) for geom, bound in posed.items()}
        table_bounds = contact_bounds
        if int(envelope.geom_type[floor]) == 0 and int(envelope.geom_bodyid[floor]) == 0:
            # 固定平面距离只受法向位移影响；非平面台面保留整体位移保护。
            normal = cast(EnvelopeData, loaded.data).geom_xmat[floor].reshape(3, 3)[:, 2]
            projected = _pose_geometry_error_bounds(
                cast(EnvelopeData, loaded.data), radii, chains, active_error, direction=normal)
            table_bounds = {geom: min(contact_bounds[geom], bound)
                            for geom, bound in projected.items()}
        data = cast(CollisionData, loaded.data)
        for i in range(data.ncon):
            contact = data.contact[i]
            a, b = int(contact.geom1), int(contact.geom2)
            cross = ((a in active_bounds['left'] and b in active_bounds['right'])
                     or (b in active_bounds['left'] and a in active_bounds['right']))
            if cross:
                required = clearance + contact_bounds[a] + contact_bounds[b]
                if contact.dist < required:
                    names = [api.mj_id2name(
                        loaded.model, api.mjtObj.mjOBJ_BODY, int(geometry.geom_bodyid[g]))
                        for g in (a, b)]
                    return (f'跨臂间隙不足{required * 1000:.3f}mm：'
                            f'{names[0]} / {names[1]}，间距{contact.dist * 1000:.3f}mm')
            if floor in (a, b):
                robot = b if a == floor else a
                if robot not in flat_bounds:
                    continue
                required = .001 + table_bounds[robot]
                if contact.dist < required:
                    name = api.mj_id2name(
                        loaded.model, api.mjtObj.mjOBJ_BODY, int(geometry.geom_bodyid[robot]))
                    return (f'{name}与台面间隙不足{required * 1000:.3f}mm，'
                            f'间距{contact.dist * 1000:.3f}mm')
        return None

    fingerprint = hashlib.sha256()
    for name in ('geom_type', 'geom_size', 'geom_pos', 'geom_quat', 'geom_bodyid',
                 'geom_contype', 'geom_conaffinity', 'body_pos', 'body_quat', 'body_parentid',
                 'jnt_pos', 'jnt_axis', 'mesh_vert', 'mesh_face'):
        value = np.asarray(getattr(loaded.model, name))
        fingerprint.update(name.encode('ascii'))
        fingerprint.update(str((value.shape, value.dtype)).encode('ascii'))
        fingerprint.update(value.tobytes())
    model_hash = fingerprint.hexdigest()
    if verified_task_geometry is not None and (
        verified_task_geometry.get('model_sha256') != model_hash
    ):
        raise ValueError('几何模型已变化，禁止复用主体检查')
    measured = {}
    for phase, plans in phases:
        if phase == 'TASK' and cached_task is not None:
            measured[phase] = dict(cached_task)
            continue
        active_error = (startup_joint_error_rad
                        if phase == 'APPROACH_FILE' and startup_joint_error_rad else error)
        active_bounds = startup_per_geom if phase == 'APPROACH_FILE' else per_geom
        flat_bounds = {geom: bound for bounds in active_bounds.values()
                       for geom, bound in bounds.items()}
        active_displacement = {side: max(bounds.values())
                               for side, bounds in active_bounds.items()}
        guarded_clearance = clearance + sum(active_displacement.values())
        guarded_table = .001 + max(active_displacement.values())
        geometry.geom_margin[:] = max(guarded_clearance, guarded_table) + .001
        try:
            measured[phase] = check_samples(
                plans, failure, max_samples=max_samples, cancelled=cancelled,
            )
        except ValueError as error:
            raise ValueError(phase + '：' + str(error)) from error
    return {'sampled_cross_arm_table_passed': True, 'phases': measured,
            'samples_checked': total, 'model_sha256': model_hash,
            'task_trajectory_sha256': task_hash,
            'task_geometry_reused': cached_task is not None,
            'cross_arm_clearance_m': clearance, 'table_clearance_m': .001,
            'table_z_m': -.01, 'interval_collision_proven': False,
            'self_collision_checked': False, 'object_collision_checked': False,
            'joint_error_rad': error, 'displacement_bounds_m': displacement,
            'startup_joint_error_rad': startup_joint_error_rad,
            'startup_displacement_bounds_m': startup_displacement,
            'displacement_bound_policy': 'per_collision_geometry_pose_radius',
            'clearance_evaluation_policy': 'per_contact_pose_bound',
            'fixed_plane_table_policy': 'normal_projection_with_ancestor_remainder',
            'geometry_displacement_bounds_m': per_geom,
            'startup_geometry_displacement_bounds_m': startup_per_geom,
            'nominal_feedback_required': bool(error and nominal_feedback_guarded),
            'error_limit_policy': ('intersection_with_runtime_nominal_limits'
                                   if error and nominal_feedback_guarded else 'full_error_box'),
            # 保留原字段兼容已有读取；数值是宽阶段检索上限，非所有零件统一门槛。
            'checked_cross_arm_clearance_m': guarded_clearance,
            'checked_table_clearance_m': guarded_table,
            'execution_ready': False,
            'scope': '当前工程刚性GP80与双臂模型的离散跨臂和台面检查，不是连续碰撞证明'}
