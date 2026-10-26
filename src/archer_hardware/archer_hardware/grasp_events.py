"""核对协同任务的夹爪位置与固定等待时序，不调用 ROS 或 SDK、不确认真实夹持。"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from archer_control.grasp_record_format import (
    GRASP_WAIT_S, GRIPPER_CLOSED_COMMAND, GRIPPER_OPEN_COMMAND,
    INSERTION_DISTANCE_M, SIMPLE_GRASP_RULE_VERSION,
)


GRASP_MOTION_STAGES = (
    'APPROACH', 'ALIGN_JOINT6', 'OPEN_FOR_GRASP', 'GRASP_WAIT', 'INSERT',
    'LIFT', 'MOVE', 'RETURN', 'LOWER', 'PLACE_SETTLE', 'RETREAT',
    'CLOSE_BEFORE_HOME', 'RETURN_HOME', 'COMPLETED',
)

SETTLE_PAIRS = {
    'APPROACH_SETTLE': 'APPROACH', 'ALIGN_SETTLE': 'ALIGN_JOINT6',
    'INSERT_SETTLE': 'INSERT', 'RETREAT_SETTLE': 'RETREAT', 'HOME_SETTLE': 'RETURN_HOME',
}


def settle_blend_info(samples: Mapping[str, Any]) -> dict[str, Any]:
    """只合并没有夹爪指令或附着变化的运动稳定尾段，开合与等待不允许穿越。"""
    labels = [p['source_stage'] for p in samples['left']['points']]
    if labels != [p['source_stage'] for p in samples['right']['points']]:
        raise ValueError('协同稳定尾段需要双侧完全相同的阶段')
    boundaries = [i for i in range(1, len(labels))
                  if SETTLE_PAIRS.get(labels[i]) == labels[i - 1]]
    ignored = []
    events = samples['reference_events']
    for i, event in enumerate(events):
        if event['stage'] not in SETTLE_PAIRS:
            continue
        if i == 0 or any(event.get(k) != events[i - 1].get(k)
                         for k in ('gripper_command', 'virtual_attached')):
            raise ValueError('稳定尾段含真实夹爪或附着事件，不允许合并')
        ignored.append(i)
    return {'policy': 'cooperative_motion_settle_v1', 'boundaries': boundaries,
            'continuous_reference_event_indices': ignored}


def target_at(schedule: Mapping[str, Any], elapsed_s: float) -> float:
    """沿共同参考时间平滑开合；搬运保持5.5，不读取虚拟附着标记。"""
    if (not math.isfinite(elapsed_s) or elapsed_s < 0.
            or schedule.get('policy') != 'cooperative_grasp_events_v1'):
        raise ValueError('夹爪计划或参考时间无效')
    phases = schedule['phase_start_ns']
    opening, waiting, closing, home = [phases[k] / 1e9 for k in (
        'OPEN_FOR_GRASP', 'GRASP_WAIT', 'CLOSE_BEFORE_HOME', 'RETURN_HOME')]
    if not 0. <= opening < waiting < closing < home:
        raise ValueError('夹爪开合阶段顺序无效')

    def blend(start: float, end: float) -> float:
        u = min(1., max(0., (elapsed_s - start) / (end - start)))
        return 6.*u**5 - 15.*u**4 + 10.*u**3

    if elapsed_s < opening:
        return GRIPPER_CLOSED_COMMAND
    if elapsed_s < waiting:
        return GRIPPER_CLOSED_COMMAND + (
            GRIPPER_OPEN_COMMAND - GRIPPER_CLOSED_COMMAND) * blend(opening, waiting)
    if elapsed_s < closing:
        return GRIPPER_OPEN_COMMAND
    if elapsed_s < home:
        return GRIPPER_OPEN_COMMAND + (
            GRIPPER_CLOSED_COMMAND - GRIPPER_OPEN_COMMAND) * blend(closing, home)
    return GRIPPER_CLOSED_COMMAND


def event_schedule(
    samples: Mapping[str, Any], prepared: Mapping[str, Any], summary: Mapping[str, Any],
) -> dict[str, Any]:
    """由双侧采样和最终时间轴重建指令计划，虚拟附着标记不会变成设备命令。"""
    rule = summary.get('simple_grasp_rule')
    if (not isinstance(rule, Mapping)
            or type(rule.get('version')) is not int
            or rule.get('version') != SIMPLE_GRASP_RULE_VERSION
            or rule.get('closed_command') != GRIPPER_CLOSED_COMMAND
            or rule.get('open_command') != GRIPPER_OPEN_COMMAND
            or rule.get('wait_s') != GRASP_WAIT_S
            or rule.get('insertion_world_m') != [0., INSERTION_DISTANCE_M, 0.]
            or rule.get('retreat_world_m') != [0., -INSERTION_DISTANCE_M, 0.]
            or rule.get('grasp_confirmation') != 'assumed_for_demo'):
        raise ValueError('需要新版协同规则：5.65闭合、5.5开口、等待5秒、进退各1cm')
    labels: list[str] = []
    stamps: list[int] = []
    for side in ('left', 'right'):
        raw, points = samples[side]['points'], prepared[side]['points']
        if len(raw) != len(points) or len(raw) < 2:
            raise ValueError('协同事件需要完整保留双臂采样')
        names = [p['source_stage'] for p in raw]
        times = []
        for point in points:
            t = point['time_from_start']
            if type(t['sec']) is not int or type(t['nanosec']) is not int:
                raise ValueError('事件时间需要整数秒与纳秒')
            times.append(t['sec'] * 10**9 + t['nanosec'])
        if side == 'left':
            labels, stamps = names, times
        elif labels != names or stamps != times:
            raise ValueError('协同事件要求双臂使用完全相同的阶段和时间轴')
    if stamps[0] != 0 or any(b <= a for a, b in zip(stamps, stamps[1:])):
        raise ValueError('协同事件时间必须从零严格递增')
    phases = [name for i, name in enumerate(labels) if i == 0 or name != labels[i-1]]
    if len(phases) != len(set(phases)):
        raise ValueError('协同阶段不允许倒退或重复进入')
    cursor = 0
    for stage in GRASP_MOTION_STAGES:
        try:
            cursor = phases.index(stage, cursor) + 1
        except ValueError as error:
            raise ValueError('协同主要阶段缺失或顺序错误：' + stage) from error
    starts = {name: stamps[labels.index(name)] for name in phases}
    opened, waiting = starts['OPEN_FOR_GRASP'], starts['GRASP_WAIT']
    inserted, closing = starts['INSERT'], starts['CLOSE_BEFORE_HOME']
    home = starts['RETURN_HOME']
    if waiting <= opened or inserted - waiting < round(GRASP_WAIT_S * 1e9) or home <= closing:
        raise ValueError('开合动作或固定五秒等待在转换中被压缩')
    return {
        'policy': 'cooperative_grasp_events_v1',
        'position_unit': 'sdk_motor_position_not_mm',
        'grasp_confirmation': 'assumed_for_demo',
        'both_grippers_synchronized': True,
        'phase_start_ns': starts,
        'events': [
            {'kind': 'position', 'phase': 'BEFORE_MOTION', 'time_ns': 0,
             'target': GRIPPER_CLOSED_COMMAND, 'requires_stationary_arms': True},
            {'kind': 'position', 'phase': 'OPEN_FOR_GRASP', 'time_ns': opened,
             'end_time_ns': waiting, 'target': GRIPPER_OPEN_COMMAND,
             'requires_stationary_arms': True},
            {'kind': 'dwell', 'phase': 'GRASP_WAIT', 'time_ns': waiting,
             'end_time_ns': inserted, 'minimum_wall_s': GRASP_WAIT_S,
             'target': GRIPPER_OPEN_COMMAND, 'requires_stationary_arms': True},
            {'kind': 'maintain', 'phase': 'INSERT_TO_RETREAT', 'time_ns': inserted,
             'end_time_ns': closing, 'target': GRIPPER_OPEN_COMMAND},
            {'kind': 'position', 'phase': 'CLOSE_BEFORE_HOME', 'time_ns': closing,
             'end_time_ns': home, 'target': GRIPPER_CLOSED_COMMAND,
             'requires_stationary_arms': True},
        ],
        'execution_ready': False,
        'scope': '只核对指令时序；未识别真实夹爪、未连接或发送命令，不确认实际夹持',
    }
