"""独立验收完整抓取记录；只读取文件，不启动 ROS 或发送动作。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..grasp_record_format import EVENT_FIELDS, SIDES, STATE_FIELDS


FloatArray = NDArray[np.float64]
ZERO_STAGES = frozenset(('OPEN_BEFORE_INSERT', 'INSERT', 'INSERT_SETTLE', 'CLOSE_DEMO',
                         'LIFT', 'MOVE', 'RETURN', 'LOWER', 'PLACE_SETTLE', 'RELEASE',
                         'OPEN_WAIT', 'RETREAT', 'RETREAT_SETTLE'))
REQUIRED_STAGES = ('READY', 'PLANNING', 'SETTLE', 'APPROACH', 'APPROACH_SETTLE',
                   'ALIGN_JOINT6', 'ALIGN_SETTLE', 'OPEN_BEFORE_INSERT', 'INSERT',
                   'INSERT_SETTLE', 'CLOSE_DEMO', 'LIFT', 'MOVE', 'RETURN', 'LOWER',
                   'PLACE_SETTLE', 'RELEASE', 'OPEN_WAIT', 'RETREAT', 'RETREAT_SETTLE',
                   'RETURN_HOME', 'HOME_SETTLE', 'COMPLETED')


@dataclass(frozen=True)
class RecordedFrame:
    time: float
    wall_time: float
    stage: str
    command: float
    attached: bool
    positions: FloatArray
    velocities: FloatArray
    inputs: FloatArray
    object_pose: FloatArray
    tcp: FloatArray


@dataclass(frozen=True)
class GraspRun:
    directory: Path
    summary: dict[str, object]
    config: dict[str, float]
    frames: tuple[RecordedFrame, ...]
    report: dict[str, object]


def number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(label + ' 必须为有限数值')
    return float(value)


def mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(label + ' 必须为对象')
    return value


def joint_snapshot(value: object, label: str) -> FloatArray:
    source = mapping(value, label)
    result = []
    for side in SIDES:
        row = source.get(side)
        if not isinstance(row, list) or len(row) != 6:
            raise ValueError(label + '/' + side + ' 需要六个关节角')
        result.append([number(v, label) for v in row])
    return np.asarray(result, dtype=np.float64)


def load_run(directory: str | Path) -> GraspRun:
    """只接受正常收尾且成功归位的一轮；停止或失败记录不能当作成功轨迹。"""
    folder = Path(directory).expanduser().resolve()
    summary = mapping(json.loads((folder / 'summary.json').read_text(encoding='utf-8')), 'summary')
    meta = mapping(summary.get('recording'), 'recording')
    if (summary.get('state') != 'COMPLETED' or summary.get('ros_stage') != 'COMPLETED'
            or meta.get('complete') is not True or meta.get('task_completed') is not True
            or summary.get('failure') != '' or summary.get('mode') != 'simple_demo'):
        raise ValueError('该记录不是完整成功的简化抓取任务；请查看 summary.json 中的失败或中断原因')
    config = {key: number(value, 'config/' + key)
              for key, value in mapping(summary.get('config'), 'config').items()}
    for key in ('object_height', 'object_diameter', 'pedestal_height'):
        if key not in config:
            raise ValueError('config 缺少 ' + key)
    if config['object_height'] <= 0. or config['object_diameter'] <= 0. or config['pedestal_height'] < 0.:
        raise ValueError('圆柱尺寸或台面高度配置无效')
    with (folder / 'state.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        headers = reader.fieldnames
        expected = list(STATE_FIELDS)
        if headers != expected:
            raise ValueError('state.csv 列名与当前记录格式不一致，或存在重复、缺失列')
        frames: list[RecordedFrame] = []
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f'state.csv 第 {line} 行列数不正确')
            values: dict[str, float] = {}
            for key in expected:
                if key != 'stage':
                    try:
                        values[key] = float(row[key])
                    except (ValueError, TypeError) as exc:
                        raise ValueError(f'第 {line} 行 {key} 不是数值') from exc
                    number(values[key], f'第 {line} 行 {key}')
            arrays = [np.asarray([[values[f'{side}_joint_{i}_{quantity}'] for i in range(1, 7)]
                                  for side in SIDES], dtype=np.float64)
                      for quantity in ('position_rad', 'velocity_rad_s', 'drive_input_rad')]
            if values['virtual_attached'] not in (0., 1.):
                raise ValueError(f'第 {line} 行附着标志不是 0 或 1')
            frames.append(RecordedFrame(values['simulation_time_s'], values['wall_elapsed_s'],
                row['stage'], values['gripper_command'], bool(values['virtual_attached']), arrays[0], arrays[1], arrays[2],
                np.asarray([values['object_' + k] for k in ('x_m', 'y_m', 'z_m', 'qw', 'qx', 'qy', 'qz')], dtype=np.float64),
                np.asarray([[values[f'{side}_tcp_{a}_m'] for a in ('x', 'y', 'z')] for side in SIDES], dtype=np.float64)))
    if len(frames) < 2 or len(frames) != meta.get('sample_count'):
        raise ValueError('采样数量与 summary.json 不一致，或记录为空')
    with (folder / 'events.csv').open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(EVENT_FIELDS):
            raise ValueError('events.csv 列名不正确')
        events = list(reader)
    history = [event['stage'] for event in events]
    if history != meta.get('observed_stages') or not history or history[-1] != 'COMPLETED':
        raise ValueError('事件日志与摘要不一致')
    # 允许暂停与恢复造成阶段重复，但主流程不得漏项或倒退。
    sequence: list[str] = []
    previous_wall = -1.
    for event in events:
        wall = float(event['wall_elapsed_s'])
        if not math.isfinite(wall) or wall < 0. or wall < previous_wall:
            raise ValueError('事件日志时间无效或倒退')
        previous_wall = wall
        stage = event['stage']
        if stage == 'PAUSED':
            continue
        if stage not in REQUIRED_STAGES:
            raise ValueError('事件日志包含未知或异常阶段：' + stage)
        if not sequence or sequence[-1] != stage:
            sequence.append(stage)
    if tuple(sequence) != REQUIRED_STAGES:
        raise ValueError('阶段日志未包含完整且有序的抓取与归位流程')
    if frames[0].stage != 'READY' or frames[-1].stage != 'COMPLETED':
        raise ValueError('状态记录起止阶段不正确')
    peak_speed, max_gap, penetration = 0., 0., 0.
    previous_time, previous_wall = -1., -1.
    for frame in frames:
        if frame.stage not in history or frame.time < previous_time or frame.wall_time < previous_wall:
            raise ValueError('采样阶段无效或时间倒退')
        if frame.time < 0. or frame.wall_time < 0.:
            raise ValueError('采样时间不能为负数')
        if previous_time >= 0.:
            max_gap = max(max_gap, frame.time - previous_time)
        previous_time, previous_wall = frame.time, frame.wall_time
        if not 0. <= frame.command <= 5.65:
            raise ValueError('夹爪命令超出 0～5.65')
        if frame.attached and not math.isclose(frame.command, 2., abs_tol=1e-12):
            raise ValueError('虚拟附着期间夹爪命令不是 2.0')
        if not math.isclose(float(np.linalg.norm(frame.object_pose[3:])), 1., abs_tol=1e-6):
            raise ValueError('圆柱姿态四元数未归一化')
        if frame.stage in ZERO_STAGES and np.max(np.abs(frame.inputs[:, 5])) > 1e-12:
            raise ValueError('进给、搬运或退开期间 J6 输入没有保持零')
        if frame.stage in ZERO_STAGES and np.max(np.abs(frame.positions[:, 5])) > .003:
            raise ValueError('进给、搬运或退开期间 J6 实际反馈未在零位容差内')
        if frame.stage in ('LIFT', 'MOVE', 'RETURN', 'LOWER') and not frame.attached:
            raise ValueError('搬运期间没有保持虚拟附着')
        peak_speed = max(peak_speed, float(np.max(np.abs(frame.velocities))))
        qw, qx, qy, qz = frame.object_pose[3:]
        cosine = min(1., abs(float(1. - 2. * (qx*qx + qy*qy))))
        extent = config['object_height'] / 2. * cosine + config['object_diameter'] / 2. * math.sqrt(max(0., 1.-cosine*cosine))
        penetration = max(penetration, -config['pedestal_height'] - float(frame.object_pose[2]) + extent)
    if max_gap > .061:
        raise ValueError(f'记录存在超过 61 ms 的仿真采样空缺：{max_gap:.6f} s')
    if penetration > .001:
        raise ValueError('圆柱底部穿入台面超过 1 mm')
    start = joint_snapshot(summary.get('start_joints_from_feedback_rad'), 'start')
    final = joint_snapshot(summary.get('final_joints_rad'), 'final')
    if not np.allclose(frames[0].positions, start, atol=1e-10, rtol=0.) or not np.allclose(frames[-1].positions, final, atol=1e-10, rtol=0.):
        raise ValueError('CSV 起终点与摘要中的反馈不一致')
    home_error = float(np.max(np.abs(final-start)))
    if home_error > .003 or frames[-1].attached or frames[-1].command != 0.:
        raise ValueError('任务未完成归位、解除附着和松开')
    if not any(frame.attached for frame in frames):
        raise ValueError('记录没有包含虚拟夹取阶段')
    final_object = summary.get('final_object_center_world_m')
    if not isinstance(final_object, list) or len(final_object) != 3:
        raise ValueError('摘要缺少最终工件中心')
    final_center = np.asarray([number(value, 'final_object_center_world_m') for value in final_object], dtype=np.float64)
    if not np.allclose(frames[-1].object_pose[:3], final_center, atol=1e-9, rtol=0.):
        raise ValueError('CSV 最终工件中心与摘要不一致')
    expected_center = np.asarray([0., .4, -config['pedestal_height']+config['object_height']/2.], dtype=np.float64)
    if np.linalg.norm(final_center-expected_center) > .001:
        raise ValueError('工件没有放回台面上的名义初始位置，误差超过 1 mm')
    if not math.isclose(frames[-1].time, number(summary.get('simulation_time_s'), 'simulation_time_s'), abs_tol=1e-9):
        raise ValueError('CSV 结束时间与摘要不一致')
    report: dict[str, object] = {'samples': len(frames), 'simulation_duration_s': frames[-1].time-frames[0].time,
        'maximum_sample_gap_s': max_gap, 'maximum_home_joint_error_rad': home_error,
        'observed_peak_joint_speed_rad_s': peak_speed, 'maximum_table_penetration_mm': penetration*1000.,
        'scope': '记录一致性验收；不代表实机限速、自碰撞或真实夹持力验证'}
    return GraspRun(folder, summary, config, tuple(frames), report)


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path, help='包含 state.csv、events.csv、summary.json 的目录')
    options = parser.parse_args(args)
    try:
        run = load_run(options.directory)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print('验收失败：' + str(exc))
        return 1
    print('通过：完整抓取记录、阶段、J6 零输入、工件台面与归位一致性')
    print(json.dumps(run.report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
