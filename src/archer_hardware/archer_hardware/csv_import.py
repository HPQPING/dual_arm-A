"""离线导入仿真CSV，按需保留采样或准备连续限速轨迹，不发送设备指令."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import yaml

from .commissioning import load_limits
from .position_plan import PositionLimits


@dataclass(frozen=True)
class Sample:
    """一臂采样包含原始纳秒时间、实际关节角及参考阶段信息."""

    stamp: int
    joints: tuple[float, ...]
    phase: str


def seconds_ns(value: str) -> int:
    """精确转换CSV秒数，避免大整数时间经浮点转换丢失纳秒精度."""
    try:
        seconds = Decimal(value)
        if not seconds.is_finite() or not 0 <= seconds <= Decimal('100000000000'):
            raise ValueError('时间必须是有限非负秒数')
        return int((seconds * 10 ** 9).to_integral_value(rounding=ROUND_HALF_EVEN))
    except InvalidOperation as error:
        raise ValueError('时间字段不是有效数值') from error


def number(value: str) -> float:
    """仅接受有限数值，不将空白或非法内容替换成默认值."""
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError('CSV字段不是有效数值') from error
    if not math.isfinite(result):
        raise ValueError('CSV字段必须为有限数值')
    return result


def import_samples(
    content: bytes, limits: dict[str, PositionLimits], *,
    max_gap_s: float = .2, pair_skew_ms: float = 5.,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """保留实际角度及采样间隔，检查名义限位；不选择或验证连续插值."""
    if (not math.isfinite(max_gap_s) or not .001 <= max_gap_s <= 1.
            or not math.isfinite(pair_skew_ms) or not 0 <= pair_skew_ms <= 5.):
        raise ValueError('最大采样间隔应为0.001到1秒，配对时差应为0到5毫秒')
    if len(content) > 50 * 1024 * 1024:
        raise ValueError('CSV超过50MiB导入上限')
    reader = csv.DictReader(io.StringIO(content.decode('utf-8-sig'), newline=''))
    header = reader.fieldnames
    if not header or len(header) != len(set(header)):
        raise ValueError('CSV表头为空或列名重复')
    grasp_names = {side: [f'{side}_joint_{i}_position_rad' for i in range(1, 7)]
                   for side in ('left', 'right')}
    tracking_names = [f'actual_J{i}_rad' for i in range(1, 7)]
    is_grasp = all(name in header for names in grasp_names.values() for name in names)
    if is_grasp:
        if not {'simulation_time_s', 'stage'} <= set(header):
            raise ValueError('state.csv缺少仿真时间或阶段')
        time_basis = 'simulation_time_s'
    elif {'side', *tracking_names} <= set(header):
        if 'simulation_time_s' in header:
            time_basis = 'simulation_time_s'
        elif 'feedback_stamp_ns' in header:
            time_basis = 'feedback_stamp_ns'
        else:
            raise ValueError('tracking.csv缺少可用的原始时间列')
    else:
        raise ValueError('缺少实际关节角列；不接受仅有目标角或drive_input_rad的文件')
    arms: dict[str, list[Sample]] = {'left': [], 'right': []}
    auxiliary: list[dict[str, Any]] = []
    duplicate_count = rows = 0
    for line, row in enumerate(reader, 2):
        rows += 1
        if rows > 200000:
            raise ValueError('CSV超过200000行导入上限')
        try:
            if None in row or any(value is None for value in row.values()):
                raise ValueError('列数与表头不一致')
            raw_time = row[time_basis]
            if time_basis == 'feedback_stamp_ns':
                if not raw_time.isascii() or not raw_time.isdigit():
                    raise ValueError('反馈时间戳必须是非负整数纳秒')
                stamp = int(raw_time)
            else:
                stamp = seconds_ns(raw_time)
            sides = ('left', 'right') if is_grasp else (row['side'],)
            if any(side not in arms for side in sides):
                raise ValueError('side必须是left或right')
            phase = row.get('stage', row.get('phase', ''))
            if is_grasp:
                extra: dict[str, Any] = {'source_time_ns': stamp, 'stage': phase}
                if 'gripper_command' in row:
                    extra['gripper_command'] = number(row['gripper_command'])
                if 'virtual_attached' in row:
                    if row['virtual_attached'] not in ('0', '1'):
                        raise ValueError('虚拟附着标记必须为0或1')
                    extra['virtual_attached'] = int(row['virtual_attached'])
                if not auxiliary or any(extra.get(key) != auxiliary[-1].get(key)
                                        for key in extra if key != 'source_time_ns'):
                    auxiliary.append(extra)
            for side in sides:
                names = grasp_names[side] if is_grasp else tracking_names
                joints = limits[side].check_positions([number(row[name]) for name in names])
                previous = arms[side][-1] if arms[side] else None
                if previous is not None and stamp <= previous.stamp:
                    if stamp == previous.stamp and joints == previous.joints:
                        duplicate_count += 1
                        continue
                    raise ValueError('时间倒退，或同一时间出现不同关节角')
                arms[side].append(Sample(stamp, joints, phase))
        except (ValueError, KeyError, TypeError) as error:
            raise ValueError(f'CSV第{line}行：{error}') from error
    if len(arms['left']) != len(arms['right']) or len(arms['left']) < 2:
        raise ValueError('左右实际采样数量不同或不足两个时间点，禁止静默丢弃未配对数据')
    pairs = list(zip(arms['left'], arms['right']))
    skew_ns = int(round(pair_skew_ms * 10 ** 6))
    maximum_skew = max(abs(left.stamp - right.stamp) for left, right in pairs)
    if maximum_skew > skew_ns:
        raise ValueError('左右对应采样时差超过配置，不能视为同一双臂时间点')
    stamps = [(left.stamp + right.stamp) // 2 for left, right in pairs]
    origin = stamps[0]
    times = [stamp - origin for stamp in stamps]
    gaps = [b - a for a, b in zip(times, times[1:])]
    if any(gap <= 0 for gap in gaps) or max(gaps) > round(max_gap_s * 1e9):
        raise ValueError('配对时间不递增或采样缺口过大，不补造缺失数据')
    if times[-1] > 3600 * 10 ** 9:
        raise ValueError('采样路径超过一小时')
    candidate: dict[str, Any] = {
        'version': 3, 'kind': 'recorded_joint_samples', 'interpolation': 'unassigned',
        'source_time_basis': time_basis, 'source_time_origin_ns': origin,
        'reference_events': auxiliary,
    }
    metrics = {}
    needs_timing = False
    for side in ('left', 'right'):
        samples = arms[side]
        candidate[side] = {
            'joint_names': [f'{side}_joint_{i}' for i in range(1, 7)],
            'points': [
                {'positions': list(sample.joints), 'source_stage': sample.phase,
                 'time_from_start': {'sec': elapsed // 10 ** 9, 'nanosec': elapsed % 10 ** 9}}
                for elapsed, sample in zip(times, samples)
            ],
        }
        velocities = [tuple((b - a) / (gap / 1e9) for a, b in zip(first.joints, last.joints))
                      for first, last, gap in zip(samples, samples[1:], gaps)]
        peak_v = [max(abs(row[i]) for row in velocities) for i in range(6)]
        accelerations = [tuple(2 * (b - a) / ((first_gap + last_gap) / 1e9)
                               for a, b in zip(first, last))
                         for first, last, first_gap, last_gap
                         in zip(velocities, velocities[1:], gaps, gaps[1:])]
        peak_a = [max((abs(row[i]) for row in accelerations), default=0.) for i in range(6)]
        bound = limits[side]
        too_fast = (any(v > cap for v, cap in zip(peak_v, bound.max_velocity))
                    or any(a > cap for a, cap in zip(peak_a, bound.max_acceleration)))
        needs_timing = needs_timing or too_fast
        metrics[side] = {'finite_difference_peak_velocity_rad_s': peak_v,
                         'finite_difference_peak_acceleration_rad_s2': peak_a,
                         'sample_difference_within_limits': not too_fast}
    return candidate, {
        'import_succeeded': True, 'execution_ready': False, 'commands_sent': False,
        'source_rows': rows, 'paired_samples': len(pairs), 'duration_s': times[-1] / 1e9,
        'contains_home_phase': any('HOME' in sample.phase for arm in arms.values()
                                   for sample in arm),
        'phase_execution_policy_verified': False,
        'duplicate_identical_arm_samples_removed': duplicate_count,
        'source_time_origin_ns': origin, 'source_time_basis': time_basis,
        'max_pair_skew_ms': maximum_skew / 1e6, 'max_gap_s': max(gaps) / 1e9,
        'requires_time_adjustment_by_sample_differences': needs_timing, 'arms': metrics,
        'selected_data': '实际position_rad或actual_J*_rad；从未读取drive_input_rad',
        'scope': '名义角度与采样差分诊断；未选择连续插值，未验证段内峰值、碰撞、实际起点或夹持',
    }


def main(args: list[str] | None = None) -> int:
    """从成功任务目录读取CSV并排他写入候选与报告，不初始化ROS或连接SDK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--limits-file', type=Path, required=True)
    parser.add_argument('--csv-name', choices=('state.csv', 'tracking.csv'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--max-gap', type=float, default=.2)
    parser.add_argument('--pair-skew-ms', type=float, default=5.)
    parser.add_argument('--prepare', action='store_true', help='生成连续插值及限速后的版本二候选')
    parser.add_argument('--margin', type=float, default=1.05, help='连续轨迹限速时间余量')
    options = parser.parse_args(args)
    root = options.run_dir.expanduser()
    stem = 'prepared_path' if options.prepare else 'recorded_path'
    output = options.output or root / (stem + '.json')
    report_path = options.report or root / (stem + '_report.json')
    existing = [name for name in ('state.csv', 'tracking.csv') if (root / name).is_file()]
    selected = options.csv_name or (existing[0] if len(existing) == 1 else None)
    if selected is None:
        parser.error('任务目录必须恰有一种可识别CSV，否则请显式指定--csv-name')
    source, summary = root / selected, root / 'summary.json'
    paths = (source, summary, options.limits_file, output, report_path)
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error('输入、配置、候选和报告不能使用相同文件路径')
    if output.exists() or report_path.exists():
        parser.error('候选或报告已存在，请指定新名称；不覆盖旧数据')
    report: dict[str, Any] = {'import_succeeded': False, 'execution_ready': False,
                              'commands_sent': False}
    candidate = None
    try:
        summary_bytes = summary.read_bytes()
        result = json.loads(summary_bytes.decode('utf-8-sig'))
        if not isinstance(result, dict):
            raise ValueError('任务摘要格式错误')
        states = [result[key] for key in ('state', 'task_state', 'ros_stage') if key in result]
        if not states or any(state != 'COMPLETED' for state in states):
            raise ValueError('仅接受摘要明确标记COMPLETED的仿真任务')
        source_bytes, limits_bytes = source.read_bytes(), options.limits_file.read_bytes()
        report.update({
            'source_csv': str(source.resolve()),
            'csv_sha256': hashlib.sha256(source_bytes).hexdigest(),
            'summary_sha256': hashlib.sha256(summary_bytes).hexdigest(),
            'limits_sha256': hashlib.sha256(limits_bytes).hexdigest(),
        })
        limits = load_limits(options.limits_file, content=limits_bytes)
        candidate, measured = import_samples(
            source_bytes, limits,
            max_gap_s=options.max_gap, pair_skew_ms=options.pair_skew_ms,
        )
        report.update(measured)
        if options.prepare:
            candidate, timing = prepare_samples(candidate, limits, margin=options.margin)
            report.update(timing)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            csv.Error, yaml.YAMLError) as error:
        candidate = None
        report['import_succeeded'] = False
        report['error'] = str(error)
    created_output = created_report = False
    try:
        with report_path.open('x', encoding='utf-8') as evidence:
            created_report = True
            if candidate is not None and report['import_succeeded']:
                content = (json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False)
                           + '\n').encode('utf-8')
                with output.open('xb') as stream:
                    created_output = True
                    stream.write(content)
                report['output_sha256'] = hashlib.sha256(content).hexdigest()
            json.dump(report, evidence, ensure_ascii=False, indent=2, allow_nan=False)
            evidence.write('\n')
    except (OSError, ValueError) as error:
        if created_output:
            output.unlink(missing_ok=True)
        if created_report:
            report_path.unlink(missing_ok=True)
        print(f'写入失败：{error}；未发送动作')
        return 1
    action = '已准备连续候选' if options.prepare else '已导入采样候选'
    print(f"{action if report['import_succeeded'] else '导入或准备失败'}：{report_path}；未发送动作")
    return 0 if report['import_succeeded'] else 1


def prepare_samples(
    samples: dict[str, Any], limits: dict[str, PositionLimits], *, margin: float = 1.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """接续本模块的采样结果，保留位置并调整双臂时间；事件仍只作为参考."""
    from archer_control.continuous_trajectory import retime_recorded_pair

    if (samples.get('version') != 3 or samples.get('kind') != 'recorded_joint_samples'
            or samples.get('interpolation') != 'unassigned'):
        raise ValueError('只能准备本模块导出的版本三原始采样')
    grids: list[tuple[int, ...]] = []
    positions = {}
    stages = {}
    for side in ('left', 'right'):
        arm = samples[side]
        if arm['joint_names'] != [f'{side}_joint_{i}' for i in range(1, 7)]:
            raise ValueError('采样关节顺序无效')
        points = arm['points']
        if not isinstance(points, list) or not 2 <= len(points) <= 20000:
            raise ValueError('准备轨迹需要2到20000个双臂采样点，不会自动删点')
        stamps = []
        for point in points:
            stamp = point['time_from_start']
            sec, ns = stamp['sec'], stamp['nanosec']
            if type(sec) is not int or type(ns) is not int or sec < 0 or not 0 <= ns < 10 ** 9:
                raise ValueError('时间字段必须是非负整数秒和合法纳秒')
            stamps.append(sec * 10 ** 9 + ns)
        grids.append(tuple(stamps))
        positions[side] = [limits[side].check_positions(p['positions']) for p in points]
        stages[side] = [p['source_stage'] for p in points]
    if grids[0] != grids[1]:
        raise ValueError('双臂采样必须共用时间轴')
    source_times = grids[0]
    stops = {0, len(source_times) - 1}
    for labels in stages.values():
        for i in range(1, len(labels)):
            if labels[i] != labels[i - 1]:
                stops.update((i - 1, i))
    events = samples['reference_events']
    origin = samples['source_time_origin_ns']
    for event in events:
        relative = event['source_time_ns'] - origin
        if not 0 <= relative <= source_times[-1]:
            raise ValueError('参考事件超出记录时间范围')
        index = min(len(source_times) - 1, bisect_left(source_times, relative))
        stops.update((max(0, index - 1), index))
    plan, measured = retime_recorded_pair(
        source_times, positions['left'], positions['right'],
        [limits[s].lower for s in ('left', 'right')],
        [limits[s].upper for s in ('left', 'right')],
        [limits[s].max_velocity for s in ('left', 'right')],
        [limits[s].max_acceleration for s in ('left', 'right')],
        stops=tuple(sorted(stops)), margin=margin,
    )
    payload = plan.payload()
    # 版本二执行字段保持原格式；来源、阶段和事件放在报告中，避免伪装成已实现的夹爪指令。
    remapped = []
    for event in events:
        elapsed_ns = math.ceil((event['source_time_ns'] - origin) * measured['uniform_time_scale'])
        remapped.append({**event, 'prepared_time_ns': elapsed_ns})
    return payload, {
        **measured, 'preparation_succeeded': True, 'execution_ready': False,
        'commands_sent': False, 'phase_execution_policy_verified': False,
        'source_stages': stages, 'reference_events': remapped,
        'geometry_preflight_passed': False, 'actual_start_verified': False,
        'scope': '保留实际关节路径并解析验证段内角度、速度及加速度；未验证碰撞、实际起点或夹爪事件执行',
    }
