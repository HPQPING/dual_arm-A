"""离线导入仿真CSV，按需保留采样或准备连续限速轨迹，不发送设备指令."""

from __future__ import annotations

import argparse
from bisect import bisect_left
import csv
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
from tempfile import mkdtemp
from typing import Any, cast

import numpy as np
import yaml

from .position_plan import load_limits
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


def replace_conversion_pair(
    output: Path, content: bytes, report_path: Path, report: bytes,
) -> None:
    """先暂存两文件再替换，普通写入异常回滚；回滚异常保留备份位置."""
    staged: list[tuple[Path, Path, Path | None]] = []
    directories: list[Path] = []
    keep_backups = False
    try:
        for target, data in ((output, content), (report_path, report)):
            if target.is_symlink() or (target.exists() and not target.is_file()):
                raise OSError('转换目标必须为普通文件，不能为目录或符号链接')
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(mkdtemp(prefix='.conversion_', dir=target.parent))
            directories.append(temporary)
            candidate = temporary / 'new'
            with candidate.open('xb') as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            backup = temporary / 'previous' if target.exists() else None
            if backup is not None:
                shutil.copyfile(target, backup)
            staged.append((target, candidate, backup))
        committed: list[tuple[Path, Path | None]] = []
        try:
            for target, candidate, backup in staged:
                os.replace(candidate, target)
                committed.append((target, backup))
        except OSError:
            try:
                for target, backup in reversed(committed):
                    if backup is None:
                        target.unlink(missing_ok=True)
                    else:
                        os.replace(backup, target)
            except OSError as error:
                keep_backups = True
                raise OSError(f'回滚失败，保留备份目录：{directories}；{error}') from error
            raise
    finally:
        if not keep_backups:
            for directory in directories:
                shutil.rmtree(directory)


def main(args: list[str] | None = None) -> int:
    """从成功任务CSV生成最新同名转换；成功替换，失败保留旧结果，不连接设备."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--limits-file', type=Path, required=True)
    parser.add_argument('--csv-name', choices=('state.csv', 'tracking.csv'))
    parser.add_argument('--output-dir', type=Path,
                        help='转换文件目录；默认工作空间/archer_results_trans/原任务目录名')
    parser.add_argument('--output', type=Path, help='显式指定轨迹文件，覆盖默认输出位置')
    parser.add_argument('--report', type=Path, help='显式指定转换报告；默认与轨迹文件放在一起')
    parser.add_argument('--max-gap', type=float, default=.2)
    parser.add_argument('--pair-skew-ms', type=float, default=5.)
    parser.add_argument('--prepare', action='store_true', help='生成连续插值及限速后的版本二候选')
    parser.add_argument('--smooth', action='store_true', help='配合--prepare平滑普通节点导数；需兼容的SciPy')
    parser.add_argument('--path-retime', action='store_true',
                        help='有界空间样条与共同弧长重定时；已包含平滑，不与--smooth混用')
    parser.add_argument('--margin', type=float, default=1.05, help='连续轨迹限速时间余量')
    parser.add_argument('--smooth-cruise', action='store_true',
                        help='配合--path-retime使用平滑速度起落及匀速段')
    parser.add_argument('--planning-jerk', type=float,
                        help='配合--path-retime约束规划加加速度rad/s³，不是厂商允许值')
    parser.add_argument('--continuous-jerk', action='store_true',
                        help='空间重定时后统一普通节点导数；需--path-retime和--planning-jerk')
    parser.add_argument('--transit-profile', action='store_true',
                        help='仅优化接近归位，保留圆弧；需平滑匀速、导数平滑及jerk约束')
    parser.add_argument('--boundary-blend', action='store_true',
                        help='仅画圆入口出口局部融合；需--transit-profile，无夹爪事件')
    parser.add_argument('--arc-smooth', action='store_true',
                        help='仅平滑两个融合窗口之间的圆弧内部；需--boundary-blend')
    parser.add_argument('--transit-timing', action='store_true',
                        help='仅压缩已平滑接近归位的内部时轴；需--arc-smooth')
    parser.add_argument('--return-profile', action='store_true',
                        help='只在画圆出口融合之后重新规划同步归位；需--transit-timing')
    options = parser.parse_args(args)
    if options.smooth and not options.prepare:
        parser.error('--smooth需要同时使用--prepare')
    if options.path_retime and (not options.prepare or options.smooth):
        parser.error('--path-retime需要--prepare，且不要同时使用--smooth')
    if (options.smooth_cruise or options.planning_jerk is not None) and not options.path_retime:
        parser.error('--smooth-cruise和--planning-jerk需要--path-retime')
    if options.continuous_jerk and (not options.path_retime or options.planning_jerk is None):
        parser.error('--continuous-jerk需要--path-retime和--planning-jerk')
    if options.transit_profile and not (options.smooth_cruise and options.continuous_jerk):
        parser.error('--transit-profile需要--smooth-cruise和--continuous-jerk')
    if options.boundary_blend and not options.transit_profile:
        parser.error('--boundary-blend需要--transit-profile')
    if options.arc_smooth and not options.boundary_blend:
        parser.error('--arc-smooth需要--boundary-blend')
    if options.transit_timing and not options.arc_smooth:
        parser.error('--transit-timing需要--arc-smooth')
    if options.return_profile and not options.transit_timing:
        parser.error('--return-profile需要--transit-timing')
    root = options.run_dir.expanduser().resolve()
    existing = [name for name in ('state.csv', 'tracking.csv') if (root / name).is_file()]
    selected = options.csv_name or (existing[0] if len(existing) == 1 else None)
    if selected is None:
        parser.error('任务目录必须恰有一种可识别CSV，否则请显式指定--csv-name')
    stem = Path(selected).stem
    workspace = (root.parent.parent if root.parent.name == 'archer_results'
                 else Path.home() / 'archer_ws')
    destination = (options.output_dir.expanduser() if options.output_dir else
                   workspace / 'archer_results_trans' / root.name)
    output = options.output.expanduser() if options.output else destination / (stem + '.json')
    report_path = (options.report.expanduser() if options.report else
                   output.with_name(output.stem + '_report.json'))
    source, summary = root / selected, root / 'summary.json'
    paths = (source, summary, options.limits_file, output, report_path)
    if len({p.resolve() for p in paths}) != len(paths):
        parser.error('输入、配置、候选和报告不能使用相同文件路径')
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
            frequency = 250
            if options.path_retime:
                configuration = cast(dict[str, Any], yaml.safe_load(limits_bytes))
                frequency = configuration['archer_position_driver'][
                    'ros__parameters'].get('control_hz', 250)
            candidate, timing = prepare_samples(candidate, limits, margin=options.margin,
                                                smooth=options.smooth,
                                                path_retime=options.path_retime,
                                                execution_hz=frequency,
                                                smooth_cruise=options.smooth_cruise,
                                                planning_jerk=options.planning_jerk,
                                                continuous_jerk=options.continuous_jerk,
                                                transit_profile=options.transit_profile,
                                                boundary_blend=options.boundary_blend,
                                                arc_smooth=options.arc_smooth,
                                                transit_timing=options.transit_timing,
                                                return_profile=options.return_profile)
            report.update(timing)
    except (OSError, ValueError, TypeError, KeyError, UnicodeError,
            csv.Error, yaml.YAMLError) as error:
        candidate = None
        report['import_succeeded'] = False
        report['error'] = str(error)
    if candidate is None or not report['import_succeeded']:
        print(f"导入或准备失败：{report.get('error', '没有有效轨迹')}；未替换旧结果，未发送动作")
        return 1
    try:
        content = (json.dumps(candidate, ensure_ascii=False, indent=2, allow_nan=False)
                   + '\n').encode('utf-8')
        report['output_sha256'] = hashlib.sha256(content).hexdigest()
        report['converted_at'] = datetime.now().astimezone().isoformat(timespec='microseconds')
        report['conversion_mode'] = ('prepared_continuous' if options.prepare
                                     else 'recorded_samples')
        evidence = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                    + '\n').encode('utf-8')
        replace_conversion_pair(output, content, report_path, evidence)
    except (OSError, ValueError) as error:
        print(f'写入失败：{error}；未发送动作')
        return 1
    action = '已准备连续候选' if options.prepare else '已导入采样候选'
    print(f"{action if report['import_succeeded'] else '导入或准备失败'}：{report_path}；未发送动作")
    return 0 if report['import_succeeded'] else 1


def prepare_samples(
    samples: dict[str, Any], limits: dict[str, PositionLimits], *, margin: float = 1.05,
    smooth: bool = False,
    path_retime: bool = False, execution_hz: int = 250,
    smooth_cruise: bool = False, planning_jerk: float | None = None,
    continuous_jerk: bool = False,
    transit_profile: bool = False,
    boundary_blend: bool = False,
    arc_smooth: bool = False,
    transit_timing: bool = False,
    return_profile: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """准备受限双臂轨迹；空间模式允许有界节点拟合，阶段事件仍仅作为参考."""
    from archer_control.continuous_trajectory import (
        RecordedTimeMap, retime_recorded_pair, smooth_recorded_pair, retime_spatial_pair,
        replace_transit_blocks, trajectory_jerk_metrics, _extrema, blend_circle_boundaries,
        smooth_circle_interior,
    )
    if type(smooth_cruise) is not bool or (
        (smooth_cruise or planning_jerk is not None) and not path_retime
    ):
        raise ValueError('平滑匀速和规划加加速度只用于路径重定时')
    if type(continuous_jerk) is not bool or (
        continuous_jerk and (not path_retime or planning_jerk is None)
    ):
        raise ValueError('普通节点导数平滑需要路径重定时和规划加加速度')
    if type(transit_profile) is not bool or (
        transit_profile and not (path_retime and smooth_cruise and continuous_jerk)
    ):
        raise ValueError('接近归位优化需要平滑匀速路径和普通节点导数平滑')
    if type(boundary_blend) is not bool or (boundary_blend and not transit_profile):
        raise ValueError('画圆边界融合需要接近归位优化')
    if type(arc_smooth) is not bool or (arc_smooth and not boundary_blend):
        raise ValueError('圆弧内部平滑需要已经启用局部边界融合')
    if type(transit_timing) is not bool or (transit_timing and not arc_smooth):
        raise ValueError('接近归位时间优化需要已平滑圆弧和融合窗口')
    if type(return_profile) is not bool or (return_profile and not transit_timing):
        raise ValueError('同步归位优化需要已经完成的接近归位时间处理')

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
    if boundary_blend and events:
        raise ValueError('画圆边界融合不处理夹爪或其他参考事件')
    origin = samples['source_time_origin_ns']
    for event in events:
        relative = event['source_time_ns'] - origin
        if not 0 <= relative <= source_times[-1]:
            raise ValueError('参考事件超出记录时间范围')
        index = min(len(source_times) - 1, bisect_left(source_times, relative))
        stops.update((max(0, index - 1), index))
    if path_retime:
        if smooth or type(execution_hz) is not int or execution_hz not in (250, 500):
            raise ValueError('路径重定时需250/500Hz配置，且不混用旧平滑选项')
        # 与现有SDK单包保护匹配：避免全程因最陡段的统一推进上限被再次放慢。
        fraction = .8 * execution_hz / 500.
        plan, measured = retime_spatial_pair(
            source_times, positions['left'], positions['right'],
            [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')],
            [[v * fraction for v in limits[s].max_velocity] for s in ('left', 'right')],
            [limits[s].max_acceleration for s in ('left', 'right')],
            stops=tuple(sorted(stops)), margin=margin,
            ramp_fraction=.25 if smooth_cruise else None, max_jerk=planning_jerk,
        )
        measured.update(planned_control_hz=execution_hz,
                        planning_velocity_fraction=fraction)
    else:
        plan, measured = retime_recorded_pair(
            source_times, positions['left'], positions['right'],
            [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')],
            [limits[s].max_velocity for s in ('left', 'right')],
            [limits[s].max_acceleration for s in ('left', 'right')],
            stops=tuple(sorted(stops)), margin=margin,
        )
    if smooth or continuous_jerk:
        fraction = .8 * execution_hz / 500. if continuous_jerk else 1.
        plan, smoothed = smooth_recorded_pair(
            plan, [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')],
            [[v * fraction for v in limits[s].max_velocity] for s in ('left', 'right')],
            [limits[s].max_acceleration for s in ('left', 'right')], margin=margin,
            max_jerk=planning_jerk if continuous_jerk else None,
            max_shape_change=.00005 if continuous_jerk else None,
        )
        # 重新登记最终时间轴，阶段和参考事件仍沿用同一映射；保持旧执行格式。
        if smooth:
            measured['max_segment_shape_change_rad'] = {
                s: measured['max_segment_shape_change_rad'][s]
                + smoothed['smoothing_max_segment_change_rad'][s] for s in ('left', 'right')}
        else:
            measured['continuous_jerk_postprocess'] = True
            measured['spatial_blocks'] = [
                {**block, 'duration_s': float(
                    plan.times[block['last']] - plan.times[block['first']])}
                for block in measured['spatial_blocks']]
        measured.update(smoothed)
        measured.update({
            'timing_policy': 'local_segment_c2', 'uniform_time_scale': None,
            'effective_time_scale': float(plan.times[-1] / (source_times[-1] / 1e9)),
            'source_times_ns': list(source_times),
            'prepared_times_ns': [int(round(float(t) * 1e9)) for t in plan.times],
            'stop_indices': list(plan.stops),
        })
    if transit_profile:
        transit_speed = [[min(v, .25) for v in limits[s].max_velocity] for s in ('left', 'right')]
        transit_accel = [[min(a, .20) for a in limits[s].max_acceleration]
                         for s in ('left', 'right')]
        replacement, _ = retime_spatial_pair(
            source_times, positions['left'], positions['right'],
            [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')], transit_speed, transit_accel,
            stops=tuple(sorted(stops)), margin=margin, ramp_fraction=.15, max_jerk=planning_jerk)
        replacement, _ = smooth_recorded_pair(
            replacement, [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')], transit_speed, transit_accel,
            margin=margin, max_jerk=planning_jerk, max_shape_change=.00005)
        plan, selected = replace_transit_blocks(plan, replacement, stages)
        measured.update(transit_profile=selected,
                        transit_velocity_limits_rad_s=transit_speed,
                        transit_acceleration_limits_rad_s2=transit_accel,
                        unchanged_phase_policy='preserve_arc_and_boundary_polynomials')
        measured['planning_velocity_fraction_scope'] = 'baseline_and_arc_only'
        measured['smoothing_max_segment_change_rad'] = {
            s: measured['smoothing_max_segment_change_rad'][s]
            + max(b['max_segment_change_rad'][s] for b in selected) for s in ('left', 'right')}
        ordinary = np.ones(len(plan.times) - 2, dtype=bool)
        ordinary[np.asarray(plan.stops[1:-1], dtype=int) - 1] = False
        gaps = np.diff(plan.times)
        measured['smoothing_jerk_jump_after_rad_s3'] = {}
        for s, polys in plan.polynomials.items():
            end = (6. * polys[:-1, :, 3] + 24. * polys[:-1, :, 4]
                   + 60. * polys[:-1, :, 5]) / gaps[:-1, None]**3
            start = 6. * polys[1:, :, 3] / gaps[1:, None]**3
            measured['smoothing_jerk_jump_after_rad_s3'][s] = float(
                np.max(abs(end[ordinary] - start[ordinary]), initial=0.))
        measured['jerk_metrics'] = trajectory_jerk_metrics(plan)
        measured.update(duration_s=float(plan.times[-1]),
                        effective_time_scale=float(plan.times[-1] / (source_times[-1] / 1e9)),
                        prepared_times_ns=[int(round(float(t) * 1e9)) for t in plan.times])
        measured['spatial_blocks'] = [
            {**block, 'duration_s': float(plan.times[block['last']] - plan.times[block['first']])}
            for block in measured['spatial_blocks']]
    if boundary_blend:
        if planning_jerk is None:
            raise ValueError('局部融合缺少明确规划jerk限值')
        plan, windows = blend_circle_boundaries(
            plan, stages, {s: np.asarray(q) for s, q in positions.items()},
            [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')],
            [[min(v, .25) for v in limits[s].max_velocity] for s in ('left', 'right')],
            [[min(a, .20) for a in limits[s].max_acceleration] for s in ('left', 'right')],
            max_jerk=planning_jerk)
        measured['boundary_blend'] = {'policy': 'circle_local_v1', 'windows': windows}
        measured['smoothing_metadata_scope'] = 'baseline_before_local_boundary_blend'
        measured['spatial_blocks_scope'] = 'retimed_baseline_before_local_boundary_blend'
        measured['stop_indices'] = list(plan.stops)
        measured['unchanged_phase_policy'] = 'preserve_all_segments_outside_blend_windows'
        measured['max_sample_position_change_rad'] = {
            s: float(np.max(abs(plan.positions[s] - positions[s]))) for s in positions}
        measured['jerk_metrics'] = trajectory_jerk_metrics(plan)
        # 先前的普通节点jerk报告不适用于已移除停车点的局部融合，避免沿用旧数值。
        measured.pop('smoothing_jerk_jump_after_rad_s3', None)
    if arc_smooth:
        if planning_jerk is None:
            raise ValueError('圆弧平滑缺少明确规划jerk限值')
        fraction = .8 * execution_hz / 500.
        plan, details = smooth_circle_interior(
            plan, stages, {s: np.asarray(q) for s, q in positions.items()},
            [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')],
            [[v * fraction for v in limits[s].max_velocity] for s in ('left', 'right')],
            [limits[s].max_acceleration for s in ('left', 'right')], max_jerk=planning_jerk)
        measured['arc_smoothing'] = details
        measured['unchanged_phase_policy'] = 'preserve_both_blend_windows_and_all_transit_segments'
        measured['max_sample_position_change_rad'] = {
            s: float(np.max(abs(plan.positions[s] - positions[s]))) for s in positions}
    if transit_timing:
        from archer_control.transit_timing import optimize_transit_timing

        if planning_jerk is None:
            raise ValueError('接近归位时间优化需要明确的规划jerk限制')
        plan, details = optimize_transit_timing(
            plan, stages, [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')],
            [[min(v, .38) for v in limits[s].max_velocity] for s in ('left', 'right')],
            [[min(a, .25) for a in limits[s].max_acceleration] for s in ('left', 'right')],
            max_jerk=planning_jerk, source={s: np.asarray(q) for s, q in positions.items()})
        measured.update(
            transit_timing=details, duration_s=float(plan.times[-1]),
            effective_time_scale=float(plan.times[-1] / (source_times[-1] / 1e9)),
            prepared_times_ns=[int(round(float(t) * 1e9)) for t in plan.times],
            unchanged_phase_policy='preserve_arc_and_both_blend_windows',
            spatial_blocks_scope='baseline_before_transit_time_warp',
            max_sample_position_change_rad={
                s: float(np.max(abs(plan.positions[s] - positions[s]))) for s in positions})
    if return_profile:
        from archer_control.transit_timing import replan_circle_return

        if planning_jerk is None:
            raise ValueError('同步归位优化需要明确的jerk限制')
        plan, details = replan_circle_return(
            plan, stages, [limits[s].lower for s in ('left', 'right')],
            [limits[s].upper for s in ('left', 'right')],
            [[min(v, .32) for v in limits[s].max_velocity] for s in ('left', 'right')],
            [[min(a, .18) for a in limits[s].max_acceleration] for s in ('left', 'right')],
            max_jerk=min(planning_jerk, .8),
            source={s: np.asarray(q) for s, q in positions.items()})
        measured.update(return_profile=details, duration_s=float(plan.times[-1]),
                        prepared_times_ns=[int(round(float(t)*1e9)) for t in plan.times],
                        effective_time_scale=float(plan.times[-1]/(source_times[-1]/1e9)),
                        unchanged_phase_policy='preserve_approach_arc_and_both_blend_windows',
                        transit_timing_scope='baseline_before_return_profile',
                        max_sample_position_change_rad={
                            s: float(np.max(abs(plan.positions[s]-positions[s])))
                            for s in positions})
    if transit_profile:
        # 多个离线处理串联时只汇总最终曲线，避免反复解析已经被后处理替换的中间段。
        measured['jerk_metrics'] = trajectory_jerk_metrics(plan)
        measured['arms'] = {
            s: {field: np.max(np.array([
                np.max(abs(_extrema(c, float(h), order)), axis=1)
                for c, h in zip(plan.polynomials[s], np.diff(plan.times))]), axis=0).tolist()
                for field, order in (('peak_velocity_rad_s', 1), ('peak_acceleration_rad_s2', 2))}
            for s in ('left', 'right')}
    payload = plan.payload()
    prepared_ns = tuple(p['time_from_start']['sec'] * 10 ** 9
                        + p['time_from_start']['nanosec'] for p in payload['left']['points'])
    held = tuple(all(positions[s][i] == positions[s][i + 1] for s in ('left', 'right'))
                 for i in range(len(source_times) - 1))
    mapping = RecordedTimeMap.from_report(measured, source_times, prepared_ns, held_intervals=held)
    # 版本二执行字段保持原格式；来源、阶段和事件放在报告中，避免伪装成已实现的夹爪指令。
    remapped = []
    for event in events:
        elapsed_ns = mapping.map_ns(event['source_time_ns'] - origin)
        remapped.append({**event, 'prepared_time_ns': elapsed_ns})
    return payload, {
        **measured, 'preparation_succeeded': True, 'execution_ready': False,
        'commands_sent': False, 'phase_execution_policy_verified': False,
        'source_stages': stages, 'reference_events': remapped,
        'geometry_preflight_passed': False, 'actual_start_verified': False,
        'scope': ('保留源采样数量和阶段；空间重定时允许有界节点拟合变化，普通模式保留原角度；'
                  '解析验证段内角度、速度及加速度，未验证碰撞、实际起点或夹爪执行'),
    }
