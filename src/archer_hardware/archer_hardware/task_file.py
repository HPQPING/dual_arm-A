"""离线核对成功任务的原始记录与连续轨迹，输出证据但不连接ROS或SDK."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable

import yaml

from archer_control.continuous_trajectory import RecordedTimeMap, circle_blend_windows
from .position_plan import load_limits
from .csv_import import import_samples
from .position_plan import ContinuousPositionTrajectory, positive, prepare_continuous_pair


@dataclass(frozen=True)
class ReviewedTask:
    """保存同一次文件核对生成的只读轨迹映射，后续预检不重新读取路径文件."""

    plans: Mapping[str, ContinuousPositionTrajectory]
    report: Mapping[str, Any]


def read_snapshot(path: Path, maximum: int) -> bytes:
    """有界读取单个文件快照，之后仅使用这份内容，防止分次读取混用."""
    with path.open('rb') as stream:
        content = stream.read(maximum + 1)
    if len(content) > maximum:
        raise ValueError(f'文件超过读取上限：{path.name}')
    return content


def json_mapping(content: bytes) -> dict[str, Any]:
    """拒绝重复键与非有限常量，避免不同JSON读取器产生歧义."""
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError('JSON存在重复字段：' + key)
            result[key] = value
        return result

    def invalid(value: str) -> object:
        raise ValueError('JSON不允许非有限常量：' + value)

    result = json.loads(content.decode('utf-8-sig'), object_pairs_hook=pairs,
                        parse_constant=invalid)
    if not isinstance(result, dict):
        raise ValueError('JSON根节点必须是映射')
    return result


def completed_task(summary: Mapping[str, Any], csv_name: str) -> str:
    """仅识别画圆和完整圆柱抓取，拒绝旧协同参考物示意及未完成结果."""
    states = [summary[k] for k in ('state', 'task_state', 'ros_stage') if k in summary]
    if not states or any(s != 'COMPLETED' for s in states):
        raise ValueError('任务摘要必须明确为COMPLETED')
    if csv_name == 'state.csv':
        recording = summary.get('recording')
        if (not isinstance(recording, dict) or recording.get('task_name') != '协同抓取'
                or any(recording.get(k) is not True
                       for k in ('complete', 'saved', 'task_completed'))):
            raise ValueError('缺少完整抓取记录元数据')
        return 'cooperative_grasp'
    if (csv_name == 'tracking.csv' and summary.get('task_name') == '画圆'
            and summary.get('task_type') == 'split_circle'
            and summary.get('reference_object_only') is False):
        return 'split_circle'
    raise ValueError('只接受画圆或完整圆柱抓取任务，不接受旧协同参考物示意')


def check_task_stages(stages: object, task: str) -> None:
    """每侧必须依次包含任务主要阶段和最终归位，禁止截取部分文件冒充整轮."""
    if not isinstance(stages, list) or any(not isinstance(s, str) for s in stages):
        raise ValueError('任务阶段必须是字符串序列')
    required = (('APPROACH', 'ARC', 'RETURN_HOME') if task == 'split_circle' else
                ('APPROACH', 'ALIGN_JOINT6', 'INSERT', 'CLOSE_DEMO', 'LIFT',
                 'RELEASE', 'RETREAT', 'RETURN_HOME', 'COMPLETED'))
    cursor = 0
    for name in required:
        try:
            cursor = stages.index(name, cursor) + 1
        except ValueError as error:
            raise ValueError('任务主要阶段缺失或顺序错误：' + name) from error


def _return_profile_range(
    report: Mapping[str, Any], task: str, plans: Mapping[str, ContinuousPositionTrajectory],
    stages: dict[str, list[str]], tolerance: float,
) -> tuple[int, int] | None:
    """仅给画圆出口融合之后的显式同步归位开放固定节点偏差，独立检查导数."""
    profile = report.get('return_profile')
    if profile is None:
        return None
    if (task != 'split_circle' or tolerance != .0005 or not isinstance(profile, dict)
            or profile.get('policy') != 'circle_return_joint_profile_v1'
            or profile.get('source_node_tolerance_rad') != .004
            or not isinstance(report.get('boundary_blend'), dict)
            or report.get('reference_events') != []):
        raise ValueError('同步归位声明只支持已融合的无事件画圆，不扩大其他阶段误差')
    required = {0, len(plans['left'].times_ns)-1}
    for labels in stages.values():
        required.update(k for i in range(1, len(labels)) if labels[i] != labels[i-1]
                        for k in (i-1, i))
    seconds = [t/1e9 for t in plans['left'].times_ns]
    windows = circle_blend_windows(seconds, stages, tuple(sorted(required)))
    first, last = windows[1]['last'], len(plans['left'].times_ns)-1
    actual_duration = (plans['left'].times_ns[last]-plans['left'].times_ns[first])/1e9
    if (type(profile.get('first')) is not int or profile['first'] != first
            or type(profile.get('last')) is not int or profile['last'] != last
            or last-first < 32 or type(profile.get('correction_s')) not in (int, float)
            or profile.get('correction_s') not in (1., 1.5, 2.)
            or type(profile.get('after_s')) not in (int, float)
            or not math.isclose(profile['after_s'], actual_duration, rel_tol=0., abs_tol=1e-9)):
        raise ValueError('同步归位范围或时间与实际画圆出口不一致')
    import numpy as np
    from archer_control.transit_timing import _jerk_bounds
    from archer_mujoco.joint_interpolation import _extrema

    for side, plan in plans.items():
        gaps = np.diff(np.asarray(plan.times_ns[first:], dtype=np.int64))/1e9
        polys = np.stack(plan.polynomials[first:])
        if np.max(abs(_jerk_bounds(polys, gaps))) > .8 + 1e-10:
            raise ValueError(side + '同步归位jerk超限')
        for order, cap, configured in ((1, .32, plan.limits.max_velocity),
                                       (2, .18, plan.limits.max_acceleration)):
            peaks = np.array([np.max(abs(_extrema(c, float(h), order)), axis=1)
                              for c, h in zip(polys, gaps)])
            if np.any(peaks > np.minimum(cap, configured)*(1.+1e-10)):
                raise ValueError(side + '同步归位速度或加速度超限')
        starts = 6.*polys[:, :, 3]/gaps[:, None]**3
        ends = (6.*polys[:, :, 3]+24.*polys[:, :, 4]+60.*polys[:, :, 5])/gaps[:, None]**3
        previous = plan.polynomials[first-1]
        h = (plan.times_ns[first]-plan.times_ns[first-1])/1e9
        previous_jerk = (6.*previous[:, 3]+24.*previous[:, 4]+60.*previous[:, 5])/h**3
        if max(float(np.max(abs(starts[1:]-ends[:-1]))),
               float(np.max(abs(starts[0]-previous_jerk)))) > 1e-5:
            raise ValueError(side + '同步归位jerk衔接跳变超限')
    return first, last


def load_reviewed_task(
    run_dir: Path, limits_file: Path, prepared_file: Path, report_file: Path, *,
    max_excursion: float,
    cancelled: Callable[[], bool] | None = None,
) -> ReviewedTask:
    """核对快照哈希、原始采样、阶段与连续整段约束，始终不授予执行许可."""
    excursion = positive(max_excursion, 'max_excursion')
    if cancelled is not None and cancelled():
        raise ValueError('任务文件核对已取消')
    root = run_dir.expanduser().resolve()
    names = [n for n in ('state.csv', 'tracking.csv') if (root / n).is_file()]
    if len(names) != 1:
        raise ValueError('目录中必须恰有一种任务CSV，不自动选择混合记录')
    paths = {
        'csv': root / names[0], 'summary': root / 'summary.json',
        'limits': limits_file.expanduser().resolve(),
        'prepared': prepared_file.expanduser().resolve(),
        'report': report_file.expanduser().resolve(),
    }
    if len({p.resolve() for p in paths.values()}) != len(paths):
        raise ValueError('五种任务输入必须是不同文件')
    snapshots = {name: read_snapshot(p, (50 if name in ('csv', 'prepared', 'report') else 2)
                                     * 1024 * 1024) for name, p in paths.items()}
    hashes = {name: hashlib.sha256(data).hexdigest() for name, data in snapshots.items()}
    summary, report, payload = (json_mapping(snapshots[n])
                                for n in ('summary', 'report', 'prepared'))
    task = completed_task(summary, names[0])
    for name, field in (('csv', 'csv_sha256'), ('summary', 'summary_sha256'),
                        ('limits', 'limits_sha256'), ('prepared', 'output_sha256')):
        if report.get(field) != hashes[name]:
            raise ValueError('文件与准备报告不一致：' + name)
    if any(report.get(k) is not True for k in (
        'import_succeeded', 'preparation_succeeded', 'continuous_constraints_passed',
    )):
        raise ValueError('需要成功的连续轨迹准备报告')
    limits = load_limits(paths['limits'], content=snapshots['limits'])
    samples, measured = import_samples(snapshots['csv'], limits)
    plans = prepare_continuous_pair(payload, limits, excursion, cancelled=cancelled)
    for field in ('source_time_origin_ns', 'source_time_basis', 'paired_samples', 'source_rows'):
        if report.get(field) != measured[field]:
            raise ValueError('准备报告与原始采样不一致：' + field)
    if report.get('stop_indices') != payload['stop_indices']:
        raise ValueError('准备报告停止点与轨迹不同')
    count = measured['paired_samples']
    if task == 'cooperative_grasp':
        recorded_count = summary['recording'].get('sample_count')
        if type(recorded_count) is not int or recorded_count != measured['source_rows']:
            raise ValueError('抓取摘要采样数与CSV不一致')
    elif summary.get('records') != {'left': count, 'right': count}:
        raise ValueError('画圆摘要双侧采样数与CSV不一致')
    if type(report.get('samples_preserved')) is not int or report['samples_preserved'] != count:
        raise ValueError('准备报告没有保留全部采样')
    stage_report = report.get('source_stages')
    if not isinstance(stage_report, dict) or set(stage_report) != {'left', 'right'}:
        raise ValueError('准备报告缺少双侧原始阶段')
    required_stops = {0, count - 1}
    tolerance = 0.
    if report.get('position_filter') is not None:
        value = report.get('position_smoothing_tolerance_rad')
        if (report['position_filter'] != 'bounded_spatial_spline'
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0. < value <= .0005):
            raise ValueError('空间路径拟合的节点误差上限无效')
        tolerance = float(value)
    return_range = _return_profile_range(report, task, plans, stage_report, tolerance)
    changes = {}
    source_times: list[int] = []
    for side in ('left', 'right'):
        raw = samples[side]['points']
        points = payload[side]['points']
        labels = [p['source_stage'] for p in raw]
        check_task_stages(labels, task)
        if labels != stage_report[side] or len(points) != len(raw):
            raise ValueError(side + '采样数量或阶段被改动')
        maximum = 0.
        for i, (original, prepared) in enumerate(zip(raw, points)):
            if cancelled is not None and cancelled():
                raise ValueError('任务采样核对已取消')
            difference = max(abs(a - b) for a, b in
                             zip(prepared['positions'], original['positions']))
            maximum = max(maximum, difference)
            allowed = .004 if return_range is not None and i > return_range[0] else tolerance
            if difference > allowed + (1e-12 if allowed else 0.):
                raise ValueError(side + '原始实际关节角被改写')
            if tolerance and i in payload['stop_indices'] and difference != 0.:
                raise ValueError(side + '阶段停车边界不得进行位置拟合修改')
            t = original['time_from_start']
            source_ns = t['sec'] * 10 ** 9 + t['nanosec']
            if side == 'left':
                source_times.append(source_ns)
            if i and labels[i] != labels[i - 1]:
                required_stops.update((i - 1, i))
        changes[side] = maximum
        if tolerance:
            reported_changes = report.get('max_sample_position_change_rad')
            if not isinstance(reported_changes, Mapping):
                raise ValueError('缺少双侧节点拟合偏差报告')
            value = reported_changes.get(side)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not math.isclose(value, maximum, rel_tol=0., abs_tol=1e-12)):
                raise ValueError(side + '节点拟合偏差与实际文件不一致')
    # 复用准备器的事件停顿规则，参考事件依然不会转换为SDK指令。
    from bisect import bisect_left

    held = tuple(all(
        samples[s]['points'][i]['positions'] == samples[s]['points'][i + 1]['positions']
        for s in ('left', 'right')) for i in range(count - 1))
    mapping = RecordedTimeMap.from_report(report, tuple(source_times), plans['left'].times_ns,
                                          held_intervals=held)
    events = []
    for event in samples['reference_events']:
        relative = event['source_time_ns'] - samples['source_time_origin_ns']
        index = min(count - 1, bisect_left(source_times, relative))
        required_stops.update((max(0, index - 1), index))
        events.append({**event, 'prepared_time_ns': mapping.map_ns(relative)})
    if report.get('reference_events') != events:
        raise ValueError('参考事件或事件时间与原始记录不同')
    blend = report.get('boundary_blend')
    if blend is not None:
        if (task != 'split_circle' or events or tolerance != .0005
                or not isinstance(blend, dict) or blend.get('policy') != 'circle_local_v1'):
            raise ValueError('局部边界融合仅支持无事件的有界空间画圆记录')
        windows = circle_blend_windows(
            [t / 1e9 for t in plans['left'].times_ns], stage_report,
            tuple(sorted(required_stops)))
        claimed = blend.get('windows')
        if not isinstance(claimed, list) or len(claimed) != len(windows):
            raise ValueError('缺少两个独立画圆衔接窗口')
        for expected, supplied in zip(windows, claimed):
            if (not isinstance(supplied, dict)
                    or any(supplied.get(k) != v for k, v in expected.items())
                    or type(supplied.get('strength')) not in (int, float)
                    or supplied['strength'] not in (1., .8, .6, .4, .2)):
                raise ValueError('画圆融合窗口与原始阶段或实际时间轴不一致')
            if any(held[expected['first']:expected['last']]) or any(
                k in payload['stop_indices'] for k in expected['removed_stops']
            ):
                raise ValueError('实际保持区间或未融合边界不得声明连续通过')
            required_stops.difference_update(expected['removed_stops'])
    if not required_stops <= set(payload['stop_indices']):
        raise ValueError('任务阶段或夹爪参考事件缺少必要停顿')
    duration = plans['left'].times_ns[-1] / 1e9
    details = {
        'task_files_verified': True, 'task_type': task, 'paired_samples': count,
        'duration_s': duration, 'max_excursion_rad': excursion, 'sha256': hashes,
        'position_smoothing_tolerance_rad': tolerance,
        'verified_max_sample_position_change_rad': changes,
        'return_profile_range': list(return_range) if return_range is not None else None,
        'execution_ready': False, 'commands_sent': False,
        'geometry_preflight_passed': False, 'actual_start_verified': False,
        'gripper_execution_verified': False,
        'scope': '验证文件快照一致、整轮阶段、采样保留及连续关节约束；不验证碰撞或实际抓取',
    }
    return ReviewedTask(MappingProxyType(plans), MappingProxyType(details))


def review_task_files(
    run_dir: Path, limits_file: Path, prepared_file: Path, report_file: Path, *,
    max_excursion: float,
) -> dict[str, Any]:
    """兼容既有只读报告接口，内存对象供几何检查直接复用."""
    reviewed = load_reviewed_task(run_dir, limits_file, prepared_file, report_file,
                                  max_excursion=max_excursion)
    return dict(reviewed.report)


def main(args: list[str] | None = None) -> int:
    """只读检查指定文件并在终端输出报告，不生成额外任务文件或初始化ROS."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--limits-file', type=Path, required=True)
    parser.add_argument('--prepared-file', type=Path, required=True)
    parser.add_argument('--report-file', type=Path, required=True)
    parser.add_argument('--max-excursion', type=float, required=True)
    parser.add_argument('--geometry', action='store_true', help='检查采样跨臂和台面间隙')
    parser.add_argument('--start-file', type=Path, help='离线双臂起点JSON，不代表实时反馈')
    parser.add_argument('--approach-max-excursion', type=float)
    parser.add_argument('--min-approach-duration', type=float, default=1.)
    parser.add_argument('--geometry-max-samples', type=int, default=100000)
    options = parser.parse_args(args)
    if options.start_file is not None and (
        not options.geometry or options.approach_max_excursion is None
    ):
        parser.error('--start-file需要--geometry及明确的--approach-max-excursion')
    if options.approach_max_excursion is not None and options.start_file is None:
        parser.error('--approach-max-excursion需要--start-file')
    try:
        reviewed = load_reviewed_task(options.run_dir, options.limits_file, options.prepared_file,
                                      options.report_file, max_excursion=options.max_excursion)
        result = dict(reviewed.report)
        if options.geometry:
            from .task_geometry import (
                assemble_task_cycle, check_task_geometry, prepare_approach, prepare_return,
            )

            approach = None
            return_home = None
            if options.start_file is not None:
                start_bytes = read_snapshot(options.start_file.expanduser(), 1024 * 1024)
                provided_start = json_mapping(start_bytes)
                approach = prepare_approach(
                    reviewed.plans, provided_start,
                    max_excursion=positive(options.approach_max_excursion,
                                           'approach_max_excursion'),
                    minimum_duration=options.min_approach_duration,
                )
                return_home = prepare_return(
                    reviewed.plans, provided_start,
                    max_excursion=positive(options.approach_max_excursion,
                                           'approach_max_excursion'),
                    minimum_duration=options.min_approach_duration,
                )
                cycle = assemble_task_cycle(
                    reviewed.plans, approach=approach, return_home=return_home,
                    max_excursion=options.max_excursion,
                )
                result['assembled_cycle'] = {
                    'duration_s': cycle['left'].duration,
                    'time_points': len(cycle['left'].times_ns),
                    'phase_start_ns': {
                        'APPROACH_FILE': 0,
                        'TASK': approach['left'].times_ns[-1],
                        'RETURN_ACTUAL': (approach['left'].times_ns[-1]
                                          + reviewed.plans['left'].times_ns[-1]),
                    },
                    'original_task_polynomials_preserved': True,
                    'returns_to_provided_start': True,
                    'live_feedback_verified': False,
                    'execution_ready': False,
                }
                result['return_actual'] = {
                    'duration_s': return_home['left'].duration,
                    'goals': {side: list(p.goal) for side, p in return_home.items()},
                }
                result['provided_start_sha256'] = hashlib.sha256(start_bytes).hexdigest()
                result['approach'] = {
                    'duration_s': approach['left'].duration,
                    'source': 'offline_provided_start', 'live_feedback_verified': False,
                    'starts': {side: list(p.initial) for side, p in approach.items()},
                    'goals': {side: list(p.goal) for side, p in approach.items()},
                }
            result['sampled_geometry'] = check_task_geometry(
                reviewed.plans, approach=approach, return_home=return_home,
                max_samples=options.geometry_max_samples,
            )
            result['scope'] = '文件一致性及连续关节约束通过；附带离散跨臂和台面检查，执行许可仍关闭'
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError,
            yaml.YAMLError, RuntimeError, ImportError) as error:
        print(json.dumps({'task_files_verified': False, 'execution_ready': False,
                          'commands_sent': False, 'error': str(error)}, ensure_ascii=False))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return 0
