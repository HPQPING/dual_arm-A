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
from typing import Any

import yaml

from .commissioning import load_limits
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


def load_reviewed_task(
    run_dir: Path, limits_file: Path, prepared_file: Path, report_file: Path, *,
    max_excursion: float,
) -> ReviewedTask:
    """核对快照哈希、原始采样、阶段与连续整段约束，始终不授予执行许可."""
    excursion = positive(max_excursion, 'max_excursion')
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
    plans = prepare_continuous_pair(payload, limits, excursion)
    scale = positive(report.get('uniform_time_scale'), 'uniform_time_scale')
    if scale < 1.:
        raise ValueError('准备过程不允许缩短原始采样时间')
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
    source_times: list[int] = []
    for side in ('left', 'right'):
        raw = samples[side]['points']
        points = payload[side]['points']
        labels = [p['source_stage'] for p in raw]
        check_task_stages(labels, task)
        if labels != stage_report[side] or len(points) != len(raw):
            raise ValueError(side + '采样数量或阶段被改动')
        for i, (original, prepared) in enumerate(zip(raw, points)):
            if prepared['positions'] != original['positions']:
                raise ValueError(side + '原始实际关节角被改写')
            t = original['time_from_start']
            source_ns = t['sec'] * 10 ** 9 + t['nanosec']
            if side == 'left':
                source_times.append(source_ns)
            # 原准备器先转秒再统一伸长，允许往返浮点和纳秒向上取整的2ns差异。
            if abs(plans[side].times_ns[i] - math.ceil(source_ns * scale)) > 2:
                raise ValueError(side + '准备时间与统一时间伸长不一致')
            if i and labels[i] != labels[i - 1]:
                required_stops.update((i - 1, i))
    # 复用准备器的事件停顿规则，参考事件依然不会转换为SDK指令。
    from bisect import bisect_left

    events = []
    for event in samples['reference_events']:
        relative = event['source_time_ns'] - samples['source_time_origin_ns']
        index = min(count - 1, bisect_left(source_times, relative))
        required_stops.update((max(0, index - 1), index))
        events.append({**event, 'prepared_time_ns': math.ceil(relative * scale)})
    if report.get('reference_events') != events:
        raise ValueError('参考事件或事件时间与原始记录不同')
    if not required_stops <= set(payload['stop_indices']):
        raise ValueError('任务阶段或夹爪参考事件缺少必要停顿')
    duration = plans['left'].times_ns[-1] / 1e9
    details = {
        'task_files_verified': True, 'task_type': task, 'paired_samples': count,
        'duration_s': duration, 'max_excursion_rad': excursion, 'sha256': hashes,
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
            from .task_geometry import check_task_geometry, prepare_approach, prepare_return

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
