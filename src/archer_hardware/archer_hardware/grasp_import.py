"""将完整协同仿真 state.csv 转换为双臂连续轨迹和夹爪时序，不连接机械臂。"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml

from .circle_import import import_samples, replace_conversion_pair
from .grasp_events import event_schedule
from .grasp_smoothing import smooth_grasp_samples
from .position_plan import load_limits
from .task_file import load_reviewed_task, read_snapshot


def convert_grasp(
    source: Path, config: Path, *, margin: float = 1.05, max_excursion: float = 3.5,
) -> tuple[bytes, bytes]:
    """先核对整轮记录，再复用连续规划；失败前不写正式输出，也不使用画圆专用优化。"""
    from archer_control.checks.grasp_record_check import load_run

    source, config = source.expanduser().resolve(), config.expanduser().resolve()
    if source.name != 'state.csv' or not source.is_file():
        raise ValueError('--input需要完整协同记录中的state.csv')
    paths = {'csv': source, 'summary': source.with_name('summary.json'),
             'events': source.with_name('events.csv'), 'limits': config}
    content = {key: read_snapshot(path, (50 if key == 'csv' else 2) * 1024 * 1024)
               for key, path in paths.items()}
    recorded = load_run(source.parent)
    for key, path in paths.items():
        if read_snapshot(path, (50 if key == 'csv' else 2) * 1024 * 1024) != content[key]:
            raise ValueError('核对期间任务输入发生变化，请重试：' + key)
    limits = load_limits(config, content=content['limits'])
    samples, report = import_samples(content['csv'], limits)
    raw_config: object = yaml.safe_load(content['limits'])
    driver = raw_config.get('archer_position_driver') if isinstance(raw_config, dict) else None
    parameters = driver.get('ros__parameters') if isinstance(driver, dict) else None
    if not isinstance(parameters, dict):
        raise ValueError('协同配置缺少archer_position_driver/ros__parameters')
    hz = parameters.get('control_hz', 250)
    if type(hz) is not int or hz not in (250, 500):
        raise ValueError('协同配置需要250或500Hz控制频率')
    prepared, timing = smooth_grasp_samples(samples, limits, execution_hz=hz, margin=margin)
    report.update(timing)
    report.update({
        'task_type': 'cooperative_grasp', 'source_csv': str(source),
        'source_events': str(paths['events']), 'limits_file': str(config),
        'csv_sha256': hashlib.sha256(content['csv']).hexdigest(),
        'summary_sha256': hashlib.sha256(content['summary']).hexdigest(),
        'events_sha256': hashlib.sha256(content['events']).hexdigest(),
        'limits_sha256': hashlib.sha256(content['limits']).hexdigest(),
        'conversion_mode': 'cooperative_continuous',
        'cooperative_timing_policy': 'bounded_phase_quintic_v2',
        'planned_control_hz': hz,
        'gripper_event_schedule': event_schedule(samples, prepared, recorded.summary),
        'execution_ready': False, 'commands_sent': False,
        'converted_at': datetime.now().astimezone().isoformat(timespec='microseconds'),
    })
    output_bytes = (json.dumps(prepared, ensure_ascii=False, indent=2, allow_nan=False)
                    + '\n').encode('utf-8')
    report['output_sha256'] = hashlib.sha256(output_bytes).hexdigest()
    report_bytes = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
                    + '\n').encode('utf-8')
    # 使用正式读取器独立检查待交付字节；任何失败都不会替换上次的转换结果。
    with TemporaryDirectory(prefix='archer_grasp_import_') as folder:
        pending = Path(folder)
        (pending / 'state.json').write_bytes(output_bytes)
        (pending / 'state_report.json').write_bytes(report_bytes)
        load_reviewed_task(source.parent, config, pending / 'state.json',
                           pending / 'state_report.json', max_excursion=max_excursion)
    return output_bytes, report_bytes


def main(args: list[str] | None = None) -> int:
    """单文件地址入口；同名成功转换替换旧结果，出错时保留旧结果。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='原始协同任务state.csv路径')
    parser.add_argument('--limits-file', type=Path, help='默认config/grasp_real.yaml')
    parser.add_argument('--output-dir', type=Path, help='默认archer_results_trans/原任务目录名')
    parser.add_argument('--margin', type=float, default=1.05)
    options = parser.parse_args(args)
    source = options.input.expanduser().resolve()
    if options.limits_file is None:
        from ament_index_python.packages import get_package_share_directory

        options.limits_file = (Path(get_package_share_directory('archer_hardware'))
                               / 'config/grasp_real.yaml')
    config = options.limits_file.expanduser().resolve()
    workspace = (source.parent.parent.parent if source.parent.parent.name == 'archer_results'
                 else Path.home() / 'archer_ws')
    destination = (options.output_dir or workspace / 'archer_results_trans' / source.parent.name
                   ).expanduser().resolve()
    output, evidence = destination / 'state.json', destination / 'state_report.json'
    inputs = {source, source.with_name('summary.json'), source.with_name('events.csv'), config}
    if output in inputs or evidence in inputs:
        parser.error('输出不能覆盖任务记录或配置')
    try:
        payload, report = convert_grasp(source, config, margin=options.margin)
        replace_conversion_pair(output, payload, evidence, report)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError,
            UnicodeError, yaml.YAMLError) as error:
        print('协同转换失败：' + str(error) + '；未替换旧结果，未发送动作')
        return 1
    details: dict[str, Any] = json.loads(report)
    print('协同连续轨迹：' + str(output))
    print('转换及夹爪时序报告：' + str(evidence))
    print(f"计划约{details['duration_s']:.2f}秒；双臂共同时间轴，开爪后直接进给，不含固定等待。")
    print('仅完成离线转换，不会启动真机；真实夹爪及执行许可尚未核验。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
