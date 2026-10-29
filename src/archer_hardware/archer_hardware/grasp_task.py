"""协同真机任务入口：读取已转换轨迹，按共同时间开合夹爪，整轮归位后关闭。"""

from __future__ import annotations

import argparse
import json
from importlib import import_module
from pathlib import Path
import sys

from .file_task import resolve_trajectory_input, run
from .cooperative_feedback import CarryFeedbackAdapter, qualify_return
from .position_plan import load_limits


def default_limits_file() -> Path:
    """独立配置与画圆配置分开，均由功能包安装。"""
    get_package_share_directory = import_module(
        'ament_index_python.packages'
    ).get_package_share_directory

    return Path(get_package_share_directory('archer_hardware')) / 'config/grasp_real.yaml'


def qualify_prepared_return(prepared: Path, config: Path) -> None:
    """原192归位预算先于创建ROS节点和SDK连接，不重复完整几何预检。"""
    payload = json.loads(prepared.read_text(encoding='utf8'))
    report = json.loads(
        prepared.with_name(prepared.stem + '_report.json').read_text(encoding='utf8')
    )
    qualify_return(payload, report, load_limits(config))


def main(args: list[str] | None = None) -> int:
    """显式许可才创建控制节点；输入为state.json，不是原始CSV或报告。"""
    raw = list(sys.argv[1:] if args is None else args)
    remove_ros_args = import_module('rclpy.utilities').remove_ros_args
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true', help='仅核对文件与归位预算，不连接设备')
    mode.add_argument(
        '--execute-authorized-full-grasp',
        action='store_true',
        help='实际连接双臂和夹爪执行完整协同任务，包含开合和自动归位',
    )
    parser.add_argument(
        '--trajectory-file',
        required=True,
        type=Path,
        help='协同转换后的state.json，自动读取同名报告和原始记录',
    )
    parser.add_argument('--run-dir', type=Path, help='原始记录迁移时显式提供新目录')
    parser.add_argument('--limits-file', type=Path, help='默认config/grasp_real.yaml')
    options = parser.parse_args(remove_ros_args(args=['hardware_grasp', *raw])[1:])
    try:
        config = options.limits_file or default_limits_file()
        source, directory, limits, prepared = resolve_trajectory_input(
            options.trajectory_file, options.run_dir, config, expected_csv_name='state.csv'
        )
        qualify_prepared_return(prepared, limits)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if options.check_only:
        print('协同轨迹、报告及归位预算核对通过；尚未连接机械臂。', flush=True)
        return 0
    return run(
        directory,
        source,
        limits,
        trajectory_file=prepared,
        ros_args=raw,
        task_type='cooperative_grasp',
        backend_type=CarryFeedbackAdapter,
    )


if __name__ == '__main__':
    raise SystemExit(main())
