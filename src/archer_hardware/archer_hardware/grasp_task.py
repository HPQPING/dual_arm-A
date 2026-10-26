"""协同真机任务入口：读取已转换轨迹，按共同时间开合夹爪，整轮归位后关闭。"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

from .file_task import resolve_trajectory_input, run


def default_limits_file() -> Path:
    """独立配置与画圆配置分开，均由功能包安装。"""
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory('archer_hardware')) / 'config/grasp_real.yaml'


def main(args: list[str] | None = None) -> int:
    """显式许可才创建控制节点；输入为state.json，不是原始CSV或报告。"""
    raw = list(sys.argv[1:] if args is None else args)
    remove_ros_args = import_module('rclpy.utilities').remove_ros_args
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-authorized-full-grasp', required=True, action='store_true',
                        help='实际连接双臂和夹爪执行完整协同任务，包含开合和自动归位')
    parser.add_argument('--trajectory-file', required=True, type=Path,
                        help='协同转换后的state.json，自动读取同名报告和原始记录')
    parser.add_argument('--run-dir', type=Path, help='原始记录迁移时显式提供新目录')
    parser.add_argument('--limits-file', type=Path, help='默认config/grasp_real.yaml')
    options = parser.parse_args(remove_ros_args(args=['hardware_grasp', *raw])[1:])
    try:
        config = options.limits_file or default_limits_file()
        source, directory, limits, prepared = resolve_trajectory_input(
            options.trajectory_file, options.run_dir, config, expected_csv_name='state.csv')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    return run(directory, source, limits, trajectory_file=prepared, ros_args=raw,
               task_type='cooperative_grasp')


if __name__ == '__main__':
    raise SystemExit(main())
