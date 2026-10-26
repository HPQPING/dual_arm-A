"""画圆真机入口；复用共用整轮调度，保持现有命令、参数和运动策略。"""

from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

from .file_task import resolve_inputs as resolve_inputs
from .file_task import resolve_trajectory_input as resolve_trajectory_input
from .file_task import run as run


def default_limits_file() -> Path:
    """从已安装功能包获取成功版配置，不依赖开发缓存目录。"""
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory('archer_hardware')) / 'config/circle_real.yaml'


def main(args: list[str] | None = None) -> int:
    """注册的ROS2入口；显式执行许可保留，支持launch追加的ROS参数。"""
    remove_ros_args = import_module('rclpy.utilities').remove_ros_args

    raw = list(sys.argv[1:] if args is None else args)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-authorized-full-circle', action='store_true', required=True,
                        help='实际连接双臂并运行整轮画圆，不是离线预览')
    parser.add_argument('--trajectory-file', type=Path,
                        help='只提供轨迹JSON，自动定位报告及原始任务记录')
    parser.add_argument('--run-dir', type=Path,
                        help='已完成画圆仿真的原始记录目录，不固定任务日期')
    parser.add_argument('--prepared-dir', type=Path,
                        help='默认同一工作空间archer_results_trans下的同名任务目录')
    parser.add_argument('--limits-file', type=Path,
                        help='默认使用功能包安装的config/circle_real.yaml')
    options = parser.parse_args(remove_ros_args(args=['hardware_circle', *raw])[1:])
    try:
        limits = options.limits_file if options.limits_file is not None else default_limits_file()
        if options.trajectory_file is not None:
            if options.prepared_dir is not None:
                parser.error('--trajectory-file与--prepared-dir不能同时指定')
            source, directory, config, prepared = resolve_trajectory_input(
                options.trajectory_file, options.run_dir, limits)
            return run(directory, source, config, ros_args=raw, trajectory_file=prepared)
        if options.run_dir is None:
            parser.error('请提供--trajectory-file，或使用原有--run-dir')
        source, directory, config = resolve_inputs(options.run_dir, options.prepared_dir, limits)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return run(directory, source, config, ros_args=raw)


if __name__ == '__main__':
    raise SystemExit(main())
