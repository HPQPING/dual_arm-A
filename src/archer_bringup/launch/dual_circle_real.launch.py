"""启动功能包内的完整画圆真机调度器，显式许可后执行并自动归位。"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, LogInfo, OpaqueFunction, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _start(context):
    """只启动一份完整任务调度器，不重复启动底层位置驱动。"""
    if LaunchConfiguration('execute').perform(context) != 'true':
        return [LogInfo(msg='未启动真机任务；确认执行时请指定 execute:=true。')]
    source = LaunchConfiguration('run_dir').perform(context).strip()
    trajectory = context.launch_configurations.get('trajectory_file', '').strip()
    if not source and not trajectory:
        raise ValueError('必须指定trajectory_file或原有run_dir，未启动设备连接')
    arguments = ['--execute-authorized-full-circle']
    if trajectory:
        if LaunchConfiguration('prepared_dir').perform(context).strip():
            raise ValueError('trajectory_file与prepared_dir不能同时指定')
        arguments.extend(['--trajectory-file', trajectory])
    if source:
        arguments.extend(['--run-dir', source])
    for name in ('prepared_dir', 'limits_file'):
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            arguments.extend(['--' + name.replace('_', '-'), value])
    task = Node(
        package='archer_hardware', executable='hardware_circle',
        arguments=arguments, output='screen', respawn=False,
        additional_env={'PYTHONUNBUFFERED': '1'},
    )
    return [
        RegisterEventHandler(OnProcessExit(
            target_action=task,
            on_exit=[EmitEvent(event=Shutdown(reason='画圆真机任务调度器已退出'))],
        )),
        task,
    ]


def generate_launch_description() -> LaunchDescription:
    """任务日期不写死；默认转换目录和成功配置由功能包入口解析。"""
    return LaunchDescription([
        DeclareLaunchArgument('trajectory_file', default_value='',
                              description='轨迹JSON路径，自动定位报告及原始任务'),
        DeclareLaunchArgument('run_dir', default_value='',
                              description='兼容原始任务目录参数，trajectory_file模式可留空'),
        DeclareLaunchArgument('prepared_dir', default_value='',
                              description='默认archer_results_trans下的同名任务目录'),
        DeclareLaunchArgument('limits_file', default_value='',
                              description='默认archer_hardware/config/circle_real.yaml'),
        DeclareLaunchArgument('execute', default_value='false', choices=['true', 'false'],
                              description='true表示实际连接双臂并执行完整画圆及归位'),
        OpaqueFunction(function=_start),
    ])
