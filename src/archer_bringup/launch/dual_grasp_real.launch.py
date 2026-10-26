"""一条命令调度双臂协同真机任务；默认不连接，execute=true才执行。"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, LogInfo, OpaqueFunction, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _start(context):
    if LaunchConfiguration('execute').perform(context) != 'true':
        return [LogInfo(msg='未连接机械臂；执行协同任务需显式指定execute:=true。')]
    trajectory = LaunchConfiguration('trajectory_file').perform(context).strip()
    if not trajectory:
        raise ValueError('必须指定协同转换后的state.json，尚未建立连接')
    arguments = ['--execute-authorized-full-grasp', '--trajectory-file', trajectory]
    for name in ('run_dir', 'limits_file'):
        value = LaunchConfiguration(name).perform(context).strip()
        if value:
            arguments.extend(['--' + name.replace('_', '-'), value])
    task = Node(package='archer_hardware', executable='hardware_grasp',
                arguments=arguments, output='both', respawn=False,
                additional_env={'PYTHONUNBUFFERED': '1'})
    return [RegisterEventHandler(OnProcessExit(
        target_action=task,
        on_exit=[EmitEvent(event=Shutdown(reason='协同任务调度器已退出'))],
    )), task]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('trajectory_file', default_value='',
                              description='转换后state.json，自动定位同名报告和原始记录'),
        DeclareLaunchArgument('run_dir', default_value='', description='可选的原始记录目录'),
        DeclareLaunchArgument('limits_file', default_value='',
                              description='默认archer_hardware/config/grasp_real.yaml'),
        DeclareLaunchArgument('execute', default_value='false', choices=['true', 'false'],
                              description='true表示允许双臂轨迹、夹爪开合及自动归位'),
        OpaqueFunction(function=_start),
    ])
