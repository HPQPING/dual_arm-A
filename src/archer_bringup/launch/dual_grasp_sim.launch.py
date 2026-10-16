"""启动独立抓取 ROS 仿真，等待服务命令，不自动运动。

同一 ROS 域内使用时，应先停止原双臂仿真，避免两套关节状态发布者。
"""

import json

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration


def _start(context) -> list[ExecuteProcess]:
    # 将路径编码成 YAML 兼容字符串，空值和带空格的路径也能正确传递。
    output = LaunchConfiguration('output_file').perform(context)
    python = LaunchConfiguration('python_executable').perform(context)
    return [ExecuteProcess(
        cmd=[python, '-m', 'archer_control.grasp_ros_node', '--ros-args',
             '-p', 'use_sim_time:=false', '-p', 'output_file:=' + json.dumps(output)],
        output='screen',
    )]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument('python_executable', default_value='/usr/bin/python3',
                              description='能够导入 ROS 2 与 MuJoCo 的 Python 解释器'),
        DeclareLaunchArgument('output_file', default_value='', description='可选 JSON 结果文件，留空仅发布话题'),
        OpaqueFunction(function=_start),
    ])
