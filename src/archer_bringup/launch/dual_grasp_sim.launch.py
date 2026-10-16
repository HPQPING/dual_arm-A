"""启动完整圆柱的简化抓取 ROS 演示，等待服务命令，不自动运动。

同一 ROS 域内使用时，应先停止原双臂仿真，避免两套关节状态发布者。
夹爪指令 2.0 表示虚拟附着；不验证真实夹紧力，不启动画圆任务。
"""

import json
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.action import Action
from launch.actions import (DeclareLaunchArgument, EmitEvent, ExecuteProcess,
                            IncludeLaunchDescription, OpaqueFunction, RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
import yaml


def validate_view_parameters(path: Path) -> None:
    """抓取仿真当前使用固定安装参数，拒绝显示另一套基座或工具坐标。"""
    raw: object = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise ValueError('显示参数必须为 YAML 映射')
    expected = {
        'left_base_xyz': [-.21, 0., 0.], 'right_base_xyz': [.21, 0., 0.],
        'left_base_rpy': [0., 0., math.pi / 2.], 'right_base_rpy': [0., 0., math.pi / 2.],
        'tcp_xyz': [0., 0., .147021696], 'mount_xyz': [0., 0., 0.], 'mount_rpy': [0., 0., 0.],
    }
    for name in ('dual_arm_scene', 'dual_arm_task'):
        node = raw.get(name)
        values = node.get('ros__parameters') if isinstance(node, dict) else None
        if not isinstance(values, dict):
            raise ValueError('显示参数缺少 ' + name + '/ros__parameters')
        if values.get('use_sim_time') is not False or values.get('left_urdf') or values.get('right_urdf'):
            raise ValueError('抓取显示要求系统时间和默认 Xacro 模型')
        for key, vector in expected.items():
            actual = values.get(key)
            if not isinstance(actual, list) or len(actual) != 3:
                raise ValueError('显示参数必须是三维向量：' + key)
            for a, b in zip(actual, vector):
                if isinstance(a, bool) or not isinstance(a, (int, float)) or not math.isclose(a, b, abs_tol=1e-9, rel_tol=0.):
                    raise ValueError('显示参数与抓取物理模型不一致：' + name + '/' + key)


def _start(context) -> list[Action]:
    # 将路径编码成 YAML 兼容字符串，空值和带空格的路径也能正确传递。
    output = LaunchConfiguration('output_file').perform(context)
    python = LaunchConfiguration('python_executable').perform(context)
    show_viewer = LaunchConfiguration('show_viewer').perform(context)
    process = ExecuteProcess(
        cmd=[python, '-m', 'archer_control.grasp_ros_node', '--ros-args',
             '-p', 'use_sim_time:=false', '-p', 'output_file:=' + json.dumps(output),
             '-p', 'show_viewer:=' + show_viewer],
        output='screen',
    )
    actions: list[Action] = [RegisterEventHandler(OnProcessExit(
        target_action=process, on_exit=[EmitEvent(event=Shutdown(reason='抓取仿真进程已退出'))])), process]
    if LaunchConfiguration('show_rviz').perform(context) == 'true':
        share = Path(get_package_share_directory('archer_bringup'))
        config = Path(LaunchConfiguration('rviz_params_file').perform(context)).expanduser().resolve()
        validate_view_parameters(config)
        rviz = share / 'rviz/dual_grasp.rviz'
        if not rviz.is_file():
            raise FileNotFoundError('缺少抓取 RViz 配置，请重新编译 archer_bringup：' + str(rviz))
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(share / 'launch/dual_arm_view.launch.py')),
            launch_arguments={'params_file': str(config), 'rviz_config': str(rviz)}.items(),
        ))
    return actions


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory('archer_bringup'))
    return LaunchDescription([
        DeclareLaunchArgument('python_executable', default_value='/usr/bin/python3',
                              description='能够导入 ROS 2 与 MuJoCo 的 Python 解释器'),
        DeclareLaunchArgument('output_file', default_value='', description='可选 JSON 结果文件，留空仅发布话题'),
        DeclareLaunchArgument('show_viewer', default_value='false', choices=['true', 'false'],
                              description='是否显示 MuJoCo 窗口；true 需要 Ubuntu 桌面显示环境'),
        DeclareLaunchArgument('show_rviz', default_value='false', choices=['true', 'false'],
                              description='是否同时启动双臂 TF、反馈汇总和抓取 RViz 界面'),
        DeclareLaunchArgument('rviz_params_file', default_value=str(share / 'config/dual_circle.yaml'),
                              description='仅用于核对显示模型安装参数，不启动画圆任务'),
        OpaqueFunction(function=_start),
    ])
