"""只读显示双臂模型、TF 和轨迹；不启动物理仿真，不发送关节命令。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.action import Action
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from archer_control.check_circle import validate_tool_chain
from archer_control.dual_arm_trajectory import rigid_transform
from archer_control.kinematics import ArmKinematics
from archer_control.single_arm_node import numbers


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(label + ' 必须是字符串键的 YAML 映射')
    return {str(key): item for key, item in value.items()}


def build_description(source: Path, parameters: dict[str, object]) -> str:
    """展开双臂模型，并核对其基座和工具坐标与运行配置一致。"""
    vectors = {name: numbers(parameters[name], name) for name in (
        'left_base_xyz', 'right_base_xyz', 'left_base_rpy', 'right_base_rpy',
        'mount_xyz', 'mount_rpy', 'tcp_xyz')}
    if any(len(value) != 3 for value in vectors.values()):
        raise ValueError('基座和工具参数必须都是三维向量')
    left, right = vectors['left_base_xyz'], vectors['right_base_xyz']
    if (left[0] >= right[0] or not np.allclose(
            [left[0] + right[0], left[1], right[1]], [0., 0., 0.], atol=1e-9, rtol=0.)):
        raise ValueError('当前双臂 Xacro 要求基座位于中点 world 的 X 轴两侧')
    if parameters.get('left_urdf') or parameters.get('right_urdf'):
        raise ValueError('本显示入口采用项目 Xacro，请将两侧自定义 URDF 路径留空')
    arguments = {
        'base_distance': str(right[0] - left[0]),
        'left_base_height': str(left[2]), 'right_base_height': str(right[2]),
        'left_base_rpy': ' '.join(str(v) for v in vectors['left_base_rpy']),
        'right_base_rpy': ' '.join(str(v) for v in vectors['right_base_rpy']),
        'left_gripper_mount_xyz': ' '.join(str(v) for v in vectors['mount_xyz']),
        'right_gripper_mount_xyz': ' '.join(str(v) for v in vectors['mount_xyz']),
        'left_gripper_mount_rpy': ' '.join(str(v) for v in vectors['mount_rpy']),
        'right_gripper_mount_rpy': ' '.join(str(v) for v in vectors['mount_rpy']),
    }
    try:
        result = subprocess.run(['xacro', str(source), *[f'{k}:={v}' for k, v in arguments.items()]],
                                check=True, capture_output=True, text=True, timeout=30)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError('双臂 Xacro 展开失败：' + exc.stderr) from exc
    root = ET.fromstring(result.stdout)
    for side in ('left', 'right'):
        model = ArmKinematics(root, side)
        validate_tool_chain(model)
        tcp = next(joint for joint in model.chain if joint.child == side + '_tcp')
        if not np.allclose(tcp.origin[:3, 3], vectors['tcp_xyz'], atol=1e-9, rtol=0.):
            raise ValueError('显示模型的 TCP 与运行参数不一致，请更新 gp80.xacro')
        candidates = []
        for joint in root.findall('joint'):
            child = joint.find('child')
            if child is not None and child.get('link') == side + '_base_link':
                candidates.append(joint)
        if len(candidates) != 1:
            raise ValueError('显示模型缺少唯一基座安装关系：' + side)
        joint = candidates[0]
        parent, origin = joint.find('parent'), joint.find('origin')
        if joint.get('type') != 'fixed' or parent is None or parent.get('link') != 'world' or origin is None:
            raise ValueError('两臂基座必须固定连接到 world')
        actual = rigid_transform([float(v) for v in origin.get('xyz', '0 0 0').split()],
                                 [float(v) for v in origin.get('rpy', '0 0 0').split()])
        expected = rigid_transform(vectors[side + '_base_xyz'], vectors[side + '_base_rpy'])
        if not np.allclose(actual, expected, atol=1e-8, rtol=0.):
            raise ValueError(side + ' 显示基座与运行参数不一致，请修改 dual_arm.xacro 的安装原点')
        mount = next(j for j in model.chain if j.child == side + '_gp80_base_link')
        expected_mount = rigid_transform(vectors['mount_xyz'], vectors['mount_rpy'])
        if not np.allclose(mount.origin, expected_mount, atol=1e-8, rtol=0.):
            raise ValueError(side + ' 夹爪安装方向与运行参数不一致')
    return result.stdout


def _start(context) -> list[Action]:
    config = Path(LaunchConfiguration('params_file').perform(context)).expanduser().resolve()
    rviz_path = Path(LaunchConfiguration('rviz_config').perform(context)).expanduser().resolve()
    if not rviz_path.is_file():
        raise FileNotFoundError('找不到 RViz 配置：' + str(rviz_path))
    with config.open(encoding='utf-8') as stream:
        raw: object = yaml.safe_load(stream)
    root = _mapping(raw, '配置文件')
    scene = _mapping(_mapping(root.get('dual_arm_scene'), 'dual_arm_scene').get('ros__parameters'), '场景参数')
    task = _mapping(_mapping(root.get('dual_arm_task'), 'dual_arm_task').get('ros__parameters'), '任务参数')
    for name in ('left_base_xyz', 'right_base_xyz', 'left_base_rpy', 'right_base_rpy',
                 'tcp_xyz', 'mount_xyz', 'mount_rpy', 'left_urdf', 'right_urdf'):
        if name not in scene or name not in task or scene[name] != task[name]:
            raise ValueError('两节点配置缺失或不一致：' + name)
    if scene.get('use_sim_time') is not False or task.get('use_sim_time') is not False:
        raise ValueError('显示入口要求与仿真一致的系统时间')
    model_source = Path(get_package_share_directory('archer_description')) / 'urdf/dual_arm.xacro'
    description = build_description(model_source, scene)
    publisher = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        name='dual_robot_state_publisher', output='screen',
        parameters=[{'robot_description': ParameterValue(description, value_type=str),
                     'publish_frequency': 50.0, 'use_sim_time': False}],
    )
    # 只汇总两侧实际反馈，不启用 GUI、默认关节位置或自主运动。
    merger = Node(
        package='joint_state_publisher', executable='joint_state_publisher',
        name='dual_joint_state_merger', output='screen',
        parameters=[{'source_list': ['/left/joint_states', '/right/joint_states'],
                     'rate': 50, 'publish_default_positions': False,
                     'publish_default_velocities': False, 'publish_default_efforts': False,
                     'use_sim_time': False}],
    )
    viewer = Node(package='rviz2', executable='rviz2', name='dual_arm_rviz',
                  arguments=['-d', str(rviz_path), '-f', 'world'], output='screen',
                  parameters=[{'use_sim_time': False}])
    # 退出这组显示进程，不影响另一个终端中的共享仿真和任务节点。
    handlers = [RegisterEventHandler(OnProcessExit(
        target_action=process,
        on_exit=[EmitEvent(event=Shutdown(reason='双臂显示进程已退出'))],
    )) for process in (publisher, merger, viewer)]
    return [*handlers, LogInfo(msg='显示模型检查通过，等待左右实际关节反馈及轨迹标记。'),
            publisher, merger, viewer]


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory('archer_bringup'))
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=str(share / 'config/dual_circle.yaml')),
        DeclareLaunchArgument('rviz_config', default_value=str(share / 'rviz/dual_arm.rviz')),
        OpaqueFunction(function=_start),
    ])