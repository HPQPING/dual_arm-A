"""统一启动共享 MuJoCo 与画圆或空载协同任务，不自动触发运动。"""

from __future__ import annotations

import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.action import Action
from launch.actions import (
    DeclareLaunchArgument, EmitEvent, ExecuteProcess, LogInfo,
    OpaqueFunction, RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
import yaml


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(label + ' 必须是字符串键组成的 YAML 映射')
    return {str(key): item for key, item in value.items()}


def _vector(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(label + ' 必须包含三个数值')
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(label + ' 必须包含三个有限数值')
        result.append(float(item))
    return result


def validate_parameters(path: Path) -> None:
    """启动前核对两节点的公共配置；详细运动学和轨迹检查仍由任务节点负责。"""
    with path.open(encoding='utf-8') as stream:
        raw: object = yaml.safe_load(stream)
    root = _mapping(raw, '配置文件')
    names = ('dual_arm_scene', 'dual_arm_task')
    parameters: list[dict[str, object]] = []
    for name in names:
        node = _mapping(root.get(name), name)
        values = _mapping(node.get('ros__parameters'), name + '/ros__parameters')
        if values.get('use_sim_time') is not False:
            raise ValueError(name + ' 必须显式设置 use_sim_time: false')
        parameters.append(values)
    scene, task = parameters
    for name in ('left_base_xyz', 'right_base_xyz', 'left_base_rpy', 'right_base_rpy',
                 'tcp_xyz', 'mount_xyz', 'mount_rpy'):
        a = _vector(scene.get(name), 'dual_arm_scene/' + name)
        b = _vector(task.get(name), 'dual_arm_task/' + name)
        if any(not math.isclose(x, y, rel_tol=0., abs_tol=1e-9) for x, y in zip(a, b)):
            raise ValueError('两节点参数不一致：' + name)
    for name in ('clearance', 'speed_limits', 'left_urdf', 'right_urdf'):
        if name not in scene or name not in task or scene[name] != task[name]:
            raise ValueError('两节点参数缺失或不一致：' + name)
    if task.get('task_type') not in ('split_circle', 'cooperative'):
        raise ValueError('task_type 必须为 split_circle 或 cooperative')


def _start(context) -> list[Action]:
    config = Path(LaunchConfiguration('params_file').perform(context)).expanduser().resolve()
    python = Path(LaunchConfiguration('python_executable').perform(context)).expanduser()
    viewer = LaunchConfiguration('show_viewer').perform(context).lower()
    mode = LaunchConfiguration('task_type').perform(context)
    if mode not in ('configured', 'split_circle', 'cooperative'):
        raise ValueError('task_type 启动参数无效')
    if not config.is_file():
        raise FileNotFoundError('找不到参数文件：' + str(config))
    if not python.is_absolute() or not python.is_file():
        raise FileNotFoundError('python_executable 必须是存在的解释器绝对路径')
    if viewer not in ('true', 'false'):
        raise ValueError('show_viewer 只能是 true 或 false')
    validate_parameters(config)
    for package in ('archer_description', 'archer_mujoco', 'archer_control'):
        get_package_share_directory(package)

    # 使用模块入口，不要求本轮额外修改两个 Python 包的 setup.py。
    scene = ExecuteProcess(
        cmd=[str(python), '-m', 'archer_mujoco.dual_sim_node', '--ros-args',
             '--params-file', str(config), '-p', 'show_viewer:=' + viewer],
        name='archer_shared_sim', output='screen', shell=False,
    )
    task_command = [str(python), '-m', 'archer_control.dual_arm_node', '--ros-args',
                    '--params-file', str(config)]
    if mode != 'configured':
        task_command += ['-p', 'task_type:=' + mode]
    task = ExecuteProcess(cmd=task_command, name='archer_dual_task', output='screen', shell=False)
    # 任一进程退出时关闭整组，避免只剩一侧控制流程继续运行。
    scene_exit = RegisterEventHandler(OnProcessExit(
        target_action=scene,
        on_exit=[EmitEvent(event=Shutdown(reason='共享仿真场景已退出'))],
    ))
    task_exit = RegisterEventHandler(OnProcessExit(
        target_action=task,
        on_exit=[EmitEvent(event=Shutdown(reason='双臂任务节点已退出'))],
    ))
    return [
        scene_exit, task_exit,
        LogInfo(msg='参数一致性检查通过；等待任务 READY 后，手动调用 /dual/task/start。'),
        scene, task,
    ]


def generate_launch_description() -> LaunchDescription:
    default_config = (Path(get_package_share_directory('archer_bringup'))
                      / 'config' / 'dual_circle.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=str(default_config),
                              description='共享场景与双臂任务的统一参数文件'),
        DeclareLaunchArgument('show_viewer', default_value='true', choices=['true', 'false'],
                              description='是否显示 MuJoCo 窗口'),
        DeclareLaunchArgument('python_executable', default_value='/usr/bin/python3',
                              description='同时能导入 ROS 2 和 MuJoCo 的 Python 解释器'),
        DeclareLaunchArgument('task_type', default_value='configured',
                              choices=['configured', 'split_circle', 'cooperative'],
                              description='沿用配置文件或显式选择画圆、协同移动任务'),
        OpaqueFunction(function=_start),
    ])