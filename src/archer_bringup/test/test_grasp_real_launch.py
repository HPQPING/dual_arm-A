"""检查协同真机launch的显式许可和唯一调度节点，不执行任何节点。"""

import importlib.util
from pathlib import Path
from unittest import TestCase

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.utilities import perform_substitutions
from launch_ros.actions import Node


def load():
    path = Path(__file__).resolve().parents[1] / 'launch/dual_grasp_real.launch.py'
    spec = importlib.util.spec_from_file_location('real_grasp_launch_test', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(execute='true'):
    result = LaunchContext()
    result.launch_configurations.update({
        'execute': execute, 'trajectory_file': '/tmp/task with spaces/state.json',
        'run_dir': '', 'limits_file': '',
    })
    return result


def test_default_launch_does_not_create_control_node():
    defaults = {a.name: a.default_value for a in load().generate_launch_description().entities
                if isinstance(a, DeclareLaunchArgument)}
    assert perform_substitutions(LaunchContext(), defaults['execute']) == 'false'
    assert not any(isinstance(action, Node) for action in load()._start(context('false')))


def test_one_runner_and_one_unsplit_json_path():
    ctx = context()
    nodes = [action for action in load()._start(ctx) if isinstance(action, Node)]
    assert len(nodes) == 1
    command = [perform_substitutions(ctx, p) for p in nodes[0].cmd]
    assert '/tmp/task with spaces/state.json' in command
    assert '--execute-authorized-full-grasp' in command
    assert any('hardware_grasp' in p for p in command)
    assert not any('hardware_position' in p for p in command)


def test_missing_trajectory_rejected_before_node_creation():
    ctx = context()
    ctx.launch_configurations['trajectory_file'] = ' '
    with TestCase().assertRaisesRegex(ValueError, 'state.json'):
        load()._start(ctx)


def test_simulation_passes_circle_initial_pose_even_without_rviz():
    """关闭RViz不应退回另一套协同初值；实际启动命令携带两侧六轴参数。"""
    path = Path(__file__).resolve().parents[1] / 'launch/dual_grasp_sim.launch.py'
    spec = importlib.util.spec_from_file_location('sim_grasp_launch_test', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = path.parents[1] / 'config/dual_circle.yaml'
    task_context = LaunchContext()
    task_context.launch_configurations.update({
        'output_file': '', 'python_executable': '/usr/bin/python3',
        'show_viewer': 'false', 'show_rviz': 'false', 'rviz_params_file': str(config)})
    actions = module._start(task_context)
    process = next(action for action in actions if isinstance(action, ExecuteProcess))
    command = [perform_substitutions(task_context, item) for item in process.cmd]
    parameters = module.initial_joint_parameters(config)
    assert command[-len(parameters):] == parameters
    assert any(item.startswith('left_initial_joints:=') for item in command)
    assert any(item.startswith('right_initial_joints:=') for item in command)


def test_no_default_trajectory_file():
    """与画圆一致：每次显式指定JSON地址，数据只在结果转换目录管理。"""
    defaults = {a.name: a.default_value for a in load().generate_launch_description().entities
                if isinstance(a, DeclareLaunchArgument)}
    assert perform_substitutions(LaunchContext(), defaults['trajectory_file']) == ''
