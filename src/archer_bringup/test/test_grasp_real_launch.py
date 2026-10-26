"""检查协同真机launch的显式许可和唯一调度节点，不执行任何节点。"""

import importlib.util
from pathlib import Path
from unittest import TestCase

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
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
