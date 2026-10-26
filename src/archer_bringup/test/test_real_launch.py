"""检查画圆真机launch只启动一份正式入口，不执行节点或连接设备。"""

import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
import pytest


def load():
    """仅加载launch定义，不执行启动动作。"""
    path = Path(__file__).resolve().parents[1] / 'launch/dual_circle_real.launch.py'
    spec = importlib.util.spec_from_file_location('real_circle_launch_test', path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(execute='true') -> LaunchContext:
    """任意任务名称用于检查参数传递，不依赖当前日期。"""
    result = LaunchContext()
    result.launch_configurations.update({
        'execute': execute, 'run_dir': '/tmp/archer_results/another circle',
        'prepared_dir': '', 'limits_file': '',
    })
    return result


def test_launch_requires_task_and_defaults_to_no_execution() -> None:
    """显式许可沿用旧入口约定，缺省不自动连接机械臂。"""
    defaults = {a.name: a.default_value for a in load().generate_launch_description().entities
                if isinstance(a, DeclareLaunchArgument)}
    assert perform_substitutions(LaunchContext(), defaults['run_dir']) == ''
    assert perform_substitutions(LaunchContext(), defaults['trajectory_file']) == ''
    assert perform_substitutions(LaunchContext(), defaults['execute']) == 'false'
    assert not any(isinstance(a, Node) for a in load()._start(context('false')))


def test_launch_starts_one_registered_runner() -> None:
    """不能额外启动hardware_position或缓存开发脚本。"""
    ctx = context()
    nodes = [a for a in load()._start(ctx) if isinstance(a, Node)]
    assert len(nodes) == 1
    node = nodes[0]
    assert node.node_package == 'archer_hardware'
    command = ' '.join(perform_substitutions(ctx, part) for part in node.cmd)
    assert 'hardware_circle' in command
    assert '--run-dir /tmp/archer_results/another circle' in command
    assert '--execute-authorized-full-circle' in command
    assert 'verify_circle116' not in command
    assert 'hardware_position' not in command


def test_explicit_paths_are_forwarded_without_shell_splitting() -> None:
    """带空格路径作为独立参数交给ROS入口，不拼接shell命令。"""
    ctx = context()
    ctx.launch_configurations.update({'prepared_dir': '/tmp/prepared folder',
                                      'limits_file': '/tmp/config file.yaml'})
    node = next(a for a in load()._start(ctx) if isinstance(a, Node))
    command = [perform_substitutions(ctx, p) if not isinstance(p, str) else p for p in node.cmd]
    assert '/tmp/prepared folder' in command
    assert '/tmp/config file.yaml' in command


def test_empty_task_is_rejected_before_node_creation() -> None:
    """空目录不允许退回某一旧任务。"""
    ctx = context()
    ctx.launch_configurations['run_dir'] = ' '
    with pytest.raises(ValueError, match='run_dir'):
        load()._start(ctx)


def test_one_trajectory_path_is_forwarded() -> None:
    """轨迹模式不需要另外填写原始目录或报告，不启动第二份驱动。"""
    ctx = context()
    ctx.launch_configurations.update({'run_dir': '', 'trajectory_file': '/tmp/new run.json'})
    node = next(a for a in load()._start(ctx) if isinstance(a, Node))
    command = [perform_substitutions(ctx, p) for p in node.cmd]
    assert '--trajectory-file' in command
    assert '/tmp/new run.json' in command
    assert '--run-dir' not in command
    assert '--prepared-dir' not in command


def test_conflicting_prepared_directory_is_rejected() -> None:
    """两种轨迹定位方式不能互相覆盖而误用另一份文件。"""
    ctx = context()
    ctx.launch_configurations.update({'trajectory_file': '/tmp/new.json',
                                      'prepared_dir': '/tmp/other'})
    with pytest.raises(ValueError, match='不能同时指定'):
        load()._start(ctx)
