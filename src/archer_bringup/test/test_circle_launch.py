"""验证两个任务的启动入口及旧名称兼容，不启动节点或连接设备."""

import copy
import importlib.util
from pathlib import Path

from launch import LaunchContext
from launch.utilities import perform_substitutions
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load(name: str):
    """仅载入启动描述源码，不执行启动动作."""
    spec = importlib.util.spec_from_file_location(
        "test_" + name.replace(".", "_"), ROOT / "launch" / name
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def context(path: Path) -> LaunchContext:
    """使用当前解释器路径和无窗口配置构造测试上下文."""
    value = LaunchContext()
    value.launch_configurations.update(
        {
            "params_file": str(path),
            "show_viewer": "false",
            "python_executable": "/usr/bin/python3",
            "record": "true",
            "show_rviz": "false",
        }
    )
    return value


def test_circle_has_no_task_selection() -> None:
    """新画圆入口只暴露场景参数，任务命令明确固定为画圆."""
    module = load("dual_circle_sim.launch.py")
    description = module.generate_launch_description()
    names = {a.name for a in description.entities if isinstance(a, DeclareLaunchArgument)}
    assert "task_type" not in names
    ctx = context(ROOT / "config/dual_circle.yaml")
    actions = module._start(ctx)
    processes = [p for p in actions if isinstance(p, ExecuteProcess)]
    assert len(processes) == 3
    commands = [" ".join((part if isinstance(part, str) else perform_substitutions(ctx, part)) for part in p.cmd) for p in processes]
    assert any("task_type:=split_circle" in c for c in commands)
    assert not any("cooperative" in c or "grasp_ros_node" in c for c in commands)


def test_old_launcher_is_removed() -> None:
    """旧入口从源目录移除，只有明确的画圆与抓取任务名称."""
    assert not (ROOT / 'launch/dual_arm_sim.launch.py').exists()



def test_old_cooperative_argument_is_explicitly_rejected() -> None:
    """旧协同参数不能被忽略后悄悄执行画圆."""
    value = context(ROOT / "config/dual_circle.yaml")
    value.launch_configurations["task_type"] = "cooperative"
    with pytest.raises(ValueError, match="dual_grasp_sim"):
        load("dual_circle_sim.launch.py")._start(value)


def test_cooperative_yaml_is_rejected(tmp_path: Path) -> None:
    """配置文件同样不能将画圆入口变回旧协同示意."""
    data = yaml.safe_load((ROOT / "config/dual_circle.yaml").read_text())
    data["dual_arm_task"]["ros__parameters"]["task_type"] = "cooperative"
    target = tmp_path / "wrong.yaml"
    target.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="dual_grasp_sim"):
        load("dual_circle_sim.launch.py").validate_parameters(target)


def test_geometry_mismatch_remains_rejected(tmp_path: Path) -> None:
    """入口整理不能绕过已有安装参数一致性检查."""
    original = yaml.safe_load((ROOT / "config/dual_circle.yaml").read_text())
    data = copy.deepcopy(original)
    data["dual_arm_task"]["ros__parameters"]["tcp_xyz"][2] += 0.01
    target = tmp_path / "wrong_tcp.yaml"
    target.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="tcp_xyz"):
        load("dual_circle_sim.launch.py").validate_parameters(target)


def test_grasp_stays_a_separate_entry() -> None:
    """完整抓取入口仍独立存在，不引入画圆任务选择."""
    description = load("dual_grasp_sim.launch.py").generate_launch_description()
    names = {a.name for a in description.entities if isinstance(a, DeclareLaunchArgument)}
    assert "task_type" not in names
    assert "show_viewer" in names


def test_both_tasks_default_to_shared_rviz() -> None:
    """两个任务默认开启MuJoCo和公共RViz，画圆默认记录保持开启."""
    for name in ('dual_circle_sim.launch.py', 'dual_grasp_sim.launch.py'):
        description = load(name).generate_launch_description()
        defaults = {}
        for argument in description.entities:
            if isinstance(argument, DeclareLaunchArgument):
                value = argument.default_value
                if value is not None:
                    defaults[argument.name] = perform_substitutions(LaunchContext(), value)
        assert defaults['show_rviz'] == 'true'
        assert defaults['show_viewer'] == 'true'
        if 'circle' in name:
            assert defaults['record'] == 'true'
    assert (ROOT / 'launch/dual_arm_view.launch.py').is_file()


def test_circle_includes_shared_display_once_when_enabled() -> None:
    """显示开关只增加一份公共显示组件，不增加场景、任务或记录进程."""
    module = load('dual_circle_sim.launch.py')
    ctx = context(ROOT / 'config/dual_circle.yaml')
    disabled = module._start(ctx)
    assert not any(isinstance(a, IncludeLaunchDescription) for a in disabled)
    ctx.launch_configurations['show_rviz'] = 'true'
    enabled = module._start(ctx)
    assert len([a for a in enabled if isinstance(a, IncludeLaunchDescription)]) == 1
    assert len([a for a in enabled if isinstance(a, ExecuteProcess)]) == 3
