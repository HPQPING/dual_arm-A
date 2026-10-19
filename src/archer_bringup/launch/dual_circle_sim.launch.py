"""启动画圆仿真和可选只读记录节点；完整抓取使用独立启动文件."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.action import Action
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
import yaml


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise ValueError(label + " 必须是字符串键组成的 YAML 映射")
    return {str(key): item for key, item in value.items()}


def _vector(value: object, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(label + " 必须包含三个数值")
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item):
            raise ValueError(label + " 必须包含三个有限数值")
        result.append(float(item))
    return result


def validate_parameters(path: Path, *, record: bool = False) -> str:
    """核对画圆配置及可选记录配置，拒绝旧空载协同模式."""
    with path.open(encoding="utf-8") as stream:
        raw: object = yaml.safe_load(stream)
    root = _mapping(raw, "配置文件")
    names = ("dual_arm_scene", "dual_arm_task")
    parameters: list[dict[str, object]] = []
    for name in names:
        node = _mapping(root.get(name), name)
        values = _mapping(node.get("ros__parameters"), name + "/ros__parameters")
        if values.get("use_sim_time") is not False:
            raise ValueError(name + " 必须显式设置 use_sim_time: false")
        parameters.append(values)
    scene, task = parameters
    for name in (
        "left_base_xyz",
        "right_base_xyz",
        "left_base_rpy",
        "right_base_rpy",
        "tcp_xyz",
        "mount_xyz",
        "mount_rpy",
    ):
        a = _vector(scene.get(name), "dual_arm_scene/" + name)
        b = _vector(task.get(name), "dual_arm_task/" + name)
        if any(not math.isclose(x, y, rel_tol=0.0, abs_tol=1e-9) for x, y in zip(a, b)):
            raise ValueError("两节点参数不一致：" + name)
    for name in ("clearance", "speed_limits", "left_urdf", "right_urdf"):
        if name not in scene or name not in task or scene[name] != task[name]:
            raise ValueError("两节点参数缺失或不一致：" + name)
    task_type = task.get("task_type")
    if not isinstance(task_type, str) or task_type != "split_circle":
        raise ValueError("此入口仅支持画圆 split_circle；抓取请使用 dual_grasp_sim.launch.py")
    if record:
        node = _mapping(root.get("dual_trace"), "dual_trace；启用记录时请使用新版参数文件")
        trace = _mapping(node.get("ros__parameters"), "dual_trace/ros__parameters")
        if trace.get("use_sim_time") is not False:
            raise ValueError("dual_trace 必须显式设置 use_sim_time: false")
        output = trace.get("output_dir")
        if not isinstance(output, str) or not output.strip():
            raise ValueError("dual_trace/output_dir 必须是非空目录路径")
        skew = trace.get("max_pair_skew_s")
        if (
            isinstance(skew, bool)
            or not isinstance(skew, (float, int))
            or not math.isfinite(skew)
            or not 0.0 <= skew <= 0.05
        ):
            raise ValueError("dual_trace/max_pair_skew_s 必须是 0～0.05 s 的有限数值")
    return task_type


def _start(context) -> list[Action]:
    config = Path(LaunchConfiguration("params_file").perform(context)).expanduser().resolve()
    python = Path(LaunchConfiguration("python_executable").perform(context)).expanduser()
    viewer = LaunchConfiguration("show_viewer").perform(context).lower()
    recording = LaunchConfiguration("record").perform(context).lower()
    # 不展示任务选择；旧协同参数明确拒绝，不能被忽略后执行画圆。
    mode = context.launch_configurations.get("task_type", "split_circle")
    if mode != "split_circle":
        raise ValueError("画圆入口不接受旧协同模式；抓取请使用 dual_grasp_sim.launch.py")
    if not config.is_file():
        raise FileNotFoundError("找不到参数文件：" + str(config))
    if not python.is_absolute() or not python.is_file():
        raise FileNotFoundError("python_executable 必须是存在的解释器绝对路径")
    if viewer not in ("true", "false"):
        raise ValueError("show_viewer 只能是 true 或 false")
    if recording not in ("true", "false"):
        raise ValueError("record 只能是 true 或 false")
    record = recording == "true"
    validate_parameters(config, record=record)
    for package in ("archer_description", "archer_mujoco", "archer_control"):
        get_package_share_directory(package)

    # 使用模块入口，不要求本轮额外修改两个 Python 包的 setup.py。
    scene = ExecuteProcess(
        cmd=[
            str(python),
            "-m",
            "archer_mujoco.dual_sim_node",
            "--ros-args",
            "--params-file",
            str(config),
            "-p",
            "show_viewer:=" + viewer,
        ],
        name="archer_shared_sim",
        output="screen",
        shell=False,
    )
    task_command = [
        str(python),
        "-m",
        "archer_control.dual_arm_node",
        "--ros-args",
        "--params-file",
        str(config),
    ]
    task_command += ["-p", "task_type:=split_circle"]
    task = ExecuteProcess(cmd=task_command, name="archer_dual_task", output="screen", shell=False)
    # 场景或任务退出时关闭本组全部进程，记录节点随组退出并保存已有数据。
    scene_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=scene,
            on_exit=[EmitEvent(event=Shutdown(reason="共享仿真场景已退出"))],
        )
    )
    task_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=task,
            on_exit=[EmitEvent(event=Shutdown(reason="双臂任务节点已退出"))],
        )
    )
    actions: list[Action] = [
        scene_exit,
        task_exit,
        LogInfo(msg="本次任务：画圆；完整抓取请使用 dual_grasp_sim.launch.py。"),
        LogInfo(msg="参数一致性检查通过；等待任务 READY 后，手动调用 /dual/task/start。"),
    ]
    if record:
        # 与任务使用同一文件；任务说明话题决定实际记录模式，避免猜测 YAML 中的默认任务。
        # JSON 字符串也是合法 YAML 标量；逐项传参且关闭 shell，保留路径中的空格等字符。
        recorder = ExecuteProcess(
            cmd=[
                str(python),
                "-m",
                "archer_control.dual_trace_node",
                "--ros-args",
                "--params-file",
                str(config),
                "-p",
                "config_file:=" + json.dumps(str(config), ensure_ascii=False),
            ],
            name="archer_dual_trace",
            output="screen",
            shell=False,
        )
        # 记录为只读观察；其退出只报告记录中断，不向任务或机械臂发送命令。
        actions.append(
            RegisterEventHandler(
                OnProcessExit(
                    target_action=recorder,
                    on_exit=[LogInfo(msg="记录节点已退出；请检查其日志和本次结果文件。")],
                )
            )
        )
        actions.append(LogInfo(msg="已启用记录；请勿再单独启动第二个 dual_trace_node。"))
        actions.extend([scene, recorder, task])
    else:
        actions.append(LogInfo(msg="本次不启动记录节点；可使用 record:=true 启用。"))
        actions.extend([scene, task])
    return actions


def generate_launch_description() -> LaunchDescription:
    default_config = (
        Path(get_package_share_directory("archer_bringup")) / "config" / "dual_circle.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=str(default_config),
                description="共享场景与双臂任务的统一参数文件",
            ),
            DeclareLaunchArgument(
                "show_viewer",
                default_value="true",
                choices=["true", "false"],
                description="是否显示 MuJoCo 窗口",
            ),
            DeclareLaunchArgument(
                "python_executable",
                default_value="/usr/bin/python3",
                description="同时能导入 ROS 2 和 MuJoCo 的 Python 解释器",
            ),
            DeclareLaunchArgument(
                "record",
                default_value="true",
                choices=["true", "false"],
                description="同时启动只读记录节点；已手动运行记录节点时保持 false",
            ),
            OpaqueFunction(function=_start),
        ]
    )
