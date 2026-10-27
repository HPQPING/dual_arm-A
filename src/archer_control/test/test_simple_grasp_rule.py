"""检查协同开爪后直接进给、共享初值及归位前闭合，不连接设备。"""

from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
import yaml

from archer_control.cooperative_planner import CooperativePlan
from archer_control.dual_arm_trajectory import DualArmPlan
from archer_control.grasp_record_format import (
    GRIPPER_CLOSED_COMMAND, GRIPPER_OPEN_COMMAND, SIMPLE_GRASP_STAGES,
)
from archer_control.simple_grasp_planner import SimpleGraspMotion
from archer_control.simple_grasp_scene import SIMULATION_INITIAL, SimpleGraspDemo
from archer_control.trajectory import JointPlan
from archer_mujoco.dual_model_loader import LoadedDualArm


def simulated_stage(stage: str) -> SimpleGraspDemo:
    """仅替换物理步进；执行生产状态机，避免真实时间等待和运动."""
    pair = DualArmPlan(
        JointPlan(tuple(f"left_joint_{i}" for i in range(1, 7)),
                  np.array([0.0, 1.0]), np.zeros((2, 6))),
        JointPlan(tuple(f"right_joint_{i}" for i in range(1, 7)),
                  np.array([0.0, 1.0]), np.zeros((2, 6))),
    )
    data = SimpleNamespace(time=0.0, qpos=np.zeros(1), qvel=np.zeros(1))

    def advance() -> None:
        data.time = round(data.time + 0.1, 8)

    loaded = SimpleNamespace(
        data=data, set_arm_targets=Mock(), set_grippers=Mock(), step=advance,
    )
    demo = SimpleNamespace(
        loaded=loaded, state=stage, stage_started=0.0, finished=False, attached=False,
        motion=SimpleGraspMotion(
            {}, pair, cast(CooperativePlan, SimpleNamespace(paired=pair)),
            pair, pair, pair, pair, cast(LoadedDualArm, None),
        ),
        grasp_wait_elapsed=0.0, tcp_points=Mock(return_value=(np.zeros(3), np.zeros(3))),
    )

    def enter(next_stage: str) -> None:
        demo.state, demo.stage_started = next_stage, data.time

    demo.enter = enter
    return cast(SimpleGraspDemo, demo)


def test_opening_advances_directly_to_insertion_without_dwell() -> None:
    """开爪动作完成后直接进给，不新增等待或提前附着."""
    demo = simulated_stage("OPEN_FOR_GRASP")
    for _ in range(9):
        SimpleGraspDemo.step(demo)
        assert demo.state == "OPEN_FOR_GRASP"
        assert not demo.attached
    SimpleGraspDemo.step(demo)
    assert demo.state == "INSERT"
    assert "GRASP_WAIT" not in SIMPLE_GRASP_STAGES
    assert demo.gripper_command == GRIPPER_OPEN_COMMAND
    assert not demo.attached


def test_open_and_close_happen_only_at_designated_boundaries() -> None:
    """到位开口、归位前闭合都平滑变化，搬运不引入其他夹爪目标."""
    demo = simulated_stage("OPEN_FOR_GRASP")
    commands = []
    for _ in range(11):
        SimpleGraspDemo.step(demo)
        commands.append(demo.gripper_command)
    assert demo.state == "INSERT"
    assert commands[0] == GRIPPER_CLOSED_COMMAND
    assert commands[-1] == GRIPPER_OPEN_COMMAND
    assert all(a >= b for a, b in zip(commands, commands[1:]))
    demo = simulated_stage("CLOSE_BEFORE_HOME")
    commands = []
    for _ in range(11):
        SimpleGraspDemo.step(demo)
        commands.append(demo.gripper_command)
    assert demo.state == "RETURN_HOME"
    assert commands[0] == GRIPPER_OPEN_COMMAND
    assert commands[-1] == GRIPPER_CLOSED_COMMAND
    assert all(a <= b for a, b in zip(commands, commands[1:]))


def test_simulation_defaults_match_circle_configuration() -> None:
    """两项仿真采用相同12轴初值，修改任意一侧默认值会被检查发现。"""
    config = Path(__file__).resolve().parents[2] / 'archer_bringup/config/dual_circle.yaml'
    raw = yaml.safe_load(config.read_text(encoding='utf-8'))
    assert isinstance(raw, dict)
    parameters = raw['dual_arm_scene']['ros__parameters']
    for side in ('left', 'right'):
        assert list(SIMULATION_INITIAL[side]) == parameters[side + '_initial_joints']
