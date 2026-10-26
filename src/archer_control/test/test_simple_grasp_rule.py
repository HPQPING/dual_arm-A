"""检查简化协同的固定等待、开口保持及归位前闭合，不连接设备."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np

from archer_control.cooperative_planner import CooperativePlan
from archer_control.dual_arm_trajectory import DualArmPlan
from archer_control.grasp_record_format import (
    GRASP_WAIT_S, GRIPPER_CLOSED_COMMAND, GRIPPER_OPEN_COMMAND,
)
from archer_control.simple_grasp_planner import SimpleGraspMotion
from archer_control.simple_grasp_scene import SimpleGraspDemo
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


def test_wait_does_not_advance_early_or_close_gripper() -> None:
    """等待期间保持 5.5，满五秒才进入进给，不能提前附着或追加夹紧."""
    demo = simulated_stage("GRASP_WAIT")
    for _ in range(49):
        SimpleGraspDemo.step(demo)
        assert demo.state == "GRASP_WAIT"
        assert not demo.attached
        assert demo.gripper_command == GRIPPER_OPEN_COMMAND
    SimpleGraspDemo.step(demo)
    assert demo.state == "INSERT"
    assert demo.grasp_wait_elapsed == GRASP_WAIT_S
    assert not demo.attached


def test_open_and_close_happen_only_at_designated_boundaries() -> None:
    """到位开口、归位前闭合都平滑变化，搬运不引入其他夹爪目标."""
    demo = simulated_stage("OPEN_FOR_GRASP")
    commands = []
    for _ in range(11):
        SimpleGraspDemo.step(demo)
        commands.append(demo.gripper_command)
    assert demo.state == "GRASP_WAIT"
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
