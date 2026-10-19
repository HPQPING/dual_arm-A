"""验证抓取仿真取消规划和窗口线程清理，不连接实机。"""

from threading import Event, Thread
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest

from archer_control.dual_arm_trajectory import DualArmKinematics
from archer_control.grasp_ros_node import GraspSimulationNode
from archer_control.simple_grasp_planner import PlanningCancelled, prepare_simple_motion
from archer_mujoco.grasp_scene import GraspConfig


def test_cancel_before_model_access() -> None:
    """已取消的任务不再读取模型或开始逆解。"""
    with pytest.raises(PlanningCancelled):
        prepare_simple_motion(cast(DualArmKinematics, None), {}, GraspConfig(),
                              cancelled=lambda: True)


def test_cancel_between_ik_candidates() -> None:
    """逆解搜索收到取消后不继续其余候选。"""
    import numpy as np

    cancel = Event()

    def solve(*args: object) -> SimpleNamespace:
        cancel.set()
        return SimpleNamespace(success=False)

    arm = SimpleNamespace(limits=np.array([[-2., 2.]] * 6), solve_ik=Mock(side_effect=solve))
    model = SimpleNamespace(left=arm, right=arm, world_from_left=np.eye(4),
                            world_from_right=np.eye(4))
    current = {side: np.zeros(6) for side in ('left', 'right')}
    with pytest.raises(PlanningCancelled):
        prepare_simple_motion(cast(DualArmKinematics, model), current, GraspConfig(),
                              cancelled=cancel.is_set)
    assert arm.solve_ik.call_count == 1


def test_close_cancels_before_join_and_is_idempotent() -> None:
    """先通知规划取消，再等待线程，重复关闭不重复清理。"""
    cancel = Event()
    calls: list[str] = []

    def shutdown(**kwargs: object) -> None:
        assert cancel.is_set()
        assert kwargs == {'wait': True, 'cancel_futures': True}
        calls.append('worker')

    node = cast(GraspSimulationNode, SimpleNamespace(
        closing=False, running=True, planning_cancel=cancel,
        timer=SimpleNamespace(cancel=lambda: calls.append('timer')),
        worker=SimpleNamespace(shutdown=shutdown), recorder=None,
        close_viewer=lambda: calls.append('viewer')))
    GraspSimulationNode.close(node)
    GraspSimulationNode.close(node)
    assert calls == ['timer', 'worker', 'viewer']
    assert not node.running


def test_viewer_thread_finishes_before_close_returns() -> None:
    """等待本次查看器线程完成清理后才允许解释器继续退出。"""
    requested, cleaned = Event(), Event()

    def render() -> None:
        requested.wait(timeout=5.)
        cleaned.set()

    thread = Thread(target=render, daemon=True)
    thread.start()
    node = cast(GraspSimulationNode, SimpleNamespace(
        resources=SimpleNamespace(close=requested.set), viewer_threads=(thread,), viewer=object()))
    GraspSimulationNode.close_viewer(node)
    assert cleaned.is_set()
    assert not thread.is_alive()
    assert node.viewer is None
    assert node.viewer_threads == ()
