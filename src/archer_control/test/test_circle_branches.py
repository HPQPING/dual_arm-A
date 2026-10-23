"""验证画圆多逆解的整轮配对选择、去重与取消，不连接机械臂。"""

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock
from unittest import TestCase

import numpy as np

from archer_control.circle_planner import _select_arc_pair, _solve_arc_candidates
from archer_control.kinematics import ArmKinematics, IKResult


def arc(values):
    """只在J6上构造候选分支；测试不把它当作实际机器人逆解。"""
    q = np.zeros((len(values), 6))
    q[:, 5] = values
    return q


def choose(left, right):
    """使用相同双臂初值和配置比较完整周期。"""
    return _select_arc_pair(left, right, (np.zeros(6), np.zeros(6)),
                            np.ones(6), .01, 1., lambda: False)


def test_return_aware_selection_uses_complete_cycle():
    """起点同样方便的分支中选择圆弧与归位代价更低的方案。"""
    old, better = arc([.1, .2, .4]), arc([.1, .15, .2])
    left, _, report = choose([old, better], [arc([0., 0., 0.])])
    np.testing.assert_array_equal(left, better)
    assert report['selected'] == [1, 0]
    assert report['cost_is_execution_duration'] is False
    assert np.all(np.array(report['selected_phase_cost_s'])
                  <= np.array(report['baseline_phase_cost_s']) + 1e-9)


def test_faster_return_must_not_worsen_approach():
    """不能用更差的接近段换取归位少转一点，未找到改进则保留旧分支。"""
    old, tradeoff = arc([.1, .2, .4]), arc([.2, .1, .05])
    left, _, report = choose([old, tradeoff], [arc([0., 0., 0.])])
    np.testing.assert_array_equal(left, old)
    assert report['selected'] == [0, 0]


def test_pair_selection_includes_slower_peer():
    """两臂代价按共同段时长比较，而非将各臂独立最优简单相加。"""
    left = [arc([.1, .2, .4]), arc([.1, .15, .2])]
    right = [arc([.3, .5, .7]), arc([.3, .35, .4])]
    lpath, rpath, report = choose(left, right)
    np.testing.assert_array_equal(lpath, left[1])
    np.testing.assert_array_equal(rpath, right[1])
    assert report['feasible_branches'] == [2, 2]


def test_duplicate_ik_seeds_trace_only_one_arc():
    """多初值收敛到同一分支时不重复求解整条圆弧。"""
    def result(v):
        return IKResult(True, np.full(6, v), 0., 0., 1, 'test')

    solve = Mock(side_effect=[result(.1)] * 4 + [result(.15), result(.2)])
    model = cast(ArmKinematics, SimpleNamespace(
        limits=np.array([[-1., 1.]] * 6), names=['left_joint_1'], solve_ik=solve))
    paths = _solve_arc_candidates(model, np.eye(4), np.zeros(6), np.zeros((3, 3)),
                                  np.eye(3), np.random.default_rng(7), 3,
                                  lambda: False, np.ones(6), True)
    assert len(paths) == 1
    assert solve.call_count == 6


def test_selection_honors_cancellation():
    """比较组合期间响应取消，不返回半成品。"""
    with TestCase().assertRaisesRegex(ValueError, 'cancelled'):
        _select_arc_pair([arc([0., .1])], [arc([0., .1])],
                         (np.zeros(6), np.zeros(6)), np.ones(6), .01, 1., lambda: True)
