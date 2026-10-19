"""验证画圆记录只写完整反馈对，不删改已经写入的历史记录."""

import csv
import io
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from archer_control.dual_trace_node import DualTrace, TaskProfile
from archer_control.trajectory import joint_names


def recorder() -> DualTrace:
    """仅替换时钟、FK和输出流，实际调用正式记录与完成回调."""
    times = np.array([0., 1., 2.], dtype=np.float64)
    identity = np.eye(4, dtype=np.float64)
    instance = SimpleNamespace(
        stream=io.StringIO(), start_ns=0, max_pair_skew=.005,
        plans={s: (times, np.zeros((3, 6))) for s in ('left', 'right')},
        feedback={s: (100000000, np.zeros(6)) for s in ('left', 'right')},
        profile=TaskProfile('split_circle', 0, 1., times, (), identity, identity, 1.5),
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=100000000)),
        last_saved={}, counts={'left': 0, 'right': 0},
        models={s: SimpleNamespace(
            forward=lambda q: identity, limits=np.array([[-10., 10.]] * 6)
        ) for s in ('left', 'right')},
        world={s: identity for s in ('left', 'right')},
        maximum_object={'left': 0., 'right': 0.},
        maximum_circle={'left': 0., 'right': 0.},
        maximum_tracking={'left': 0., 'right': 0.},
        arc_counts={'left': 0, 'right': 0}, traces={'left': [], 'right': []},
        center=np.zeros(3), radius=.2, record_relative=Mock(), finish=Mock(),
    )
    instance.record = lambda: DualTrace.record(cast(DualTrace, instance))
    return cast(DualTrace, instance)


def rows(trace: DualTrace) -> list[list[str]]:
    """读取测试流，不创建任务结果文件."""
    assert isinstance(trace.stream, io.StringIO)
    return list(csv.reader(io.StringIO(trace.stream.getvalue())))


def test_pair_is_written_together_and_not_duplicated():
    """完整新鲜帧写两行，重复采样不重复写入."""
    trace = recorder()
    trace.record()
    assert [row[2] for row in rows(trace)] == ['left', 'right']
    assert trace.counts == {'left': 1, 'right': 1}
    trace.record()
    assert len(rows(trace)) == 2


def test_missing_or_old_side_does_not_write_orphan():
    """缺一侧，或一侧已记录而另一侧更新时，都不写单侧行."""
    trace = recorder()
    right = trace.feedback.pop('right')
    trace.record()
    assert rows(trace) == []
    trace.feedback['right'] = right
    trace.last_saved['right'] = right[0]
    trace.record()
    assert rows(trace) == []
    assert trace.last_saved == {'right': right[0]}


def test_bad_skew_stale_future_and_before_start_are_rejected():
    """时间差、过期、未来和任务开始前的帧不能被当成一对."""
    for left, right in ((100000000, 90000000), (-500000001, 100000000),
                        (100000001, 100000000), (-1, -1)):
        trace = recorder()
        trace.feedback['left'] = (left, np.zeros(6))
        trace.feedback['right'] = (right, np.zeros(6))
        trace.record()
        assert rows(trace) == []
        assert trace.counts == {'left': 0, 'right': 0}


def test_valid_pair_preserves_actual_angles_and_original_stamps():
    """记录配对不平均或伪造实际角度，也不改写各侧原始时间戳."""
    trace = recorder()
    trace.feedback['left'] = (99999000, np.arange(6, dtype=np.float64) * .001)
    trace.feedback['right'] = (100000000, -np.arange(6, dtype=np.float64) * .002)
    trace.record()
    actual = rows(trace)
    assert [int(row[0]) for row in actual] == [99999000, 100000000]
    for side, row in zip(('left', 'right'), actual):
        np.testing.assert_array_equal([float(x) for x in row[4:10]], trace.feedback[side][1])


def test_completion_does_not_append_unpaired_tail():
    """完成回调最后再采样，也不会追加只有一侧更新的尾帧."""
    trace = recorder()
    trace.last_saved = {'right': trace.feedback['right'][0]}
    DualTrace.on_status(trace, String(data='COMPLETED'))
    assert rows(trace) == []
    assert trace.task_state == 'COMPLETED'
    cast(Mock, trace.finish).assert_called_once_with('COMPLETED')


def test_second_feedback_callback_records_pair_without_timer():
    """两侧回调到齐即可成对写入，不依赖定时器与回调的相对顺序."""
    trace = recorder()
    trace.feedback.clear()
    for side in ('left', 'right'):
        message = JointState(name=joint_names(side), position=[0.] * 6)
        message.header.stamp.nanosec = 100000000
        DualTrace.on_feedback(trace, side, message)
        assert len(rows(trace)) == (0 if side == 'left' else 2)
    trace.record()
    assert len(rows(trace)) == 2
