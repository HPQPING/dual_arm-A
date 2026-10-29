"""211版协同反馈控制：沿用206已验证的双J2修正、归位软边界和搬运双J3低通。"""

from __future__ import annotations
from collections import deque
import json
import math
import time
from typing import Any
from .gripper_adapter import CooperativePositionAdapter
from .position_plan import ContinuousPositionTrajectory, TrajectoryCancelled
from .sdk_cycle import PositionSample
from .position_feedback import BoundedPositionFeedback

RETURN_LIMIT_BAND_RAD = 1.25
RETURN_MAX_J2_SPEED_RAD_S = 0.16
RETURN_MAX_J2_ACCEL_RAD_S2 = 0.22
RETURN_CAP_TRANSITION_S = 1.0
MAX_OFFSET = 0.012
GAIN = 0.25
OMEGA = 1.5
RAMP_S = 2.0


class FeedbackCooperativeAdapter(CooperativePositionAdapter):
    """211版双J2反馈基类：沿用已验证参数，保持POSITION及原轨迹保护。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._feedback_correction = BoundedPositionFeedback(
            self.limits.lower,
            self.limits.upper,
            enabled_axes=(False, True, False, False, False, False),
            max_velocity=self.limits.max_velocity,
            max_acceleration=self.limits.max_acceleration,
        )
        self._feedback_gate: Any = None
        self._feedback_sample_at: float | None = None

    def _stream_position_sample(self) -> PositionSample | None:
        sample = super()._stream_position_sample()
        if sample is None or self._trajectory_side not in ('left', 'right'):
            return sample
        gate, move, frame = (self._trajectory_gate, self._move, self._feedback)
        if (
            gate is not None
            and getattr(gate, 'cancelled', False)
            or self.state in ('STOPPED', 'FAULT')
        ):
            raise TrajectoryCancelled('反馈候选收到共同取消，不再生成位置目标')
        if gate is None or not isinstance(move, ContinuousPositionTrajectory) or frame is None:
            raise RuntimeError('反馈候选缺少已验证轨迹或新鲜反馈')
        if gate is not self._feedback_gate:
            self._feedback_correction.reset()
            self._feedback_sample_at = None
            self._feedback_gate = gate
        now = time.perf_counter()
        dt = (
            1.0 / self.control_hz
            if self._feedback_sample_at is None
            else now - self._feedback_sample_at
        )
        self._feedback_sample_at = now
        value = gate.playback_status()['reference_elapsed_s']
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError('反馈候选参考时间无效')
        elapsed = float(value)
        program = self.gripper.program
        phases = {} if program is None else program['phase_start_ns']
        active = (
            bool(phases)
            and phases['OPEN_FOR_GRASP'] / 1000000000.0
            <= elapsed
            < phases['CLOSE_BEFORE_HOME'] / 1000000000.0
        )
        command = self._feedback_correction.step(
            sample.target,
            frame.positions,
            dt,
            enabled=active,
            reference_velocity=move.sample(elapsed, 1),
            reference_acceleration=move.sample(elapsed, 2),
        )
        self.limits.check_positions(command)
        return PositionSample(command, sample.acknowledge)

    def close(self) -> None:
        """关闭控制后打印两侧J2的末段摘要，确认修正实际发送，不干扰周期线程。"""
        rows = self.tracking_trace()
        side = self._trajectory_side
        super().close()
        if rows:
            offsets = [
                r['sent_rad'][1] - r['desired_rad'][1] for r in rows if r['sent_rad'] is not None
            ]
            last = rows[-1]
            result = {
                'side': side,
                'max_command_correction_rad': max(map(abs, offsets)) if offsets else None,
                'max_actual_tracking_error_rad': max(
                    (abs(r['actual_rad'][1] - r['desired_rad'][1]) for r in rows)
                ),
                'last_desired_rad': last['desired_rad'][1],
                'last_actual_rad': last['actual_rad'][1],
                'last_sent_rad': last['sent_rad'][1] if last['sent_rad'] else None,
                'last_device_target_rad': last['device_rad'][1],
            }
            print('候选J2末段反馈（已关闭控制）：' + json.dumps(result), flush=True)


def limit_weight(q: float, low: float, high: float) -> float:
    """五次C2窗口：修正在限位前渐消，保持真实关节范围不变。"""
    u = min(1.0, max(0.0, min(q - low, high - q) / RETURN_LIMIT_BAND_RAD))
    return u**3 * (10.0 + u * (-15.0 + 6.0 * u))


class ReturnFeedbackAdapter(FeedbackCooperativeAdapter):
    """复用原滤波状态与约束，归位前不改变采样/确认，末端保持最后有效指令。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._return_gate: Any = None
        self._return_last: tuple[float, ...] | None = None
        self._return_peak = 0.0
        self._return_filter: BoundedPositionFeedback | None = None

    def _stream_position_sample(self) -> PositionSample | None:
        gate, program = (self._trajectory_gate, self.gripper.program)
        if gate is None or program is None:
            return super()._stream_position_sample()
        start = program['phase_start_ns'].get('RETURN_HOME')
        value = gate.playback_status().get('reference_elapsed_s')
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or (not isinstance(value, (int, float)))
            or isinstance(value, bool)
            or (value < start / 1000000000.0)
        ):
            return super()._stream_position_sample()
        if getattr(gate, 'cancelled', False) or self.state in ('STOPPED', 'FAULT'):
            raise TrajectoryCancelled('归位已取消，不再生成修正指令')
        if self.state == 'HOLDING':
            return self._hold_return_command(gate)
        sample = CooperativePositionAdapter._stream_position_sample(self)
        if sample is None:
            return None
        move, frame = (self._move, self._feedback)
        if not isinstance(move, ContinuousPositionTrajectory) or frame is None:
            raise RuntimeError('归位缺少已验证轨迹或实际反馈')
        if getattr(gate, 'cancelled', False) or self.state in ('STOPPED', 'FAULT'):
            raise TrajectoryCancelled('归位取样期间取消，不再发送旧指令')
        if gate is not self._feedback_gate:
            self._feedback_correction.reset()
            self._feedback_sample_at = None
            self._feedback_gate = gate
        if gate is not self._return_gate:
            self._return_gate, self._return_last, self._return_peak = (gate, None, 0.0)
            original = self._feedback_correction
            self._return_filter = BoundedPositionFeedback(
                (-1.0,) * 6,
                (1.0,) * 6,
                enabled_axes=original.axes,
                max_velocity=self.limits.max_velocity,
                max_acceleration=self.limits.max_acceleration,
            )
            for attr in (
                'gain',
                'max_offset',
                'max_rate',
                'max_acceleration',
                'max_jerk',
                'omega',
            ):
                setattr(self._return_filter, attr, getattr(original, attr))
            for attr in ('offset', 'velocity', 'acceleration'):
                setattr(self._return_filter, attr, list(getattr(original, attr)))
            for i, enabled in enumerate(original.axes):
                if (
                    enabled
                    and limit_weight(sample.target[i], self.limits.lower[i], self.limits.upper[i])
                    < 1.0
                    and (abs(original.offset[i]) > 1e-09)
                ):
                    raise RuntimeError('归位起点在软边界内且已有修正，不能保证连续接入')
        now = time.perf_counter()
        dt = (
            1.0 / self.control_hz
            if self._feedback_sample_at is None
            else now - self._feedback_sample_at
        )
        self._feedback_sample_at = now
        elapsed = gate.playback_status()['reference_elapsed_s']
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise RuntimeError('归位参考时间无效')
        assert self._return_filter is not None
        u = min(1.0, max(0.0, (float(elapsed) - start / 1000000000.0) / RETURN_CAP_TRANSITION_S))
        blend = u**3 * (10.0 + u * (-15.0 + 6.0 * u))
        self._return_filter.max_rate = self._feedback_correction.max_rate * (1.0 - 0.5 * blend)
        self._return_filter.max_acceleration = self._feedback_correction.max_acceleration * (
            1.0 - 0.5 * blend
        )
        correction = self._return_filter.step(
            (0.0,) * 6,
            tuple((a - q for a, q in zip(frame.positions, sample.target))),
            dt,
            enabled=True,
            reference_velocity=move.sample(float(elapsed), 1),
            reference_acceleration=move.sample(float(elapsed), 2),
        )
        command = tuple(
            (
                q + delta * limit_weight(q, low, high)
                for q, delta, low, high in zip(
                    sample.target, correction, self.limits.lower, self.limits.upper
                )
            )
        )
        self.limits.check_positions(command)
        self._return_last = command
        self._return_peak = max(self._return_peak, abs(command[1] - sample.target[1]))
        return PositionSample(command, sample.acknowledge)

    def _hold_return_command(self, gate: Any) -> PositionSample | None:
        """参考JTC成功末点保持；保持阶段仍核对原实际误差和反馈有效期。"""
        if (
            gate is not self._return_gate
            or self._return_last is None
            or (not gate.playback_status().get('end_transport_delivered'))
        ):
            return None
        frame, move = (self._feedback, self._move)
        if (
            frame is None
            or not 0.0
            <= (time.perf_counter_ns() - frame.source_ns) / 1000000000.0
            <= self.stale_timeout
        ):
            raise RuntimeError('归位完成交接缺少新鲜反馈')
        if not isinstance(move, ContinuousPositionTrajectory):
            raise RuntimeError('归位完成交接缺少原轨迹')
        self.limits.check_positions(frame.positions)
        tolerance = (
            min(move.tracking_error_at(move.duration), self.tracking_tolerance)
            if move.joint_error_rad
            else self.tracking_tolerance
        )
        if max((abs(q - r) for q, r in zip(frame.positions, move.goal))) > tolerance:
            raise RuntimeError('归位完成交接超出原路径误差界限')
        self.limits.check_positions(self._return_last)
        return PositionSample(self._return_last, lambda: None)

    def close(self) -> None:
        """先关闭原控制连接再报告归位修正峰值，不在SDK周期中打印。"""
        peak = self._return_peak
        super().close()
        if peak:
            print(f'归位独立候选修正峰值：{peak:.6f}rad（已关闭连接）', flush=True)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(name + '必须是有限数值')
    return float(value)


def _ramp(value: float) -> float:
    """七次平滑阶跃：两端一至三阶导数为零。"""
    u = min(1.0, max(0.0, value))
    return u**4 * (35.0 + u * (-84.0 + u * (70.0 - 20.0 * u)))


def stage_weight(elapsed: float, start: float, end: float) -> float:
    """只在完整搬运阶段渐入/渐出，不在MOVE/RETURN接缝重置滤波器。"""
    elapsed, start, end = (
        _finite(v, n) for v, n in zip((elapsed, start, end), ('参考时间', '搬运起点', '搬运终点'))
    )
    if end - start < 2.0 * RAMP_S:
        raise ValueError('搬运阶段不足以平滑接入并退出J3修正')
    if elapsed <= start or elapsed >= end:
        return 0.0
    return _ramp((elapsed - start) / RAMP_S) * _ramp((end - elapsed) / RAMP_S)


class CarryJ3Filter:
    """有界比例误差的三极点低通，精确离散化；不积分误差、不使用速度差分。"""

    def __init__(self) -> None:
        self.z1 = self.z2 = self.z3 = 0.0

    def step(self, error: float, dt: float, elapsed: float, start: float, end: float) -> float:
        error, dt = (_finite(error, 'J3误差'), _finite(dt, '更新间隔'))
        if not 0.0 < dt <= 0.04:
            raise ValueError('J3修正更新间隔需在0到40ms内')
        weight = stage_weight(elapsed, start, end)
        if weight == 0.0:
            return 0.0
        goal = -MAX_OFFSET * math.tanh(GAIN * error / MAX_OFFSET)
        h = OMEGA * dt
        decay = math.exp(-h)
        a, b, c = (self.z1 - goal, self.z2 - goal, self.z3 - goal)
        self.z1 = goal + decay * a
        self.z2 = goal + decay * (b + h * a)
        self.z3 = goal + decay * (c + h * b + 0.5 * h * h * a)
        return weight * self.z3


class CarryFeedbackAdapter(ReturnFeedbackAdapter):
    """先执行原J2/几何/新鲜度检查，再仅给搬运J3叠加≤.012rad低带宽修正。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._tracking_trace = deque(maxlen=8000)
        self._carry_filter = CarryJ3Filter()
        self._carry_gate: Any = None
        self._carry_sample_at: float | None = None

    def _stream_position_sample(self) -> PositionSample | None:
        sample = super()._stream_position_sample()
        if sample is None:
            return None
        gate, program = (self._trajectory_gate, self.gripper.program)
        if gate is None or program is None:
            return sample
        if getattr(gate, 'cancelled', False) or self.state in ('STOPPED', 'FAULT'):
            raise TrajectoryCancelled('搬运修正已取消，不再生成位置指令')
        phases = program['phase_start_ns']
        start, end = (phases.get('LIFT'), phases.get('PLACE_SETTLE'))
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or (not isinstance(end, int))
            or isinstance(end, bool)
        ):
            raise ValueError('搬运J3修正缺少完整阶段时间')
        elapsed = gate.playback_status()['reference_elapsed_s']
        if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
            raise ValueError('搬运J3修正参考时间无效')
        elapsed = _finite(elapsed, '参考时间')
        if elapsed <= start / 1000000000.0 or elapsed >= end / 1000000000.0:
            return sample
        if not isinstance(self._move, ContinuousPositionTrajectory) or self._feedback is None:
            raise RuntimeError('搬运J3修正缺少已验证轨迹或反馈')
        if gate is not self._carry_gate:
            self._carry_filter = CarryJ3Filter()
            self._carry_sample_at = None
            self._carry_gate = gate
        now = time.perf_counter()
        dt = (
            1.0 / self.control_hz if self._carry_sample_at is None else now - self._carry_sample_at
        )
        self._carry_sample_at = now
        delta = self._carry_filter.step(
            self._feedback.positions[2] - sample.target[2],
            dt,
            elapsed,
            start / 1000000000.0,
            end / 1000000000.0,
        )
        command = sample.target[:2] + (sample.target[2] + delta,) + sample.target[3:]
        self.limits.check_positions(command)
        if getattr(gate, 'cancelled', False) or self.state in ('STOPPED', 'FAULT'):
            raise TrajectoryCancelled('搬运修正取样期间已取消，不发送旧指令')
        return PositionSample(command, sample.acknowledge)


def qualify_return(payload, report, limits):
    """连接前仅解析归位J2导数，预算覆盖窗口链式项及既有实际起点衔接。"""
    import numpy as np
    from archer_mujoco.joint_interpolation import coefficients, _extrema

    width = RETURN_LIMIT_BAND_RAD
    cap_v = RETURN_MAX_J2_SPEED_RAD_S
    cap_a = RETURN_MAX_J2_ACCEL_RAD_S2
    result = {}
    for side in ('left', 'right'):
        points = payload[side]['points']
        index = report['source_stages'][side].index('RETURN_HOME')
        q = points[index]['positions'][1]
        low, high = (limits[side].lower[1], limits[side].upper[1])
        if width > (high - low) / 2.0:
            raise ValueError(side + '修正窗口超出半关节范围，不能保证C2')
        vmax = amax = 0.0
        for first, last in zip(points[index:], points[index + 1 :]):

            def stamp(p):
                return p['time_from_start']['sec'] + p['time_from_start']['nanosec'] / 1000000000.0

            h = stamp(last) - stamp(first)
            values = [
                np.asarray(p[k], dtype=np.float64)
                for k in ('positions', 'velocities', 'accelerations')
                for p in (first, last)
            ]
            c = coefficients(values[0], values[1], values[2], values[3], values[4], values[5], h)
            vmax = max(vmax, float(np.max(abs(_extrema(c, h, 1)[1]))))
            amax = max(amax, float(np.max(abs(_extrema(c, h, 2)[1]))))
        vmax = max(vmax, 1.875 * 0.25 / 3.15)
        amax = max(amax, 10.0 * math.sqrt(3.0) / 3.0 * 0.25 / 3.15**2)
        if vmax > cap_v or amax > cap_a:
            raise ValueError(side + '归位曲线超过本候选已分配的J2导数预算')
        if min(q - low, high - q) - cap_v * 1.04 < width:
            raise ValueError(side + '归位首1秒无法保证完整窗口，禁止连接')
        w1, w2 = (1.875 / width, 10.0 * math.sqrt(3.0) / 3.0 / width**2)
        correction_v = 0.005 + 0.02 * w1 * cap_v
        correction_a = 0.02 + 2 * w1 * cap_v * 0.005 + 0.02 * (w2 * cap_v**2 + w1 * cap_a)
        if (
            correction_v > 0.01
            or cap_v + 0.01 > limits[side].max_velocity[1]
            or max(cap_a + correction_a, cap_a + 0.04) > limits[side].max_acceleration[1]
        ):
            raise ValueError(side + '最终修正/目标导数预算不足，禁止连接')
        result[side] = {
            'nominal_v_bound': vmax,
            'nominal_a_bound': amax,
            'window_correction_v_bound': correction_v,
            'whole_return_correction_v_bound': 0.01,
            'combined_a_bound': max(cap_a + correction_a, cap_a + 0.04),
        }
    return result
