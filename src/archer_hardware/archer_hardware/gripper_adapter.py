"""GP80复用机械臂SDK连接；显式启用才启动官方夹爪周期，不发送力矩或零力矩。"""

from __future__ import annotations

import asyncio
from collections import deque
from copy import deepcopy
import hashlib
from importlib import import_module
import json
import math
from pathlib import Path
from threading import Event, RLock
import time
from typing import Any

from .grasp_events import target_at
from .position_adapter import HexPositionAdapter
from .position_plan import ContinuousPositionTrajectory, PairedTrajectoryStart


HANDS_SHA256 = '2cdfd268ba9c64cb58be3e05cfe138910603b61b538b0afdeaf71777b093c00a'


def verified_hands() -> Any:
    """核对Ubuntu已读取的Hands实现，SDK升级后需重新检查。"""
    module = import_module('hex_device.hands')
    source = module.__file__
    if (not isinstance(source, str)
            or hashlib.sha256(Path(source).read_bytes()).hexdigest() != HANDS_SHA256):
        raise RuntimeError('Hands SDK源码已变化，需要重新核对夹爪控制')
    return module.Hands


class HexGripperAdapter:
    """一端口一只GP80，状态机与任务分工参考ALOHA；保留HEX原生编码与限幅。"""

    def __init__(self, owner: HexPositionAdapter) -> None:
        """仅分配状态，不连接、不启用夹爪。"""
        self.owner = owner
        self.lock = RLock()
        self.api: Any = None
        self.device: Any = None
        self.frame: tuple[Any, int, float] | None = None
        self.requested = False
        self.active = False
        self.closed = False
        self.mode = 'disabled'
        self.error = ''
        self.updated_at = 0.
        self.started_at = 0.
        self.seed = 0.
        self.command = 0.
        self.delivered = Event()
        self.future: Any = None
        self.program: dict[str, Any] | None = None
        self.gate: PairedTrajectoryStart | None = None
        self.expected: bytes | None = None
        self.encoded_command: float | None = None
        self.sent_command: float | None = None
        self._hands_type: Any = None
        self._last_hand_stamp = -1

    def attach(self, api: Any) -> None:
        """只监听已存在连接的附属反馈，保持夹爪周期关闭。"""
        self._hands_type = verified_hands()
        self.api = api
        original = api._process_api_up

        def observe(message: Any) -> None:
            received = time.perf_counter_ns()
            original(message)
            if self.closed:
                return
            for secondary in message.secondary_device_status:
                if secondary.device_id != 1:
                    continue
                device = api.find_optional_device_by_id(1)
                if isinstance(device, self._hands_type):
                    status = device.get_simple_motor_status(pop=False)
                    if status is not None:
                        stamp = status['ts']
                        seconds, nanos = stamp['s'], stamp['ns']
                        if (type(seconds) is not int or type(nanos) is not int
                                or seconds < 0 or not 0 <= nanos < 10**9):
                            continue
                        source_ns = seconds * 10**9 + nanos
                        with self.lock:
                            # 重复的SDK状态不能通过新的本地读取时刻续活。
                            if source_ns <= self._last_hand_stamp:
                                continue
                            self._last_hand_stamp = source_ns
                            self.frame = (device, received, self._one(status['pos']))

        api._process_api_up = observe

    @staticmethod
    def _one(values: Any) -> float:
        if values is None or len(values) != 1 or isinstance(values[0], (bool, str)):
            raise ValueError('GP80需要一个有限电机位置')
        result = float(values[0])
        if not math.isfinite(result):
            raise ValueError('夹爪反馈不是有限数值')
        return result

    def _feedback(self) -> tuple[Any, float]:
        if self.frame is None or (time.perf_counter_ns() - self.frame[1]) / 1e9 > .1:
            raise RuntimeError('夹爪缺少100ms内的新鲜反馈')
        device, _, position = self.frame
        protocol = import_module('hex_device.generated.public_api_types_pb2')
        if (device.motor_count != 1 or device.device_id != 1
                or device.get_hand_type() != protocol.SecondaryDeviceType.SdtHandGp80G1
                or device.get_joint_limits()[:2] != [0., 5.65]):
            raise RuntimeError('当前端口夹爪不是已核对的单电机GP80G1/ID1')
        if device.get_motor_state(0) == 'error':
            raise RuntimeError('夹爪报告电机故障')
        return device, position

    def request_enable(self) -> None:
        """任务线程只提交意图，SDK操作仍由机械臂工作线程执行。"""
        with self.lock:
            if self.closed or self.error or self.owner.state != 'HOLDING':
                raise RuntimeError('夹爪仅允许在已准备的原位保持阶段启用')
            self.requested = True

    def bind_program(self, schedule: dict[str, Any], gate: PairedTrajectoryStart) -> None:
        """绑定已核对的整轮阶段时钟，不另起播放时钟或争夺控制连接。"""
        with self.lock:
            if not self.status()['ready']:
                raise RuntimeError('夹爪初始指令尚未交付，不能启动协同路径')
            copy = deepcopy(schedule)
            target_at(copy, 0.)
            self.program, self.gate = copy, gate

    def _enable(self) -> None:
        device, position = self._feedback()
        if not 0. <= position <= 5.65:
            raise RuntimeError('夹爪实际初值超出SDK目标范围，请先检查当前摆放或零位')
        self.device, self.seed = device, position
        self.started_at = self.updated_at = time.monotonic()
        self.mode, self.active = 'position', True
        self.command = position
        types = self.owner._command_types
        # 先提供实测位置，再启动原生周期，避免None看门狗或默认目标触发动作。
        device.motor_command(types.POSITION, [position])
        original = device._construct_joint_command_msg

        def encode() -> Any:
            try:
                with self.lock:
                    if self.closed:
                        raise RuntimeError('夹爪构包已关闭')
                    if self.mode == 'brake':
                        message = original()
                        self._validate_message(message, 'brake')
                        return message
                    if not self.active or self.mode != 'position':
                        raise RuntimeError('夹爪位置构包已失效')
                    if time.monotonic() - self.updated_at > .04:
                        raise RuntimeError('夹爪项目目标超过40ms未刷新')
                    message = original()
                    self._validate_message(message, 'position')
                    self.expected = message.SerializeToString()
                    self.encoded_command = self.command
                    return message
            except Exception as error:
                with self.lock:
                    self.error = str(error)
                raise

        device._construct_joint_command_msg = encode
        device._set_send_message_callback(self.dispatch)
        loop = getattr(self.api, '_HexDeviceApi__loop', None)
        if loop is None or not loop.is_running():
            raise RuntimeError('SDK事件循环尚未就绪')
        self.future = asyncio.run_coroutine_threadsafe(device._periodic(), loop)

    def tick(self) -> None:
        """复用机械臂工作线程刷新目标；官方Hands周期负责实际编码和传输。"""
        with self.lock:
            if self.error:
                raise RuntimeError(self.error)
            if self.requested and not self.active:
                self._enable()
            if not self.active:
                return
            if self.gate is not None and self.gate.cancelled:
                self.stop()
                return
            self._feedback()
            if self.future is not None and self.future.done():
                raise RuntimeError('官方夹爪周期已意外退出')
            if self.program is not None and self.gate is not None:
                elapsed = self.gate.playback_status()['reference_elapsed_s']
                if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
                    raise RuntimeError('夹爪共同参考时间不是有限数值')
                self.command = target_at(self.program, float(elapsed))
            else:
                u = min(1., max(0., time.monotonic() - self.started_at))
                blend = 6.*u**5 - 15.*u**4 + 10.*u**3
                self.command = self.seed + (5.65 - self.seed) * blend
            if not 0. <= self.command <= 5.65:
                raise RuntimeError('夹爪目标超出SDK位置范围')
            self.device.motor_command(self.owner._command_types.POSITION, [self.command])
            self.updated_at = time.monotonic()

    @staticmethod
    def _validate_message(message: Any, kind: str) -> None:
        if {f.name for f, _ in message.ListFields()} != {'secondary_device_command'}:
            raise RuntimeError('夹爪通道拒绝非附属设备命令')
        secondary = message.secondary_device_command
        if ({f.name for f, _ in secondary.ListFields()} != {'device_id', 'hand_command'}
                or secondary.device_id != 1
                or {f.name for f, _ in secondary.hand_command.ListFields()} != {'motor_targets'}):
            raise RuntimeError('夹爪命令设备或字段不匹配')
        targets = secondary.hand_command.motor_targets.targets
        if len(targets) != 1 or {f.name for f, _ in targets[0].ListFields()} != {kind}:
            raise RuntimeError('夹爪只允许单电机位置或显式制动，拒绝MIT/速度/力矩')
        if kind == 'brake' and targets[0].brake is not True:
            raise RuntimeError('拒绝解除制动')

    async def dispatch(self, message: Any) -> None:
        """发送出口再次检查生命周期、控制权和夹爪专属报文。"""
        try:
            with self.lock:
                if self.closed:
                    return
                if self.mode == 'position' and self.gate is not None and self.gate.cancelled:
                    self.stop()
                    return
                if self.mode == 'brake':
                    try:
                        self._validate_message(message, 'position')
                    except RuntimeError:
                        pass
                    else:
                        # 制动前已编码的合法位置包只丢弃，不转成新的故障或重新发送。
                        return
                kind = 'position' if self.mode == 'position' else 'brake'
                self._validate_message(message, kind)
                if kind == 'position':
                    if not self.active or time.monotonic() - self.updated_at > .04:
                        raise RuntimeError('夹爪发送许可过期')
                    self._feedback()
                    if message.SerializeToString() != self.expected:
                        raise RuntimeError('夹爪实际编码与本次构包不匹配')
                    if self.owner.state not in ('HOLDING', 'ARMED', 'RUNNING'):
                        raise RuntimeError('机械臂状态不允许夹爪位置控制')
                    if not all(self.owner._health(require_calibrated=True)):
                        raise RuntimeError('夹爪发送前机械臂控制准备已失效')
                    frame = self.owner._feedback
                    if frame is None or not 0. <= (
                        time.perf_counter_ns() - frame.source_ns
                    ) / 1e9 <= .1:
                        raise RuntimeError('夹爪发送前缺少新鲜机械臂反馈')
                command = self.encoded_command if kind == 'position' else None
                arm = self.owner._arm()
                if arm.get_my_session_id() <= 0 or arm.get_session_holder() != (
                    arm.get_my_session_id()
                ):
                    raise RuntimeError('夹爪发送前机械臂控制权已经丢失')
            # 复用同一连接的官方传输；机械臂的native发送准入保持原样。
            base = import_module('hex_device.hex_device_api').HexDeviceApi
            await base._send_down_message(self.api, message)
            with self.lock:
                if self.closed or self.mode != kind:
                    return
                self.sent_command = command
                self.delivered.set()
        except Exception as error:
            with self.lock:
                first = not self.error
                self.error = str(error)
            if first:
                self.owner._native_failure('夹爪：' + str(error))

    def status(self) -> dict[str, Any]:
        """仅报告指令交付，不把位置误差或电流当作真实夹持确认。"""
        with self.lock:
            ready = (self.active and not self.error and self.delivered.is_set()
                     and time.monotonic() - self.started_at >= 1.
                     and self.sent_command is not None
                     and abs(self.sent_command - 5.65) <= 1e-9)
            return {'active': self.active, 'ready': ready, 'error': self.error,
                    'command': self.command, 'sent_command': self.sent_command,
                    'actual_position': None if self.frame is None else self.frame[2],
                    'feedback_age_s': (None if self.frame is None else
                                       max(0., (time.perf_counter_ns()-self.frame[1])/1e9)),
                    'transport_delivered': self.delivered.is_set(),
                    'physical_grasp_verified': False}

    def stop(self, *, wait: bool = False) -> None:
        """取消夹爪位置计划并尽力BRAKE；不张开、不闭合、不归位。"""
        with self.lock:
            self.requested = False
            self.program = self.gate = None
            if self.device is None:
                return
            if self.active:
                self.active, self.mode = False, 'brake'
                self.delivered.clear()
                self.device.motor_command(self.owner._command_types.BRAKE, [True])
        if wait and self.mode == 'brake':
            self.delivered.wait(.05)

    def close(self) -> None:
        """先关闭准入再取消唯一的夹爪周期，防止连接关闭时重发位置。"""
        self.stop(wait=True)
        with self.lock:
            self.closed = True
            if self.future is not None:
                self.future.cancel()


class CooperativePositionAdapter(HexPositionAdapter):
    """仅协同入口选择此类型；普通画圆仍使用原HexPositionAdapter。"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.gripper = HexGripperAdapter(self)
        self._tracking_trace: deque[dict[str, Any]] = deque(maxlen=251)
        self._tracking_trace_lock = RLock()
        self._tracking_trace_last_ns = 0
        self._tracking_trace_error = ''

    def connect(self) -> None:
        verified_hands()
        if self.gripper.closed:
            self.gripper = HexGripperAdapter(self)
        super().connect()
        self.gripper.attach(self._api)
        with self._tracking_trace_lock:
            self._tracking_trace.clear()
            self._tracking_trace_last_ns = 0
            self._tracking_trace_error = ''
        original = self._api._process_api_up

        def observe(message: Any) -> None:
            original(message)
            self.observe_tracking(message)

        self._api._process_api_up = observe

    def observe_tracking(self, message: Any) -> None:
        """只观察已有同帧反馈，最多50Hz/251帧；诊断失败不得改变控制状态。"""
        if self.state != 'RUNNING':
            return
        now = time.perf_counter_ns()
        if now - self._tracking_trace_last_ns < 20_000_000:
            return
        try:
            move, gate = self._move, self._trajectory_gate
            if not isinstance(move, ContinuousPositionTrajectory) or gate is None:
                return
            motors = message.arm_status.motor_status
            if len(motors) != 6:
                return
            frame = self._api._project_feedback_frames.get(message.robot_type)
            if frame is None:
                return
            value = gate.playback_status()['reference_elapsed_s']
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError('跟踪观测参考时间不是数值')
            elapsed = float(value)
            if not math.isfinite(elapsed):
                raise ValueError('跟踪观测参考时间不是有限数值')
            status = frame[2]
            device = tuple(
                float(m.current_target.position)*2.*math.pi/float(m.pulse_per_rotation)
                if m.current_target.HasField('position') and m.pulse_per_rotation > 0 else None
                for m in motors)
            timing = self._native.position_timing() if self._native is not None else None
            row = {'reference_s': elapsed, 'received_ns': frame[1],
                   'desired_rad': move.sample(elapsed),
                   'actual_rad': tuple(float(x) for x in status['pos']),
                   'velocity_rad_s': tuple(float(x) for x in status['vel']),
                   'effort': tuple(float(x) for x in status['eff']),
                   'device_rad': device, 'sent_rad': timing[0] if timing else None}
            with self._tracking_trace_lock:
                self._tracking_trace.append(row)
                self._tracking_trace_last_ns = now
        except Exception as error:
            # 该旁路不发指令、不参与准入；观测字段缺失只在故障上下文注明。
            self._tracking_trace_error = str(error)

    def tracking_trace(self) -> tuple[dict[str, Any], ...]:
        """返回有界浅副本供错误说明使用；不写CSV/JSON，不触发设备读取。"""
        lock = getattr(self, '_tracking_trace_lock', None)
        if lock is None:
            return ()
        with lock:
            return tuple(self._tracking_trace)

    def _tracking_detail(self, error: Exception, *, native: bool = False) -> str:
        """保留原停止判断，仅补充同一时刻的最差关节与反馈年龄供定位。"""
        text = str(error)
        if not any(value in text for value in (
            '实际反馈超出完整路径几何误差界限', 'SDK周期取样目标与实际反馈偏差超限',
        )):
            return text
        frame, target, move = self._feedback, self._target, self._move
        gate = self._trajectory_gate
        elapsed = 0.
        if gate is not None:
            value = gate.playback_status()['reference_elapsed_s']
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                elapsed = float(value)
        if native and isinstance(move, ContinuousPositionTrajectory):
            target = move.sample(elapsed)
        if frame is None or target is None:
            return text
        errors = [abs(a - b) for a, b in zip(frame.positions, target)]
        joint = max(range(6), key=errors.__getitem__)
        cap = (move.tracking_error_at(elapsed) if isinstance(move, ContinuousPositionTrajectory)
               else self.tracking_tolerance)
        age = (time.perf_counter_ns() - frame.source_ns) / 1e9
        rows = self.tracking_trace()
        context = ''
        if rows:
            selected = sorted({round(i*(len(rows)-1)/min(11, len(rows)-1))
                               for i in range(min(12, len(rows)))}) if len(rows) > 1 else [0]
            values = [[round(rows[i]['reference_s'], 4),
                       round(rows[i]['desired_rad'][joint], 6),
                       round(rows[i]['actual_rad'][joint], 6),
                       round(rows[i]['velocity_rad_s'][joint], 6),
                       (round(rows[i]['sent_rad'][joint], 6)
                        if rows[i]['sent_rad'] is not None else None),
                       (round(rows[i]['device_rad'][joint], 6)
                        if rows[i]['device_rad'][joint] is not None else None),
                       round(rows[i]['effort'][joint], 6)] for i in selected]
            context = ('；同轴近5秒[参考s,期望rad,实际rad,实际速度rad/s,已发rad,设备目标rad,反馈力矩]='
                       + json.dumps(values, separators=(',', ':')))
        trace_error = getattr(self, '_tracking_trace_error', '')
        if trace_error:
            context += '；跟踪观测字段不可用：' + trace_error
        return (text + f'；J{joint + 1} 目标={target[joint]:.6f}rad'
                f'，反馈={frame.positions[joint]:.6f}rad，偏差={errors[joint]:.6f}rad'
                f'，上限={cap:.6f}rad，参考时间={elapsed:.3f}s，反馈年龄={age:.4f}s' + context)

    def _tick(self) -> None:
        """沿用原驱动判断，在原方法清理轨迹对象之前提取故障上下文。"""
        try:
            super()._tick()
        except RuntimeError as error:
            detail = self._tracking_detail(error)
            if detail == str(error):
                raise
            raise RuntimeError(detail) from error

    def _stream_position_sample(self) -> Any:
        """原生SDK采样仍按原规则停止；只增强协同任务的错误说明。"""
        try:
            return super()._stream_position_sample()
        except RuntimeError as error:
            detail = self._tracking_detail(error, native=True)
            if detail == str(error):
                raise
            raise RuntimeError(detail) from error

    def tick(self) -> str:
        state = super().tick()
        try:
            if state in ('HOLDING', 'ARMED', 'RUNNING'):
                self.gripper.tick()
        except Exception as error:
            self.last_error = str(error)
            self.stop(fault=True)
            raise
        return self.state

    def stop(self, *, fault: bool = False, wait: bool = True) -> None:
        self.gripper.stop(wait=wait)
        super().stop(fault=fault, wait=wait)

    def close(self) -> None:
        try:
            self.gripper.close()
        finally:
            super().close()
