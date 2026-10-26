"""将项目目标交给官方 SDK 周期发送，逐包复核权限、有效期和原始编码."""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
from threading import Event, RLock
import time
from typing import Any, Callable

from .position_plan import TrajectoryCancelled


@dataclass(frozen=True)
class PositionSample:
    """绑定本次连续取样目标与该包实际发送后的确认，不表示控制器收包。"""

    target: tuple[float, ...]
    acknowledge: Callable[[], None]


@dataclass
class PendingCommand:
    """保存本代命令的预期编码和实际发送完成事件."""

    encoded: bytes
    epoch: int
    deadline: float
    delivered: Event
    target: tuple[float, ...] | None = None
    acknowledge: Callable[[], None] | None = None


class SdkCycleSender:
    """不创建发送线程，所有传输仅发生在官方 Arm 周期回调内."""

    def __init__(
        self, check: Callable[[Any, str], None], fail: Callable[[str], None],
        epoch: Callable[[], int], command_types: Any,
    ) -> None:
        """构造只保存回调，不建立网络连接."""
        self.check = check
        self.fail = fail
        self.epoch = epoch
        self.types = command_types
        self.lock = RLock()
        self.pending: dict[str, PendingCommand] = {}
        self.closed = False
        self.last_position: tuple[float, ...] | None = None
        self.last_position_at: float | None = None
        self.sample_position: Callable[[], PositionSample | None] | None = None
        self.cancel_position: Callable[[], None] | None = None
        self._position_cancelled = False

    def _brake_message(self, arm: Any) -> Any:
        """为SDK已复制的旧位置命令返回合法制动编码，仍由发送出口核对准入。"""
        targets = arm._construct_custom_motor_msg(self.types.BRAKE, [True] * 6)
        return arm._construct_custom_joint_command_msg(targets)

    def _cancel_position(self) -> None:
        """复用调用方停止状态和BRAKE生命周期，不在SDK线程等待自身发送。"""
        if self.cancel_position is not None:
            self.cancel_position()
        else:
            self.cancel()

    def encode_position(self, arm: Any, command: Any, encode: Callable[[Any], Any]) -> Any:
        """在官方周期实际构包时取样，避免反复发送旧缓存后跳到较晚目标."""
        with self.lock:
            if (command.command_type == self.types.POSITION
                    and (self.closed or self._position_cancelled)):
                return self._brake_message(arm)
            request = self.pending.get('position')
            if (self.closed or self.sample_position is None or request is None
                    or command.command_type != self.types.POSITION):
                return encode(command)
            if request.epoch != self.epoch():
                raise RuntimeError('SDK 构包时目标代次已失效')
            if time.perf_counter() > request.deadline:
                raise RuntimeError('SDK 目标超过40ms未更新，禁止周期取样复活旧任务')
            try:
                sample = self.sample_position()
            except TrajectoryCancelled:
                self._cancel_position()
                return self._brake_message(arm)
            if sample is None:
                return encode(command)
            sampled = deepcopy(command)
            sampled.position_command = list(sample.target)
            message = encode(sampled)
            self.check(message, 'position')
            request.encoded = message.arm_command.SerializeToString()
            request.target = sample.target
            request.acknowledge = sample.acknowledge
            return message

    def position_timing(self) -> tuple[tuple[float, ...], float] | None:
        """返回上一实际发送位置及其单调时间间隔，不使用规划缓存更新时刻."""
        with self.lock:
            if self.last_position is None or self.last_position_at is None:
                return None
            return self.last_position, time.perf_counter() - self.last_position_at

    def publish(
        self, arm: Any, message: Any, kind: str, target: tuple[float, ...] | None,
    ) -> Event:
        """更新官方命令缓存；不调用任何传输函数，不积压位置队列."""
        with self.lock:
            if self.closed:
                raise RuntimeError('SDK 发送器正在退出')
            self.check(message, kind)
            delivered = Event()
            if kind == 'position':
                if target is None:
                    raise RuntimeError('位置目标为空')
                arm.motor_command(self.types.POSITION, list(target))
                self._position_cancelled = False
            elif kind == 'brake':
                self.pending.clear()
                self._position_cancelled = True
                arm.motor_command(self.types.BRAKE, [True] * 6)
            elif kind == 'initialize':
                arm.start()
            elif kind == 'clear_api_timeout':
                arm.clear_parking_stop()
            elif kind not in ('calibrate', 'heartbeat'):
                raise ValueError('未知 SDK 周期请求')
            encoded = message.arm_command.SerializeToString()
            previous = self.pending.get(kind)
            if (kind == 'position' and previous is not None
                    and previous.encoded == encoded and previous.epoch == self.epoch()
                    and previous.target == target and not previous.delivered.is_set()):
                # 重复同一目标刷新有效期，但保留正在等待的发送确认事件。
                previous.deadline = time.perf_counter() + .04
                return previous.delivered
            self.pending[kind] = PendingCommand(
                encoded, self.epoch(),
                time.perf_counter() + .04, delivered,
                target if kind == 'position' else None,
            )
            return delivered

    def cancel(self) -> None:
        """立即使缓存目标失效；之后仍可显式提交本代 BRAKE."""
        with self.lock:
            self.pending.clear()
            self._position_cancelled = True
            self.last_position = None
            self.last_position_at = None

    def close(self) -> None:
        """关闭发送准入，不发失能、归位或零力矩命令."""
        with self.lock:
            self.closed = True
            self._position_cancelled = True
            self.pending.clear()
            self.last_position = None
            self.last_position_at = None

    def wait_pending_brake(self, timeout: float = .05) -> bool:
        """等待已发布BRAKE的底层发送完成，不发布新命令，也不证明物理制动."""
        with self.lock:
            request = self.pending.get('brake')
        return request is None or request.delivered.wait(timeout)

    async def dispatch(self, api: Any, base: Any, arm: Any, message: Any) -> None:
        """只转发仍有效且与项目预期完全相同的厂商原生周期报文."""
        try:
            with self.lock:
                if self.closed:
                    return
                fields = {field.name for field, _ in message.ListFields()}
                if fields != {'arm_command'}:
                    raise RuntimeError('SDK 周期产生了未授权的顶层命令')
                outer = message.arm_command
                outer_fields = {f.name for f, _ in outer.ListFields()}
                command = outer.arm_exclusive_command
                field = command.WhichOneof('exclusive_command')
                if outer_fields == {'arm_shared_command'}:
                    shared = outer.arm_shared_command
                    if ({f.name for f, _ in shared.ListFields()} != {'clear_parking_stop'}
                            or shared.clear_parking_stop is not True):
                        return
                    kind = 'clear_api_timeout'
                elif outer_fields != {'arm_exclusive_command'}:
                    return
                elif field == 'api_control_initialize':
                    kind = 'initialize'
                    if command.api_control_initialize is not True:
                        return
                elif field == 'calibrate':
                    kind = 'calibrate'
                    if kind not in self.pending and 'heartbeat' in self.pending:
                        message = arm._construct_empty_message()
                        kind = 'heartbeat'
                elif field == 'arm_api_control_command':
                    control = command.arm_api_control_command
                    if {f.name for f, _ in control.ListFields()} != {'motor_targets'}:
                        raise RuntimeError('禁止 SDK 特殊控制模式')
                    targets = control.motor_targets.targets
                    if len(targets) == 0:
                        kind = 'heartbeat'
                    elif len(targets) == 6 and all(
                        {f.name for f, _ in p.ListFields()} == {'position'} for p in targets
                    ):
                        kind = 'position'
                    elif len(targets) == 6 and all(
                        {f.name for f, _ in p.ListFields()} == {'brake'} and p.brake is True
                        for p in targets
                    ):
                        kind = 'brake'
                    else:
                        raise RuntimeError('禁止 SDK 速度、力矩、MIT 或非六轴目标')
                else:
                    return
                request = self.pending.get(kind)
                if request is None or request.epoch != self.epoch():
                    return
                if kind != 'brake' and time.perf_counter() > request.deadline:
                    raise RuntimeError('SDK 目标超过40ms未更新，禁止重复过期目标')
                if message.arm_command.SerializeToString() != request.encoded:
                    # 新目标已发布时，周期任务可能刚刚复制了旧目标；不能发旧包。
                    return
                self.check(message, kind)
                captured = request
                sent_target = captured.target
                acknowledge = captured.acknowledge
            # 只有这一处调用官方传输；没有第二套项目周期发送器。
            sent_at = time.perf_counter()
            await base._send_down_message(api, message)
            with self.lock:
                captured.delivered.set()
                if (kind == 'position' and captured.epoch == self.epoch()
                        and not self.closed and not self._position_cancelled):
                    # 实际发送已完成，即使缓存已更新，也须登记这份包的原始目标。
                    # 不读可能已被后来构造操作更新的SDK历史，更不复活取消代次。
                    self.last_position = sent_target
                    self.last_position_at = sent_at
                    if acknowledge is not None:
                        acknowledge()
                if self.pending.get(kind) is captured:
                    if kind in ('initialize', 'calibrate', 'clear_api_timeout'):
                        self.pending.pop(kind)
        except TrajectoryCancelled:
            self._cancel_position()
        except Exception as error:
            self.fail(str(error))
