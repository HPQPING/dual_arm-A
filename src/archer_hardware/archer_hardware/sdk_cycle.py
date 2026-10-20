"""将项目目标交给官方 SDK 周期发送，逐包复核权限、有效期和原始编码."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event, RLock
import time
from typing import Any, Callable


@dataclass
class PendingCommand:
    """保存本代命令的预期编码和实际发送完成事件."""

    encoded: bytes
    epoch: int
    deadline: float
    delivered: Event


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
            elif kind == 'brake':
                self.pending.clear()
                arm.motor_command(self.types.BRAKE, [True] * 6)
            elif kind == 'initialize':
                arm.start()
            elif kind == 'clear_api_timeout':
                arm.clear_parking_stop()
            elif kind not in ('calibrate', 'heartbeat'):
                raise ValueError('未知 SDK 周期请求')
            self.pending[kind] = PendingCommand(
                message.arm_command.SerializeToString(), self.epoch(),
                time.perf_counter() + .04, delivered,
            )
            return delivered

    def cancel(self) -> None:
        """立即使缓存目标失效；之后仍可显式提交本代 BRAKE."""
        with self.lock:
            self.pending.clear()
            self.last_position = None

    def close(self) -> None:
        """关闭发送准入，不发失能、归位或零力矩命令."""
        with self.lock:
            self.closed = True
            self.pending.clear()
            self.last_position = None

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
            # 只有这一处调用官方传输；没有第二套项目周期发送器。
            await base._send_down_message(api, message)
            with self.lock:
                captured.delivered.set()
                if self.pending.get(kind) is captured:
                    if kind == 'position':
                        positions = arm.get_last_positions()
                        self.last_position = tuple(positions) if positions is not None else None
                    elif kind in ('initialize', 'calibrate', 'clear_api_timeout'):
                        self.pending.pop(kind)
        except Exception as error:
            self.fail(str(error))
