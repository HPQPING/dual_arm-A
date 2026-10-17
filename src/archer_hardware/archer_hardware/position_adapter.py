"""
独立的 SDK 关节位置控制接入，默认只连接反馈，不自动申请控制权.

调用方需显式申请控制并持续调用 tick；软件制动请求不等同于实体急停。
本轮通过离线测试验证，尚未连接实机确认固件响应。
"""

from __future__ import annotations

import asyncio
from importlib import import_module
import time
from typing import Any, Iterable, cast

from .hex_adapter import (
    HexFeedbackAdapter, JointFeedback, report_only, verified_sdk,
)
from .position_plan import JointPositionMove, PositionLimits, StopAndReturnMove, positive


class HexPositionAdapter(HexFeedbackAdapter):
    """每个实例管理一只臂，不接收 MIT、力矩或速度模式指令."""

    def __init__(
        self, url: str, limits: PositionLimits, *, control_hz: int = 250,
        max_excursion: float = 0.05, tracking_tolerance: float = 0.10,
    ) -> None:
        """要求显式传入限位，小幅试运行默认每关节最多变化 0.05 rad."""
        super().__init__(url, stale_timeout=0.1)
        if type(control_hz) is not int or control_hz not in (250, 500):
            raise ValueError('当前接入层只接受 250 或 500 Hz 的目标发送频率')
        self.limits = limits
        self.control_hz = control_hz
        self.max_excursion = positive(max_excursion, 'max_excursion')
        self.tracking_tolerance = positive(tracking_tolerance, 'tracking_tolerance')
        self.state = 'DISCONNECTED'
        self.last_error = ''
        self.initial_positions: tuple[float, ...] | None = None
        self._feedback: JointFeedback | None = None
        self._target: tuple[float, ...] | None = None
        self._move: JointPositionMove | StopAndReturnMove | None = None
        self._base: Any = None
        self._command_types: Any = None
        self._epoch = 0
        self._requested_at = 0.0
        self._last_tick = 0.0
        self._move_started = 0.0
        self._settled_since: float | None = None
        self._nominal_required = False
        self._recovering_limits = False

    def connect(self) -> None:
        """核对 SDK 后建立反馈连接，不启动厂商设备控制周期任务."""
        if self.state != 'DISCONNECTED':
            raise RuntimeError('需要先关闭原连接')
        module = verified_sdk()
        base = cast(Any, module.HexDeviceApi)

        class PositionAPI(base):
            """官方周期发送关闭；设备回调只能下发反馈频率设置."""

            def _start_device_periodic_task(self, device_id: int) -> None:
                """禁止默认自动清错、校准和抢占控制逻辑."""
                return None

            async def _send_down_message(self, message: Any) -> None:
                """厂商内部回调仅允许反馈配置，不发送控制消息."""
                if report_only(message):
                    await super()._send_down_message(message)

        self._base = base
        self._arm_type = import_module('hex_device.arm').Arm
        self._command_types = import_module('hex_device.motor_base').CommandType
        self._api = PositionAPI(
            ws_url=self.url, control_hz=self.control_hz, enable_kcp=False,
        )
        self.last_error = ''
        self._epoch += 1
        self.state = 'CONNECTED'

    def _arm(self) -> Any:
        arms = [d for d in self._api.device_list if isinstance(d, self._arm_type)]
        if len(arms) != 1 or arms[0].motor_count != 6:
            raise RuntimeError('尚未发现唯一的六轴机械臂')
        return arms[0]

    def read_feedback(self) -> JointFeedback | None:
        """读取或复用仍有效的反馈，供控制线程统一维护同一份缓存."""
        frame = super().read()
        if frame is not None:
            self._feedback = frame
        if self._feedback is None:
            return None
        age = (time.perf_counter_ns() - self._feedback.source_ns) / 1e9
        if not 0 <= age <= self.stale_timeout:
            raise RuntimeError('真实关节反馈过期')
        return self._feedback

    def _fresh_feedback(self) -> JointFeedback:
        frame = self.read_feedback()
        if frame is None:
            raise RuntimeError('尚未收到真实关节反馈')
        self.limits.check_feedback(frame.positions)
        if self._nominal_required:
            self.limits.check_positions(frame.positions)
        return frame

    def _health(self) -> tuple[bool, bool]:
        arm = self._arm()
        if arm.get_parking_stop_detail().ListFields():
            raise RuntimeError('机械臂报告停车故障；本接口不会自动清除')
        errors = arm.get_motor_error_codes()
        if errors is None or len(errors) != 6:
            raise RuntimeError('尚未取得六个电机的故障状态')
        if any(code is not None for code in errors):
            raise RuntimeError(f'电机报告故障：{errors}')
        with arm._status_lock:
            calibrated = arm._calibrated
            initialized = arm._api_control_initialized
            holder = arm._session_holder
            mine = arm._my_session_id
        if calibrated is not True:
            raise RuntimeError('机械臂尚未校准；本接口不会自动校准')
        if mine <= 0:
            raise RuntimeError('尚未取得有效会话编号')
        if holder not in (0, mine):
            raise RuntimeError('控制权属于其他会话，不抢占')
        return holder == mine, initialized is True

    def _send(self, message: Any, kind: str) -> None:
        """有界等待官方发送协程，过期排队消息在发送前作废."""
        api = self._api
        if api is None:
            raise RuntimeError('设备未连接')
        epoch = self._epoch
        deadline = time.perf_counter() + 0.04

        async def deliver() -> None:
            if epoch != self._epoch or time.perf_counter() > deadline:
                raise RuntimeError('控制消息已作废或发送超时')
            permitted = (
                kind == 'initialize' and self.state == 'REQUESTING'
                or kind == 'position' and self.state in ('HOLDING', 'RUNNING', 'RETURNING')
                or kind == 'brake' and self.state in ('STOPPED', 'FAULT')
            )
            if not permitted:
                raise RuntimeError('当前状态不允许下发此类消息')
            # 只由下面固定的官方构造器产生消息，不提供任意消息发送入口。
            await self._base._send_down_message(api, message)

        loop = api._HexDeviceApi__loop
        if loop is None or not loop.is_running():
            raise RuntimeError('SDK 通信线程未运行')
        future = asyncio.run_coroutine_threadsafe(deliver(), loop)
        try:
            future.result(timeout=0.05)
        except Exception:
            future.cancel()
            self._epoch += 1
            raise

    def request_control(self) -> None:
        """在新鲜、静止反馈和无故障状态下显式申请控制权."""
        if self.state != 'CONNECTED':
            raise RuntimeError('仅允许从 CONNECTED 状态申请控制')
        frame = self._fresh_feedback()
        self._health()
        if max(abs(v) for v in frame.velocities) > 0.02:
            raise RuntimeError('机械臂仍在运动，不能申请初始位置保持')
        self.state = 'REQUESTING'
        self._requested_at = time.monotonic()
        try:
            self._send(self._arm()._construct_init_message(True), 'initialize')
        except Exception as error:
            self.last_error = str(error)
            self.stop(fault=True)
            raise

    def move_to(self, goal: Iterable[object], duration: float) -> None:
        """从当前实际静止位置规划单段小幅移动，不抢占正在运行的轨迹."""
        if self.state != 'HOLDING':
            raise RuntimeError('需要先获得控制权并处于 HOLDING 状态')
        frame = self._fresh_feedback()
        owned, initialized = self._health()
        if not owned or not initialized:
            raise RuntimeError('控制会话无效')
        if max(abs(v) for v in frame.velocities) > 0.02:
            raise RuntimeError('需要等待机械臂静止后再开始新轨迹')
        move = JointPositionMove.prepare(
            frame.positions, goal, duration, self.limits, self.max_excursion,
        )
        self._recovering_limits = not self.limits.within_nominal(frame.positions)
        self._nominal_required = not self._recovering_limits
        self._move = move
        self._move_started = time.monotonic()
        self._settled_since = None
        self.state = 'RUNNING'

    def return_to_initial(self, duration: float) -> None:
        """普通停止时先按实际速度减速，再受控返回本轮初始位置."""
        if self.initial_positions is None:
            raise RuntimeError('尚未记录真实初始位置')
        if self.state not in ('HOLDING', 'RUNNING'):
            raise RuntimeError('当前不允许受控归位；故障时应立即停止')
        frame = self._fresh_feedback()
        owned, initialized = self._health()
        if not owned or not initialized:
            raise RuntimeError('归位前控制会话无效')
        move = StopAndReturnMove.prepare(
            frame.positions, frame.velocities, self.initial_positions, duration, self.limits,
        )
        self._epoch += 1
        self._nominal_required = False
        self._recovering_limits = False
        self._move = move
        self._move_started = time.monotonic()
        self._settled_since = None
        self.state = 'RETURNING'

    def tick(self) -> str:
        """由单一调用线程按目标频率执行；异常锁存并请求软件制动."""
        if self.state not in ('REQUESTING', 'HOLDING', 'RUNNING', 'RETURNING'):
            return self.state
        try:
            self._tick()
        except Exception as error:
            self.last_error = str(error)
            self.stop(fault=True)
            raise
        return self.state

    def _tick(self) -> None:
        now = time.monotonic()
        frame = self._fresh_feedback()
        owned, initialized = self._health()
        if self.state == 'REQUESTING':
            if now - self._requested_at > 2.0:
                raise RuntimeError('申请控制权超时')
            if not owned or not initialized:
                return
            if max(abs(v) for v in frame.velocities) > 0.02:
                raise RuntimeError('取得控制权时机械臂未静止')
            self.initial_positions = frame.positions
            self._target = frame.positions
            self._last_tick = now
            self.state = 'HOLDING'
        if not owned or not initialized:
            raise RuntimeError('控制权已丢失或控制模式未初始化')
        if now - self._last_tick > 0.04:
            raise RuntimeError('调用 tick 的间隔超过 40 ms')
        self._last_tick = now
        if self.state in ('RUNNING', 'RETURNING'):
            assert self._move is not None
            elapsed = now - self._move_started
            self._target = self._move.sample(elapsed)
            if (self._recovering_limits and self.limits.within_nominal(frame.positions)
                    and self.limits.within_nominal(self._target)):
                self._nominal_required = True
            if elapsed >= self._move.duration:
                error = max(abs(a - b) for a, b in zip(frame.positions, self._move.goal))
                stopped = max(abs(v) for v in frame.velocities) <= 0.02
                recovered = (not self._recovering_limits
                             or self.limits.within_nominal(frame.positions))
                if error <= 0.01 and stopped and recovered:
                    if self._settled_since is None:
                        self._settled_since = now
                    if now - self._settled_since >= 0.2:
                        self.state = 'HOLDING'
                        self._recovering_limits = False
                else:
                    self._settled_since = None
                if self.state in ('RUNNING', 'RETURNING') and elapsed > self._move.duration + 2.0:
                    raise RuntimeError('目标位置未在允许时间内到位')
        assert self._target is not None
        if max(abs(a - b) for a, b in zip(frame.positions, self._target)) > (
            self.tracking_tolerance
        ):
            raise RuntimeError('实际位置与目标位置偏差超限')
        arm = self._arm()
        # 复用官方弧度到编码器转换；限位和限速由本实例整段解析预检完成。
        targets = arm._construct_custom_motor_msg(
            self._command_types.POSITION, list(self._target),
        )
        self._send(arm._construct_custom_joint_command_msg(targets), 'position')

    def stop(self, *, fault: bool = False) -> None:
        """
        取消位置发送并尝试制动，保留连接；失败不转零力矩或失能.

        STOPPED 仅表示软件轨迹已取消，不证明实际已制动或锁定。
        """
        active = self.state in ('REQUESTING', 'HOLDING', 'RUNNING', 'RETURNING')
        self._epoch += 1
        self._move = None
        self.state = 'FAULT' if fault else 'STOPPED'
        if active and self._api is not None:
            try:
                arm = self._arm()
                mine = arm.get_my_session_id()
                if mine <= 0 or arm.get_session_holder() != mine:
                    return
                targets = arm._construct_custom_motor_msg(
                    self._command_types.BRAKE, [True] * 6,
                )
                self._send(arm._construct_custom_joint_command_msg(targets), 'brake')
            except Exception as error:
                self.last_error += f'；软件制动发送未确认：{error}'

    def close(self) -> None:
        """关闭连接前取消位置发送，不自动返回初始位置."""
        if self._api is not None:
            self.stop()
        try:
            super().close()
        finally:
            self.state = 'DISCONNECTED'
            self._feedback = None
            self.initial_positions = None
            self._nominal_required = False
            self._recovering_limits = False
