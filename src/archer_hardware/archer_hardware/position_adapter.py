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
    HexFeedbackAdapter, JointFeedback, managed_sdk_class, report_only, verified_sdk,
)
from .position_plan import (
    ContinuousPositionTrajectory, JointPositionMove, JointPositionTrajectory,
    PairedTrajectoryStart, PositionLimits,
    positive,
)


class HexPositionAdapter(HexFeedbackAdapter):
    """每个实例管理一只臂，不接收 MIT、力矩或速度模式指令."""

    def __init__(
        self, url: str, limits: PositionLimits, *, control_hz: int = 250,
        max_excursion: float = 0.05, tracking_tolerance: float = 0.10,
        allow_api_calibration: bool = False, readiness_timeout: float = 10.0,
    ) -> None:
        """要求显式传入限位，小幅试运行默认每关节最多变化 0.05 rad."""
        super().__init__(url, stale_timeout=0.1)
        if type(control_hz) is not int or control_hz not in (250, 500):
            raise ValueError('当前接入层只接受 250 或 500 Hz 的目标发送频率')
        self.limits = limits
        self.control_hz = control_hz
        self.max_excursion = positive(max_excursion, 'max_excursion')
        self.tracking_tolerance = positive(tracking_tolerance, 'tracking_tolerance')
        if type(allow_api_calibration) is not bool:
            raise ValueError('allow_api_calibration 必须为布尔值')
        self.allow_api_calibration = allow_api_calibration
        self.readiness_timeout = positive(readiness_timeout, 'readiness_timeout')
        if self.readiness_timeout > 30.0:
            raise ValueError('readiness_timeout 不能超过 30 秒')
        self.readiness_stage = 'IDLE'
        self._calibration_sent = False
        self._request_positions: tuple[float, ...] | None = None
        self.state = 'DISCONNECTED'
        self.last_error = ''
        self.initial_positions: tuple[float, ...] | None = None
        self._feedback: JointFeedback | None = None
        self._target: tuple[float, ...] | None = None
        self._move: (
            JointPositionMove | JointPositionTrajectory | ContinuousPositionTrajectory | None
        ) = None
        self._trajectory_gate: PairedTrajectoryStart | None = None
        self.trajectory_start_at: float | None = None
        self.trajectory_observed_start_at: float | None = None
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

        managed_base = managed_sdk_class(base)

        class PositionAPI(managed_base):
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
        self.readiness_stage = 'CONNECTED'

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

    def _health(self, *, require_calibrated: bool = True) -> tuple[bool, bool, bool]:
        arm = self._arm()
        # SDK 的 type 26 是只读 HelloArcher；不能因六轴反馈正常而允许控制。
        if arm.robot_type not in (16, 17, 25):
            raise RuntimeError(
                f'设备型号 {arm.robot_type} 不在已核对的位置控制型号中；'
                '只读 Hello 型号禁止申请控制'
            )
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
        if require_calibrated and calibrated is not True:
            raise RuntimeError(
                '设备尚未报告 calibrated=true，禁止发送位置目标；'
                '本接口不会自动校准'
            )
        if mine <= 0:
            raise RuntimeError('尚未取得有效会话编号')
        if holder not in (0, mine):
            raise RuntimeError('控制权属于其他会话，不抢占')
        return holder == mine, initialized is True, calibrated is True

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
            if (kind == 'position' and self._trajectory_gate is not None
                    and self._trajectory_gate.cancelled):
                raise RuntimeError('双臂轨迹已取消，禁止发送排队的位置消息')
            permitted = (
                kind == 'initialize' and self.state == 'REQUESTING'
                or kind == 'calibrate' and self.state == 'REQUESTING'
                and self.allow_api_calibration and self._calibration_sent
                or kind == 'position' and self.state in (
                    'HOLDING', 'ARMED', 'RUNNING',
                )
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
        self._health(require_calibrated=False)
        if max(abs(v) for v in frame.velocities) > 0.02:
            raise RuntimeError('机械臂仍在运动，不能申请初始位置保持')
        self.state = 'REQUESTING'
        self.readiness_stage = 'WAITING_SESSION'
        self._calibration_sent = False
        self._request_positions = frame.positions
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
        owned, initialized, _ = self._health()
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
        self._trajectory_gate = None
        self._move_started = time.monotonic()
        self._settled_since = None
        self.state = 'RUNNING'

    def _trajectory_feedback(self) -> JointFeedback:
        """两类完整轨迹共用起步条件，要求控制会话有效及新鲜静止反馈."""
        if self.state != 'HOLDING':
            raise RuntimeError('完整轨迹需要先处于 HOLDING 状态')
        frame = self._fresh_feedback()
        owned, initialized, _ = self._health()
        if not owned or not initialized:
            raise RuntimeError('完整轨迹开始前控制会话无效')
        if max(abs(v) for v in frame.velocities) > 0.02:
            raise RuntimeError('完整轨迹开始前机械臂必须静止')
        return frame

    def follow_trajectory(
        self, times_ns: Iterable[int], positions: Iterable[Iterable[object]],
    ) -> None:
        """执行版本一逐点停车轨迹，保留原有接口及预检行为."""
        frame = self._trajectory_feedback()
        move = JointPositionTrajectory.prepare(
            times_ns, positions, self.limits, self.max_excursion,
            actual_start=frame.positions,
        )
        self._move = move
        self._trajectory_gate = None
        self._target = frame.positions
        self._move_started = time.monotonic()
        self._settled_since = None
        self._recovering_limits = False
        self._nominal_required = True
        self.state = 'RUNNING'

    def arm_continuous(
        self, plan: ContinuousPositionTrajectory, side: str, gate: PairedTrajectoryStart,
    ) -> None:
        """接收已预检连续轨迹，核对实际起点后等待双臂共同启动，不进行全程重算."""
        if side not in ('left', 'right') or gate.cancelled:
            raise ValueError('连续轨迹侧别错误或共同启动已取消')
        frame = self._trajectory_feedback()
        move = plan.aligned_start(frame.positions, self.limits, self.max_excursion)
        self._move = move
        self._target = frame.positions
        self._trajectory_gate = gate
        self._settled_since = None
        self._recovering_limits = False
        self._nominal_required = True
        self.trajectory_start_at = None
        self.trajectory_observed_start_at = None
        self.state = 'ARMED'
        gate.ready(side)

    def arm_trajectory(
        self, times_ns: tuple[int, ...], positions: tuple[tuple[float, ...], ...],
        side: str, gate: PairedTrajectoryStart,
    ) -> None:
        """在线程内重新预检后保持实际位置，双方准备成功才允许共同启动."""
        self.follow_trajectory(times_ns, positions)
        self._trajectory_gate = gate
        self.trajectory_start_at = None
        self.trajectory_observed_start_at = None
        self.state = 'ARMED'
        gate.ready(side)

    def tick(self) -> str:
        """由单一调用线程按目标频率执行；异常锁存并请求软件制动."""
        if self.state not in ('REQUESTING', 'HOLDING', 'ARMED', 'RUNNING'):
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
        owned, initialized, calibrated = self._health(require_calibrated=False)
        if self.state == 'REQUESTING':
            if now - self._requested_at > self.readiness_timeout:
                waiting = []
                if not owned:
                    waiting.append('尚未取得控制会话')
                if not initialized:
                    waiting.append('API 初始化尚未确认')
                if not calibrated:
                    waiting.append('设备尚未报告 calibrated=true')
                    if not self.allow_api_calibration:
                        waiting.append('API 校准未获显式允许')
                raise RuntimeError('等待控制就绪超时：' + '；'.join(waiting))
            if max(abs(v) for v in frame.velocities) > 0.02:
                raise RuntimeError('等待控制就绪期间机械臂出现运动')
            assert self._request_positions is not None
            if max(abs(a - b) for a, b in zip(
                frame.positions, self._request_positions,
            )) > 0.01:
                raise RuntimeError('等待控制就绪期间关节漂移超过 0.01 rad')
            if self._calibration_sent and (not owned or not initialized):
                raise RuntimeError('API 校准请求后控制会话或初始化状态丢失')
            if not owned:
                self.readiness_stage = 'WAITING_SESSION'
                return
            if not initialized:
                self.readiness_stage = 'WAITING_INITIALIZATION'
                return
            if not calibrated:
                if not self.allow_api_calibration:
                    self.readiness_stage = 'CALIBRATION_PERMISSION_REQUIRED'
                elif not self._calibration_sent:
                    # 发送前锁存，传输失败也不自动重试未知的校准操作。
                    self._calibration_sent = True
                    self.readiness_stage = 'WAITING_CALIBRATION'
                    self._send(self._arm()._construct_calibrate_message(), 'calibrate')
                return
            self.initial_positions = frame.positions
            self._target = frame.positions
            self._last_tick = now
            self.state = 'HOLDING'
            self.readiness_stage = 'READY'
        if not owned or not initialized:
            raise RuntimeError('控制权已丢失或控制模式未初始化')
        if not calibrated:
            raise RuntimeError('设备 calibrated 状态失效，停止发送位置目标')
        if self._trajectory_gate is not None and self._trajectory_gate.cancelled:
            raise RuntimeError('双臂轨迹已取消，停止后续位置发送')
        if now - self._last_tick > 0.04:
            raise RuntimeError('调用 tick 的间隔超过 40 ms')
        self._last_tick = now
        if self.state == 'ARMED':
            assert self._trajectory_gate is not None
            start = self._trajectory_gate.start_time()
            if (isinstance(self._move, ContinuousPositionTrajectory)
                    and self._move.preserve_path and frame.positions != self._move.initial):
                raise RuntimeError('完整路径等待启动时起点变化，取消本次运行并重新预检')
            if (max(abs(v) for v in frame.velocities) > 0.02
                    or self._target is None or max(abs(a - b) for a, b in zip(
                        frame.positions, self._target,
                    )) > 0.005):
                raise RuntimeError('双臂等待启动时实际位置或速度发生变化')
            if start is not None and now >= start:
                if now - start > 0.02:
                    raise RuntimeError('双臂轨迹启动迟到超过 20 ms，取消本次运行')
                self._move_started = start
                self.trajectory_start_at = start
                self.trajectory_observed_start_at = now
                self.state = 'RUNNING'
        if self.state == 'RUNNING':
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
                if self.state == 'RUNNING' and elapsed > self._move.duration + 2.0:
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
        active = self.state in ('REQUESTING', 'HOLDING', 'ARMED', 'RUNNING')
        if self._trajectory_gate is not None:
            self._trajectory_gate.cancel()
        self._epoch += 1
        self._move = None
        self.state = 'FAULT' if fault else 'STOPPED'
        self.readiness_stage = self.state
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
            self.readiness_stage = 'DISCONNECTED'
