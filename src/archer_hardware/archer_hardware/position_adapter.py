"""
独立的 SDK 关节位置控制接入，默认只连接反馈，不自动申请控制权.

调用方需显式申请控制并持续调用 tick；软件制动请求不等同于实体急停。
显式准备时采用官方 SDK 流程；任务中故障取消执行，不自动续发旧目标。
"""

from __future__ import annotations

from importlib import import_module
from threading import Event, RLock
import time
from typing import Any, Iterable, cast

from .hex_adapter import (
    HexFeedbackAdapter, JointFeedback, managed_sdk_class, report_only, verified_sdk,
)
from .sdk_cycle import SdkCycleSender
from .sdk_position import isolated_arm_class
from .position_plan import (
    ContinuousPositionTrajectory, JointPositionMove, JointPositionTrajectory,
    PairedTrajectoryStart, PositionLimits,
    limit_preparation_complete, limit_preparation_goal, positive,
)


class HexPositionAdapter(HexFeedbackAdapter):
    """每个实例管理一只臂，不接收 MIT、力矩或速度模式指令."""

    def __init__(
        self, url: str, limits: PositionLimits, *, control_hz: int = 250,
        max_excursion: float = 0.05, tracking_tolerance: float = 0.10,
        readiness_timeout: float = 10.0,
    ) -> None:
        """要求显式传入限位，小幅试运行默认每关节最多变化 0.05 rad."""
        super().__init__(url, stale_timeout=0.1)
        if type(control_hz) is not int or control_hz not in (250, 500):
            raise ValueError('当前接入层只接受 250 或 500 Hz 的目标发送频率')
        self.limits = limits
        self.control_hz = control_hz
        self.max_excursion = positive(max_excursion, 'max_excursion')
        self.tracking_tolerance = positive(tracking_tolerance, 'tracking_tolerance')
        self.readiness_timeout = positive(readiness_timeout, 'readiness_timeout')
        if self.readiness_timeout > 30.0:
            raise ValueError('readiness_timeout 不能超过 30 秒')
        self.readiness_stage = 'IDLE'
        self._calibration_sent = False
        self._initialization_seen = False
        self._startup_clear_source_ns: int | None = None
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
        self._trajectory_side: str | None = None
        self._delivery_generation: int | None = None
        self._delivery_event: Event | None = None
        self.trajectory_start_at: float | None = None
        self.trajectory_observed_start_at: float | None = None
        self._native: SdkCycleSender | None = None
        self._native_error = ''
        self._command_types: Any = None
        self._epoch = 0
        self._requested_at = 0.0
        self._last_tick = 0.0
        self._move_started = 0.0
        self._settled_since: float | None = None
        self._nominal_required = False
        self._recovering_limits = False
        self._limit_preparation = False
        self._control_period_lock = RLock()
        self._control_period_enabled = False

    def connect(self) -> None:
        """建立反馈连接；显式申请控制时才启动有准入过滤的官方控制周期."""
        if self.state != 'DISCONNECTED':
            raise RuntimeError('需要先关闭原连接')
        module = verified_sdk()
        base = cast(Any, module.HexDeviceApi)
        managed_base = managed_sdk_class(base)
        arm_type = isolated_arm_class()
        owner = self
        self._control_period_enabled = False
        self._command_types = import_module('hex_device.motor_base').CommandType
        self._native = SdkCycleSender(
            self._check_dispatch, self._native_failure, lambda: self._epoch, self._command_types,
        )
        self._native.sample_position = self._stream_position_sample

        class PositionAPI(managed_base):
            """替换机械臂类并保留附属设备反馈，控制下行经项目核对."""

            def _register_device_class(self, device_class: Any) -> None:
                super()._register_device_class(
                    arm_type if device_class is owner._arm_type else device_class,
                )

            def _start_device_periodic_task(self, device_id: int) -> None:
                with owner._control_period_lock:
                    arm = self._device_id_map.get(device_id)
                    if isinstance(arm, arm_type):
                        arm._project_position_error = owner._native_failure
                        if owner._control_period_enabled:
                            super()._start_device_periodic_task(device_id)

            def _project_start_control_periodic(self) -> None:
                """显式准备后才启动官方控制周期；接收反馈不依赖此周期。"""
                with owner._control_period_lock:
                    if owner.state != 'REQUESTING':
                        raise RuntimeError('只有显式准备控制时才能启动SDK控制周期')
                    loop = getattr(self, '_HexDeviceApi__loop', None)
                    if loop is None or not loop.is_running():
                        raise RuntimeError('SDK事件循环未就绪，不能启动控制周期')
                    devices = [(i, d) for i, d in self._device_id_map.items()
                               if isinstance(d, arm_type)]
                    if len(devices) != 1:
                        raise RuntimeError('需要唯一的六轴机械臂才能启动控制周期')
                    owner._control_period_enabled = True
                    for device_id, _ in devices:
                        if device_id not in self._device_tasks:
                            self._start_device_periodic_task(device_id)

            async def _send_down_message(self, message: Any) -> None:
                if self._HexDeviceApi__is_closing.is_set():
                    return
                if report_only(message):
                    await super()._send_down_message(message)
                elif owner._native is not None:
                    arms = [d for d in self.device_list if isinstance(d, arm_type)]
                    if len(arms) == 1:
                        await owner._native.dispatch(self, base, arms[0], message)

        self._arm_type = import_module('hex_device.arm').Arm
        self._native_error = ''
        self._epoch += 1
        self._api = PositionAPI(
            ws_url=self.url, control_hz=self.control_hz, enable_kcp=False,
        )
        self.last_error = ''
        self.state = 'CONNECTED'
        self.readiness_stage = 'CONNECTED'

    def _native_failure(self, reason: str) -> None:
        """SDK 线程锁存故障并缓存 BRAKE，不等待自身线程或自动恢复运动."""
        if self._native_error or self.state in ('DISCONNECTED', 'CONNECTED'):
            return
        self._native_error = reason
        self.last_error = reason
        self.stop(fault=True, wait=False)

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

    def _startup_timeout_only(self) -> bool:
        """只识别厂商可远程清除的API通信超时，不把其他停车当作可恢复状态."""
        detail = self._arm().get_parking_stop_detail()
        category = import_module('hex_device.generated.public_api_types_pb2').ParkingStopCategory
        return (detail.category == category.PscAPICommunicationTimeout
                and detail.reason == 'API communication timeout'
                and detail.is_remotely_clearable is True)

    def _health(
        self, *, require_calibrated: bool = True, allow_startup_timeout: bool = False,
    ) -> tuple[bool, bool, bool]:
        arm = self._arm()
        # SDK 的 type 26 是只读 HelloArcher；不能因六轴反馈正常而允许控制。
        if arm.robot_type not in (16, 17, 25):
            raise RuntimeError(
                f'设备型号 {arm.robot_type} 不在已核对的位置控制型号中；'
                '只读 Hello 型号禁止申请控制'
            )
        if arm.get_parking_stop_detail().ListFields() and not (
            allow_startup_timeout and self.state in ('CONNECTED', 'REQUESTING')
            and self._startup_timeout_only()
        ):
            raise RuntimeError('机械臂报告停车故障；仅启动准备允许清除指定API通信超时')
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
                '需要重新完成控制准备'
            )
        if mine <= 0:
            raise RuntimeError('尚未取得有效会话编号')
        if holder not in (0, mine):
            raise RuntimeError('控制权属于其他会话，不抢占')
        return holder == mine, initialized is True, calibrated is True

    def _check_dispatch(self, message: Any, kind: str) -> None:
        """发布缓存及官方真实发送前都复核会话、故障、反馈与双臂取消."""
        if (kind == 'position' and self._trajectory_gate is not None
                and self._trajectory_gate.cancelled):
            raise RuntimeError('双臂轨迹已取消，禁止发送排队的位置消息')
        permitted = (
            kind == 'clear_api_timeout' and self.state == 'REQUESTING'
            and self.readiness_stage == 'CLEARING_API_TIMEOUT'
            and self._startup_clear_source_ns is not None
            or kind == 'initialize' and self.state == 'REQUESTING'
            and self.readiness_stage != 'CLEARING_API_TIMEOUT'
            or kind == 'calibrate' and self.state == 'REQUESTING'
            and self._calibration_sent
            or kind == 'heartbeat' and self.state == 'REQUESTING'
            or kind == 'position' and self.state in (
                'HOLDING', 'ARMED', 'RUNNING',
            )
            or kind == 'brake' and self.state in ('STOPPED', 'FAULT')
        )
        if not permitted:
            raise RuntimeError('当前状态不允许下发此类消息')
        # 通信队列等待期间设备状态可能变化；下发前重新核对，不复用先前结论。
        if kind == 'brake':
            arm = self._arm()
            with arm._status_lock:
                holder, mine = arm._session_holder, arm._my_session_id
            if mine <= 0 or holder != mine:
                raise RuntimeError('下发制动前控制权已改变，不向其他会话发送')
            # 制动用于故障处理，不因电机故障或calibrated失效而阻止尝试。
        else:
            owned, initialized, calibrated = self._health(
                require_calibrated=kind == 'position',
                allow_startup_timeout=kind == 'clear_api_timeout',
            )
            frame = self._feedback
            if frame is None or not 0 <= (
                time.perf_counter_ns() - frame.source_ns
            ) / 1e9 <= self.stale_timeout:
                raise RuntimeError('下发前缺少新鲜关节反馈，消息作废')
            self.limits.check_feedback(frame.positions)
            if kind == 'clear_api_timeout':
                if (not self._startup_timeout_only()
                        or max(abs(v) for v in frame.velocities) > .02
                        or self._request_positions is None
                        or max(abs(a - b) for a, b in zip(
                            frame.positions, self._request_positions,
                        )) > .01):
                    raise RuntimeError('清除通信超时前状态或静止条件已改变')
                outer = message.arm_command
                shared = outer.arm_shared_command
                if ({f.name for f, _ in message.ListFields()} != {'arm_command'}
                        or {f.name for f, _ in outer.ListFields()} != {'arm_shared_command'}
                        or {f.name for f, _ in shared.ListFields()} != {'clear_parking_stop'}
                        or shared.clear_parking_stop is not True):
                    raise RuntimeError('启动清除只能包含单独的clear_parking_stop=true')
            elif kind == 'position':
                if not owned or not initialized:
                    raise RuntimeError('下发位置前控制权或初始化状态已改变')
                if self._nominal_required:
                    self.limits.check_positions(frame.positions)
            elif kind == 'calibrate' and (not owned or calibrated):
                raise RuntimeError('下发校准前控制权或校准状态已改变')
            elif kind == 'heartbeat':
                if not owned:
                    raise RuntimeError('会话心跳不能在失去控制权后继续发送')
                command = message.arm_command.arm_exclusive_command
                control = command.arm_api_control_command
                if (command.WhichOneof('exclusive_command') != 'arm_api_control_command'
                        or {f.name for f, _ in control.ListFields()} != {'motor_targets'}
                        or len(control.motor_targets.targets) != 0):
                    raise RuntimeError('会话心跳必须只有空电机目标，禁止任何运动模式')

    def _send(self, message: Any, kind: str) -> Event:
        """只发布到官方 SDK 缓存，不运行项目传输协程或周期发送线程."""
        if self._native is None:
            raise RuntimeError('官方 SDK 发送器未连接')
        self._arm()._project_position_timing = self._native.position_timing
        self._arm()._project_position_encoder = self._native.encode_position
        event = self._native.publish(self._arm(), message, kind, self._target)
        if kind == 'brake' and not event.wait(.05):
            raise RuntimeError('BRAKE 未在50ms内确认底层发送，仍将继续关闭连接')
        return event

    def _stream_position_sample(self) -> tuple[float, ...] | None:
        """完整文件在SDK构包时按共同时钟采样，同时复核实际反馈误差。"""
        move, gate, side = self._move, self._trajectory_gate, self._trajectory_side
        if (self.state != 'RUNNING' or not isinstance(move, ContinuousPositionTrajectory)
                or not move.preserve_path or gate is None or side is None
                or gate.playback_status()['stream_period_s'] is None):
            return None
        elapsed, _, _ = gate.playback(side, None)
        target = move.sample(elapsed)
        frame = self._feedback
        if frame is None or not 0. <= (
            time.perf_counter_ns() - frame.source_ns
        ) / 1e9 <= self.stale_timeout:
            raise RuntimeError('SDK周期取样前缺少新鲜反馈')
        self.limits.check_positions(frame.positions)
        tolerance = (min(move.tracking_error_at(elapsed), self.tracking_tolerance)
                     if move.joint_error_rad else self.tracking_tolerance)
        if max(abs(a - b) for a, b in zip(frame.positions, target)) > tolerance:
            raise RuntimeError('SDK周期取样目标与实际反馈偏差超限')
        return target

    def request_control(self) -> None:
        """在新鲜、静止反馈和无故障状态下显式申请控制权."""
        if self.state != 'CONNECTED':
            raise RuntimeError('仅允许从 CONNECTED 状态申请控制')
        frame = self._fresh_feedback()
        self._health(require_calibrated=False, allow_startup_timeout=True)
        if max(abs(v) for v in frame.velocities) > 0.02:
            raise RuntimeError('机械臂仍在运动，不能申请初始位置保持')
        self.state = 'REQUESTING'
        self.readiness_stage = 'WAITING_SESSION'
        self._calibration_sent = False
        self._initialization_seen = False
        self._request_positions = frame.positions
        self._requested_at = time.monotonic()
        self._startup_clear_source_ns = None
        try:
            arm = self._arm()
            if arm.get_parking_stop_detail().ListFields():
                self.readiness_stage = 'CLEARING_API_TIMEOUT'
                self._startup_clear_source_ns = frame.source_ns
                self._send(arm._construct_clear_parking_stop_message(), 'clear_api_timeout')
            else:
                self._send(arm._construct_init_message(True), 'initialize')
            start_periodic = getattr(self._api, '_project_start_control_periodic', None)
            if start_periodic is not None:
                start_periodic()
        except Exception as error:
            self.last_error = str(error)
            self.stop(fault=True)
            raise

    def move_to(self, goal: Iterable[object], duration: float) -> None:
        """从当前实际静止位置规划单段小幅移动，不抢占正在运行的轨迹."""
        frame = self._single_move_feedback()
        self._start_move(frame, goal, duration, self.max_excursion)

    def recover_limits(self, duration: float) -> None:
        """显式准备入限及边界余量，不改读取容差或普通移动的精度标准."""
        frame = self._single_move_feedback()
        goal = limit_preparation_goal(self.limits, frame.positions)
        if goal == frame.positions:
            return
        self._start_move(
            frame, goal, duration,
            min(self.max_excursion, 0.02),
        )
        self._limit_preparation = True

    def _single_move_feedback(self) -> JointFeedback:
        """单点移动和限位收回共用新鲜、静止及有效控制会话检查."""
        if self.state != 'HOLDING':
            raise RuntimeError('需要先获得控制权并处于 HOLDING 状态')
        frame = self._fresh_feedback()
        owned, initialized, _ = self._health()
        if not owned or not initialized:
            raise RuntimeError('控制会话无效')
        if max(abs(v) for v in frame.velocities) > 0.02:
            raise RuntimeError('需要等待机械臂静止后再开始新轨迹')
        return frame

    def _start_move(
        self, frame: JointFeedback, goal: Iterable[object], duration: float,
        max_excursion: float,
    ) -> None:
        """验证并安装单段计划；不重读反馈或改变原始初始位置记录."""
        move = JointPositionMove.prepare(
            frame.positions, goal, duration, self.limits, max_excursion,
        )
        self._recovering_limits = not self.limits.within_nominal(frame.positions)
        self._nominal_required = not self._recovering_limits
        self._move = move
        self._limit_preparation = False
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
        self._limit_preparation = False
        self._trajectory_gate = None
        self._target = frame.positions
        self._move_started = time.monotonic()
        self._settled_since = None
        self._recovering_limits = False
        self._nominal_required = True
        self.state = 'RUNNING'

    @property
    def commanded_positions(self) -> tuple[float, ...] | None:
        """只在保持状态公开已发送目标，不能用最新反馈冒充指令锚点."""
        if self.state != 'HOLDING':
            return None
        if self._native is not None and self._native.last_position != self._target:
            return None
        return self._target

    def arm_continuous(
        self, plan: ContinuousPositionTrajectory, side: str, gate: PairedTrajectoryStart,
    ) -> None:
        """接收已预检连续轨迹，核对实际起点后等待双臂共同启动，不进行全程重算."""
        if side not in ('left', 'right') or gate.cancelled:
            raise ValueError('连续轨迹侧别错误或共同启动已取消')
        frame = self._trajectory_feedback()
        move = plan.aligned_start(frame.positions, self.limits, self.max_excursion)
        if move.joint_error_rad and self._target != move.initial:
            raise ValueError('保持指令目标与已检路径起点不同，必须重新预检')
        self._move = move
        self._limit_preparation = False
        self._target = move.initial if move.joint_error_rad else frame.positions
        self._trajectory_gate = gate
        self._trajectory_side = side
        self._delivery_generation = None
        self._delivery_event = None
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
            if self._native_error:
                raise RuntimeError(self._native_error)
            self._tick()
        except Exception as error:
            self.last_error = str(error)
            self.stop(fault=True)
            raise
        return self.state

    def _tick(self) -> None:
        now = time.monotonic()
        frame = self._fresh_feedback()
        clearing = self.state == 'REQUESTING' and self.readiness_stage == 'CLEARING_API_TIMEOUT'
        owned, initialized, calibrated = self._health(
            require_calibrated=False, allow_startup_timeout=clearing,
        )
        if self.state == 'REQUESTING':
            if now - self._requested_at > self.readiness_timeout:
                waiting = []
                if clearing:
                    waiting.append('API通信超时停车尚未确认清除')
                if not owned:
                    waiting.append('尚未取得控制会话')
                if not initialized:
                    waiting.append('API 初始化尚未确认')
                if not calibrated:
                    waiting.append('设备尚未报告 calibrated=true')
                raise RuntimeError('等待控制就绪超时：' + '；'.join(waiting))
            if max(abs(v) for v in frame.velocities) > 0.02:
                raise RuntimeError('等待控制就绪期间机械臂出现运动')
            assert self._request_positions is not None
            if max(abs(a - b) for a, b in zip(
                frame.positions, self._request_positions,
            )) > 0.01:
                raise RuntimeError('等待控制就绪期间关节漂移超过 0.01 rad')
            if clearing:
                if (self._arm().get_parking_stop_detail().ListFields()
                        or self._startup_clear_source_ns is None
                        or frame.source_ns <= self._startup_clear_source_ns):
                    return
                # 只有更新的实测帧确认停车已消失后才申请控制，不保留排队的清除请求。
                if self._native is not None:
                    self._native.cancel()
                self.readiness_stage = 'WAITING_SESSION'
                self._send(self._arm()._construct_init_message(True), 'initialize')
                return
            if self._calibration_sent and (
                not owned or (self._initialization_seen and not initialized)
            ):
                raise RuntimeError('API 校准请求后控制会话或初始化状态丢失')
            if not owned:
                self.readiness_stage = 'WAITING_SESSION'
                return
            self._initialization_seen = self._initialization_seen or initialized
            if not calibrated:
                self.readiness_stage = 'WAITING_CALIBRATION'
                if not self._calibration_sent:
                    # 发送前锁存，传输失败也不自动重试未知的校准操作。
                    self._calibration_sent = True
                    self._send(self._arm()._construct_calibrate_message(), 'calibrate')
                else:
                    self._send(self._arm()._construct_empty_message(), 'heartbeat')
                return
            if not initialized:
                self.readiness_stage = 'WAITING_INITIALIZATION'
                self._send(self._arm()._construct_empty_message(), 'heartbeat')
                return
            arm = self._arm()
            if self._native is not None:
                lower = [min(lo, q) for lo, q in zip(self.limits.lower, frame.positions)]
                upper = [max(hi, q) for hi, q in zip(self.limits.upper, frame.positions)]
                config = {'name': 'project_position_limits', 'dof_num': 'six_axis',
                          'motor_model': [128] * 6,
                          'joints': [{'joint_name': f'joint_{i + 1}',
                                      'joint_limit': [lo, hi, -v, v, 0., 0.]}
                                     for i, (lo, hi, v) in enumerate(zip(
                                         lower, upper, self.limits.max_velocity))]}
                if not arm.reload_arm_config_from_dict(config):
                    raise RuntimeError('本臂 SDK 位置配置加载失败')
                arm.set_initial_positions(list(frame.positions))
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
            raise RuntimeError('位置目标更新间隔超过 40 ms')
        if self.state == 'ARMED':
            assert self._trajectory_gate is not None
            start = self._trajectory_gate.start_time()
            if (isinstance(self._move, ContinuousPositionTrajectory)
                    and self._move.preserve_path
                    and (self._target != self._move.initial or max(
                        abs(a - b) for a, b in zip(frame.positions, self._move.initial)
                    ) > self._move.joint_error_rad)):
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
        playback_generation: int | None = None
        tracking_elapsed = (self._move.duration if self.state == 'HOLDING'
                            and isinstance(self._move, ContinuousPositionTrajectory) else 0.)
        if self.state == 'RUNNING':
            assert self._move is not None
            elapsed = now - self._move_started
            terminal_age: float | None = elapsed - self._move.duration
            paced = (self._trajectory_gate is not None
                     and self._trajectory_gate.delivery_paced)
            if paced:
                assert self._trajectory_gate is not None and self._trajectory_side is not None
                acknowledged = (self._delivery_generation if self._delivery_event is not None
                                and self._delivery_event.is_set() else None)
                tracking_ratio = None
                if self._trajectory_gate.following_feedback:
                    move = self._move
                    if (not isinstance(move, ContinuousPositionTrajectory)
                            or not move.joint_error_rad):
                        raise RuntimeError('实际跟踪门控需要已检完整路径误差界限')

                    def ratio_at(reference: float) -> float:
                        # 锁内取得共用参考时间，避免另一臂推进后仍用上一代目标作比较。
                        target = move.sample(reference)
                        tolerance = min(move.tracking_error_at(reference), self.tracking_tolerance)
                        return max(abs(a - b) for a, b in zip(frame.positions, target)) / tolerance

                    tracking_ratio = ratio_at

                elapsed, playback_generation, terminal_age = self._trajectory_gate.playback(
                    self._trajectory_side, acknowledged, tracking_ratio=tracking_ratio)
            self._target = self._move.sample(elapsed)
            tracking_elapsed = elapsed
            if elapsed >= self._move.duration and terminal_age is not None:
                error = max(abs(a - b) for a, b in zip(frame.positions, self._move.goal))
                stopped = max(abs(v) for v in frame.velocities) <= 0.02
                recovered = (not self._recovering_limits
                             or self.limits.within_nominal(frame.positions))
                # 仅显式入限准备按实际范围和余量准入，普通移动与完整轨迹保持原精度。
                if self._limit_preparation:
                    assert isinstance(self._move, JointPositionMove)
                    arrived = limit_preparation_complete(
                        self.limits, self._move.start, self._move.goal, frame.positions,
                    ) and error <= self.tracking_tolerance
                elif (isinstance(self._move, ContinuousPositionTrajectory)
                      and self._move.preserve_path and self._move.joint_error_rad):
                    # 完整文件到位条件沿用已做几何预检的误差界限，不叠加另一套精度。
                    arrived = error <= min(self._move.joint_error_rad, self.tracking_tolerance)
                else:
                    arrived = error <= min(0.003, self.tracking_tolerance)
                if arrived and stopped and recovered:
                    if self._settled_since is None:
                        self._settled_since = now
                    if now - self._settled_since >= 0.2:
                        self.state = 'HOLDING'
                        self._recovering_limits = False
                        self._limit_preparation = False
                        # 入限过程允许既有限定的初值误差，完成后恢复严格限位。
                        # 不能因途中一帧刚过边界便切换，造成边界量化波动误停。
                        self._nominal_required = True
                else:
                    self._settled_since = None
                if self.state == 'RUNNING' and terminal_age > 2.0:
                    raise RuntimeError('目标位置未在允许时间内到位')
        assert self._target is not None
        path_error = max(abs(a - b) for a, b in zip(frame.positions, self._target))
        if (isinstance(self._move, ContinuousPositionTrajectory)
                and self._move.joint_error_rad
                and path_error > self._move.tracking_error_at(tracking_elapsed)):
            raise RuntimeError('实际反馈超出完整路径几何误差界限，取消后续位置发送')
        if max(abs(a - b) for a, b in zip(frame.positions, self._target)) > (
            self.tracking_tolerance
        ):
            raise RuntimeError('实际位置与目标位置偏差超限')
        arm = self._arm()
        # 复用官方弧度到编码器转换；限位和限速由本实例整段解析预检完成。
        targets = arm._construct_custom_motor_msg(
            self._command_types.POSITION, list(self._target),
        )
        event = self._send(arm._construct_custom_joint_command_msg(targets), 'position')
        if self.state == 'RUNNING' and playback_generation is not None:
            self._delivery_generation = playback_generation
            self._delivery_event = event
        # 记录成功更新缓存的时刻；上一轮计算或锁等待不能冒充后续调用空档。
        # SDK发送器仍独立检查目标40ms有效期，不因本时间戳更新而延长旧请求。
        self._last_tick = time.monotonic()

    def stop(self, *, fault: bool = False, wait: bool = True) -> None:
        """
        取消位置发送并尝试制动，保留连接；失败不转零力矩或失能.

        STOPPED 仅表示软件轨迹已取消，不证明实际已制动或锁定。
        """
        active = self.state in ('REQUESTING', 'HOLDING', 'ARMED', 'RUNNING')
        if not active:
            # 重复停止不能清掉正在由 SDK 周期交付的 BRAKE，也不能降级故障。
            if fault and self.state == 'STOPPED':
                self.state = 'FAULT'
                self.readiness_stage = 'FAULT'
            return
        if self._trajectory_gate is not None:
            self._trajectory_gate.cancel()
        self._epoch += 1
        if self._native is not None:
            self._native.cancel()
        self._move = None
        self._limit_preparation = False
        self._delivery_generation = None
        self._delivery_event = None
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
                message = arm._construct_custom_joint_command_msg(targets)
                if not wait and self._native is not None:
                    self._native.publish(arm, message, 'brake', None)
                else:
                    self._send(message, 'brake')
            except Exception as error:
                self.last_error += f'；软件制动发送未确认：{error}'

    def close(self) -> None:
        """关闭连接前取消位置发送，不自动返回初始位置."""
        if self._api is not None:
            self.stop()
        if self._native is not None:
            # SDK线程故障已缓存BRAKE时，stop()会保留FAULT，不再重复发布。
            # 先给原生周期一次有界的发送机会，再清空缓存和关闭准入。
            if not self._native.wait_pending_brake():
                self.last_error += '；关闭前缓存BRAKE的底层发送未确认'
            self._native.close()
        try:
            super().close()
        finally:
            self._control_period_enabled = False
            self.state = 'DISCONNECTED'
            self._feedback = None
            self.initial_positions = None
            self._nominal_required = False
            self._recovering_limits = False
            self._limit_preparation = False
            self._trajectory_side = None
            self._delivery_generation = None
            self._delivery_event = None
            self._move = None
            self._target = None
            self._trajectory_gate = None
            self._request_positions = None
            self._startup_clear_source_ns = None
            self._calibration_sent = False
            self._initialization_seen = False
            self._settled_since = None
            self.trajectory_start_at = None
            self.trajectory_observed_start_at = None
            self.readiness_stage = 'DISCONNECTED'
