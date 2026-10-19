"""以每臂独立线程调度位置接口，服务回调不直接访问 SDK."""

from __future__ import annotations

from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
import time
from typing import Callable, Protocol
from uuid import uuid4

from .hex_adapter import JointFeedback
from .position_plan import ContinuousPositionTrajectory, PairedTrajectoryStart


ALLOWED_STATES = {
    'connect': ('DISCONNECTED',),
    'enable': ('CONNECTED',),
    'move': ('HOLDING',),
    'trajectory': ('HOLDING',),
    'disconnect': ('CONNECTED', 'STOPPED', 'FAULT', 'DISCONNECTED'),
}


class PositionBackend(Protocol):
    """实际 SDK 接口与离线替身共用的最小协议."""

    state: str
    last_error: str
    initial_positions: tuple[float, ...] | None

    def connect(self) -> None:
        """建立反馈连接，不自动使能."""
        ...

    def read_feedback(self) -> JointFeedback | None:
        """读取新鲜反馈，未到首帧时返回空值."""
        ...

    def request_control(self) -> None:
        """显式申请位置控制."""
        ...

    def move_to(self, goal: tuple[float, ...], duration: float) -> None:
        """从实际反馈规划移动."""
        ...

    def arm_trajectory(
        self, times_ns: tuple[int, ...], positions: tuple[tuple[float, ...], ...],
        side: str, gate: PairedTrajectoryStart,
    ) -> None:
        """准备完整轨迹，和另一侧共用启动门控."""
        ...

    def tick(self) -> str:
        """推进控制周期."""
        ...

    def arm_continuous(
        self, plan: ContinuousPositionTrajectory, side: str, gate: PairedTrajectoryStart,
    ) -> None:
        """核对实际起点后准备已预检的连续曲线，不重算全程多项式."""
        ...

    def stop(self, *, fault: bool = False) -> None:
        """取消目标并尽力制动."""
        ...

    def close(self) -> None:
        """关闭连接，不自动运动."""
        ...


@dataclass(frozen=True)
class WorkerSnapshot:
    """供 ROS 线程读取的不可变快照."""

    state: str
    feedback: JointFeedback | None = None
    initial_positions: tuple[float, ...] | None = None
    error: str = ''
    command_id: str = ''
    command: str = ''
    phase: str = ''
    message: str = ''
    readiness_stage: str = ''
    trajectory_start_at: float | None = None
    trajectory_observed_start_at: float | None = None
    commanded_positions: tuple[float, ...] | None = None


@dataclass(frozen=True)
class WorkerCommand:
    """带有效期和停止代次的单次请求."""

    identifier: str
    kind: str
    generation: int
    expires: float
    goal: tuple[float, ...] = ()
    duration: float = 0.0
    times_ns: tuple[int, ...] = ()
    positions: tuple[tuple[float, ...], ...] = ()
    gate: PairedTrajectoryStart | None = None
    continuous: ContinuousPositionTrajectory | None = None


class PositionWorker:
    """一个线程独占一只臂；普通队列至多保留一条，停止走独立事件."""

    def __init__(
        self, side: str, factory: Callable[[], PositionBackend] | None,
        on_fault: Callable[[str], None], *, control_hz: int = 250,
        startup_timeout: float = 10.0,
    ) -> None:
        """构造时不连接、不启动线程，由调用方统一启动."""
        if type(control_hz) is not int or control_hz not in (250, 500):
            raise ValueError('control_hz 仅接受 250 或 500')
        if startup_timeout <= 0:
            raise ValueError('startup_timeout 必须为正数')
        self.side = side
        self.factory = factory
        self.on_fault = on_fault
        self.period = 1.0 / control_hz
        self.startup_timeout = startup_timeout
        self._backend: PositionBackend | None = None
        self._queue: Queue[WorkerCommand] = Queue(maxsize=1)
        self._command_lock = Lock()
        self._snapshot_lock = Lock()
        self._snapshot = WorkerSnapshot('DISCONNECTED' if factory else 'UNCONFIGURED')
        self._wake = Event()
        self._stop = Event()
        self._shutdown = Event()
        self._generation = 0
        self._pending = False
        self._job: WorkerCommand | None = None
        self._phase = ''
        self._message = ''
        self._error = ''
        self._frame: JointFeedback | None = None
        self._connected_at = 0.0
        self._thread = Thread(target=self._run, name=f'{side}_position', daemon=True)

    def start(self) -> None:
        """启动调度线程，仍不建立设备连接."""
        self._thread.start()

    def snapshot(self) -> WorkerSnapshot:
        """读取与 SDK 对象分离的状态快照."""
        with self._snapshot_lock:
            return self._snapshot

    def submit(
        self, kind: str, *, goal: tuple[float, ...] = (), duration: float = 0.0,
        times_ns: tuple[int, ...] = (), positions: tuple[tuple[float, ...], ...] = (),
        gate: PairedTrajectoryStart | None = None,
        continuous: ContinuousPositionTrajectory | None = None,
    ) -> str:
        """异步受理请求；返回编号不表示设备执行成功."""
        if kind not in ALLOWED_STATES:
            raise ValueError('未知控制请求')
        if continuous is not None and (
            kind != 'trajectory' or times_ns or positions
            or not isinstance(continuous, ContinuousPositionTrajectory)
        ):
            raise ValueError('连续对象只能用于轨迹请求，不能混入旧格式点数组')
        with self._command_lock:
            if self._shutdown.is_set():
                raise RuntimeError('线程正在退出')
            if self.factory is None:
                raise RuntimeError('请先填写并确认本臂真实限位，然后重启节点')
            if self._stop.is_set() and kind != 'disconnect':
                raise RuntimeError('停止处理中，请等待状态更新')
            if self._pending:
                raise RuntimeError('已有等待处理的请求，请先查看状态')
            if self.snapshot().state not in ALLOWED_STATES[kind]:
                raise RuntimeError(f'当前状态不允许 {kind} 请求')
            identifier = uuid4().hex[:12]
            command = WorkerCommand(
                identifier, kind, self._generation, time.monotonic() + 0.5,
                goal, duration, times_ns, positions, gate, continuous,
            )
            try:
                self._queue.put_nowait(command)
                self._pending = True
            except Full as error:
                raise RuntimeError('已有等待处理的请求，请先查看状态') from error
        self._wake.set()
        return identifier

    def request_stop(self) -> None:
        """作废等待请求，停止事件无需等待普通队列."""
        with self._command_lock:
            self._generation += 1
            self._stop.set()
            self._pending = False
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
        self._wake.set()

    def shutdown(self, timeout: float = 12.0) -> None:
        """先请求停止再结束线程，不在 ROS 回调线程关闭 SDK."""
        self.request_stop()
        self._shutdown.set()
        self._wake.set()
        if self._thread.ident is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                raise RuntimeError(f'{self.side} SDK 线程未在超时内退出')

    def _publish(self) -> None:
        backend = self._backend
        state = backend.state if backend else ('DISCONNECTED' if self.factory else 'UNCONFIGURED')
        snapshot = WorkerSnapshot(
            state, self._frame,
            backend.initial_positions if backend else None,
            self._error or (backend.last_error if backend else ''),
            self._job.identifier if self._job else '',
            self._job.kind if self._job else '', self._phase, self._message,
            getattr(backend, 'readiness_stage', '') if backend else '',
            getattr(backend, 'trajectory_start_at', None) if backend else None,
            getattr(backend, 'trajectory_observed_start_at', None) if backend else None,
            getattr(backend, 'commanded_positions', None) if state == 'HOLDING' else None,
        )
        with self._snapshot_lock:
            self._snapshot = snapshot

    def _fail(self, error: Exception) -> None:
        self._error = str(error)
        self._phase = 'FAILED'
        self._message = str(error)
        if self._backend is not None:
            try:
                self._backend.stop(fault=True)
            except Exception as stop_error:
                self._error += f'；制动请求异常：{stop_error}'
        self._publish()
        self.on_fault(f'{self.side}: {self._error}')

    def _handle_stop(self) -> None:
        if not self._stop.is_set():
            return
        if self._backend is not None:
            fault = self._backend.state == 'FAULT'
            self._backend.stop(fault=fault)
        if self._phase in ('WAITING', 'RUNNING'):
            self._phase = 'CANCELLED'
            self._message = '已取消后续目标，软件制动请求不代表物理停止确认'
        self._stop.clear()

    def _execute(self, command: WorkerCommand) -> None:
        self._job = command
        self._phase = 'WAITING'
        self._message = '请求已进入执行线程'
        with self._command_lock:
            valid = command.generation == self._generation and not self._stop.is_set()
        if not valid or time.monotonic() > command.expires:
            self._phase = 'REJECTED'
            self._message = '请求已过期或被停止作废'
            return
        current = self._backend.state if self._backend else 'DISCONNECTED'
        if current not in ALLOWED_STATES[command.kind]:
            self._phase = 'REJECTED'
            self._message = f'执行前状态已变化：{current}'
            return
        try:
            if command.kind == 'connect':
                assert self.factory is not None
                self._backend = self.factory()
                self._backend.connect()
                self._connected_at = time.monotonic()
                self._error = ''
                self._frame = None
                self._phase = 'SUCCEEDED'
                self._message = '通信对象已创建，等待新鲜反馈；尚未申请控制'
                return
            backend = self._backend
            if backend is None:
                if command.kind == 'disconnect':
                    self._phase = 'SUCCEEDED'
                    return
                raise RuntimeError('设备尚未连接')
            if command.kind == 'enable':
                backend.request_control()
                self._message = '已申请控制，等待设备确认'
            elif command.kind == 'move':
                backend.move_to(command.goal, command.duration)
                self._phase = 'RUNNING'
            elif command.kind == 'trajectory':
                if command.gate is None:
                    raise ValueError('成对轨迹缺少共同启动门控')
                if command.continuous is None:
                    backend.arm_trajectory(command.times_ns, command.positions,
                                           self.side, command.gate)
                else:
                    backend.arm_continuous(command.continuous, self.side, command.gate)
                self._message = '本臂已准备，等待两臂共同启动'
            elif command.kind == 'disconnect':
                backend.close()
                self._backend = None
                self._frame = None
                self._phase = 'SUCCEEDED'
                self._message = '连接已关闭，未执行归位'
        except ValueError as error:
            if command.kind == 'trajectory' and self._backend is not None:
                # 成对轨迹准备失败需取消双方，不能让另一侧继续等待或执行。
                self._fail(error)
            else:
                self._phase = 'REJECTED'
                self._message = str(error)
        except Exception as error:
            if self._backend and self._backend.state in (
                'REQUESTING', 'HOLDING', 'ARMED', 'RUNNING', 'FAULT',
            ):
                self._fail(error)
            else:
                self._phase = 'REJECTED'
                self._message = str(error)
                self._error = str(error)
                if command.kind == 'connect' and self._backend is not None:
                    try:
                        self._backend.close()
                    except Exception as close_error:
                        self._error += f'；关闭失败：{close_error}'
                    self._backend = None

    def _cycle(self) -> None:
        backend = self._backend
        if backend is None or backend.state == 'DISCONNECTED':
            return
        terminal = backend.state in ('STOPPED', 'FAULT')
        try:
            frame = backend.read_feedback()
            if frame is not None:
                self._frame = frame
            elif time.monotonic() - self._connected_at > self.startup_timeout:
                raise RuntimeError('等待真实反馈首帧超时')
            previous = backend.state
            if previous in ('REQUESTING', 'HOLDING', 'ARMED', 'RUNNING'):
                backend.tick()
                if backend.state == 'RUNNING' and previous == 'ARMED':
                    self._phase = 'RUNNING'
                    self._message = '按共同单调时刻推进轨迹，非控制器硬同步'
                if backend.state == 'HOLDING' and previous in (
                    'REQUESTING', 'ARMED', 'RUNNING',
                ):
                    self._phase = 'SUCCEEDED'
                    self._message = ('控制权已确认，保持实际当前位置' if previous == 'REQUESTING'
                                     else '实际反馈已稳定到位')
        except Exception as error:
            if terminal:
                # 停止后继续读取反馈；读取失败不重复触发两臂联锁。
                self._error = self._error or str(error)
            else:
                self._fail(error)

    def _run(self) -> None:
        deadline = time.monotonic()
        try:
            while not self._shutdown.is_set():
                self._wake.clear()
                self._handle_stop()
                try:
                    command = self._queue.get_nowait()
                except Empty:
                    command = None
                if command is not None:
                    self._execute(command)
                now = time.monotonic()
                if now >= deadline:
                    # 命令执行期间到来的停止优先于下一次位置 tick。
                    self._handle_stop()
                    self._cycle()
                    deadline += self.period
                    if deadline < time.monotonic():
                        deadline = time.monotonic() + self.period
                self._publish()
                with self._command_lock:
                    if self._queue.empty():
                        self._pending = False
                self._wake.wait(max(0.0, deadline - time.monotonic()))
        except Exception as error:
            self._fail(error)
        finally:
            if self._backend is not None:
                try:
                    self._backend.close()
                except Exception as error:
                    self._error += f'；关闭失败：{error}'
            self._publish()
