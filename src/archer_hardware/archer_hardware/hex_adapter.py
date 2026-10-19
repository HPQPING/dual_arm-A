"""读取已核对版本的 HEX SDK 反馈，禁止下发机械臂控制消息."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import hashlib
from importlib import import_module
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Iterable, cast
from urllib.parse import urlparse


SDK_HASHES = {
    'hex_device_api.py':
        'b08b04633d47cf88ad406ada1d2c06c39c3e42e20fb6276684f0ab95e3a0658c',
    'arm.py':
        'a03ab5404fc88f2f51edae99109849342e6874fa7058de811665010562655be7',
    'device_base.py':
        '2d4116fe6891d2f9551595bc4ed1976267dc48bdbeddeb724a934d625be327f8',
    'motor_base.py':
        'a23fe91324555c4d9c777db37391ecb5426c95ad947c1aa50ba4717943e977a3',
    'arm_config.py':
        'ce1dd4228c9ded01f80d91898516f77b8cee4929493dc415cdbc89523f60aa3a',
    'device_factory.py':
        '75ac688170a91fa7c9c811f46083fdc184470226c13220ed88ba6f1f0c80b36e',
}


@dataclass(frozen=True)
class JointFeedback:
    """同一帧的六关节反馈；单位依次为 rad、rad/s 和 N·m."""

    positions: tuple[float, ...]
    velocities: tuple[float, ...]
    efforts: tuple[float, ...]
    source_ns: int


def six_values(values: Iterable[object]) -> tuple[float, ...]:
    """拒绝非六维、布尔值和非有限数值，不改写实际关节角."""
    items = tuple(values)
    if len(items) != 6 or any(isinstance(v, (bool, str)) for v in items):
        raise ValueError('反馈必须包含六个数值')
    result = tuple(float(cast(Any, v)) for v in items)
    if not all(math.isfinite(v) for v in result):
        raise ValueError('反馈包含 NaN 或无穷大')
    return result


def report_only(message: Any) -> bool:
    """只允许反馈频率消息，混入任何控制字段都整包拒绝."""
    fields = {field.name for field, _ in message.ListFields()}
    return 'set_report_frequency' in fields and fields <= {
        'set_report_frequency', 'protocol_major_version',
        'protocol_minor_version',
    }


def verified_sdk() -> Any:
    """在创建连接前核对 SDK 实现和反馈时间基准."""
    if 'HEX_PTP_CLOCK' in os.environ:
        raise RuntimeError('当前反馈时效校验要求不设置 HEX_PTP_CLOCK')
    module = import_module('hex_device.hex_device_api')
    root = Path(cast(str, module.__file__)).parent
    for name, expected in SDK_HASHES.items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'SDK 文件 {name} 已变化，需重新核对后接入')
    return module


def managed_sdk_class(base: Any) -> Any:
    """修正已核对 SDK 的退出顺序，不修改厂商文件或发送控制消息."""

    class ManagedAPI(base):
        """先完成传输清理，再允许 SDK 主循环退出."""

        async def _HexDeviceApi__async_close(self) -> None:
            """关闭握手有界等待，失败时记录异常并中止本地传输."""
            waiter = getattr(self, '_project_close_waiter', None)
            if waiter is not None:
                await asyncio.shield(waiter)
                return
            waiter = asyncio.get_running_loop().create_future()
            self._project_close_waiter = waiter
            self._HexDeviceApi__is_closing.set()
            try:
                # 厂商 _stop_all_device_tasks 固定等0.1秒，仍会留下看门狗空窗。
                # 先取消任务并立即关闭传输；最终循环清理负责等待取消完成。
                tasks = getattr(self, '_device_tasks', {})
                for future in list(tasks.values()):
                    future.cancel()
                tasks.clear()
                await asyncio.sleep(0)
                client = self._HexDeviceApi__kcp_client
                if client is not None:
                    client.stop()
                    self._HexDeviceApi__kcp_client = None
                websocket = self._HexDeviceApi__websocket
                if websocket is not None:
                    try:
                        await asyncio.wait_for(websocket.close(), timeout=2.0)
                    except Exception:
                        # 官方当前版本使用 websockets 的 asyncio 传输。
                        transport = getattr(websocket, 'transport', None)
                        if transport is not None:
                            transport.abort()
                        raise
                    finally:
                        self._HexDeviceApi__websocket = None
            except Exception as error:
                self._project_close_error = (
                    f'{type(error).__name__}: {error}'
                )
                raise
            finally:
                waiter.set_result(None)
                event = self._HexDeviceApi__shutdown_event
                if event is not None:
                    event.set()

        def _HexDeviceApi__loop_start(self) -> None:
            """保留 SDK 主循环，退出时回收残留任务并释放事件循环."""
            try:
                super()._HexDeviceApi__loop_start()
            finally:
                loop = self._HexDeviceApi__loop
                if loop is not None and not loop.is_closed():
                    pending = asyncio.all_tasks(loop)
                    for task in pending:
                        task.cancel()
                    if pending:
                        done, remaining = loop.run_until_complete(
                            asyncio.wait(pending, timeout=1.0)
                        )
                        for task in done:
                            if not task.cancelled():
                                task.exception()
                        if remaining:
                            self._project_close_error = '后台异步任务未在期限内退出'
                    loop.run_until_complete(loop.shutdown_asyncgens())
                    loop.close()

        def close(self) -> None:
            """直接关闭传输，不调用 SDK 的设备 stop 或隐式退出控制请求."""
            loop = getattr(self, '_HexDeviceApi__loop', None)
            worker = getattr(self, '_HexDeviceApi__loop_thread', None)
            if worker is threading.current_thread():
                if (loop is not None and loop.is_running()
                        and not self._HexDeviceApi__is_closing.is_set()):
                    # SDK 解析失败时会在自己的事件线程调用 close。
                    task = loop.create_task(self._HexDeviceApi__async_close())
                    task.add_done_callback(
                        lambda done: None if done.cancelled() else done.exception()
                    )
                return
            if loop is not None and loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    self._HexDeviceApi__async_close(), loop,
                )
                try:
                    future.result(timeout=3.0)
                except Exception as error:
                    self._project_close_error = f'{type(error).__name__}: {error}'
            if worker is not None:
                worker.join(timeout=1.5)
                if worker.is_alive():
                    raise RuntimeError('SDK 后台线程未退出，连接清理尚未确认')
            error = getattr(self, '_project_close_error', '')
            if error:
                raise RuntimeError(f'SDK 连接清理失败：{error}')

    return ManagedAPI


class HexFeedbackAdapter:
    """显式连接后读取反馈；本类没有运动、使能或清故障接口."""

    def __init__(self, url: str, stale_timeout: float = 0.5) -> None:
        """保存连接参数，构造时不导入 SDK、不建立网络连接."""
        parsed = urlparse(url)
        if (parsed.scheme != 'ws' or not parsed.hostname
                or parsed.port is None or parsed.username is not None
                or parsed.password is not None or parsed.query
                or parsed.fragment or parsed.path not in ('', '/')):
            raise ValueError('地址必须为 ws://主机:端口')
        if not math.isfinite(stale_timeout) or stale_timeout <= 0:
            raise ValueError('反馈超时时间必须为正数')
        self.url = url
        self.stale_timeout = stale_timeout
        self._api: Any = None
        self._arm_type: Any = None
        self._last_source_ns = -1

    def connect(self) -> None:
        """核对 SDK 后连接，关闭设备控制循环并过滤所有控制下行."""
        if self._api is not None:
            raise RuntimeError('已经连接，不能重复连接')
        module = verified_sdk()
        base = managed_sdk_class(module.HexDeviceApi)

        class FeedbackAPI(base):
            """隔离官方默认后台校准、清错及控制权申请行为."""

            def _start_device_periodic_task(self, device_id: int) -> None:
                """反馈接入不启动任何设备控制周期任务."""
                return None

            async def _send_down_message(self, data: Any) -> None:
                """在实际发送出口阻断控制消息，只放行反馈频率."""
                if report_only(data):
                    await super()._send_down_message(data)

        self._arm_type = import_module('hex_device.arm').Arm
        self._api = FeedbackAPI(
            ws_url=self.url, control_hz=50, enable_kcp=False,
        )

    def read(self) -> JointFeedback | None:
        """读取最新一帧；重复帧不重发，断线、过期或异常帧报错."""
        if self._api is None:
            return None
        if self._api.is_api_exit() or self._api.is_websocket_recv_timeout():
            raise RuntimeError('SDK 已退出或通信超时')
        arms = [d for d in self._api.device_list
                if isinstance(d, self._arm_type)]
        if not arms:
            return None
        if len(arms) != 1 or arms[0].motor_count != 6:
            raise RuntimeError('每个地址必须对应一台六关节机械臂')
        if arms[0].robot_type not in (16, 17, 25, 26):
            raise RuntimeError('反馈设备不是已核对的六轴 Archer 系列')
        status = arms[0].get_simple_motor_status(pop=False)
        if status is None:
            return None
        stamp = status['ts']
        seconds, nanos = stamp['s'], stamp['ns']
        if (type(seconds) is not int or type(nanos) is not int
                or seconds < 0 or not 0 <= nanos < 1_000_000_000):
            raise ValueError('SDK 时间戳无效')
        source_ns = seconds * 1_000_000_000 + nanos
        age = (time.perf_counter_ns() - source_ns) / 1e9
        if age < -0.05 or age > self.stale_timeout:
            raise RuntimeError('关节反馈过期或时间基准不匹配')
        if source_ns < self._last_source_ns:
            raise RuntimeError('反馈时间倒退，请断开后重新检查设备')
        if source_ns == self._last_source_ns:
            return None
        feedback = JointFeedback(
            six_values(status['pos']), six_values(status['vel']),
            six_values(status['eff']), source_ns,
        )
        self._last_source_ns = source_ns
        return feedback

    def close(self) -> None:
        """关闭连接；过滤器在 SDK 清理过程中仍然生效."""
        if self._api is not None:
            self._api.close()
            self._api = None
        self._last_source_ns = -1
