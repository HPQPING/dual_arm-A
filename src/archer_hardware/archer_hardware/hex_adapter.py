"""读取已核对版本的 HEX SDK 反馈，禁止下发机械臂控制消息."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib import import_module
import math
import os
from pathlib import Path
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
        base = cast(Any, module.HexDeviceApi)

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
