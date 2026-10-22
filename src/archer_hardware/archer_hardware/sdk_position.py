"""为官方 SDK 位置命令隔离每臂配置与历史；本模块不连接设备或启动任务."""

from __future__ import annotations

from copy import deepcopy
from importlib import import_module
import math
from threading import RLock
import time
from typing import Any

from .hex_adapter import six_values, verified_sdk


def isolated_arm_class() -> Any:
    """构造可注册到 SDK 工厂的 Arm 子类，保留官方周期与编码器转换."""
    verified_sdk()
    module = import_module('hex_device.arm')
    manager_type = import_module('hex_device.arm_config').ArmConfigManager
    base: Any = module.Arm

    class Arm(base):
        """每实例保存厂商限位器历史，禁止静默裁剪后继续发送路径."""

        SUPPORTED_ROBOT_TYPES = (16, 17, 25)

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            """构造独立厂商管理器，尚不接受任何位置目标."""
            self._project_config = manager_type()
            self._project_config_lock = RLock()
            self._project_position_ready = False
            self._project_config_loaded = False
            self._project_position_error: Any = None
            self._project_position_timing: Any = None
            self._project_position_encoder: Any = None
            super().__init__(*args, **kwargs)

        def _construct_joint_command_msg(self, command: Any) -> Any:
            """编码异常传给项目故障锁存，防止厂商循环仅写日志后继续重试."""
            try:
                if self._project_position_encoder is not None:
                    return self._project_position_encoder(
                        self, command, super()._construct_joint_command_msg)
                return super()._construct_joint_command_msg(command)
            except Exception as error:
                if self._project_position_error is not None:
                    self._project_position_error(str(error))
                raise

        def reload_arm_config_from_dict(self, config_data: dict[str, Any]) -> bool:
            """配置仅进入当前实例；配置变化后必须重新提供实测起点."""
            with self._project_config_lock:
                # 新管理器避免厂商首次添加配置路径跳过完整验证。
                candidate = manager_type()
                try:
                    if not candidate.reload_from_dict(self.robot_type, deepcopy(config_data)):
                        return False
                    config = candidate.get_config(self.robot_type)
                    if config is None or len(config.robot_config.joints) != self.motor_count:
                        return False
                    for joint in config.robot_config.joints:
                        values = joint.joint_limit
                        if (len(values) != 6 or not all(math.isfinite(v) for v in values)
                                or values[0] >= values[1] or values[2] >= values[3]
                                or values[4] > values[5]):
                            return False
                except (KeyError, TypeError, ValueError):
                    return False
                self._project_config = candidate
                self._project_position_ready = False
                self._project_config_loaded = True
                return True

        def set_initial_positions(self, positions: Any) -> None:
            """保存调用方核对过的新鲜实测起点；这里不读取网络反馈."""
            values = list(six_values(positions))
            with self._project_config_lock:
                if not self._project_config_loaded:
                    raise RuntimeError('需要先加载明确的项目限位配置')
                bounds = self._project_config.get_joint_limits(self.robot_type)
                if bounds is None or len(bounds) != 6 or any(
                    not bound[0] <= value <= bound[1]
                    for bound, value in zip(bounds, values)
                ):
                    raise ValueError('SDK 起点不在当前实例配置范围内')
                self._project_config.set_initial_positions(self.robot_type, values)
                self._project_position_ready = True

        def validate_joint_positions(self, positions: Any, dt: float = 0.002) -> list[float]:
            """按实际发送间隔调用厂商限速器；不扩大速度上限或允许过期续发。"""
            requested = list(six_values(positions))
            if isinstance(dt, bool) or not math.isfinite(dt) or dt <= 0:
                raise ValueError('SDK 位置校验周期必须为有限正数')
            with self._project_config_lock:
                previous = self.get_last_positions()
                if not self._project_position_ready or previous is None:
                    raise RuntimeError('SDK 位置历史未初始化，需要重新读取实际起点')
                if self._project_position_timing is not None:
                    timing = self._project_position_timing()
                    if timing is not None:
                        sent, elapsed = timing
                        if (isinstance(elapsed, bool) or not math.isfinite(elapsed)
                                or not 0. <= elapsed <= .04):
                            raise RuntimeError('SDK 实际位置发送间隔超过40ms或时钟无效')
                        # SDK漏过缓存时，实际包间隔大于名义周期。使用实际上一包
                        # 而非未发送的构造历史；早到周期仍沿用厂商名义限速步长。
                        dt = max(dt, elapsed)
                        previous = list(six_values(sent))
                        self._project_config.set_last_positions(
                            self.robot_type, previous)
                result = self._project_config.validate_joint_positions(
                    self.robot_type, requested, dt,
                )
                if result != requested:
                    self._project_config.set_last_positions(self.robot_type, previous)
                    axis = next(i for i, (a, b) in enumerate(zip(result, requested)) if a != b)
                    raise ValueError(
                        'SDK 将裁剪位置或速度，拒绝改写已验证路径：'
                        f'J{axis + 1}，前目标={previous[axis]:.9f}rad，'
                        f'新目标={requested[axis]:.9f}rad，周期={dt:.6f}s，'
                        f'包间速度={(requested[axis] - previous[axis]) / dt:.6f}rad/s')
                return list(result)

        def get_last_positions(self) -> list[float] | None:
            """返回副本，调用方不能通过列表引用修改内部历史."""
            with self._project_config_lock:
                return deepcopy(self._project_config.get_last_positions(self.robot_type))

        def get_arm_config(self) -> Any:
            """只返回当前实例配置的副本."""
            with self._project_config_lock:
                return deepcopy(self._project_config.get_config(self.robot_type))

        def get_joint_limits(self) -> Any:
            """读取当前实例限位，不读取厂商全局同型号配置."""
            with self._project_config_lock:
                return deepcopy(self._project_config.get_joint_limits(self.robot_type))

        def get_joint_names(self) -> Any:
            """关节名也从本实例的配置中读取."""
            with self._project_config_lock:
                return deepcopy(self._project_config.get_joint_names(self.robot_type))

        def get_expected_motor_count(self) -> Any:
            """沿用厂商接口，查询本实例配置的轴数."""
            with self._project_config_lock:
                return self._project_config.get_motor_count(self.robot_type)

        def get_arm_name(self) -> str | None:
            """返回本实例名称，避免读取全局同型号配置."""
            config = self.get_arm_config()
            return config.name if config is not None else None

        def validate_joint_velocities(self, velocities: Any, dt: float = 0.002) -> Any:
            """本接入不提供速度控制入口."""
            raise ValueError('当前位置接入不允许速度模式')

        def set_initial_velocities(self, velocities: Any) -> None:
            """不向 SDK 全局速度历史写入任何数据."""
            raise ValueError('当前位置接入不允许速度模式')

        def get_last_velocities(self) -> None:
            """位置接入没有速度指令历史."""
            return None

        def clear_velocity_history(self) -> None:
            """仅清理本实例厂商管理器中的速度历史."""
            with self._project_config_lock:
                self._project_config.clear_velocity_history(self.robot_type)

        def clear_position_history(self) -> None:
            """制动或取消后使本臂起点失效，不触碰另一只臂."""
            with self._project_config_lock:
                self._project_config.clear_position_history(self.robot_type)
                self._project_position_ready = False

        def clear_last_positions(self) -> None:
            """显式清空后禁止用默认位置自动补齐起点."""
            self.clear_position_history()

        def clear_motion_history(self) -> None:
            """当前接入仅使用位置历史，不开启速度控制模式."""
            self.clear_position_history()

        def motor_command(self, command_type: Any, values: Any) -> None:
            """公开命令入口只允许 POSITION 和六轴 BRAKE."""
            if command_type == module.CommandType.POSITION:
                requested = list(six_values(values))
                if not self._project_position_ready:
                    raise RuntimeError('SDK 位置历史未初始化')
                # POSITION 路径沿用官方公开入口，不进入其全局清历史分支。
                super().motor_command(command_type, requested)
            elif command_type == module.CommandType.BRAKE:
                brakes = tuple(values)
                if len(brakes) != 6 or any(value is not True for value in brakes):
                    raise ValueError('BRAKE 必须为六个 True')
                # Arm 的非位置分支会清除同型号全局历史，改调用其官方基类入口。
                module.MotorBase.motor_command(self, command_type, list(brakes))
                self._last_command_time = time.perf_counter()
                self.clear_position_history()
            else:
                raise ValueError('当前位置接入只允许 POSITION 或 BRAKE')

    return Arm
