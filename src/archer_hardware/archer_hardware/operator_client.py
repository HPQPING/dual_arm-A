"""提供显式真机操作及离线检查入口，不直接调用SDK或自动使能."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .position_plan import positive


COMMANDS = ('status', 'connect', 'enable', 'move', 'halt', 'disconnect')
TERMINAL_FAILURES = ('FAILED', 'REJECTED', 'CANCELLED')


def request_path(command: str, side: str | None) -> str:
    """故障停止使用全局接口，其他操作必须明确指定一只臂."""
    if command not in COMMANDS or command == 'status':
        raise ValueError('该命令没有控制服务入口')
    if command == 'halt':
        if side is not None:
            raise ValueError('halt 是全局命令，请不要指定 --side')
        return f'/hardware/position/{command}'
    if side not in ('left', 'right'):
        raise ValueError('必须用 --side left/right 指定一只臂')
    return f'/hardware/position/{side}/{command}'


def operation_result(
    status: dict[str, Any], side: str, identifier: str, command: str,
) -> bool | None:
    """仅匹配本次编号；忽略此前操作成功状态，不把受理当作完成."""
    arms = status.get('arms', {})
    frame = arms.get(side, {})
    if command != 'disconnect' and status.get('interlock'):
        raise RuntimeError(f"软件联锁：{status['interlock']}")
    if frame.get('command_id') != identifier:
        return None
    phase = frame.get('phase')
    if phase in TERMINAL_FAILURES:
        raise RuntimeError(f"{phase}：{frame.get('message', '')}；{frame.get('error', '')}")
    if phase != 'SUCCEEDED':
        return None
    expected = {'connect': 'CONNECTED', 'enable': 'HOLDING', 'move': 'HOLDING',
                'disconnect': 'DISCONNECTED'}[command]
    if frame.get('state') != expected:
        raise RuntimeError('操作编号已完成，但当前状态已改变，请检查完整状态')
    return True


class OperatorClient(Node):
    """订阅状态并按请求调用服务，没有任何启动时控制动作."""

    def __init__(self) -> None:
        """构造时只订阅状态，不连接设备或申请控制权."""
        super().__init__('archer_hardware_operator')
        self.status: dict[str, Any] | None = None
        self.version = 0
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.subscription_kept = self.create_subscription(
            String, '/hardware/position/status', self.receive, qos,
        )
        self.request_issued = False
        self.halt_on_failure = False

    def receive(self, message: String) -> None:
        """只接受具有预期顶层结构的状态 JSON."""
        try:
            value = json.loads(message.data)
            if (not isinstance(value, dict) or not isinstance(value.get('arms'), dict)
                    or not isinstance(value.get('interlock'), str)):
                return
            if any(not isinstance(arm, dict) for arm in value['arms'].values()):
                return
            self.status = value
            self.version += 1
        except (ValueError, TypeError):
            return

    def wait_status(self, timeout: float) -> dict[str, Any]:
        """等待状态；不通过连接或使能来主动获取数据."""
        deadline = time.monotonic() + timeout
        while self.status is None and time.monotonic() < deadline and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
        if self.status is None:
            raise RuntimeError('没有收到位置控制节点状态')
        return self.status

    def trigger(self, path: str, timeout: float) -> Trigger.Response:
        """单次发送服务请求，不重试，超时不能推断设备未执行."""
        client = self.create_client(Trigger, path)
        deadline = time.monotonic() + timeout
        try:
            while not client.service_is_ready():
                if time.monotonic() >= deadline or not rclpy.ok():
                    raise RuntimeError(f'服务未就绪：{path}')
                rclpy.spin_once(self, timeout_sec=0.05)
            future = client.call_async(Trigger.Request())
            self.request_issued = True
            while not future.done() and time.monotonic() < deadline and rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)
            if not future.done():
                raise RuntimeError('服务响应超时；请求可能已受理，不自动重试')
            response = future.result()
            if response is None:
                raise RuntimeError('服务没有返回结果')
            if not response.success:
                self.request_issued = False
                raise RuntimeError(response.message)
            return response
        finally:
            self.destroy_client(client)

    def run_command(self, command: str, side: str | None, timeout: float) -> dict[str, Any]:
        """等待实际状态，返回时不关闭设备连接、不自动归位."""
        if command == 'status':
            return {'result': 'STATUS', 'status': self.wait_status(timeout)}
        path = request_path(command, side)
        self.halt_on_failure = command in ('connect', 'enable', 'move')
        baseline = self.version
        response = self.trigger(path, min(timeout, 5.0))
        deadline = time.monotonic() + timeout
        if command == 'halt':
            while time.monotonic() < deadline and rclpy.ok():
                status = self.status
                if status is not None and self.version > baseline:
                    states = [value.get('state') for value in status['arms'].values()]
                    if (command == 'halt' and status['interlock'] and states
                            and all(value in ('STOPPED', 'FAULT', 'DISCONNECTED', 'UNCONFIGURED')
                                    for value in states)):
                        return {'result': 'SOFTWARE_HALT_OBSERVED',
                                'physical_stop_confirmed': False, 'status': status}
                rclpy.spin_once(self, timeout_sec=0.05)
            raise RuntimeError('未观察到停止后的软件状态；不代表实际物理停止已完成')
        match = re.search(r'编号 ([0-9a-f]{12})(?:，|\s|$)', response.message)
        if match is None or side is None:
            raise RuntimeError('服务没有返回可识别的操作编号，不能确认执行结果')
        identifier = match.group(1)
        while time.monotonic() < deadline and rclpy.ok():
            if (self.status is not None
                    and operation_result(self.status, side, identifier, command)):
                return {'result': 'SUCCEEDED', 'command_id': identifier, 'status': self.status}
            rclpy.spin_once(self, timeout_sec=0.05)
        raise RuntimeError('等待本次操作完成超时；不能按成功处理，也不自动重试')


def main(args: list[str] | None = None) -> int:
    """默认只看状态；异常时对已发出的活动请求尽力补发全局 halt."""
    arguments = list(sys.argv[1:] if args is None else args)
    if arguments and arguments[0] == 'csv-import':
        from .csv_import import main as csv_import_main
        return csv_import_main(arguments[1:])
    # 联调工具只在显式选择时载入；不创建操作客户端或控制服务请求。
    if arguments and arguments[0] in ('audit', 'trial', 'plan-check', 'plan-retime'):
        from .commissioning import audit_main, plan_check_main, plan_retime_main, trial_main
        entry = {'audit': audit_main, 'trial': trial_main, 'plan-check': plan_check_main,
                 'plan-retime': plan_retime_main}[
            arguments[0]
        ]
        return entry(arguments[1:])
    parser = argparse.ArgumentParser(
        description='统一硬件操作；默认 status，不自动连接或使能',
        epilog='离线计划检查：hardware_operator plan-check --help；另有audit/trial子命令',
    )
    parser.add_argument(
        'command', choices=COMMANDS + (
            'audit', 'trial', 'plan-check', 'plan-retime', 'csv-import',
        ),
        nargs='?', default='status',
    )
    parser.add_argument('--side', choices=['left', 'right'])
    parser.add_argument('--timeout', type=float, default=30.0)
    options = parser.parse_args(arguments)
    if options.command in ('audit', 'trial', 'plan-check', 'plan-retime', 'csv-import'):
        parser.error('audit/trial/plan-check/plan-retime/csv-import 子命令必须放在参数最前面')
    positive(options.timeout, 'timeout')
    if options.command != 'status':
        request_path(options.command, options.side)
    rclpy.init(args=[])
    node: OperatorClient | None = None
    try:
        node = OperatorClient()
        result = node.run_command(options.command, options.side, options.timeout)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (RuntimeError, ValueError, OSError, KeyboardInterrupt) as error:
        print(f'操作未确认成功：{error}')
        if node is not None and node.request_issued and node.halt_on_failure:
            try:
                node.trigger('/hardware/position/halt', 1.0)
                print('已请求全局软件 halt；不代表物理停止已确认，未自动归位或断开。')
            except Exception as halt_error:
                print(f'软件 halt 未确认：{halt_error}')
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    raise SystemExit(main())
