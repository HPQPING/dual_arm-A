"""提供显式真机操作、键盘停止及离线检查入口，不直接调用SDK或自动使能."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any, Callable

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rclpy.signals import SignalHandlerOptions
from rclpy.task import Future
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .position_plan import positive


COMMANDS = ('status', 'connect', 'enable', 'move', 'stop', 'halt', 'disconnect')
TERMINAL_FAILURES = ('FAILED', 'REJECTED', 'CANCELLED')


def return_identifiers(message: str) -> dict[str, str]:
    """解析同一归位请求中的各臂编号，命令行和键盘共用."""
    payload = json.loads(message)
    identifiers = payload.get('commands') if isinstance(payload, dict) else None
    if (not isinstance(identifiers, dict) or not identifiers
            or any(side not in ('left', 'right') or not isinstance(identifier, str)
                   or re.fullmatch(r'[0-9a-f]{12}', identifier) is None
                   for side, identifier in identifiers.items())):
        raise ValueError('归位受理结果缺少有效操作编号')
    return identifiers


def request_path(command: str, side: str | None) -> str:
    """两种停止使用全局接口，其他操作必须明确指定一只臂."""
    if command not in COMMANDS or command == 'status':
        raise ValueError('该命令没有控制服务入口')
    if command in ('stop', 'halt'):
        if side is not None:
            raise ValueError('stop/halt 是全局命令，请不要指定 --side')
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
                'return': 'HOLDING', 'disconnect': 'DISCONNECTED'}[command]
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
        self.halt_on_failure = command in ('connect', 'enable', 'move', 'stop')
        baseline = self.version
        response = self.trigger(path, min(timeout, 5.0))
        deadline = time.monotonic() + timeout
        if command == 'stop':
            identifiers = return_identifiers(response.message)
            while time.monotonic() < deadline and rclpy.ok():
                if self.status is not None:
                    results = [operation_result(self.status, s, identifier, 'return')
                               for s, identifier in identifiers.items()]
                    if all(result is True for result in results):
                        return {'result': 'RETURNED_TO_INITIAL_TARGET',
                                'commands': identifiers, 'status': self.status}
                rclpy.spin_once(self, timeout_sec=0.05)
            raise RuntimeError('等待停止并归位完成超时')
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


def keyboard_actions(data: bytes, escape: int = 0) -> tuple[list[str], int]:
    """只识别S和Ctrl+C，忽略空格、方向键等其他输入."""
    actions: list[str] = []
    for value in data:
        if value == 3:
            actions.append('exit')
            escape = 0
        elif escape == 1:
            escape = 2 if value in (ord('['), ord('O')) else 0
        elif escape == 2:
            if 0x40 <= value <= 0x7E:
                escape = 0
        elif value == 27:
            escape = 1
        elif value in (ord('s'), ord('S')):
            actions.append('stop')
    return actions, escape


class KeyboardStops:
    """异步处理两种停止，归位等待期间不会阻塞立即停止键."""

    def __init__(
        self, node: OperatorClient, timeout: float,
        report: Callable[[str], None] = print,
    ) -> None:
        """只创建两个停止服务客户端，不连接设备或申请控制."""
        self.node = node
        self.timeout = positive(timeout, 'timeout')
        self.report = report
        self.stop_client = node.create_client(Trigger, '/hardware/position/stop')
        self.halt_client = node.create_client(Trigger, '/hardware/position/halt')
        self.stop_future: Future | None = None
        self.halt_future: Future | None = None
        self.identifiers: dict[str, str] = {}
        self.stop_deadline = 0.0
        self.halt_deadline = 0.0
        self.halt_latched = False
        self.halt_phase = 'IDLE'
        self.halt_baseline = 0
        self.exit_requested = False

    def press_stop(self) -> None:
        """S只提交一次归位请求；服务未就绪不排队等待迟发."""
        if self.halt_latched or (self.node.status and self.node.status['interlock']):
            self.report('本次运行已结束，禁止归位。')
        elif self.stop_future is not None or self.identifiers:
            self.report('归位请求正在处理，未重复发送。')
        elif not self.stop_client.service_is_ready():
            self.report('归位服务未就绪，未发送；立即停止请按Ctrl+C。')
        else:
            self.stop_future = self.stop_client.call_async(Trigger.Request())
            self.stop_deadline = time.monotonic() + self.timeout
            self.report('已请求完成后归位；Ctrl+C仍可立即请求 halt。')

    def press_halt(self, *, exit_requested: bool = False) -> None:
        """立即停止优先于归位，之后不允许重新发起归位."""
        self.exit_requested = self.exit_requested or exit_requested
        if self.halt_latched:
            return
        self.halt_latched = True
        self.identifiers.clear()
        # 丢弃本地等待不等于撤销远端归位；由独立halt请求取消远端动作。
        self.stop_future = None
        self.halt_baseline = self.node.version
        self.halt_deadline = time.monotonic() + 5.0
        self.halt_phase = 'WAITING_SERVICE'
        self.report('已按下立即停止：正在请求 halt，不归位。')

    def poll(self) -> None:
        """只检查就绪状态和异步结果，任何分支都不阻塞等服务."""
        now = time.monotonic()
        if self.halt_latched:
            self.poll_halt(now)
            return
        try:
            if self.stop_future is not None and self.stop_future.done():
                response = self.stop_future.result()
                self.stop_future = None
                if response is None:
                    raise RuntimeError('归位服务未返回结果')
                if not response.success:
                    self.report(f'归位未受理：{response.message}')
                    return
                self.identifiers = return_identifiers(response.message)
                self.report('归位已受理，等待所有已配置臂完成。')
            if self.identifiers and self.node.status is not None:
                results = [operation_result(self.node.status, side, identifier, 'return')
                           for side, identifier in self.identifiers.items()]
                if all(result is True for result in results):
                    self.identifiers.clear()
                    self.report('归位已完成并保持；键盘继续监听。')
            if (self.stop_future is not None or self.identifiers) and now > self.stop_deadline:
                raise RuntimeError('归位响应或完成超时，结果未确认')
        except Exception as error:
            self.report(f'归位异常：{error}')
            self.press_halt()

    def poll_halt(self, now: float) -> None:
        """保持halt优先；超时明确报告未确认，不冒充物理停止."""
        if self.halt_phase in ('COMPLETE', 'UNCONFIRMED'):
            return
        try:
            if self.halt_phase == 'WAITING_SERVICE' and self.halt_client.service_is_ready():
                self.halt_future = self.halt_client.call_async(Trigger.Request())
                self.halt_phase = 'WAITING_RESPONSE'
            if self.halt_future is not None and self.halt_future.done():
                response = self.halt_future.result()
                self.halt_future = None
                if response is None or not response.success:
                    raise RuntimeError('halt未被服务确认受理')
                self.halt_phase = 'WAITING_STATUS'
            status = self.node.status
            if (self.halt_phase == 'WAITING_STATUS' and status is not None
                    and self.node.version > self.halt_baseline and status['interlock']
                    and status['arms'] and all(
                        arm.get('state') in ('STOPPED', 'FAULT', 'DISCONNECTED', 'UNCONFIGURED')
                        for arm in status['arms'].values()
                    )):
                self.halt_phase = 'COMPLETE'
                self.report('已观察到软件停止状态；不代表物理停止已确认。本次运行不可恢复。')
            elif now > self.halt_deadline:
                raise RuntimeError('halt服务或停止状态超时；不能确认实际停止')
        except Exception as error:
            self.halt_phase = 'UNCONFIRMED'
            self.report(f'立即停止未确认：{error}。请使用实体急停处理现场异常。')

    def close(self) -> None:
        """释放两个ROS客户端，不断开机械臂SDK连接."""
        self.node.destroy_client(self.stop_client)
        self.node.destroy_client(self.halt_client)


def keyboard_loop(node: OperatorClient, timeout: float) -> int:
    """在前台Ubuntu终端监听按键；Ctrl+C请求halt后退出并恢复终端."""
    import os
    import select
    import termios
    import tty

    descriptor = sys.stdin.fileno()
    original = termios.tcgetattr(descriptor)
    controller = KeyboardStops(node, timeout)
    escape = 0
    try:
        tty.setcbreak(descriptor)
        # 键盘监控异常退出时，复用main的尽力halt兜底。
        node.request_issued = True
        node.halt_on_failure = True
        print('键盘已启用：S=完成后归位；Ctrl+C=立即请求halt并退出；空格无操作。', flush=True)
        print('只在本终端获得焦点时生效；本入口不会连接、使能或启动运动。', flush=True)
        while rclpy.ok():
            try:
                if select.select([descriptor], [], [], 0)[0]:
                    data = os.read(descriptor, 64)
                    actions, escape = keyboard_actions(data, escape)
                    if not data or 'exit' in actions:
                        controller.press_halt(exit_requested=True)
                    elif 'stop' in actions:
                        controller.press_stop()
                controller.poll()
                if (controller.exit_requested
                        and controller.halt_phase in ('COMPLETE', 'UNCONFIRMED')):
                    return 0 if controller.halt_phase == 'COMPLETE' else 1
                rclpy.spin_once(node, timeout_sec=0.02)
            except KeyboardInterrupt:
                controller.press_halt(exit_requested=True)
        return 1
    finally:
        try:
            controller.close()
        finally:
            termios.tcsetattr(descriptor, termios.TCSANOW, original)


def main(args: list[str] | None = None) -> int:
    """默认只看状态；异常时对已发出的活动请求尽力补发全局 halt."""
    arguments = list(sys.argv[1:] if args is None else args)
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
        'command', choices=COMMANDS + ('audit', 'trial', 'plan-check', 'plan-retime', 'keyboard'),
        nargs='?', default='status',
    )
    parser.add_argument('--side', choices=['left', 'right'])
    parser.add_argument('--timeout', type=float, default=30.0)
    options = parser.parse_args(arguments)
    if options.command in ('audit', 'trial', 'plan-check', 'plan-retime'):
        parser.error('audit/trial/plan-check/plan-retime 子命令必须放在参数最前面')
    positive(options.timeout, 'timeout')
    if options.command == 'keyboard' and (options.side is not None or not sys.stdin.isatty()):
        parser.error('keyboard须在前台交互终端运行，不加--side；不接受管道或后台输入')
    if options.command not in ('status', 'keyboard'):
        request_path(options.command, options.side)
    rclpy.init(args=[], signal_handler_options=(
        SignalHandlerOptions.NO if options.command == 'keyboard' else None
    ))
    node: OperatorClient | None = None
    try:
        node = OperatorClient()
        if options.command == 'keyboard':
            return keyboard_loop(node, options.timeout)
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
