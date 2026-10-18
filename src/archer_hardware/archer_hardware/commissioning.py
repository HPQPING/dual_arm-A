"""提供只读反馈验收、小幅试动文件生成及离线轨迹预检与时间调整，不连接SDK或发送控制命令."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, cast

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
import yaml

from .hex_adapter import six_values
from .position_plan import JointPositionMove, JointPositionTrajectory, PositionLimits, positive


class FeedbackWindow:
    """检查关节名称、连续时间戳、静止速度及配置角度边界."""

    def __init__(self, side: str, limits: PositionLimits) -> None:
        """保存约束；此处不建立任何设备或 ROS 连接."""
        self.side = side
        self.limits = limits
        self.count = 0
        self.observed = 0
        self.invalid = 0
        self.errors: list[str] = []
        self.first_at: float | None = None
        self.last_at: float | None = None
        self.last_stamp = -1
        self.max_gap = 0.0
        self.max_speed = 0.0
        self.overrun = [0.0] * 6
        self.initial: tuple[float, ...] | None = None
        self.latest: tuple[float, ...] | None = None

    def add(self, message: JointState, received: float, ros_now_ns: int) -> None:
        """无效帧计入失败，不能由后续正常帧清除历史错误."""
        try:
            names = list(message.name)
            expected = [f'{self.side}_joint_{i}' for i in range(1, 7)]
            if len(names) != 6 or set(names) != set(expected):
                raise ValueError('关节名称缺失、重复或混臂')
            positions = six_values(message.position)
            velocities = six_values(message.velocity)
            six_values(message.effort)
            order = [names.index(name) for name in expected]
            q = tuple(positions[i] for i in order)
            v = tuple(velocities[i] for i in order)
            sec = message.header.stamp.sec
            ns = message.header.stamp.nanosec
            stamp = sec * 1_000_000_000 + ns
            if sec < 0 or not 0 <= ns < 1_000_000_000 or stamp <= self.last_stamp:
                raise ValueError('ROS 接收时间戳无效、重复或倒退')
            age = (ros_now_ns - stamp) / 1e9
            if not -0.05 <= age <= 0.5:
                raise ValueError('ROS 消息过期或时间基准不一致')
            # 结构与时间合法的原始反馈先留证；越界仍拒绝，但不丢失诊断数据。
            if self.last_at is not None:
                self.max_gap = max(self.max_gap, received - self.last_at)
            if self.initial is None:
                self.initial = q
                self.first_at = received
            self.latest = q
            self.last_at = received
            self.last_stamp = stamp
            self.max_speed = max(self.max_speed, max(abs(value) for value in v))
            for i, (value, low, high) in enumerate(zip(q, self.limits.lower, self.limits.upper)):
                self.overrun[i] = max(self.overrun[i], low - value, value - high)
            self.observed += 1
            self.limits.check_feedback(q)
            self.count += 1
        except (ValueError, TypeError, OverflowError) as error:
            self.invalid += 1
            if len(self.errors) < 10:
                self.errors.append(str(error))

    def report(
        self, now: float, minimum: int, gap: float, speed: float, minimum_span: float = 0.0,
    ) -> dict[str, Any]:
        """样本不足、中断、尾帧过期或观察到运动时不通过."""
        tail = None if self.last_at is None else now - self.last_at
        span = (0.0 if self.first_at is None or self.last_at is None
                else self.last_at - self.first_at)
        passed = (self.invalid == 0 and self.count >= minimum and tail is not None
                  and 0 <= tail <= gap and self.max_gap <= gap and self.max_speed <= speed
                  and span >= minimum_span)
        return {
            'passed': passed, 'samples': self.count, 'invalid_frames': self.invalid,
            'structurally_valid_observations': self.observed,
            'errors': self.errors, 'observed_span_s': span,
            'observed_rate_hz': (self.observed - 1) / span if span > 0 else None,
            'max_arrival_gap_s': self.max_gap, 'last_arrival_age_s': tail,
            'max_abs_velocity_rad_s': self.max_speed,
            'initial_observed_rad': self.initial, 'latest_rad': self.latest,
            'max_nominal_limit_overrun_rad': self.overrun,
            'latest_within_nominal_limits': (
                self.limits.within_nominal(self.latest) if self.latest is not None else None
            ),
        }


class FeedbackAuditNode(Node):
    """仅创建左右反馈订阅，不发布消息或创建控制客户端."""

    def __init__(self, limits: dict[str, PositionLimits]) -> None:
        """使用独立验收节点，拒绝仿真时间."""
        super().__init__('archer_feedback_audit')
        if self.get_parameter('use_sim_time').value:
            raise ValueError('反馈验收不能使用仿真时间')
        self.windows = {side: FeedbackWindow(side, value) for side, value in limits.items()}
        self.subscriptions_kept = [
            self.create_subscription(
                JointState, f'/hardware/{side}/joint_states', self.callback(side),
                qos_profile_sensor_data,
            ) for side in self.windows
        ]

    def callback(self, side: str) -> Callable[[JointState], None]:
        """返回单参数类型明确的回调，避免默认参数被识别为消息元数据."""
        def receive(message: JointState) -> None:
            self.windows[side].add(
                message, time.monotonic(), self.get_clock().now().nanoseconds,
            )
        return receive


def load_limits(path: Path, *, content: bytes | None = None) -> dict[str, PositionLimits]:
    """读取已有位置配置作比较，不启用其中连接或控制开关."""
    text = path.read_text(encoding='utf-8') if content is None else content.decode('utf-8')
    data = yaml.safe_load(text)
    parameters = data['archer_position_driver']['ros__parameters']
    return {
        side: PositionLimits(
            tuple(parameters[f'{side}_lower']), tuple(parameters[f'{side}_upper']),
            tuple(parameters[f'{side}_max_velocity']),
            tuple(parameters[f'{side}_max_acceleration']),
            parameters['feedback_tolerance'],
        ) for side in ('left', 'right')
    }


def _measure_joint_plan(
    data: object, limits: dict[str, PositionLimits],
) -> tuple[dict[str, Any], dict[str, tuple[list[int], list[tuple[float, ...]]]]]:
    """解析结构、名义位置和共享时间轴，测量峰值供预检与时间调整共用."""
    if not isinstance(data, dict) or set(data) != {
        'version', 'interpolation', 'left', 'right',
    }:
        raise ValueError('计划需要 version、interpolation、left、right 四个字段')
    if type(data['version']) is not int or data['version'] != 1:
        raise ValueError('计划 version 必须为整数 1')
    if data['interpolation'] != 'quintic_stop':
        raise ValueError('仅检查每段端点速度及加速度为零的 quintic_stop 插值')
    grids: dict[str, list[int]] = {}
    parsed: dict[str, tuple[list[int], list[tuple[float, ...]]]] = {}
    arms = {}
    for side in ('left', 'right'):
        arm = data[side]
        if not isinstance(arm, dict) or set(arm) != {'joint_names', 'points'}:
            raise ValueError(f'{side} 需要 joint_names 和 points')
        expected = [f'{side}_joint_{i}' for i in range(1, 7)]
        if arm['joint_names'] != expected:
            raise ValueError(f'{side} 关节名必须按 joint_1 至 joint_6 顺序排列')
        points = arm['points']
        if not isinstance(points, list) or len(points) < 2:
            raise ValueError(f'{side} 至少需要起点及一个目标点')
        times: list[int] = []
        positions: list[tuple[float, ...]] = []
        peak_speed = [0.0] * 6
        peak_acceleration = [0.0] * 6
        bound = limits[side]
        for index, point in enumerate(points):
            if not isinstance(point, dict) or set(point) != {'positions', 'time_from_start'}:
                raise ValueError(f'{side} 第 {index} 点字段不符合位置计划格式')
            stamp = point['time_from_start']
            if not isinstance(stamp, dict) or set(stamp) != {'sec', 'nanosec'}:
                raise ValueError(f'{side} 第 {index} 点需要 sec 和 nanosec')
            seconds, nanos = stamp['sec'], stamp['nanosec']
            if (type(seconds) is not int or type(nanos) is not int
                    or not 0 <= seconds <= 2_147_483_647 or not 0 <= nanos < 10 ** 9):
                raise ValueError(f'{side} 第 {index} 点时间无效')
            timestamp = seconds * 10 ** 9 + nanos
            if (not times and timestamp != 0) or (times and timestamp <= times[-1]):
                raise ValueError(f'{side} 时间必须从零开始并严格递增')
            raw = point['positions']
            if not isinstance(raw, list):
                raise ValueError(f'{side} 第 {index} 点 positions 必须为六维列表')
            # 正式计划全程使用名义限位，初值工程偏差另走已有准备流程。
            current = bound.check_positions(raw)
            if positions:
                duration = (timestamp - times[-1]) / 1e9
                previous = positions[-1]
                for joint, (a, b) in enumerate(zip(previous, current)):
                    distance = abs(b - a)
                    peak_speed[joint] = max(peak_speed[joint], 1.875 * distance / duration)
                    peak_acceleration[joint] = max(
                        peak_acceleration[joint], (10 / math.sqrt(3)) * distance / duration ** 2,
                    )
            times.append(timestamp)
            positions.append(current)
        grids[side] = times
        excursion = max(
            abs(q - first) for row in positions for q, first in zip(row, positions[0])
        )
        parsed[side] = times, positions
        arms[side] = {
            'points': len(times), 'segments': len(times) - 1,
            'initial_planned_rad': positions[0], 'final_planned_rad': positions[-1],
            'peak_velocity_rad_s': peak_speed,
            'peak_acceleration_rad_s2': peak_acceleration,
            'max_excursion_from_start_rad': excursion,
        }
    if grids['left'] != grids['right']:
        raise ValueError('双臂必须使用相同的纳秒时间轴，不能只比较总时长')
    return {
        'passed': True, 'interpolation': 'quintic_stop',
        'duration_s': grids['left'][-1] / 1e9, 'arms': arms,
        'execution_supported_by_current_driver': False,
        'paired_interface_available': True,
        'scope': '仅关节计划解析预检；不含实际初值、TCP、碰撞或夹紧力验证，不发送动作',
        'remaining': '成对接口需显式许可、在线初值预检及完整任务几何检查',
    }, parsed


def review_joint_plan(data: object, limits: dict[str, PositionLimits]) -> dict[str, Any]:
    """完整预检仍使用执行核心的速度、加速度及一小时时长规则."""
    report, parsed = _measure_joint_plan(data, limits)
    for side, (times, positions) in parsed.items():
        excursion = report['arms'][side]['max_excursion_from_start_rad']
        try:
            JointPositionTrajectory.prepare(
                times, positions, limits[side], max(excursion, 1e-12),
                actual_start=positions[0],
            )
        except ValueError as error:
            raise ValueError(f'{side} {error}') from error
    return report


def retime_joint_plan(
    data: object, limits: dict[str, PositionLimits], margin: float = 1.05,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """仅放慢共享时间轴；不改关节角、插值、名义限位或驱动配置."""
    margin = positive(margin, 'margin')
    if not 1.001 <= margin <= 2.0:
        raise ValueError('margin 必须在 1.001 到 2.0 之间')
    measured, parsed = _measure_joint_plan(data, limits)
    original = parsed['left'][0]
    if original[-1] > 3600 * 10 ** 9:
        raise ValueError('原始轨迹总时长已经超过一小时，不能通过放慢修复')
    required = 0.0
    for side, bounds in limits.items():
        arm = measured['arms'][side]
        for speed, acceleration, vmax, amax in zip(
            arm['peak_velocity_rad_s'], arm['peak_acceleration_rad_s2'],
            bounds.max_velocity, bounds.max_acceleration,
        ):
            required = max(required, speed / vmax, math.sqrt(acceleration / amax))
    factor = max(1.0, required * margin)
    if not math.isfinite(factor) or original[-1] * factor > 3600 * 10 ** 9:
        raise ValueError('满足限制所需总时长超过一小时，未生成计划')
    times = [0]
    for before, after in zip(original, original[1:]):
        times.append(times[-1] + math.ceil((after - before) * factor))
    # 输入结构已完成严格解析；深复制以免调用者原始计划被修改。
    result: dict[str, Any] = deepcopy(cast(dict[str, Any], data))
    for side in ('left', 'right'):
        for point, stamp in zip(result[side]['points'], times):
            point['time_from_start'] = {'sec': stamp // 10 ** 9, 'nanosec': stamp % 10 ** 9}
    verified = review_joint_plan(result, limits)
    verified.update({
        'source_duration_s': original[-1] / 1e9, 'requested_margin': margin,
        'time_scale': factor, 'positions_unchanged': True,
        'scope': '仅调整时间并复核关节约束；未验证实际起点、碰撞或夹持，不发送动作',
    })
    return result, verified


def plan_retime_main(args: list[str] | None = None) -> int:
    """显式生成新计划与哈希报告，不初始化 ROS、不连接设备或自动执行."""
    parser = argparse.ArgumentParser(description='离线放慢双臂轨迹，保持关节路径与配合关系')
    parser.add_argument('--plan-file', type=Path, required=True)
    parser.add_argument('--limits-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--margin', type=float, default=1.05)
    options = parser.parse_args(args)
    paths = [options.plan_file, options.limits_file, options.output, options.report]
    if len({p.resolve() for p in paths}) != 4:
        parser.error('输入、限制、输出计划和报告必须使用四个不同的文件路径')
    if options.output.exists() or options.report.exists():
        parser.error('输出计划或报告已存在，请指定新文件名')
    report: dict[str, Any] = {'passed': False, 'commands_sent': False}
    plan_created = False
    try:
        plan_bytes = options.plan_file.read_bytes()
        limits_bytes = options.limits_file.read_bytes()
        report.update({
            'source_plan': str(options.plan_file.resolve()),
            'source_sha256': hashlib.sha256(plan_bytes).hexdigest(),
            'limits_file': str(options.limits_file.resolve()),
            'limits_sha256': hashlib.sha256(limits_bytes).hexdigest(),
        })
        result, verified = retime_joint_plan(
            yaml.safe_load(plan_bytes.decode('utf-8')),
            load_limits(options.limits_file, content=limits_bytes), options.margin,
        )
        content = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
                   + '\n').encode('utf-8')
        # 先排他保留报告路径，再排他创建计划；不会覆盖输入或其他进程的文件。
        with options.report.open('x', encoding='utf-8') as evidence:
            try:
                with options.output.open('xb') as stream:
                    plan_created = True
                    stream.write(content)
                report.update(verified)
                report.update({
                    'output_plan': str(options.output.resolve()),
                    'output_sha256': hashlib.sha256(content).hexdigest(),
                })
                json.dump(report, evidence, ensure_ascii=False, indent=2, allow_nan=False)
                evidence.write('\n')
            except (OSError, ValueError, TypeError):
                if plan_created:
                    options.output.unlink()
                raise
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        print(f'时间调整失败：{error}；未发送动作')
        return 1
    print(f"已生成：{options.output}；{verified['source_duration_s']:.3f}s → "
          f"{verified['duration_s']:.3f}s；未发送动作")
    return 0


def plan_check_main(args: list[str] | None = None) -> int:
    """读取显式计划文件，排他写入通过或失败报告，不初始化 ROS."""
    parser = argparse.ArgumentParser(description='离线双臂关节计划预检，不连接设备或ROS')
    parser.add_argument('--plan-file', type=Path, required=True)
    parser.add_argument('--limits-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args(args)
    if options.output.exists():
        parser.error('报告已存在，请指定新文件名')
    report: dict[str, Any] = {'passed': False, 'commands_sent': False}
    try:
        plan_bytes = options.plan_file.read_bytes()
        limits_bytes = options.limits_file.read_bytes()
        report.update({
            'plan_file': str(options.plan_file.resolve()),
            'plan_sha256': hashlib.sha256(plan_bytes).hexdigest(),
            'limits_file': str(options.limits_file.resolve()),
            'limits_sha256': hashlib.sha256(limits_bytes).hexdigest(),
        })
        report.update(review_joint_plan(
            yaml.safe_load(plan_bytes.decode('utf-8')),
            load_limits(options.limits_file, content=limits_bytes),
        ))
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, yaml.YAMLError) as error:
        report['error'] = str(error)
    with options.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(f"{'通过' if report['passed'] else '未通过'}：离线报告 {options.output}；未发送动作")
    return 0 if report['passed'] else 1


def audit_main(args: list[str] | None = None) -> int:
    """观察指定时间并排他写入报告；不通过返回退出码一."""
    parser = argparse.ArgumentParser(description='只读反馈验收，不连接 SDK、不发送动作')
    parser.add_argument('--limits-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--min-samples', type=int, default=100)
    parser.add_argument('--max-gap', type=float, default=0.25)
    parser.add_argument('--stationary-speed', type=float, default=0.02)
    options = parser.parse_args(args)
    for name in ('duration', 'max_gap', 'stationary_speed'):
        positive(getattr(options, name), name)
    if options.min_samples < 2:
        parser.error('--min-samples 至少为二')
    if options.output.exists():
        parser.error('报告已存在，请指定新文件名')
    limits = load_limits(options.limits_file)
    rclpy.init(args=[])
    node: FeedbackAuditNode | None = None
    try:
        node = FeedbackAuditNode(limits)
        deadline = time.monotonic() + options.duration
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))
        now = time.monotonic()
        arms = {side: window.report(now, options.min_samples, options.max_gap,
                                    options.stationary_speed, options.duration * 0.8)
                for side, window in node.windows.items()}
        passed = all(value['passed'] for value in arms.values())
        report = {
            'passed': passed, 'limits_file': str(options.limits_file.resolve()),
            'thresholds': {'min_samples': options.min_samples, 'max_gap_s': options.max_gap,
                           'stationary_speed_rad_s': options.stationary_speed,
                           'minimum_span_s': options.duration * 0.8},
            'arms': arms,
            'scope': '仅 ROS 接收窗口；不证明数据来自实机，不验证制动、夹紧力或控制安全',
            'timestamp_scope': '驱动接收时间，非设备源时间；设备时效由只读驱动另行检查',
        }
        with options.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(f"{'通过' if passed else '未通过'}：报告已写入 {options.output}")
        return 0 if passed else 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


# 以下为准备及试动文件生成，和反馈检查共用配置读取。


def trial_target(
    side: str, joint: int, delta: float, duration: float,
    positions: tuple[float, ...], limits: PositionLimits,
) -> dict[str, Any]:
    """仅改变选定关节，保留其余实际角度，解析校验整段速度与加速度."""
    if side not in ('left', 'right') or type(joint) is not int or not 1 <= joint <= 6:
        raise ValueError('必须指定 left/right 和一至六号关节')
    if isinstance(delta, bool) or not math.isfinite(delta) or not 0 < abs(delta) <= 0.02:
        raise ValueError('试动增量须非零，绝对值不超过 0.02 rad')
    seconds = positive(duration, 'duration')
    start = limits.check_feedback(positions)
    goal = list(start)
    goal[joint - 1] += delta
    return target_message(side, start, tuple(goal), seconds, limits)


def target_message(
    side: str, start: tuple[float, ...], goal: tuple[float, ...],
    seconds: float, limits: PositionLimits,
) -> dict[str, Any]:
    """按各轴最大位移计算准备或试动时长，并生成同格式目标."""
    seconds = positive(seconds, 'duration')
    for a, b, velocity, acceleration in zip(
        start, goal, limits.max_velocity, limits.max_acceleration,
    ):
        distance = abs(b - a)
        seconds = max(seconds, 1.875 * distance / velocity,
                      math.sqrt((10 / math.sqrt(3)) * distance / acceleration))
    # 向上取整到纳秒后再次预检，避免临界舍入缩短计划时间。
    nanoseconds = math.ceil(seconds * 1e9)
    if nanoseconds // 1_000_000_000 > 2_147_483_647:
        raise ValueError('目标时间超过 ROS Duration 的范围')
    JointPositionMove.prepare(start, goal, nanoseconds / 1e9, limits, 0.02)
    return {
        'joint_names': [f'{side}_joint_{i}' for i in range(1, 7)],
        'points': [{'positions': list(goal), 'time_from_start': {
            'sec': nanoseconds // 1_000_000_000,
            'nanosec': nanoseconds % 1_000_000_000,
        }}],
    }


def read_positions(message: JointState, side: str, ros_now_ns: int) -> tuple[float, ...]:
    """核对单帧接收时间和静止反馈，按标准名称重排实际角度."""
    names = list(message.name)
    expected = [f'{side}_joint_{i}' for i in range(1, 7)]
    if len(names) != 6 or set(names) != set(expected):
        raise ValueError('反馈关节名缺失、重复或混臂')
    positions = six_values(message.position)
    velocities = six_values(message.velocity)
    if max(abs(value) for value in velocities) > 0.02:
        raise ValueError('反馈显示关节仍在运动，不生成试动目标')
    sec, ns = message.header.stamp.sec, message.header.stamp.nanosec
    stamp = sec * 1_000_000_000 + ns
    if sec < 0 or not 0 <= ns < 1_000_000_000:
        raise ValueError('反馈时间戳格式错误')
    if not -0.05 <= (ros_now_ns - stamp) / 1e9 <= 0.1:
        raise ValueError('反馈接收时间过期或时间基准不一致')
    return tuple(positions[names.index(name)] for name in expected)


class TrialObserver(Node):
    """只订阅反馈，不发布目标、不创建控制服务客户端."""

    def __init__(self, side: str, prefix: str) -> None:
        """默认读取只读驱动；可显式选择位置驱动的反馈命名空间."""
        super().__init__('archer_joint_trial')
        if self.get_parameter('use_sim_time').value:
            raise ValueError('实机试动文件不能使用仿真时间')
        self.side = side
        self.positions: tuple[float, ...] | None = None
        self.error = ''
        self.received = 0.0
        self.subscription_kept = self.create_subscription(
            JointState, f'{prefix}/{side}/joint_states', self.receive,
            qos_profile_sensor_data,
        )

    def receive(self, message: JointState) -> None:
        """仅保留最新一帧，坏帧会使此前缓存失效."""
        self.positions = None
        try:
            self.positions = read_positions(message, self.side, self.get_clock().now().nanoseconds)
            self.received = time.monotonic()
            self.error = ''
        except (ValueError, TypeError, OverflowError) as error:
            self.error = str(error)


def trial_main(args: list[str] | None = None) -> int:
    """生成供检查的 YAML；运行本命令不会使机械臂移动."""
    parser = argparse.ArgumentParser(description='读取反馈并生成小幅试动文件，不发送动作')
    parser.add_argument('--side', choices=['left', 'right'], required=True)
    parser.add_argument('--joint', type=int, choices=range(1, 7))
    parser.add_argument('--delta', type=float)
    parser.add_argument('--recover-limits', action='store_true',
                        help='只生成越界关节收回限位内的准备目标，不执行动作')
    parser.add_argument('--duration', type=float, default=2.0)
    parser.add_argument('--timeout', type=float, default=5.0)
    parser.add_argument('--limits-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--feedback-prefix', choices=['/hardware', '/hardware/position'],
                        default='/hardware')
    options = parser.parse_args(args)
    positive(options.timeout, 'timeout')
    limits = load_limits(options.limits_file)[options.side]
    # 在等待反馈之前拒绝非法增量、时长或已经存在的文件。
    if options.recover_limits:
        if options.joint is not None or options.delta is not None:
            parser.error('--recover-limits 不能与 --joint/--delta 同用')
    elif (options.joint is None or options.delta is None or not math.isfinite(options.delta)
          or not 0 < abs(options.delta) <= 0.02):
        parser.error('试动须指定 --joint 及非零、绝对值不超过 0.02 rad 的 --delta')
    positive(options.duration, 'duration')
    if options.output.exists():
        parser.error('输出文件已存在，请使用新文件名')
    rclpy.init(args=[])
    node: TrialObserver | None = None
    try:
        node = TrialObserver(options.side, options.feedback_prefix)
        deadline = time.monotonic() + options.timeout
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.positions is not None:
                break
        if node.positions is None or time.monotonic() - node.received > 0.1:
            raise RuntimeError(node.error or '没有收到新鲜的静止关节反馈')
        if options.recover_limits:
            target = target_message(options.side, node.positions,
                                    limits.recovery_goal(node.positions), options.duration, limits)
        else:
            assert options.joint is not None and options.delta is not None
            target = trial_target(options.side, options.joint, options.delta,
                                  options.duration, node.positions, limits)
        with options.output.open('x', encoding='utf-8') as stream:
            stream.write('# 根据一次反馈生成的预览文件；未暂存目标、未执行动作。\n')
            stream.write('# 执行前必须重新读取反馈并通过位置驱动的当前状态预检。\n')
            yaml.safe_dump(target, stream, allow_unicode=True, sort_keys=False)
        print(f'已生成 {options.output}；未发送目标，未执行动作。')
        return 0
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
