"""共用画圆与协同的文件核对、SDK准备、连续播放、归位及退出流程。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
from importlib import import_module
import json
from pathlib import Path
import time
from typing import Any
from uuid import uuid4


TRACKING_LIMIT_RAD = .05
COOPERATIVE_TRACKING_LIMIT_RAD = .045


def worker_failure(workers: Mapping[str, Any]) -> str | None:
    """先报告已锁存故障，避免另一侧正常取消或滞后的线程快照掩盖首因。"""
    faults, stopped = [], []
    for side, worker in workers.items():
        state = worker.snapshot()
        backend = getattr(worker, '_backend', None)
        if state.state == 'FAULT' or state.phase in ('FAILED', 'REJECTED'):
            reason = state.error or getattr(backend, 'last_error', '') or state.message
            faults.append(side + ': ' + (reason or '任务故障'))
        elif backend is not None and (
                getattr(backend, 'state', None) == 'FAULT'
                or bool(getattr(backend, '_native_error', ''))):
            reason = getattr(backend, '_native_error', '') or getattr(backend, 'last_error', '')
            faults.append(side + ': ' + (reason or 'SDK任务故障'))
        elif state.state == 'STOPPED':
            stopped.append(side + ': ' + (state.error or '任务已停止'))
    return '；'.join(faults or stopped) or None


def show_gripper_feedback(workers: Mapping[str, Any]) -> None:
    """只打印已有夹爪缓存，区分指令、底层发送和物理反馈，不确认真实夹持。"""
    for side, worker in workers.items():
        gripper = getattr(getattr(worker, '_backend', None), 'gripper', None)
        if gripper is None:
            continue
        state = gripper.status()
        print(f'{side}夹爪：指令={state["command"]}，底层已发={state["sent_command"]}，'
              f'实际反馈={state["actual_position"]}，反馈年龄s={state["feedback_age_s"]}',
              flush=True)


def run(
    prepared_dir: Path, run_dir: Path, config: Path, *, ros_args: list[str] | None = None,
    trajectory_file: Path | None = None, task_type: str = 'split_circle',
    backend_type: type[Any] | None = None,
) -> int:
    """显式启动后才连接；中间采样点不等待到位，不生成额外诊断文件。"""
    if task_type not in ('split_circle', 'cooperative_grasp'):
        raise ValueError('未知完整任务类型')
    cooperative = task_type == 'cooperative_grasp'
    # 每项任务的几何预检与运行保护使用同一个值；协同0.045已离线核对间隙。
    tracking_limit = COOPERATIVE_TRACKING_LIMIT_RAD if cooperative else TRACKING_LIMIT_RAD
    csv_stem = 'state' if cooperative else 'tracking'
    task_name = '协同搬运' if cooperative else '画圆'
    rclpy = import_module('rclpy')
    from archer_hardware.position_adapter import HexPositionAdapter
    from archer_hardware.position_node import PositionControlNode, prepare_file_core_message

    parameters = {
        'left_limits_confirmed': 'true', 'right_limits_confirmed': 'true',
        'allow_paired_trajectory': 'true', 'allow_continuous_trajectory': 'true',
        'allow_task_file_trajectory': 'true', 'max_excursion': '3.5',
        'task_joint_error_rad': str(tracking_limit),
        'task_startup_joint_error_rad': str(tracking_limit),
        'tracking_tolerance': str(tracking_limit),
    }
    arguments = [*(ros_args or []), '--ros-args', '--params-file', str(config)]
    for name, value in parameters.items():
        arguments.extend(['-p', name + ':=' + value])
    node = None
    run_started = time.monotonic()
    core_cancel = None
    rclpy.init(args=arguments)
    try:
        if cooperative:
            from .gripper_adapter import CooperativePositionAdapter

            node = PositionControlNode(
                task_type=task_type, backend_type=backend_type or CooperativePositionAdapter)
        else:
            node = PositionControlNode()

        def wait_until(condition: Callable[[], bool], timeout: float, stage: str) -> None:
            """只等待当前流程结束，不根据中间点偏差控制播放进度。"""
            assert node is not None
            deadline = time.monotonic() + timeout
            next_print = time.monotonic() + 15.
            previous_phase = ''
            while rclpy.ok() and time.monotonic() < deadline:
                if node._interlock_reason:
                    raise RuntimeError(node._interlock_reason)
                failure = worker_failure(node.workers)
                if failure is not None:
                    raise RuntimeError(failure)
                if condition():
                    return
                gate = node._paired_gate
                if node._paired_active and gate is not None:
                    value = gate.playback_status()['reference_elapsed_s']
                    elapsed = float(value) if isinstance(value, (int, float)) else 0.
                    phases = node._paired_details.get('phase_start_ns', {})
                    if cooperative and 'gripper_event_schedule' in node._paired_details:
                        phases = dict(sorted({
                            **phases,
                            **node._paired_details['gripper_event_schedule']['phase_start_ns'],
                        }.items(), key=lambda item: item[1]))
                    phase = next((name for name, stamp in reversed(list(phases.items()))
                                  if elapsed >= stamp / 1e9), '')
                    if phase and phase != previous_phase:
                        print('轨迹阶段：' + phase + f'，参考时间{elapsed:.2f}s', flush=True)
                        if cooperative and phase in ('OPEN_FOR_GRASP', 'INSERT'):
                            show_gripper_feedback(node.workers)
                        previous_phase = phase
                if time.monotonic() >= next_print:
                    print(stage + '进行中，可按Ctrl+C取消', flush=True)
                    next_print = time.monotonic() + 15.
                rclpy.spin_once(node, timeout_sec=.002)
            if not rclpy.ok():
                raise KeyboardInterrupt
            raise RuntimeError(stage + f'超过{timeout:.0f}秒等待上限')

        # 阶段串联参考MTC SerialContainer思想；数值轨迹和SDK周期仍复用本项目实现。
        prepared_file = trajectory_file or prepared_dir / (csv_stem + '.json')
        report_file = prepared_file.with_name(prepared_file.stem + '_report.json')
        identifier = uuid4().hex
        request = {
            'request_id': identifier,
            'run_dir': str(run_dir),
            'limits_file': str(config),
            'prepared_file': str(prepared_file),
            'report_file': str(report_file),
            'transition_max_excursion': .25 if cooperative else .05,
        }
        request_text = json.dumps(request)
        print('1/4 连接机械臂前检查完整文件主体，不建立SDK连接', flush=True)
        assert node._preparer is not None and node._preparation_manager is not None
        core_cancel = node._preparation_manager.Event()
        core_started = time.monotonic()
        future = node._preparer.submit(
            prepare_file_core_message, request_text, dict(node.limits), node.max_excursion,
            core_cancel, joint_error_rad=tracking_limit,
            control_hz=round(1. / node._control_period),
            **({'task_type': task_type} if cooperative else {}))
        wait_until(future.done, 125., '文件主体预检（尚未连接机械臂）')
        prepared_core = future.result()
        print(f'主体预检完成，耗时{time.monotonic() - core_started:.3f}秒', flush=True)

        print('2/4 主体预检已通过，连接双臂并读取初始角度', flush=True)
        for side in node.sides:
            node.handle_request(side, 'connect')
        wait_until(lambda: all(w.snapshot().state == 'CONNECTED'
                               and w.snapshot().feedback is not None
                               for w in node.workers.values()), 10., '读取初始角度')
        for side, worker in node.workers.items():
            frame = worker.snapshot().feedback
            if frame is None or max(abs(v) for v in frame.velocities) > .02:
                raise RuntimeError(side + '缺少静止的初始反馈，请结束其他控制后再运行')
            backend = worker._backend
            if not isinstance(backend, HexPositionAdapter):
                raise RuntimeError(side + '连接尚未准备完成')
            arm = backend._arm()
            backend._health(require_calibrated=False, allow_startup_timeout=True)
            if backend._startup_timeout_only():
                print(side + '读取到上轮API通信超时停车；将在第3步按规则处理，当前未申请控制。',
                      flush=True)
            holder = arm.get_session_holder()
            if (holder > 0 and holder != arm.get_my_session_id()
                    and arm.get_parking_stop_detail().reason != 'API communication timeout'):
                raise RuntimeError(side + '仍有其他控制会话，未申请控制')

        print('3/4 按官方SDK流程准备位置模式，重新读取起点并保存归位角度', flush=True)
        for side in node.sides:
            node.handle_request(side, 'enable')
        wait_until(lambda: all(w.snapshot().state == 'HOLDING'
                               and w.snapshot().commanded_positions is not None
                               for w in node.workers.values()), 15., 'SDK准备')
        print('仅核对最新起点的接近、归位段及主体模型一致性，尚未开始播放', flush=True)
        node.queue_pair_preparation(request_text, task_file=True, prepared_core=prepared_core)
        wait_until(lambda: node._paired_phase == 'STAGED' or bool(node._file_rejection)
                   or node._paired_phase.startswith('REJECTED'),
                   node._preparation_timeout + 5., '完整路径检查')
        if node._paired_phase != 'STAGED':
            reason = node._file_rejection.get('reason') or node._paired_phase
            raise RuntimeError('路径未通过：' + str(reason))

        if cooperative:
            from .gripper_adapter import CooperativePositionAdapter

            backends: list[CooperativePositionAdapter] = []
            for worker in node.workers.values():
                backend = worker._backend
                if not isinstance(backend, CooperativePositionAdapter):
                    raise RuntimeError('协同任务缺少双侧夹爪接入')
                backends.append(backend)
            for backend in backends:
                backend.gripper.request_enable()
            wait_until(lambda: all(backend.gripper.status()['ready'] for backend in backends),
                       3., '夹爪初始闭合指令交付')
            print('两侧夹爪初始指令已交付；沿文件时序开合，不代表实际夹持确认。', flush=True)
        duration = node._paired_details['duration_s']
        print(f'4/4 连续播放：接近、{task_name}、归位，计划约{duration:.1f}秒；'
              f'运行偏差保护{tracking_limit}rad，不等待中间点到位', flush=True)
        node.request_paired_trajectory(expected_request_id=identifier)
        playback_started = time.monotonic()
        try:
            wait_until(lambda: node._paired_phase == 'FINISHED' and not node._paired_active,
                       duration + 120., '连续播放')
        finally:
            print(f'本轮播放实际耗时：{time.monotonic() - playback_started:.3f}秒', flush=True)
        for side, worker in node.workers.items():
            frame = worker.snapshot().feedback
            target = node._paired_details['return_targets'][side]
            if frame is not None:
                error = max(abs(a - b) for a, b in zip(frame.positions, target))
                print(f'{side} 归位目标最大角度差：{error:.6f}rad', flush=True)
        print('完整路径含归位段播放完成，正在关闭连接；不代表轨迹精度验收通过。', flush=True)
        return 0
    except KeyboardInterrupt:
        print('已取消：停止后续目标，尽力制动并关闭连接，不执行故障归位。', flush=True)
        return 130
    except Exception as error:
        print('执行结束：' + str(error), flush=True)
        if node is not None and node._paired_gate is not None:
            print('停止时播放进度：' + str(node._paired_gate.playback_status()), flush=True)
        return 1
    finally:
        if core_cancel is not None:
            core_cancel.set()
        if node is not None:
            for side, worker in node.workers.items():
                frame = worker.snapshot().feedback
                target = node._paired_details.get('return_targets', {}).get(side)
                if frame is not None and target is not None:
                    error = max(abs(a - b) for a, b in zip(frame.positions, target))
                    print(f'{side}退出前最后反馈与本轮归位目标最大差：{error:.6f}rad', flush=True)
        try:
            if node is not None:
                try:
                    node.close_workers()
                finally:
                    node.destroy_node()
        finally:
            if rclpy.ok():
                rclpy.shutdown()
            print(f'本轮连接至关闭总耗时：{time.monotonic() - run_started:.3f}秒', flush=True)


def resolve_inputs(
    run_dir: Path, prepared_dir: Path | None, limits_file: Path, *,
    trajectory_file: Path | None = None, csv_name: str = 'tracking.csv',
) -> tuple[Path, Path, Path]:
    """解析本轮目录并在创建节点之前检查必要文件，不创建SDK连接。"""
    source = run_dir.expanduser().resolve()
    workspace = source.parent.parent if source.parent.name == 'archer_results' else (
        Path.home() / 'archer_ws')
    directory = (prepared_dir or workspace / 'archer_results_trans' / source.name
                 ).expanduser().resolve()
    config = limits_file.expanduser().resolve()
    if not all((source / name).is_file() for name in (csv_name, 'summary.json')):
        raise ValueError(f'任务目录缺少{csv_name}或summary.json，未连接机械臂')
    prepared = trajectory_file or directory / (Path(csv_name).stem + '.json')
    report = prepared.with_name(prepared.stem + '_report.json')
    if not all(p.is_file() for p in (prepared, report)):
        raise ValueError(f'转换目录需要{prepared.name}及{report.name}，未连接机械臂')
    if not config.is_file():
        raise ValueError('找不到限位配置，未连接机械臂：' + str(config))
    return source, directory, config


def resolve_trajectory_input(
    trajectory_file: Path, run_dir: Path | None, limits_file: Path, *,
    expected_csv_name: str = 'tracking.csv',
) -> tuple[Path, Path, Path, Path]:
    """由轨迹定位同名报告和原始记录，在创建节点前核对所有文件哈希。"""
    from .task_file import json_mapping, read_snapshot

    prepared = trajectory_file.expanduser().resolve()
    if prepared.suffix.lower() != '.json' or prepared.stem.endswith('_report'):
        stem = Path(expected_csv_name).stem
        raise ValueError(f'请输入轨迹{stem}.json，不是{stem}_report.json')
    report_path = prepared.with_name(prepared.stem + '_report.json')
    report = json_mapping(read_snapshot(report_path, 50 * 1024 * 1024))
    csv_name = report.get('source_csv')
    if not isinstance(csv_name, str) or not csv_name.strip():
        raise ValueError('转换报告缺少source_csv，不能自动定位原始任务')
    original_csv = Path(csv_name).expanduser()
    if not original_csv.is_absolute():
        original_csv = report_path.parent / original_csv
    if original_csv.name != expected_csv_name:
        raise ValueError('完整真机入口仅执行画圆tracking.csv，不执行抓取记录'
                         if expected_csv_name == 'tracking.csv'
                         else '协同入口只接受state.csv对应的轨迹文件')
    source = run_dir if run_dir is not None else original_csv.parent
    source, directory, config = resolve_inputs(
        source, prepared.parent, limits_file, trajectory_file=prepared,
        csv_name=expected_csv_name)
    paths = ((prepared, 'output_sha256', 50), (source / expected_csv_name, 'csv_sha256', 50),
             (source / 'summary.json', 'summary_sha256', 2), (config, 'limits_sha256', 2))
    for path, field, maximum in paths:
        if hashlib.sha256(read_snapshot(path, maximum * 1024 * 1024)).hexdigest() != (
                report.get(field)):
            raise ValueError('文件与转换报告不一致，未连接机械臂：' + str(path))
    return source, directory, config, prepared
