"""通过真实 ROS 服务和线程验证位置节点，设备始终使用离线替身."""

from dataclasses import replace
import json
from threading import Event
import time
from typing import Callable, cast
import unittest
from unittest.mock import patch

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from archer_hardware.hex_adapter import JointFeedback
from archer_hardware.position_node import PositionControlNode
from archer_hardware.position_plan import (
    JointPositionTrajectory, PairedTrajectoryStart, PositionLimits,
)
from archer_hardware.position_worker import PositionWorker, WorkerCommand


class FakeBackend:
    """只记录调用并产生模拟反馈，没有 socket 或厂商 API 连接."""

    def __init__(self, url='', limits: PositionLimits | None = None, **kwargs):
        """测试参数不代表真实机械臂限位."""
        self.state = 'DISCONNECTED'
        self.readiness_stage = 'IDLE'
        self.options = kwargs
        self.limits = limits
        self.reject_trajectory = False
        self._gate: PairedTrajectoryStart | None = None
        self._paired_plan: JointPositionTrajectory | None = None
        self.trajectory_start_at: float | None = None
        self.trajectory_observed_start_at: float | None = None
        self.last_error = ''
        self.initial_positions: tuple[float, ...] | None = None
        self.positions = (0.1,) * 6
        self.goal = self.positions
        self.calls = []
        self.fail = Event()
        self.complete_move = Event()
        self.complete_move.set()
        self.block_connect = Event()
        self.block_connect.set()
        self.entered_connect = Event()

    def connect(self) -> None:
        """记录连接请求，必要时模拟阻塞初始化."""
        self.calls.append(('connect', ()))
        self.entered_connect.set()
        if not self.block_connect.wait(2.0):
            raise RuntimeError('模拟连接超时')
        self.state = 'CONNECTED'
        self.readiness_stage = 'CONNECTED'

    def read_feedback(self) -> JointFeedback:
        """源时间来自单调时钟，供真实节点执行时效校验."""
        if self.fail.is_set():
            raise RuntimeError('模拟反馈故障')
        return JointFeedback(self.positions, (0.0,) * 6, (0.0,) * 6,
                             time.perf_counter_ns())

    def request_control(self) -> None:
        """仅记录申请，确认由后续 tick 模拟."""
        if self.state != 'CONNECTED':
            raise RuntimeError('尚未连接')
        self.calls.append(('enable', ()))
        self.state = 'REQUESTING'
        self.readiness_stage = 'WAITING_SESSION'

    def move_to(self, goal: tuple[float, ...], duration: float) -> None:
        """只记录目标，不驱动任何设备."""
        if self.state != 'HOLDING':
            raise RuntimeError('不在 HOLDING 状态')
        self.calls.append(('move', goal))
        self.goal = goal
        self.state = 'RUNNING'

    def return_to_initial(self, duration: float) -> None:
        """模拟显式归位请求."""
        if self.initial_positions is None:
            raise RuntimeError('尚未记录初始位置')
        self.calls.append(('return', self.initial_positions))
        self.goal = self.initial_positions
        self.state = 'RETURNING'

    def arm_trajectory(self, times_ns, positions, side, gate) -> None:
        """验证真实轨迹数学，但启动和位置变化仅写入内存."""
        if self.reject_trajectory:
            raise ValueError('模拟一侧准备失败')
        assert self.limits is not None
        self._paired_plan = JointPositionTrajectory.prepare(
            times_ns, positions, self.limits, self.options.get('max_excursion', 0.05),
            actual_start=self.positions,
        )
        self.goal = self._paired_plan.goal
        self._gate = gate
        self.state = 'ARMED'
        gate.ready(side)

    def tick(self) -> str:
        """模拟取得控制和到位，轨迹数学由其他测试单独验证."""
        if self._gate is not None and self._gate.cancelled:
            raise RuntimeError('模拟成对取消')
        if self.state == 'ARMED':
            assert self._gate is not None
            start = self._gate.start_time()
            if start is None or time.monotonic() < start:
                return self.state
            self.trajectory_start_at = start
            self.trajectory_observed_start_at = time.monotonic()
            self.calls.append(('trajectory_start', (start,)))
            self.state = 'RUNNING'
        if self.state == 'REQUESTING':
            self.initial_positions = self.positions
            self.state = 'HOLDING'
            self.readiness_stage = 'READY'
        elif self.state in ('RUNNING', 'RETURNING') and self.complete_move.is_set():
            if (self._paired_plan is not None and self.trajectory_start_at is not None
                    and time.monotonic() < self.trajectory_start_at + self._paired_plan.duration):
                return self.state
            self.positions = self.goal
            self.state = 'HOLDING'
        return self.state

    def stop(self, *, fault: bool = False) -> None:
        """记录停止，不自动归位."""
        if self._gate is not None:
            self._gate.cancel()
        self.calls.append(('stop', ()))
        self.state = 'FAULT' if fault else 'STOPPED'

    def close(self) -> None:
        """记录关闭连接，不自动执行其他动作."""
        self.calls.append(('close', ()))
        self.state = 'DISCONNECTED'


class PositionWorkerTests(unittest.TestCase):
    """验证有界队列、停止优先和执行前状态检查."""

    def test_old_queued_command_cannot_survive_stop(self) -> None:
        """停止使旧代次的使能请求失效."""
        fake = FakeBackend()
        fake.state = 'CONNECTED'
        worker = PositionWorker('left', lambda: fake, lambda reason: None)
        worker._backend = fake
        worker._publish()
        worker.submit('enable')
        command = worker._queue.get_nowait()
        worker.request_stop()
        worker._execute(command)
        self.assertEqual(worker._phase, 'REJECTED')
        self.assertFalse(any(kind == 'enable' for kind, _ in fake.calls))

    def test_expired_command_is_rejected(self) -> None:
        """过期请求不执行 SDK 调用."""
        fake = FakeBackend()
        worker = PositionWorker('left', lambda: fake, lambda reason: None)
        job = WorkerCommand('old', 'connect', 0, time.monotonic() - 1.0)
        worker._execute(job)
        self.assertEqual(fake.calls, [])
        self.assertEqual(worker._phase, 'REJECTED')

    def test_queue_capacity_and_state_change(self) -> None:
        """队列不积压目标，出队后再次检查实际状态."""
        fake = FakeBackend()
        fake.state = 'CONNECTED'
        worker = PositionWorker('left', lambda: fake, lambda reason: None)
        worker._backend = fake
        worker._publish()
        worker.submit('enable')
        with self.assertRaisesRegex(RuntimeError, '等待处理'):
            worker.submit('enable')
        job = worker._queue.get_nowait()
        fake.state = 'STOPPED'
        worker._execute(job)
        self.assertEqual(worker._phase, 'REJECTED')
        self.assertEqual(fake.calls, [])

    def test_stop_during_connection_prevents_enable(self) -> None:
        """停止事件不会因普通队列阻塞而丢失."""
        fake = FakeBackend()
        fake.block_connect.clear()
        worker = PositionWorker('left', lambda: fake, lambda reason: None)
        worker.start()
        try:
            worker.submit('connect')
            self.assertTrue(fake.entered_connect.wait(1.0))
            worker.request_stop()
            fake.block_connect.set()
            deadline = time.monotonic() + 1.0
            while worker.snapshot().state != 'STOPPED' and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(worker.snapshot().state, 'STOPPED')
            self.assertFalse(any(kind in ('enable', 'move') for kind, _ in fake.calls))
        finally:
            fake.block_connect.set()
            worker.shutdown()

    def test_stopped_backend_still_reports_feedback_without_motion(self) -> None:
        """停止后继续读取实际状态，读取失败不重复触发联锁."""
        fake = FakeBackend()
        fake.state = 'STOPPED'
        faults = []
        worker = PositionWorker('left', lambda: fake, faults.append)
        worker._backend = fake
        worker._cycle()
        self.assertIsNotNone(worker._frame)
        self.assertEqual(fake.calls, [])
        fake.fail.set()
        worker._cycle()
        self.assertEqual(faults, [])
        self.assertEqual(fake.state, 'STOPPED')

    def test_failed_return_stops_original_motion(self) -> None:
        """归位预检失败后锁存故障，不能继续被用户取消的原动作."""
        fake = FakeBackend()
        fake.state = 'RUNNING'
        fake.initial_positions = (0.1,) * 6
        faults = []
        worker = PositionWorker('left', lambda: fake, faults.append)
        worker._backend = fake
        job = WorkerCommand('return', 'return', 0, time.monotonic() + 1.0, duration=2.0)
        with patch.object(fake, 'return_to_initial', side_effect=ValueError('减速距离超界')):
            worker._execute(job)
        self.assertEqual(fake.state, 'FAULT')
        self.assertEqual(worker._phase, 'FAILED')
        self.assertTrue(faults)
        self.assertIn(('stop', ()), fake.calls)


class PositionNodeTests(unittest.TestCase):
    """使用真实 ROS 消息及服务，位置后端始终被替换."""

    def setUp(self) -> None:
        """准备独立节点清理列表."""
        self.nodes: list[Node] = []
        self.executor: SingleThreadedExecutor | None = None

    def tearDown(self) -> None:
        """先停止工作线程，再释放 ROS 实体."""
        for node in self.nodes:
            if isinstance(node, PositionControlNode):
                node.close_workers()
        if self.executor is not None:
            self.executor.shutdown()
        for node in self.nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def make_node(
        self, configured: bool = True, *, allow_api_calibration: bool = False,
        allow_paired_trajectory: bool = False,
    ) -> PositionControlNode:
        """虚构测试限位只通过参数传入，不写入正式配置."""
        args = ['--ros-args', '-p',
                f'allow_api_calibration:={str(allow_api_calibration).lower()}', '-p',
                f'allow_paired_trajectory:={str(allow_paired_trajectory).lower()}']
        if configured:
            for side in ('left', 'right'):
                args.extend(['-p', f'{side}_limits_confirmed:=true'])
                for suffix, values in (
                    ('lower', [-2.0] * 6), ('upper', [2.0] * 6),
                    ('max_velocity', [0.1] * 6), ('max_acceleration', [0.2] * 6),
                ):
                    args.extend(['-p', f'{side}_{suffix}:={json.dumps(values)}'])
        rclpy.init(args=args)
        with patch('archer_hardware.position_node.HexPositionAdapter', FakeBackend):
            node = PositionControlNode()
        self.nodes.append(node)
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(node)
        self.observer = Node('position_test_observer')
        self.nodes.append(self.observer)
        self.executor.add_node(self.observer)
        return node

    def wait_until(self, predicate: Callable[[], bool], timeout: float = 3.0) -> None:
        """推进 ROS 通信，等待工作线程或服务结果."""
        assert self.executor is not None
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            self.executor.spin_once(timeout_sec=0.005)
        self.assertTrue(predicate())

    def service(self, name: str) -> Trigger.Response:
        """经真实 ROS 服务传输请求，返回受理结果."""
        client = self.observer.create_client(Trigger, '/hardware/position/' + name)
        self.assertTrue(client.wait_for_service(timeout_sec=2.0))
        future = client.call_async(Trigger.Request())
        self.wait_until(future.done)
        result = cast(Trigger.Response, future.result())
        self.observer.destroy_client(client)
        return result

    def backend(self, node: PositionControlNode, side: str) -> FakeBackend:
        """显式检查替身类型，避免对真实 SDK 类型访问测试字段."""
        backend = node.workers[side]._backend
        assert isinstance(backend, FakeBackend)
        return backend

    def enable_both(self, node: PositionControlNode) -> None:
        """显式连接后再显式使能，确认未发生自动使能."""
        for side in ('left', 'right'):
            self.assertTrue(self.service(side + '/connect').success)
            self.wait_until(lambda: node.workers[side].snapshot().feedback is not None)
            self.assertFalse(any(kind == 'enable' for kind, _ in self.backend(node, side).calls))
            self.assertTrue(self.service(side + '/enable').success)
            self.wait_until(lambda: node.workers[side].snapshot().state == 'HOLDING')

    def target(self, side: str, positions=None) -> JointTrajectory:
        """生成仅供测试的单点关节目标."""
        message = JointTrajectory()
        message.joint_names = [f'{side}_joint_{i}' for i in range(1, 7)]
        point = JointTrajectoryPoint()
        point.positions = [0.12] * 6 if positions is None else positions
        point.time_from_start.sec = 2
        message.points = [point]
        return message

    def test_default_unconfigured_and_late_status(self) -> None:
        """默认启动没有后端连接，晚加入订阅者仍可读取状态."""
        node = self.make_node(False)
        self.assertTrue(all(w._backend is None for w in node.workers.values()))
        self.assertFalse(self.service('left/connect').success)
        received = []

        def on_status(message: String) -> None:
            received.append(json.loads(message.data))

        qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        subscription = self.observer.create_subscription(
            String, '/hardware/position/status', on_status, qos,
        )
        self.wait_until(lambda: bool(received))
        self.assertIsNotNone(subscription)
        self.assertEqual(received[-1]['arms']['left']['state'], 'UNCONFIGURED')

    def test_target_topic_requires_explicit_move_and_is_single_use(self) -> None:
        """发布目标不运动，move 只执行一次，归位使用实际初始角度."""
        node = self.make_node()
        self.enable_both(node)
        publisher = self.observer.create_publisher(
            JointTrajectory, '/hardware/position/left/target', 1,
        )
        self.wait_until(lambda: publisher.get_subscription_count() > 0)
        publisher.publish(self.target('left'))
        self.wait_until(lambda: 'left' in node._targets)
        backend = self.backend(node, 'left')
        self.assertFalse(any(kind == 'move' for kind, _ in backend.calls))
        response = self.service('left/move')
        self.assertTrue(response.success)
        self.assertIn('已受理编号', response.message)
        self.wait_until(lambda: any(kind == 'move' for kind, _ in backend.calls))
        self.wait_until(lambda: node.workers['left'].snapshot().phase == 'SUCCEEDED')
        self.assertFalse(self.service('left/move').success)
        self.assertTrue(self.service('stop').success)
        self.wait_until(lambda: ('return', (0.1,) * 6) in backend.calls)

    def test_target_validation_and_reordering(self) -> None:
        """按关节名重新排序，拒绝重复名字、多点、其他控制字段和越界目标."""
        node = self.make_node()
        self.enable_both(node)
        message = self.target('left', [0.101, 0.102, 0.103, 0.104, 0.105, 0.106])
        message.joint_names = list(reversed(list(message.joint_names)))
        node.stage_target('left', message)
        self.assertEqual(node._targets['left'].positions[0], 0.106)
        for variant in ('duplicate', 'wrong_side', 'multiple', 'velocity', 'stamp', 'limit'):
            target = self.target('left')
            point = list(target.points)[0]
            if variant == 'duplicate':
                target.joint_names = ['left_joint_1'] * 6
            elif variant == 'wrong_side':
                target.joint_names = [f'right_joint_{i}' for i in range(1, 7)]
            elif variant == 'multiple':
                target.points = [point, point]
            elif variant == 'velocity':
                point.velocities = [0.1] * 6
            elif variant == 'stamp':
                target.header.stamp.sec = 1
            elif variant == 'limit':
                point.positions = [3.0] * 6
            with self.subTest(variant=variant), self.assertRaises(ValueError):
                node.stage_target('left', target)

    def test_completion_holds_goal_and_does_not_return_automatically(self) -> None:
        """正常完成后保持目标位置，归位仅在明确请求时发生."""
        node = self.make_node()
        self.enable_both(node)
        backend = self.backend(node, 'left')
        node.stage_target('left', self.target('left'))
        self.assertTrue(self.service('left/move').success)
        self.wait_until(lambda: backend.positions == (0.12,) * 6)
        self.wait_until(lambda: node.workers['left'].snapshot().phase == 'SUCCEEDED')
        self.assertEqual(node.workers['left'].snapshot().state, 'HOLDING')
        self.assertFalse(any(kind == 'return' for kind, _ in backend.calls))
        self.assertEqual(backend.initial_positions, (0.1,) * 6)

    def test_normal_stop_rejects_running_motion_without_return(self) -> None:
        """普通停止只用于完成后归位，运动中只接受独立 halt."""
        node = self.make_node()
        self.enable_both(node)
        backend = self.backend(node, 'left')
        backend.complete_move.clear()
        node.stage_target('left', self.target('left'))
        self.assertTrue(self.service('left/move').success)
        self.wait_until(lambda: node.workers['left'].snapshot().state == 'RUNNING')
        self.assertFalse(self.service('stop').success)
        self.assertFalse(any(kind == 'return' for kind, _ in backend.calls))
        self.assertEqual(node.workers['left'].snapshot().state, 'RUNNING')
        self.assertTrue(self.service('halt').success)
        self.wait_until(lambda: node.workers['left'].snapshot().state == 'STOPPED')

    def test_halt_neither_returns_nor_disconnects(self) -> None:
        """立即停止保持反馈连接，不触发归位；断开仍是独立操作."""
        node = self.make_node()
        self.enable_both(node)
        self.assertTrue(self.service('halt').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'STOPPED' for w in node.workers.values()
        ))
        for side in ('left', 'right'):
            calls = self.backend(node, side).calls
            self.assertFalse(any(kind in ('return', 'close') for kind, _ in calls))
        self.assertFalse(self.service('stop').success)
        self.assertFalse(self.service('left/enable').success)
        with self.assertRaises(ValueError):
            node.handle_request(None, 'reset')

    def test_only_two_stop_services_are_exposed(self) -> None:
        """移除复位和单臂停止别名，公开停止入口只有 stop/halt."""
        node = self.make_node()
        names = {name for name, _ in node.get_service_names_and_types()}
        self.assertIn('/hardware/position/stop', names)
        self.assertIn('/hardware/position/halt', names)
        self.assertNotIn('/hardware/position/reset', names)
        for side in ('left', 'right'):
            for kind in ('stop', 'halt', 'return_initial'):
                self.assertNotIn(f'/hardware/position/{side}/{kind}', names)

    def test_halt_is_accepted_before_any_connection(self) -> None:
        """即使未配置或未连接也可请求 halt，本次运行随后不可继续."""
        node = self.make_node(configured=False)
        self.assertTrue(self.service('halt').success)
        self.assertTrue(node._interlock_reason)
        self.assertFalse(self.service('stop').success)
        self.assertFalse(self.service('left/connect').success)

    def test_normal_stop_returns_both_without_ending_session(self) -> None:
        """运动完成后按普通停止，两臂归位并保持，不自动断开."""
        node = self.make_node()
        self.enable_both(node)
        for side in ('left', 'right'):
            node.stage_target(side, self.target(side))
            self.assertTrue(self.service(side + '/move').success)
        self.wait_until(lambda: all(
            w.snapshot().phase == 'SUCCEEDED'
            for w in node.workers.values()
        ))
        response = self.service('stop')
        self.assertTrue(response.success)
        identifiers = json.loads(response.message)['commands']
        self.assertEqual(set(identifiers), {'left', 'right'})
        self.wait_until(lambda: all(
            w.snapshot().command == 'return'
            and w.snapshot().phase == 'SUCCEEDED'
            for w in node.workers.values()
        ))
        self.wait_until(lambda: not node._return_active)
        self.assertEqual(node._interlock_reason, '')
        for side in ('left', 'right'):
            backend = self.backend(node, side)
            self.assertEqual(backend.positions, (0.1,) * 6)
            self.assertFalse(any(kind == 'close' for kind, _ in backend.calls))

    def test_partial_return_acceptance_halts_both(self) -> None:
        """一侧归位入队失败，不能留另一侧独自继续执行."""
        node = self.make_node()
        self.enable_both(node)
        with patch.object(node.workers['right'], 'submit', side_effect=RuntimeError('queue busy')):
            self.assertFalse(self.service('stop').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'STOPPED'
            for w in node.workers.values()
        ))
        self.assertTrue(node._interlock_reason)

    def test_halt_cancels_return_and_never_allows_resume(self) -> None:
        """归位过程中 halt 仍可抢占；结束后没有复位继续入口."""
        node = self.make_node()
        self.enable_both(node)
        for side in ('left', 'right'):
            self.backend(node, side).complete_move.clear()
        self.assertTrue(self.service('stop').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'RETURNING'
            for w in node.workers.values()
        ))
        with self.assertRaises(RuntimeError):
            node.stage_target('left', self.target('left'))
        self.assertFalse(self.service('stop').success)
        self.assertTrue(self.service('halt').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'STOPPED'
            for w in node.workers.values()
        ))
        self.assertFalse(node._return_active)
        self.assertFalse(self.service('stop').success)

    def test_expired_target_never_moves(self) -> None:
        """目标过期后必须重新发布，不能重用旧坐标."""
        node = self.make_node()
        self.enable_both(node)
        node.stage_target('left', self.target('left'))
        with node._lock:
            staged = node._targets['left']
            node._targets['left'] = replace(staged, received=time.monotonic() - 10.0)
        self.assertFalse(self.service('left/move').success)
        self.assertNotIn('left', node._targets)
        self.assertFalse(any(k == 'move' for k, _ in self.backend(node, 'left').calls))

    def test_fault_ends_session_even_after_disconnect(self) -> None:
        """故障联停后允许清理连接，但不能清锁、重连或恢复任务."""
        node = self.make_node()
        self.enable_both(node)
        node.stage_target('right', self.target('right'))
        self.backend(node, 'left').fail.set()
        self.wait_until(lambda: node.workers['right'].snapshot().state == 'STOPPED')
        self.assertEqual(node.workers['left'].snapshot().state, 'FAULT')
        self.assertEqual(node._targets, {})
        self.assertFalse(self.service('right/enable').success)
        with self.assertRaises(ValueError):
            node.handle_request(None, 'reset')
        for side in ('left', 'right'):
            self.assertTrue(self.service(side + '/disconnect').success)
            self.wait_until(lambda: node.workers[side].snapshot().state == 'DISCONNECTED')
        self.assertTrue(node._interlock_reason)
        self.assertFalse(self.service('left/connect').success)
        self.assertFalse(self.service('stop').success)

    def test_manual_stop_and_immutable_parameters(self) -> None:
        """停止请求联锁两臂，运行时不能改限位或启用仿真时钟."""
        node = self.make_node()
        self.enable_both(node)
        self.assertTrue(self.service('halt').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'STOPPED' for w in node.workers.values()
        ))
        self.assertFalse(self.service('right/enable').success)
        result = node.set_parameters([Parameter('use_sim_time', value=True)])
        self.assertFalse(result[0].successful)
        result = node.set_parameters([Parameter('max_excursion', value=1.0)])
        self.assertFalse(result[0].successful)
        result = node.set_parameters([Parameter('allow_api_calibration', value=True)])
        self.assertFalse(result[0].successful)
        result = node.set_parameters([Parameter('allow_paired_trajectory', value=True)])
        self.assertFalse(result[0].successful)

    def paired_plan(self) -> dict:
        """短小合成计划，不是真机任务或实机初值."""
        return {'version': 1, 'interpolation': 'quintic_stop', **{
            side: {'joint_names': [f'{side}_joint_{i}' for i in range(1, 7)], 'points': [
                {'positions': [0.1] * 6, 'time_from_start': {'sec': 0, 'nanosec': 0}},
                {'positions': [0.1002] * 6,
                 'time_from_start': {'sec': 0, 'nanosec': 200_000_000}},
            ]} for side in ('left', 'right')
        }}

    def test_paired_default_closed(self) -> None:
        """默认关闭成对接口，不会因计划到达而连接或执行."""
        node = self.make_node()
        with self.assertRaisesRegex(RuntimeError, '显式允许'):
            node.stage_paired_target(self.paired_plan())
        self.assertFalse(self.service('paired_move').success)
        self.assertTrue(all(w._backend is None for w in node.workers.values()))

    def test_paired_topic_stage_start_complete_and_explicit_return(self) -> None:
        """经真实ROS话题和服务暂存再执行，双方共用启动时间，完成后保持."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        publisher = self.observer.create_publisher(String, '/hardware/position/paired_target', 1)
        self.wait_until(lambda: publisher.get_subscription_count() > 0)
        message = String()
        message.data = json.dumps(self.paired_plan())
        publisher.publish(message)
        self.wait_until(lambda: node._paired_target is not None)
        self.assertTrue(all(self.backend(node, side).state == 'HOLDING' for side in node.sides))
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: node._paired_phase == 'FINISHED')
        left, right = (self.backend(node, side) for side in node.sides)
        self.assertIsNotNone(left.trajectory_start_at)
        self.assertEqual(left.trajectory_start_at, right.trajectory_start_at)
        self.assertEqual(left.positions, (0.1002,) * 6)
        self.assertEqual(right.positions, (0.1002,) * 6)
        self.assertFalse(any(kind == 'return' for kind, _ in left.calls))
        self.assertTrue(self.service('stop').success)
        self.wait_until(lambda: not node._return_active)
        self.assertEqual(left.positions, (0.1,) * 6)
        self.assertEqual(right.positions, (0.1,) * 6)

    def test_one_prepare_rejection_prevents_both_starts(self) -> None:
        """一侧执行线程拒绝准备时共同门控不释放，两侧均取消."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        self.backend(node, 'right').reject_trajectory = True
        node.stage_paired_target(self.paired_plan())
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: bool(node._interlock_reason))
        self.wait_until(lambda: all(w.snapshot().state in ('STOPPED', 'FAULT')
                                    for w in node.workers.values()))
        for side in node.sides:
            self.assertFalse(any(kind == 'trajectory_start'
                                 for kind, _ in self.backend(node, side).calls))

    def test_halt_before_shared_start_prevents_trajectory(self) -> None:
        """双方已准备但共同时间尚未来到，halt作废两侧及暂存计划."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        node.stage_paired_target(self.paired_plan())
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: all(w.snapshot().state == 'ARMED' for w in node.workers.values()))
        self.assertFalse(self.service('paired_move').success)
        self.assertFalse(self.service('left/enable').success)
        self.assertFalse(self.service('stop').success)
        self.assertTrue(self.service('halt').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'STOPPED' for w in node.workers.values()
        ))
        for side in node.sides:
            self.assertFalse(any(kind == 'trajectory_start'
                                 for kind, _ in self.backend(node, side).calls))
        self.assertFalse(self.service('paired_move').success)

    def test_expired_pair_is_consumed_without_submitting_motion(self) -> None:
        """暂存超时不能执行，也不能再次使用同一份过期暂存."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        node.stage_paired_target(self.paired_plan())
        assert node._paired_target is not None
        node._paired_target = replace(node._paired_target,
                                      received=time.monotonic() - node.target_timeout - 1)
        self.assertFalse(self.service('paired_move').success)
        self.assertIsNone(node._paired_target)
        self.assertFalse(node._paired_active)

    def test_runtime_fault_cancels_both_and_cannot_resume(self) -> None:
        """一侧执行中反馈出错时全局联停，不能重新运行或自动归位."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        for side in node.sides:
            self.backend(node, side).complete_move.clear()
        node.stage_paired_target(self.paired_plan())
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'RUNNING' for w in node.workers.values()
        ))
        self.backend(node, 'left').fail.set()
        self.wait_until(lambda: bool(node._interlock_reason))
        self.wait_until(lambda: all(w.snapshot().state in ('STOPPED', 'FAULT')
                                    for w in node.workers.values()))
        self.assertFalse(self.service('paired_move').success)
        self.assertFalse(self.service('stop').success)

    def test_bad_paired_message_invalidates_previous_stage(self) -> None:
        """收到无效新消息后不能误执行此前有效但已被替换的计划."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        node.stage_paired_target(self.paired_plan())
        message = String()
        message.data = '{invalid json'
        node._paired_target_callback(message)
        self.assertIsNone(node._paired_target)
        self.assertFalse(self.service('paired_move').success)

    def test_stale_worker_snapshot_cancels_pair(self) -> None:
        """即使工作线程未主动报告故障，ROS监视发现过期反馈仍取消两侧."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        for side in node.sides:
            self.backend(node, side).complete_move.clear()
        node.stage_paired_target(self.paired_plan())
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'RUNNING' for w in node.workers.values()
        ))
        snapshot = node.workers['left'].snapshot()
        assert snapshot.feedback is not None
        stale = replace(snapshot, feedback=replace(
            snapshot.feedback, source_ns=time.perf_counter_ns() - 200_000_000,
        ))
        with patch.object(node.workers['left'], 'snapshot', return_value=stale):
            node.check_paired_progress()
        self.assertIn('新鲜反馈', node._interlock_reason)
        self.wait_until(lambda: all(w.snapshot().state in ('STOPPED', 'FAULT')
                                    for w in node.workers.values()))

    def test_calibration_configuration_and_stage_reach_status(self) -> None:
        """显式许可只传入后端配置，连接不等于使能，阶段可从状态读取."""
        node = self.make_node(allow_api_calibration=True)
        self.assertTrue(self.service('left/connect').success)
        self.wait_until(lambda: node.workers['left'].snapshot().feedback is not None)
        backend = self.backend(node, 'left')
        self.assertTrue(backend.options['allow_api_calibration'])
        self.assertEqual(backend.options['readiness_timeout'], 10.0)
        self.assertEqual([kind for kind, _ in backend.calls], ['connect'])
        self.assertEqual(node.workers['left'].snapshot().readiness_stage, 'CONNECTED')
        node.publish_state()
        status = json.loads(node._last_status)
        self.assertEqual(status['arms']['left']['readiness_stage'], 'CONNECTED')
        self.assertTrue(status['arms']['left']['allow_api_calibration'])


if __name__ == '__main__':
    unittest.main()
