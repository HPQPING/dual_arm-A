"""通过真实 ROS 服务和线程验证位置节点，设备始终使用离线替身."""

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json
from threading import Event, Thread
import time
from typing import Callable, cast
import unittest
from unittest.mock import Mock, patch

import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from std_srvs.srv import Trigger
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from archer_hardware.hex_adapter import JointFeedback
from archer_hardware.position_node import PositionControlNode, StagedPair
from archer_hardware.position_plan import (
    ContinuousPositionTrajectory, JointPositionTrajectory, PairedTrajectoryStart, PositionLimits,
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
        self._paired_plan: JointPositionTrajectory | ContinuousPositionTrajectory | None = None
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

    @property
    def commanded_positions(self) -> tuple[float, ...] | None:
        """替身保持目标独立于模拟传感器读数."""
        return self.goal if self.state == 'HOLDING' else None

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

    def recover_limits(self, duration: float) -> None:
        """替身仅在内存收回超限关节，不构造任何实际设备."""
        assert self.limits is not None
        if not self.limits.within_nominal(self.positions):
            self.move_to(self.limits.recovery_goal(self.positions), duration)

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
        elif self.state in ('RUNNING',) and self.complete_move.is_set():
            if (self._paired_plan is not None and self.trajectory_start_at is not None
                    and time.monotonic() < self.trajectory_start_at + self._paired_plan.duration):
                return self.state
            self.positions = self.goal
            self.state = 'HOLDING'
        return self.state

    def arm_continuous(
        self, plan: ContinuousPositionTrajectory, side: str, gate: PairedTrajectoryStart,
    ) -> None:
        """连续对象使用真实起点对齐，但执行只改变内存状态，不接触设备."""
        if self.reject_trajectory:
            raise ValueError('模拟一侧连续准备失败')
        assert self.limits is not None
        prepared = plan.aligned_start(
            self.positions, self.limits, self.options.get('max_excursion', .05),
        )
        self._paired_plan = prepared
        self.goal = prepared.goal
        self._gate = gate
        self.state = 'ARMED'
        self.calls.append(('continuous_armed', ()))
        gate.ready(side)

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

    def test_recovery_queued_before_stop_cannot_execute(self) -> None:
        """停止代次同样作废尚未执行的限位收回请求."""
        fake = FakeBackend()
        fake.state = 'HOLDING'
        worker = PositionWorker('left', lambda: fake, lambda reason: None)
        worker._backend = fake
        worker._publish()
        worker.submit('recover_limits', duration=2.0)
        command = worker._queue.get_nowait()
        worker.request_stop()
        worker._execute(command)
        self.assertEqual(worker._phase, 'REJECTED')
        self.assertEqual(fake.calls, [])

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

    def test_legacy_return_rejected_before_queue(self) -> None:
        """旧return已从调度器删除，不会隐藏触发单臂动作或改变当前运动."""
        fake = FakeBackend()
        fake.state = 'RUNNING'
        worker = PositionWorker('left', lambda: fake, lambda error: None)
        worker._backend = fake
        with self.assertRaisesRegex(ValueError, '未知控制请求'):
            worker.submit('return', duration=2.0)
        self.assertTrue(worker._queue.empty())
        self.assertEqual(fake.calls, [])
        self.assertEqual(fake.state, 'RUNNING')

    def test_failed_pair_prepare_faults_worker(self) -> None:
        """清理旧归位后，成对轨迹准备失败仍必须故障取消并通知调用者."""
        fake = FakeBackend()
        fake.state = 'HOLDING'
        faults = []
        worker = PositionWorker('left', lambda: fake, faults.append)
        worker._backend = fake
        job = WorkerCommand('pair', 'trajectory', 0, time.monotonic() + 1.0,
                            gate=PairedTrajectoryStart())
        with patch.object(fake, 'arm_trajectory', side_effect=ValueError('invalid pair')):
            worker._execute(job)
        self.assertEqual(fake.state, 'FAULT')
        self.assertEqual(worker._phase, 'FAILED')
        self.assertTrue(faults)
        self.assertIn(('stop', ()), fake.calls)

    def test_sdk_background_fault_is_forwarded_once(self) -> None:
        """后台已先锁存故障时，工作线程仍须通知双臂联锁，且不重复通知."""
        fake = FakeBackend()
        fake.state = 'FAULT'
        fake.last_error = 'SDK 编码失败'
        faults = []
        worker = PositionWorker('left', lambda: fake, faults.append)
        worker._backend = fake
        worker._cycle()
        worker._cycle()
        self.assertEqual(len(faults), 1)
        self.assertIn('SDK 编码失败', faults[0])


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
        self, configured: bool = True, *,
        allow_paired_trajectory: bool = False,
        allow_continuous_trajectory: bool = False,
        allow_task_file_trajectory: bool = False,
        task_joint_error_rad: float = 0.,
    ) -> PositionControlNode:
        """虚构测试限位只通过参数传入，不写入正式配置."""
        args = ['--ros-args', '-p',
                f'allow_paired_trajectory:={str(allow_paired_trajectory).lower()}', '-p',
                f'allow_continuous_trajectory:={str(allow_continuous_trajectory).lower()}', '-p',
                f'allow_task_file_trajectory:={str(allow_task_file_trajectory).lower()}']
        args.extend(['-p', f'task_joint_error_rad:={task_joint_error_rad}'])
        if configured:
            for side in ('left', 'right'):
                args.extend(['-p', f'{side}_limits_confirmed:=true'])
                for suffix, values in (
                    ('lower', [-2.0] * 6), ('upper', [2.0] * 6),
                    ('max_velocity', [0.1] * 6), ('max_acceleration', [0.2] * 6),
                ):
                    args.extend(['-p', f'{side}_{suffix}:={json.dumps(values)}'])
        rclpy.init(args=args)
        with patch('archer_hardware.position_node.HexPositionAdapter', FakeBackend), patch(
            'archer_hardware.position_node.create_preparation_resources',
            side_effect=lambda: (ThreadPoolExecutor(max_workers=1), None),
        ):
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
        self.assertTrue(predicate(), str([
            (node._paired_phase, node._interlock_reason)
            for node in self.nodes if isinstance(node, PositionControlNode)
        ]))

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
        """发布目标不运动，move只执行一次；halt不追加归位."""
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
        self.assertTrue(self.service('halt').success)
        self.assertFalse(any(kind == 'return' for kind, _ in backend.calls))

    def test_recovery_is_explicit_single_arm_and_clears_old_target(self) -> None:
        """连接使能不自动收回；单臂服务使用最新反馈且清除该臂旧单点目标."""
        node = self.make_node()
        self.assertFalse(self.service('left/recover_limits').success)
        self.enable_both(node)
        left = self.backend(node, 'left')
        right = self.backend(node, 'right')
        self.assertFalse(any(kind == 'move' for kind, _ in left.calls + right.calls))
        node.stage_target('left', self.target('left'))
        left.positions = (-2.001, 0.101) + (0.1,) * 4
        response = self.service('left/recover_limits')
        self.assertTrue(response.success)
        identifier = response.message.split()[1].split('，', 1)[0]

        def recovery_completed() -> bool:
            """不能把上一条使能请求的SUCCEEDED误当成这次收回已完成."""
            snapshot = node.workers['left'].snapshot()
            return snapshot.command_id == identifier and snapshot.phase == 'SUCCEEDED'

        self.wait_until(recovery_completed)
        self.assertEqual(left.goal, (-1.995, 0.101) + (0.1,) * 4)
        self.assertNotIn('left', node._targets)
        self.assertFalse(any(kind == 'move' for kind, _ in right.calls))
        self.assertFalse(self.service('left/move').success)
        self.assertTrue(self.service('halt').success)
        self.assertFalse(self.service('left/recover_limits').success)

    def test_recovery_nominal_reports_success_without_motion(self) -> None:
        """已在范围内的请求按本次编号完成，不能一直等待不存在的运动."""
        node = self.make_node()
        self.enable_both(node)
        self.assertTrue(self.service('left/recover_limits').success)
        self.wait_until(lambda: node.workers['left'].snapshot().command == 'recover_limits')
        snapshot = node.workers['left'].snapshot()
        self.assertEqual(snapshot.phase, 'SUCCEEDED')
        self.assertIn('未启动', snapshot.message)
        self.assertFalse(any(kind == 'move' for kind, _ in self.backend(node, 'left').calls))

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
        """底层单点完成后保持；任务归位必须已包含在上层完整计划中."""
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

    def test_removed_stop_does_not_interrupt_motion(self) -> None:
        """旧stop命令被拒绝，运行中halt可以取消运动."""
        node = self.make_node()
        self.enable_both(node)
        backend = self.backend(node, 'left')
        backend.complete_move.clear()
        node.stage_target('left', self.target('left'))
        self.assertTrue(self.service('left/move').success)
        self.wait_until(lambda: node.workers['left'].snapshot().state == 'RUNNING')
        with self.assertRaises(ValueError):
            node.handle_request(None, 'stop')
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
        with self.assertRaises(ValueError):
            node.handle_request(None, 'stop')
        self.assertFalse(self.service('left/enable').success)
        with self.assertRaises(ValueError):
            node.handle_request(None, 'reset')

    def test_only_fault_halt_service_is_exposed(self) -> None:
        """移除键盘stop归位接口；保留全局故障halt，归位属于任务轨迹."""
        node = self.make_node()
        names = {name for name, _ in node.get_service_names_and_types()}
        self.assertNotIn('/hardware/position/stop', names)
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
        with self.assertRaises(ValueError):
            node.handle_request(None, 'stop')
        self.assertFalse(self.service('left/connect').success)

    def test_complete_paired_cycle_returns_inside_plan(self) -> None:
        """完整轨迹含归位点，双臂正常结束保持，不追加独立return或close."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        plan = self.paired_plan()
        for side in node.sides:
            plan[side]['points'].append({
                'positions': [0.1] * 6,
                'time_from_start': {'sec': 0, 'nanosec': 400_000_000},
            })
        node.stage_paired_target(plan)
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: node._paired_phase == 'FINISHED')
        for side in node.sides:
            backend = self.backend(node, side)
            self.assertEqual(backend.positions, (0.1,) * 6)
            self.assertFalse(any(kind in ('return', 'close') for kind, _ in backend.calls))
        self.assertEqual(node._interlock_reason, '')

    def test_removed_return_request_never_queues_a_worker(self) -> None:
        """旧stop不会悄悄转成单臂归位或联锁，完整轨迹由任务层提供."""
        node = self.make_node()
        self.enable_both(node)
        with patch.object(node.workers['left'], 'submit') as submit:
            with self.assertRaises(ValueError):
                node.handle_request(None, 'stop')
            submit.assert_not_called()
        self.assertEqual(node._interlock_reason, '')

    def test_driver_ctrl_c_closes_workers_before_destroy(self) -> None:
        """Ctrl+C走驱动退出清理，不调用归位服务或启动新的任务."""
        from archer_hardware import position_node
        fake = Mock()
        order = []
        fake.close_workers.side_effect = lambda: order.append('close')
        fake.destroy_node.side_effect = lambda: order.append('destroy')
        with patch.object(position_node.rclpy, 'init'), patch.object(
            position_node, 'PositionControlNode', return_value=fake,
        ), patch.object(position_node.rclpy, 'spin', side_effect=KeyboardInterrupt), patch.object(
            position_node.rclpy, 'ok', return_value=True,
        ), patch.object(position_node.rclpy, 'shutdown') as shutdown:
            position_node.main([])
            shutdown.assert_called_once()
        self.assertEqual(order, ['close', 'destroy'])

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
        with self.assertRaises(ValueError):
            node.handle_request(None, 'stop')

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

    def test_paired_topic_stage_start_complete_and_hold(self) -> None:
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
        self.assertTrue(self.service('halt').success)
        self.assertEqual(left.positions, (0.1002,) * 6)
        self.assertEqual(right.positions, (0.1002,) * 6)

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
        with self.assertRaises(ValueError):
            node.handle_request(None, 'stop')
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
        with self.assertRaises(ValueError):
            node.handle_request(None, 'stop')

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

    def test_preparation_stage_reaches_status_without_implicit_enable(self) -> None:
        """连接只读取反馈，显式准备后才进入保持，阶段可从状态读取."""
        node = self.make_node()
        self.assertTrue(self.service('left/connect').success)
        self.wait_until(lambda: node.workers['left'].snapshot().feedback is not None)
        backend = self.backend(node, 'left')
        self.assertEqual(backend.options['readiness_timeout'], 10.0)
        self.assertEqual([kind for kind, _ in backend.calls], ['connect'])
        self.assertEqual(node.workers['left'].snapshot().readiness_stage, 'CONNECTED')
        node.publish_state()
        status = json.loads(node._last_status)
        self.assertEqual(status['arms']['left']['readiness_stage'], 'CONNECTED')
        self.assertTrue(self.service('left/enable').success)
        self.wait_until(lambda: node.workers['left'].snapshot().state == 'HOLDING')
        node.publish_state()
        self.assertEqual(json.loads(node._last_status)['arms']['left']['readiness_stage'], 'READY')

    def continuous_payload(self) -> dict:
        """合成小幅闭合路径，不使用实机或仿真任务初值."""
        from archer_control.continuous_trajectory import retime_recorded_pair
        import numpy as np

        q = np.repeat(np.asarray([.1, .1002, .1004, .1001, .1])[:, None], 6, axis=1)
        bound = PositionLimits((-2.,) * 6, (2.,) * 6, (.1,) * 6, (.2,) * 6)
        plan, _ = retime_recorded_pair(
            (0, 100000000, 200000000, 300000000, 400000000), q, q,
            [bound.lower] * 2, [bound.upper] * 2,
            [bound.max_velocity] * 2, [bound.max_acceleration] * 2,
        )
        return plan.payload()

    def test_continuous_entry_remains_closed_by_default(self) -> None:
        """只允许旧成对接口不能隐式打开连续接口，也不会因收到文件而运动."""
        node = self.make_node(allow_paired_trajectory=True)
        self.enable_both(node)
        node.stage_paired_target(self.continuous_payload())
        self.wait_until(lambda: node._preparation is None)
        self.assertIsNone(node._paired_target)
        self.assertIn('尚未显式允许', node._paired_phase)
        self.assertFalse(self.service('paired_move').success)

    def test_continuous_topic_preflight_then_explicit_shared_execution(self) -> None:
        """真实ROS只暂存后台结果，显式开始后两线程执行连续对象并共用启动时刻."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        self.enable_both(node)
        publisher = self.observer.create_publisher(String, '/hardware/position/paired_target', 1)
        self.wait_until(lambda: publisher.get_subscription_count() > 0)
        message = String()
        message.data = json.dumps(self.continuous_payload())
        publisher.publish(message)
        self.wait_until(lambda: node._paired_phase == 'STAGED')
        assert node._paired_target is not None
        self.assertIsNotNone(node._paired_target.continuous)
        for side in node.sides:
            self.assertFalse(any(kind == 'continuous_armed'
                                 for kind, _ in self.backend(node, side).calls))
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: node._paired_phase == 'FINISHED')
        left, right = (self.backend(node, side) for side in node.sides)
        self.assertIsNotNone(left.trajectory_start_at)
        self.assertEqual(left.trajectory_start_at, right.trajectory_start_at)
        self.assertEqual(left.positions, (.1,) * 6)
        self.assertEqual(right.positions, (.1,) * 6)
        self.assertTrue(all(any(kind == 'continuous_armed' for kind, _ in backend.calls)
                            for backend in (left, right)))
        self.assertEqual(node._interlock_reason, '')

    def slow_preflight(self, entered: Event, release: Event):
        """模拟不响应取消的慢库，证明即使迟到返回成功也无法复活旧结果."""
        from archer_hardware.position_node import prepare_pair_message

        def prepare(text, limits, excursion, allowed, cancelled):
            entered.set()
            if not release.wait(3.):
                raise RuntimeError('test did not release preparation')
            return prepare_pair_message(text, limits, excursion, allowed, Event())
        return prepare

    def test_halt_works_during_preflight_and_late_result_is_discarded(self) -> None:
        """预检未返回时ROS仍响应halt；迟到的成功结果不能暂存或执行."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        self.enable_both(node)
        entered, release = Event(), Event()
        with patch('archer_hardware.position_node.prepare_pair_message',
                   side_effect=self.slow_preflight(entered, release)):
            try:
                node.stage_paired_target(self.continuous_payload())
                self.assertTrue(entered.wait(1.))
                started = time.monotonic()
                self.assertTrue(self.service('halt').success)
                self.assertLess(time.monotonic() - started, 1.)
                self.assertFalse(release.is_set())
            finally:
                release.set()
            self.wait_until(lambda: node._preparation is None)
        self.assertIsNone(node._paired_target)
        self.assertFalse(self.service('paired_move').success)
        self.assertTrue(node._interlock_reason)

    def test_busy_new_message_cancels_old_without_queueing_another(self) -> None:
        """正在预检时收到新消息会作废旧结果，不积压新的任务或误用旧暂存."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        self.enable_both(node)
        entered, release = Event(), Event()
        with patch('archer_hardware.position_node.prepare_pair_message',
                   side_effect=self.slow_preflight(entered, release)) as prepare:
            try:
                node.stage_paired_target(self.continuous_payload())
                self.assertTrue(entered.wait(1.))
                first = node._preparation
                message = String()
                message.data = '{bad new json'
                node._paired_target_callback(message)
                self.assertIs(node._preparation, first)
                self.assertEqual(prepare.call_count, 1)
                self.assertFalse(self.service('paired_move').success)
            finally:
                release.set()
            self.wait_until(lambda: node._preparation is None)
        self.assertIsNone(node._paired_target)
        self.assertTrue(node._paired_phase.startswith('REJECTED'))

    def test_actual_start_is_rechecked_after_background_finishes(self) -> None:
        """预检期间实际角度变化超过容差，不能沿用提交时的反馈通过暂存."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        self.enable_both(node)
        entered, release = Event(), Event()
        with patch('archer_hardware.position_node.prepare_pair_message',
                   side_effect=self.slow_preflight(entered, release)):
            try:
                node.stage_paired_target(self.continuous_payload())
                self.assertTrue(entered.wait(1.))
                backend = self.backend(node, 'left')
                backend.positions = (.106,) * 6

                def moved_feedback():
                    frame = node.workers['left'].snapshot().feedback
                    return frame is not None and frame.positions == (.106,) * 6

                self.wait_until(moved_feedback)
            finally:
                release.set()
            self.wait_until(lambda: node._preparation is None)
        self.assertIsNone(node._paired_target)
        self.assertIn('实际初始角度', node._paired_phase)
        self.assertFalse(self.service('paired_move').success)

    def test_expired_feedback_rejects_completed_preflight(self) -> None:
        """后台结果正确但最新反馈失效时，仍必须拒绝暂存."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        self.enable_both(node)
        entered, release = Event(), Event()
        with patch('archer_hardware.position_node.prepare_pair_message',
                   side_effect=self.slow_preflight(entered, release)):
            try:
                node.stage_paired_target(self.continuous_payload())
                self.assertTrue(entered.wait(1.))
                snapshot = node.workers['left'].snapshot()
                assert snapshot.feedback is not None
                stale = replace(snapshot, feedback=replace(
                    snapshot.feedback, source_ns=time.perf_counter_ns() - 200000000,
                ))
                with patch.object(node.workers['left'], 'snapshot', return_value=stale):
                    release.set()
                    self.wait_until(lambda: node._preparation is None)
            finally:
                release.set()
        self.assertIsNone(node._paired_target)
        self.assertIn('新鲜静止反馈', node._paired_phase)

    def test_continuous_one_arm_failure_prevents_both_starts(self) -> None:
        """一侧执行前复核失败时取消共同门控，不能另一侧单独开始."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        self.enable_both(node)
        node.stage_paired_target(self.continuous_payload())
        self.wait_until(lambda: node._paired_phase == 'STAGED')
        self.backend(node, 'right').reject_trajectory = True
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: bool(node._interlock_reason))
        self.wait_until(lambda: all(w.snapshot().state in ('STOPPED', 'FAULT')
                                    for w in node.workers.values()))
        for side in node.sides:
            self.assertFalse(any(kind == 'trajectory_start'
                                 for kind, _ in self.backend(node, side).calls))

    def test_preflight_timeout_discards_late_success(self) -> None:
        """用测试时钟模拟超时，无需等待30秒，迟到结果仍不能进入暂存."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        self.enable_both(node)
        entered, release = Event(), Event()
        with patch('archer_hardware.position_node.prepare_pair_message',
                   side_effect=self.slow_preflight(entered, release)):
            try:
                node.stage_paired_target(self.continuous_payload())
                self.assertTrue(entered.wait(1.))
                node._preparation_started = time.monotonic() - 31.
                node.check_preparation_progress()
                self.assertIn('超过30秒', node._paired_phase)
                self.assertFalse(self.service('paired_move').success)
            finally:
                release.set()
            self.wait_until(lambda: node._preparation is None)
        self.assertIsNone(node._paired_target)
        self.assertEqual(node._interlock_reason, '')

    def test_start_tolerance_is_not_accumulated_between_staging_and_execution(self) -> None:
        """暂存时允许的微小差异不能再次叠加，使执行起点偏离文件超过容差."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        self.enable_both(node)
        entered, release = Event(), Event()
        with patch('archer_hardware.position_node.prepare_pair_message',
                   side_effect=self.slow_preflight(entered, release)):
            try:
                payload = self.continuous_payload()
                # 给首段足够时间容纳4毫弧度的对齐，单独测试容差不能累计。
                for side in ('left', 'right'):
                    for point in payload[side]['points']:
                        stamp = point['time_from_start']
                        ns = (stamp['sec'] * 10 ** 9 + stamp['nanosec']) * 5
                        point['time_from_start'] = {'sec': ns // 10 ** 9, 'nanosec': ns % 10 ** 9}
                        point['velocities'] = [v / 5. for v in point['velocities']]
                        point['accelerations'] = [a / 25. for a in point['accelerations']]
                node.stage_paired_target(payload)
                self.assertTrue(entered.wait(1.))
                self.backend(node, 'left').positions = (.104,) * 6

                def staged_feedback():
                    frame = node.workers['left'].snapshot().feedback
                    return frame is not None and frame.positions == (.104,) * 6

                self.wait_until(staged_feedback)
            finally:
                release.set()
            self.wait_until(lambda: node._paired_phase == 'STAGED')
        assert node._paired_target is not None and node._paired_target.continuous is not None
        self.assertEqual(node._paired_target.continuous['left'].initial, (.1,) * 6)
        self.backend(node, 'left').positions = (.108,) * 6
        self.assertTrue(self.service('paired_move').success)
        self.wait_until(lambda: bool(node._interlock_reason))
        for side in node.sides:
            self.assertFalse(any(kind == 'trajectory_start'
                                 for kind, _ in self.backend(node, side).calls))

    def file_preflight(self, text, limits, excursion, starts, cancelled, *, saved_initial=None,
                       hold_start=None, joint_error_rad=0., startup_joint_error_rad=0.,
                       control_hz=500):
        """合成三段数学路径，替代磁盘和几何耗时步骤，始终不访问设备."""
        from archer_hardware.task_geometry import assemble_task_cycle, prepare_approach
        from archer_hardware.task_geometry import prepare_return
        self.assertIn(control_hz, (250, 500))

        task = {side: ContinuousPositionTrajectory.prepare(
            (0, 500000000), ((.1,) * 6, (.105,) * 6),
            ((0.,) * 6,) * 2, ((0.,) * 6,) * 2, limits[side], excursion,
        ) for side in ('left', 'right')}
        approach = prepare_approach(task, starts if hold_start is None else hold_start,
                                    max_excursion=excursion)
        targets = {side: limits[side].return_goal((saved_initial or starts)[side])
                   for side in ('left', 'right')}
        home = prepare_return(task, targets, max_excursion=excursion)
        cycle = assemble_task_cycle(task, approach=approach, return_home=home,
                                    max_excursion=excursion, return_targets=targets)
        identifier = json.loads(text)['request_id']
        return StagedPair(cycle['left'].times_ns, {}, 0., cycle, {
            'request_id': identifier, 'preserve_path': True, 'task_type': 'split_circle',
            'duration_s': cycle['left'].duration,
        })

    def file_node(self, error: float = 0.) -> PositionControlNode:
        """仅向替身节点显式开放全部开关，不修改正式默认值."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True,
                              allow_task_file_trajectory=True, task_joint_error_rad=error)
        self.enable_both(node)
        return node

    def file_message(self, identifier='a' * 32) -> String:
        """请求仅用于替身后台；真实文件字段由专门的文件测试覆盖."""
        message = String()
        message.data = json.dumps({'request_id': identifier})
        return message

    def test_guarded_file_preparation_binds_raw_feedback_and_hold_target(self) -> None:
        """真实ROS线程快照分别保存反馈和保持目标，预检结束再核对目标."""
        node = self.file_node(.0001)
        for side in node.sides:
            self.backend(node, side).positions = (.10005,) * 6
        self.wait_until(lambda: all(
            (frame := node.workers[s].snapshot().feedback) is not None
            and frame.positions == (.10005,) * 6 for s in node.sides
        ))
        captured = {}

        def preflight(text, limits, excursion, starts, cancelled, **options):
            captured.update(starts=starts, **options)
            result = self.file_preflight(text, limits, excursion, options['hold_start'], cancelled)
            assert result.continuous is not None and result.details is not None
            return replace(result, continuous={
                side: replace(p, joint_error_rad=options['joint_error_rad'])
                for side, p in result.continuous.items()
            }, details={**result.details, 'joint_error_rad': options['joint_error_rad'],
                        'sampled_geometry': {'joint_error_rad': options['joint_error_rad'],
                                             'nominal_feedback_required': True}})

        with patch('archer_hardware.position_node.prepare_file_task_message',
                   side_effect=preflight):
            node._file_target_callback(self.file_message())
            self.wait_until(lambda: node._paired_phase == 'STAGED')
        self.assertEqual(captured['starts']['left'], (.10005,) * 6)
        self.assertEqual(captured['hold_start']['left'], (.1,) * 6)
        assert node._paired_target is not None and node._paired_target.continuous is not None
        self.assertEqual(node._paired_target.continuous['left'].initial, (.1,) * 6)
        self.backend(node, 'left').goal = (.10001,) * 6
        self.wait_until(lambda: node.workers['left'].snapshot().commanded_positions
                        == (.10001,) * 6)
        node._file_execute_callback(String(data='a' * 32))
        self.assertIn('保持指令目标', node._file_rejection['reason'])
        self.assertFalse(node._paired_active)

    def test_guarded_file_rejects_feedback_beyond_envelope_before_preflight(self) -> None:
        """超差立即拒绝，不启动后台计算，不沿用旧暂存."""
        node = self.file_node(.0001)
        self.backend(node, 'left').positions = (.1002,) * 6
        self.wait_until(lambda: (frame := node.workers['left'].snapshot().feedback) is not None
                        and frame.positions == (.1002,) * 6)
        with patch('archer_hardware.position_node.prepare_file_task_message') as prepare:
            node._file_target_callback(self.file_message())
            prepare.assert_not_called()
        self.assertIn('几何误差界限', node._file_rejection['reason'])

    def test_file_entry_default_closed(self) -> None:
        """普通连续许可不能自动开放文件入口，也不会因文件消息自动使能."""
        node = self.make_node(allow_paired_trajectory=True, allow_continuous_trajectory=True)
        with patch('archer_hardware.position_node.prepare_file_task_message') as prepare:
            node._file_target_callback(self.file_message())
        prepare.assert_not_called()
        self.assertIn('默认关闭', node._file_rejection['reason'])
        self.assertTrue(all(w._backend is None for w in node.workers.values()))

    def test_file_cycle_correlated_start_home_and_single_use(self) -> None:
        """文件经ROS提交后不运动，匹配编号才执行，完成含归位且不能重复消费."""
        node = self.file_node()
        publisher = self.observer.create_publisher(String, '/hardware/position/task_file', 1)
        self.wait_until(lambda: publisher.get_subscription_count() > 0)
        with patch('archer_hardware.position_node.prepare_file_task_message',
                   side_effect=self.file_preflight):
            publisher.publish(self.file_message())
            self.wait_until(lambda: node._paired_phase == 'STAGED')
        self.assertFalse(self.service('paired_move').success)
        node._file_execute_callback(String(data='b' * 32))
        self.assertIsNotNone(node._paired_target)
        for side in node.sides:
            self.assertFalse(any(kind == 'continuous_armed'
                                 for kind, _ in self.backend(node, side).calls))
        node._file_execute_callback(String(data='a' * 32))
        self.wait_until(lambda: node._paired_phase == 'FINISHED', timeout=5.)
        assert node._paired_gate is not None
        self.assertEqual(node._paired_gate.playback_status()['stream_period_s'], .004)
        for side in node.sides:
            backend = self.backend(node, side)
            self.assertEqual(backend.positions, (.1,) * 6)
            self.assertIsNotNone(backend._paired_plan)
            assert backend._paired_plan is not None
            self.assertEqual(backend._paired_plan.goal, (.1,) * 6)
            self.assertEqual(sum(kind == 'continuous_armed' for kind, _ in backend.calls), 1)
        self.assertEqual(self.backend(node, 'left').trajectory_start_at,
                         self.backend(node, 'right').trajectory_start_at)
        node._file_execute_callback(String(data='a' * 32))
        self.assertIn('没有有效成对暂存计划', node._file_rejection['reason'])

    def test_file_start_drift_after_staging_prevents_both_submissions(self) -> None:
        """即使漂移远小于旧容差，已检文件也必须重新预检而非重建首段."""
        node = self.file_node()
        with patch('archer_hardware.position_node.prepare_file_task_message',
                   side_effect=self.file_preflight):
            node._file_target_callback(self.file_message())
            self.wait_until(lambda: node._paired_phase == 'STAGED')
        self.backend(node, 'right').positions = (.100001,) * 6

        def changed():
            frame = node.workers['right'].snapshot().feedback
            return frame is not None and frame.positions == (.100001,) * 6

        self.wait_until(changed)
        node._file_execute_callback(String(data='a' * 32))
        self.assertIn('禁止重建首段', node._file_rejection['reason'])
        self.assertFalse(node._paired_active)
        for side in node.sides:
            self.assertFalse(any(kind == 'continuous_armed'
                                 for kind, _ in self.backend(node, side).calls))

    def test_file_snapshot_rejects_skew_without_changing_legacy_policy(self) -> None:
        """文件起点快照要求双臂时间差受控，旧通路仍按原新鲜度规则处理."""
        node = self.file_node()
        now = time.perf_counter_ns()
        snapshots = {side: node.workers[side].snapshot() for side in node.sides}
        left, right = snapshots['left'], snapshots['right']
        assert left.feedback is not None and right.feedback is not None
        left = replace(left, feedback=replace(left.feedback, source_ns=now - 25000000))
        right = replace(right, feedback=replace(right.feedback, source_ns=now))
        with patch.object(node.workers['left'], 'snapshot', return_value=left), \
                patch.object(node.workers['right'], 'snapshot', return_value=right):
            self.assertEqual(set(node._pair_feedback()), {'left', 'right'})
            with self.assertRaisesRegex(RuntimeError, '时间差超过20ms'):
                node._pair_feedback(coherent=True)

    def test_file_halt_during_preflight_discards_late_result(self) -> None:
        """文件后台仍在计算时halt立即结束本轮，迟到成功不能重新暂存."""
        node = self.file_node()
        entered, release = Event(), Event()

        def slow(*args, **kwargs):
            entered.set()
            if not release.wait(2.):
                raise RuntimeError('测试后台等待超时')
            return self.file_preflight(*args, **kwargs)

        with patch('archer_hardware.position_node.prepare_file_task_message', side_effect=slow):
            try:
                node._file_target_callback(self.file_message())
                self.assertTrue(entered.wait(1.))
                self.assertTrue(self.service('halt').success)
                self.assertFalse(node._preparation_valid)
            finally:
                release.set()
            self.wait_until(lambda: node._preparation is None)
        self.assertIsNone(node._paired_target)
        self.assertTrue(node._interlock_reason)

    def test_file_one_arm_rejection_cancels_both(self) -> None:
        """一侧执行线程拒绝完整文件时，另一侧不得越过共同启动门控."""
        node = self.file_node()
        with patch('archer_hardware.position_node.prepare_file_task_message',
                   side_effect=self.file_preflight):
            node._file_target_callback(self.file_message())
            self.wait_until(lambda: node._paired_phase == 'STAGED')
        self.backend(node, 'right').reject_trajectory = True
        node._file_execute_callback(String(data='a' * 32))
        self.wait_until(lambda: bool(node._interlock_reason))
        for side in node.sides:
            self.assertFalse(any(kind == 'trajectory_start'
                                 for kind, _ in self.backend(node, side).calls))

    def test_task_operator_file_cycle_over_ros(self) -> None:
        """真实ROS通道贯通操作客户端和位置节点，左右后端仅为内存替身."""
        from archer_hardware.operator_client import OperatorClient

        node = self.file_node()
        client = OperatorClient()
        self.nodes.append(client)
        assert self.executor is not None
        executor = self.executor
        thread = Thread(target=executor.spin, daemon=True)
        thread.start()
        try:
            with patch('archer_hardware.position_node.prepare_file_task_message',
                       side_effect=self.file_preflight):
                result = client.run_task_file({'request_id': 'c' * 32}, 3.)
            self.assertEqual(result['result'], 'SUCCEEDED')
            self.assertEqual(result['request_id'], 'c' * 32)
            for side in node.sides:
                calls = self.backend(node, side).calls
                self.assertEqual(sum(kind == 'connect' for kind, _ in calls), 1)
                self.assertEqual(sum(kind == 'enable' for kind, _ in calls), 1)
                self.assertEqual(sum(kind == 'continuous_armed' for kind, _ in calls), 1)
                self.assertFalse(any(kind == 'return' for kind, _ in calls))
        finally:
            executor.shutdown(timeout_sec=2.)
            thread.join(timeout=2.)
            self.executor = None


if __name__ == '__main__':
    unittest.main()
