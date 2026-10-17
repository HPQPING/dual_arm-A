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
from archer_hardware.position_worker import PositionWorker, WorkerCommand


class FakeBackend:
    """只记录调用并产生模拟反馈，没有 socket 或厂商 API 连接."""

    def __init__(self, url='', limits=None, **kwargs):
        """测试参数不代表真实机械臂限位."""
        self.state = 'DISCONNECTED'
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

    def tick(self) -> str:
        """模拟取得控制和到位，轨迹数学由其他测试单独验证."""
        if self.state == 'REQUESTING':
            self.initial_positions = self.positions
            self.state = 'HOLDING'
        elif self.state in ('RUNNING', 'RETURNING') and self.complete_move.is_set():
            self.positions = self.goal
            self.state = 'HOLDING'
        return self.state

    def stop(self, *, fault: bool = False) -> None:
        """记录停止，不自动归位."""
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

    def make_node(self, configured: bool = True) -> PositionControlNode:
        """虚构测试限位只通过参数传入，不写入正式配置."""
        args = []
        if configured:
            args = ['--ros-args']
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
        self.assertTrue(self.service('left/return_initial').success)
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

    def test_return_request_during_running_motion(self) -> None:
        """运行期间可以显式切换为归位，不重新采样覆盖本轮初始角度."""
        node = self.make_node()
        self.enable_both(node)
        backend = self.backend(node, 'left')
        backend.complete_move.clear()
        node.stage_target('left', self.target('left'))
        self.assertTrue(self.service('left/move').success)
        self.wait_until(lambda: node.workers['left'].snapshot().state == 'RUNNING')
        self.assertTrue(self.service('left/return_initial').success)
        self.wait_until(lambda: node.workers['left'].snapshot().state == 'RETURNING')
        backend.complete_move.set()
        self.wait_until(lambda: node.workers['left'].snapshot().state == 'HOLDING')
        self.assertIn(('return', (0.1,) * 6), backend.calls)

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

    def test_one_arm_fault_stops_both_and_requires_disconnect_reset(self) -> None:
        """一臂反馈故障联锁两臂，复位不自动连接或继续旧目标."""
        node = self.make_node()
        self.enable_both(node)
        node.stage_target('right', self.target('right'))
        self.backend(node, 'left').fail.set()
        self.wait_until(lambda: node.workers['right'].snapshot().state == 'STOPPED')
        self.assertEqual(node.workers['left'].snapshot().state, 'FAULT')
        self.assertEqual(node._targets, {})
        self.assertFalse(self.service('right/enable').success)
        self.assertFalse(self.service('reset').success)
        for side in ('left', 'right'):
            self.assertTrue(self.service(side + '/disconnect').success)
            self.wait_until(lambda: node.workers[side].snapshot().state == 'DISCONNECTED')
        self.assertTrue(self.service('reset').success)
        self.assertEqual(node._interlock_reason, '')
        self.assertTrue(all(w._backend is None for w in node.workers.values()))

    def test_manual_stop_and_immutable_parameters(self) -> None:
        """停止请求联锁两臂，运行时不能改限位或启用仿真时钟."""
        node = self.make_node()
        self.enable_both(node)
        self.assertTrue(self.service('left/stop').success)
        self.wait_until(lambda: all(
            w.snapshot().state == 'STOPPED' for w in node.workers.values()
        ))
        self.assertFalse(self.service('right/enable').success)
        result = node.set_parameters([Parameter('use_sim_time', value=True)])
        self.assertFalse(result[0].successful)
        result = node.set_parameters([Parameter('max_excursion', value=1.0)])
        self.assertFalse(result[0].successful)


if __name__ == '__main__':
    unittest.main()
