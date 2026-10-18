"""在真实 ROS 通信中验证只读节点，设备连接始终用替身隔离."""

import json
import time
import unittest
from unittest.mock import Mock, patch

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from archer_hardware.driver_node import HardwareFeedbackNode, main
from archer_hardware.position_plan import positive as positive_seconds
from archer_hardware.hex_adapter import JointFeedback


class FakeAdapter:
    """只提供一帧数据，不含任何网络调用."""

    def __init__(self, url, stale_timeout):
        """根据地址区分左右臂的反馈值."""
        self.connected = False
        self.closed = False
        self.error = False
        value = 0.1 if ':8439' in url else -0.1
        self.frame: JointFeedback | None = JointFeedback(
            (value,) * 6, (0.0,) * 6, (0.0,) * 6,
            time.perf_counter_ns(),
        )

    def connect(self):
        """仅记录连接请求."""
        self.connected = True

    def read(self):
        """模拟单次有效反馈和断线."""
        if self.error:
            raise RuntimeError('模拟通信中断')
        result, self.frame = self.frame, None
        return result

    def close(self):
        """仅记录清理请求."""
        self.closed = True


class FeedbackNodeTests(unittest.TestCase):
    """验证默认不连接、晚订阅状态、关节反馈及超时处理."""

    def setUp(self):
        """每项测试使用独立 ROS 上下文."""
        self.nodes = []

    def tearDown(self):
        """释放所有测试节点，连接替身不会访问设备."""
        for node in self.nodes:
            if isinstance(node, HardwareFeedbackNode):
                node.close_connections()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    def make_node(self, connect):
        """在构造节点前替换连接适配器."""
        rclpy.init(args=[
            '--ros-args', '-p', f'connect:={str(connect).lower()}',
        ])
        with patch('archer_hardware.driver_node.HexFeedbackAdapter', FakeAdapter):
            node = HardwareFeedbackNode()
        self.nodes.append(node)
        return node

    def test_timeout_parameter_validation(self):
        """显式拒绝空值、布尔值、非数值及无效超时."""
        for value in (None, True, '0.5', [0.5], 0, -1,
                      float('nan'), float('inf'), 10 ** 1000):
            with self.subTest(value=value), self.assertRaises(ValueError):
                positive_seconds(value, 'stale_timeout')
        for value in (1, 0.5):
            result = positive_seconds(value, 'stale_timeout')
            self.assertIsInstance(result, float)
            self.assertEqual(result, value)

    def test_default_never_connects(self):
        """默认启动不连接设备，状态可以被晚加入订阅者接收."""
        node = self.make_node(False)
        for adapter in node.adapters.values():
            assert isinstance(adapter, FakeAdapter)
            self.assertFalse(adapter.connected)
        observer = Node('feedback_test_observer')
        self.nodes.append(observer)
        received = []
        qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        def on_status(message: String) -> None:
            received.append(message.data)

        subscription = observer.create_subscription(
            String, '/hardware/status', on_status, qos,
        )
        deadline = time.monotonic() + 3.0
        while not received and time.monotonic() < deadline:
            rclpy.spin_once(observer, timeout_sec=0.05)
        self.assertIsNotNone(subscription)
        self.assertEqual(json.loads(received[-1]), {
            'left': 'DISCONNECTED', 'right': 'DISCONNECTED',
        })

    def test_joint_messages_and_fault(self):
        """验证左右关节顺序、实际数值和故障后停止发布."""
        node = self.make_node(True)
        observer = Node('feedback_test_joint_observer')
        self.nodes.append(observer)
        received = {'left': [], 'right': []}

        def left_callback(msg: JointState) -> None:
            received['left'].append(msg)

        def right_callback(msg: JointState) -> None:
            received['right'].append(msg)

        subscriptions = [
            observer.create_subscription(
                JointState, '/hardware/left/joint_states', left_callback, 10),
            observer.create_subscription(
                JointState, '/hardware/right/joint_states', right_callback, 10),
        ]
        # 等待 DDS 匹配，然后手动执行一次轮询，不启动设备线程。
        deadline = time.monotonic() + 3.0
        while (any(p.get_subscription_count() == 0
                   for p in node.feedback_pubs.values())
               and time.monotonic() < deadline):
            rclpy.spin_once(observer, timeout_sec=0.05)
        node.poll_feedback()
        deadline = time.monotonic() + 3.0
        while not all(received.values()) and time.monotonic() < deadline:
            rclpy.spin_once(observer, timeout_sec=0.05)
        self.assertEqual(len(subscriptions), 2)
        for side, value in (('left', 0.1), ('right', -0.1)):
            message = received[side][0]
            self.assertEqual(list(message.name), [
                f'{side}_joint_{i}' for i in range(1, 7)
            ])
            self.assertEqual(list(message.position), [value] * 6)
        left_adapter = node.adapters['left']
        assert isinstance(left_adapter, FakeAdapter)
        left_adapter.error = True
        node.poll_feedback()
        self.assertTrue(node.states['left'].startswith('ERROR'))
        self.assertTrue(left_adapter.closed)
        # 右臂没有新帧，也不得用旧角度刷新就绪状态。
        node.last_received['right'] -= 1.0
        node.poll_feedback()
        self.assertTrue(node.states['right'].startswith('ERROR'))

    def test_first_frame_timeout(self):
        """没有首帧时等待超时，不用固定关节角代替."""
        node = self.make_node(True)
        for adapter in node.adapters.values():
            assert isinstance(adapter, FakeAdapter)
            adapter.frame = None
        node.started -= node.startup_timeout + 1.0
        node.poll_feedback()
        self.assertTrue(all(s.startswith('ERROR') for s in node.states.values()))

    def test_poll_cleanup_failure_does_not_skip_other_arm(self):
        """一侧断线且关闭失败时保留错误说明，另一侧仍处理当前帧."""
        node = self.make_node(True)
        left = node.adapters['left']
        assert isinstance(left, FakeAdapter)
        left.error = True
        with patch.object(left, 'close', side_effect=RuntimeError('close failed')):
            node.poll_feedback()
        self.assertIn('模拟通信中断', node.states['left'])
        self.assertIn('连接清理未确认', node.states['left'])
        self.assertEqual(node.states['right'], 'READY')
        self.assertIn('right', node.last_received)

    def test_shutdown_attempts_both_connections(self):
        """一侧清理失败不能跳过另一侧，也不能把清理结果报告为成功."""
        node = self.make_node(True)
        with patch.object(node.adapters['left'], 'close',
                          side_effect=RuntimeError('left close failed')):
            with self.assertRaisesRegex(RuntimeError, 'left: left close failed'):
                node.close_connections()
        right = node.adapters['right']
        assert isinstance(right, FakeAdapter)
        self.assertTrue(right.closed)

    def test_partial_startup_failure_cleans_both_sides(self):
        """第二侧连接失败时尝试两侧清理，并保留原连接和清理异常."""
        rclpy.init(args=['--ros-args', '-p', 'connect:=true'])
        left = FakeAdapter('ws://127.0.0.1:8439', 0.5)
        right = FakeAdapter('ws://127.0.0.1:9439', 0.5)
        original_destroy = Node.destroy_node
        destroyed = []

        def cleanup(instance):
            destroyed.append(instance)
            return original_destroy(instance)

        with patch('archer_hardware.driver_node.HexFeedbackAdapter', side_effect=[left, right]):
            with patch.object(right, 'connect', side_effect=RuntimeError('right connect failed')):
                with patch.object(left, 'close', side_effect=RuntimeError('left close failed')):
                    with patch.object(HardwareFeedbackNode, 'destroy_node', cleanup):
                        with self.assertRaisesRegex(
                            RuntimeError, 'right connect failed.*left close failed',
                        ):
                            HardwareFeedbackNode()
        self.assertEqual(len(destroyed), 1)
        self.assertTrue(right.closed)

    def test_main_releases_ros_after_connection_cleanup_error(self):
        """退出清理抛错时仍销毁节点和关闭 ROS，异常继续报告给调用者."""
        node = Mock()
        node.close_connections.side_effect = RuntimeError('close failed')
        with patch('archer_hardware.driver_node.HardwareFeedbackNode', return_value=node):
            with patch('archer_hardware.driver_node.rclpy.spin', side_effect=KeyboardInterrupt):
                with self.assertRaisesRegex(RuntimeError, 'close failed'):
                    main(args=[])
        node.destroy_node.assert_called_once()
        self.assertFalse(rclpy.ok())


if __name__ == '__main__':
    unittest.main()
