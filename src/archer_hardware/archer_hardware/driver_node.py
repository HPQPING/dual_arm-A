"""发布双臂实机只读反馈，默认不连接设备，不接收运动指令."""

from __future__ import annotations

import json
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from .hex_adapter import HexFeedbackAdapter
from .position_plan import positive


class HardwareFeedbackNode(Node):
    """用独立硬件话题发布反馈，避免覆盖当前仿真反馈."""

    def __init__(self) -> None:
        """先校验所有参数，再按显式开关决定是否连接."""
        super().__init__('archer_hardware_feedback')
        self.declare_parameter('connect', False)
        self.declare_parameter('left_url', 'ws://192.168.1.100:8439')
        self.declare_parameter('right_url', 'ws://192.168.1.100:9439')
        self.declare_parameter('stale_timeout', 0.5)
        self.declare_parameter('startup_timeout', 10.0)
        connect = self.get_parameter('connect').value
        left = self.get_parameter('left_url').value
        right = self.get_parameter('right_url').value
        stale = positive(
            self.get_parameter('stale_timeout').value, 'stale_timeout',
        )
        startup = positive(
            self.get_parameter('startup_timeout').value, 'startup_timeout',
        )
        if type(connect) is not bool:
            raise ValueError('connect 必须为布尔值')
        if not isinstance(left, str) or not isinstance(right, str) or left == right:
            raise ValueError('左右臂地址必须是不同的字符串')
        if self.get_parameter('use_sim_time').value:
            raise ValueError('实机反馈节点不能使用仿真时间')
        self.adapters = {
            'left': HexFeedbackAdapter(left, stale),
            'right': HexFeedbackAdapter(right, stale),
        }
        self.enabled = connect
        self.startup_timeout = startup
        self.stale_timeout = stale
        self.started = time.monotonic()
        self.last_received: dict[str, float] = {}
        self.states = {side: 'DISCONNECTED' for side in self.adapters}
        self.feedback_pubs = {
            side: self.create_publisher(
                JointState, f'/hardware/{side}/joint_states', 10,
            ) for side in self.adapters
        }
        qos = QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.status_pub = self.create_publisher(
            String, '/hardware/status', qos,
        )
        self.previous_status = ''
        if connect:
            try:
                for side, adapter in self.adapters.items():
                    adapter.connect()
                    self.states[side] = 'WAITING'
            except Exception:
                self.close_connections()
                raise
        self.started = time.monotonic()
        self.create_timer(0.02, self.poll_feedback)
        self.publish_status()

    def publish_status(self) -> None:
        """启动及状态变化时发布可供晚加入订阅者读取的状态."""
        text = json.dumps(self.states, ensure_ascii=False, sort_keys=True)
        if text != self.previous_status:
            message = String()
            message.data = text
            self.status_pub.publish(message)
            self.get_logger().info(text)
            self.previous_status = text

    def poll_feedback(self) -> None:
        """仅发布新鲜的完整帧，异常时关闭对应连接并等待人工重启."""
        if not self.enabled:
            return
        now = time.monotonic()
        for side, adapter in self.adapters.items():
            if self.states[side].startswith('ERROR'):
                continue
            try:
                feedback = adapter.read()
                if feedback is None:
                    last = self.last_received.get(side)
                    deadline = (self.started + self.startup_timeout if last is None
                                else last + self.stale_timeout)
                    if now > deadline:
                        raise RuntimeError('等待首帧或新反馈超时')
                    continue
                message = JointState()
                # ROS 时间戳表示此节点收到数据的时刻，不冒充设备墙钟。
                message.header.stamp = self.get_clock().now().to_msg()
                message.name = [f'{side}_joint_{i}' for i in range(1, 7)]
                message.position = list(feedback.positions)
                message.velocity = list(feedback.velocities)
                message.effort = list(feedback.efforts)
                self.feedback_pubs[side].publish(message)
                self.last_received[side] = now
                self.states[side] = 'READY'
            except Exception as error:
                self.states[side] = f'ERROR: {error}'
                adapter.close()
        self.publish_status()

    def close_connections(self) -> None:
        """释放两侧连接，不触发回零或其他退出运动."""
        for adapter in self.adapters.values():
            adapter.close()


def main(args: list[str] | None = None) -> None:
    """启动只读节点，并在退出时清理连接及 ROS 上下文."""
    rclpy.init(args=args)
    node: HardwareFeedbackNode | None = None
    try:
        node = HardwareFeedbackNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close_connections()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
