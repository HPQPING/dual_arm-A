"""Display GP80 fingertip coordinates in RViz, without changing its mesh."""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray


def opening_from_position(position: float) -> float:
    """Simulation mapping: 0.0 -> 80 mm open; 5.65 -> zero gap."""
    if not math.isfinite(position) or not 0.0 <= position <= 5.65:
        raise ValueError('Gripper position must be finite and between 0.0 and 5.65')
    return 0.080 * (1.0 - position / 5.65)


def fingertip_coordinates(position: float, tcp_xyz: list[float]):
    """Fingertips symmetric about TCP, along the gripper local Y axis."""
    if len(tcp_xyz) != 3 or not all(math.isfinite(v) for v in tcp_xyz):
        raise ValueError('tcp_xyz must contain three finite values')
    half_gap = opening_from_position(position) / 2.0
    x, y, z = tcp_xyz
    return (x, y + half_gap, z), (x, y - half_gap, z)


class GripperMarkers(Node):
    def __init__(self):
        super().__init__('gripper_markers')
        self.declare_parameter('side', 'left')
        self.declare_parameter('tcp_xyz', [0.0, 0.0, 0.147])
        self.declare_parameter('initial_position', 0.0)
        side = self.get_parameter('side').value
        if not isinstance(side, str) or side not in ('left', 'right'):
            raise ValueError('side must be left or right')
        self.side = side
        # Generated ParameterValue arrays are iterable even when not Python lists.
        self.tcp = list(self.get_parameter('tcp_xyz').get_parameter_value().double_array_value)
        if len(self.tcp) != 3 or not all(math.isfinite(v) for v in self.tcp):
            raise ValueError('tcp_xyz must contain three finite numbers')
        initial = self.get_parameter('initial_position').value
        if isinstance(initial, bool) or not isinstance(initial, (float, int)):
            raise ValueError('initial_position must be a number')
        self.position = float(initial)
        opening_from_position(self.position)
        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        base = '/' + self.side
        self.marker_pub = self.create_publisher(MarkerArray, base + '/gripper_markers', qos)
        self.position_pub = self.create_publisher(Float64, base + '/gripper_position', qos)
        self.command_sub = self.create_subscription(
            Float64, base + '/gripper_position_command', self.on_command, 10
        )
        self.timer = self.create_timer(0.05, self.publish_markers)
        self.get_logger().info(
            f'{self.side} coordinate gripper: 0.0=open, 5.65=closed. '
            'Markers only; no hardware or grasp-contact simulation.'
        )

    def on_command(self, message: Float64):
        value = float(message.data)
        try:
            opening_from_position(value)
        except ValueError as exc:
            self.get_logger().error(str(exc))
            return
        self.position = value
        self.publish_markers()

    def make_sphere(self, marker_id, xyz, color):
        marker = Marker()
        marker.header.frame_id = self.side + '_gp80_base_link'
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = self.side + '_gripper_coordinates'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = float(xyz[0])
        marker.pose.position.y = float(xyz[1])
        marker.pose.position.z = float(xyz[2])
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.008
        marker.scale.y = 0.008
        marker.scale.z = 0.008
        marker.color.r = float(color[0])
        marker.color.g = float(color[1])
        marker.color.b = float(color[2])
        marker.color.a = 1.0
        return marker

    def publish_markers(self):
        left, right = fingertip_coordinates(self.position, self.tcp)
        markers = MarkerArray()
        markers.markers = [
            self.make_sphere(0, left, (0.0, 1.0, 0.0)),
            self.make_sphere(1, right, (0.0, 0.6, 1.0)),
            self.make_sphere(2, self.tcp, (1.0, 0.0, 0.0)),
        ]
        self.marker_pub.publish(markers)
        state = Float64()
        state.data = self.position
        self.position_pub.publish(state)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = GripperMarkers()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()