"""离线验证反馈验收，ROS 消息使用合成数据，不连接设备."""

import unittest

import rclpy
from sensor_msgs.msg import JointState

from archer_hardware.commissioning import FeedbackAuditNode, FeedbackWindow
from archer_hardware.position_plan import PositionLimits


def limits() -> PositionLimits:
    """虚构测试限位，不用于实机."""
    return PositionLimits((-1.0,) * 6, (1.0,) * 6, (0.1,) * 6, (0.2,) * 6, 0.002)


def message(seconds: float, side: str = 'left') -> JointState:
    """生成包含六个关节的合成消息."""
    result = JointState()
    stamp = int(round(seconds * 1e9))
    result.header.stamp.sec = stamp // 1_000_000_000
    result.header.stamp.nanosec = stamp % 1_000_000_000
    result.name = [f'{side}_joint_{i}' for i in range(1, 7)]
    result.position = [0.0] * 6
    result.velocity = [0.0] * 6
    result.effort = [0.0] * 6
    return result


class FeedbackAuditTests(unittest.TestCase):
    """检查失联、运动、乱序、坏帧、轻微越界和短时突发."""

    def setUp(self) -> None:
        """创建无网络的观察窗口."""
        self.window = FeedbackWindow('left', limits())

    def add(self, stamp: float, packet: JointState | None = None) -> None:
        """以一致的合成墙钟和单调时间送入数据."""
        self.window.add(message(stamp) if packet is None else packet,
                        stamp, int(round(stamp * 1e9)))

    def test_reorder_and_raw_tolerance(self) -> None:
        """乱序关节按名称重排，容差内实测越界保留原值."""
        for stamp in (1.0, 1.1, 1.2):
            packet = message(stamp)
            packet.name = list(reversed(list(packet.name)))
            packet.position = [0.0] * 5 + [-1.001]
            self.add(stamp, packet)
        result = self.window.report(1.21, 3, 0.25, 0.02)
        self.assertTrue(result['passed'])
        self.assertEqual(self.window.initial, (-1.001,) + (0.0,) * 5)
        self.assertAlmostEqual(result['max_nominal_limit_overrun_rad'][0], 0.001)
        self.assertFalse(result['latest_within_nominal_limits'])

    def test_missing_and_stale(self) -> None:
        """无消息和尾帧过期都不能通过."""
        self.assertFalse(self.window.report(1.0, 2, 0.25, 0.02)['passed'])
        self.add(1.0)
        self.add(1.1)
        self.assertFalse(self.window.report(2.0, 2, 0.25, 0.02)['passed'])

    def test_gap_and_motion_latched(self) -> None:
        """历史中断和观察到的运动不会被末帧静止覆盖."""
        packet = message(1.0)
        packet.velocity = [0.03] * 6
        self.add(1.0, packet)
        self.add(1.5)
        self.add(1.6)
        result = self.window.report(1.61, 2, 0.25, 0.02)
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['max_arrival_gap_s'], 0.5)
        self.assertAlmostEqual(result['max_abs_velocity_rad_s'], 0.03)

    def test_invalid_frame_latched(self) -> None:
        """坏帧被记录，不接受重复、混臂、非有限值或越界数据."""
        for kind in ('duplicate', 'side', 'nan', 'limit', 'missing', 'old'):
            with self.subTest(kind=kind):
                self.window = FeedbackWindow('left', limits())
                self.add(1.0)
                packet = message(1.1)
                if kind == 'duplicate':
                    packet.name = ['left_joint_1'] * 6
                elif kind == 'side':
                    packet = message(1.1, 'right')
                elif kind == 'nan':
                    packet.position = [float('nan')] * 6
                elif kind == 'limit':
                    packet.position = [-1.003] * 6
                elif kind == 'missing':
                    packet.velocity = []
                else:
                    packet = message(1.0)
                self.add(1.1, packet)
                self.add(1.2)
                result = self.window.report(1.21, 2, 0.25, 0.02)
                self.assertFalse(result['passed'])
                self.assertEqual(result['invalid_frames'], 1)

    def test_burst_is_not_long_observation(self) -> None:
        """短时间大量消息不能冒充完整时段采集."""
        self.add(1.0)
        self.add(1.01)
        self.assertFalse(self.window.report(1.02, 2, 0.25, 0.02, 8.0)['passed'])

    def test_stale_header_rejected(self) -> None:
        """收到旧消息，即使到达时间新鲜仍拒绝."""
        self.window.add(message(1.0), 3.0, 3_000_000_000)
        self.assertEqual(self.window.invalid, 1)

    def test_rejected_limit_frames_preserve_raw_evidence(self) -> None:
        """超限反馈保留实际角度与通信统计，不能将全拒绝误报为没收到数据."""
        for stamp in (1.0, 1.1, 1.2):
            packet = message(stamp)
            packet.position = [-1.03] * 6
            self.add(stamp, packet)
        result = self.window.report(1.21, 2, 0.25, 0.02)
        self.assertFalse(result['passed'])
        self.assertEqual(result['samples'], 0)
        self.assertEqual(result['structurally_valid_observations'], 3)
        self.assertEqual(result['invalid_frames'], 3)
        self.assertEqual(result['latest_rad'], (-1.03,) * 6)
        self.assertAlmostEqual(result['max_nominal_limit_overrun_rad'][0], 0.03)
        self.assertAlmostEqual(result['observed_rate_hz'], 10.0)

    def test_malformed_frames_do_not_replace_evidence(self) -> None:
        """结构错误或时间戳过期的消息不覆盖最近一帧合法结构的数据."""
        self.add(1.0)
        bad = message(1.1)
        bad.position = [float('nan')] * 6
        self.add(1.1, bad)
        self.assertEqual(self.window.latest, (0.0,) * 6)
        self.assertEqual(self.window.observed, 1)
        self.assertEqual(self.window.invalid, 1)

    def test_node_has_no_control_interfaces(self) -> None:
        """真实 ROS 图上只订阅数据，不创建运动服务客户端或话题发布."""
        rclpy.init(args=[])
        node = None
        try:
            node = FeedbackAuditNode({'left': limits(), 'right': limits()})
            topics = {topic for topic, _ in node.get_subscriber_names_and_types_by_node(
                node.get_name(), node.get_namespace(),
            )}
            self.assertIn('/hardware/left/joint_states', topics)
            self.assertIn('/hardware/right/joint_states', topics)
            self.assertEqual(list(node.clients), [])
        finally:
            if node is not None:
                node.destroy_node()
            rclpy.shutdown()

    def test_startup_candidates_preserve_raw_and_only_recover_outside_joint(self) -> None:
        """原始越界初值不变，准备目标只收回越界关节，归位候选取名义边界."""
        first = message(1.0)
        first.position = [-1.001, .3] + [0.0] * 4
        self.add(1.0, first)
        last = message(1.1)
        last.position = [-1.0011, .30001] + [0.0] * 4
        self.add(1.1, last)
        result = self.window.report(1.11, 2, .25, .02)
        startup = result['startup_review']
        self.assertTrue(result['passed'])
        self.assertFalse(startup['execution_ready'])
        self.assertTrue(startup['requires_nominal_recovery'])
        self.assertEqual(result['initial_observed_rad'][0], -1.001)
        self.assertEqual(result['latest_rad'][0], -1.0011)
        self.assertEqual(startup['recovery_candidate_rad'], (-.995, .30001) + (0.,) * 4)
        self.assertEqual(startup['saved_start_return_candidate_rad'], (-1., .3) + (0.,) * 4)

    def test_bad_or_missing_feedback_has_no_startup_candidate(self) -> None:
        """坏帧或缺反馈不能产生可被误用的准备位置."""
        empty = self.window.report(1.0, 2, .25, .02)['startup_review']
        self.assertIsNone(empty['recovery_candidate_rad'])
        self.assertIsNone(empty['requires_nominal_recovery'])
        self.add(1.0)
        bad = message(1.1)
        bad.position = [1.1] * 6
        self.add(1.1, bad)
        startup = self.window.report(1.11, 2, .25, .02)['startup_review']
        self.assertIsNone(startup['recovery_candidate_rad'])
        self.assertIsNone(startup['saved_start_return_candidate_rad'])
        self.assertFalse(startup['execution_ready'])

    def test_position_variation_preserves_small_changes(self) -> None:
        """往返变化及微小抖动都保留，不以首尾相同冒充全程不变."""
        for index, q in enumerate((0.0, 1e-7, -2e-7, 0.0)):
            stamp = 1.0 + index * .1
            packet = message(stamp)
            packet.position = [q, .5] + [0.0] * 4
            self.add(stamp, packet)
        result = self.window.report(1.31, 4, .25, .02)
        self.assertTrue(result['passed'])
        self.assertFalse(result['positions_exactly_constant'])
        self.assertEqual(result['position_change_counts'], (3, 0, 0, 0, 0, 0))
        self.assertAlmostEqual(result['position_range_rad'][0], 3e-7)
        self.assertAlmostEqual(result['max_adjacent_position_change_rad'][0], 3e-7)
        self.assertEqual(result['minimum_observed_rad'][0], -2e-7)
        self.assertEqual(result['initial_observed_rad'], result['latest_rad'])

    def test_no_or_single_frame_is_not_constant_evidence(self) -> None:
        """没有变化区间时输出未知，至少两帧才能报告逐位不变."""
        for count in (0, 1, 2):
            if count:
                self.add(1.0 + count * .1)
            result = self.window.report(1.21, 2, .25, .02)
            if count < 2:
                self.assertIsNone(result['positions_exactly_constant'])
                self.assertIsNone(result['max_adjacent_position_change_rad'])
            else:
                self.assertTrue(result['positions_exactly_constant'])
                self.assertEqual(result['position_range_rad'], (0.0,) * 6)

    def test_bad_frames_do_not_pollute_position_statistics(self) -> None:
        """坏帧不混入范围统计；超限但结构合法的真实读数仍留证且判失败."""
        self.add(1.0)
        packet = message(1.1)
        packet.position = [float('nan')] * 6
        self.add(1.1, packet)
        packet = message(1.2)
        packet.position = [1.03] + [0.0] * 5
        self.add(1.2, packet)
        result = self.window.report(1.21, 2, .25, .02)
        self.assertFalse(result['passed'])
        self.assertEqual(result['position_range_rad'][0], 1.03)
        self.assertEqual(result['position_change_counts'][0], 1)
