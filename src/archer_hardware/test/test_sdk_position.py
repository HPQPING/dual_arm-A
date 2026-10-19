"""使用真实 SDK 离线验证双臂历史隔离和目标拒改，不建立网络连接."""

from copy import deepcopy
from importlib import import_module
import unittest
from typing import Any

import numpy as np

from archer_hardware.sdk_position import isolated_arm_class


def configuration(velocity: float = .1) -> dict[str, Any]:
    """仅为内存编码测试提供虚构边界，不作为真机参数."""
    return {
        'name': 'offline_only', 'dof_num': 'six_axis', 'motor_model': [128] * 6,
        'joints': [{'joint_name': f'joint_{i}',
                    'joint_limit': [-2., 2., -velocity, velocity, 0., 0.]}
                   for i in range(6)],
    }


class SdkPositionTests(unittest.TestCase):
    """直接实例化厂商对象并捕获编码结果，不创建 HexDeviceApi."""

    def setUp(self) -> None:
        self.module = import_module('hex_device.arm')
        self.command = self.module.CommandType
        self.global_manager = import_module('hex_device.arm_config').arm_config_manager
        self.global_before = deepcopy(self.global_manager.__dict__)
        arm_type = isolated_arm_class()
        self.left = arm_type(robot_type=25, motor_count=6, proto_version=(1, 6))
        self.right = arm_type(robot_type=25, motor_count=6, proto_version=(1, 6))
        for arm in (self.left, self.right):
            self.assertTrue(arm.reload_arm_config_from_dict(configuration()))
            arm.set_initial_positions([0.] * 6)
            # 模拟上行已提供的编码器参数，只使用内存，不伪装实机反馈。
            arm._pulse_per_rotation = np.array([100000.] * 6)

    def tearDown(self) -> None:
        self.assertEqual(self.global_manager.__dict__, self.global_before)

    def packet(self, arm: Any, target: list[float]) -> Any:
        """使用公开 POSITION 入口和厂商原始报文构造逻辑."""
        arm.motor_command(self.command.POSITION, target)
        with arm._command_lock:
            command = deepcopy(arm._target_command)
        return arm._construct_joint_command_msg(command)

    def test_same_model_history_and_brake_are_isolated(self) -> None:
        """左右同型号不同起点交替编码，制动一侧不清除另一侧历史."""
        self.right.set_initial_positions([1.] * 6)
        self.packet(self.left, [.0001] * 6)
        self.packet(self.right, [1.0001] * 6)
        self.assertEqual(self.left.get_last_positions(), [.0001] * 6)
        self.assertEqual(self.right.get_last_positions(), [1.0001] * 6)
        self.left.motor_command(self.command.BRAKE, [True] * 6)
        self.assertIsNone(self.left.get_last_positions())
        self.assertEqual(self.right.get_last_positions(), [1.0001] * 6)
        with self.assertRaisesRegex(RuntimeError, '未初始化'):
            self.packet(self.left, [.0001] * 6)

    def test_position_and_speed_clipping_rejected_without_history_change(self) -> None:
        """禁止裁剪越界或过快目标，也禁止失败编码污染后续限速历史."""
        for value in (3., .01):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, '裁剪'):
                self.packet(self.left, [value] * 6)
            self.assertEqual(self.left.get_last_positions(), [0.] * 6)
        message = self.packet(self.left, [.0001] * 6)
        control = message.arm_command.arm_exclusive_command.arm_api_control_command
        targets = control.motor_targets.targets
        expected = self.left.convert_rad_to_positions(
            np.array([.0001] * 6), self.left.get_motor_pulse_per_rotations(),
        )
        self.assertEqual([p.position for p in targets], [int(x) for x in expected])
        self.assertTrue(all({f.name for f, _ in p.ListFields()} == {'position'}
                            for p in targets))

    def test_config_and_returned_history_do_not_leak(self) -> None:
        """修改返回值或重配左臂，不改变右臂或原配置字典."""
        values = self.left.get_last_positions()
        assert values is not None
        values[0] = 99.
        self.assertEqual(self.left.get_last_positions(), [0.] * 6)
        config = configuration(.02)
        self.assertTrue(self.left.reload_arm_config_from_dict(config))
        config['joints'][0]['joint_limit'][0] = -99.
        self.assertEqual(self.left.get_joint_limits()[0][0], -2.)
        self.assertEqual(self.right.get_joint_limits()[0][3], .1)
        with self.assertRaisesRegex(RuntimeError, '未初始化'):
            self.packet(self.left, [0.] * 6)

    def test_invalid_config_and_modes_are_rejected(self) -> None:
        """坏配置保持原配置，速度、零力矩及 MIT 均不能进入缓存."""
        for bad in (float('nan'), float('inf')):
            config = configuration()
            config['joints'][0]['joint_limit'][0] = bad
            self.assertFalse(self.left.reload_arm_config_from_dict(config))
        self.assertEqual(self.left.get_last_positions(), [0.] * 6)
        for mode in (self.command.SPEED, self.command.TORQUE, self.command.MIT):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, '只允许'):
                self.left.motor_command(mode, [0.] * 6)
        with self.assertRaises(ValueError):
            self.left.motor_command(self.command.BRAKE, [False] * 6)
        for values in ([True] * 6, [0.] * 5, [float('nan')] * 6):
            with self.assertRaises(ValueError):
                self.left.motor_command(self.command.POSITION, values)

    def test_sdk_factory_can_construct_isolated_arm(self) -> None:
        """核对厂商按类名解析构造参数的工厂，无网络也能发现六轴."""
        factory = import_module('hex_device.device_factory').DeviceFactory()
        factory.register_device_class(type(self.left))
        up = import_module('hex_device.generated.public_api_up_pb2').APIUp()
        up.protocol_major_version = 1
        up.protocol_minor_version = 6
        for _ in range(6):
            up.arm_status.motor_status.add()
        arm = factory.create_device_for_robot_type(25, 250, api_up=up)
        self.assertIsInstance(arm, type(self.left))
        self.assertEqual(arm.motor_count, 6)


if __name__ == '__main__':
    unittest.main()
