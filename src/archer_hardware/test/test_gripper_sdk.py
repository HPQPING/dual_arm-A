"""运行官方Hands周期并截获全部传输，核对真实编码与BRAKE，不创建网络API。"""

import asyncio
from importlib import import_module
import math
from types import SimpleNamespace
from typing import cast
import time
import unittest
from unittest.mock import patch

from archer_hardware.gripper_adapter import HexGripperAdapter, verified_hands
from archer_hardware.hex_adapter import JointFeedback
from archer_hardware.position_adapter import HexPositionAdapter


class GripperSdkTests(unittest.TestCase):
    def test_official_cycle_sends_only_expected_position_and_brake(self):
        """保留官方插补、扭矩限幅和周期，仅替换最终传输。"""
        async def run():
            hand_type = verified_hands()
            types = import_module('hex_device.generated.public_api_types_pb2')
            motor = import_module('hex_device.motor_base')
            sdk = import_module('hex_device.hex_device_api').HexDeviceApi
            hand = hand_type(
                device_id=1, device_type=types.SecondaryDeviceType.SdtHandGp80G1,
                motor_count=1, proto_version=(1, 6), control_hz=250,
                send_message_callback=None,
            )
            hand._push_motor_data([types.MotorStatus(
                position=int(5.60 / (2.*math.pi) * 100000), speed=0, torque=0,
                pulse_per_rotation=100000,
            )], motor.Timestamp.from_ns(time.perf_counter_ns()))
            self.assertIsNotNone(hand.get_motor_positions(pop=False))
            faults, sent = [], []
            arm = SimpleNamespace(get_my_session_id=lambda: 7, get_session_holder=lambda: 7)
            owner = cast(HexPositionAdapter, SimpleNamespace(
                state='HOLDING', _command_types=motor.CommandType,
                _feedback=SimpleNamespace(source_ns=time.perf_counter_ns()),
                _health=lambda **kw: (True, True, True), _arm=lambda: arm,
                _native_failure=faults.append,
            ))
            gripper = HexGripperAdapter(owner)
            gripper.api = SimpleNamespace(_HexDeviceApi__loop=asyncio.get_running_loop())

            def refresh():
                owner._feedback = JointFeedback(
                    (0.,) * 6, (0.,) * 6, (0.,) * 6, time.perf_counter_ns())
                gripper.frame = (hand, time.perf_counter_ns(),
                                 float(hand.get_motor_positions(pop=False)[0]))

            async def capture(api, message):
                self.assertIs(api, gripper.api)
                packet = type(message)()
                packet.CopyFrom(message)
                sent.append(packet)

            refresh()
            with patch.object(sdk, '_send_down_message', capture):
                try:
                    gripper.request_enable()
                    gripper.tick()
                    gripper.started_at -= 2.
                    for _ in range(15):
                        refresh()
                        gripper.tick()
                        await asyncio.sleep(.004)
                    self.assertTrue(gripper.status()['ready'], gripper.status())
                    self.assertTrue(sent)
                    for packet in sent:
                        HexGripperAdapter._validate_message(packet, 'position')
                    gripper.stop(wait=False)
                    owner.state = 'STOPPED'
                    await asyncio.sleep(.012)
                    self.assertTrue(gripper.delivered.is_set())
                    HexGripperAdapter._validate_message(sent[-1], 'brake')
                    self.assertEqual(faults, [])
                finally:
                    gripper.close()
                    await asyncio.sleep(0)
        asyncio.run(run())
