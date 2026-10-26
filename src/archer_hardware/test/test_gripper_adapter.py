"""以SDK替身验证夹爪发送准入、过期取消、端口隔离和制动，不连接设备。"""

import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from typing import Any, cast
import unittest
from unittest.mock import patch

from archer_hardware.gripper_adapter import HexGripperAdapter, verified_hands
from archer_hardware.position_adapter import HexPositionAdapter


def run_immediate(coroutine):
    """替身传输立即返回，无需创建会使用本机socketpair的事件循环。"""
    try:
        coroutine.send(None)
    except StopIteration:
        return
    finally:
        coroutine.close()
    raise AssertionError('替身传输不应包含实际异步等待')


class Wire:
    def __init__(self, fields: dict[str, Any]):
        self.fields = fields
        self.__dict__.update(fields)

    def ListFields(self):
        return [(SimpleNamespace(name=name), value) for name, value in self.fields.items()]


class Message(Wire):
    def __init__(self, kind='position', value=5.6, device_id=1):
        self.kind, self.value, self.device_id = kind, value, device_id
        super().__init__({'secondary_device_command': Wire({
            'device_id': device_id, 'hand_command': Wire({
                'motor_targets': Wire({'targets': [Wire({kind: value})]}),
            }),
        })})

    def SerializeToString(self):
        return repr((self.kind, self.value, self.device_id)).encode()


class Hand:
    device_id = 1
    motor_count = 1

    def __init__(self):
        self.commands = []
        self.callback = None
        self.feedback = {'pos': [5.6], 'ts': {'s': 1, 'ns': 0}}

    def get_simple_motor_status(self, pop=False):
        return self.feedback

    def get_hand_type(self):
        return 4

    def get_joint_limits(self):
        return [0., 5.65]

    def get_motor_state(self, index):
        return 'normal'

    def motor_command(self, kind, values):
        self.commands.append((kind, tuple(values)))

    def _construct_joint_command_msg(self):
        kind, values = self.commands[-1]
        return Message('brake' if kind == 'BRAKE' else 'position', values[0])

    def _set_send_message_callback(self, callback):
        self.callback = callback

    async def _periodic(self):
        pass


class GripperAdapterTests(unittest.TestCase):
    def setUp(self):
        self.sent = []
        self.failures = []
        self.future = SimpleNamespace(
            done=lambda: False, cancel=lambda: self.sent.append('cancel'))
        arm = SimpleNamespace(get_my_session_id=lambda: 1, get_session_holder=lambda: 1)
        self.owner = cast(HexPositionAdapter, SimpleNamespace(
            state='HOLDING', _command_types=SimpleNamespace(POSITION='POSITION', BRAKE='BRAKE'),
            _feedback=SimpleNamespace(source_ns=time.perf_counter_ns()),
            _arm=lambda: arm, _health=lambda **kw: (True, True, True),
            _native_failure=self.failures.append,
        ))
        self.gripper = HexGripperAdapter(self.owner)
        self.hand = Hand()
        self.gripper.api = SimpleNamespace(
            _HexDeviceApi__loop=SimpleNamespace(is_running=lambda: True))
        self.gripper.frame = (self.hand, time.perf_counter_ns(), 5.60)

        async def send(api, message):
            self.sent.append((api, message.kind, message.value))

        def module(name):
            if name.endswith('public_api_types_pb2'):
                return SimpleNamespace(SecondaryDeviceType=SimpleNamespace(SdtHandGp80G1=4))
            if name == 'hex_device.hex_device_api':
                return SimpleNamespace(HexDeviceApi=SimpleNamespace(_send_down_message=send))
            raise AssertionError('Unexpected SDK import: ' + name)

        def schedule(coroutine, loop):
            coroutine.close()
            return self.future

        self.patch_module = patch(
            'archer_hardware.gripper_adapter.import_module', side_effect=module)
        self.patch_periodic = patch(
            'archer_hardware.gripper_adapter.asyncio.run_coroutine_threadsafe',
            side_effect=schedule)
        self.patch_module.start()
        self.patch_periodic.start()
        self.addCleanup(self.patch_module.stop)
        self.addCleanup(self.patch_periodic.stop)

    def activate(self):
        self.gripper.request_enable()
        self.gripper.tick()

    def test_enable_is_explicit_and_starts_from_feedback(self):
        self.assertEqual(self.hand.commands, [])
        self.gripper.tick()
        self.assertEqual(self.hand.commands, [])
        self.activate()
        self.assertEqual(self.hand.commands[0], ('POSITION', (5.60,)))
        self.assertEqual(self.sent, [])

    def test_only_expected_position_reaches_same_connection(self):
        self.activate()
        message = self.hand._construct_joint_command_msg()
        run_immediate(self.gripper.dispatch(message))
        self.assertEqual(self.sent[0][0], self.gripper.api)
        self.assertEqual(self.sent[0][1], 'position')
        self.assertEqual(self.failures, [])

    def test_late_target_cannot_be_sent(self):
        self.activate()
        message = self.hand._construct_joint_command_msg()
        self.gripper.updated_at = time.monotonic() - .041
        run_immediate(self.gripper.dispatch(message))
        self.assertEqual(self.sent, [])
        self.assertIn('过期', self.failures[0])

    def test_replaced_device_and_control_loss_are_rejected(self):
        self.hand.device_id = 2
        with self.assertRaisesRegex(RuntimeError, 'GP80G1'):
            self.activate()
        self.hand.device_id = 1
        self.activate()
        message = self.hand._construct_joint_command_msg()
        self.owner._arm().get_session_holder = lambda: 2
        run_immediate(self.gripper.dispatch(message))
        self.assertEqual(self.sent, [])
        self.assertIn('控制权', self.failures[0])

    def test_cancelled_gate_prevents_queued_position(self):
        self.activate()
        message = self.hand._construct_joint_command_msg()
        self.gripper.gate = cast(Any, SimpleNamespace(cancelled=True))
        run_immediate(self.gripper.dispatch(message))
        self.assertEqual(self.sent, [])
        self.assertEqual(self.failures, [])
        self.assertFalse(self.gripper.active)

    def test_stopped_gripper_drops_previously_encoded_position(self):
        self.activate()
        old = self.hand._construct_joint_command_msg()
        self.gripper.stop()
        self.owner.state = 'STOPPED'
        run_immediate(self.gripper.dispatch(old))
        self.assertEqual(self.sent, [])
        self.assertEqual(self.failures, [])
        self.assertFalse(self.gripper.delivered.is_set())
        brake = self.hand._construct_joint_command_msg()
        run_immediate(self.gripper.dispatch(brake))
        self.assertEqual(self.sent[-1][1:], ('brake', True))

    def test_late_position_ack_cannot_acknowledge_new_brake(self):
        self.activate()
        message = self.hand._construct_joint_command_msg()

        async def send_and_cancel(api, packet):
            self.gripper.stop()
            self.owner.state = 'STOPPED'

        module = SimpleNamespace(HexDeviceApi=SimpleNamespace(_send_down_message=send_and_cancel))
        from archer_hardware import gripper_adapter
        original_import = gripper_adapter.import_module

        def imported(name):
            return module if name == 'hex_device.hex_device_api' else original_import(name)

        with patch('archer_hardware.gripper_adapter.import_module', side_effect=imported):
            run_immediate(self.gripper.dispatch(message))
        self.assertEqual(self.failures, [])
        self.assertFalse(self.gripper.delivered.is_set())

    def test_stop_encodes_brake_and_does_not_open_or_close(self):
        self.activate()
        self.gripper.stop()
        self.owner.state = 'STOPPED'
        message = self.hand._construct_joint_command_msg()
        run_immediate(self.gripper.dispatch(message))
        self.assertEqual(self.sent[0][1:], ('brake', True))
        self.assertEqual(self.hand.commands[-1], ('BRAKE', (True,)))
        self.assertFalse(self.gripper.active)
        self.gripper.stop()
        self.assertTrue(self.gripper.delivered.is_set())

    def test_unknown_modes_or_device_ids_never_pass_wire_filter(self):
        for message in (Message('torque', 0.), Message('velocity', .1),
                        Message(device_id=2), Message('brake', False)):
            with self.subTest(kind=message.kind), self.assertRaises(RuntimeError):
                HexGripperAdapter._validate_message(message, 'position')

    def test_stale_gripper_feedback_blocks_update(self):
        self.gripper.frame = (self.hand, time.perf_counter_ns() - 101_000_000, 5.6)
        with self.assertRaisesRegex(RuntimeError, '100ms'):
            self.activate()
        self.assertEqual(self.hand.commands, [])

    def test_close_without_enable_does_not_send_brake(self):
        self.gripper.close()
        self.assertEqual(self.hand.commands, [])
        self.assertTrue(self.gripper.closed)

    def test_duplicate_sdk_feedback_cannot_refresh_its_age(self):
        api = SimpleNamespace(
            _process_api_up=lambda message: None,
            find_optional_device_by_id=lambda identifier: self.hand,
        )
        with patch('archer_hardware.gripper_adapter.verified_hands', return_value=Hand):
            self.gripper.attach(api)
        message = SimpleNamespace(secondary_device_status=[SimpleNamespace(device_id=1)])
        api._process_api_up(message)
        first = self.gripper.frame
        api._process_api_up(message)
        self.assertIs(self.gripper.frame, first)

    def test_constructor_and_sdk_change_do_not_connect(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'hands.py'
            path.write_bytes(b'known source')
            module = SimpleNamespace(__file__=str(path), Hands=Hand)
            with patch('archer_hardware.gripper_adapter.import_module', return_value=module), \
                    patch('archer_hardware.gripper_adapter.HANDS_SHA256',
                          hashlib.sha256(path.read_bytes()).hexdigest()):
                self.assertIs(verified_hands(), Hand)
                path.write_bytes(b'changed source')
                with self.assertRaisesRegex(RuntimeError, '源码已变化'):
                    verified_hands()


if __name__ == '__main__':
    unittest.main()
