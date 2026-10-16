"""双臂关节驱动的同物件上下夹持演示，不发布 ROS 或实机命令。

任务开始时读取实际 qpos，保存起点，规划接近、夹持、放置、退开和返回起点。
左臂夹上部、右臂夹下部，物体仅通过指腹接触和摩擦被搬动。
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from importlib import import_module
import json
import math
from pathlib import Path
import time
from typing import cast

import numpy as np

from .check_circle import validate_tool_chain
from .cooperative_planner import sample_object
from .dual_arm_trajectory import DualArmKinematics, DualArmPlan, rigid_transform
from .grasp_task_planner import GraspMotion, prepare_grasp_motion
from .kinematics import ArmKinematics, rotation_error
from .trajectory import FloatArray, JointPlan, sample_joint_plan, validate_joints
from archer_mujoco.grasp_model_loader import load_grasp_model, physics
from archer_mujoco.grasp_scene import GraspConfig, ViewerAPI, _smooth


def sample_position_velocity(plan: JointPlan, elapsed: float) -> tuple[FloatArray, FloatArray]:
    """读取位置并计算同一五次轨迹的解析速度，保持首尾速度为零。"""
    position, _ = sample_joint_plan(plan, elapsed)
    if elapsed <= 0. or elapsed >= plan.duration:
        return position, np.zeros(6, dtype=np.float64)
    index = int(np.searchsorted(plan.times, elapsed, side='right')) - 1
    duration = float(plan.times[index + 1] - plan.times[index])
    u = float((elapsed - plan.times[index]) / duration)
    derivative = 30. * u * u * (1. - u) ** 2 / duration
    velocity = np.asarray((plan.positions[index + 1] - plan.positions[index]) * derivative, dtype=np.float64)
    return position, velocity


class DualGraspDemo:
    def __init__(self, left_urdf: str | None = None, right_urdf: str | None = None,
                 config: GraspConfig = GraspConfig(),
                 left_initial: FloatArray | None = None, right_initial: FloatArray | None = None) -> None:
        config.validate()
        self.config = config
        self.left_urdf, self.right_urdf = left_urdf, right_urdf
        tcp = [0., 0., .147021696]
        zero = [0., 0., 0.]
        self.models = {
            side: (ArmKinematics.from_urdf(path, side) if path else
                   ArmKinematics.from_description(side, tcp, zero, zero))
            for side, path in (('left', left_urdf), ('right', right_urdf))
        }
        for model in self.models.values():
            validate_tool_chain(model)
        self.kinematics = DualArmKinematics(
            self.models['left'], self.models['right'],
            rigid_transform([-.21, 0., 0.], [0., 0., np.pi / 2.]),
            rigid_transform([.21, 0., 0.], [0., 0., np.pi / 2.]))
        default = np.array([0., -.8, 1.6, -.8, 0., 0.], dtype=np.float64)
        seeds = {
            'left': validate_joints(default if left_initial is None else left_initial, self.models['left'].limits),
            'right': validate_joints(default if right_initial is None else right_initial, self.models['right'].limits),
        }
        self.loaded = load_grasp_model(seeds['left'], seeds['right'], left_urdf, right_urdf, config)
        # 加载模型与规划器必须具有同一 TCP 变换，防止只显示一致而实际使用不同工具长度。
        world_poses = self.kinematics.forward_world(seeds['left'], seeds['right'])
        self.model_tcp_error = 0.
        for side, expected in zip(('left', 'right'), world_poses):
            arm = self.loaded.arms[side]
            actual = self.loaded.data.site_xpos[arm.tcp_site_id]
            distance = float(np.linalg.norm(actual - expected[:3, 3]))
            angle = float(np.linalg.norm(rotation_error(expected[:3, :3],
                self.loaded.data.site_xmat[arm.tcp_site_id].reshape(3, 3))))
            self.model_tcp_error = max(self.model_tcp_error, distance)
            # URDF 转 MJCF 会舍入参数；允许 1 微米平移和 1e-5 弧度姿态数值差。
            if distance > 1e-6 or angle > 1e-5:
                raise ValueError(f'{side} 场景 TCP 与运动学 TCP 不一致：'
                                 f'{actual.tolist()} / {expected[:3, 3].tolist()}')
        self.initial_qpos = self.loaded.data.qpos.copy()
        self.initial_ctrl = self.loaded.data.ctrl.copy()
        self.reset()
        print(f'场景 READY；尚未规划或执行接近。物体中心 Z={config.center_z:.3f} m，'
              f'台面 Z={config.table_z:.3f} m。')

    def reset(self) -> None:
        physics.mj_resetData(self.loaded.model, self.loaded.data)
        self.loaded.data.qpos[:] = self.initial_qpos
        self.loaded.data.ctrl[:] = self.initial_ctrl
        physics.mj_forward(self.loaded.model, self.loaded.data)
        self.motion: GraspMotion | None = None
        self.state = 'READY'
        self.stage_started = 0.
        self.motion_started = 0.
        self.stable_since: float | None = None
        self.failure = ''
        self.finished = False
        self.verified = False
        self.verified_time = 0.
        self.maximum_object_error = 0.
        self.maximum_relative_error = 0.
        self.maximum_relative_angle = 0.
        self.maximum_object_tilt = 0.
        self.contact_loss_time = 0.
        self.maximum_contact_loss_time = 0.
        self.minimum_lift_clearance = math.inf
        self.history: list[str] = ['READY']
        self.home_error: dict[str, float] = {}

    def read_joints(self) -> dict[str, FloatArray]:
        return {side: validate_joints(self.loaded.data.qpos[arm.qpos_indices], arm.joint_limits)
                for side, arm in self.loaded.arms.items()}

    def enter(self, state: str) -> None:
        self.state = state
        self.stage_started = float(self.loaded.data.time)
        self.stable_since = None
        if not self.history or self.history[-1] != state:
            self.history.append(state)

    def fail(self, reason: str) -> None:
        self.failure, self.finished = reason, True
        self.enter('FAILED')

    def start(self) -> bool:
        """只从静止且张开的实际场景开始；本轮起点从 qpos 读取并保存。"""
        if self.state != 'READY':
            return False
        try:
            physics.mj_forward(self.loaded.model, self.loaded.data)
            current = self.read_joints()
            for side, arm in self.loaded.arms.items():
                if float(np.max(np.abs(self.loaded.data.qvel[arm.dof_indices]))) > .01:
                    raise ValueError(side + ' 尚未静止，拒绝从运动中启动')
                if float(np.max(np.abs(self.loaded.data.qpos[self.loaded.finger_positions[side]]))) > .001:
                    raise ValueError(side + ' 夹爪尚未完全张开，请先重置场景')
            self.enter('PLANNING')
            print('读取本轮起点：', {side: q.tolist() for side, q in current.items()})
            motion = prepare_grasp_motion(self.kinematics, current, self.config, self.left_urdf, self.right_urdf)
            for side, actual in self.read_joints().items():
                if float(np.max(np.abs(actual - current[side]))) > .002:
                    raise ValueError('规划期间姿态发生变化，拒绝使用过期起点')
            self.motion = motion
            self.enter('SETTLE')
            return True
        except (ValueError, RuntimeError, OSError) as exc:
            self.fail('规划或起点检查失败：' + str(exc))
            return False

    def converged(self, paired: DualArmPlan, tolerance: float = .003) -> bool:
        for side, plan in (('left', paired.left), ('right', paired.right)):
            arm = self.loaded.arms[side]
            error = float(np.max(np.abs(self.loaded.data.qpos[arm.qpos_indices] - plan.positions[-1])))
            speed = float(np.max(np.abs(self.loaded.data.qvel[arm.dof_indices])))
            if error > tolerance or speed > .01:
                return False
        return True

    def grounded(self) -> bool:
        data = self.loaded.data
        return (abs(self.loaded.object_bottom() - self.config.table_z) < .001
                and float(np.linalg.norm(data.qvel[self.loaded.object_dof:self.loaded.object_dof + 6])) < .01
                and float(data.xmat[self.loaded.object_body, 8]) > math.cos(math.radians(2.)))

    def wait_stable(self, condition: bool, next_state: str, dwell: float = .2, timeout: float = 3.) -> None:
        now = float(self.loaded.data.time)
        if condition:
            if self.stable_since is None:
                self.stable_since = now
            elif now - self.stable_since >= dwell:
                self.enter(next_state)
                return
        else:
            self.stable_since = None
        if now - self.stage_started > timeout:
            self.fail(self.state + ' 等待实际反馈稳定超时')

    def guard_grasp(self, motion: GraspMotion, elapsed: float, fingers: dict[str, float], table: float) -> None:
        loaded = self.loaded
        expected = sample_object(motion.cooperative, elapsed)
        error = float(np.linalg.norm(loaded.data.xpos[loaded.object_body] - expected[:3, 3]))
        self.maximum_object_error = max(self.maximum_object_error, error)
        actual = self.read_joints()
        left, right = self.kinematics.forward_world(actual['left'], actual['right'])
        relative = np.linalg.inv(left) @ right
        desired = np.linalg.inv(motion.cooperative.object_from_left_tcp) @ motion.cooperative.object_from_right_tcp
        distance = float(np.linalg.norm(relative[:3, 3] - desired[:3, 3]))
        angle = float(np.linalg.norm(rotation_error(desired[:3, :3], relative[:3, :3])))
        tilt = math.acos(min(1., max(-1., float(loaded.data.xmat[loaded.object_body, 8]))))
        self.maximum_relative_error = max(self.maximum_relative_error, distance)
        self.maximum_relative_angle = max(self.maximum_relative_angle, angle)
        self.maximum_object_tilt = max(self.maximum_object_tilt, tilt)
        placing = (self.state == 'LOWER' and abs(loaded.object_bottom() - self.config.table_z) < .001 and table > .05)
        lost = not placing and min(fingers.values()) < .01
        self.contact_loss_time = self.contact_loss_time + self.config.timestep if lost else 0.
        self.maximum_contact_loss_time = max(self.maximum_contact_loss_time, self.contact_loss_time)
        if error > .003 or self.contact_loss_time > .05:
            self.fail('物件误差超过 3 mm，或某个指腹连续失去有效接触超过 50 ms')
        elif distance > .005 or angle > .03 or tilt > .03:
            self.fail('TCP 相对位姿或物体倾斜超过演示阈值')
        if self.state in ('MOVE', 'RETURN'):
            self.minimum_lift_clearance = min(self.minimum_lift_clearance,
                                               loaded.object_bottom() - self.config.table_z)

    def step(self) -> None:
        motion = self.motion
        if self.finished or motion is None:
            return
        data = self.loaded.data
        now = float(data.time)
        elapsed = now - self.stage_started
        active = self.state
        close, sample_time = 0., 0.
        paired = motion.approach
        if active in ('APPROACH', 'APPROACH_SETTLE'):
            sample_time = elapsed if active == 'APPROACH' else paired.duration
        elif active in ('CLOSE', 'VERIFY', 'PLACE_SETTLE', 'OPEN', 'OPEN_WAIT'):
            paired = motion.cooperative.paired
            sample_time = 0. if active in ('CLOSE', 'VERIFY') else paired.duration
            close = _smooth(elapsed / 1.5) if active == 'CLOSE' else 1.
            if active == 'OPEN':
                close = 1. - _smooth(elapsed)
            elif active == 'OPEN_WAIT':
                close = 0.
        elif active in ('LIFT', 'MOVE', 'RETURN', 'LOWER'):
            paired = motion.cooperative.paired
            sample_time = now - self.motion_started
            leg = min(3, int(np.searchsorted(paired.left.times[[16, 32, 48, 64]], sample_time, side='right')))
            label = ('LIFT', 'MOVE', 'RETURN', 'LOWER')[leg]
            if label != self.state:
                self.enter(label)
            close = 1.
        elif active in ('RETREAT', 'RETREAT_SETTLE'):
            paired = motion.retreat
            sample_time = elapsed if active == 'RETREAT' else paired.duration
        elif active in ('RETURN_HOME', 'HOME_SETTLE'):
            paired = motion.home
            sample_time = elapsed if active == 'RETURN_HOME' else paired.duration
        left, lv = sample_position_velocity(paired.left, sample_time)
        right, rv = sample_position_velocity(paired.right, sample_time)
        self.loaded.set_arm_targets(left, right, lv, rv)
        self.loaded.set_grippers(5.65 * close)
        self.loaded.step()
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            self.fail('仿真状态出现非有限值')
            return
        fingers, table = self.loaded.contact_forces()
        elapsed = float(data.time) - self.stage_started
        if self.state == 'SETTLE' and elapsed >= .8:
            if not self.grounded():
                self.fail('圆柱尚未在台面稳定落定')
            else:
                self.enter('APPROACH')
        elif self.state == 'APPROACH' and elapsed >= motion.approach.duration:
            self.enter('APPROACH_SETTLE')
        elif self.state == 'APPROACH_SETTLE':
            self.wait_stable(self.converged(motion.approach), 'CLOSE')
        elif self.state == 'CLOSE' and elapsed >= 1.5:
            self.enter('VERIFY')
        elif self.state == 'VERIFY':
            self.verified_time = self.verified_time + self.config.timestep if min(fingers.values()) > .02 else 0.
            if self.verified_time >= .2:
                self.verified = True
                self.motion_started = float(data.time)
                self.enter('LIFT')
            elif elapsed > .8:
                self.fail('上下夹爪持续接触确认失败，禁止抬升')
        elif self.state in ('LIFT', 'MOVE', 'RETURN', 'LOWER'):
            if not self.verified:
                self.fail('夹持未经确认')
                return
            motion_time = float(data.time) - self.motion_started
            self.guard_grasp(motion, motion_time, fingers, table)
            if not self.finished and motion_time >= motion.cooperative.paired.duration:
                self.enter('PLACE_SETTLE')
        elif self.state == 'PLACE_SETTLE':
            self.wait_stable(self.grounded() and self.converged(motion.cooperative.paired), 'OPEN')
        elif self.state == 'OPEN' and elapsed >= 1.:
            self.enter('OPEN_WAIT')
        elif self.state == 'OPEN_WAIT':
            opened = all(float(np.max(np.abs(data.qpos[indices]))) < .001
                         for indices in self.loaded.finger_positions.values())
            # 实际张开、物体落稳且台面已承重后，才允许退开。
            self.wait_stable(opened and self.grounded() and table > .05, 'RETREAT')
        elif self.state == 'RETREAT' and elapsed >= motion.retreat.duration:
            self.enter('RETREAT_SETTLE')
        elif self.state == 'RETREAT_SETTLE':
            self.wait_stable(self.converged(motion.retreat), 'RETURN_HOME')
        elif self.state == 'RETURN_HOME' and elapsed >= motion.home.duration:
            self.enter('HOME_SETTLE')
        elif self.state == 'HOME_SETTLE':
            actual = self.read_joints()
            self.home_error = {side: float(np.max(np.abs(actual[side] - motion.start_joints[side])))
                               for side in ('left', 'right')}
            self.wait_stable(self.converged(motion.home) and self.grounded(), 'COMPLETED', dwell=.3)
            if self.state == 'COMPLETED':
                if not self.verified or self.minimum_lift_clearance < self.config.lift_height * .8:
                    self.fail('抬升验证未通过')
                else:
                    self.finished = True
        if self.state in ('APPROACH', 'APPROACH_SETTLE', 'RETREAT', 'RETREAT_SETTLE', 'RETURN_HOME', 'HOME_SETTLE'):
            nominal = np.array([0., .4, self.config.center_z], dtype=np.float64)
            if float(np.linalg.norm(data.xpos[self.loaded.object_body] - nominal)) > .005:
                self.fail('接近或回程期间物件被扰动超过 5 mm')

    def summary(self) -> dict[str, object]:
        return {'state': self.state, 'failure': self.failure, 'config': asdict(self.config),
                'simulation_time_s': float(self.loaded.data.time), 'stages': self.history,
                'start_joints_from_feedback_rad': ({side: q.tolist() for side, q in self.motion.start_joints.items()}
                                                   if self.motion else {}),
                'final_joints_rad': {side: q.tolist() for side, q in self.read_joints().items()},
                'home_maximum_joint_error_rad': self.home_error,
                'maximum_object_position_error_mm': self.maximum_object_error * 1000.,
                'maximum_relative_tcp_position_error_mm': self.maximum_relative_error * 1000.,
                'maximum_relative_tcp_rotation_error_rad': self.maximum_relative_angle,
                'maximum_object_tilt_rad': self.maximum_object_tilt,
                'maximum_model_tcp_difference_mm': self.model_tcp_error * 1000.,
                'maximum_continuous_contact_loss_ms': self.maximum_contact_loss_time * 1000.,
                'minimum_lift_clearance_mm': (self.minimum_lift_clearance * 1000.
                                              if math.isfinite(self.minimum_lift_clearance) else None),
                'final_object_center_world_m': self.loaded.data.xpos[self.loaded.object_body].tolist(),
                'scope': '双臂关节动力学与简化指腹接触；从反馈起点接近、放置后退开并返回；未接 ROS 或实机，未含连续碰撞、自碰撞和线缆检查'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-urdf')
    parser.add_argument('--right-urdf')
    parser.add_argument('--left-initial', type=float, nargs=6, help='场景初值，弧度；任务起点仍从反馈读取')
    parser.add_argument('--right-initial', type=float, nargs=6, help='场景初值，弧度')
    parser.add_argument('--headless', action='store_true', help='不开窗口，自动执行并输出结果')
    parser.add_argument('--output', type=Path, help='可选的 JSON 结果文件')
    args = parser.parse_args()
    try:
        demo = DualGraspDemo(args.left_urdf, args.right_urdf,
            left_initial=None if args.left_initial is None else np.asarray(args.left_initial, dtype=np.float64),
            right_initial=None if args.right_initial is None else np.asarray(args.right_initial, dtype=np.float64))
        if args.headless:
            if demo.start() and demo.motion is not None:
                motion = demo.motion
                duration = sum(plan.duration for plan in (motion.approach, motion.cooperative.paired, motion.retreat, motion.home))
                for _ in range(math.ceil((duration + 25.) / demo.config.timestep)):
                    demo.step()
                    if demo.finished:
                        break
                if not demo.finished:
                    demo.fail('完整任务总时间超限')
        else:
            viewer_api = cast(ViewerAPI, import_module('mujoco.viewer'))
            keys: list[int] = []
            running = False
            print('空格键：读取当前起点并开始，或暂停/继续；R：重置整个场景。')
            with viewer_api.launch_passive(demo.loaded.model, demo.loaded.data,
                                            key_callback=keys.append) as viewer:
                with viewer.lock():
                    viewer.cam.lookat[:] = [0., .25, .3]
                    viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 90., -15.
                last = ''
                while viewer.is_running():
                    started = time.monotonic()
                    while keys:
                        key = keys.pop(0)
                        if key == 32:
                            running = demo.start() if demo.state == 'READY' else not running
                        elif key in (82, 114):
                            demo.reset()
                            running = False
                    with viewer.lock():
                        if running and not demo.finished:
                            for _ in range(max(1, round(.01 / demo.config.timestep))):
                                demo.step()
                    if demo.state != last:
                        print(demo.state, demo.failure)
                        last = demo.state
                    viewer.sync()
                    time.sleep(max(0., .01 - (time.monotonic() - started)))
        result = demo.summary()
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        print(text)
        if args.output is not None:
            output = args.output.expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + '\n', encoding='utf-8')
        return 0 if demo.state == 'COMPLETED' else 1
    except (ValueError, RuntimeError, OSError) as exc:
        print('双臂夹取验证失败：' + str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())