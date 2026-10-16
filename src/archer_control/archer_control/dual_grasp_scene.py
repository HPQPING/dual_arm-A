"""双臂关节驱动的同物件上下夹持演示，不发布 ROS 或实机命令。

初始关节角由 IK 设置为张开夹爪的预抓取姿态；尚未演示从任意姿态接近物体。
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

from .check_circle import CollisionModel, validate_tool_chain
from .circle_planner import inward_rotation
from .circle_sim import cross_arm_failure
from .cooperative_planner import interpolate_pose, plan_cooperative, sample_errors, sample_object
from .dual_arm_trajectory import DualArmKinematics, rigid_transform
from .kinematics import ArmKinematics, rotation_error
from .trajectory import FloatArray, JointPlan, sample_joint_plan
from archer_mujoco.dual_model_loader import load_dual_arm
from archer_mujoco.grasp_model_loader import load_grasp_model, physics
from archer_mujoco.grasp_scene import GraspConfig, ViewerAPI, _smooth
from archer_mujoco.model_loader import mujoco


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
                 config: GraspConfig = GraspConfig()) -> None:
        config.validate()
        self.config = config
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
        self.object_start = rigid_transform([0., .4, config.center_z], zero)
        grasps: dict[str, FloatArray] = {}
        for side, height in (('left', config.grasp_offset), ('right', -config.grasp_offset)):
            grasp = rigid_transform([0., 0., height], zero)
            grasp[:3, :3] = inward_rotation(side, float(np.deg2rad(10.)))
            grasps[side] = grasp
        seeds: dict[str, FloatArray] = {}
        for side, base, seed in (('left', self.kinematics.world_from_left, 7),
                                 ('right', self.kinematics.world_from_right, 8)):
            arm = self.models[side]
            target = np.linalg.inv(base) @ self.object_start @ grasps[side]
            initial = np.array([0., -.8, 1.6, -.8, 0., 0.], dtype=np.float64)
            candidates: list[FloatArray] = []
            generator = np.random.default_rng(seed)
            for index in range(33):
                guess = initial if index == 0 else generator.uniform(arm.limits[:, 0], arm.limits[:, 1])
                solution = arm.solve_ik(target, guess)
                if solution.success:
                    candidates.append(solution.joints.copy())
            if not candidates:
                raise ValueError(side + ' 预抓取姿态没有找到 IK 解')
            seeds[side] = min(candidates, key=lambda q: float(np.max(np.abs(q - initial))))
        # 每条腿独立插值：抬升、前移、返回、落下；两臂保持固定抓取关系。
        corners = [self.object_start.copy() for _ in range(5)]
        corners[1][2, 3] += config.lift_height
        corners[2][2, 3] += config.lift_height
        corners[2][1, 3] += config.move_distance
        corners[3][2, 3] += config.lift_height
        poses = [corners[0]]
        for start, end in zip(corners[:-1], corners[1:]):
            poses.extend(interpolate_pose(start, end, float(u)) for u in np.linspace(0., 1., 17)[1:])
        self.cooperative = plan_cooperative(self.kinematics, seeds['left'], seeds['right'], poses,
                                             grasps['left'], grasps['right'], minimum_duration=.25)
        self.breakpoints = self.cooperative.paired.left.times[[16, 32, 48, 64]].copy()
        print('开始双臂轨迹与离散跨臂预检……')
        contact = load_dual_arm(left_urdf, right_urdf, left_base_xyz=(-.21, 0., 0.),
            right_base_xyz=(.21, 0., 0.), left_base_rpy=(0., 0., np.pi / 2.),
            right_base_rpy=(0., 0., np.pi / 2.), tcp_xyz=tcp, enable_contacts=True)
        cast(CollisionModel, contact.model).geom_margin[:] = .006
        duration = self.cooperative.paired.duration
        samples = set(float(t) for t in self.cooperative.paired.left.times)
        samples.update(float(t) for t in np.linspace(0., duration, math.ceil(duration / .02) + 1))
        for elapsed in sorted(samples):
            for side, plan in (('left', self.cooperative.paired.left), ('right', self.cooperative.paired.right)):
                contact.data.qpos[contact.arms[side].qpos_indices] = sample_joint_plan(plan, elapsed)[0]
            mujoco.mj_forward(contact.model, contact.data)
            failure = cross_arm_failure(contact, .01)
            if failure is not None:
                raise ValueError('离散跨臂预检失败：' + failure)
            errors = sample_errors(self.kinematics, self.cooperative, elapsed)
            if (max(errors['world_position_m'], errors['relative_position_m']) > .0005
                    or max(errors['world_rotation_rad'], errors['relative_rotation_rad']) > .003):
                raise ValueError('协同路径插值误差超过预检阈值')
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
        print(f'预检通过；初始已放在预抓取姿态。物体中心 Z={config.center_z:.3f} m，'
              f'台面 Z={config.table_z:.3f} m。')

    def reset(self) -> None:
        physics.mj_resetData(self.loaded.model, self.loaded.data)
        self.loaded.data.qpos[:] = self.initial_qpos
        self.loaded.data.ctrl[:] = self.initial_ctrl
        physics.mj_forward(self.loaded.model, self.loaded.data)
        self.state = 'SETTLE'
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
        self.history: list[str] = []

    def fail(self, reason: str) -> None:
        self.state, self.failure, self.finished = 'FAILED', reason, True

    def step(self) -> None:
        if self.finished:
            return
        loaded = self.loaded
        now = float(loaded.data.time)
        motion_time = now - 2.7
        duration = self.cooperative.paired.duration
        if now < .8:
            state, close = 'SETTLE', 0.
        elif now < 2.3:
            state, close = 'CLOSE', _smooth((now - .8) / 1.5)
        elif now < 2.7:
            state, close = 'VERIFY', 1.
        elif motion_time < duration:
            leg = min(3, int(np.searchsorted(self.breakpoints, motion_time, side='right')))
            state, close = ('LIFT', 'MOVE', 'RETURN', 'LOWER')[leg], 1.
        elif motion_time < duration + 1.:
            state, close = 'OPEN', 1. - _smooth(motion_time - duration)
        elif motion_time < duration + 2.:
            state, close = 'FINAL_SETTLE', 0.
        else:
            vertical = abs(float(loaded.data.xmat[loaded.object_body, 8])) > math.cos(math.radians(2.))
            stationary = float(np.linalg.norm(loaded.data.qvel[loaded.object_dof:loaded.object_dof + 6])) < .01
            if (self.verified and self.minimum_lift_clearance > self.config.lift_height * .8
                    and abs(loaded.object_bottom() - self.config.table_z) < .001 and vertical and stationary):
                self.state, self.finished = 'COMPLETED', True
            else:
                self.fail('最终落台、竖直稳定或抬升高度未通过')
            self.history.append(self.state)
            return
        if self.state == 'SETTLE' and state != 'SETTLE':
            speed = np.linalg.norm(loaded.data.qvel[loaded.object_dof:loaded.object_dof + 6])
            if abs(loaded.object_bottom() - self.config.table_z) > .001 or float(speed) > .01:
                self.fail('圆柱没有在台面稳定落定')
                return
        self.state = state
        if not self.history or self.history[-1] != state:
            self.history.append(state)
        left, left_velocity = sample_position_velocity(self.cooperative.paired.left, motion_time)
        right, right_velocity = sample_position_velocity(self.cooperative.paired.right, motion_time)
        loaded.set_arm_targets(left, right, left_velocity, right_velocity)
        loaded.set_grippers(5.65 * close)
        loaded.step()
        if not np.all(np.isfinite(loaded.data.qpos)) or not np.all(np.isfinite(loaded.data.qvel)):
            self.fail('仿真状态出现非有限值')
            return
        fingers, table_force = loaded.contact_forces()
        if state == 'VERIFY':
            self.verified_time = self.verified_time + self.config.timestep if min(fingers.values()) > .02 else 0.
            if self.verified_time > .2:
                self.verified = True
            if now > 2.68 and not self.verified:
                self.fail('上下夹爪尚未形成持续四指接触，禁止抬升')
        if state in ('LIFT', 'MOVE', 'RETURN', 'LOWER'):
            if not self.verified:
                self.fail('夹持未经确认')
                return
            expected = sample_object(self.cooperative, motion_time)
            error = float(np.linalg.norm(loaded.data.xpos[loaded.object_body] - expected[:3, 3]))
            self.maximum_object_error = max(self.maximum_object_error, error)
            actual_left, actual_right = self.kinematics.forward_world(
                loaded.data.qpos[loaded.arms['left'].qpos_indices],
                loaded.data.qpos[loaded.arms['right'].qpos_indices])
            desired = np.linalg.inv(self.cooperative.object_from_left_tcp) @ self.cooperative.object_from_right_tcp
            relative = np.linalg.inv(actual_left) @ actual_right
            self.maximum_relative_error = max(self.maximum_relative_error,
                                               float(np.linalg.norm(relative[:3, 3] - desired[:3, 3])))
            angle = float(np.linalg.norm(rotation_error(desired[:3, :3], relative[:3, :3])))
            self.maximum_relative_angle = max(self.maximum_relative_angle, angle)
            cosine = min(1., max(-1., float(loaded.data.xmat[loaded.object_body, 8])))
            tilt = math.acos(cosine)
            self.maximum_object_tilt = max(self.maximum_object_tilt, tilt)
            placing = (state == 'LOWER' and abs(loaded.object_bottom() - self.config.table_z) < .001
                       and table_force > .05)
            lost = not placing and min(fingers.values()) < .01
            self.contact_loss_time = self.contact_loss_time + self.config.timestep if lost else 0.
            self.maximum_contact_loss_time = max(self.maximum_contact_loss_time, self.contact_loss_time)
            if error > .003 or self.contact_loss_time > .05:
                self.fail('物件误差超过 3 mm，或某个指腹连续失去有效接触超过 50 ms')
            elif self.maximum_relative_error > .005 or angle > .03 or tilt > .03:
                self.fail('两 TCP 相对位姿或物体倾斜超过演示阈值')
            if state in ('MOVE', 'RETURN'):
                self.minimum_lift_clearance = min(self.minimum_lift_clearance,
                                                   loaded.object_bottom() - self.config.table_z)

    def summary(self) -> dict[str, object]:
        return {'state': self.state, 'failure': self.failure, 'config': asdict(self.config),
                'simulation_time_s': float(self.loaded.data.time), 'stages': self.history,
                'maximum_object_position_error_mm': self.maximum_object_error * 1000.,
                'maximum_relative_tcp_position_error_mm': self.maximum_relative_error * 1000.,
                'maximum_relative_tcp_rotation_error_rad': self.maximum_relative_angle,
                'maximum_object_tilt_rad': self.maximum_object_tilt,
                'maximum_model_tcp_difference_mm': self.model_tcp_error * 1000.,
                'maximum_continuous_contact_loss_ms': self.maximum_contact_loss_time * 1000.,
                'minimum_lift_clearance_mm': (self.minimum_lift_clearance * 1000.
                                              if math.isfinite(self.minimum_lift_clearance) else None),
                'final_object_center_world_m': self.loaded.data.xpos[self.loaded.object_body].tolist(),
                'scope': '双臂关节动力学与简化指腹接触；预抓取姿态初始化；未接入 ROS 或实机，未含整臂自碰撞、台面或线缆检查'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-urdf')
    parser.add_argument('--right-urdf')
    parser.add_argument('--headless', action='store_true', help='不开窗口，自动执行并输出结果')
    parser.add_argument('--output', type=Path, help='可选的 JSON 结果文件')
    args = parser.parse_args()
    try:
        demo = DualGraspDemo(args.left_urdf, args.right_urdf)
        if args.headless:
            steps = math.ceil((demo.cooperative.paired.duration + 6.) / demo.config.timestep)
            for _ in range(steps):
                demo.step()
                if demo.finished:
                    break
        else:
            viewer_api = cast(ViewerAPI, import_module('mujoco.viewer'))
            keys: list[int] = []
            running = False
            print('空格键：开始或暂停；R：重置。橙色左臂夹上部，青色右臂夹下部。')
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
                            running = not running
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