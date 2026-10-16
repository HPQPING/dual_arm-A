"""完整圆柱的简化抓取演示：指令 2.0 表示虚拟附着，不验证真实夹持力。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .cooperative_planner import sample_object
from .dual_grasp_scene import DualGraspDemo, sample_position_velocity
from .dual_arm_trajectory import plan_paired_joints
from .kinematics import ArmKinematics
from .simple_grasp_planner import (SimpleGraspMotion, check_runtime_paths,
                                  joint6_zero_target, prepare_simple_motion)
from .trajectory import FloatArray, joint_vector
from archer_mujoco.grasp_model_loader import GraspModel, physics
from archer_mujoco.grasp_scene import GraspConfig


# 仅用于仿真世界启动及重置；每轮任务仍从当前 qpos 保存实际起点。
SIMULATION_INITIAL = {
    'left': (0.0016936991186183399, -1.5557692501183322, 3.131012580369181,
             -1.5715998405409322, -0.05458065430333334, 1.6680315339869978),
    'right': (-0.008299789093679984, -1.5717756091728772, 3.132204154731199,
              -1.5717483593281518, -0.02457215514032048, 1.4469849024526769),
}


class DemoGeometry(GraspModel, Protocol):
    geom_contype: NDArray[np.int32]
    geom_conaffinity: NDArray[np.int32]


class SimpleGraspDemo(DualGraspDemo):
    """沿用真实关节动力学与完整工件几何；搬运工件使用显式虚拟附着。"""

    def __init__(self, left_urdf: str | None = None, right_urdf: str | None = None,
                 config: GraspConfig = GraspConfig(), left_initial: FloatArray | None = None,
                 right_initial: FloatArray | None = None) -> None:
        requested: dict[str, FloatArray] = {}
        applied: dict[str, FloatArray] = {}
        for side, path, initial in (('left', left_urdf, left_initial), ('right', right_urdf, right_initial)):
            arm = (ArmKinematics.from_urdf(path, side) if path else
                   ArmKinematics.from_description(side, (0., 0., .147021696), (0., 0., 0.), (0., 0., 0.)))
            values = joint_vector(SIMULATION_INITIAL[side] if initial is None else initial, side + '_initial')
            # 原记录略超 URDF 两位小数限位；只允许微小仿真修正，边界留数值裕量。
            bounded = np.clip(values, arm.limits[:, 0] + 1e-5, arm.limits[:, 1] - 1e-5)
            if np.max(np.abs(bounded - values)) > .002:
                raise ValueError(side + ' 仿真初值超限超过 0.002 rad，拒绝自动修正')
            requested[side], applied[side] = values, bounded
            if not np.array_equal(values, bounded):
                print(side + ' 仿真启动角限位修正 rad：', (bounded - values).tolist())
        self.simulation_initial_requested = {side: q.tolist() for side, q in requested.items()}
        self.simulation_initial_applied = {side: q.tolist() for side, q in applied.items()}
        super().__init__(left_urdf, right_urdf, config, applied['left'], applied['right'])
        geometry = cast(DemoGeometry, self.loaded.model)
        # 此模式不让简化指腹与工件求解接触，避免把指令 2.0 误当成真实夹紧。
        for index in self.loaded.finger_geoms:
            geometry.geom_contype[index] = 0
            geometry.geom_conaffinity[index] = 0
        geometry.geom_conaffinity[self.loaded.object_geom] = 1
        self.get_mode_notice()

    def get_mode_notice(self) -> None:
        print('简化演示：完整 65 cm 圆柱，夹爪目标 2.0，工件虚拟附着；不是摩擦抓取验证。')

    def reset(self) -> None:
        super().reset()
        self.attached = False
        self.gripper_command = 0.
        self.attachment_center = np.zeros(3, dtype=np.float64)
        self.attachment_tcp_midpoint = np.zeros(3, dtype=np.float64)
        self.attachment_relative = np.zeros(3, dtype=np.float64)
        self.maximum_attachment_error = 0.
        self.maximum_alignment_tcp_drift = 0.
        self.alignment_tcp: tuple[FloatArray, FloatArray] | None = None
        self.insertion_start_tcp: tuple[FloatArray, FloatArray] | None = None
        self.insertion_displacements: dict[str, list[float]] = {}
        self.plane_alignment_errors: dict[str, dict[str, float]] = {}
        self.arrival_joint6: dict[str, dict[str, float]] = {}

    def start(self) -> bool:
        if self.state != 'READY':
            return False
        try:
            physics.mj_forward(self.loaded.model, self.loaded.data)
            current = self.read_joints()
            for side, arm in self.loaded.arms.items():
                if float(np.max(np.abs(self.loaded.data.qvel[arm.dof_indices]))) > .01:
                    raise ValueError(side + ' 尚未静止')
                if float(np.max(np.abs(self.loaded.data.qpos[self.loaded.finger_positions[side]]))) > .001:
                    raise ValueError(side + ' 夹爪尚未完全张开')
            self.enter('PLANNING')
            motion = prepare_simple_motion(self.kinematics, current, self.config, self.left_urdf, self.right_urdf)
            if any(np.max(np.abs(self.read_joints()[side] - q)) > .002 for side, q in current.items()):
                raise ValueError('规划期间起点变化')
            self.motion = motion
            self.enter('SETTLE')
            return True
        except (ValueError, RuntimeError, OSError) as exc:
            self.fail('简化抓取规划失败：' + str(exc))
            return False

    def tcp_points(self) -> tuple[FloatArray, FloatArray]:
        return (self.loaded.data.site_xpos[self.loaded.arms['left'].tcp_site_id].copy(),
                self.loaded.data.site_xpos[self.loaded.arms['right'].tcp_site_id].copy())

    def joint6_arrived(self) -> bool:
        # 验收到位后的实际 J6 接近用户指定的零目标，不以平面求解覆盖指令。
        return all(abs(float(self.loaded.data.qpos[arm.qpos_indices[5]])) < .003
                   for arm in self.loaded.arms.values())

    def record_plane(self, label: str, normal_axis: int) -> bool:
        errors: dict[str, float] = {}
        for side, arm in self.loaded.arms.items():
            matrix = self.loaded.data.site_xmat[arm.tcp_site_id].reshape(3, 3)
            errors[side] = math.acos(min(1., abs(float(matrix[normal_axis, 0]))))
        self.plane_alignment_errors[label] = errors
        return max(errors.values()) < .01

    def prepare_arrival_alignment(self, motion: SimpleGraspMotion) -> None:
        """到位稳定后读取实际反馈，仅将两侧 J6 目标设为零，并重检路径。"""
        actual = self.read_joints()
        targets: dict[str, FloatArray] = {}
        for side, arm, old in (
            ('left', self.models['left'], motion.alignment.left),
            ('right', self.models['right'], motion.alignment.right),
        ):
            targets[side] = joint6_zero_target(arm, actual[side])
            if np.max(np.abs(targets[side] - old.positions[-1])) > .003:
                raise ValueError(side + ' 到位反馈修正超出既有后续路径范围，需要重新规划')
            self.arrival_joint6[side] = {'actual_rad': float(actual[side][5]),
                'target_rad': float(targets[side][5]),
                'rotation_rad': float(targets[side][5] - actual[side][5])}
        alignment = plan_paired_joints(actual['left'], actual['right'], [targets['left']], [targets['right']],
            left_limits=self.models['left'].limits, right_limits=self.models['right'].limits, minimum_duration=1.5)
        insertion = plan_paired_joints(targets['left'], targets['right'], motion.insertion.left.positions[1:],
            motion.insertion.right.positions[1:], left_limits=self.models['left'].limits,
            right_limits=self.models['right'].limits, minimum_duration=.125)
        check_runtime_paths(motion.collision, (('ALIGN_JOINT6', alignment), ('INSERT', insertion)))
        self.motion = replace(motion, alignment=alignment, insertion=insertion)
        print('到位后读取 J6，并将两侧绝对目标设为 0：', self.arrival_joint6)

    def attach(self) -> None:
        left, right = self.tcp_points()
        self.attachment_center = self.loaded.data.xpos[self.loaded.object_body].copy()
        self.attachment_tcp_midpoint = (left + right) / 2.
        self.attachment_relative = right - left
        self.attached = True

    def follow_virtual_attachment(self) -> None:
        """工件中心跟随双 TCP 实际位移的平均值，竖直姿态保持不变。"""
        left, right = self.tcp_points()
        error = float(np.linalg.norm((right - left) - self.attachment_relative))
        self.maximum_attachment_error = max(self.maximum_attachment_error, error)
        if error > .003:
            self.fail('虚拟夹持期间两 TCP 相对位置漂移超过 3 mm')
            return
        position = self.attachment_center + (left + right) / 2. - self.attachment_tcp_midpoint
        # 放置末段允许小量跟踪误差，但显示几何不得穿过台面。
        if position[2] < self.config.center_z - .001:
            self.fail('虚拟工件的底部会穿过台面超过 1 mm')
            return
        position[2] = max(position[2], self.config.center_z)
        q = self.loaded.object_qpos
        self.loaded.data.qpos[q:q + 3] = position
        self.loaded.data.qpos[q + 3:q + 7] = [1., 0., 0., 0.]
        d = self.loaded.object_dof
        self.loaded.data.qvel[d:d + 6] = 0.
        physics.mj_forward(self.loaded.model, self.loaded.data)

    def step(self) -> None:
        motion = self.motion
        if self.finished or not isinstance(motion, SimpleGraspMotion):
            return
        data = self.loaded.data
        now = float(data.time)
        elapsed = now - self.stage_started
        active = self.state
        paired, sample_time, command = motion.approach, 0., 0.
        if active in ('APPROACH', 'APPROACH_SETTLE'):
            sample_time = elapsed if active == 'APPROACH' else paired.duration
        elif active in ('ALIGN_JOINT6', 'ALIGN_SETTLE', 'OPEN_BEFORE_INSERT'):
            paired = motion.alignment
            sample_time = elapsed if active == 'ALIGN_JOINT6' else paired.duration
        elif active in ('INSERT', 'INSERT_SETTLE', 'CLOSE_DEMO'):
            paired = motion.insertion
            sample_time = elapsed if active == 'INSERT' else paired.duration
            if active == 'CLOSE_DEMO':
                u = min(1., elapsed)
                command = 2. * (6. * u**5 - 15. * u**4 + 10. * u**3)
        elif active in ('LIFT', 'MOVE', 'RETURN', 'LOWER', 'PLACE_SETTLE', 'RELEASE', 'OPEN_WAIT'):
            paired = motion.cooperative.paired
            sample_time = now - self.motion_started if active in ('LIFT', 'MOVE', 'RETURN', 'LOWER') else paired.duration
            command = 2.
            if active == 'RELEASE':
                u = min(1., elapsed)
                command = 2. * (1. - (6. * u**5 - 15. * u**4 + 10. * u**3))
            elif active == 'OPEN_WAIT':
                command = 0.
            if active in ('LIFT', 'MOVE', 'RETURN', 'LOWER'):
                leg = min(3, int(np.searchsorted(paired.left.times[[16, 32, 48, 64]], sample_time, side='right')))
                label = ('LIFT', 'MOVE', 'RETURN', 'LOWER')[leg]
                if label != self.state:
                    self.enter(label)
        elif active in ('RETREAT', 'RETREAT_SETTLE'):
            paired = motion.retreat
            sample_time = elapsed if active == 'RETREAT' else paired.duration
        elif active in ('RETURN_HOME', 'HOME_SETTLE'):
            paired = motion.home
            sample_time = elapsed if active == 'RETURN_HOME' else paired.duration
        left, lv = sample_position_velocity(paired.left, sample_time)
        right, rv = sample_position_velocity(paired.right, sample_time)
        self.loaded.set_arm_targets(left, right, lv, rv)
        self.gripper_command = command
        self.loaded.set_grippers(command)
        self.loaded.step()
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            self.fail('仿真反馈出现非有限值')
            return
        if self.attached:
            self.follow_virtual_attachment()
            if self.finished:
                return
        if active == 'ALIGN_JOINT6' and self.alignment_tcp is not None:
            self.maximum_alignment_tcp_drift = max(self.maximum_alignment_tcp_drift,
                *(float(np.linalg.norm(a-b)) for a, b in zip(self.tcp_points(), self.alignment_tcp)))
            if self.maximum_alignment_tcp_drift > .0005:
                self.fail('仅 J6 对齐时 TCP 漂移超过 0.5 mm')
                return
        elapsed = float(data.time) - self.stage_started
        if self.state == 'SETTLE' and elapsed >= .8:
            if self.grounded():
                self.enter('APPROACH')
            else:
                self.fail('完整圆柱未在台面稳定落定')
        elif self.state == 'APPROACH' and elapsed >= motion.approach.duration:
            self.enter('APPROACH_SETTLE')
        elif self.state == 'APPROACH_SETTLE':
            self.wait_stable(self.converged(motion.approach), 'ALIGN_JOINT6')
            if self.state == 'ALIGN_JOINT6':
                try:
                    self.prepare_arrival_alignment(motion)
                except (ValueError, RuntimeError, OSError) as exc:
                    self.fail('到位后 J6 对齐计算或预检失败：' + str(exc))
                    return
                self.alignment_tcp = self.tcp_points()
        elif self.state == 'ALIGN_JOINT6' and elapsed >= motion.alignment.duration:
            self.enter('ALIGN_SETTLE')
        elif self.state == 'ALIGN_SETTLE':
            self.wait_stable(self.converged(motion.alignment) and self.joint6_arrived(), 'OPEN_BEFORE_INSERT')
            if self.state == 'OPEN_BEFORE_INSERT':
                self.record_plane('after_XOY_error_rad', 2)
        elif self.state == 'OPEN_BEFORE_INSERT' and elapsed >= .5:
            self.insertion_start_tcp = self.tcp_points()
            self.enter('INSERT')
        elif self.state == 'INSERT' and elapsed >= motion.insertion.duration:
            self.enter('INSERT_SETTLE')
        elif self.state == 'INSERT_SETTLE':
            self.wait_stable(self.converged(motion.insertion) and self.joint6_arrived(), 'CLOSE_DEMO')
            if self.state == 'CLOSE_DEMO' and self.insertion_start_tcp is not None:
                for side, start, end in zip(('left', 'right'), self.insertion_start_tcp, self.tcp_points()):
                    displacement = end - start
                    self.insertion_displacements[side] = displacement.tolist()
                    if np.linalg.norm(displacement - [0., .01, 0.]) > .0005:
                        self.fail(side + ' 实际 +Y 进给与 1 cm 目标偏差超过 0.5 mm')
                        return
        elif self.state == 'CLOSE_DEMO' and elapsed >= 1.:
            self.attach()
            self.motion_started = float(data.time)
            self.enter('LIFT')
        elif self.state in ('LIFT', 'MOVE', 'RETURN', 'LOWER'):
            expected = sample_object(motion.cooperative, float(data.time) - self.motion_started)
            self.maximum_object_error = max(self.maximum_object_error,
                float(np.linalg.norm(data.xpos[self.loaded.object_body] - expected[:3, 3])))
            if self.maximum_object_error > .005:
                self.fail('虚拟工件轨迹跟踪误差超过 5 mm')
                return
            if self.state in ('MOVE', 'RETURN'):
                self.minimum_lift_clearance = min(self.minimum_lift_clearance,
                    self.loaded.object_bottom() - self.config.table_z)
            if float(data.time) - self.motion_started >= motion.cooperative.paired.duration:
                self.enter('PLACE_SETTLE')
        elif self.state == 'PLACE_SETTLE':
            self.wait_stable(self.converged(motion.cooperative.paired) and self.grounded(), 'RELEASE')
            if self.state == 'RELEASE':
                self.attached = False
        elif self.state == 'RELEASE' and elapsed >= 1.:
            self.enter('OPEN_WAIT')
        elif self.state == 'OPEN_WAIT':
            opened = all(np.max(np.abs(data.qpos[indices])) < .001 for indices in self.loaded.finger_positions.values())
            self.wait_stable(bool(opened) and self.grounded(), 'RETREAT')
        elif self.state == 'RETREAT' and elapsed >= motion.retreat.duration:
            self.enter('RETREAT_SETTLE')
        elif self.state == 'RETREAT_SETTLE':
            self.wait_stable(self.converged(motion.retreat), 'RETURN_HOME')
        elif self.state == 'RETURN_HOME' and elapsed >= motion.home.duration:
            self.enter('HOME_SETTLE')
        elif self.state == 'HOME_SETTLE':
            actual = self.read_joints()
            self.home_error = {side: float(np.max(np.abs(actual[side] - motion.start_joints[side]))) for side in actual}
            self.wait_stable(self.converged(motion.home) and self.grounded(), 'COMPLETED', dwell=.3)
            self.finished = self.state == 'COMPLETED'

    def summary(self) -> dict[str, object]:
        result = super().summary()
        result.pop('maximum_continuous_contact_loss_ms', None)
        result.pop('maximum_relative_tcp_rotation_error_rad', None)
        result.pop('maximum_relative_tcp_position_error_mm', None)
        result.pop('maximum_object_tilt_rad', None)
        result.update({'mode': 'simple_demo', 'attachment_method': 'virtual_translation_follow',
            'gripper_closed_command': 2., 'gripper_command': self.gripper_command, 'attached': self.attached,
            'actual_insertion_displacements_world_m': self.insertion_displacements,
            'gripper_plane': '真实 GP80 网格的工具 YZ 平面，法向为工具 X',
            'gripper_visual': 'original_GP80_static_mesh',
            'initial_position_source': '任务开始时读取当前 MuJoCo data.qpos',
            'simulation_initial_requested_rad': self.simulation_initial_requested,
            'simulation_initial_applied_rad': self.simulation_initial_applied,
            'plane_alignment_errors': self.plane_alignment_errors,
            'arrival_joint6': self.arrival_joint6,
            'maximum_attachment_relative_error_mm': self.maximum_attachment_error * 1000.,
            'maximum_alignment_tcp_drift_mm': self.maximum_alignment_tcp_drift * 1000.,
            'scope': '完整圆柱几何；真实关节动力学；虚拟附着演示，不验证夹紧力或摩擦，不连接实机'})
        return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true', help='此独立验证入口始终无窗口')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    demo = SimpleGraspDemo()
    if demo.start() and isinstance(demo.motion, SimpleGraspMotion):
        motion = demo.motion
        duration = sum(p.duration for p in (motion.approach, motion.alignment, motion.insertion,
                       motion.cooperative.paired, motion.retreat, motion.home))
        for _ in range(math.ceil((duration + 25.) / demo.config.timestep)):
            demo.step()
            if demo.finished:
                break
        if not demo.finished:
            demo.fail('简化演示总时间超限')
    text = json.dumps(demo.summary(), ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.output:
        args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.output.expanduser().write_text(text + '\n', encoding='utf-8')
    return 0 if demo.state == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
