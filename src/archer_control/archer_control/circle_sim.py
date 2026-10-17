"""在共享 MuJoCo 场景执行候选双臂半圆并保存跟踪记录，不连接实机。

空格开始或暂停，S 停止并保持当前位置，R 重置仿真。
本程序不启动 ROS 节点；不要同时使用其它程序修改该查看器的数据。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
import csv
from datetime import datetime
from importlib import import_module
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import time
from typing import Protocol, cast

import numpy as np

from archer_mujoco.dual_model_loader import LoadedDualArm, load_dual_arm
from archer_mujoco.model_loader import DataInterface, ModelInterface, mujoco
from .model_checks import CollisionModel, validate_tool_chain
# 兼容原先从 circle_sim 导入间隙检查的外部脚本。
from .model_checks import cross_arm_failure as cross_arm_failure
from .circle_planner import (
    CIRCLE_CENTER, CIRCLE_RADIUS, MOUNT_XYZ, TCP_XYZ, CircleResult, plan_circle,
)
from .circle_visuals import VisualScene, draw_paths
from .dual_arm_trajectory import DualArmKinematics, rigid_transform
from .kinematics import ArmKinematics
from .trajectory import FloatArray, sample_joint_plan


class Camera(Protocol):
    lookat: FloatArray
    distance: float
    azimuth: float
    elevation: float


class Viewer(Protocol):
    cam: Camera
    user_scn: VisualScene
    def lock(self) -> AbstractContextManager[object]: ...
    def sync(self) -> None: ...
    def is_running(self) -> bool: ...
    def close(self) -> None: ...


class ViewerAPI(Protocol):
    def launch_passive(self, model: ModelInterface, data: DataInterface,
                       key_callback: Callable[[int], None] | None = None) -> Viewer: ...


def preflight(loaded: LoadedDualArm, result: CircleResult, clearance: float) -> None:
    """用独立接触模型检查计划路径，避免影响随后执行的动力学状态。"""
    cast(CollisionModel, loaded.model).geom_margin[:] = clearance * .6
    times = set(float(t) for t in result.plan.left.times)
    times.update(float(t) for t in np.linspace(0., result.plan.duration,
                 int(np.ceil(result.plan.duration / .02)) + 1))
    for elapsed in sorted(times):
        for side, plan in (('left', result.plan.left), ('right', result.plan.right)):
            loaded.data.qpos[loaded.arms[side].qpos_indices] = sample_joint_plan(plan, elapsed)[0]
        mujoco.mj_forward(loaded.model, loaded.data)
        reason = cross_arm_failure(loaded, clearance)
        if reason is not None:
            raise ValueError(f'计划路径跨臂间隙不足，t={elapsed:.3f}s：{reason}')


class CircleSimulation:
    def __init__(self, loaded: LoadedDualArm, contact_model: LoadedDualArm,
                 kinematics: DualArmKinematics, result: CircleResult,
                 center: FloatArray, radius: float, clearance: float) -> None:
        self.loaded = loaded
        self.contact_model = contact_model
        self.kinematics = kinematics
        self.result = result
        self.center = center.copy()
        self.radius = radius
        self.clearance = clearance
        self.initial = loaded.data.qpos.copy()
        self.targets = {'left': result.plan.left.positions[0].copy(),
                        'right': result.plan.right.positions[0].copy()}
        self.state = 'READY'
        self.reason = ''
        self.start_time = 0.
        self.goal_since: float | None = None
        self.last_record = -float('inf')
        self.rows: list[list[object]] = []
        self.traces: dict[str, list[FloatArray]] = {'left': [], 'right': []}
        self.maximum_tracking: dict[str, float] = {'left': 0., 'right': 0.}
        self.maximum_circle: dict[str, float] = {'left': 0., 'right': 0.}

    def start_or_pause(self) -> None:
        if self.state == 'READY':
            self.start_time = float(self.loaded.data.time)
            self.state = 'RUNNING'
            print('开始双臂轨迹。')
        elif self.state in ('RUNNING', 'SETTLING'):
            self.state = 'PAUSED'
            print('仿真暂停，空格继续。')
        elif self.state == 'PAUSED':
            self.state = 'RUNNING'
            print('继续轨迹。')

    def stop(self, reason: str, failed: bool = False) -> None:
        for side, arm in self.loaded.arms.items():
            self.targets[side] = self.loaded.data.qpos[arm.qpos_indices].copy()
        self.state = 'FAILED' if failed else 'STOPPED'
        self.reason = reason
        print(self.state + '：' + reason)

    def reset(self) -> None:
        mujoco.mj_resetData(self.loaded.model, self.loaded.data)
        self.loaded.data.qpos[:] = self.initial
        for side, arm in self.loaded.arms.items():
            self.targets[side] = self.initial[arm.qpos_indices].copy()
            self.loaded.data.ctrl[arm.actuator_indices] = self.targets[side]
        mujoco.mj_forward(self.loaded.model, self.loaded.data)
        self.state = 'READY'
        self.reason = ''
        self.goal_since = None
        self.last_record = -float('inf')
        self.rows.clear()
        self.traces = {'left': [], 'right': []}
        self.maximum_tracking = {'left': 0., 'right': 0.}
        self.maximum_circle = {'left': 0., 'right': 0.}
        print('仿真重置，空格开始；本次内存记录已清空。')

    def tick(self) -> None:
        if self.state == 'PAUSED':
            return
        active = self.state in ('RUNNING', 'SETTLING')
        elapsed = max(0., float(self.loaded.data.time) - self.start_time)
        if active:
            for side, plan in (('left', self.result.plan.left), ('right', self.result.plan.right)):
                self.targets[side] = sample_joint_plan(plan, elapsed)[0]
            if elapsed >= self.result.plan.duration:
                self.state = 'SETTLING'
        for side, arm in self.loaded.arms.items():
            self.loaded.data.ctrl[arm.actuator_indices] = self.targets[side]
        # 仿真偏置力补偿；这里不生成任何实机力矩指令。
        mujoco.mj_forward(self.loaded.model, self.loaded.data)
        self.loaded.data.qfrc_applied[:] = self.loaded.data.qfrc_bias
        mujoco.mj_step(self.loaded.model, self.loaded.data)
        if not np.all(np.isfinite(self.loaded.data.qpos)) or not np.all(np.isfinite(self.loaded.data.qvel)):
            self.stop('动力学状态出现非有限数值', failed=True)
            return
        if active and float(self.loaded.data.time) - self.last_record >= .02 - 1e-9:
            self.last_record = float(self.loaded.data.time)
            # mj_step 积分后重新计算位姿，确保记录中的 qpos 和 TCP 属于同一状态。
            mujoco.mj_forward(self.loaded.model, self.loaded.data)
            # 独立接触模型按实际关节角检查，动力学模型本身关闭接触响应。
            self.contact_model.data.qpos[:] = self.loaded.data.qpos
            mujoco.mj_forward(self.contact_model.model, self.contact_model.data)
            failure = cross_arm_failure(self.contact_model, self.clearance)
            if failure is not None:
                self.stop('实际仿真姿态的跨臂间隙不足：' + failure, failed=True)
                return
            self.record(elapsed)
        if self.state == 'SETTLING':
            errors = [float(np.max(np.abs(self.loaded.data.qpos[arm.qpos_indices] - self.targets[side])))
                      for side, arm in self.loaded.arms.items()]
            settled = max(errors) <= .005 and float(np.max(np.abs(self.loaded.data.qvel))) <= .01
            now = float(self.loaded.data.time)
            if settled:
                if self.goal_since is None:
                    self.goal_since = now
                elif now - self.goal_since >= .3:
                    self.state = 'COMPLETED'
                    print('双臂轨迹完成，保持终点。')
            else:
                self.goal_since = None
            if self.state != 'COMPLETED' and elapsed > self.result.plan.duration + 5.:
                self.stop('终点未在规定时间内稳定', failed=True)

    def record(self, elapsed: float) -> None:
        expected = self.kinematics.forward_world(self.targets['left'], self.targets['right'])
        on_arc = elapsed >= float(self.result.plan.left.times[self.result.arc_first_index])
        for side, pose in zip(('left', 'right'), expected):
            arm = self.loaded.arms[side]
            actual = self.loaded.data.site_xpos[arm.tcp_site_id].copy()
            desired = pose[:3, 3]
            tracking = float(np.linalg.norm(actual - desired)) * 1000.
            circle_error: float | None = None
            if on_arc:
                radial = float(np.hypot(actual[0] - self.center[0], actual[2] - self.center[2]))
                circle_error = float(np.hypot(radial - self.radius, actual[1] - self.center[1])) * 1000.
                self.maximum_circle[side] = max(self.maximum_circle[side], circle_error)
                self.traces[side].append(actual)
            self.maximum_tracking[side] = max(self.maximum_tracking[side], tracking)
            self.rows.append([float(self.loaded.data.time), elapsed, 'ARC' if on_arc else 'APPROACH', side,
                              *[float(q) for q in self.targets[side]],
                              *[float(q) for q in self.loaded.data.qpos[arm.qpos_indices]],
                              *[float(x) for x in desired], *[float(x) for x in actual], tracking, circle_error])

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        header = ['simulation_time_s', 'trajectory_time_s', 'phase', 'side']
        header += [f'target_J{i}_rad' for i in range(1, 7)]
        header += [f'actual_J{i}_rad' for i in range(1, 7)]
        header += ['target_x_m', 'target_y_m', 'target_z_m', 'actual_x_m', 'actual_y_m', 'actual_z_m',
                   'tracking_error_mm', 'distance_to_circle_mm']
        with (directory / 'tracking.csv').open('w', newline='', encoding='utf-8-sig') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(self.rows)
        summary = {'state': self.state, 'reason': self.reason,
                   'planned_duration_s': self.result.plan.duration,
                   'maximum_tracking_error_mm': self.maximum_tracking if self.rows else None,
                   'maximum_distance_to_circle_mm': {
                       side: self.maximum_circle[side] if self.traces[side] else None for side in ('left', 'right')},
                   'records': len(self.rows),
                   'scope': '候选安装模型的仿真结果；只监测跨臂间隙，不含自碰撞、桌面或线缆；不可直接作为实机轨迹许可'}
        (directory / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print('记录保存到：' + str(directory.resolve()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--left-urdf')
    parser.add_argument('--right-urdf')
    parser.add_argument('--headless', action='store_true', help='无窗口运行完整仿真')
    parser.add_argument('--auto-start', action='store_true', help='显示窗口后自动开始')
    parser.add_argument('--mount-yaw-deg', type=float, default=0., help='候选夹爪安装偏航角')
    parser.add_argument('--tilt-deg', type=float, default=10., help='左右夹爪内倾角')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    output = args.output_dir or Path.home() / 'archer_ws/circle_results' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    viewer: Viewer | None = None
    simulation: CircleSimulation | None = None
    try:
        tcp = list(TCP_XYZ)
        mount = list(MOUNT_XYZ)
        mount_rpy = [0., 0., float(np.deg2rad(args.mount_yaw_deg))]
        left = (ArmKinematics.from_urdf(args.left_urdf, 'left') if args.left_urdf else
                ArmKinematics.from_description('left', tcp, mount, mount_rpy))
        right = (ArmKinematics.from_urdf(args.right_urdf, 'right') if args.right_urdf else
                 ArmKinematics.from_description('right', tcp, mount, mount_rpy))
        validate_tool_chain(left)
        validate_tool_chain(right)
        world_left = rigid_transform([-.21, 0., 0.], [0., 0., np.pi / 2.])
        world_right = rigid_transform([.21, 0., 0.], [0., 0., np.pi / 2.])
        kinematics = DualArmKinematics(left, right, world_left, world_right)
        def load(contacts: bool) -> LoadedDualArm:
            return load_dual_arm(left_urdf=args.left_urdf, right_urdf=args.right_urdf,
                left_base_xyz=[-.21, 0., 0.], right_base_xyz=[.21, 0., 0.],
                left_base_rpy=[0., 0., np.pi / 2.], right_base_rpy=[0., 0., np.pi / 2.],
                tcp_xyz=tcp, mount_xyz=mount, mount_rpy=mount_rpy, enable_contacts=contacts)
        loaded = load(False)
        # 核对运动学与仿真加载的 TCP，避免两套模型配置不一致。
        poses = kinematics.forward_world(
            loaded.data.qpos[loaded.arms['left'].qpos_indices],
            loaded.data.qpos[loaded.arms['right'].qpos_indices])
        for side, pose in zip(('left', 'right'), poses):
            arm = loaded.arms[side]
            if not np.allclose(pose[:3, 3], loaded.data.site_xpos[arm.tcp_site_id],
                               atol=1e-5, rtol=0.):
                raise ValueError(f'{side} 运动学与仿真 TCP 不一致，请检查 Xacro 和安装参数')
        result = plan_circle(kinematics, loaded.data.qpos[loaded.arms['left'].qpos_indices],
                             loaded.data.qpos[loaded.arms['right'].qpos_indices], inward_tilt=float(np.deg2rad(args.tilt_deg)))
        collision_model = load(True)
        preflight(collision_model, result, .01)
        print(f'预检通过，候选安装偏航角 {args.mount_yaw_deg:.3f}°；轨迹约 {result.plan.duration:.2f} 秒。')
        print('灰线为参考圆，橙线为左臂实际轨迹，青线为右臂实际轨迹。')
        simulation = CircleSimulation(loaded, collision_model, kinematics, result,
                                      np.array(CIRCLE_CENTER), CIRCLE_RADIUS, .01)
        keys: SimpleQueue[int] = SimpleQueue()
        def on_key(key: int) -> None:
            keys.put(key)
        if not args.headless:
            module = cast(ViewerAPI, import_module('mujoco.viewer'))
            viewer = module.launch_passive(loaded.model, loaded.data, key_callback=on_key)
            with viewer.lock():
                viewer.cam.lookat[:] = [0., .18, .30]
                viewer.cam.distance = 1.6
                viewer.cam.azimuth = 90.
                viewer.cam.elevation = -12.
            print('空格开始/暂停，S停止，R重置。Control 滑条由轨迹控制器管理。')
        if args.headless or args.auto_start:
            simulation.start_or_pause()
        last_view = 0.
        while viewer is None or viewer.is_running():
            began = time.monotonic()
            with nullcontext() if viewer is None else viewer.lock():
                while True:
                    try:
                        key = keys.get_nowait()
                    except Empty:
                        break
                    if key == 32:
                        simulation.start_or_pause()
                    elif key == 83:
                        simulation.stop('用户停止')
                    elif key == 82:
                        simulation.reset()
                # 非有限状态出现后冻结物理状态，只保留查看和重置操作。
                if not (simulation.state == 'FAILED' and (
                        not np.all(np.isfinite(loaded.data.qpos)) or not np.all(np.isfinite(loaded.data.qvel)))):
                    simulation.tick()
                if viewer is not None and began - last_view >= 1. / 30.:
                    draw_paths(viewer.user_scn, result.left_targets, result.right_targets,
                               simulation.traces['left'], simulation.traces['right'])
            if viewer is not None:
                if began - last_view >= 1. / 30.:
                    viewer.sync()
                    last_view = began
                time.sleep(max(0., .002 - (time.monotonic() - began)))
            elif simulation.state in ('COMPLETED', 'FAILED', 'STOPPED'):
                break
    except KeyboardInterrupt:
        if simulation is not None and simulation.state != 'COMPLETED':
            simulation.stop('用户中断')
    except (ValueError, OSError, RuntimeError) as exc:
        print('仿真失败：' + str(exc))
        if simulation is not None:
            simulation.stop(str(exc), failed=True)
        else:
            return 1
    finally:
        if viewer is not None:
            viewer.close()
        if simulation is not None:
            if simulation.state in ('RUNNING', 'SETTLING', 'PAUSED'):
                simulation.stop('窗口在轨迹完成前关闭')
            simulation.save(output)
    return 0 if simulation is not None and simulation.state == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())