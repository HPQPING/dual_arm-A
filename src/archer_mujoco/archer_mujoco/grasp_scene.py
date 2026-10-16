"""同一圆柱上下夹持的接触验证台，不连接 ROS 或实机。

采用简化平行指腹和运动学支撑机构，圆柱是受重力和接触作用的自由刚体。
不使用物体焊接、吸附、重力补偿或直接改写物体位姿来模拟抓取。
本模块尚未接入 Archer 的十二个关节或现有任务节点。
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from importlib import import_module
import math
import time
from typing import Protocol, TypeAlias, cast
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]


class Model(Protocol):
    body_mocapid: NDArray[np.int32]


class Contact(Protocol):
    geom1: int
    geom2: int
    dist: float


class ContactList(Protocol):
    def __getitem__(self, index: int) -> Contact: ...


class Data(Protocol):
    time: float
    qpos: FloatArray
    qvel: FloatArray
    ctrl: FloatArray
    mocap_pos: FloatArray
    xpos: FloatArray
    xmat: FloatArray
    ncon: int
    contact: ContactList


class ModelFactory(Protocol):
    def from_xml_string(self, xml: str) -> Model: ...


class DataFactory(Protocol):
    def __call__(self, model: Model) -> Data: ...


class ObjectKinds(Protocol):
    mjOBJ_BODY: object
    mjOBJ_GEOM: object
    mjOBJ_ACTUATOR: object


class Physics(Protocol):
    MjModel: ModelFactory
    MjData: DataFactory
    mjtObj: ObjectKinds
    def mj_name2id(self, model: Model, kind: object, name: str) -> int: ...
    def mj_forward(self, model: Model, data: Data) -> None: ...
    def mj_step(self, model: Model, data: Data) -> None: ...
    def mj_resetData(self, model: Model, data: Data) -> None: ...
    def mj_contactForce(self, model: Model, data: Data, index: int, force: FloatArray) -> None: ...


class Camera(Protocol):
    lookat: FloatArray
    distance: float
    azimuth: float
    elevation: float


class Viewer(Protocol):
    cam: Camera
    def is_running(self) -> bool: ...
    def sync(self) -> None: ...
    def lock(self) -> AbstractContextManager[object]: ...


class ViewerAPI(Protocol):
    def launch_passive(self, model: Model, data: Data, *,
                       key_callback: Callable[[int], None]) -> AbstractContextManager[Viewer]: ...


physics = cast(Physics, import_module('mujoco'))


@dataclass(frozen=True)
class GraspConfig:
    # 用户已确认的尺寸；world 与左右机械臂基座原点同高。
    pedestal_height: float = .01
    object_height: float = .65
    object_diameter: float = .01
    object_mass: float = .05
    # 以下为独立接触测试参数，并非实机夹爪标定或力参数。
    initial_clearance: float = .001
    grasp_offset: float = .20
    lift_height: float = .04
    move_distance: float = .02
    friction: float = .8
    finger_force: float = 2.
    timestep: float = .001

    @property
    def table_z(self) -> float:
        return -self.pedestal_height

    @property
    def center_z(self) -> float:
        return self.table_z + self.object_height / 2.

    def validate(self) -> None:
        for name, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(name + ' 必须为有限数值')
        if not 0. <= self.pedestal_height <= .5:
            raise ValueError('垫架高度必须在 0～0.5 m 内')
        if not .05 <= self.object_height <= 1. or not .002 <= self.object_diameter < .075:
            raise ValueError('测试物件高度须为 0.05～1 m，直径须为 0.002～0.075 m')
        if not .001 <= self.object_mass <= 1.:
            raise ValueError('测试物件质量须为 0.001～1 kg')
        if not 0. <= self.initial_clearance <= .003:
            raise ValueError('初始离台间隙须为 0～3 mm')
        if not .02 < self.grasp_offset < self.object_height / 2. - .02:
            raise ValueError('上下抓取位置必须在圆柱内部，并留出指腹高度')
        if not .01 <= self.lift_height <= .10 or not 0. <= self.move_distance <= .10:
            raise ValueError('测试抬升须为 0.01～0.10 m，平移须为 0～0.10 m')
        if not 0. <= self.friction <= 2. or not 0. < self.finger_force <= 10.:
            raise ValueError('测试摩擦系数须为 0～2，单指执行器力上限须为 0～10 N')
        if not .0002 <= self.timestep <= .002:
            raise ValueError('时间步长须为 0.0002～0.002 s')


def _text(values: tuple[float, ...]) -> str:
    return ' '.join(format(value, '.12g') for value in values)


def build_xml(config: GraspConfig) -> str:
    """生成独立验证台：工作台、自由圆柱、上下两组可动指腹。"""
    config.validate()
    root = ET.Element('mujoco', model='上下夹持圆柱接触验证')
    ET.SubElement(root, 'compiler', angle='radian', autolimits='true')
    ET.SubElement(root, 'option', timestep=str(config.timestep), gravity='0 0 -9.81',
                  integrator='implicitfast', iterations='100', cone='elliptic')
    default = ET.SubElement(root, 'default')
    ET.SubElement(default, 'geom', friction=_text((config.friction, .005, .0001)),
                  condim='4', solref='.004 1', solimp='.99 .999 .0001')
    world = ET.SubElement(root, 'worldbody')
    ET.SubElement(world, 'light', pos='0 -.2 1.8', dir='0 .4 -1', diffuse='.8 .8 .8')
    ET.SubElement(world, 'geom', name='table', type='box',
                  pos=_text((0., .35, config.table_z - .02)), size='.6 .7 .02',
                  rgba='.25 .3 .3 1', contype='1', conaffinity='2')
    # 模型底面仅作初始化避让；后续由重力和接触求解，不持续保留这 1 mm。
    cylinder = ET.SubElement(world, 'body', name='object',
                             pos=_text((0., .40, config.center_z + config.initial_clearance)))
    ET.SubElement(cylinder, 'freejoint', name='object_free')
    ET.SubElement(cylinder, 'geom', name='object_geom', type='cylinder',
                  size=_text((config.object_diameter / 2., config.object_height / 2.)),
                  mass=str(config.object_mass), rgba='.2 .8 .3 1', contype='2', conaffinity='5')
    actuators = ET.SubElement(root, 'actuator')
    equality = ET.SubElement(root, 'equality')
    for side, height, color in (('left', config.center_z + config.grasp_offset, '1 .5 .1 1'),
                                ('right', config.center_z - config.grasp_offset, '0 .7 1 1')):
        # 支撑目标通过约束驱动有质量的夹爪，接触计算包含夹爪的真实动力学速度。
        # 约束仅连接支撑目标与夹爪，绝不连接或吸附圆柱。
        ET.SubElement(world, 'body', name=side + '_target', mocap='true',
                      pos=_text((0., .4, height)))
        carrier = ET.SubElement(world, 'body', name=side + '_carrier',
                                pos=_text((0., .4, height)))
        ET.SubElement(carrier, 'freejoint', name=side + '_carrier_free')
        ET.SubElement(equality, 'weld', body1=side + '_target', body2=side + '_carrier',
                      solref='.004 1', solimp='.99 .999 .001')
        # 支撑件仅显示；本轮不代表机械臂、GP80 连杆或真实可达空间。
        ET.SubElement(carrier, 'geom', type='box', pos='0 -.045 0', size='.06 .006 .02',
                      mass='.10', rgba=color, contype='0', conaffinity='0')
        for label, sign in (('a', -1.), ('b', 1.)):
            name = side + '_finger_' + label
            finger = ET.SubElement(carrier, 'body', name=name, pos=_text((sign * .044, 0., 0.)))
            ET.SubElement(finger, 'joint', name=name, type='slide',
                          axis=_text((-sign, 0., 0.)), range='0 .04', damping='1')
            ET.SubElement(finger, 'geom', name=name, type='box', size='.004 .012 .015',
                          mass='.02', rgba=color, contype='4', conaffinity='2')
            ET.SubElement(actuators, 'position', name=name, joint=name, kp='800',
                          ctrlrange='0 .04', forcerange=_text((-config.finger_force, config.finger_force)))
    return ET.tostring(root, encoding='unicode')


def _smooth(value: float) -> float:
    u = min(1., max(0., value))
    return 6. * u**5 - 15. * u**4 + 10. * u**3


class GraspBench:
    # 本测试流程自动执行，但与现有 ROS 任务节点和实机完全隔离。
    stages: tuple[tuple[str, float], ...] = (
        ('SETTLE', .8), ('CLOSE', 1.5), ('VERIFY', .4), ('LIFT', 2.),
        ('HOLD', .6), ('MOVE', 2.), ('RETURN', 2.), ('LOWER', 2.),
        ('OPEN', 1.), ('FINAL_SETTLE', 1.),
    )

    def __init__(self, config: GraspConfig = GraspConfig()) -> None:
        self.config = config
        self.model = physics.MjModel.from_xml_string(build_xml(config))
        self.data = physics.MjData(self.model)
        kinds = physics.mjtObj
        self.object_body = self._id(kinds.mjOBJ_BODY, 'object')
        self.object_geom = self._id(kinds.mjOBJ_GEOM, 'object_geom')
        self.table_geom = self._id(kinds.mjOBJ_GEOM, 'table')
        self.fingers = {self._id(kinds.mjOBJ_GEOM, side + '_finger_' + label): side + '_' + label
                        for side in ('left', 'right') for label in ('a', 'b')}
        self.controls = [self._id(kinds.mjOBJ_ACTUATOR, side + '_finger_' + label)
                         for side in ('left', 'right') for label in ('a', 'b')]
        self.carriers = [int(self.model.body_mocapid[self._id(kinds.mjOBJ_BODY, side + '_target')])
                         for side in ('left', 'right')]
        self.reset()

    def _id(self, kind: object, name: str) -> int:
        index = physics.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError('模型缺少：' + name)
        return index

    def reset(self) -> None:
        physics.mj_resetData(self.model, self.data)
        physics.mj_forward(self.model, self.data)
        self.initial_carriers = self.data.mocap_pos[self.carriers].copy()
        self.rest_center = np.array([0., .4, self.config.center_z], dtype=np.float64)
        self.state = 'SETTLE'
        self.failure = ''
        self.finished = False
        self.grip_verified = False
        self.verified_time = 0.
        self.maximum_tracking_error = 0.
        self.minimum_lift_clearance = math.inf
        self.maximum_penetration = 0.
        self.table_normal_force = 0.
        self.finger_normal_forces = {name: 0. for name in self.fingers.values()}

    def set_gripper_position(self, position: float) -> None:
        """沿用 0 张开、5.65 闭合的仿真指令约定，不声明其为真实电机行程。"""
        if not math.isfinite(position) or not 0. <= position <= 5.65:
            raise ValueError('夹爪指令须在 0～5.65 内')
        self.data.ctrl[self.controls] = .04 * position / 5.65

    def object_bottom(self) -> float:
        """按圆柱实际姿态计算最低点，倾斜时也不能只减去半高。"""
        cosine = min(1., abs(float(self.data.xmat[self.object_body, 8])))
        extent = (self.config.object_height / 2. * cosine
                  + self.config.object_diameter / 2. * math.sqrt(max(0., 1. - cosine * cosine)))
        return float(self.data.xpos[self.object_body, 2]) - extent

    def _contacts(self) -> None:
        forces = {name: 0. for name in self.fingers.values()}
        table_force = 0.
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
            force = np.zeros(6, dtype=np.float64)
            physics.mj_contactForce(self.model, self.data, index, force)
            if other in self.fingers:
                forces[self.fingers[other]] += max(0., float(force[0]))
            elif other == self.table_geom:
                table_force += max(0., float(force[0]))
            self.maximum_penetration = max(self.maximum_penetration, -float(contact.dist))
        self.finger_normal_forces = forces
        self.table_normal_force = table_force

    def _fail(self, reason: str) -> None:
        self.failure = reason
        self.state = 'FAILED'
        self.finished = True

    def step(self) -> None:
        if self.finished:
            return
        elapsed = float(self.data.time)
        boundary = 0.
        stage, fraction = 'FINAL_SETTLE', 1.
        for candidate, duration in self.stages:
            if elapsed < boundary + duration:
                stage, fraction = candidate, (elapsed - boundary) / duration
                break
            boundary += duration
        else:
            self._finish()
            return
        previous = self.state
        self.state = stage
        if previous == 'SETTLE' and stage != 'SETTLE':
            if (abs(self.object_bottom() - self.config.table_z) > .001
                    or float(np.linalg.norm(self.data.qvel[:6])) > .01):
                self._fail('圆柱未在台面稳定落定')
                return
            self.rest_center = self.data.xpos[self.object_body].copy()
        progress = _smooth(fraction)
        close = 0. if stage == 'SETTLE' else progress if stage == 'CLOSE' else 1.
        if stage == 'OPEN':
            close = 1. - progress
        elif stage == 'FINAL_SETTLE':
            close = 0.
        dz = 0. if stage in ('SETTLE', 'CLOSE', 'VERIFY', 'OPEN', 'FINAL_SETTLE') else self.config.lift_height
        dy = 0.
        if stage == 'LIFT':
            dz *= progress
        elif stage == 'MOVE':
            dy = self.config.move_distance * progress
        elif stage == 'RETURN':
            dy = self.config.move_distance * (1. - progress)
        elif stage == 'LOWER':
            dz *= 1. - progress
        offset = np.array([0., dy, dz], dtype=np.float64)
        self.data.mocap_pos[self.carriers] = self.initial_carriers + offset
        self.set_gripper_position(5.65 * close)
        physics.mj_step(self.model, self.data)
        # 接触力保留本物理步的求解结果；刷新运动学后读取最新物体位置。
        self._contacts()
        physics.mj_forward(self.model, self.data)
        if not np.all(np.isfinite(self.data.qpos)) or not np.all(np.isfinite(self.data.qvel)):
            self._fail('仿真状态出现非有限值')
            return
        if stage == 'VERIFY':
            contacting = min(self.finger_normal_forces.values()) > .02
            self.verified_time = self.verified_time + self.config.timestep if contacting else 0.
            if self.verified_time >= .2:
                self.grip_verified = True
            if fraction > .95 and not self.grip_verified:
                self._fail('上下夹爪未形成持续的双指接触')
        if stage in ('LIFT', 'HOLD', 'MOVE', 'RETURN', 'LOWER'):
            if not self.grip_verified:
                self._fail('未确认夹持，禁止抬起')
                return
            error = float(np.linalg.norm(self.data.xpos[self.object_body] - self.rest_center - offset))
            self.maximum_tracking_error = max(self.maximum_tracking_error, error)
            # 放置末段允许台面接管支撑力；离台运动仍要求四个指腹持续有效接触。
            placing = (stage == 'LOWER' and abs(self.object_bottom() - self.config.table_z) < .001
                       and self.table_normal_force > .05)
            if error > .003 or (not placing and min(self.finger_normal_forces.values()) < .01):
                self._fail('物体滑移超过 3 mm，或某个指腹失去有效接触')
            if stage in ('HOLD', 'MOVE', 'RETURN'):
                self.minimum_lift_clearance = min(self.minimum_lift_clearance,
                                                   self.object_bottom() - self.config.table_z)

    def _finish(self) -> None:
        grounded = abs(self.object_bottom() - self.config.table_z) < .001
        upright = abs(float(self.data.xmat[self.object_body, 8])) > math.cos(math.radians(2.))
        stopped = float(np.linalg.norm(self.data.qvel[:6])) < .01
        if (not self.grip_verified or self.minimum_lift_clearance < self.config.lift_height * .8
                or not grounded or not upright or not stopped):
            self._fail('最终落台、竖直稳定或抬升验证未通过')
            return
        self.state = 'COMPLETED'
        self.finished = True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diameter-mm', type=float, default=10.)
    parser.add_argument('--mass-g', type=float, default=50.)
    args = parser.parse_args()
    bench = GraspBench(GraspConfig(object_diameter=args.diameter_mm / 1000., object_mass=args.mass_g / 1000.))
    viewer_api = cast(ViewerAPI, import_module('mujoco.viewer'))
    actions: list[int] = []
    running = False
    def on_key(key: int) -> None:
        # 查看器线程仅排队按键，主线程修改物理状态。
        actions.append(key)
    print('空格键：开始或暂停；R：重置；关闭窗口退出。仅为接触验证台，未接入机械臂。')
    last_state = ''
    with viewer_api.launch_passive(bench.model, bench.data, key_callback=on_key) as viewer:
        with viewer.lock():
            viewer.cam.lookat[:] = [0., .4, .30]
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 135.
            viewer.cam.elevation = -15.
        while viewer.is_running():
            started = time.monotonic()
            while actions:
                key = actions.pop(0)
                if key == 32:
                    running = not running
                elif key in (82, 114):
                    bench.reset()
                    running = False
            with viewer.lock():
                if running and not bench.finished:
                    for _ in range(max(1, round(.01 / bench.config.timestep))):
                        bench.step()
            if bench.state != last_state:
                print(bench.state, bench.failure)
                last_state = bench.state
            viewer.sync()
            time.sleep(max(0., .01 - (time.monotonic() - started)))


if __name__ == '__main__':
    main()