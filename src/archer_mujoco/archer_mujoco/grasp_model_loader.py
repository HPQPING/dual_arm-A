"""
构建双臂、四个简化指腹和自由圆柱，不修改原来的十二关节模型加载器.

沿用原有 URDF 转换工具。真实 GP80 整体网格默认用于外观，不参与夹取碰撞； 夹取使用局部 X 方向开合的简化平行指腹，不代表 GP80 实际连杆或开合轴标定。
本模型的动力学接触只覆盖圆柱、指腹和台面；整臂碰撞需另行预检。
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
import tempfile
from typing import Protocol, cast
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import NDArray

from .dual_model_loader import _convert_arm, quaternion_from_rpy
from .grasp_scene import ContactList, GraspConfig
from .model_loader import DataInterface, FloatArray, IndexArray, LoadedArm, ModelInterface


class GraspModel(ModelInterface, Protocol):
    body_mocapid: NDArray[np.int32]


class GraspData(DataInterface, Protocol):
    mocap_pos: FloatArray
    xpos: FloatArray
    xmat: FloatArray
    ncon: int
    contact: ContactList


class ModelFactory(Protocol):
    def from_xml_path(self, filename: str) -> GraspModel: ...


class DataFactory(Protocol):
    def __call__(self, model: GraspModel) -> GraspData: ...


class ObjectKinds(Protocol):
    mjOBJ_JOINT: object
    mjOBJ_ACTUATOR: object
    mjOBJ_SITE: object
    mjOBJ_BODY: object
    mjOBJ_GEOM: object


class Physics(Protocol):
    MjModel: ModelFactory
    MjData: DataFactory
    mjtObj: ObjectKinds

    def mj_name2id(self, model: GraspModel, kind: object, name: str) -> int: ...

    def mj_forward(self, model: GraspModel, data: GraspData) -> None: ...

    def mj_step(self, model: GraspModel, data: GraspData) -> None: ...

    def mj_resetData(self, model: GraspModel, data: GraspData) -> None: ...

    def mj_contactForce(
        self, model: GraspModel, data: GraspData, index: int, force: FloatArray
    ) -> None: ...


physics = cast(Physics, import_module("mujoco"))


def _text(values: tuple[float, ...] | FloatArray) -> str:
    return " ".join(format(float(value), ".12g") for value in values)


@dataclass
class LoadedGrasp:
    model: GraspModel
    data: GraspData
    arms: dict[str, LoadedArm]
    finger_controls: dict[str, IndexArray]
    finger_positions: dict[str, IndexArray]
    finger_geoms: dict[int, str]
    object_body: int
    object_geom: int
    table_geom: int
    object_qpos: int
    object_dof: int
    config: GraspConfig

    def set_arm_targets(
        self,
        left: FloatArray,
        right: FloatArray,
        left_velocity: FloatArray | None = None,
        right_velocity: FloatArray | None = None,
    ) -> None:
        pending: list[tuple[IndexArray, FloatArray]] = []
        for side, target, velocity in (
            ("left", left, left_velocity),
            ("right", right, right_velocity),
        ):
            arm = self.arms[side]
            values = np.asarray(target, dtype=np.float64)
            if (
                values.shape != (6,)
                or not np.all(np.isfinite(values))
                or np.any(values < arm.joint_limits[:, 0])
                or np.any(values > arm.joint_limits[:, 1])
            ):
                raise ValueError(side + " 关节目标不是限位内的六个有限数值")
            if velocity is not None:
                speed = np.asarray(velocity, dtype=np.float64)
                if speed.shape != (6,) or not np.all(np.isfinite(speed)):
                    raise ValueError(side + " 速度参考必须包含六个有限数值")
                # 位置驱动器的阻尼作用于实际速度；加入参考速度项得到完整的轨迹跟踪 PD。
                # 仅用于本仿真的 kp=200、kv=20，不能直接转为实机控制参数。
                values = np.clip(
                    values + 0.1 * speed, arm.joint_limits[:, 0], arm.joint_limits[:, 1]
                )
            pending.append((arm.actuator_indices, values.copy()))
        # 两侧参数全部合法后再写入，避免右侧报错时只更改了左侧目标。
        for indices, values in pending:
            self.data.ctrl[indices] = values

    def set_grippers(self, position: float) -> None:
        if isinstance(position, bool) or not np.isfinite(position) or not 0.0 <= position <= 5.65:
            raise ValueError("夹爪指令必须在 0～5.65 之间")
        for controls in self.finger_controls.values():
            self.data.ctrl[controls] = 0.04 * position / 5.65

    def step(self) -> None:
        # 仅补偿十二个机械臂自由度；不补偿圆柱，也不覆盖它的自由关节状态。
        self.data.qfrc_applied[:] = 0.0
        for arm in self.arms.values():
            self.data.qfrc_applied[arm.dof_indices] = self.data.qfrc_bias[arm.dof_indices]
        physics.mj_step(self.model, self.data)
        physics.mj_forward(self.model, self.data)

    def contact_forces(self) -> tuple[dict[str, float], float]:
        fingers = {name: 0.0 for name in self.finger_geoms.values()}
        table = 0.0
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            if self.object_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == self.object_geom else contact.geom1
            force = np.zeros(6, dtype=np.float64)
            physics.mj_contactForce(self.model, self.data, index, force)
            if other in self.finger_geoms:
                fingers[self.finger_geoms[other]] += max(0.0, float(force[0]))
            elif other == self.table_geom:
                table += max(0.0, float(force[0]))
        return fingers, table

    def object_bottom(self) -> float:
        cosine = min(1.0, abs(float(self.data.xmat[self.object_body, 8])))
        extent = (
            self.config.object_height / 2.0 * cosine
            + self.config.object_diameter / 2.0 * np.sqrt(max(0.0, 1.0 - cosine * cosine))
        )
        return float(self.data.xpos[self.object_body, 2] - extent)


def load_grasp_model(
    left_initial: FloatArray,
    right_initial: FloatArray,
    left_urdf: str | Path | None = None,
    right_urdf: str | Path | None = None,
    config: GraspConfig = GraspConfig(),
    show_original_gripper: bool = True,
) -> LoadedGrasp:
    """创建专用抓取场景；左右基座及工具链继续采用已经确定的工程约定."""
    config.validate()
    bounds: dict[str, FloatArray] = {}
    tcp = np.array([0.0, 0.0, 0.147021696], dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    with tempfile.TemporaryDirectory(prefix="archer_grasp_") as directory:
        staging = Path(directory)
        root = ET.Element("mujoco", model="archer_dual_cylinder_grasp")
        ET.SubElement(root, "compiler", angle="radian", fusestatic="false", meshdir=str(staging))
        ET.SubElement(
            root,
            "option",
            timestep=str(config.timestep),
            gravity="0 0 -9.81",
            integrator="implicitfast",
            iterations="100",
            cone="elliptic",
        )
        default = ET.SubElement(root, "default")
        ET.SubElement(
            default,
            "geom",
            friction=_text((config.friction, 0.005, 0.0001)),
            condim="4",
            solref=".004 1",
            solimp=".99 .999 .0001",
        )
        assets = ET.SubElement(root, "asset")
        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "light", pos="0 -.2 2", dir="0 .4 -1")
        ET.SubElement(
            world,
            "geom",
            name="grasp_table",
            type="box",
            pos=_text((0.0, 0.35, config.table_z - 0.02)),
            size=".65 .75 .02",
            rgba=".25 .3 .3 1",
            contype="1",
            conaffinity="2",
        )
        actuators = ET.SubElement(root, "actuator")
        for side, path, x in (("left", left_urdf, -0.21), ("right", right_urdf, 0.21)):
            imported, bounds[side] = _convert_arm(side, staging, path, tcp, zero, zero)
            imported_assets = imported.find("asset")
            if imported_assets is not None:
                assets.extend(list(imported_assets))
            imported_world = imported.find("worldbody")
            if imported_world is None:
                raise ValueError("机械臂转换结果缺少 worldbody")
            base = ET.SubElement(
                world,
                "body",
                name=side + "_scene_base",
                pos=_text((x, 0.0, 0.0)),
                quat=_text(quaternion_from_rpy((0.0, 0.0, np.pi / 2.0))),
            )
            base.extend(list(imported_world))
            # 原机械臂外形保留显示；本轮接触模型只验证物件与指腹、台面的作用。
            for geom in base.findall(".//geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")
            tool = base.find(f".//body[@name='{side}_gp80_base_link']")
            tip = base.find(f".//body[@name='{side}_tcp']")
            if tool is None or tip is None:
                raise ValueError("模型缺少独立 GP80 或 TCP；请使用项目已更新的 Xacro")
            for geom in tool.findall("geom"):
                geom.set("rgba", "1 1 1 1" if show_original_gripper else "0 0 0 0")
            ET.SubElement(
                tool,
                "geom",
                type="cylinder",
                pos="0 0 .04",
                size=".025 .04",
                rgba="0 0 0 0" if show_original_gripper else ".3 .3 .35 1",
                contype="0",
                conaffinity="0",
                mass="0",
            )
            ET.SubElement(tip, "site", name=side + "_tcp_site", size=".002", rgba="1 0 0 1")
            color = (
                "0 0 0 0"
                if show_original_gripper
                else ("1 .5 .1 1" if side == "left" else "0 .7 1 1")
            )
            for label, sign in (("a", -1.0), ("b", 1.0)):
                name = side + "_grasp_finger_" + label
                finger = ET.SubElement(tip, "body", name=name, pos=_text((sign * 0.044, 0.0, 0.0)))
                ET.SubElement(
                    finger,
                    "joint",
                    name=name,
                    type="slide",
                    axis=_text((-sign, 0.0, 0.0)),
                    range="0 .04",
                    damping="1",
                )
                ET.SubElement(
                    finger,
                    "geom",
                    name=name,
                    type="box",
                    size=".004 .015 .012",
                    mass=".02",
                    rgba=color,
                    contype="4",
                    conaffinity="2",
                )
                ET.SubElement(
                    actuators,
                    "position",
                    name=name,
                    joint=name,
                    kp="800",
                    ctrlrange="0 .04",
                    forcerange=_text((-config.finger_force, config.finger_force)),
                )
            for index, (low, high) in enumerate(bounds[side], start=1):
                name = f"{side}_joint_{index}"
                # 此增益仅用于接触仿真，不作为实机 SDK 的刚度或 MIT 参数。
                ET.SubElement(
                    actuators,
                    "position",
                    name=name + "_position",
                    joint=name,
                    kp="200",
                    kv="20",
                    ctrlrange=f"{low} {high}",
                )
        obj = ET.SubElement(
            world,
            "body",
            name="grasp_object",
            pos=_text((0.0, 0.4, config.center_z + config.initial_clearance)),
        )
        ET.SubElement(obj, "freejoint", name="grasp_object_free")
        ET.SubElement(
            obj,
            "geom",
            name="grasp_object_geom",
            type="cylinder",
            size=_text((config.object_diameter / 2.0, config.object_height / 2.0)),
            mass=str(config.object_mass),
            rgba=".2 .8 .3 1",
            contype="2",
            conaffinity="5",
        )
        filename = staging / "grasp_scene.xml"
        ET.ElementTree(root).write(filename, encoding="utf-8", xml_declaration=True)
        model = physics.MjModel.from_xml_path(str(filename))
    if (model.nq, model.nv, model.nu) != (23, 22, 16):
        raise ValueError("模型应包含十二个臂关节、四个指腹关节和一个自由物件")
    data = physics.MjData(model)
    kinds = physics.mjtObj

    def identifier(kind: object, name: str) -> int:
        index = physics.mj_name2id(model, kind, name)
        if index < 0:
            raise ValueError("模型缺少：" + name)
        return index

    arms: dict[str, LoadedArm] = {}
    controls: dict[str, IndexArray] = {}
    positions: dict[str, IndexArray] = {}
    geoms: dict[int, str] = {}
    for side, initial in (("left", left_initial), ("right", right_initial)):
        names = tuple(f"{side}_joint_{index}" for index in range(1, 7))
        joints = np.array([identifier(kinds.mjOBJ_JOINT, name) for name in names], dtype=np.intp)
        actions = np.array(
            [identifier(kinds.mjOBJ_ACTUATOR, name + "_position") for name in names], dtype=np.intp
        )
        arm = LoadedArm(
            model,
            data,
            names,
            np.asarray(model.jnt_qposadr[joints], dtype=np.intp),
            np.asarray(model.jnt_dofadr[joints], dtype=np.intp),
            actions,
            bounds[side],
            identifier(kinds.mjOBJ_SITE, side + "_tcp_site"),
        )
        values = np.asarray(initial, dtype=np.float64)
        if (
            values.shape != (6,)
            or not np.all(np.isfinite(values))
            or np.any(values < bounds[side][:, 0])
            or np.any(values > bounds[side][:, 1])
        ):
            raise ValueError(side + " 初始关节角超出模型限位")
        data.qpos[arm.qpos_indices] = values
        data.ctrl[arm.actuator_indices] = values
        arms[side] = arm
        finger_names = [side + "_grasp_finger_" + label for label in ("a", "b")]
        controls[side] = np.array(
            [identifier(kinds.mjOBJ_ACTUATOR, name) for name in finger_names], dtype=np.intp
        )
        indices = np.array(
            [identifier(kinds.mjOBJ_JOINT, name) for name in finger_names], dtype=np.intp
        )
        positions[side] = np.asarray(model.jnt_qposadr[indices], dtype=np.intp)
        for name in finger_names:
            geoms[identifier(kinds.mjOBJ_GEOM, name)] = name
    free = identifier(kinds.mjOBJ_JOINT, "grasp_object_free")
    loaded = LoadedGrasp(
        model,
        data,
        arms,
        controls,
        positions,
        geoms,
        identifier(kinds.mjOBJ_BODY, "grasp_object"),
        identifier(kinds.mjOBJ_GEOM, "grasp_object_geom"),
        identifier(kinds.mjOBJ_GEOM, "grasp_table"),
        int(model.jnt_qposadr[free]),
        int(model.jnt_dofadr[free]),
        config,
    )
    physics.mj_forward(model, data)
    return loaded
