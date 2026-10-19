"""加载 Archer-Y6 单臂模型到 MuJoCo，不访问实机."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import hashlib
from math import isfinite
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from typing import Protocol, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.intp]


# 用明确接口描述扩展模块的动态导出，不修改实际 MuJoCo 对象。
class ModelInterface(Protocol):
    nq: int
    nv: int
    nu: int
    jnt_qposadr: NDArray[np.int32]
    jnt_dofadr: NDArray[np.int32]


class DataInterface(Protocol):
    time: float
    qpos: FloatArray
    qvel: FloatArray
    ctrl: FloatArray
    qfrc_applied: FloatArray
    qfrc_bias: FloatArray
    qfrc_actuator: FloatArray
    site_xpos: FloatArray
    site_xmat: FloatArray


class ModelFactory(Protocol):
    def from_xml_path(self, filename: str) -> ModelInterface: ...


class DataFactory(Protocol):
    def __call__(self, model: ModelInterface) -> DataInterface: ...


class ObjectKinds(Protocol):
    mjOBJ_JOINT: object
    mjOBJ_ACTUATOR: object
    mjOBJ_SITE: object


class MuJoCoInterface(Protocol):
    MjModel: ModelFactory
    MjData: DataFactory
    mjtObj: ObjectKinds

    def mj_saveLastXML(self, filename: str, model: ModelInterface) -> None: ...

    def mj_name2id(self, model: ModelInterface, kind: object, name: str) -> int: ...

    def mj_forward(self, model: ModelInterface, data: DataInterface) -> None: ...

    def mj_step(self, model: ModelInterface, data: DataInterface) -> None: ...

    def mj_resetData(self, model: ModelInterface, data: DataInterface) -> None: ...


mujoco = cast(MuJoCoInterface, import_module("mujoco"))


@dataclass
class LoadedArm:
    model: ModelInterface
    data: DataInterface

    joint_names: tuple[str, ...]
    qpos_indices: IndexArray
    dof_indices: IndexArray
    actuator_indices: IndexArray
    joint_limits: FloatArray
    tcp_site_id: int


def _required_element(parent: ET.Element, tag: str) -> ET.Element:
    """读取必要的 XML 元素，缺少时明确报错."""
    element = parent.find(tag)

    if element is None:
        raise ValueError(f"Missing XML element: {tag}")

    return element


def _package_share(name: str) -> Path:
    """获取已安装 ROS 2 包的资源目录."""
    from ament_index_python.packages import get_package_share_directory

    return Path(get_package_share_directory(name))


def _resolve_mesh(filename: str, urdf_directory: Path) -> Path:
    """将网格资源路径转换为实际文件路径."""
    if filename.startswith("package://"):
        package, relative = filename[len("package://"):].split("/", 1)
        path = _package_share(package) / relative

    elif filename.startswith("file://"):
        path = Path(filename[len("file://"):])

    else:
        path = Path(filename)

        if not path.is_absolute():
            path = urdf_directory / path

    path = path.resolve()

    if not path.is_file():
        raise FileNotFoundError(f"Mesh not found: {path}")

    return path


def load_single_arm(
    side: str = "left",
    urdf_path: str | Path | None = None,
    tcp_xyz: ArrayLike = (0.0, 0.0, 0.147),
    mount_xyz: ArrayLike = (0.0, 0.0, 0.0),
    mount_rpy: ArrayLike = (0.0, 0.0, 0.0),
    timestep: float = 0.002,
    position_kp: float = 80.0,
    position_kv: float = 8.0,
    enable_contacts: bool = True,
) -> LoadedArm:
    """
    加载单臂，返回 MuJoCo 模型、状态和关节映射.

    side:
        left 或 right。

    urdf_path:
        不指定时，展开 archer_description 中的 single_arm.xacro。
        指定时，直接加载已经展开的 URDF。

    tcp_xyz:
        TCP 在夹爪模型坐标系中的位置。
        默认值为仿真候选值，尚未实机标定。

    mount_xyz / mount_rpy:
        夹爪模型相对法兰参考坐标的安装变换。

    timestep:
        物理仿真步长，单位：秒。

    position_kp / position_kv:
        仿真位置执行器增益，不作为实机控制参数。

    enable_contacts:
        是否启用碰撞接触。
        STL 碰撞使用 MuJoCo 凸包，不能代表精确的夹持接触。

    注意：
        指定 urdf_path 后，使用该文件中的安装和 TCP 定义，
        不再通过上述参数修改它们。
    """
    if side not in ("left", "right"):
        raise ValueError("side must be left or right")

    values = (timestep, position_kp, position_kv)

    if not all(isfinite(v) and v > 0 for v in values):
        raise ValueError("Timestep and position gains must be finite and positive")

    vectors: dict[str, FloatArray] = {}
    for name, values in (
        ("tcp_xyz", tcp_xyz),
        ("mount_xyz", mount_xyz),
        ("mount_rpy", mount_rpy),
    ):
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain three finite values")
        vectors[name] = vector

    prefix = side + "_"

    joint_names = tuple(f"{prefix}joint_{i}" for i in range(1, 7))

    with tempfile.TemporaryDirectory(prefix="archer_mujoco_") as directory:

        staging = Path(directory)

        # ---------- 读取或展开 URDF ----------

        if urdf_path is None:
            source = _package_share("archer_description") / "urdf" / "single_arm.xacro"

            command = [
                "xacro",
                str(source),
                f"side:={side}",
                "show_tcp:=false",
                "tcp_xyz:=" + " ".join(str(v) for v in vectors["tcp_xyz"]),
                "gripper_mount_xyz:=" + " ".join(str(v) for v in vectors["mount_xyz"]),
                "gripper_mount_rpy:=" + " ".join(str(v) for v in vectors["mount_rpy"]),
            ]

            try:
                result = subprocess.run(
                    command,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            except FileNotFoundError as exc:
                raise RuntimeError(
                    "xacro was not found. " "Source /opt/ros/humble/setup.bash first."
                ) from exc

            except subprocess.CalledProcessError as exc:
                raise RuntimeError(f"Xacro failed:\n{exc.stderr}") from exc

            root = ET.fromstring(result.stdout)
            urdf_directory = source.parent

        else:
            source = Path(urdf_path).expanduser().resolve()
            root = ET.parse(source).getroot()
            urdf_directory = source.parent

        # 移除显示球，保留 TCP 坐标系。
        # 后面使用 MuJoCo site 显示 TCP。
        tcp_link = root.find(f"link[@name='{prefix}tcp']")

        if tcp_link is None:
            raise ValueError(f"Missing TCP link: {prefix}tcp")

        for visual in list(tcp_link.findall("visual")):
            tcp_link.remove(visual)

        # ---------- 读取六个关节的位置限制 ----------

        joint_limits: list[tuple[float, float]] = []

        for name in joint_names:
            joint = root.find(f"joint[@name='{name}']")

            if joint is None or joint.get("type") != "revolute":
                raise ValueError(f"Missing revolute joint: {name}")

            limit = joint.find("limit")

            if limit is None:
                raise ValueError(f"Missing joint limits: {name}")

            bounds = (
                float(limit.attrib["lower"]),
                float(limit.attrib["upper"]),
            )

            if not all(isfinite(value) for value in bounds) or bounds[0] >= bounds[1]:
                raise ValueError(f"Invalid joint limits: {name}")

            joint_limits.append(bounds)

        # ---------- 处理网格资源 ----------

        # 只在临时目录生成唯一文件名，不修改原始 STL。
        # 避免 MuJoCo 内部资源名称因同名文件发生冲突。
        for mesh in root.findall(".//mesh"):
            path = _resolve_mesh(
                mesh.attrib["filename"],
                urdf_directory,
            )

            digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]

            target = staging / f"mesh_{digest}{path.suffix.lower()}"

            if not target.exists():
                shutil.copyfile(path, target)

            mesh.set("filename", str(target))

        # ---------- 设置 URDF 导入选项 ----------

        extension = root.find("mujoco")

        if extension is None:
            extension = ET.SubElement(root, "mujoco")

        compiler = extension.find("compiler")

        if compiler is None:
            compiler = ET.SubElement(
                extension,
                "compiler",
            )

        compiler.set("meshdir", str(staging))
        compiler.set("discardvisual", "false")

        # 保留固定链接，避免 TCP 坐标系被融合掉。
        compiler.set("fusestatic", "false")

        prepared = staging / "prepared.urdf"

        ET.ElementTree(root).write(
            prepared,
            encoding="utf-8",
            xml_declaration=True,
        )

        imported = mujoco.MjModel.from_xml_path(str(prepared))

        # ---------- 转为 MJCF 并加入仿真配置 ----------

        converted = staging / "converted.xml"

        mujoco.mj_saveLastXML(
            str(converted),
            imported,
        )

        mjcf = ET.parse(converted).getroot()

        _required_element(
            mjcf,
            "compiler",
        ).set("meshdir", str(staging))

        option = mjcf.find("option")

        if option is None:
            option = ET.SubElement(mjcf, "option")

        option.set("timestep", str(timestep))
        option.set("gravity", "0 0 -9.81")
        option.set("integrator", "implicitfast")

        # ---------- 添加灯光和地面 ----------

        world = _required_element(
            mjcf,
            "worldbody",
        )

        ET.SubElement(
            world,
            "light",
            {
                "name": "scene_light",
                "pos": "1 -1 2",
                "dir": "-1 1 -2",
            },
        )

        ET.SubElement(
            world,
            "geom",
            {
                "name": "floor",
                "type": "plane",
                "pos": "0 0 -0.005",
                "size": "1 1 0.1",
                "rgba": "0.25 0.28 0.3 1",
            },
        )

        if not enable_contacts:
            for geom in mjcf.findall(".//geom"):
                geom.set("contype", "0")
                geom.set("conaffinity", "0")

        # ---------- 添加 TCP site ----------

        tcp_body = mjcf.find(f".//body[@name='{prefix}tcp']")

        if tcp_body is None:
            raise ValueError("TCP frame was lost during URDF conversion")

        ET.SubElement(
            tcp_body,
            "site",
            {
                "name": prefix + "tcp_site",
                "pos": "0 0 0",
                "size": "0.004",
                "rgba": "1 0 0 1",
            },
        )

        # ---------- 添加六个位置执行器 ----------

        actuators = mjcf.find("actuator")

        if actuators is None:
            actuators = ET.SubElement(
                mjcf,
                "actuator",
            )

        for name, bounds in zip(
            joint_names,
            joint_limits,
        ):
            ET.SubElement(
                actuators,
                "position",
                {
                    "name": name + "_position",
                    "joint": name,
                    "kp": str(position_kp),
                    "kv": str(position_kv),
                    "ctrllimited": "true",
                    "ctrlrange": (f"{bounds[0]} {bounds[1]}"),
                },
            )

        final_xml = staging / "scene.xml"

        ET.ElementTree(mjcf).write(
            final_xml,
            encoding="utf-8",
            xml_declaration=True,
        )

        model = mujoco.MjModel.from_xml_path(str(final_xml))

    # 模型编译后，网格数据已在内存中，
    # 不再依赖上面的临时文件。

    # ---------- 建立名称与索引映射 ----------

    joint_ids = np.array(
        [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_JOINT,
                name,
            )
            for name in joint_names
        ],
        dtype=np.intp,
    )

    actuator_ids = np.array(
        [
            mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                name + "_position",
            )
            for name in joint_names
        ],
        dtype=np.intp,
    )

    tcp_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        prefix + "tcp_site",
    )

    if np.any(joint_ids < 0) or np.any(actuator_ids < 0) or tcp_id < 0:
        raise ValueError("Incomplete joint, actuator or TCP mappings")

    if model.nq != 6 or model.nv != 6 or model.nu != 6:
        raise ValueError("Expected exactly six arm joints and six actuators")

    data = mujoco.MjData(model)

    qpos_indices = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.intp).copy()

    dof_indices = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.intp).copy()

    # 仿真初始姿态，不向实机发送。
    initial = np.array([0.0, -0.8, 1.6, -0.8, 0.0, 0.0], dtype=np.float64)

    bounds = np.array(joint_limits, dtype=np.float64)

    if np.any(initial < bounds[:, 0]) or np.any(initial > bounds[:, 1]):
        raise ValueError("Simulation initial pose is outside the model limits")

    data.qpos[qpos_indices] = initial
    data.ctrl[actuator_ids] = initial

    mujoco.mj_forward(model, data)

    return LoadedArm(
        model=model,
        data=data,
        joint_names=joint_names,
        qpos_indices=qpos_indices,
        dof_indices=dof_indices,
        actuator_indices=actuator_ids,
        joint_limits=bounds,
        tcp_site_id=tcp_id,
    )
