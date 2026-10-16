"""Build one 12-joint MuJoCo scene from the existing single-arm URDFs.

No hardware access. GP80 remains a rigid mesh; TCP is the configured candidate.
Original meshes and URDFs are never modified. Assets are staged temporarily.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import TypeAlias

import numpy as np

from .model_loader import (
    DataInterface, FloatArray, IndexArray, LoadedArm, ModelInterface, mujoco,
)

VectorInput: TypeAlias = Sequence[float] | FloatArray


def _package_share(name: str) -> Path:
    from ament_index_python.packages import get_package_share_directory
    return Path(get_package_share_directory(name))


def _resolve_mesh(filename: str, urdf_directory: Path) -> Path:
    if filename.startswith('package://'):
        package, separator, relative = filename[len('package://'):].partition('/')
        if not separator or not package or not relative:
            raise ValueError('Invalid package mesh path: ' + filename)
        path = _package_share(package) / relative
    elif filename.startswith('file://'):
        path = Path(filename[len('file://'):])
    else:
        path = Path(filename)
        if not path.is_absolute():
            path = urdf_directory / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError('Mesh not found: ' + str(path))
    return path


@dataclass
class LoadedDualArm:
    model: ModelInterface
    data: DataInterface
    arms: dict[str, LoadedArm]


def vector3(values: VectorInput, name: str) -> FloatArray:
    result: FloatArray = np.asarray(values, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f'{name} needs three finite numbers')
    return result


def quaternion_from_rpy(values: VectorInput) -> FloatArray:
    roll, pitch, yaw = vector3(values, 'base_rpy') * 0.5
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                     cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy], dtype=np.float64)


def _text(values: FloatArray) -> str:
    return ' '.join(str(float(v)) for v in values.flat)


def _convert_arm(
    side: str, directory: Path, urdf_path: str | Path | None,
    tcp_xyz: FloatArray, mount_xyz: FloatArray, mount_rpy: FloatArray,
) -> tuple[ET.Element, FloatArray]:
    if urdf_path is None:
        source = _package_share('archer_description') / 'urdf/single_arm.xacro'
        command = ['xacro', str(source), f'side:={side}', 'show_tcp:=false',
                   'tcp_xyz:=' + _text(tcp_xyz),
                   'gripper_mount_xyz:=' + _text(mount_xyz),
                   'gripper_mount_rpy:=' + _text(mount_rpy)]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
        except FileNotFoundError as exc:
            raise RuntimeError('xacro unavailable; source ROS Humble first') from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError('Xacro failed: ' + exc.stderr) from exc
        root = ET.fromstring(result.stdout)
    else:
        source = Path(urdf_path).expanduser().resolve()
        root = ET.parse(source).getroot()
    tcp = root.find(f"link[@name='{side}_tcp']")
    if tcp is None:
        raise ValueError(f'Missing {side}_tcp in URDF')
    for visual in list(tcp.findall('visual')):
        tcp.remove(visual)
    limits: list[tuple[float, float]] = []
    for index in range(1, 7):
        joint = root.find(f"joint[@name='{side}_joint_{index}']")
        if joint is None or joint.get('type') != 'revolute':
            raise ValueError(f'Missing {side} revolute joint {index}')
        limit = joint.find('limit')
        if limit is None:
            raise ValueError('Missing joint limits')
        low, high = float(limit.attrib['lower']), float(limit.attrib['upper'])
        if not np.isfinite(low) or not np.isfinite(high) or low >= high:
            raise ValueError('Invalid joint limits')
        limits.append((low, high))
    for mesh in root.findall('.//mesh'):
        path = _resolve_mesh(mesh.attrib['filename'], source.parent)
        digest = hashlib.sha256(str(path).encode('utf-8')).hexdigest()[:16]
        target = directory / f'{side}_mesh_{digest}{path.suffix.lower()}'
        if not target.exists():
            shutil.copyfile(path, target)
        mesh.set('filename', str(target))
    extension = root.find('mujoco')
    if extension is None:
        extension = ET.SubElement(root, 'mujoco')
    compiler = extension.find('compiler')
    if compiler is None:
        compiler = ET.SubElement(extension, 'compiler')
    compiler.set('meshdir', str(directory))
    compiler.set('discardvisual', 'false')
    compiler.set('fusestatic', 'false')
    prepared = directory / f'{side}.urdf'
    ET.ElementTree(root).write(prepared, encoding='utf-8', xml_declaration=True)
    imported = mujoco.MjModel.from_xml_path(str(prepared))
    converted = directory / f'{side}.xml'
    mujoco.mj_saveLastXML(str(converted), imported)
    mjcf = ET.parse(converted).getroot()
    # The supported source is the project's six-joint URDF, not arbitrary MJCF.
    if any(child.tag not in ('compiler', 'option', 'asset', 'worldbody') for child in mjcf):
        raise ValueError('Unexpected URDF conversion sections; inspect source before merging')
    renamed: dict[str, str] = {}
    for element in mjcf.iter():
        name = element.get('name')
        if name is not None and name:
            renamed[name] = name if name.startswith(side + '_') else side + '_model_' + name
    references = ('name', 'mesh', 'material', 'texture', 'joint', 'body', 'site')
    for element in mjcf.iter():
        for attribute in references:
            value = element.get(attribute)
            if value is not None and value in renamed:
                element.set(attribute, renamed[value])
    return mjcf, np.asarray(limits, dtype=np.float64)


def load_dual_arm(
    left_urdf: str | Path | None = None, right_urdf: str | Path | None = None,
    left_base_xyz: VectorInput = (0., 0.42, 0.), right_base_xyz: VectorInput = (0., 0., 0.),
    left_base_rpy: VectorInput = (0., 0., 0.), right_base_rpy: VectorInput = (0., 0., 0.),
    tcp_xyz: VectorInput = (0., 0., 0.147), mount_xyz: VectorInput = (0., 0., 0.),
    mount_rpy: VectorInput = (0., 0., 0.), timestep: float = 0.002,
    position_kp: float = 80., position_kv: float = 8., enable_contacts: bool = False,
) -> LoadedDualArm:
    if not all(np.isfinite(v) and v > 0 for v in (timestep, position_kp, position_kv)):
        raise ValueError('Physics step and gains must be positive and finite')
    tcp = vector3(tcp_xyz, 'tcp_xyz')
    mount = vector3(mount_xyz, 'mount_xyz')
    mount_rotation = vector3(mount_rpy, 'mount_rpy')
    translations = {'left': vector3(left_base_xyz, 'left_base_xyz'),
                    'right': vector3(right_base_xyz, 'right_base_xyz')}
    rotations = {'left': quaternion_from_rpy(left_base_rpy), 'right': quaternion_from_rpy(right_base_rpy)}
    bounds: dict[str, FloatArray] = {}
    with tempfile.TemporaryDirectory(prefix='archer_dual_') as directory:
        staging = Path(directory)
        scene = ET.Element('mujoco', {'model': 'archer_dual_y6_gp80'})
        ET.SubElement(scene, 'compiler', {'angle': 'radian', 'meshdir': str(staging), 'fusestatic': 'false'})
        ET.SubElement(scene, 'option', {'timestep': str(timestep), 'gravity': '0 0 -9.81', 'integrator': 'implicitfast'})
        assets = ET.SubElement(scene, 'asset')
        world = ET.SubElement(scene, 'worldbody')
        ET.SubElement(world, 'light', {'name': 'scene_light', 'pos': '1 -1 2', 'dir': '-1 1 -2'})
        ET.SubElement(world, 'geom', {'name': 'floor', 'type': 'plane', 'pos': '0 0 -0.005',
                                    'size': '1.5 1.5 0.1', 'rgba': '0.25 0.28 0.3 1'})
        actuators = ET.SubElement(scene, 'actuator')
        for side, source in (('left', left_urdf), ('right', right_urdf)):
            imported, bounds[side] = _convert_arm(side, staging, source, tcp, mount, mount_rotation)
            imported_assets = imported.find('asset')
            if imported_assets is not None:
                assets.extend(list(imported_assets))
            imported_world = imported.find('worldbody')
            if imported_world is None:
                raise ValueError('URDF conversion lost worldbody')
            base = ET.SubElement(world, 'body', {'name': side + '_scene_base',
                                                'pos': _text(translations[side]), 'quat': _text(rotations[side])})
            base.extend(list(imported_world))
            tcp_body = base.find(f".//body[@name='{side}_tcp']")
            if tcp_body is None:
                raise ValueError(f'Missing {side} TCP body after conversion')
            ET.SubElement(tcp_body, 'site', {'name': side + '_tcp_site', 'size': '0.004', 'rgba': '1 0 0 1'})
            for index, (low, high) in enumerate(bounds[side], start=1):
                name = f'{side}_joint_{index}'
                ET.SubElement(actuators, 'position', {'name': name + '_position', 'joint': name,
                    'kp': str(position_kp), 'kv': str(position_kv), 'ctrllimited': 'true', 'ctrlrange': f'{low} {high}'})
        if not enable_contacts:
            for geom in scene.findall('.//geom'):
                geom.set('contype', '0')
                geom.set('conaffinity', '0')
        filename = staging / 'dual_scene.xml'
        ET.ElementTree(scene).write(filename, encoding='utf-8', xml_declaration=True)
        model = mujoco.MjModel.from_xml_path(str(filename))
    if (model.nq, model.nv, model.nu) != (12, 12, 12):
        raise ValueError('Expected twelve joints and twelve actuators in the shared scene')
    data = mujoco.MjData(model)
    arms: dict[str, LoadedArm] = {}
    initial = np.array([0., -0.8, 1.6, -0.8, 0., 0.], dtype=np.float64)
    for side in ('left', 'right'):
        names = tuple(f'{side}_joint_{index}' for index in range(1, 7))
        joints: IndexArray = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names], dtype=np.intp)
        controls: IndexArray = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name + '_position') for name in names], dtype=np.intp)
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, side + '_tcp_site')
        if np.any(joints < 0) or np.any(controls < 0) or site < 0:
            raise ValueError('Incomplete dual-arm joint mappings')
        if np.any(initial < bounds[side][:, 0]) or np.any(initial > bounds[side][:, 1]):
            raise ValueError('Initial pose exceeds URDF limits')
        qpos: IndexArray = np.asarray(model.jnt_qposadr[joints], dtype=np.intp).copy()
        dofs: IndexArray = np.asarray(model.jnt_dofadr[joints], dtype=np.intp).copy()
        arms[side] = LoadedArm(model, data, names, qpos, dofs, controls, bounds[side], site)
        data.qpos[qpos] = initial
        data.ctrl[controls] = initial
    mujoco.mj_forward(model, data)
    return LoadedDualArm(model, data, arms)