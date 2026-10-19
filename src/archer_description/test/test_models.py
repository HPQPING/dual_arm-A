"""检查Xacro关节、工具坐标链和安装网格资源，不启动仿真或硬件."""

import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
import pytest


ROOT = Path(__file__).resolve().parents[1]


def expand(name: str, *arguments: str) -> ET.Element:
    """展开源文件，包含的包资源按已安装工程解析."""
    result = subprocess.run(
        ["xacro", str(ROOT / "urdf" / name), *arguments],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )
    return ET.fromstring(result.stdout)


@pytest.mark.parametrize("side", ["left", "right"])
def test_single_arm_has_six_finite_revolute_joints(side: str) -> None:
    """单臂六关节均有有限有序限位和单位旋转轴，名称与反馈一致."""
    robot = expand("single_arm.xacro", "side:=" + side)
    joints = [j for j in robot.findall("joint") if j.get("type") == "revolute"]
    assert {j.get("name") for j in joints} == {f"{side}_joint_{i}" for i in range(1, 7)}
    for joint in joints:
        limit, axis = joint.find("limit"), joint.find("axis")
        assert limit is not None and axis is not None
        low, high = float(limit.attrib["lower"]), float(limit.attrib["upper"])
        assert math.isfinite(low) and math.isfinite(high) and low < high
        vector = [float(x) for x in axis.attrib["xyz"].split()]
        assert len(vector) == 3 and math.isclose(sum(v * v for v in vector), 1.0, abs_tol=1e-9)


def test_dual_arm_tree_and_tools() -> None:
    """双臂拓扑只有world根，法兰8mm与候选TCP偏移各出现一次."""
    robot = expand("dual_arm.xacro")
    links = [x.attrib["name"] for x in robot.findall("link")]
    assert len(links) == len(set(links))
    children = {}
    for joint in robot.findall("joint"):
        child, parent = joint.find("child"), joint.find("parent")
        assert child is not None and parent is not None
        name = child.attrib["link"]
        assert name not in children and name in links and parent.attrib["link"] in links
        children[name] = joint
    assert set(links) - set(children) == {"world"}
    for side in ("left", "right"):
        for name, xyz in (
            ("flange", (0.0, 0.0, 0.008)),
            ("gp80_base_link", (0.0, 0.0, 0.0)),
            ("tcp", (0.0, 0.0, 0.147021696)),
        ):
            joint = children[side + "_" + name]
            assert joint.get("type") == "fixed"
            origin = joint.find("origin")
            assert origin is not None
            actual = [float(x) for x in origin.attrib["xyz"].split()]
            assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(actual, xyz))


def test_all_mesh_references_are_installed() -> None:
    """真实模型网格引用必须在安装资源中存在，不能用简化替身冒充."""
    robot = expand("dual_arm.xacro")
    meshes = robot.findall(".//mesh")
    assert meshes
    for mesh in meshes:
        filename = mesh.attrib["filename"]
        assert filename.startswith("package://")
        package, relative = filename[len("package://"):].split("/", 1)
        assert (Path(get_package_share_directory(package)) / relative).is_file()
