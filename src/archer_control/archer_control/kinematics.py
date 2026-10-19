"""
从URDF建立六轴正运动学、雅可比矩阵与阻尼最小二乘逆运动学模型.

位姿均相对于对应机械臂的base_link表达，不是world。 不执行运动、硬件连接、碰撞检查、力控制或动力学补偿。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np
from numpy.typing import ArrayLike

from .trajectory import FloatArray, joint_names, joint_vector


def _vector3(values: ArrayLike, label: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{label} must contain three finite numbers")
    return result


def _xml_vector(text: str, label: str) -> FloatArray:
    return _vector3([float(value) for value in text.split()], label)


def _skew(axis: FloatArray) -> FloatArray:
    x, y, z = axis
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _rotation(axis: FloatArray, angle: float) -> FloatArray:
    k = _skew(axis)
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


def _origin(element: ET.Element) -> FloatArray:
    pose = np.eye(4, dtype=np.float64)
    origin = element.find("origin")
    if origin is None:
        return pose
    pose[:3, 3] = _xml_vector(origin.attrib.get("xyz", "0 0 0"), "origin xyz")
    roll, pitch, yaw = _xml_vector(origin.attrib.get("rpy", "0 0 0"), "origin rpy")
    pose[:3, :3] = (
        _rotation(np.array([0.0, 0.0, 1.0]), float(yaw))
        @ _rotation(np.array([0.0, 1.0, 0.0]), float(pitch))
        @ _rotation(np.array([1.0, 0.0, 0.0]), float(roll))
    )
    return pose


def rotation_error(target: FloatArray, current: FloatArray) -> FloatArray:
    """SO(3) logarithm of target @ current.T, expressed in the base frame."""
    rotation = target @ current.T
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    theta = float(np.arccos(cosine))
    vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if theta < 1e-7:
        return vee / 2.0
    if np.pi - theta < 1e-5:
        # Near pi the usual theta/sin(theta) expression is ill-conditioned.
        _, eigenvectors = np.linalg.eigh((rotation + rotation.T + 2 * np.eye(3)) / 4.0)
        axis = eigenvectors[:, -1]
        if float(np.dot(axis, vee)) < 0:
            axis = -axis
        return theta * axis
    return theta / (2.0 * np.sin(theta)) * vee


def _pose(values: ArrayLike) -> FloatArray:
    pose = np.asarray(values, dtype=np.float64)
    if pose.shape != (4, 4) or not np.all(np.isfinite(pose)):
        raise ValueError("target_pose must be a finite 4x4 matrix")
    rotation = pose[:3, :3]
    if not np.allclose(pose[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0):
        raise ValueError("Invalid homogeneous transform bottom row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(
        np.linalg.det(rotation), 1.0, atol=1e-6
    ):
        raise ValueError("Target rotation must be orthonormal with determinant +1")
    return pose.copy()


@dataclass(frozen=True)
class JointInfo:
    name: str
    parent: str
    child: str
    kind: str
    origin: FloatArray
    axis: FloatArray
    lower: float
    upper: float


@dataclass(frozen=True)
class IKResult:
    success: bool
    joints: FloatArray
    position_error: float
    orientation_error: float
    iterations: int
    message: str


class ArmKinematics:
    def __init__(self, root: ET.Element, side: str):
        self.names = joint_names(side)
        self.base_frame = side + "_base_link"
        self.tcp_frame = side + "_tcp"
        links = {element.attrib["name"] for element in root.findall("link")}
        if self.base_frame not in links or self.tcp_frame not in links:
            raise ValueError(f"Model needs {self.base_frame} and {self.tcp_frame}")
        by_child: dict[str, ET.Element] = {}
        for element in root.findall("joint"):
            child = element.find("child")
            if child is None:
                raise ValueError("Joint is missing its child link")
            name = child.attrib["link"]
            if name in by_child:
                raise ValueError(f"Multiple parents for link {name}")
            by_child[name] = element
        reversed_path: list[ET.Element] = []
        current = self.tcp_frame
        visited: set[str] = set()
        while current != self.base_frame:
            if current in visited or current not in by_child:
                raise ValueError("No valid chain from base to TCP")
            visited.add(current)
            element = by_child[current]
            reversed_path.append(element)
            parent = element.find("parent")
            if parent is None:
                raise ValueError("Joint is missing its parent link")
            current = parent.attrib["link"]
        self.chain: list[JointInfo] = []
        moving_names: list[str] = []
        bounds: list[tuple[float, float]] = []
        for element in reversed(reversed_path):
            kind = element.attrib["type"]
            name = element.attrib["name"]
            parent = element.find("parent")
            child = element.find("child")
            if parent is None or child is None:
                raise ValueError("Joint is missing parent/child")
            axis = np.zeros(3, dtype=np.float64)
            lower, upper = 0.0, 0.0
            if kind == "revolute":
                axis_element = element.find("axis")
                if axis_element is None:
                    raise ValueError(f"Missing axis for {name}")
                axis = _xml_vector(axis_element.attrib["xyz"], "joint axis")
                norm = float(np.linalg.norm(axis))
                if norm < 1e-12:
                    raise ValueError(f"Zero axis for {name}")
                axis = axis / norm
                limit = element.find("limit")
                if limit is None:
                    raise ValueError(f"Missing limits for {name}")
                lower, upper = float(limit.attrib["lower"]), float(limit.attrib["upper"])
                if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                    raise ValueError(f"Invalid limits for {name}")
                moving_names.append(name)
                bounds.append((lower, upper))
            elif kind != "fixed":
                raise ValueError(f"Unsupported joint type: {kind}")
            self.chain.append(
                JointInfo(
                    name,
                    parent.attrib["link"],
                    child.attrib["link"],
                    kind,
                    _origin(element),
                    axis,
                    lower,
                    upper,
                )
            )
        if tuple(moving_names) != self.names:
            raise ValueError("Expected the six Archer joints in order from base to TCP")
        self.limits = np.asarray(bounds, dtype=np.float64)

    @classmethod
    def from_urdf(cls, path: str | Path, side: str = "left") -> ArmKinematics:
        return cls(ET.parse(Path(path).expanduser()).getroot(), side)

    @classmethod
    def from_description(
        cls,
        side: str = "left",
        tcp_xyz: ArrayLike = (0.0, 0.0, 0.147),
        mount_xyz: ArrayLike = (0.0, 0.0, 0.0),
        mount_rpy: ArrayLike = (0.0, 0.0, 0.0),
    ) -> ArmKinematics:
        """Expand the installed project Xacro; requires a sourced ROS workspace."""
        from ament_index_python.packages import get_package_share_directory

        joint_names(side)
        path = Path(get_package_share_directory("archer_description")) / "urdf/single_arm.xacro"
        command = ["xacro", str(path), f"side:={side}", "show_tcp:=false"]
        for name, values in (
            ("tcp_xyz", tcp_xyz),
            ("gripper_mount_xyz", mount_xyz),
            ("gripper_mount_rpy", mount_rpy),
        ):
            vector = _vector3(values, name)
            command.append(name + ":=" + " ".join(str(float(value)) for value in vector))
        try:
            result = subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=30
            )
        except FileNotFoundError as exc:
            raise RuntimeError("xacro not found; source ROS Humble first") from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Xacro failed: {exc.stderr}") from exc
        return cls(ET.fromstring(result.stdout), side)

    def _checked_joints(self, values: ArrayLike) -> FloatArray:
        q = joint_vector(values)
        if np.any(q < self.limits[:, 0]) or np.any(q > self.limits[:, 1]):
            raise ValueError("Joint vector is outside this URDF model limits")
        return q

    def forward_with_jacobian(self, values: ArrayLike) -> tuple[FloatArray, FloatArray]:
        q = self._checked_joints(values)
        transform = np.eye(4, dtype=np.float64)
        axes: list[FloatArray] = []
        origins: list[FloatArray] = []
        index = 0
        for joint in self.chain:
            transform = transform @ joint.origin
            if joint.kind == "revolute":
                origins.append(transform[:3, 3].copy())
                axes.append(transform[:3, :3] @ joint.axis)
                motion = np.eye(4, dtype=np.float64)
                motion[:3, :3] = _rotation(joint.axis, float(q[index]))
                transform = transform @ motion
                index += 1
        jacobian = np.zeros((6, 6), dtype=np.float64)
        for index, (axis, origin) in enumerate(zip(axes, origins)):
            jacobian[:3, index] = np.cross(axis, transform[:3, 3] - origin)
            jacobian[3:, index] = axis
        return transform, jacobian

    def forward(self, values: ArrayLike) -> FloatArray:
        return self.forward_with_jacobian(values)[0]

    def solve_ik(
        self,
        target_pose: ArrayLike,
        seed: ArrayLike,
        max_iterations: int = 200,
        position_tolerance: float = 1e-4,
        orientation_tolerance: float = 1e-3,
        orientation_weight: float = 0.2,
        damping: float = 1e-3,
        maximum_step: float = 0.1,
    ) -> IKResult:
        """
        Numerical 6D IK with joint limits and backtracking line search.

        Local solver: a reachable target can still fail from a poor seed.
        Never use failed joints as a motion target. No collision guarantee.
        """
        target = _pose(target_pose)
        q = self._checked_joints(seed)
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        parameters = (
            position_tolerance,
            orientation_tolerance,
            orientation_weight,
            damping,
            maximum_step,
        )
        if not all(np.isfinite(value) and value > 0 for value in parameters):
            raise ValueError("IK tolerances, weight, damping and step must be positive and finite")
        reason = "Maximum iterations reached"
        iteration = 0
        for iteration in range(max_iterations + 1):
            pose, jacobian = self.forward_with_jacobian(q)
            position_error = target[:3, 3] - pose[:3, 3]
            angle_error = rotation_error(target[:3, :3], pose[:3, :3])
            p_norm = float(np.linalg.norm(position_error))
            r_norm = float(np.linalg.norm(angle_error))
            if p_norm <= position_tolerance and r_norm <= orientation_tolerance:
                return IKResult(True, q.copy(), p_norm, r_norm, iteration, "Converged")
            if iteration == max_iterations:
                break
            error = np.concatenate((position_error, orientation_weight * angle_error))
            weighted = jacobian.copy()
            weighted[3:] *= orientation_weight
            delta = weighted.T @ np.linalg.solve(
                weighted @ weighted.T + damping**2 * np.eye(6), error
            )
            norm = float(np.linalg.norm(delta))
            if not np.isfinite(norm):
                reason = "Non-finite numerical step"
                break
            if norm > maximum_step:
                delta *= maximum_step / norm
            cost = float(np.dot(error, error))
            improved = False
            for factor in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                candidate = np.clip(q + factor * delta, self.limits[:, 0], self.limits[:, 1])
                next_pose = self.forward(candidate)
                next_error = np.concatenate(
                    (
                        target[:3, 3] - next_pose[:3, 3],
                        orientation_weight * rotation_error(target[:3, :3], next_pose[:3, :3]),
                    )
                )
                if float(np.dot(next_error, next_error)) < cost - 1e-14:
                    q = candidate
                    improved = True
                    break
            if not improved:
                reason = "No improving step; try a different seed or check reachability"
                break
        final = self.forward(q)
        return IKResult(
            False,
            q.copy(),
            float(np.linalg.norm(target[:3, 3] - final[:3, 3])),
            float(np.linalg.norm(rotation_error(target[:3, :3], final[:3, :3]))),
            iteration,
            reason,
        )
