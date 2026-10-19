"""公共工具链校验与离散跨臂间隙检查，不执行任务或发送运动指令."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from .circle_planner import MOUNT_XYZ, TCP_XYZ
from .kinematics import ArmKinematics

if TYPE_CHECKING:
    from archer_mujoco.dual_model_loader import LoadedDualArm


class CollisionModel(Protocol):
    geom_margin: NDArray[np.float64]
    geom_bodyid: NDArray[np.int32]


class Contact(Protocol):
    geom1: int
    geom2: int
    dist: float


class Contacts(Protocol):
    def __getitem__(self, index: int) -> Contact: ...


class CollisionData(Protocol):
    ncon: int
    contact: Contacts


class BodyKinds(Protocol):
    mjOBJ_BODY: object


class NameAPI(Protocol):
    mjtObj: BodyKinds

    def mj_id2name(self, model: object, kind: object, index: int) -> str | None: ...


def validate_tool_chain(model: ArmKinematics) -> None:
    """检查工程坐标约定，避免旧模型或重复偏移被悄悄使用；不代表实机标定."""
    prefix = model.base_frame.removesuffix("base_link")
    expected = (
        (prefix + "link_6", prefix + "flange", (0.0, 0.0, 0.008)),
        (prefix + "flange", prefix + "gp80_base_link", MOUNT_XYZ),
        (prefix + "gp80_base_link", prefix + "tcp", TCP_XYZ),
    )
    for parent, child, translation in expected:
        matches = [
            joint for joint in model.chain if joint.parent == parent and joint.child == child
        ]
        if len(matches) != 1 or matches[0].kind != "fixed":
            raise ValueError(f"模型缺少固定坐标关系：{parent} → {child}，请先更新 Xacro")
        joint = matches[0]
        if not np.allclose(joint.origin[:3, 3], translation, atol=1e-8, rtol=0.0):
            raise ValueError(
                f"{parent} → {child} 平移应为 {translation} m；"
                f"实际为 {joint.origin[:3, 3].tolist()}，请检查偏移是否遗漏或重复"
            )
        # 夹爪安装旋转可按模型调整；其余两个参考帧按当前约定保持同向。
        if child != prefix + "gp80_base_link" and not np.allclose(
            joint.origin[:3, :3], np.eye(3), atol=1e-8, rtol=0.0
        ):
            raise ValueError(f"{parent} → {child} 的旋转应为零，请检查模型坐标约定")


def cross_arm_failure(loaded: LoadedDualArm, clearance: float) -> str | None:
    """只检测两臂之间的采样间隙，不把相邻链接装配重叠当作跨臂碰撞."""
    geometry = cast(CollisionModel, loaded.model)
    data = cast(CollisionData, loaded.data)
    api = cast(NameAPI, import_module("mujoco"))
    for index in range(data.ncon):
        contact = data.contact[index]
        a = (
            api.mj_id2name(
                loaded.model, api.mjtObj.mjOBJ_BODY, int(geometry.geom_bodyid[contact.geom1])
            )
            or "world"
        )
        b = (
            api.mj_id2name(
                loaded.model, api.mjtObj.mjOBJ_BODY, int(geometry.geom_bodyid[contact.geom2])
            )
            or "world"
        )
        cross = (a.startswith("left_") and b.startswith("right_")) or (
            a.startswith("right_") and b.startswith("left_")
        )
        if cross and contact.dist < clearance:
            return f"{a} / {b}，间距 {contact.dist * 1000:.3f} mm"
    return None
