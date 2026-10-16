"""在 MuJoCo 查看器中绘制参考半圆和实际 TCP 轨迹。

本模块文件名必须为 circle_visuals.py，与 circle_sim.py 的导入一致。
"""

from __future__ import annotations

from collections.abc import Sequence
from importlib import import_module
from typing import Protocol, cast

import numpy as np

from .trajectory import FloatArray


class VisualGeom(Protocol):
    pass


class Geoms(Protocol):
    def __getitem__(self, index: int) -> VisualGeom: ...


class VisualScene(Protocol):
    geoms: Geoms
    ngeom: int
    maxgeom: int


class GeomKinds(Protocol):
    mjGEOM_LINE: object
    mjGEOM_SPHERE: object


class DrawingAPI(Protocol):
    mjtGeom: GeomKinds
    def mjv_initGeom(self, geom: VisualGeom, kind: object, size: FloatArray,
                     pos: FloatArray, mat: FloatArray, rgba: FloatArray) -> None: ...
    def mjv_connector(self, geom: VisualGeom, kind: object, width: float,
                      start: FloatArray, end: FloatArray) -> None: ...


def _line(scene: VisualScene, start: FloatArray, end: FloatArray,
          color: FloatArray, api: DrawingAPI) -> None:
    if scene.ngeom >= scene.maxgeom or float(np.linalg.norm(end - start)) < 1e-8:
        return
    geom = scene.geoms[scene.ngeom]
    api.mjv_initGeom(geom, api.mjtGeom.mjGEOM_LINE, np.zeros(3), start,
                    np.eye(3).ravel(), color)
    # 线宽按像素设置，仅用于显示，不参与动力学或碰撞。
    api.mjv_connector(geom, api.mjtGeom.mjGEOM_LINE, 3., start, end)
    scene.ngeom += 1


def _sphere(scene: VisualScene, position: FloatArray, color: FloatArray,
            api: DrawingAPI) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    api.mjv_initGeom(scene.geoms[scene.ngeom], api.mjtGeom.mjGEOM_SPHERE,
                    np.array([.004, 0., 0.]), position, np.eye(3).ravel(), color)
    scene.ngeom += 1


def _path(scene: VisualScene, points: Sequence[FloatArray], color: FloatArray,
          api: DrawingAPI, maximum: int = 160) -> None:
    if len(points) < 2:
        return
    # 均匀抽取显示点，保留完整轨迹范围，避免耗尽查看器的几何槽位。
    indices = np.linspace(0, len(points) - 1, min(maximum, len(points)), dtype=np.intp)
    for first, second in zip(indices[:-1], indices[1:]):
        _line(scene, points[int(first)], points[int(second)], color, api)


def draw_paths(scene: VisualScene, left_targets: FloatArray, right_targets: FloatArray,
               left_trace: Sequence[FloatArray], right_trace: Sequence[FloatArray]) -> None:
    """灰线为参考圆；橙线为左臂实际轨迹；青线为右臂实际轨迹。"""
    api = cast(DrawingAPI, import_module('mujoco'))
    scene.ngeom = 0
    gray = np.array([.8, .8, .8, 1.], dtype=np.float64)
    orange = np.array([1., .45, .05, 1.], dtype=np.float64)
    cyan = np.array([0., .85, 1., 1.], dtype=np.float64)
    _path(scene, list(left_targets), gray, api)
    _path(scene, list(right_targets), gray, api)
    _path(scene, left_trace, orange, api)
    _path(scene, right_trace, cyan, api)
    if left_trace:
        _sphere(scene, left_trace[-1], orange, api)
    if right_trace:
        _sphere(scene, right_trace[-1], cyan, api)