"""Static measured-pose comparison; no ROS topics, physics stepping or hardware.

Red = +X, green = +Y, blue = +Z. Long axes at bases; short axes at TCPs.
--base-yaw-deg changes only the proposed base-to-world installation rotation.
"""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from importlib import import_module
import math
import time
from typing import Protocol, cast

import numpy as np

from .dual_model_loader import LoadedDualArm, load_dual_arm
from .model_loader import FloatArray, ModelInterface, DataInterface, mujoco


LEFT_MEASURED = (
    0.0016936991186183399, -1.5557692501183322, 3.131012580369181,
    -1.5715998405409322, -0.05458065430333334, 1.6680315339869978,
)
RIGHT_MEASURED = (
    -0.008299789093679984, -1.5717756091728772, 3.132204154731199,
    -1.5717483593281518, -0.02457215514032048, 1.4469849024526769,
)


class VisualGeom(Protocol):
    pass


class GeomArray(Protocol):
    def __getitem__(self, index: int) -> VisualGeom: ...


class VisualScene(Protocol):
    geoms: GeomArray
    ngeom: int
    maxgeom: int


class Camera(Protocol):
    lookat: FloatArray
    distance: float
    azimuth: float
    elevation: float


class AxisViewer(Protocol):
    cam: Camera
    user_scn: VisualScene
    def lock(self) -> AbstractContextManager[object]: ...
    def sync(self) -> None: ...
    def is_running(self) -> bool: ...
    def close(self) -> None: ...


class ViewerAPI(Protocol):
    def launch_passive(self, model: ModelInterface, data: DataInterface) -> AxisViewer: ...


class GeomKinds(Protocol):
    mjGEOM_ARROW: object


class DrawingAPI(Protocol):
    mjtGeom: GeomKinds
    def mjv_initGeom(self, geom: VisualGeom, kind: object, size: FloatArray,
                     pos: FloatArray, mat: FloatArray, rgba: FloatArray) -> None: ...
    def mjv_connector(self, geom: VisualGeom, kind: object, width: float,
                      start: FloatArray, end: FloatArray) -> None: ...


def apply_measured_pose(loaded: LoadedDualArm) -> None:
    """Set qpos exactly for geometry display, keeping the original model limits."""
    for side, values in (('left', LEFT_MEASURED), ('right', RIGHT_MEASURED)):
        arm = loaded.arms[side]
        q = np.asarray(values, dtype=np.float64)
        for index, (position, bounds) in enumerate(zip(q, arm.joint_limits), start=1):
            if position < bounds[0] or position > bounds[1]:
                print(f'NOTE: {side} J{index}={position:.8f} rad is outside '
                      f'model [{bounds[0]:.8f}, {bounds[1]:.8f}]. '
                      'Displayed exactly for comparison; limits are unchanged.')
        loaded.data.qpos[arm.qpos_indices] = q
        loaded.data.ctrl[arm.actuator_indices] = q
    loaded.data.qvel[:] = 0.
    # FK computes the geometric pose without advancing time or enforcing a motion.
    mujoco.mj_forward(loaded.model, loaded.data)


def base_rotation(yaw: float) -> FloatArray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]], dtype=np.float64)


def draw_axes(scene: VisualScene, origin: FloatArray, rotation: FloatArray,
              length: float, drawing: DrawingAPI) -> None:
    colors = ((1., 0., 0., 1.), (0., 1., 0., 1.), (0., 0.4, 1., 1.))
    if scene.ngeom + 3 > scene.maxgeom:
        raise ValueError('Viewer has insufficient custom-geometry capacity')
    for index, color in enumerate(colors):
        geom = scene.geoms[scene.ngeom]
        drawing.mjv_initGeom(geom, drawing.mjtGeom.mjGEOM_ARROW,
                            np.zeros(3, dtype=np.float64), origin,
                            np.eye(3, dtype=np.float64).ravel(), np.asarray(color, dtype=np.float64))
        drawing.mjv_connector(geom, drawing.mjtGeom.mjGEOM_ARROW, 0.004,
                             origin, origin + rotation[:, index] * length)
        scene.ngeom += 1


def update_axes(scene: VisualScene, loaded: LoadedDualArm, yaw: float) -> None:
    drawing = cast(DrawingAPI, import_module('mujoco'))
    scene.ngeom = 0
    rotation = base_rotation(yaw)
    # Raise base arrows 2 mm above the mounting plane for visibility.
    draw_axes(scene, np.array([-.42, 0., .002]), rotation, .15, drawing)
    draw_axes(scene, np.array([0., 0., .002]), rotation, .15, drawing)
    for side in ('left', 'right'):
        site = loaded.arms[side].tcp_site_id
        draw_axes(scene, loaded.data.site_xpos[site].copy(),
                  loaded.data.site_xmat[site].reshape(3, 3).copy(), .06, drawing)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-yaw-deg', type=float, default=0.)
    parser.add_argument('--left-urdf', help='Optional expanded left URDF')
    parser.add_argument('--right-urdf', help='Optional expanded right URDF')
    parser.add_argument('--no-viewer', action='store_true', help='Print comparison data only')
    args = parser.parse_args()
    if not math.isfinite(args.base_yaw_deg):
        parser.error('base-yaw-deg must be finite')
    yaw = math.radians(args.base_yaw_deg)
    loaded = load_dual_arm(
        left_urdf=args.left_urdf, right_urdf=args.right_urdf,
        left_base_xyz=[-.42, 0., 0.], right_base_xyz=[0., 0., 0.],
        left_base_rpy=[0., 0., yaw], right_base_rpy=[0., 0., yaw],
        enable_contacts=False,
    )
    apply_measured_pose(loaded)
    print('Static measured pose. Red +X; green +Y; blue +Z.')
    print(f'Candidate installation yaw: {args.base_yaw_deg:.1f} degrees')
    for side in ('left', 'right'):
        arm = loaded.arms[side]
        position = loaded.data.site_xpos[arm.tcp_site_id]
        rotation = loaded.data.site_xmat[arm.tcp_site_id].reshape(3, 3)
        print(f'{side} TCP world XYZ (m): {np.round(position, 6).tolist()}')
        print(f'{side} TCP local +Z in world: {np.round(rotation[:, 2], 6).tolist()}')
    if args.no_viewer:
        return 0
    module = cast(ViewerAPI, import_module('mujoco.viewer'))
    viewer = module.launch_passive(loaded.model, loaded.data)
    try:
        with viewer.lock():
            viewer.cam.lookat[:] = [-.21, 0., .25]
            viewer.cam.distance = 1.35
            viewer.cam.azimuth = 135.
            viewer.cam.elevation = -15.
            update_axes(viewer.user_scn, loaded, yaw)
        while viewer.is_running():
            viewer.sync()
            time.sleep(1. / 30.)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())