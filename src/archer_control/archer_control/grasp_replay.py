"""在独立 MuJoCo 场景中回放抓取记录，不启动 ROS，不执行动力学或控制指令."""

from __future__ import annotations

import argparse
from importlib import import_module
import math
import os
from pathlib import Path
import sys
import time
from typing import cast

import numpy as np

from .checks.grasp_record_check import GraspRun, RecordedFrame, load_run
from archer_mujoco.grasp_model_loader import LoadedGrasp, load_grasp_model, physics
from archer_mujoco.grasp_scene import GraspConfig, ViewerAPI


def apply_frame(loaded: LoadedGrasp, frame: RecordedFrame) -> None:
    """只重建可见几何；隐藏指腹没有被记录，不伪造其开合过程."""
    for index, side in enumerate(("left", "right")):
        arm = loaded.arms[side]
        loaded.data.qpos[arm.qpos_indices] = frame.positions[index]
        loaded.data.qvel[arm.dof_indices] = frame.velocities[index]
    q = loaded.object_qpos
    loaded.data.qpos[q: q + 7] = frame.object_pose
    loaded.data.time = frame.time
    # mj_forward 只更新几何和派生量；不调用 mj_step，也不向驱动器下发记录输入。
    physics.mj_forward(loaded.model, loaded.data)


def verify_model(run: GraspRun) -> tuple[LoadedGrasp, float]:
    config = GraspConfig(**run.config)
    config.validate()
    initial = run.frames[0]
    loaded = load_grasp_model(initial.positions[0], initial.positions[1], config=config)
    maximum = 0.0
    for frame in run.frames:
        for index, side in enumerate(("left", "right")):
            bounds = loaded.arms[side].joint_limits
            # 实际反馈允许极小数值误差，不能直接裁剪或修改记录。
            if np.any(frame.positions[index] < bounds[:, 0] - 1e-5) or np.any(
                frame.positions[index] > bounds[:, 1] + 1e-5
            ):
                raise ValueError(f"{side} 在 {frame.time:.3f} s 的反馈超出当前模型关节限位")
        apply_frame(loaded, frame)
        actual = np.asarray(
            [loaded.data.site_xpos[loaded.arms[side].tcp_site_id] for side in ("left", "right")],
            dtype=np.float64,
        )
        error = float(np.max(np.linalg.norm(actual - frame.tcp, axis=1)))
        maximum = max(maximum, error)
        if error > 1e-6:
            raise ValueError(
                f"{frame.time:.3f} s 的记录 TCP 与当前模型不一致：{error*1000.:.6f} mm"
            )
    apply_frame(loaded, initial)
    return loaded, maximum


def main(args: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="已完成的一轮抓取记录目录")
    parser.add_argument("--headless", action="store_true", help="逐帧核对模型，不打开窗口")
    parser.add_argument("--speed", type=float, default=1.0, help="回放速度，0.1～10 倍")
    options = parser.parse_args(args)
    if not math.isfinite(options.speed) or not 0.1 <= options.speed <= 10.0:
        parser.error("--speed 必须在 0.1～10 之间")
    try:
        run = load_run(options.directory)
        loaded, maximum = verify_model(run)
        print(f"通过：{len(run.frames)} 帧，当前模型与记录 TCP 最大差 {maximum*1000.:.9f} mm")
        if options.headless:
            return 0
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise ValueError("没有 DISPLAY；请在 Ubuntu 桌面终端运行，或使用 --headless")
        api = cast(ViewerAPI, import_module("mujoco.viewer"))
        print("只读几何回放：完整圆柱与真实静态 GP80；关闭窗口结束，不会启动任务。")
        with api.launch_passive(
            loaded.model, loaded.data, key_callback=lambda key: None
        ) as viewer:
            with viewer.lock():
                viewer.cam.lookat[:] = [0.0, 0.25, 0.3]
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.7, 90.0, -15.0
            started = time.monotonic()
            origin = run.frames[0].time
            index, last_stage = 0, ""
            while viewer.is_running():
                elapsed = (time.monotonic() - started) * options.speed
                while (
                    index + 1 < len(run.frames) and run.frames[index + 1].time - origin <= elapsed
                ):
                    index += 1
                frame = run.frames[index]
                with viewer.lock():
                    apply_frame(loaded, frame)
                if frame.stage != last_stage:
                    print(f"{frame.time:.3f} s：{frame.stage}；夹爪指令 {frame.command:.1f}")
                    last_stage = frame.stage
                viewer.sync()
                if index == len(run.frames) - 1:
                    print("回放完成；保留末帧，关闭窗口即可退出。")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.02)
                    break
                time.sleep(0.01)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as exc:
        print("回放失败：" + str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
