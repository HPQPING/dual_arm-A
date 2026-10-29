"""开发工具：从版本一逐点计划生成并复核版本二连续画圆候选，不连接真机."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from archer_mujoco.dual_model_loader import load_dual_arm
from archer_mujoco.model_loader import mujoco
from ..circle_planner import CIRCLE_CENTER, CIRCLE_RADIUS, MOUNT_XYZ, TCP_XYZ, inward_rotation
from ..continuous_trajectory import plan_continuous_pair
from ..dual_arm_trajectory import rigid_transform
from ..kinematics import ArmKinematics, rotation_error
from ..model_checks import CollisionModel, cross_arm_failure, validate_tool_chain
from ..trajectory import FloatArray, joint_names


def read_source(data: Any) -> dict[str, FloatArray]:
    """严格读取上一轮导出的旧格式；仅取圆弧路径，不沿用逐点停车时序."""
    if (
        not isinstance(data, dict)
        or set(data) != {"version", "interpolation", "left", "right"}
        or type(data["version"]) is not int
        or data["version"] != 1
        or data["interpolation"] != "quintic_stop"
    ):
        raise ValueError("输入必须为上一轮导出的版本一逐点停车计划")
    result = {}
    grids = []
    for side in ("left", "right"):
        arm = data[side]
        if (
            not isinstance(arm, dict)
            or set(arm) != {"joint_names", "points"}
            or arm["joint_names"] != list(joint_names(side))
            or not isinstance(arm["points"], list)
            or not 18 <= len(arm["points"]) <= 258
        ):
            raise ValueError("圆轨迹需要正确关节顺序、一个初值和至少17个圆弧点")
        times, positions = [], []
        for point in arm["points"]:
            if not isinstance(point, dict) or set(point) != {"positions", "time_from_start"}:
                raise ValueError("输入点字段错误")
            stamp = point["time_from_start"]
            if not isinstance(stamp, dict) or set(stamp) != {"sec", "nanosec"}:
                raise ValueError("输入时间格式错误")
            seconds, nanos = stamp["sec"], stamp["nanosec"]
            if (
                type(seconds) is not int
                or type(nanos) is not int
                or seconds < 0
                or not 0 <= nanos < 10**9
            ):
                raise ValueError("输入时间必须是非负整数秒及合法纳秒")
            value = seconds * 10**9 + nanos
            if (not times and value != 0) or (times and value <= times[-1]):
                raise ValueError("输入时间必须从零开始且递增")
            q = point["positions"]
            if (
                not isinstance(q, list)
                or len(q) != 6
                or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                    for v in q
                )
            ):
                raise ValueError("输入位置必须是有限六维数值列表")
            times.append(value)
            positions.append(q)
        grids.append(times)
        result[side] = np.asarray(positions, dtype=np.float64)
    if grids[0] != grids[1]:
        raise ValueError("左右时间轴必须一致")
    return result


def main() -> int:
    """先完成解析约束、TCP和跨臂采样检查，全部通过后才导出新格式候选."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-file", type=Path, required=True)
    parser.add_argument("--limits-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--center", nargs=3, type=float, default=list(CIRCLE_CENTER))
    parser.add_argument("--radius", type=float, default=CIRCLE_RADIUS)
    parser.add_argument("--tilt-deg", type=float, default=10.0)
    parser.add_argument("--check-dt", type=float, default=0.02)
    args = parser.parse_args()
    paths = (args.plan_file, args.limits_file, args.output, args.report)
    if len({p.resolve() for p in paths}) != 4:
        parser.error("输入、限制、输出及报告路径必须不同")
    if args.output.exists() or args.report.exists():
        parser.error("输出或报告已存在，请选择新文件名")
    if (
        not math.isfinite(args.check_dt)
        or not 0.001 <= args.check_dt <= 0.02
        or not math.isfinite(args.radius)
        or args.radius <= 0.0
        or not all(math.isfinite(v) for v in args.center)
    ):
        parser.error("圆参数必须有限有效；检查间隔应在0.001到0.02秒之间")
    report: dict[str, Any] = {
        "passed": False,
        "commands_sent": False,
        "execution_supported": False,
    }
    payload = None
    try:
        source, config = args.plan_file.read_bytes(), args.limits_file.read_bytes()
        report.update(
            {
                "source_sha256": hashlib.sha256(source).hexdigest(),
                "limits_sha256": hashlib.sha256(config).hexdigest(),
            }
        )
        points = read_source(yaml.safe_load(source.decode("utf-8")))
        settings = cast(dict[str, Any], yaml.safe_load(config.decode("utf-8")))
        params = settings["archer_position_driver"]["ros__parameters"]
        constraints = [
            np.asarray([params[f"{side}_{field}"] for side in ("left", "right")], dtype=np.float64)
            for field in ("lower", "upper", "max_velocity", "max_acceleration")
        ]
        for field in ("lower", "upper", "max_velocity", "max_acceleration"):
            raw = [params[f"{side}_{field}"] for side in ("left", "right")]
            if any(isinstance(value, bool) for row in raw for value in row):
                raise ValueError("限制中不允许布尔值")
        count = len(points["left"])
        candidate, metrics = plan_continuous_pair(
            points["left"],
            points["right"],
            *constraints,
            stops=(0, 1, count - 1),
        )
        report.update(metrics)
        center = np.asarray(args.center, dtype=np.float64)
        models = {
            side: ArmKinematics.from_description(side, TCP_XYZ, MOUNT_XYZ)
            for side in ("left", "right")
        }
        worlds = {
            side: rigid_transform([x, 0.0, 0.0], [0.0, 0.0, np.pi / 2.0])
            for side, x in (("left", -0.21), ("right", 0.21))
        }
        rotations = {
            side: inward_rotation(side, float(np.deg2rad(args.tilt_deg)))
            for side in ("left", "right")
        }
        for model in models.values():
            validate_tool_chain(model)
        loaded = load_dual_arm(
            left_base_xyz=[-0.21, 0.0, 0.0],
            right_base_xyz=[0.21, 0.0, 0.0],
            left_base_rpy=[0.0, 0.0, np.pi / 2.0],
            right_base_rpy=[0.0, 0.0, np.pi / 2.0],
            tcp_xyz=TCP_XYZ,
            mount_xyz=MOUNT_XYZ,
            enable_contacts=True,
        )
        cast(CollisionModel, loaded.model).geom_margin[:] = 0.006
        duration = float(candidate.times[-1])
        times = set(float(t) for t in candidate.times)
        grid = np.linspace(0.0, duration, int(np.ceil(duration / args.check_dt)) + 1)
        times.update(float(t) for t in grid)
        max_distance = max_rotation = max_phase = 0.0
        last_angle = {"left": np.pi / 2.0, "right": 3.0 * np.pi / 2.0}
        for elapsed in sorted(times):
            phases = []
            segment = int(np.searchsorted(candidate.times, elapsed, side="right")) - 1
            index = min(count - 2, max(1, segment))
            step = candidate.times[index + 1] - candidate.times[index]
            u = (elapsed - candidate.times[index]) / step
            for side, start_angle in (("left", np.pi / 2.0), ("right", 3.0 * np.pi / 2.0)):
                q = candidate.sample(side, elapsed)
                loaded.data.qpos[loaded.arms[side].qpos_indices] = q
                if elapsed + 1e-10 < candidate.times[1]:
                    continue
                pose = worlds[side] @ models[side].forward(q)
                offset = pose[:3, 3] - center
                error = math.hypot(
                    math.hypot(float(offset[0]), float(offset[2])) - args.radius, float(offset[1])
                )
                max_distance = max(max_distance, error)
                max_rotation = max(
                    max_rotation,
                    float(
                        np.linalg.norm(
                            rotation_error(
                                rotations[side],
                                pose[:3, :3],
                            )
                        )
                    ),
                )
                reference = start_angle + ((index - 1) + u) * np.pi / (count - 2)
                angle = math.atan2(float(offset[2]), float(offset[0]))
                angle += 2.0 * np.pi * round((reference - angle) / (2.0 * np.pi))
                if angle < last_angle[side] - 1e-5 or angle > start_angle + np.pi + 0.001:
                    raise ValueError("圆弧插值出现倒退或越过终点")
                last_angle[side] = angle
                phases.append(angle - start_angle)
            if len(phases) == 2:
                max_phase = max(max_phase, abs(phases[0] - phases[1]))
            mujoco.mj_forward(loaded.model, loaded.data)
            failure = cross_arm_failure(loaded, 0.01)
            if failure is not None:
                raise ValueError(f"{elapsed:.6f}s 跨臂间隙不足：{failure}")
        # 近圆误差之外，核对首尾确为两个半圆的起止点，防止误读其他任务文件。
        for side, start_angle in (("left", np.pi / 2.0), ("right", 3.0 * np.pi / 2.0)):
            for index, angle in ((1, start_angle), (-1, start_angle + np.pi)):
                expected = center + args.radius * np.array([np.cos(angle), 0.0, np.sin(angle)])
                actual = (worlds[side] @ models[side].forward(points[side][index]))[:3, 3]
                if np.linalg.norm(expected - actual) > 0.0005:
                    raise ValueError("输入圆弧首尾与指定圆参数不一致")
        report.update(
            {
                "samples": len(times),
                "max_circle_distance_m": max_distance,
                "max_orientation_error_rad": max_rotation,
                "max_paired_arc_phase_difference_rad": max_phase,
            }
        )
        if max_distance > 0.0005 or max_rotation > 0.003 or max_phase > 0.003:
            raise ValueError("圆弧位置、姿态或双臂进度误差超出候选阈值")
        payload = candidate.payload()
        report["passed"] = True
        report["scope"] = (
            "离线候选：解析关节约束及采样TCP/跨臂检查；不含实际反馈、自碰撞、桌面、线缆或夹持"
        )
    except (ValueError, TypeError, KeyError, OSError, RuntimeError, yaml.YAMLError) as error:
        report["error"] = str(error)
    try:
        with args.report.open("x", encoding="utf-8") as evidence:
            if payload is not None and report["passed"]:
                text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
                content = text.encode("utf-8")
                with args.output.open("xb") as stream:
                    stream.write(content)
                report["output_sha256"] = hashlib.sha256(content).hexdigest()
            json.dump(report, evidence, ensure_ascii=False, indent=2, allow_nan=False)
            evidence.write("\n")
    except OSError as error:
        print(f"写入失败：{error}；未发送动作")
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
