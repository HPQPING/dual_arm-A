"""无窗口验证同一圆柱的上下夹持、抬升、平移及释放；不发布运动指令。"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path

from .grasp_scene import GraspBench, GraspConfig


def run_check(config: GraspConfig) -> dict[str, object]:
    bench = GraspBench(config)
    initial_bottom = bench.object_bottom()
    stages: list[str] = []
    maximum_steps = math.ceil((sum(duration for _, duration in bench.stages) + 1.) / config.timestep)
    for _ in range(maximum_steps):
        bench.step()
        if not stages or stages[-1] != bench.state:
            stages.append(bench.state)
        if bench.finished:
            break
    return {
        'state': bench.state,
        'failure': bench.failure,
        'config': asdict(config),
        'table_z_m': config.table_z,
        'nominal_object_center_world_m': [0., .4, config.center_z],
        'initial_bottom_clearance_mm': (initial_bottom - config.table_z) * 1000.,
        'final_bottom_clearance_mm': (bench.object_bottom() - config.table_z) * 1000.,
        'final_object_center_world_m': bench.data.xpos[bench.object_body].tolist(),
        'maximum_motion_tracking_error_mm': bench.maximum_tracking_error * 1000.,
        'minimum_lift_clearance_mm': (bench.minimum_lift_clearance * 1000.
                                      if math.isfinite(bench.minimum_lift_clearance) else None),
        'maximum_contact_penetration_mm': bench.maximum_penetration * 1000.,
        'simulation_time_s': float(bench.data.time),
        'stages': stages,
        'scope': '简化指腹与运动学支撑机构的接触测试，未接入 Archer 关节、ROS 或实机',
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diameter-mm', type=float, default=10.)
    parser.add_argument('--mass-g', type=float, default=50.)
    parser.add_argument('--output', type=Path, help='可选的 JSON 结果文件路径')
    args = parser.parse_args()
    try:
        result = run_check(GraspConfig(object_diameter=args.diameter_mm / 1000.,
                                       object_mass=args.mass_g / 1000.))
    except (ValueError, RuntimeError) as exc:
        print('检查失败：' + str(exc))
        return 1
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    print(text)
    if args.output is not None:
        path = args.output.expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + '\n', encoding='utf-8')
    return 0 if result['state'] == 'COMPLETED' else 1


if __name__ == '__main__':
    raise SystemExit(main())