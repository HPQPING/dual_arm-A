"""同步记录抓取仿真的实际状态，不发布控制命令，不连接实机。"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, TextIO

from .grasp_record_format import EVENT_FIELDS, STATE_FIELDS
from .result_paths import (
    discard_failed_run, discard_recording, new_recording_directory, publish_recording,
)

if TYPE_CHECKING:
    from .simple_grasp_scene import SimpleGraspDemo


class GraspRunRecorder:
    """主线程采样；规划线程运行期间不得访问仿真数据。"""

    def __init__(self, directory: Path) -> None:
        self.directory = directory.expanduser()
        self.active = False
        self.folder: Path | None = None
        self.staging: Path | None = None
        self.started_at: datetime | None = None
        self.trace: TextIO | None = None
        self.events: TextIO | None = None
        self.count = 0
        self.history: list[str] = []
        self.last_sample = -math.inf
        self.last_stage = ''
        self.started_wall = 0.
        self.last_flush = 0.

    def begin(self, demo: SimpleGraspDemo) -> None:
        if self.active:
            raise RuntimeError('上一轮记录尚未关闭')
        self.started_at = datetime.now().astimezone()
        self.folder = new_recording_directory()
        self.staging = self.folder
        try:
            self.trace = (self.folder / 'state.csv').open('x', encoding='utf-8', newline='')
            self.events = (self.folder / 'events.csv').open('x', encoding='utf-8', newline='')
            csv.writer(self.trace).writerow(STATE_FIELDS)
            csv.writer(self.events).writerow(EVENT_FIELDS)
            self.active = True
            self.count, self.history = 0, []
            self.last_sample, self.last_stage = -math.inf, ''
            self.started_wall = self.last_flush = time.monotonic()
            self.event('READY')
            self.sample(demo, 'READY', force=True)
        except Exception:
            self.close()
            raise

    def event(self, stage: str) -> None:
        """只写阶段信息；可在后台规划期间调用，不访问模型。"""
        if not self.active or self.events is None or (self.history and self.history[-1] == stage):
            return
        csv.writer(self.events).writerow([datetime.now(timezone.utc).isoformat(),
                                         time.monotonic() - self.started_wall, stage])
        self.history.append(stage)
        self.events.flush()

    def sample(self, demo: SimpleGraspDemo, stage: str, *, force: bool = False) -> None:
        if not self.active or self.trace is None:
            return
        data = demo.loaded.data
        now = float(data.time)
        if not force and stage == self.last_stage and now - self.last_sample < .05 - 1e-9:
            return
        # 一行中所有数组来自同一次主线程采样，不拼接不同时间的左右消息。
        values: list[float] = []
        for quantity, source in enumerate((data.qpos, data.qvel, data.ctrl)):
            for side in ('left', 'right'):
                arm = demo.loaded.arms[side]
                indices = (arm.qpos_indices, arm.dof_indices, arm.actuator_indices)[quantity]
                values.extend(float(value) for value in source[indices])
        q = demo.loaded.object_qpos
        values.extend(float(value) for value in data.qpos[q:q + 7])
        for side in ('left', 'right'):
            values.extend(float(value) for value in data.site_xpos[demo.loaded.arms[side].tcp_site_id])
        if not all(math.isfinite(value) for value in (*values, now, demo.gripper_command)):
            raise ValueError('记录状态出现非有限数值')
        csv.writer(self.trace).writerow([time.monotonic() - self.started_wall, now, stage,
                                        demo.gripper_command, int(demo.attached), *values])
        self.last_sample, self.last_stage = now, stage
        self.count += 1
        if force or time.monotonic() - self.last_flush >= 1.:
            self.trace.flush()
            self.last_flush = time.monotonic()

    def finish(self, demo: SimpleGraspDemo, result: dict[str, object]) -> dict[str, object]:
        if not self.active or self.folder is None:
            return {'enabled': True, 'active': False}
        stage = str(result.get('ros_stage', result.get('state', 'UNKNOWN')))
        if result.get('state') != 'COMPLETED' or stage != 'COMPLETED':
            self.close()
            return {'enabled': True, 'complete': False, 'task_completed': False,
                    'saved': False, 'reason': '未成功完成，不保存任务文件'}
        final: Path | None = None
        try:
            self.event(stage)
            self.sample(demo, stage, force=True)
            staging = self.folder
            self.close(discard=False)
            final = publish_recording(staging, self.directory, '协同抓取', self.started_at)
            self.staging = None
            self.folder = final
            metadata: dict[str, object] = {
                'enabled': True, 'complete': True, 'saved': True, 'task_name': '协同抓取',
                'directory': str(self.folder),
                'started_at': self.started_at.isoformat() if self.started_at is not None else None,
                'task_completed': result.get('state') == 'COMPLETED' and stage == 'COMPLETED',
                'state_csv': str(self.folder / 'state.csv'), 'events_csv': str(self.folder / 'events.csv'),
                'summary_json': str(self.folder / 'summary.json'), 'sample_count': self.count,
                'observed_stages': list(self.history),
                'sampling': '同一 MuJoCo 状态，约每 0.05 仿真秒及阶段变化时记录',
                'drive_input_note': '驱动输入含仿真参考速度前馈，不能直接当作实机位置指令',
                'scope': '仅仿真记录；虚拟附着不代表已验证真实抓取，不可直接下发实机',
            }
            summary = dict(result)
            summary['recording'] = metadata
            temporary = self.folder / 'summary.json.tmp'
            temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
            temporary.replace(self.folder / 'summary.json')
            return metadata
        except Exception:
            if final is not None:
                discard_failed_run(final, self.directory)
                self.folder = None
            raise
        finally:
            self.close()

    def close(self, *, discard: bool = True) -> None:
        self.active = False
        try:
            if self.trace is not None:
                self.trace.close()
        finally:
            self.trace = None
            if self.events is not None:
                try:
                    self.events.close()
                finally:
                    self.events = None
            if discard and self.staging is not None:
                discard_recording(self.staging)
                if self.folder == self.staging:
                    self.folder = None
                self.staging = None
