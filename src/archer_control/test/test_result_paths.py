"""验证任务结果分目录与抓取记录，使用合成状态，不启动仿真或连接真机."""

from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import patch
import unittest

import numpy as np

from archer_control.grasp_recording import GraspRunRecorder
from archer_control.result_paths import (
    discard_failed_run,
    discard_recording,
    new_run_directory,
)

if TYPE_CHECKING:
    from archer_control.simple_grasp_scene import SimpleGraspDemo


def synthetic_demo() -> "SimpleGraspDemo":
    """生成合成状态，避免访问MuJoCo或实际设备."""
    qpos = np.zeros(19, dtype=np.float64)
    qpos[15] = 1.0
    data = SimpleNamespace(
        time=0.0, qpos=qpos, qvel=np.zeros(12), ctrl=np.zeros(12), site_xpos=np.zeros((2, 3))
    )
    arms = {
        side: SimpleNamespace(
            qpos_indices=np.arange(start, start + 6),
            dof_indices=np.arange(start, start + 6),
            actuator_indices=np.arange(start, start + 6),
            tcp_site_id=index,
        )
        for index, (side, start) in enumerate((("left", 0), ("right", 6)))
    }
    return cast(
        "SimpleGraspDemo",
        SimpleNamespace(
            loaded=SimpleNamespace(data=data, arms=arms, object_qpos=12),
            gripper_command=2.0,
            attached=False,
        ),
    )


class ResultDirectoryTests(unittest.TestCase):
    """检查并发目录不覆盖、文件阻挡和连续两轮记录."""

    def test_parallel_runs_have_local_timestamp_directories(self) -> None:
        """同秒产生的多轮任务仍各自保存，目录名便于按日期排序."""
        with TemporaryDirectory() as folder:
            root = Path(folder) / "archer_results"
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(lambda _: new_run_directory(root), range(16)))
            self.assertEqual(len(set(paths)), 16)
            for path in paths:
                self.assertEqual(path.parent, root)
                self.assertRegex(path.name, r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}_任务$")
                self.assertTrue(path.is_dir())

    def test_start_time_is_preserved_when_archiving_later(self) -> None:
        """按实际开始时间命名，晚几分钟归档也不能改成完成时间."""
        with TemporaryDirectory() as folder:
            began = datetime.now().astimezone() - timedelta(minutes=5)
            first = new_run_directory(Path(folder), "画圆", began)
            second = new_run_directory(Path(folder), "画圆", began)
            self.assertTrue(first.name.startswith(began.strftime("%Y-%m-%d_%H-%M-%S_%f")))
            self.assertTrue(second.name.startswith(began.strftime("%Y-%m-%d_%H-%M-%S_%f")))
            self.assertNotEqual(first, second)

    def test_existing_file_is_not_replaced(self) -> None:
        """同名文件阻挡目录时明确失败，不删除或覆盖原始文件."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "archer_results"
            path.write_text("preserve", encoding="utf-8")
            with self.assertRaises(OSError):
                new_run_directory(path)
            self.assertEqual(path.read_text(encoding="utf-8"), "preserve")

    def test_grasp_runs_only_write_json_csv_and_keep_previous_run(self) -> None:
        """保留原有状态、事件及摘要数据，第二轮不覆盖第一轮，不生成图片."""
        demo = synthetic_demo()
        with TemporaryDirectory() as folder:
            recorder = GraspRunRecorder(Path(folder))
            previous = None
            saved = None
            for _ in range(3):
                recorder.begin(demo)
                current = recorder.folder
                assert current is not None
                with self.assertRaises(RuntimeError):
                    recorder.begin(demo)
                demo.loaded.data.time += 0.1
                recorder.sample(demo, "RUNNING")
                result = recorder.finish(demo, {"state": "COMPLETED", "ros_stage": "COMPLETED"})
                self.assertFalse(recorder.active)
                self.assertFalse(current.exists())
                current = Path(str(result["directory"]))
                self.assertTrue(current.name.endswith("_协同抓取"))
                self.assertEqual(
                    {p.name for p in current.iterdir()},
                    {"state.csv", "events.csv", "summary.json"},
                )
                self.assertEqual(result["directory"], str(current))
                self.assertTrue(
                    json.loads((current / "summary.json").read_text())["recording"][
                        "task_completed"
                    ]
                )
                with (current / "state.csv").open(newline="", encoding="utf-8") as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(len(rows), 4)
                self.assertEqual(len({len(row) for row in rows}), 1)
                if previous is not None:
                    self.assertNotEqual(previous, current)
                    self.assertEqual((previous / "summary.json").read_bytes(), saved)
                previous = current
                saved = (current / "summary.json").read_bytes()

    def test_failure_and_abandoned_run_leave_no_archive(self) -> None:
        """失败、停止、中断或直接关闭时，清理临时采样，不创建最终任务目录."""
        demo = synthetic_demo()
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for state in ("FAILED", "STOPPED", "INTERRUPTED"):
                recorder = GraspRunRecorder(root)
                recorder.begin(demo)
                staging = recorder.folder
                assert staging is not None
                result = recorder.finish(demo, {"state": state, "ros_stage": state})
                self.assertFalse(result["saved"])
                self.assertFalse(staging.exists())
                self.assertEqual(list(root.iterdir()), [])
            recorder = GraspRunRecorder(root)
            recorder.begin(demo)
            staging = recorder.folder
            assert staging is not None
            recorder.close()
            self.assertFalse(staging.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_copy_failure_rolls_back_partial_archive(self) -> None:
        """归档复制失败不能留下看似成功的目录，同时清理本轮临时数据."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            recorder = GraspRunRecorder(root)
            demo = synthetic_demo()
            recorder.begin(demo)
            staging = recorder.folder
            assert staging is not None
            with patch("archer_control.result_paths.shutil.copy2", side_effect=OSError("test")):
                with self.assertRaises(OSError):
                    recorder.finish(demo, {"state": "COMPLETED", "ros_stage": "COMPLETED"})
            self.assertFalse(staging.exists())
            self.assertEqual(list(root.iterdir()), [])

    def test_cleanup_rejects_unowned_paths(self) -> None:
        """清理函数拒绝任意路径，不能删掉其他用户文件夹."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            other = root / "user_data"
            other.mkdir()
            (other / "keep.txt").write_text("preserve")
            with self.assertRaises(ValueError):
                discard_recording(other)
            with self.assertRaises(ValueError):
                discard_failed_run(other, root)
            self.assertTrue((other / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
