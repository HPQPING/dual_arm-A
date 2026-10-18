"""统一成功任务的日期时间与名称目录，临时采样不进入结果根目录。"""

from datetime import datetime
from pathlib import Path
import re
import shutil
import tempfile


DEFAULT_RESULTS_DIR = Path.home() / 'archer_ws' / 'archer_results'


def new_run_directory(
    root: Path = DEFAULT_RESULTS_DIR, task_name: str = '任务',
    started_at: datetime | None = None,
) -> Path:
    """排他创建每轮任务目录，保留微秒以避免覆盖同秒完成的其他任务。"""
    if not isinstance(task_name, str) or re.fullmatch(r'[\w\-]{1,32}', task_name) is None:
        raise ValueError('任务名称只允许中文、字母、数字、下划线和短横线，最长32字符')
    directory = root.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    began = datetime.now().astimezone() if started_at is None else started_at.astimezone()
    stamp = began.strftime('%Y-%m-%d_%H-%M-%S_%f')
    for index in range(100):
        suffix = '' if index == 0 else f'_{index:02d}'
        result = directory / (stamp + '_' + task_name + suffix)
        try:
            result.mkdir(exist_ok=False)
        except FileExistsError:
            continue
        return result
    raise FileExistsError('连续时间戳冲突，未覆盖已有任务目录')


def new_recording_directory() -> Path:
    """流式采样写入系统临时目录，成功前不占用archer_results中的任务位置。"""
    return Path(tempfile.mkdtemp(prefix='archer_record_'))


def discard_recording(directory: Path) -> None:
    """只清理由本项目创建的系统临时目录，拒绝任意路径或符号链接。"""
    if (directory.is_symlink()
            or directory.parent.resolve() != Path(tempfile.gettempdir()).resolve()
            or not directory.name.startswith('archer_record_')):
        raise ValueError('拒绝清理非本项目临时记录目录')
    if directory.exists():
        shutil.rmtree(directory)


def discard_failed_run(directory: Path, root: Path) -> None:
    """归档写入失败时清理刚创建的目标目录，路径必须直接位于指定根目录下。"""
    if (directory.is_symlink() or directory.parent.resolve() != root.expanduser().resolve()
            or re.fullmatch(r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_\d{6}_[\w\-]{1,32}',
                            directory.name) is None):
        raise ValueError('拒绝清理根目录之外或名称不合法的目录')
    if directory.exists():
        shutil.rmtree(directory)


def publish_recording(
    staging: Path, root: Path, task_name: str, started_at: datetime | None = None,
) -> Path:
    """仅在任务完成后转存JSON和CSV，失败则撤销本次空目录或不完整副本。"""
    if (staging.is_symlink() or staging.parent.resolve() != Path(tempfile.gettempdir()).resolve()
            or not staging.name.startswith('archer_record_')):
        raise ValueError('输入必须是本项目临时采样目录')
    files = list(staging.iterdir())
    if not files or any(p.is_symlink() or not p.is_file() or p.suffix not in ('.json', '.csv')
                        for p in files):
        raise ValueError('临时记录必须仅包含JSON和CSV文件')
    target = new_run_directory(root, task_name, started_at)
    try:
        for source in files:
            shutil.copy2(source, target / source.name)
        discard_recording(staging)
    except Exception:
        discard_failed_run(target, root)
        raise
    return target
