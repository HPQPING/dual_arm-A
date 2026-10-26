"""验证正式画圆入口的许可、路径及ROS参数处理，不连接机械臂。"""

from pathlib import Path
import hashlib
import json
from unittest.mock import patch

import pytest
import rclpy

from archer_hardware import circle_task


def inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """仅创建用于入口验证的空占位文件，不作为可执行轨迹。"""
    source = tmp_path / 'archer_results' / '2030-01-02_03-04-05_画圆'
    prepared = tmp_path / 'archer_results_trans' / source.name
    source.mkdir(parents=True)
    prepared.mkdir(parents=True)
    for name in ('tracking.csv', 'summary.json'):
        (source / name).write_text('', encoding='utf8')
    for name in ('tracking.json', 'tracking_report.json'):
        (prepared / name).write_text('', encoding='utf8')
    config = tmp_path / 'limits.yaml'
    config.write_text('/**:\n  ros__parameters: {}\n', encoding='utf8')
    return source, prepared, config


def test_default_prepared_directory_follows_workspace(tmp_path: Path) -> None:
    """默认目录跟随本轮任务，不读取旧日期或缓存候选。"""
    source, prepared, config = inputs(tmp_path)
    assert circle_task.resolve_inputs(source, None, config) == (source, prepared, config)


def test_missing_files_are_rejected_before_running(tmp_path: Path) -> None:
    """缺少输入文件时不得创建控制节点。"""
    source, prepared, config = inputs(tmp_path)
    (prepared / 'tracking_report.json').unlink()
    with patch.object(circle_task, 'run') as runner:
        with pytest.raises(SystemExit):
            circle_task.main(['--run-dir', str(source), '--limits-file', str(config),
                              '--execute-authorized-full-circle'])
        runner.assert_not_called()


def test_explicit_permission_is_required(tmp_path: Path) -> None:
    """只指定目录不能隐式开始连接或运动。"""
    source, _, config = inputs(tmp_path)
    with patch.object(circle_task, 'run') as runner:
        with pytest.raises(SystemExit):
            circle_task.main(['--run-dir', str(source), '--limits-file', str(config)])
        runner.assert_not_called()


def test_installed_config_and_launch_ros_arguments(tmp_path: Path) -> None:
    """launch附加ROS参数仍能解析，完整参数可交给rclpy初始化。"""
    source, prepared, config = inputs(tmp_path)
    arguments = ['--run-dir', str(source), '--execute-authorized-full-circle',
                 '--ros-args', '-r', '__node:=circle_entry_test']

    def run(directory, source_dir, limits, *, ros_args):
        assert (directory, source_dir, limits) == (prepared, source, config)
        assert ros_args == arguments
        context = rclpy.Context()
        try:
            rclpy.init(args=[*ros_args, '--ros-args', '--params-file', str(limits)],
                       context=context)
            assert context.ok()
        finally:
            if context.ok():
                context.shutdown()
        return 7

    with patch.object(circle_task, 'default_limits_file', return_value=config):
        with patch.object(circle_task, 'run', side_effect=run):
            assert circle_task.main(arguments) == 7


def test_unknown_option_is_not_silently_ignored(tmp_path: Path) -> None:
    """拼错参数须退出，避免误用默认文件开始任务。"""
    source, _, config = inputs(tmp_path)
    with patch.object(circle_task, 'run') as runner:
        with pytest.raises(SystemExit):
            circle_task.main(['--run-dir', str(source), '--limits-file', str(config),
                              '--execute-authorized-full-circle', '--prepard-dir', 'wrong'])
        runner.assert_not_called()


def report_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """创建仅用于路径和哈希核对的文件，不代表通过完整轨迹预检。"""
    source, prepared, config = inputs(tmp_path)
    hashes = {
        'source_csv': str(source / 'tracking.csv'),
        'csv_sha256': hashlib.sha256((source / 'tracking.csv').read_bytes()).hexdigest(),
        'summary_sha256': hashlib.sha256((source / 'summary.json').read_bytes()).hexdigest(),
        'output_sha256': hashlib.sha256((prepared / 'tracking.json').read_bytes()).hexdigest(),
        'limits_sha256': hashlib.sha256(config.read_bytes()).hexdigest(),
    }
    (prepared / 'tracking_report.json').write_text(json.dumps(hashes), encoding='utf8')
    return source, prepared, config


def test_trajectory_path_locates_report_source_and_config(tmp_path: Path) -> None:
    """只给轨迹文件时通过报告定位原始记录，不选择其他日期的任务。"""
    source, prepared, config = report_inputs(tmp_path)
    trajectory = prepared / 'tracking.json'
    assert circle_task.resolve_trajectory_input(trajectory, None, config) == (
        source, prepared, config, trajectory)
    with patch.object(circle_task, 'run', return_value=0) as runner:
        assert circle_task.main([
            '--trajectory-file', str(trajectory), '--limits-file', str(config),
            '--execute-authorized-full-circle']) == 0
        assert runner.call_args.kwargs['trajectory_file'] == trajectory


@pytest.mark.parametrize('changed', ['tracking.csv', 'summary.json', 'limits.yaml'])
def test_stale_inputs_are_rejected_before_node_creation(tmp_path: Path, changed: str) -> None:
    """原始记录或配置改变时不能借单文件入口绕过来源检查。"""
    source, prepared, config = report_inputs(tmp_path)
    path = config if changed == 'limits.yaml' else source / changed
    path.write_text('changed', encoding='utf8')
    with patch.object(circle_task, 'run') as runner:
        with pytest.raises(SystemExit):
            circle_task.main(['--trajectory-file', str(prepared / 'tracking.json'),
                              '--limits-file', str(config), '--execute-authorized-full-circle'])
        runner.assert_not_called()


def test_report_file_is_not_accepted_as_trajectory(tmp_path: Path) -> None:
    """用户应输入tracking.json，不能把报告当作关节轨迹执行。"""
    _, prepared, config = report_inputs(tmp_path)
    with pytest.raises(ValueError, match='不是tracking_report'):
        circle_task.resolve_trajectory_input(prepared / 'tracking_report.json', None, config)


def test_custom_trajectory_basename_uses_matching_report(tmp_path: Path) -> None:
    """显式另存的轨迹文件也使用同主文件名的报告。"""
    source, prepared, config = report_inputs(tmp_path)
    (prepared / 'tracking.json').rename(prepared / 'chosen.json')
    (prepared / 'tracking_report.json').rename(prepared / 'chosen_report.json')
    assert circle_task.resolve_trajectory_input(prepared / 'chosen.json', None, config) == (
        source, prepared, config, prepared / 'chosen.json')
