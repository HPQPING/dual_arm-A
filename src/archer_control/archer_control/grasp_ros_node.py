"""为独立抓取仿真提供 ROS 服务与关节反馈，不连接实机。

暂停和停止均冻结整个仿真世界；这不是实机急停或带重力的物理保持。
规划在后台进行，完成预检后才允许推进仿真。
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from importlib import import_module
import json
import os
from pathlib import Path
import sys
from threading import Event, Thread, enumerate as enumerate_threads
from typing import cast

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from .simple_grasp_scene import SimpleGraspDemo
from .grasp_recording import GraspRunRecorder
from .result_paths import DEFAULT_RESULTS_DIR
from .kinematics import rotation_error
from archer_mujoco.grasp_scene import Viewer, ViewerAPI


def text_message(value: str) -> String:
    message = String()
    message.data = value
    return message


class GraspSimulationNode(Node):
    def __init__(self) -> None:
        super().__init__('dual_grasp_sim')
        self.declare_parameter('output_file', '')
        self.declare_parameter('show_viewer', False)
        self.declare_parameter('record_dir', str(DEFAULT_RESULTS_DIR))
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        if self.get_parameter('use_sim_time').value is not False:
            raise ValueError('本节点发布墙钟时间戳，use_sim_time 必须为 false')
        output = self.get_parameter('output_file').value
        if not isinstance(output, str):
            raise ValueError('output_file 必须为字符串')
        self.output = Path(output).expanduser() if output else None
        record_dir = self.get_parameter('record_dir').value
        if not isinstance(record_dir, str):
            raise ValueError('record_dir 必须为字符串，留空可关闭记录')
        self.recorder = GraspRunRecorder(Path(record_dir)) if record_dir else None
        self.recording_error = ''
        show_viewer = self.get_parameter('show_viewer').value
        if not isinstance(show_viewer, bool):
            raise ValueError('show_viewer 必须为布尔值')
        if show_viewer and sys.platform.startswith('linux') and not os.environ.get('DISPLAY'):
            raise ValueError('没有可用的 DISPLAY；请在 Ubuntu 桌面终端启动，或设置 show_viewer:=false')
        self.demo = SimpleGraspDemo()
        self.resources = ExitStack()
        self.viewer: Viewer | None = None
        self.viewer_threads: tuple[Thread, ...] = ()
        if show_viewer:
            api = cast(ViewerAPI, import_module('mujoco.viewer'))
            existing_threads = set(enumerate_threads())
            try:
                self.viewer = self.resources.enter_context(api.launch_passive(
                    self.demo.loaded.model, self.demo.loaded.data, key_callback=lambda key: None))
                # 此时尚未创建规划线程；保留本次同步启动查看器所创建的线程。
                self.viewer_threads = tuple(t for t in enumerate_threads()
                                            if t not in existing_threads)
                with self.viewer.lock():
                    self.viewer.cam.lookat[:] = [0., .25, .3]
                    self.viewer.cam.distance = 1.7
                    self.viewer.cam.azimuth = 90.
                    self.viewer.cam.elevation = -15.
            except Exception:
                self.close_viewer()
                raise
        self.closing = False
        self.planning_cancel = Event()
        self.running = False
        self.stopped = False
        self.planning: Future[bool] | None = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='grasp_plan')
        self.last_stage = ''
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.stage_pub = self.create_publisher(String, '/dual/grasp/stage', qos)
        self.result_pub = self.create_publisher(String, '/dual/grasp/result', qos)
        self.marker_pub = self.create_publisher(MarkerArray, '/dual/grasp/markers', qos)
        self.gripper_pub = self.create_publisher(Float64, '/dual/grasp/gripper_command', qos)
        self.visual_ticks = 0
        self.joint_pubs = {
            side: self.create_publisher(JointState, '/' + side + '/joint_states', 10)
            for side in ('left', 'right')
        }
        self.services_kept = [
            self.create_service(Trigger, '/dual/grasp/start', self.on_start),
            self.create_service(Trigger, '/dual/grasp/pause', self.on_pause),
            self.create_service(Trigger, '/dual/grasp/resume', self.on_resume),
            self.create_service(Trigger, '/dual/grasp/stop', self.on_stop),
            self.create_service(Trigger, '/dual/grasp/reset', self.on_reset),
        ]
        self.timer = self.create_timer(.01, self.tick)
        self.publish_feedback()
        self.publish_markers()
        self.publish_stage()
        self.get_logger().info('抓取仿真 READY；调用 /dual/grasp/start 开始，启动节点不会自动运动。')
        self.get_logger().info('当前为简化抓取演示：夹爪目标 2.0，工件虚拟附着，不验证实际接触力。')
        self.get_logger().info('起点与归位目标在每次开始时读取当前仿真反馈；到位后左右 J6 目标均为 0 rad。')
        self.get_logger().info('仿真启动与重置使用用户给定的固定姿态；GP80 保留真实静态外形，暂不播放开合动画。')

    def stage(self) -> str:
        if self.stopped:
            return 'STOPPED'
        if self.planning is not None:
            return 'PLANNING'
        if not self.running and not self.demo.finished and self.demo.state != 'READY':
            return 'PAUSED'
        return self.demo.state

    def publish_stage(self) -> None:
        stage = self.stage()
        if stage != self.last_stage:
            self.record_event(stage)
            self.stage_pub.publish(text_message(stage))
            self.get_logger().info('抓取阶段：' + stage)
            self.last_stage = stage

    def publish_feedback(self) -> None:
        # 后台规划也会读取模型；规划期间不并发读取或发布它的可变数据。
        if self.planning is not None:
            return
        stamp = self.get_clock().now().to_msg()
        command = Float64()
        command.data = self.demo.gripper_command
        self.gripper_pub.publish(command)
        for side, arm in self.demo.loaded.arms.items():
            message = JointState()
            message.header.stamp = stamp
            message.name = list(arm.joint_names)
            message.position = self.demo.loaded.data.qpos[arm.qpos_indices].tolist()
            message.velocity = self.demo.loaded.data.qvel[arm.dof_indices].tolist()
            self.joint_pubs[side].publish(message)

    def publish_result(self) -> None:
        result = self.demo.summary()
        result['ros_stage'] = self.stage()
        result['recording'] = self.finish_recording(result)
        text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        self.result_pub.publish(text_message(text))
        if self.output is not None and result.get('state') == 'COMPLETED':
            try:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                self.output.write_text(text + '\n', encoding='utf-8')
            except OSError as exc:
                self.get_logger().error('结果已发布，但文件保存失败：' + str(exc))

    def record_error(self, exc: Exception) -> None:
        self.recording_error = str(exc)
        if self.recorder is not None:
            try:
                self.recorder.close()
            except OSError:
                pass
        self.get_logger().error('仿真继续执行，但记录失败：' + self.recording_error)

    def record_event(self, stage: str) -> None:
        if self.recorder is not None and not self.recording_error:
            try:
                self.recorder.event(stage)
            except (OSError, ValueError) as exc:
                self.record_error(exc)

    def record_sample(self) -> None:
        if self.recorder is not None and not self.recording_error and self.planning is None:
            try:
                self.recorder.sample(self.demo, self.stage())
            except (OSError, ValueError) as exc:
                self.record_error(exc)

    def finish_recording(self, result: dict[str, object]) -> dict[str, object]:
        if self.recorder is None:
            return {'enabled': False}
        if not self.recording_error:
            try:
                return self.recorder.finish(self.demo, result)
            except (OSError, ValueError) as exc:
                self.record_error(exc)
        return {'enabled': True, 'complete': False, 'error': self.recording_error}

    def publish_markers(self) -> None:
        """显示实际自由物件和 TCP，绝不使用参考轨迹冒充物件反馈。"""
        loaded = self.demo.loaded
        markers: list[Marker] = []
        stamp = self.get_clock().now().to_msg()
        for index, name, kind, position, scale, color in (
            (0, 'object', Marker.CYLINDER, loaded.data.xpos[loaded.object_body],
             (self.demo.config.object_diameter, self.demo.config.object_diameter, self.demo.config.object_height),
             (.95, .65, .10, 1.)),
            (1, 'table', Marker.CUBE, (0., .35, self.demo.config.table_z - .02),
             (1.3, 1.5, .04), (.35, .45, .45, .45)),
            (2, 'left_tcp', Marker.SPHERE, loaded.data.site_xpos[loaded.arms['left'].tcp_site_id],
             (.012, .012, .012), (1., .25, .10, 1.)),
            (3, 'right_tcp', Marker.SPHERE, loaded.data.site_xpos[loaded.arms['right'].tcp_site_id],
             (.012, .012, .012), (.10, .70, 1., 1.)),
        ):
            marker = Marker()
            marker.header.frame_id, marker.header.stamp = 'world', stamp
            marker.ns, marker.id = 'grasp_' + name, index
            marker.type, marker.action = kind, Marker.ADD
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, position)
            marker.pose.orientation.w = 1.
            if index == 0:
                matrix = loaded.data.xmat[loaded.object_body].reshape(3, 3)
                vector = rotation_error(matrix, np.eye(3, dtype=np.float64))
                angle = float(np.linalg.norm(vector))
                xyz = np.zeros(3) if angle < 1e-12 else vector * np.sin(angle / 2.) / angle
                marker.pose.orientation.x = float(xyz[0])
                marker.pose.orientation.y = float(xyz[1])
                marker.pose.orientation.z = float(xyz[2])
                marker.pose.orientation.w = float(np.cos(angle / 2.))
            marker.scale.x, marker.scale.y, marker.scale.z = scale
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
            markers.append(marker)
        message = MarkerArray()
        message.markers = markers
        self.marker_pub.publish(message)

    def prepare(self) -> bool:
        # 查看器和规划器共享模型，规划期间持锁且主线程不执行 sync。
        with self.viewer.lock() if self.viewer is not None else nullcontext():
            return self.demo.start(cancelled=self.planning_cancel.is_set)

    def on_start(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.planning is not None or self.stopped or self.demo.state != 'READY':
            response.success, response.message = False, '只允许从 READY 启动；停止或失败后请 reset'
            return response
        self.planning_cancel.clear()
        self.recording_error = ''
        if self.recorder is not None:
            try:
                self.recorder.begin(self.demo)
                self.get_logger().info('本轮临时采样目录（成功后归档）：' + str(self.recorder.folder))
            except (OSError, ValueError, RuntimeError) as exc:
                self.record_error(exc)
        self.planning = self.worker.submit(self.prepare)
        response.success, response.message = True, '已接受，正在后台规划；通过 stage/result 查询最终结果'
        self.publish_stage()
        return response

    def on_pause(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = self.running and not self.stopped
        response.message = '整个仿真世界已冻结' if response.success else '当前没有正在执行的任务'
        if response.success:
            self.running = False
        self.record_sample()
        self.publish_stage()
        return response

    def on_resume(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = self.stage() == 'PAUSED'
        response.message = '继续仿真' if response.success else '仅允许从 PAUSED 继续'
        if response.success:
            self.running = True
        self.record_sample()
        self.publish_stage()
        return response

    def on_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.demo.finished and self.planning is None:
            response.success, response.message = False, '任务已结束；需要重新执行时请 reset'
            return response
        self.running, self.stopped = False, True
        self.planning_cancel.set()
        if self.planning is None:
            self.demo.fail('用户停止：冻结仿真，必须重置后重新启动')
            self.publish_result()
        response.success, response.message = True, '已冻结仿真并取消后台规划，未完成结果不会执行'
        self.publish_stage()
        return response

    def on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.planning is not None or self.running:
            response.success, response.message = False, '请先停止，且等待后台规划结束后重置'
            return response
        if self.recorder is not None and self.recorder.active:
            result = self.demo.summary()
            result.update({'state': 'INTERRUPTED', 'ros_stage': 'RESET'})
            self.finish_recording(result)
        with self.viewer.lock() if self.viewer is not None else nullcontext():
            self.demo.reset()
        self.stopped = False
        # 清除上一轮锁存结果，防止新订阅者把旧成功结果当成本轮结果。
        self.result_pub.publish(text_message('{}'))
        self.publish_feedback()
        self.publish_markers()
        self.publish_stage()
        response.success, response.message = True, '整个仿真场景已恢复初值，状态 READY'
        return response

    def tick(self) -> None:
        if self.viewer is not None and not self.viewer.is_running():
            self.on_stop(Trigger.Request(), Trigger.Response())
            self.viewer = None
            self.get_logger().info('查看器已关闭；冻结当前仿真，ROS 节点仍可查询和重置。')
        if self.planning is not None:
            if not self.planning.done():
                return
            future, self.planning = self.planning, None
            try:
                planned = future.result()
            except Exception as exc:
                self.demo.fail('后台规划异常：' + str(exc))
                planned = False
            if self.stopped:
                self.demo.fail('规划期间用户停止，预检结果已丢弃')
            self.running = planned and not self.stopped
            if not self.running:
                self.publish_result()
        if self.running:
            try:
                # 每次最多推进 10 ms；调度变慢时仿真变慢，不补跑大段时间。
                with self.viewer.lock() if self.viewer is not None else nullcontext():
                    for _ in range(round(.01 / self.demo.config.timestep)):
                        self.demo.step()
                        if self.demo.finished:
                            break
            except Exception as exc:
                self.demo.fail('仿真执行异常：' + str(exc))
            if self.demo.finished:
                self.running = False
                self.publish_result()
        self.publish_feedback()
        self.record_sample()
        self.visual_ticks += 1
        if self.visual_ticks % 5 == 0:
            self.publish_markers()
        self.publish_stage()
        if self.viewer is not None:
            self.viewer.sync()

    def close_viewer(self) -> None:
        """先通知窗口退出，再等待渲染清理完成，避免 GLFW 与解释器退出竞争。"""
        self.resources.close()
        for thread in self.viewer_threads:
            thread.join()
        self.viewer_threads = ()
        self.viewer = None

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.running = False
        self.planning_cancel.set()
        self.timer.cancel()
        # 先取消，再等待当前有界计算退出，避免销毁仍被规划线程使用的模型。
        self.worker.shutdown(wait=True, cancel_futures=True)
        if self.recorder is not None and self.recorder.active:
            result = self.demo.summary()
            result.update({'state': 'INTERRUPTED', 'ros_stage': 'NODE_CLOSED'})
            self.finish_recording(result)
        self.close_viewer()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: GraspSimulationNode | None = None
    try:
        node = GraspSimulationNode()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
