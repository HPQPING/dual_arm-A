"""为独立抓取仿真提供 ROS 服务与关节反馈，不连接实机。

暂停和停止均冻结整个仿真世界；这不是实机急停或带重力的物理保持。
规划在后台进行，完成预检后才允许推进仿真。
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
from pathlib import Path

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import String
from std_srvs.srv import Trigger

from .dual_grasp_scene import DualGraspDemo


def text_message(value: str) -> String:
    message = String()
    message.data = value
    return message


class GraspSimulationNode(Node):
    def __init__(self) -> None:
        super().__init__('dual_grasp_sim')
        self.declare_parameter('output_file', '')
        if not self.has_parameter('use_sim_time'):
            self.declare_parameter('use_sim_time', False)
        if self.get_parameter('use_sim_time').value is not False:
            raise ValueError('本节点发布墙钟时间戳，use_sim_time 必须为 false')
        output = self.get_parameter('output_file').value
        if not isinstance(output, str):
            raise ValueError('output_file 必须为字符串')
        self.output = Path(output).expanduser() if output else None
        self.demo = DualGraspDemo()
        self.running = False
        self.stopped = False
        self.planning: Future[bool] | None = None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='grasp_plan')
        self.last_stage = ''
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.stage_pub = self.create_publisher(String, '/dual/grasp/stage', qos)
        self.result_pub = self.create_publisher(String, '/dual/grasp/result', qos)
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
        self.publish_stage()
        self.get_logger().info('抓取仿真 READY；调用 /dual/grasp/start 开始，启动节点不会自动运动。')

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
            self.stage_pub.publish(text_message(stage))
            self.get_logger().info('抓取阶段：' + stage)
            self.last_stage = stage

    def publish_feedback(self) -> None:
        # 后台规划也会读取模型；规划期间不并发读取或发布它的可变数据。
        if self.planning is not None:
            return
        stamp = self.get_clock().now().to_msg()
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
        text = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2)
        self.result_pub.publish(text_message(text))
        if self.output is not None:
            try:
                self.output.parent.mkdir(parents=True, exist_ok=True)
                self.output.write_text(text + '\n', encoding='utf-8')
            except OSError as exc:
                self.get_logger().error('结果已发布，但文件保存失败：' + str(exc))

    def on_start(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.planning is not None or self.stopped or self.demo.state != 'READY':
            response.success, response.message = False, '只允许从 READY 启动；停止或失败后请 reset'
            return response
        self.planning = self.worker.submit(self.demo.start)
        response.success, response.message = True, '已接受，正在后台规划；通过 stage/result 查询最终结果'
        self.publish_stage()
        return response

    def on_pause(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = self.running and not self.stopped
        response.message = '整个仿真世界已冻结' if response.success else '当前没有正在执行的任务'
        if response.success:
            self.running = False
        self.publish_stage()
        return response

    def on_resume(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        response.success = self.stage() == 'PAUSED'
        response.message = '继续仿真' if response.success else '仅允许从 PAUSED 继续'
        if response.success:
            self.running = True
        self.publish_stage()
        return response

    def on_stop(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.demo.finished and self.planning is None:
            response.success, response.message = False, '任务已结束；需要重新执行时请 reset'
            return response
        self.running, self.stopped = False, True
        if self.planning is None:
            self.demo.fail('用户停止：冻结仿真，必须重置后重新启动')
            self.publish_result()
        response.success, response.message = True, '已禁止推进仿真；后台预检若未结束，将结束后丢弃结果'
        self.publish_stage()
        return response

    def on_reset(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        if self.planning is not None or self.running:
            response.success, response.message = False, '请先停止，且等待后台规划结束后重置'
            return response
        self.demo.reset()
        self.stopped = False
        # 清除上一轮锁存结果，防止新订阅者把旧成功结果当成本轮结果。
        self.result_pub.publish(text_message('{}'))
        self.publish_feedback()
        self.publish_stage()
        response.success, response.message = True, '整个仿真场景已恢复初值，状态 READY'
        return response

    def tick(self) -> None:
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
        self.publish_stage()

    def close(self) -> None:
        self.running = False
        self.timer.cancel()
        # 等待后台只读规划结束后再销毁模型和 ROS 节点。
        self.worker.shutdown(wait=True, cancel_futures=True)


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
