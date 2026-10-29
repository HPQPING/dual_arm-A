# archer_control：运动规划、仿真任务与记录

更新：2026-10-29。本文以当前源码、注册入口和实际安装检查为准。

## 两个主任务

双臂画圆在共享MuJoCo场景中由左右臂分别执行半圆；简化协同搬运在模型中展示完整工件，按接近、J6调姿、夹爪事件、进给、抬升/搬运/放回、退开和归位执行。该包不连接hex_device，不直接向真机发送指令。

```bash
ros2 launch archer_bringup dual_circle_sim.launch.py
ros2 service call /dual/task/start std_srvs/srv/Trigger '{}'
```

另一任务使用独立启动/服务：

```bash
ros2 launch archer_bringup dual_grasp_sim.launch.py
ros2 service call /dual/grasp/start std_srvs/srv/Trigger '{}'
```

先启动launch，另一个已加载环境的终端等待READY后调用服务。两者默认打开MuJoCo和RViz，不在启动时自动运动。画圆服务和抓取服务不是同一个名字。

## 文件与职责

| 文件 | 基本功能 |
|---|---|
| [__init__.py](archer_control/__init__.py) | 声明Python包，不启动节点或任务。 |
| [circle_planner.py](archer_control/circle_planner.py) | 离线规划双臂半圆轨迹及夹爪内倾姿态，不访问 ROS 接口或实机. |
| [circle_sim.py](archer_control/circle_sim.py) | 独立共享MuJoCo画圆验证程序；可无ROS运行，不是硬件入口。 |
| [circle_visuals.py](archer_control/circle_visuals.py) | 在 MuJoCo 查看器中绘制参考半圆和实际 TCP 轨迹. 本模块文件名必须为 circle_visuals.py，与 circle_sim.py 的导入一致。 |
| [continuous_trajectory.py](archer_control/continuous_trajectory.py) | 连续五次轨迹、共享时间重分配、圆弧/接近归位平滑和最终导数检查；离线规划数学。 |
| [cooperative_planner.py](archer_control/cooperative_planner.py) | 按物体位姿和左右抓持变换计算共同路径、相对误差；仍被当前简化规划与辅助检查调用。 |
| [dual_arm_node.py](archer_control/dual_arm_node.py) | 共享仿真的ROS任务节点：后台规划、预览、共同时间下发、起点反馈与执行状态；标准画圆launch只选择split_circle。 |
| [dual_arm_trajectory.py](archer_control/dual_arm_trajectory.py) | 双臂轨迹规划与文件导出，不访问硬件或发送 ROS 指令. 世界坐标与基座变换由调用者传入；共享时间轴不代表硬件同步启动。 |
| [dual_grasp_scene.py](archer_control/dual_grasp_scene.py) | 辅助完整关节接触抓持演示及当前简化场景的共用父类；不是主dual_grasp_sim入口，不能整删。 |
| [dual_trace_node.py](archer_control/dual_trace_node.py) | 画圆及辅助双臂演示的只读记录器；配对新鲜反馈、计算TCP/误差，完成后发布成功任务目录。 |
| [grasp_record_format.py](archer_control/grasp_record_format.py) | 协同CSV字段及统一简化规则常量：5.65闭合、5.5任务开口、进退0.01m，无5秒等待。 |
| [grasp_recording.py](archer_control/grasp_recording.py) | 同步记录抓取仿真的实际状态，不发布控制命令，不连接实机. |
| [grasp_replay.py](archer_control/grasp_replay.py) | 在独立 MuJoCo 场景中回放抓取记录，不启动 ROS，不执行动力学或控制指令. |
| [grasp_ros_node.py](archer_control/grasp_ros_node.py) | 为独立抓取仿真提供 ROS 服务与关节反馈，不连接实机. 暂停和停止均冻结整个仿真世界；这不是实机急停或带重力的物理保持。 |
| [grasp_task_planner.py](archer_control/grasp_task_planner.py) | 从反馈关节快照规划接近、上下协同搬运、张开退开及返回起点. |
| [kinematics.py](archer_control/kinematics.py) | 从URDF建立六轴正运动学、雅可比矩阵与阻尼最小二乘逆运动学模型. 位姿均相对于对应机械臂的base_link表达，不是world。 |
| [model_checks.py](archer_control/model_checks.py) | 公共工具链校验与离散跨臂间隙检查，不执行任务或发送运动指令. |
| [parameter_validation.py](archer_control/parameter_validation.py) | 公共数值参数检查，不导入 ROS 节点或启动任务. |
| [result_paths.py](archer_control/result_paths.py) | 任务开始时生成日期/时间/微秒/名称；临时采样只有成功才发布，失败不保留正式任务文件夹。 |
| [simple_grasp_planner.py](archer_control/simple_grasp_planner.py) | 当前协同关节/笛卡尔路径及分阶段计划；复用旧规划器的数据结构和几何工具。 |
| [simple_grasp_scene.py](archer_control/simple_grasp_scene.py) | 当前简化协同演示：读取仿真关节状态，执行夹爪5.65/5.5、进退1cm和归位；工件采用明确的演示附着。 |
| [single_arm_node.py](archer_control/single_arm_node.py) | 提供单臂关节和采样笛卡尔仿真任务，通过ROS接口发布任务轨迹. |
| [trajectory.py](archer_control/trajectory.py) | 规划分段五次关节轨迹并按需转换ROS消息，不访问硬件. 单位为弧度、弧度每秒和秒；支持普通停车点和带连续导数的路径。 |
| [transit_timing.py](archer_control/transit_timing.py) | 离线优化画圆过渡时轴和同步归位，保留画圆及衔接窗口，不连接设备. |

### 开发检查目录

| 文件 | 基本功能 |
|---|---|
| [__init__.py](archer_control/checks/__init__.py) | 声明Python包，不启动节点或任务。 |
| [check_circle.py](archer_control/checks/check_circle.py) | 离线预检双臂圆轨迹，不发布运动指令，不连接实机. 模型应已将 link_6 → flange 设置为局部 +Z 8 mm；法兰 → 夹爪零平移。 |
| [check_continuous_circle.py](archer_control/checks/check_continuous_circle.py) | 从版本一逐点计划离线生成并复核版本二连续画圆候选，不连接真机。 |
| [check_cooperative.py](archer_control/checks/check_cooperative.py) | 离线演示同一物体上、下两处抓取关系，不发布轨迹，不控制夹爪或实机. |
| [check_dual_plan.py](archer_control/checks/check_dual_plan.py) | 离线检查双臂配对规划，不发布轨迹或连接机械臂. 运行：python3 -m archer_control.checks.check_dual_plan。 |
| [check_kinematics.py](archer_control/checks/check_kinematics.py) | 离线检查正运动学、雅可比矩阵和逆运动学，不发布机械臂控制指令. |
| [grasp_record_check.py](archer_control/checks/grasp_record_check.py) | 独立验收完整抓取记录；只读取文件，不启动 ROS 或发送动作. |

[checks/README.md](archer_control/checks/README.md)列出全部独立检查命令。主要数学模块被任务共用，不应为了复用小函数反向依赖开发检查入口。

### 配置与包元数据

| 文件 | 用途 |
|---|---|
| [config/tasks.yaml](config/tasks.yaml) | 独立单臂关节/笛卡尔/小圆示例。不是双臂主任务配置。 |

| 文件/目录 | 用途 |
|---|---|
| [package.xml](package.xml) | ROS依赖、包名与ament构建类型。 |
| [setup.py](setup.py) | 注册Python可执行入口并安装配置和README。 |
| [setup.cfg](setup.cfg) | 把console脚本安装到ROS包的lib目录。 |
| resource/archer_control | ament索引标记，不是执行程序。 |

## ROS接口

| 入口/接口 | 功能 |
|---|---|
| `/dual/task/start`、`/dual/task/stop` | 共享画圆任务开始/仿真停止。 |
| `/dual/task/status`、`stage`、`profile` | 当前执行状态、阶段和规划时间/来源信息。 |
| `/left/joint_trajectory`、`/right/joint_trajectory` | 共享仿真执行端接收的关节轨迹；不是发布joint_states控制真机。 |
| `/dual/grasp/start`、`pause`、`resume`、`stop`、`reset` | 简化协同仿真生命周期。 |
| `/dual/grasp/stage`、`result`、`markers`、`gripper_command` | 协同阶段、结果、工件标记及夹爪示意指令。 |
| `/left/joint_states`、`/right/joint_states` | 仿真状态反馈，供记录与显示。 |

协同reset/暂停是仿真功能，不是硬件复位。画圆和协同任务不能在同一套话题上同时运行。

## 记录与回放

成功画圆目录含tracking.csv和summary.json；协同含state.csv、events.csv、summary.json。文件放archer_results/开始日期_时间_微秒_任务名称；失败、取消的临时记录不发布。基础记录不自动生成大量图片。

```bash
/usr/bin/python3 -m archer_control.checks.grasp_record_check "/原始协同记录目录"
/usr/bin/python3 -m archer_control.grasp_replay "/原始协同记录目录"
/usr/bin/python3 -m archer_control.grasp_replay "/原始协同记录目录" --headless
```

回放只重建可见记录，不模拟SDK、真实夹持或重做动力学精度验收。CSV的仿真时间/参考/实际位置、输入值不能任意互换为真机指令。

## 辅助入口与保留理由

`ros2 run archer_control single_arm_task`用于单臂仿真调试，默认不运动；需先有对应单臂仿真反馈。`ros2 run archer_control dual_grasp_scene --help`是辅助接触抓持示例，不是主简化任务。`/usr/bin/python3 -m archer_control.simple_grasp_scene --headless`是独立无ROS验证，按模块实现运行，不连接真机。

旧接触链路仍提供父类、配置协议、路径数据结构和模型加载器；将其整删会破坏主简化任务。单臂仿真同样是双臂执行端的基类来源，不按文件名“single”就删除。公共数学保留清晰职责，不把所有规划器合成一个巨型模块。

## 测试文件

测试文件是开发回归，不随正常任务自动运行。功能回归与完整ament检查方法见项目根README；版权/许可证检查存在占位或跳过，不把功能测试通过写成全部发布检查通过。

| 文件 | 基本功能 |
|---|---|
| [test_circle_branches.py](test/test_circle_branches.py) | 验证画圆多逆解的整轮配对选择、去重与取消，不连接机械臂。 |
| [test_continuous_trajectory.py](test/test_continuous_trajectory.py) | 离线验证连续插值、整段约束及候选格式，不连接设备. |
| [test_copyright.py](test/test_copyright.py) | 检查功能包源文件的版权声明格式，不启动节点或连接设备. |
| [test_flake8.py](test/test_flake8.py) | 检查功能包Python代码的格式和静态风格，不启动节点或连接设备. |
| [test_grasp_shutdown.py](test/test_grasp_shutdown.py) | 验证抓取仿真取消规划和窗口线程清理，不连接实机. |
| [test_pep257.py](test/test_pep257.py) | 检查功能包Python文档字符串是否符合PEP257，不启动任务或连接设备. |
| [test_recorded_timing.py](test/test_recorded_timing.py) | 检查采样轨迹的分段时间、连续插值及数据保留，不依赖ROS或机械臂. |
| [test_result_paths.py](test/test_result_paths.py) | 验证任务结果分目录与抓取记录，使用合成状态，不启动仿真或连接真机. |
| [test_simple_grasp_rule.py](test/test_simple_grasp_rule.py) | 检查协同开爪后直接进给、共享初值及归位前闭合，不连接设备。 |
| [test_task_cycle.py](test/test_task_cycle.py) | 检查任务归位的共享时间轴、限位及输入不变性，不连接机械臂. |
| [test_trace_pairing.py](test/test_trace_pairing.py) | 验证画圆记录只写完整反馈对，不删改已经写入的历史记录. |
| [test_transit_timing.py](test/test_transit_timing.py) | 离线核对接近归位提速、边界连续和圆弧保留，不连接设备. |
