# archer_bringup：统一仿真、显示和真机启动

更新：2026-10-29。本文以当前源码、注册入口和实际安装检查为准。

本包只组织节点/进程、参数、资源和退出事件；运动算法在control/hardware，模型在description/mujoco。主任务只有画圆和简化协同搬运，辅助显示/单臂入口不表示新增主任务。

## 常用命令

### 仿真

```bash
ros2 launch archer_bringup dual_circle_sim.launch.py
ros2 launch archer_bringup dual_grasp_sim.launch.py
```

选择一个任务启动；准备就绪后另开已加载环境的终端分别调用：

```bash
ros2 service call /dual/task/start std_srvs/srv/Trigger '{}'
ros2 service call /dual/grasp/start std_srvs/srv/Trigger '{}'
```

第一条开始画圆，第二条开始搬运；不要在画圆后端等待抓取服务。两种launch均默认show_viewer:=true、show_rviz:=true；无窗口统一加show_viewer:=false show_rviz:=false。画圆默认record:=true；已单独运行记录器时可设false避免重复记录。

### 真机（显式执行后会运动）

```bash
ros2 launch archer_bringup dual_circle_real.launch.py execute:=true trajectory_file:="/转换目录/tracking.json"
ros2 launch archer_bringup dual_grasp_real.launch.py execute:=true trajectory_file:="/转换目录/state.json"
```

每次显式指定转换轨迹文件，不设默认路径，不在功能包内保存JSON轨迹。报告由调度器自动从同目录查找，不把_report.json当输入。原始记录迁移时支持run_dir，特殊配置可指定limits_file；配置字节需与报告绑定一致。

不加execute:=true仅打印未连接；执行许可和轨迹路径都会在创建控制节点前核对。console与launch调用同一任务调度器，不用另开hardware_position或旧缓存脚本。当前协同正式版211含沿用206反馈及两处时间优化；真机整合版平稳性仍待实际验证。

## 每份启动文件

| 文件 | 基本功能 |
|---|---|
| [dual_arm_view.launch.py](launch/dual_arm_view.launch.py) | 公共双臂TF、joint_state_publisher汇总、robot_state_publisher与RViz；由两份主仿真launch包含，也可独立显示。 |
| [dual_circle_real.launch.py](launch/dual_circle_real.launch.py) | 一份hardware_circle整轮调度器，显式execute和JSON/记录定位；默认不连接，不重生进程。 |
| [dual_circle_sim.launch.py](launch/dual_circle_sim.launch.py) | 主画圆仿真：共享MuJoCo后端、后台任务规划、自动记录及可选公共双臂显示；默认不启动运动。 |
| [dual_grasp_real.launch.py](launch/dual_grasp_real.launch.py) | 一份hardware_grasp正式211调度器，显式execute与state.json，处理夹爪、反馈修正和归位。 |
| [dual_grasp_sim.launch.py](launch/dual_grasp_sim.launch.py) | 主简化协同仿真：grasp_ros_node、完整工件、夹爪示意、进退1cm和归位；共用画圆启动姿态。 |
| [single_arm_display.launch.py](launch/single_arm_display.launch.py) | 单臂Xacro、滑块反馈、TF及RViz，供模型调试；不是MuJoCo动力学。 |
| [single_arm_sim.launch.py](launch/single_arm_sim.launch.py) | 单臂MuJoCo、反馈/轨迹接收、夹爪标记与可选RViz；供单臂任务联调。 |

## 参数与显示资源

| 文件 | 基本功能 |
|---|---|
| [config/dual_circle.yaml](config/dual_circle.yaml) | 共享双臂安装、仿真固定初角、画圆几何/速度/加速度和记录器参数；协同也复用其初角/显示安装。 |
| [rviz/dual_arm.rviz](rviz/dual_arm.rviz) | 双臂模型、TF、半圆/TCP轨迹等显示布局。 |
| [rviz/dual_grasp.rviz](rviz/dual_grasp.rviz) | 协同模型、工件/阶段标记等显示布局。 |

| 文件/目录 | 用途 |
|---|---|
| [package.xml](package.xml) | ROS依赖、包名与ament构建类型。 |
| [CMakeLists.txt](CMakeLists.txt) | 安装launch、config、rviz和README并注册启动回归。 |

主画圆支持params_file、python_executable、show_viewer、show_rviz、record；主协同支持python_executable、output_file、show_viewer、show_rviz、rviz_params_file。
output_file仅为显式额外结果JSON，不代替archer_results自动成功记录。rviz_params_file关闭RViz时仍提供共用初值核对，不能因此换回另一套仿真起点。

## 辅助调用与退出

```bash
ros2 launch archer_bringup single_arm_sim.launch.py side:=left
ros2 launch archer_bringup single_arm_display.launch.py side:=right
ros2 launch archer_bringup dual_arm_view.launch.py
```

公共显示不会生成真实反馈或启动运动；主仿真已默认包含它，无需再开相同显示。显示launch不从单臂任务节点导入参数工具，而直接使用纯公共校验模块。

关闭独立显示组只影响其自身；如果它由主launch包含，RViz退出发出的Shutdown会结束同一个launch服务的主任务进程。画圆关闭MuJoCo窗口通常结束整组；协同关闭查看器会冻结场景但可保留ROS服务，结束整组用launch终端Ctrl+C。真机任务Ctrl+C执行软件取消/BRAKE关闭，不故障归位。

## 测试文件

测试文件是开发回归，不随正常任务自动运行。功能回归与完整ament检查方法见项目根README；版权/许可证检查存在占位或跳过，不把功能测试通过写成全部发布检查通过。

| 文件 | 基本功能 |
|---|---|
| [test_circle_launch.py](test/test_circle_launch.py) | 验证两个任务的启动入口及旧名称兼容，不启动节点或连接设备. |
| [test_grasp_real_launch.py](test/test_grasp_real_launch.py) | 检查协同真机launch的显式许可和唯一调度节点，不执行任何节点。 |
| [test_real_launch.py](test/test_real_launch.py) | 检查画圆真机launch只启动一份正式入口，不执行节点或连接设备。 |
