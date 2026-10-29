# archer_mujoco：模型加载与MuJoCo执行端

更新：2026-10-29。本文以当前源码、注册入口和实际安装检查为准。

## 职责与环境

本包加载archer_description的URDF/Xacro和网格，构建单臂或双臂共享MuJoCo模型，接收关节轨迹并发布仿真反馈。使用能同时导入rclpy与mujoco的解释器；MuJoCo由当前环境提供，不在setup.py中擅自升级或替换。不存在真机SDK连接。

常规入口由bringup组合：

```bash
ros2 launch archer_bringup dual_circle_sim.launch.py
ros2 launch archer_bringup dual_grasp_sim.launch.py
```

两者是不同任务，选择其一。先启动后调用对应start服务，具体命令见根README。无窗口时加show_viewer:=false show_rviz:=false。

## 每份实现文件

| 文件 | 基本功能 |
|---|---|
| [__init__.py](archer_mujoco/__init__.py) | 声明Python包，不启动节点或任务。 |
| [check_grasp.py](archer_mujoco/check_grasp.py) | 无窗口验证同一圆柱的上下夹持、抬升、平移及释放；不发布运动指令. |
| [dual_model_loader.py](archer_mujoco/dual_model_loader.py) | 从现有单臂URDF组合十二关节MuJoCo场景，不访问硬件. GP80仍为刚性网格，TCP使用当前配置的候选值。 |
| [dual_sim_node.py](archer_mujoco/dual_sim_node.py) | 共享双臂ROS执行端，复用SingleArmSim的关节逻辑；同一物理模型推进并发布左右状态。 |
| [grasp_model_loader.py](archer_mujoco/grasp_model_loader.py) | 把完整双臂、圆柱与台面加入同一场景；供完整关节接触演示、简化场景父类及记录回放使用。 |
| [grasp_scene.py](archer_mujoco/grasp_scene.py) | 独立简化指腹/圆柱GraspBench接触验证台；公共配置和查看器协议还被完整任务共用。 |
| [gripper_markers.py](archer_mujoco/gripper_markers.py) | 在RViz发布GP80指尖坐标标记，不修改夹爪网格或发送夹爪控制命令. |
| [joint_interpolation.py](archer_mujoco/joint_interpolation.py) | 统一五次系数、Horner取样和解析极值；硬件及离线规划也复用，不在SDK周期中依赖ROS。 |
| [model_loader.py](archer_mujoco/model_loader.py) | 加载 Archer-Y6 单臂模型到 MuJoCo，不访问实机. |
| [pose_check.py](archer_mujoco/pose_check.py) | 静态打印/显示左右安装与TCP坐标轴，使用模块内参考姿态，不能当实时真机初值。 |
| [sim_node.py](archer_mujoco/sim_node.py) | 单臂ROS执行、连续五次插值与反馈；是双臂SceneArmEndpoint的父实现，不能整删。 |

## 注册入口及调用

| console入口 | 使用方式/目的 |
|---|---|
| sim_node | `ros2 run archer_mujoco sim_node`，单臂后端，普通使用single_arm_sim.launch.py。 |
| dual_sim_node | `ros2 run archer_mujoco dual_sim_node`，双臂后端，普通使用dual_circle_sim.launch.py。 |
| gripper_markers | `ros2 run archer_mujoco gripper_markers`，发布工具坐标系下的指尖/TCP标记，由RViz按TF显示，不驱动模型。 |
| pose_check | `ros2 run archer_mujoco pose_check --no-viewer`，静态姿态核对。 |
| check_grasp | `ros2 run archer_mujoco check_grasp --diameter-mm 10 --mass-g 50`，辅助接触验证，非主协同任务。 |
| grasp_scene | `ros2 run archer_mujoco grasp_scene --help`，独立GraspBench参数帮助。 |

这些是可独立调试的辅助入口；不是在主任务之外另加生产任务。不需要同时启动相同后端。

## 配置与资源

[config/simulation.yaml](config/simulation.yaml)是单臂示例：2ms物理步长、状态/显示频率、仿真PD增益、接触开关及TCP。双臂主画圆配置在bringup/config/dual_circle.yaml。仿真增益和物理步长不是实机电机参数；500Hz物理仿真不表示SDK必须500Hz。

模型网格位于archer_description/meshes；本包原models目录为空，已取消无效安装声明，实际仅安装config目录中的YAML。不删除description的STL或公共加载器。旧接触台的几何与主简化演示的工件附着规则不同，README不把二者当成相同夹持物理模型。

| 文件/目录 | 用途 |
|---|---|
| [package.xml](package.xml) | ROS依赖、包名与ament构建类型。 |
| [setup.py](setup.py) | 注册Python可执行入口并安装配置和README。 |
| [setup.cfg](setup.cfg) | 把console脚本安装到ROS包的lib目录。 |
| resource/archer_mujoco | ament索引标记，不是执行程序。 |

## 退出语义

单臂/双臂画圆MuJoCo窗口关闭会触发对应仿真主循环退出，主launch结束同组节点。简化协同的查看器由control/grasp_ros_node管理：关闭窗口会冻结并停止当前仿真任务，ROS服务仍可保留查询/重置；要结束整组仍在该launch终端按Ctrl+C。后台规划取消、渲染线程与executor的清理职责不同，不能只因退出代码相似强行合并。

## 测试文件

测试文件是开发回归，不随正常任务自动运行。功能回归与完整ament检查方法见项目根README；版权/许可证检查存在占位或跳过，不把功能测试通过写成全部发布检查通过。

| 文件 | 基本功能 |
|---|---|
| [test_copyright.py](test/test_copyright.py) | 检查功能包源文件的版权声明格式，不启动节点或连接设备. |
| [test_flake8.py](test/test_flake8.py) | 检查功能包Python代码的格式和静态风格，不启动节点或连接设备. |
| [test_joint_interpolation.py](test/test_joint_interpolation.py) | 核对高频五次取样优化的数值等价性，不启动仿真或连接设备。 |
| [test_pep257.py](test/test_pep257.py) | 检查功能包Python文档字符串是否符合PEP257，不启动任务或连接设备. |
