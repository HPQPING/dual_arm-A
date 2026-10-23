# Archer 启动入口

常用任务只有两个，启动后均等待服务命令，不自动运动：

```bash
source /opt/ros/humble/setup.bash
source ~/archer_ws/install/setup.bash

# 画圆：默认同时显示MuJoCo和RViz，记录开启；启动后等待开始服务。
ros2 launch archer_bringup dual_circle_sim.launch.py
# 另一个已source环境的终端开始画圆。
ros2 service call /dual/task/start std_srvs/srv/Trigger '{}'

# 完整圆柱抓取：另一次独立任务，不能与画圆仿真同时运行。
ros2 launch archer_bringup dual_grasp_sim.launch.py
# 另一个已source环境的终端开始抓取。
ros2 service call /dual/grasp/start std_srvs/srv/Trigger '{}'
```

旧画圆启动名称已移除，请统一使用 `dual_circle_sim.launch.py`。新入口不显示 `task_type` 任务选择，旧 `cooperative` 参数或配置会明确拒绝，并提示完整抓取入口。内部早期参考物协同规划代码保留，不作为常用任务入口。

两个任务的 `show_rviz` 均默认true，自动包含同一个 `dual_arm_view.launch.py`，不需再另开显示launch。
该公共文件只负责模型、TF、实际反馈汇总与RViz，没有运动规划，保留供两个任务复用。
关闭RViz可加 `show_rviz:=false`。完全无窗口时同时加 `show_viewer:=false show_rviz:=false`。
两个任务的MuJoCo窗口也均默认开启；其余默认值和开始服务不变。
`single_arm_display.launch.py`、`single_arm_sim.launch.py` 用于单臂调试，保留供开发排查。

只允许同时运行一套仿真场景，避免重复发布左右关节反馈。以上仿真入口不启动实机SDK。

## 画圆仿真初值

`config/dual_circle.yaml`的`dual_arm_scene.ros__parameters.left_initial_joints`和
`right_initial_joints`已更新为2026-10-20 21:29只读采集的实际静止角度，未裁剪或改写。
每组按J1到J6填写，单位rad；本次两臂均在名义限位内。
它们是固定的仿真参数，启动仿真不会重新连接真机。实机姿态改变后，需要重新取得实际角度再更新。

完整画圆仍按“读取仿真反馈起点→接近→左右半圆→自动归位→COMPLETED”执行。
沿用上面的一个launch命令和开始服务即可，成功后只在archer_results下生成一份新的日期时间任务目录。
画圆的新初值不修改独立抓取任务的配置，也不改变真机驱动的实时读取规则。


## 166：画圆上游的归位感知逆解选择

`archer_control/circle_planner.py`保留相同圆弧点、工具姿态和随机种子，
收集去重后能够连续完成圆弧的IK分支，比较左右臂配对的接近、圆弧、归位代价。
代价使用当前速度配置和共同五次段时长估算；只有各阶段代价均不变差且总成本
有实际下降（或同等时间下关节总路程改善）时才改选，否则保留原首条可行分支。
`plan_circle(..., return_aware=False)`供离线旧算法对照；正常调用默认启用整轮选择。
任务节点传入实际`speed_limits`，不再让候选排序和初始关节规划隐式使用默认速度。
不会直接修改J6目标、工具旋转、限位、夹爪、硬件配置或已有CSV/JSON。

参考MoveIt Task Constructor的ComputeIK多解去重及SerialContainer完整任务代价思想：
<https://moveit.picknik.ai/main/doc/concepts/moveit_task_constructor/wrappers.html>
<https://moveit.picknik.ai/main/doc/concepts/moveit_task_constructor/moveit_task_constructor.html>
这里使用项目自身IK与代价比较，未集成MTC求解器；不保证找全分支或全局最优。
最终连续轨迹的V/A和跨臂离散间隙仍由既有完整路径检查复核，候选代价不是实际耗时承诺。

使用当前dual_circle.yaml初值、相同32次随机尝试离线对比：两臂各找到一个可行分支，
新旧关节节点完全相同；完整上游仿真计划71.2208秒，其中接近14.2504、归位14.2505秒。
TCP节点位置误差最大0.0592mm、姿态误差0.000193rad，完整跨臂离散预检通过。
因此当前上游尚无分支提速收益，不能把下游归位较慢主要归因于J6大幅度。
这些时间是上游计划值，不是记录CSV重定时后的真机执行时间。
原启动命令保持不变；既有158/164缓存文件不会因本次源码变更而自动改变。
