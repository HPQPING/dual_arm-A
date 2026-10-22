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
