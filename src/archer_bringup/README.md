# Archer 启动入口

常用任务只有两个，启动后均等待服务命令，不自动运动：

```bash
source /opt/ros/humble/setup.bash
source ~/archer_ws/install/setup.bash

# 画圆：接近、画左右半圆、返回本次起点。
ros2 launch archer_bringup dual_circle_sim.launch.py
# 另一个已source环境的终端开始画圆。
ros2 service call /dual/task/start std_srvs/srv/Trigger '{}'

# 完整圆柱抓取：另一次独立任务，不能与画圆仿真同时运行。
ros2 launch archer_bringup dual_grasp_sim.launch.py show_viewer:=true show_rviz:=true
# 另一个已source环境的终端开始抓取。
ros2 service call /dual/grasp/start std_srvs/srv/Trigger '{}'
```

旧画圆启动名称已移除，请统一使用 `dual_circle_sim.launch.py`。新入口不显示 `task_type` 任务选择，旧 `cooperative` 参数或配置会明确拒绝，并提示完整抓取入口。内部早期参考物协同规划代码保留，不作为常用任务入口。

`dual_arm_view.launch.py` 只启动双臂显示、TF和反馈汇总；它没有运动规划。`single_arm_display.launch.py`、`single_arm_sim.launch.py` 用于单臂调试，保留供开发排查。

只允许同时运行一套仿真场景，避免重复发布左右关节反馈。以上仿真入口不启动实机SDK。
