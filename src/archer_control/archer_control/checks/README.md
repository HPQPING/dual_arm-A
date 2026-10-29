# 开发检查工具

更新：2026-10-29。这些入口不随普通任务启动自动运行，也不连接真实机械臂。先加载 `/opt/ros/humble/setup.bash` 和 `~/archer_ws/install/setup.bash`。

| 文件 | 基本功能 | 调用 |
|---|---|---|
| check_kinematics.py | 单臂正逆运动学与关节范围核对 | `/usr/bin/python3 -m archer_control.checks.check_kinematics` |
| check_dual_plan.py | 双臂配对计划、公共时间与间隙检查 | `/usr/bin/python3 -m archer_control.checks.check_dual_plan` |
| check_circle.py | 画圆几何、安装与分支/间隙预检 | `/usr/bin/python3 -m archer_control.checks.check_circle` |
| check_cooperative.py | 辅助物体路径与双臂相对抓持误差核对 | `/usr/bin/python3 -m archer_control.checks.check_cooperative` |
| check_continuous_circle.py | 连续画圆候选、导数及采样几何检查；需指定输入/输出参数 | `/usr/bin/python3 -m archer_control.checks.check_continuous_circle --help` |
| grasp_record_check.py | 已完成协同记录的阶段/状态/事件一致性 | `/usr/bin/python3 -m archer_control.checks.grasp_record_check "/记录目录"` |
| __init__.py | 声明开发检查子包，不执行检查 | 无需直接运行 |

具体参数在每条命令后加 `--help` 查看。需要输出时显式指定地址；开发结果不当作archer_results里的生产任务记录。
记录回放使用 `/usr/bin/python3 -m archer_control.grasp_replay "/记录目录"`；加 `--headless` 只核对模型。

公共几何工具在 `archer_control.model_checks`，数值校验在 `archer_control.parameter_validation`。普通任务使用这些公共模块，不为了复用函数反向依赖检查程序。旧根模块检查路径已迁移，不保留重复启动入口。

离散几何或记录检查不证明连续碰撞、实际夹持力或实机轨迹精度；真机仍需正式文件核对、实际反馈及执行流程。
