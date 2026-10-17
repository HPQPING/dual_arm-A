# 检查工具

这些命令用于开发和回归，不会随普通任务启动自动执行。先运行：

```bash
source ~/archer_ws/install/setup.bash
```

| 用途 | 命令 |
|---|---|
| 单臂正逆运动学 | `/usr/bin/python3 -m archer_control.checks.check_kinematics` |
| 双臂配对轨迹 | `/usr/bin/python3 -m archer_control.checks.check_dual_plan` |
| 双臂画圆预检 | `/usr/bin/python3 -m archer_control.checks.check_circle` |
| 协同路径预检 | `/usr/bin/python3 -m archer_control.checks.check_cooperative` |
| 抓取记录验收 | `/usr/bin/python3 -m archer_control.checks.grasp_record_check /记录目录` |

原来根模块下的五条检查路径已迁移，不再保留重复入口。可在每条命令后添加 `--help` 查看参数。抓取记录仍使用原有 CSV 格式，无需转换。

回放命令保持 `/usr/bin/python3 -m archer_control.grasp_replay /记录目录`，内部会先调用新的记录验收模块。仅核对模型时添加 `--headless`。

公共工具链与碰撞检查在 `archer_control.model_checks`；参数检查在 `archer_control.parameter_validation`。普通运行程序使用公共模块，不应为了复用小函数反向依赖检查入口。记录回放调用记录验收属于明确的功能依赖。

这些工具不连接实机。离散预检与记录验收不代表连续碰撞、真实夹持力或实机安全验证。
