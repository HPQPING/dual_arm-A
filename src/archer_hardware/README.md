# 真机接口与完整任务流程

更新日期：2026-10-18。

正常任务采用“启动 → 接近 → 任务动作 → 计划内归位 → 保持”。归位属于同一份完整任务计划，执行前一并检查。协同抓取仿真已有释放、退开及归位；画圆仿真补入共享时间轴归位段。底层单点 move 或任意关节轨迹只执行给定目标，不会自行猜测和追加归位。

## 退出与故障停止

无需启动键盘监听。退出位置驱动时，在**驱动所在前台终端**按 Ctrl+C：先取消两臂的活动请求，再等待执行线程关闭 SDK 连接，最后退出 ROS。不中途归位，不发送零力矩或 MIT 命令。现有适配器关闭时会尽力 BRAKE 后断开；这里没有新增失能指令。按用户确认的设备特性，断连后由设备自锁。

Ctrl+C 只作用于对应终端的进程；退出 RViz、状态客户端或只读报告不等于关闭另一个终端的位置驱动。强杀进程、断电与网络故障不保证软件清理完成。

移除 `hardware_operator keyboard`、`hardware_operator stop` 及 `/hardware/position/stop`。不再通过停止命令发起独立归位。保留全局 `/hardware/position/halt` 供程序故障联停：取消运动、作废目标、本次运行不可恢复；halt 服务本身保留反馈连接，与退出驱动不同。没有 reset/resume 入口。

## 目录与基本功能

```text
archer_hardware/
├── archer_hardware/
│   ├── hex_adapter.py          # SDK依赖、只读反馈与状态解析
│   ├── driver_node.py          # 左右臂只读反馈
│   ├── position_plan.py        # 关节限位、位置轨迹和公共校验
│   ├── position_adapter.py     # SDK位置模式执行与关闭清理
│   ├── position_worker.py      # 每臂执行线程、取消和状态快照
│   ├── position_node.py        # ROS目标暂存、显式执行与故障联停
│   ├── operator_client.py      # 显式操作入口，无专用键盘监听
│   ├── commissioning.py        # 联调报告与离线计划检查
│   └── csv_import.py           # 成功任务CSV转为待检查的候选路径
├── config/
├── test/                      # 离线替身与开发检查
├── setup.py
└── package.xml
```

## 不连接设备的查询命令

```bash
source /opt/ros/humble/setup.bash
source ~/archer_ws/install/setup.bash
ros2 run archer_hardware hardware_operator --help
ros2 run archer_hardware hardware_operator status --timeout 5
ros2 run archer_hardware hardware_operator csv-import --help
ros2 run archer_hardware hardware_operator plan-check --help
```

status 只订阅已存在驱动的状态，不自动连接或使能。hex_device 仍是必要依赖，适配器没有替代官方 SDK。

## 当前边界

连接、使能、单点及成对位置轨迹接口已有离线替身检查。CSV 导入生成候选数据，不代表现有版本可以直接播放真实任务；连续轨迹执行、实际起点接入和整条路径的实机验收仍需后续完成。仿真与离线检查不等同于实机测试。

再次连接机械臂前先由用户选择；本轮只通过 SSH 修改 Ubuntu 代码，没有建立机械臂连接或执行动作。
