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

## 控制流程清理

完整任务归位只使用常规轨迹执行器，不再保留停止后减速归位的第二套执行逻辑。
删除 StopAndReturnMove、return_to_initial、RETURNING 状态、线程 return 命令和
失去用途的 return_duration 参数；保留原始初值、名义限位内目标计算及越界起步收回检查。

归位记录只是同一次任务 CSV 的 phase=RETURN_HOME，不产生独立文件、节点或任务。
保留该字段是为了区分画圆段和接近/归位段，避免误算圆轨迹误差。
本轮不修改采样频率、结果文件数量、任务目录规则或故障联停与退出行为。

## CSV连续轨迹准备

CSV读取已支持在同一入口增加 `--prepare`：保留全部实际关节采样和顺序，用两臂共同时间比例放慢路径，准备位置、速度、加速度连续的版本二轨迹。非均匀采样间隔和静止区间保留；阶段边界及夹爪参考事件处设置停点，段内角度、峰值速度和加速度均进行解析检查。

```bash
source ~/archer_ws/install/setup.bash
ros2 run archer_hardware hardware_operator csv-import \
  --run-dir ~/archer_ws/archer_results/某次已完成任务目录 \
  --limits-file ~/archer_ws/src/archer_hardware/config/position_control.yaml \
  --prepare
```

成功后生成 `prepared_path.json` 和 `prepared_path_report.json`，不覆盖原CSV、摘要或之前的候选。也可以用 `--output`、`--report` 显式指定已存在目录中的新文件名。移除 `--prepare` 时继续沿用仅导入原采样的版本三行为。

准备结果依然标记 `execution_ready=false`：它尚未通过实际起点和完整几何检查；参考夹爪事件不会作为真实夹爪指令执行。现有版本一执行器继续拒绝版本二文件，不会偷偷退回逐点停车方式。连续执行通路需要显式许可，后续还需完整任务门禁和专用文件播放入口；连接机械臂前仍先由用户选择。

## 连续执行核心

底层新增版本二读取与执行核心：`prepare_continuous_pair` 在准备阶段重新解析左右时间、关节顺序、停止点及位置/速度/加速度，并检查整个多项式段内约束；生成只读的预检对象。`arm_continuous` 接收该对象，核对新鲜静止反馈与实际起点，等待现有双臂共同启动门控。

起点仅允许0.005 rad内的小差异；有差异时重建并重新检查首段。首段即使起点差符合容差，也可能因速度或加速度过大而拒绝。其余路径和时间不变，起点不匹配时不自动规划接近动作。

250/500Hz位置发送循环只查询当前段并计算目标，不逐点补发过期样本，也不重复做全程实根检查。使用原有位置模式、实际到位保持、故障取消和关闭清理，不发送速度模式、MIT或零力矩。速度和加速度字段只用于重建位置曲线。

上一轮打通的执行核心现已接入ROS后台预检与工作线程，但默认不开放版本二。完整几何检查、起点接近、夹爪事件和专用文件播放入口属于后续衔接，不能将底层接入通过测试理解为完整CSV任务已经可以在实机运行。

## 后台预检与双臂线程衔接

`/hardware/position/paired_target` 的回调仅提交字符串，JSON解析及整段约束检查交给单一后台计算进程（spawn方式启动）。通过后再次核对最新静止反馈和实际起点，暂存为STAGED；收到显式paired_move后，才把连续对象交给左右执行线程。两侧各自复核起点并通过共同启动门控，任一侧失败则取消双方。

新增只读参数 `allow_continuous_trajectory=false`，必须与既有成对许可及两臂已确认限位同时满足才允许版本二接入；默认不会连接、使能或执行。现有小幅位移限制不变，不会因版本二文件到达而扩大限制。

后台任务最多一个，不排队积压。跨进程返回的数值数组会在暂存前重新冻结；取消信号独立传递，故障时先请求两臂执行线程停止，再通知计算进程退出。预检未完成时收到新消息会作废旧结果并拒绝新请求，等待后台退出后需重新提交。预检期间禁止单臂运动和提前开始；halt、显式断开、驱动退出均作废预检结果。耗时检查逐段响应取消，超出30秒即取消；迟到成功也不能重新暂存。

状态话题提供VALIDATING、STAGED等阶段与preflight_busy。预检只检查关节约束，不代表任务几何与夹爪事件已通过；Linux线程及JSON解析也不构成硬实时停止保证。本轮只用替身驱动验证上述通路，没有连接机械臂。


## 完整任务文件只读核对

先使用已有 `csv-import --prepare` 生成连续候选，再使用 `task-review` 核对同一轮记录：

```bash
ros2 run archer_hardware hardware_operator task-review \
  --run-dir /path/to/archer_results/日期_时间_任务名称 \
  --limits-file ~/archer_ws/src/archer_hardware/config/position_control.yaml \
  --prepared-file /path/to/prepared_path.json \
  --report-file /path/to/prepared_path_report.json \
  --max-excursion 0.05
```

`--max-excursion` 是本次离线检查的明确关节位移上限，单位 rad，不修改驱动配置。完整任务若超过0.05会被拒绝；只有明确评估后才能为离线检查选择其他值，不能据此推导已允许驱动发送大范围轨迹。

入口只读取五种文件快照：CSV、成功摘要、限位配置、连续候选和准备报告。校验哈希对应、任务种类、原始实际角度逐点保留、统一时间伸长、两侧完整阶段及事件停顿，再复用连续多项式约束检查器验证全段角度、速度和加速度。仅接受画圆 `split_circle` 或完整圆柱抓取记录，不接受旧参考物协同示意。

结果仅打印至终端，不新增任务文件、不初始化ROS、不加载SDK或连接机械臂。通过时 `task_files_verified=true`，`execution_ready=false`。这不是文件签名或厂商认证：哈希只证明当前这些快照与准备报告一致，不能防止整套记录与报告被一起伪造。碰撞、工件真实性、实际起点和夹爪执行仍需后续验证。报告中的这些许可字段不作为本入口的信任依据。

该入口没有播放动作，也没有自动接近、使能、归位或夹爪命令。后续执行必须使用已核对的同一份内存快照，不能检查后再次读取并发送一个可能已变化的路径文件。
