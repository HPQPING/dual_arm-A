# 真机接口与完整任务流程

更新日期：2026-10-19。

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

连接、使能、单点及成对位置轨迹接口已有离线替身检查。CSV 导入生成候选数据，连续执行及实际起点衔接已有默认关闭的画圆文件入口；整条路径的实机验收尚未完成。抓取文件只做离线检查，真实夹爪尚未接入。仿真与离线检查不等同于实机测试。

只读反馈验收报告增加逐关节最小值、最大值、变化范围、最大相邻变化量和变化次数。
`positions_exactly_constant` 只说明本次窗口内至少两帧读数是否逐位相同；无样本或单帧时为null。
这些统计不自动放宽起点或限位，不代表设备可控，也不能用一次不变的观察保证以后没有抖动。
统计包括结构与时间合法的超限读数以保留证据；坏帧仍判失败。报告频率是收到反馈的频率，
不是250/500Hz控制目标发送频率。

反馈报告的 `startup_review` 单独列出是否需要收回名义限位、准备位置候选和本轮原始初值对应的
限位内归位候选。准备候选复用已有 `recovery_goal`：只改变超限关节，并在边界内留出0.005rad；
归位候选复用 `return_goal`，超限初值对应名义边界，实际初始记录保持原值。
存在坏帧或没有反馈时不给位置候选；无论只读窗口是否通过，`execution_ready`始终为false。
候选不是已检查的可执行轨迹，不会自动使能、收回或执行。轻微抖动也不触发反馈裁剪或四舍五入。

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

准备结果依然标记 `execution_ready=false`：它尚未通过实际起点和完整几何检查；参考夹爪事件不会作为真实夹爪指令执行。版本二使用独立的连续执行核心，不偷偷退回逐点停车方式。画圆完整文件的专用入口见本文末尾，默认关闭；抓取文件尚不能进入真实执行。连接机械臂前仍先由用户选择。

## 连续执行核心

底层新增版本二读取与执行核心：`prepare_continuous_pair` 在准备阶段重新解析左右时间、关节顺序、停止点及位置/速度/加速度，并检查整个多项式段内约束；生成只读的预检对象。`arm_continuous` 接收该对象，核对新鲜静止反馈与实际起点，等待现有双臂共同启动门控。

普通版本二关节候选允许0.005 rad内的小差异，有差异时重建并重新检查首段；首段仍可能因速度或加速度超限而拒绝。经过几何检查的完整文件使用 `preserve_path=true`，要求实际起点精确匹配，禁止套用上述首段重建逻辑；变化后必须重新规划和检查。

250/500Hz位置发送循环只查询当前段并计算目标，不逐点补发过期样本，也不重复做全程实根检查。使用原有位置模式、实际到位保持、故障取消和关闭清理，不发送速度模式、MIT或零力矩。速度和加速度字段只用于重建位置曲线。

连续执行核心和画圆完整文件已接入ROS后台预检与工作线程，但默认不开放。只用替身测试打通通道，不能理解为已经完成实机联调或真实抓取。

## 后台预检与双臂线程衔接

`/hardware/position/paired_target` 的回调仅提交字符串，JSON解析及整段约束检查交给单一后台计算进程（spawn方式启动）。通过后再次核对最新静止反馈和实际起点，暂存为STAGED；收到显式paired_move后，才把连续对象交给左右执行线程。两侧各自复核起点并通过共同启动门控，任一侧失败则取消双方。

新增只读参数 `allow_continuous_trajectory=false`，必须与既有成对许可及两臂已确认限位同时满足才允许版本二接入；默认不会连接、使能或执行。现有小幅位移限制不变，不会因版本二文件到达而扩大限制。

后台任务最多一个，不排队积压。跨进程返回的数值数组会在暂存前重新冻结；取消信号独立传递，故障时先请求两臂执行线程停止，再通知计算进程退出。预检未完成时收到新消息会作废旧结果并拒绝新请求，等待后台退出后需重新提交。预检期间禁止单臂运动和提前开始；halt、显式断开、驱动退出均作废预检结果。普通候选预检上限30秒，含文件核对与模型检查的完整文件上限120秒；迟到成功也不能重新暂存。

状态话题提供VALIDATING、STAGED等阶段与preflight_busy。普通paired_target仅检查关节约束；文件入口还做本文所述离散几何检查，但没有真实夹爪验证。Linux线程及JSON解析不构成硬实时停止保证，替身测试也不能代替实际控制器验收。


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


## 离线起点衔接与采样几何检查

在 `task-review` 后增加 `--geometry`，使用同一次校验保留下来的只读轨迹对象进行几何检查，不重新打开候选路径文件。可选 `--start-file` 提供离线起点，并明确指定 `--approach-max-excursion`：

```bash
ros2 run archer_hardware hardware_operator task-review \
  --run-dir /path/to/archer_results/日期_时间_任务名称 \
  --limits-file ~/archer_ws/src/archer_hardware/config/position_control.yaml \
  --prepared-file /path/to/prepared_path.json \
  --report-file /path/to/prepared_path_report.json \
  --max-excursion 6.3 --geometry \
  --start-file /path/to/start.json --approach-max-excursion 0.05
```

起点文件仅包含 `left`、`right` 两个六维关节角数组，单位rad。离线文件不能冒充新鲜反馈；未来执行仍须重新读取并核对双臂实际反馈、状态和时间。略超名义限位的读数会拒绝，要求先走既有的独立限位恢复流程，不自动裁剪反馈。

三段方案是：本轮起点→文件起点的接近、完整原文件、文件末点→本轮保存起点的归位。两个边界段都用零速零加速度五次插值，按更严格一侧的限速自动选择共同时间；文件主体不改写。这里只生成内存预检对象，没有播放命令，也没有新增独立归位结果文件。

几何检查沿实际连续多项式采样，按解析速度峰值加密，相邻点最多0.1秒、单关节行程目标上限0.005rad；默认总预算100000点。预算不足直接拒绝，不降低精度或只检查前半段。检查当前安装模型下跨臂10mm和机器人与台面1mm间隙；台面Z=-0.01m，双基座X=±0.21m、yaw=90°，TCP使用当前工程候选值。模型检查GP80和法兰固定链，哈希记录加载后的几何与运动学数据。

通过时 `sampled_geometry.sampled_cross_arm_table_passed=true`；`interval_collision_proven=false`，因为加密采样仍不是连续碰撞证明。自碰撞、工件、线缆、外部障碍与真实抓取接触没有纳入。总执行许可和实时起点验证保持false，不会据此连接、使能或下发动作。

`--max-excursion 6.3` 仅是本例的离线全文件检查范围，正式驱动0.05rad小幅限制及默认关闭许可不变。接近和归位共用单独的0.05rad范围，不能拿全文件范围替代边界段保护。

## 画圆完整文件执行通道

本节描述已实现的接口，不表示现在已允许连接或执行真机。默认配置仍为不自动连接、不自动使能、限位未确认，`allow_paired_trajectory`、`allow_continuous_trajectory`、`allow_task_file_trajectory` 全为false，0.05rad小幅限制不变。本轮不提供自动打开这些门禁的启动命令。

完成后续实机验收并明确开放时，执行入口形式如下；四个路径必须定位驱动所在电脑上的同一轮记录与准备文件：

```bash
ros2 run archer_hardware hardware_operator task-run \
  --run-dir /path/to/archer_results/某轮画圆 \
  --limits-file ~/archer_ws/src/archer_hardware/config/position_control.yaml \
  --prepared-file /path/to/development_cache/prepared_path.json \
  --report-file /path/to/development_cache/prepared_path_report.json \
  --transition-max-excursion 0.05
```

该入口不连接或使能，要求双臂已经处于HOLDING并有新鲜静止反馈。驱动保存本轮双臂角度快照（两帧时间差不超过20ms），计算进程读取并核对文件快照、当前限位和完整阶段，规划接近与返回本轮起点，再检查原任务及两段过渡。整轮范围仍由驱动的max_excursion约束，不从文件请求接受扩大值。

通过后重新核对反馈并暂存STAGED；客户端再用同一唯一编号显式消费一次计划。路径在同一共同时间轴中包含接近、完整文件和归位；左右线程复核后通过共同门控启动。完成需匹配两侧本次命令编号、SUCCEEDED、HOLDING及新鲜反馈，不能把“收到请求”或另一轮FINISHED当作成功。正常返回操作客户端只关闭客户端，位置驱动继续保持最终目标。

文件定位话题为 `/hardware/position/task_file`；带编号的执行话题为 `/hardware/position/task_execute`。普通paired_move服务不能消费完整文件，以免把旧的预检编号或另一份暂存误当成本次请求。正式状态在 `/hardware/position/status` 的paired_trajectory下，包含request_id、task_file、last_file_rejection和左右命令编号。

文件内容不会在检查后重新打开用于发送；跨进程返回数组重新冻结。预检后、左右准备时或等待共同启动时，起点发生变化都不能重建首段继续执行。当前采取精确起点匹配，编码器噪声或漂移可能导致拒绝；实际稳定性与允许误差需下一步只读实测后评估，不在本轮凭经验放宽。

`--timeout`对task-run默认120秒，表示通信及预检等待窗口；任务执行等待按路径自身时长再加该窗口，不将902秒轨迹误当120秒失败，也不因更改timeout而加速任务。调用中断或执行结果不明时沿用既有尽力halt；不会自动重发、重新连接、归位、发送零力矩或失能。退出真实驱动仍在驱动终端按Ctrl+C，取消后关闭连接。

抓取记录会明确拒绝进入此运动通道，因为虚拟附着和夹爪命令2.0尚不能映射为已验证的真实夹爪动作；仍可使用task-review做离线核对。当前几何检查依然不含自碰撞、工件、线缆或连续碰撞证明，也未解决calibrated状态说明、实际TCP标定和真实控制频率抖动；这些属于执行真机任务前的后续工作。

新增只读开关会改变配置文件哈希。旧准备报告若不再匹配，应从同一份原CSV重新准备到开发缓存，保留原始采样；不要手工改哈希字段冒充验证通过。

## 关节误差的几何包围检查

几何检查函数可显式指定 `joint_error_rad`，默认0，不改变现有执行起点规则。
非零值仅用于离线评估，不能超过0.001rad或已有反馈读取容差；这不是厂商精度参数。
检查器沿碰撞几何到各祖先转动关节的固定链，利用偏移长度和几何包围球求转动半径上界，
再用各关节 `R*误差角` 之和保守覆盖任意碰撞外形点的位移。
跨臂要求增加左右位移上界之和，台面要求增加两侧上界的较大值；角度误差包围也必须在名义与模型限位内。
不支持的活动关节、无效包围数据或间隙不足均拒绝，不把误差容差当作忽略碰撞的理由。

报告同时输出误差角、外形位移上界和实际检查的间隙要求。结果仍是离散姿态检查，
没有证明采样间隔内的运动安全，也未覆盖自碰撞、线缆、工件或实际跟踪动态。
执行线程尚未使用该误差值放行起步，`preserve_path`的精确匹配保持不变。
本阶段先建立几何依据，后续还须绑定实际保持目标，保证不改写已检多项式或造成指令跳变。
