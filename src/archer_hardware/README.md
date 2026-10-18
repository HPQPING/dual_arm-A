# 真机接口：两种停止与终端键盘操作

更新日期：2026-10-18。本轮补齐前台键盘入口，不连接机械臂、不使能或执行真机动作。主要运行模块仍为8份，没有增加新的运行文件。

## 一、键盘使用

在Ubuntu的独立前台终端运行：

```bash
source /opt/ros/humble/setup.bash
source ~/archer_ws/install/setup.bash
ros2 run archer_hardware hardware_operator keyboard --timeout 180
```

| 按键 | 行为 |
| --- | --- |
| S或s，不需要回车 | 调用普通stop；当前运动已完成后，返回本次初始位置对应的限位内目标，完成后继续监听 |
| Ctrl+C | 立即请求halt，不归位；等待软件结果后退出监听。本次驱动运行不可恢复 |
| 空格及其他普通字符 | 无操作；空格已按用户要求取消绑定 |

S不是Ctrl+S。重复按S不会重复提交正在处理的归位；方向键、功能键的终端转义序列不应被误识别为S。归位等待采用异步服务和状态检查，Ctrl+C不会因等待归位而被阻塞。

Ctrl+C不是直接销毁客户端。keyboard模式关闭rclpy默认SIGINT处理，在保留ROS通信的情况下捕获KeyboardInterrupt，先请求已有halt接口，再退出并恢复终端设置。halt服务或软件停止状态最多等待约5秒；超时明确报告未确认，不能把程序退出当作实际已停稳。程序被强杀或系统/通信异常时仍可能无法完成发送。

只在这个键盘终端获得焦点时有效，不是全局热键。其他终端中的Ctrl+C只作用于那个终端的进程，不等于本入口绑定。管道、重定向及没有交互终端的后台运行会在ROS初始化前被拒绝。没有位置驱动服务时不会自动启动、连接或使能设备。

本入口只创建stop/halt服务客户端；不会发送零力矩、MIT或失能。软件停止不能替代实体急停。

## 二、两种停止语义保持

普通stop只能在所有已配置臂完成当前控制请求并处于HOLDING、有新鲜静止反馈和本次初值时受理。运动中S会被驱动拒绝，原运动不会变成归位；遇到异常用Ctrl+C。

正常运动完成保持终点，不自动归位。显式stop后，各已配置臂归位并保持；若实际初值略越界，其指令目标为对应名义边界，初始记录不改。

halt可以在归位期间请求，作废后续目标并尝试BRAKE。本次驱动进程结束标记不可清除，无reset恢复入口；disconnect只清理连接，不能解除结束状态。后续人工断电复位后，另行启动新运行并重新读取实际初值。软件不能仅凭程序重启证明设备已经断电。

当前通过终端和ROS命令实现，未绑定实体按钮或图形按钮。两个执行线程的归位不是已验收的双臂同步/碰撞规划。

## 三、目录与本轮改动

```text
~/archer_ws/src/archer_hardware/
├── archer_hardware/
│   ├── __init__.py
│   ├── hex_adapter.py
│   ├── driver_node.py
│   ├── position_plan.py
│   ├── position_adapter.py
│   ├── position_worker.py
│   ├── position_node.py
│   ├── operator_client.py          # 本轮：异步键盘停止、Ctrl+C退出及共用编号解析
│   └── commissioning.py
├── config/
│   ├── hardware.yaml
│   └── position_control.yaml
├── test/
│   └── test_operator_client.py     # 本轮：键盘及ROS替身回归；其他测试保留
├── resource/
├── README.md
├── package.xml
├── setup.cfg
└── setup.py
```

本轮修改operator_client.py、其测试和README。没有修改位置驱动、SDK适配、控制限位或默认连接参数。

| 现有模块 | 职责 |
| --- | --- |
| hex_adapter.py | 官方SDK哈希核对、只读过滤和反馈解析 |
| driver_node.py | 独立双臂只读反馈节点 |
| position_plan.py | 公共校验、限位、准备和归位轨迹计算 |
| position_adapter.py | 单臂SDK位置控制与反馈状态 |
| position_worker.py | 每臂执行线程、请求有效期及取消 |
| position_node.py | ROS接口、两种停止及故障结束状态 |
| operator_client.py | 显式命令、请求编号确认和前台键盘 |
| commissioning.py | 连续反馈报告、准备及试动文件生成 |

## 四、仿真外功能与当前边界

实际只读连接曾验证双臂约50Hz反馈、六轴robot_type=25及状态字段。实测初值记录、0.005rad初始读取容差、准备收回、正式名义限位监测、位置插值、保持、边界归位、故障联停和两种停止均已有代码与离线测试。

实机使能/位置运动、制动/自锁效果、真实夹爪动作、画圆/共同夹取完整任务执行尚未验收。SDK的calibrated=false含义及关闭超时仍待核对。键盘功能不代表这些问题已经解决。需要再次实际连接机械臂时仍先让用户选择；本轮没有建立设备连接。

## 五、兼容命令

既有hardware_operator status/connect/enable/move/stop/halt/disconnect保留；audit/trial仍转发联调工具。移除的return/reset没有恢复。

```bash
# 查看状态，不连接设备。
ros2 run archer_hardware hardware_operator status --timeout 5

# 查看联调工具帮助。
ros2 run archer_hardware hardware_operator audit --help
ros2 run archer_hardware hardware_operator trial --help
```

旧hardware_feedback_audit与hardware_joint_trial仍通过setup.py指向commissioning同一实现。SDK的hex_device依赖仍必须保留。

## 六、验证和交付修正

- Ubuntu构建通过；97项测试通过、1项版权模板跳过、8条插件提示。
- 实际ROS环境Pyright零错误零警告；flake8、PEP257、git diff --check通过。
- 真实伪终端运行安装后的ros2命令，使用隔离ROS域与内存服务：启动无服务调用；空格无操作；S提交一次归位；重复S不重发；归位等待中Ctrl+C请求halt后退出；终端设置恢复；退出码0。
- 安装后的operator_client与源码逐字节一致。测试没有创建SDK连接或运动机械臂。

部署前发现本地交付目录混放顶层副本与标准包内副本，使第十二轮ZIP的包内模块可能保留旧版本。Ubuntu实际源码始终是已经验证的两种停止逻辑，未被这些副本覆盖。本轮以Ubuntu源码归档重建了唯一标准目录，保留原本地目录及原ZIP在work下供追溯；hardware_two_stops_round12.zip已按Ubuntu当时实际版本修正，本轮完整包也采用标准目录。校验归档时核对所有包内文件与实际源码，禁止顶层Python模块副本，避免再次出现版本覆盖。

备份：~/archer_ws/dual_results/hardware_keyboard_backup_20261018_111855/。

终端记录：~/archer_ws/dual_results/hardware_keyboard_terminal_20261018/terminal_output.txt。

摘要：~/archer_ws/dual_results/hardware_keyboard_validation.json。

本轮交付为完整archer_hardware源码包，含配置、测试、当前README与验证摘要，不是全五包工作区镜像。历史报告保留；键盘绑定以本轮S/Ctrl+C为准，空格不再使用。

## 七、SDK连接清理修正（2026-10-18，第十四轮）

已离线复现官方SDK 1.4.8的关闭竞争：`__async_close`先设置主循环退出事件，再等待WebSocket关闭；`run_until_complete(__main_loop())`可能先返回，导致关闭协程及等待结果尚未完成。此结论来自运行实际SDK主循环和内存传输，不是对实机日志的单纯猜测。

修改范围：

```text
archer_ws/src/archer_hardware/
├── archer_hardware/
│   ├── hex_adapter.py
│   └── position_adapter.py
├── test/
│   └── test_feedback_adapter.py
└── README.md
```

反馈与位置适配器共用`managed_sdk_class`：先完成WebSocket关闭握手，再通知SDK主循环退出；握手等待上限2秒，失败时中止本地传输并向调用方报告。退出时取消残留任务、关闭事件循环，并确认后台线程退出。SDK解析回调内调用close时不阻塞自身事件线程；并发关闭共享握手结果，重复清理仍可调用。厂商源文件及四份SDK哈希未改变，兼容层仅适用于当前已核对版本。

这只修正通信资源清理，不增加运动、使能、校准或断电功能。halt仍保持连接及反馈，stop仍是满足完成条件后归位；退出位置驱动前既有软件制动请求仍由原控制层处理。连接关闭不证明实机已经停稳，也不改变固件的失联处理配置。

`calibrated`由`Arm._update`从`ArmStatus`读取，协议为没有字段存在性信息、默认false的布尔值。因此SDK读到false不能单独证明机械零位校准丢失，也不能判断固件是否省略了该字段。已有协议对该字段没有给出Archer-Y6/type25的进一步解释。使能门槛继续保留，报错改为“尚未报告calibrated=true”，不自动校准、不放宽判断。

本轮Ubuntu构建通过；102项测试通过、1项跳过；实际ROS类型检查零错误零警告；样式、文档字符串及diff检查通过。新增离线覆盖正常握手、回调内关闭、并发/重复关闭、握手失败和超时。另在127.0.0.1模拟服务器运行实际SDK和WebSocket连接：约0.303秒完成关闭，服务端确认关闭、SDK线程退出、事件循环关闭，发送内容仅含反馈频率设置。未连接192.168.1.100、未下发真机命令；实机关闭表现仍待经用户选择后的复测。

备份：`~/archer_ws/dual_results/hardware_shutdown_backup_20261018_114231/`。摘要：`hardware_sdk_cleanup_validation.json`、`hardware_shutdown_loopback.json`。完整交付包：`hardware_sdk_cleanup_round14.zip`。下一步需要确认当前固件的calibrated含义，之后才能安排真机控制验证；真实连接与运动分别征求用户选择。

## 八、设备型号约束与异常清理（2026-10-18，第十五轮）

本轮继续离线检查，没有连接机械臂。修改位置适配器、只读节点及两份原有测试，不增加运行文件。

```text
archer_ws/src/archer_hardware/
├── archer_hardware/
│   ├── position_adapter.py
│   └── driver_node.py
├── test/
│   ├── test_position_control.py
│   └── test_feedback_node.py
└── README.md
```

位置驱动在发送初始化前明确检查已核对型号16、17、25；Hello/type26及未知型号不能因具备六轴反馈而被允许控制。SDK只读适配器仍可以读取type26。当前实测Archer/type25保留原有calibrated、故障、会话和静止反馈检查，不放宽使能门槛。该型号区分对应已安装SDK的Arm映射；[厂商ROS驱动说明](https://github.com/hexfellow/hex_ros_robot_arm#4-control-modes)也将Hello定义为只读设备。

一侧反馈异常且连接关闭失败时，节点保留原始异常及清理异常，继续处理另一侧反馈。节点退出时始终尝试两侧清理，之后统一报告失败；即使连接清理抛错，仍销毁节点并关闭ROS上下文。第二侧启动连接失败时同样清理已创建的两侧适配器、销毁节点，原连接错误不会被清理错误掩盖。现有/hardware/status结构、关节话题和停止按键不变。

新增六项离线测试，覆盖只读/未知型号拒绝、当前type25校准检查、轮询中一侧清理失败、退出时两侧清理、部分启动失败和ROS上下文最终释放。Ubuntu构建通过；108项测试通过、1项跳过，实际ROS类型检查零错误零警告，样式和diff检查通过。

现有资料仍不能确认当前固件calibrated=false的具体原因，不能据此直接运行校准或绕过检查。下一步可以在用户选择后只读重连，采集新反馈并复测实机连接清理；该操作不包括申请控制、运动、制动、归位或校准。即使只读验收通过，也不表示已授权运动。

备份：`~/archer_ws/dual_results/hardware_readiness_backup_20261018_115711/`。完整交付：`hardware_readiness_round15.zip`。本轮未改变官方SDK文件、参数配置及默认关闭的连接开关。

## 九、初始化、API校准和位置控制分阶段处理（2026-10-18，第十七轮）

本节取代前文“calibrated=false时禁止申请控制”的旧规则。此前第十六轮已将申请初始化与校准检查分开，111项离线测试通过；该轮中断后未单独打包。本轮完整包包含该轮及本轮全部修改。

```text
archer_ws/src/archer_hardware/
├── archer_hardware/
│   ├── position_adapter.py
│   ├── position_worker.py
│   └── position_node.py
├── config/
│   └── position_control.yaml
├── test/
│   ├── test_position_control.py
│   └── test_position_node.py
└── README.md
```

### 流程

显式enable读取新鲜静止反馈并检查型号、限位、故障及会话；允许calibrated=false时申请API初始化，但不发送位置目标。等待当前会话取得控制、API初始化确认后，若calibrated仍为false，仅在allow_api_calibration=true时发送一次官方Calibrate(true)消息。发送前锁存，发送失败、超时或未确认均不自动重发。设备已报告true时跳过校准。全部条件确认后，记录就绪时实际关节角并保持该位置，之后才可接受移动目标。

该流程参考[官方Rust示例](https://github.com/hexfellow/robot-demos/blob/main/examples/arm-ez-control.rs#L213)和Python SDK的分阶段处理，未恢复厂商后台自动清错、自动校准、抢占控制等行为。官方示例后续包含零力矩命令，本项目没有照搬这些动作。

### 配置和观察

allow_api_calibration默认false，严格布尔类型，是启动时只读参数。仅设置此参数不会连接设备、申请控制或校准；仍须显式connect/enable。已有自动连接开关和限位确认开关均保持false。真实API校准操作须用户另行选择，目前没有执行。

readiness_timeout默认10秒，允许范围为大于0且不超过30秒，覆盖整个控制申请、初始化及校准等待过程，并非厂商承诺耗时。超过期限进入故障处理，不继续尝试。把时间加长不能代替缺失的设备确认。

原有/hardware/position/status每臂增加readiness_stage及allow_api_calibration字段，命令行查看：

```bash
source /opt/ros/humble/setup.bash
source ~/archer_ws/install/setup.bash
ros2 run archer_hardware hardware_operator status --timeout 5
```

该命令只读取正在运行的位置驱动状态，不启动驱动或连接机械臂。阶段值：

| readiness_stage | 含义 |
| --- | --- |
| IDLE / CONNECTED | 尚未申请控制 / 已创建反馈连接 |
| WAITING_SESSION | 等待控制会话属于本连接 |
| WAITING_INITIALIZATION | 等待API初始化确认 |
| CALIBRATION_PERMISSION_REQUIRED | 设备未报告校准就绪，当前未允许发送校准 |
| WAITING_CALIBRATION | 一次校准请求已提交，等待设备确认 |
| READY | 设备条件确认，已进入实际位置保持 |
| STOPPED / FAULT / DISCONNECTED | 已停止 / 故障 / 已关闭连接 |

### 未知行为的处理边界

等待期间持续检查设备源反馈时效、限位和故障；速度超过0.02 rad/s，或相对申请前反馈累计漂移超过0.01 rad，取消后续命令并沿用已有尽力BRAKE及两臂故障联停。校准请求后控制会话或初始化丢失，同样结束本次运行。阈值为工程初始策略，不是厂家安全等级保证，不代表未知命令不会引起运动或能在伤害前制止运动。

halt继续不归位、不主动断开、不切零力矩；不会因校准确认迟到而恢复运动。普通stop仍只在当前操作完成后归位。所有停止语义及限位规则保持。

当前固件中校准命令的具体物理动作、零点影响和实际耗时尚未验证。使用显式默认关闭开关、单次请求、反馈确认及可中止等待来管理这些不确定性；不把本地状态直接改成true，不把通信发送成功当作校准完成。

### 验证和交付

Ubuntu构建通过，122项测试通过、1项跳过；实际ROS环境类型检查零错误零警告，样式及diff检查通过。新增覆盖显式许可、单次请求、实际官方校准消息字段、发送出口状态门槛、已就绪跳过、失败/超时不重试、运动/漂移异常、初始化丢失、halt后迟到确认以及ROS配置和阶段传播。测试仅使用离线SDK对象与内存后端，没有连接实机或运行厂商示例。

备份：`~/archer_ws/dual_results/hardware_api_calibration_backup_20261018_131848/`。完整功能包：`hardware_api_calibration_round17.zip`。验证摘要：`hardware_api_calibration_validation.json`。主运行模块仍为8份。

## 十、离线双臂关节计划预检（2026-10-18，第十八轮）

上一轮源码、说明和完整归档已收尾。继续检查任务接口发现：现有dual_arm_node使用仿真话题、仿真后端状态与停止服务；真机位置驱动目前只支持单段小幅位置目标，不能直接把仿真话题重映射到它来执行完整画圆或抓取任务。

本轮复用commissioning和hardware_operator，新增plan-check子命令；没有新增运行模块、独立检查脚本或控制服务。

```text
archer_ws/src/archer_hardware/
├── archer_hardware/
│   ├── commissioning.py
│   └── operator_client.py
├── test/
│   ├── test_joint_trial.py
│   └── test_operator_client.py
└── README.md
```

### 输入与检查范围

输入为YAML或JSON的明确计划格式：根字段version=1、interpolation=quintic_stop、left、right。每臂包含按joint_1至joint_6排序的joint_names，以及至少两个points；每点只有positions与time_from_start，后者含整数sec和nanosec。两臂首点时间必须为零且共用完全相同、严格递增的纳秒时间轴。位置单位rad、时间单位秒。

quintic_stop表示每段端点速度和加速度为零的五次插值。逐段解析检查名义关节限位、峰值速度1.875×位移/时长与峰值加速度(10/√3)×位移/时长²；不只检查离散点或平均速度。第一点及全部正式计划位置必须在名义限位内，初值工程偏差仍走已有准备流程。不能将其他插值、实际记录CSV或只有几何信息的profile.json直接改名当作该输入。

```bash
source /opt/ros/humble/setup.bash
source ~/archer_ws/install/setup.bash
ros2 run archer_hardware hardware_operator plan-check --help

# 输入必须是按上述格式准备的计划，输出使用尚不存在的新文件名。
ros2 run archer_hardware hardware_operator plan-check --plan-file /path/to/paired_plan.yaml --limits-file ~/archer_ws/install/archer_hardware/share/archer_hardware/config/position_control.yaml --output ~/archer_ws/dual_results/plan_review_new.json
```

本命令不初始化ROS、不创建控制客户端、不导入或连接SDK实例，只读取文件并生成报告。通过退出码0，拒绝退出码1。报告记录输入与限位文件的SHA256、解析峰值、计划起终点及commands_sent=false，文件不覆盖。校验使用同一份限位字节内容，避免哈希对应另一版本。

### 通过不等于可执行

报告execution_supported_by_current_driver始终false，明确当前驱动尚无完整双臂任务执行接口。预检不读取实际初始位置，也不验证TCP、碰撞、桌面、物体或夹紧力，不替代既有运动学/碰撞检查。两个时间轴相同只是计划同步要求，不代表实际控制线程已经同步。计划导出、完整轨迹执行、双臂启动与取消协议、反馈跟踪及执行前实际初值比对仍待补齐；本轮没有移除小幅联调的位移限制或开启自动连接。

### 验证和交付

Ubuntu构建通过，130项测试通过、1项跳过，实际ROS类型检查零错误零警告，样式及diff检查通过。安装命令对离线合成示例验证：合法计划通过；超速拒绝；总时长相同但中间时间不同拒绝。另覆盖加速度超限、角度越界、错误关节名、非有限值、无效时间、未知插值、报告不覆盖及命令不初始化ROS。

示例位于交付包根目录offline_examples/，仅用于文件格式和预检演示，不是实机动作。Ubuntu记录位于`~/archer_ws/dual_results/hardware_plan_review_smoke/`；备份`hardware_plan_review_backup_20261018_133259/`。完整功能包`hardware_plan_review_round18.zip`包含上一轮API校准代码；当前主运行模块仍8份。本轮没有连接机械臂或发送实机命令。

## 十一、多段关节轨迹执行核心（2026-10-18，第十九轮）

本轮为完整任务补齐内部执行基础，尚未开放新的ROS整段运动入口，也没有实现双臂同步执行。前文“ROS接口只支持小幅单点”的边界仍成立。本轮不连接实机或发送控制命令。

```text
archer_ws/src/archer_hardware/
├── archer_hardware/
│   ├── position_plan.py
│   ├── position_adapter.py
│   └── commissioning.py
├── test/
│   ├── test_position_control.py
│   └── test_joint_trial.py
└── README.md
```

JointPositionTrajectory由整数纳秒时间轴和不可变的多段JointPositionMove组成，至少两个点、从零严格递增、总时长不超过一小时。各段五次插值在连接点位置连续，速度和加速度为零。按绝对已用时间选择分段，不逐点补发积压数据，最终保持末端目标。

prepare要求actual_start：正式轨迹与实际起点均须在名义限位内，初值工程偏差先使用既有准备流程处理。计划首点与实际角度最大偏差超过0.005 rad时拒绝，需要重新规划接近段；容差内将首段起点换为实际角度，并重新验证首段峰值约束，避免第一条位置指令跳向历史角度。起点容差是工程值，不表示TCP几何误差或碰撞已验证；后续任务接口仍需使用实际起点规划并保留几何预检。

整条轨迹所有点相对实际起点的位移都必须符合当前max_excursion，不能通过拆分小段绕过累计限制。每段速度和加速度使用既有解析峰值检查。没有自动放宽限位、速度、加速度或默认0.05 rad小幅联调位移限制。

HexPositionAdapter.follow_trajectory是内部方法：仅从HOLDING开始，重新读取新鲜静止反馈并核对控制会话、初始化、校准及全程约束。执行复用现有tick的反馈时效、调度间隔、跟踪误差和故障处理。到位需实际反馈稳定确认，随后保持最终位置，原始归位初值不变；不自动归位。halt清空整条轨迹，之后tick不再恢复或补发剩余点。

plan-check改为复用同一个轨迹prepare核心，报告仍是离线关节数学检查，execution_supported_by_current_driver保持false。这样检查与未来执行不会因两套插值/导数规则而出现差异。当前ROS完整双臂接口、同步启动取消、规划导出、在线任务反馈和夹爪接入仍待完成，不能据本轮核心方法存在就直接执行画圆或实物抓取。

Ubuntu构建通过，140项测试通过、1项跳过，实际ROS类型检查零错误零警告，样式与diff检查通过。新增覆盖连接点连续性、采样、起点不符、容差内对齐后重验首段、累计位移限制、正式名义限位、非法时间/超长计划、完整到位保持、保留初值及halt取消。所有SDK发送由替身接收。

备份：`~/archer_ws/dual_results/hardware_full_trajectory_backup_20261018_134343/`。完整功能包：`hardware_full_trajectory_round19.zip`。主要运行模块仍8份；已有ROS入口、配置和键盘行为不变。

## 十二、双臂软件同步暂存与执行（2026-10-18，第二十轮）

用户再次确认：两臂连接同一台控制器192.168.1.100，左臂can0/8439，右臂can1/9439。保持两个独立WebSocket连接及各自端口返回的会话检查，不要求左右会话编号相同。共用控制器不能仅凭端口信息推断为硬同步。

```text
archer_ws/src/archer_hardware/
├── archer_hardware/
│   ├── position_plan.py
│   ├── position_adapter.py
│   ├── position_worker.py
│   ├── position_node.py
│   └── commissioning.py
├── config/position_control.yaml
├── test/
│   ├── test_position_control.py
│   └── test_position_node.py
└── README.md
```

### 新接口及默认行为

allow_paired_trajectory默认false，为启动时只读布尔参数。默认配置也保持connect_on_start=false、两臂限位确认false、API校准许可false。本轮不连接机械臂，不申请控制、不校准或执行实机命令。

新增/hardware/position/paired_target（std_msgs/msg/String），内容是plan-check同一version1/quintic_stop双臂计划的JSON。需要显式允许成对接口、both配置、两臂已HOLDING及新鲜静止反馈，才可暂存。解析验证全轨迹后，还按照实际起点和当前max_excursion逐臂重新预检。暂存不执行，错误新消息使此前成对暂存失效；单臂目标与成对暂存互斥。

新增/hardware/position/paired_move（std_srvs/srv/Trigger）消费一次暂存，过期或重复请求拒绝。返回两个操作编号只是受理，不是动作已完成。两臂工作线程再次读取反馈、预检起点及约束，然后进入ARMED位置保持。

接口调用格式如下，仅是代码接口说明；本轮没有执行真机调用：

```bash
ros2 service call /hardware/position/paired_move std_srvs/srv/Trigger '{}'
ros2 run archer_hardware hardware_operator status --timeout 5
```

执行前需已有有效成对JSON暂存，不能把YAML或仿真话题直接重映射成这个消息。现有hardware_operator尚无成对计划上传/执行子命令，高层计划导出与操作客户端衔接留在后续完成；手工服务调用不替代这些预检和记录。

### 软件同步与取消

共享PairedTrajectoryStart要求左右两个不同侧都准备成功后，才确定共同的time.monotonic时刻。准备最多等待0.5秒，双方准备后预留0.2秒启动窗口，等待期间发送的是当前位置保持目标。任何侧准备失败、停止或故障使门控不可恢复地取消；尚未到共同时间时不推进轨迹。

开始推进后，两臂按相同绝对时间轴采样。各侧首次推进迟到超过20 ms，或等待期间速度超过0.02 rad/s、相对保持位置漂移超过0.005 rad，取消本次执行。时间值及阈值为工程策略，不是硬实时或物理动作保证。线程阻塞、通信和控制器延迟可能导致检测与制动迟到；不能撤回已进入网络或控制器的指令。

ROS节点按两个操作编号监视完成，不能把其他操作成功状态当成本次完成。任一拒绝、故障、取消、异常状态、操作丢失或反馈超过0.1秒没有更新，取消门控并请求两臂停止。SDK发送出口也检查门控，禁止取消后继续发送尚未进入出口的位置消息。两臂各自真正到位稳定并HOLDING后，才标记FINISHED，不自动归位。

halt保持现有语义：任意阶段作废后续动作、尽力BRAKE、不归位、不主动断开、不切零力矩，本次运行不能恢复。普通stop在成对执行尚未完成时拒绝，完成后按原初始记录归位。普通归位仍是两臂独立线程的既有流程，本轮没有新增同步或无碰撞归位规划。

### 状态与验证边界

/hardware/position/status新增paired_trajectory，包含allowed、staged、active、phase与两个commands编号。阶段有IDLE、STAGED、PREPARING、RUNNING、FINISHED、CANCELLED及REJECTED说明。每臂新增trajectory_start_at_monotonic（共同计划时刻）与trajectory_observed_start_at_monotonic（本臂首次推进的主机时刻）。这不是控制器时间戳、网络发送时刻或机械臂实际启动测量，不能据此宣称物理同步精度。

plan-check仍只给出离线关节数学结果，新增paired_interface_available=true不等于授权执行，原保守执行标记保持false。几何/TCP、桌面、线缆、物体及夹紧力验证没有被本轮替代；规划器导出、任务客户端及实机联调仍待完成，不能直接宣称完整实物抓取已可运行。

Ubuntu构建通过，154项测试通过、1项跳过；实际ROS类型检查零错误零警告，样式及diff检查通过。真实ROS话题与服务配合内存后端验证暂存不执行、共同时间、到位保持与显式归位、单侧准备失败、启动前halt、重复提交拒绝、过期/坏计划、运行故障联停及过期反馈监视。适配器及发送出口另验证门控取消、等待漂移与迟到拒绝。安装节点默认启动烟雾验证没有调用SDK，端口正确，新服务存在但默认拒绝成对执行。

备份：`~/archer_ws/dual_results/hardware_paired_start_backup_20261018_135732/`。完整功能包：`hardware_paired_start_round20.zip`；摘要：`hardware_paired_start_validation.json`。没有新增运行模块，仍8份主要模块。


## 离线轨迹时间调整

```bash
ros2 run archer_hardware hardware_operator plan-retime --plan-file dual_results/cooperative_export_round21.json --limits-file src/archer_hardware/config/position_control.yaml --output dual_results/cooperative_slow.json --report dual_results/cooperative_slow_report.json
```

仅适用于现有 version=1、quintic_stop、共用纳秒时间轴的双臂计划。使用左右臂所有段的解析峰值选择统一放慢倍数，默认保留5%的时间余量；不改关节角、名义限位或驱动限速。逐段时间向上取整到纳秒，两臂使用同一结果。原本足够慢的文件保持时间不变，不会加速。

输出前使用执行核心重新检查名义角度、速度、加速度及一小时上限；输入、配置、输出和报告必须为不同路径，输出文件不能已存在。成功报告包含原文件、配置及新计划SHA256。失败不会返回可执行计划；不会初始化ROS、连接设备、暂存或执行。`--margin`可选范围为1.001～2.0，默认1.05。

时间调整保留同段进度下的关节空间路径和双臂关系；此前通过的几何假设没有得到新的实物验证，也不意味着动态接触、跟踪误差或实际起点已经验收。当前驱动的最大试动幅度仍保持0.05 rad，完整大幅度任务不会因离线时间预检通过而自动放行。


## 成功仿真任务的CSV路径导入

```bash
ros2 run archer_hardware hardware_operator csv-import --run-dir "archer_results/实际日期_开始时间_微秒_协同抓取" --limits-file src/archer_hardware/config/position_control.yaml
```

此命令只读源文件，在同一目录新增`recorded_path.json`和`recorded_path_report.json`，不覆盖已有文件，不创建新的任务文件夹。摘要必须明确标记COMPLETED。支持同一行包含左右臂的`state.csv`、左右分行的独立画圆`tracking.csv`和ROS反馈`tracking.csv`；目录同时有两种文件时须用`--csv-name`明确选择。

只提取实际`position_rad`或`actual_J*_rad`，不使用仿真`drive_input_rad`。优先采用仿真时间；ROS反馈格式使用整数纳秒时间戳，按左右对应样本配对，最大允许时差默认5 ms。保留采样间隔，仅将第一点时间平移为零。相同时间且角度完全相同的重复记录可以合并；倒退、同刻不同角、缺口超过0.2 s、未配对数据、非法列或名义角度越界均拒绝，不静默丢帧、夹紧角度或补造数据。两个间隔参数可显式设置：`--max-gap`范围0.001～1秒，`--pair-skew-ms`范围0～5毫秒。

候选为版本三`recorded_joint_samples`，插值标记`unassigned`。导入检查与采样差分诊断不等同于连续运动峰值、碰撞、实际初值或夹持验证；报告始终`execution_ready=false`。差分超限时保留原始时序并标记需要调整，不能把导入成功当作速度校验通过。旧执行接口会拒绝该格式，尚不发送运动。参考阶段和夹爪示意状态仅保留供分析，不转成实机夹爪指令。

通过报告内CSV、摘要、限位配置及输出SHA256核对文件身份；输入CSV和summary.json保持不变。离线命令不会初始化ROS、连接SDK或自动使能。
