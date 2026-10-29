# archer_hardware：真机接入与完整任务执行

更新：2026-10-29。本文以当前源码、注册入口和实际安装检查为准。

## 职责和当前版本

本包负责厂商SDK的连接、反馈、SDK准备、POSITION发送、双臂共同参考时间、夹爪事件、故障取消以及关闭。规划数学与模型检查复用control/mujoco；普通任务无需单独启动底层驱动。

正式协同版本为211：反馈控制沿用206实机认可逻辑，J6调姿与归位开始分臂时间采用211候选。J6名义主段9.887→7.423秒，分臂11.337→8.956秒；它们是计划值，211整合版真实平稳性仍待验证。

## 主要命令

先加载ROS/工作空间环境。转换输入是**原始CSV**，执行输入是**转换后的轨迹JSON**，不是报告JSON。

```bash
ros2 run archer_hardware hardware_operator circle-import --input "/原始任务目录/tracking.csv" --profile circle
ros2 run archer_hardware hardware_operator grasp-import --input "/原始任务目录/state.csv"
ros2 launch archer_bringup dual_circle_real.launch.py execute:=true trajectory_file:="/转换目录/tracking.json"
ros2 launch archer_bringup dual_grasp_real.launch.py execute:=true trajectory_file:="/转换目录/state.json"
```

两条launch会运动，内部各只创建一份完整任务调度器。数据统一保存在archer_results_trans，每次显式指定文件；不安装任务数据、不猜测最新日期。通过同名_report.json定位绑定的原始CSV/summary/events；迁移原始记录时可加run_dir:=/新原始目录。

仅核对协同文件，不连接SDK：

```bash
ros2 run archer_hardware hardware_grasp --check-only --trajectory-file "/转换目录/state.json"
```

直接console入口与launch使用同一实现，可通过 `ros2 run archer_hardware hardware_grasp --help` 或 `hardware_circle --help` 查看参数。正常使用优先上面的launch；无需同时运行hardware_position。

## 每份实现文件

| 文件 | 基本功能 |
|---|---|
| [__init__.py](archer_hardware/__init__.py) | 声明Python包，不启动节点或任务。 |
| [circle_import.py](archer_hardware/circle_import.py) | 画圆CSV转换与同名输出替换，可显式启用连续样条、共同时间重分配和圆弧/边界平滑。 |
| [circle_task.py](archer_hardware/circle_task.py) | 画圆真机入口；复用共用整轮调度，保持现有命令、参数和运动策略。 |
| [commissioning.py](archer_hardware/commissioning.py) | 提供只读反馈验收、小幅试动文件生成及普通双臂计划校验，不连接SDK或发送控制命令. |
| [cooperative_feedback.py](archer_hardware/cooperative_feedback.py) | 211运行反馈：双J2有界修正、归位软边界、末点保持和搬运双J3三极点低通；内存观测保留但不注册全局任务列表。 |
| [cooperative_timing.py](archer_hardware/cooperative_timing.py) | 离线两窗口优化：正C3共同时间密度、固定节点和约束最小jerk/snap重建；不参与SDK实时发送。 |
| [driver_node.py](archer_hardware/driver_node.py) | 发布双臂实机只读反馈，默认不连接设备，不接收运动指令. |
| [file_task.py](archer_hardware/file_task.py) | 画圆/协同公共全流程：离线预检、SDK准备、实际起点衔接、连续回放、归位和退出；显式注入协同后端。 |
| [grasp_events.py](archer_hardware/grasp_events.py) | 核对协同任务的夹爪位置与开爪后直接进给时序，不调用 ROS 或 SDK、不确认真实夹持。 |
| [grasp_import.py](archer_hardware/grasp_import.py) | 协同CSV通用转换，保留关节轨迹、夹爪事件和来源哈希；不会隐式启用211特定输入优化。 |
| [grasp_smoothing.py](archer_hardware/grasp_smoothing.py) | 按协同阶段拟合有界五次样条，抑制仿真采样微波动，保留事件停顿与双臂共同时间。 |
| [grasp_task.py](archer_hardware/grasp_task.py) | 协同真机任务入口：读取已转换轨迹，按共同时间开合夹爪，整轮归位后关闭。 |
| [gripper_adapter.py](archer_hardware/gripper_adapter.py) | GP80复用机械臂SDK连接；显式启用才启动官方夹爪周期，不发送力矩或零力矩。 |
| [hex_adapter.py](archer_hardware/hex_adapter.py) | 封装hex_device依赖、反馈解析、设备状态和连接清理；仍需要Ubuntu安装的官方SDK。 |
| [operator_client.py](archer_hardware/operator_client.py) | 提供显式真机操作及离线检查入口，不直接调用SDK或自动使能. |
| [position_adapter.py](archer_hardware/position_adapter.py) | 独立的 SDK 关节位置控制接入，默认只连接反馈，不自动申请控制权. 调用方需显式申请控制并持续调用 tick；软件制动请求不等同于实体急停。 |
| [position_feedback.py](archer_hardware/position_feedback.py) | 协同双J2有界平滑反馈滤波；不导入ROS/SDK或发送指令。 |
| [position_node.py](archer_hardware/position_node.py) | 提供显式连接、使能、目标预检、执行和停止的 ROS 位置接口. |
| [position_plan.py](archer_hardware/position_plan.py) | 检查关节限制并生成位置轨迹与双臂启动屏障，不连接设备. |
| [position_worker.py](archer_hardware/position_worker.py) | 以每臂独立线程调度位置接口，服务回调不直接访问 SDK. |
| [sdk_cycle.py](archer_hardware/sdk_cycle.py) | 将项目目标交给官方 SDK 周期发送，逐包复核权限、有效期和原始编码. |
| [sdk_position.py](archer_hardware/sdk_position.py) | 为官方 SDK 位置命令隔离每臂配置与历史；本模块不连接设备或启动任务. |
| [task_file.py](archer_hardware/task_file.py) | 离线核对成功任务的原始记录与连续轨迹，输出证据但不连接ROS或SDK. |
| [task_geometry.py](archer_hardware/task_geometry.py) | 离线拼接接近、文件任务与归位轨迹，并对连续插值做离散几何预检. |

## 配置、安装与测试文件

| 配置 | 功能 |
|---|---|
| [config/hardware.yaml](config/hardware.yaml) | 只读左右臂反馈节点的连接/发布参数。 |
| [config/position_control.yaml](config/position_control.yaml) | 底层位置驱动基础配置，默认不自动连接、不开放完整任务许可。 |
| [config/circle_real.yaml](config/circle_real.yaml) | 画圆转换与整轮执行的绑定配置。 |
| [config/grasp_real.yaml](config/grasp_real.yaml) | 协同转换与执行配置，包含两臂范围、250Hz、0.5rad/s和0.3rad/s²上限。 |

YAML字节参与SHA256报告绑定。本轮保留全部真机YAML字节，避免已有转换因仅改注释失效。其中历史“尚未接入夹爪/未实机”注释不能代表当前能力，以本文和正式入口为准。直接启动底层驱动时许可默认关闭；完整调度器会显式设置本次所需许可与运行误差界限，不意味着修改配置文件。

| 文件/目录 | 用途 |
|---|---|
| [package.xml](package.xml) | ROS依赖、包名与ament构建类型。 |
| [setup.py](setup.py) | 注册Python可执行入口并安装配置和README。 |
| [setup.cfg](setup.cfg) | 把console脚本安装到ROS包的lib目录。 |
| resource/archer_hardware | ament索引标记，不是执行程序。 |

测试文件是开发回归，不随正常任务自动运行。功能回归与完整ament检查方法见项目根README；版权/许可证检查存在占位或跳过，不把功能测试通过写成全部发布检查通过。

| 文件 | 基本功能 |
|---|---|
| [test_circle_import.py](test/test_circle_import.py) | 验证仿真CSV离线导入、原始时间和角度保留，不连接设备或执行运动. |
| [test_circle_task.py](test/test_circle_task.py) | 验证正式画圆入口的许可、路径及ROS参数处理，不连接机械臂。 |
| [test_cooperative_carry_feedback.py](test/test_cooperative_carry_feedback.py) | 离线验证搬运J3修正与成功204其他阶段隔离，不建立设备连接。 |
| [test_cooperative_package.py](test/test_cooperative_package.py) | 正式协同入口的默认文件、显式许可和后端注入，不连接设备。 |
| [test_cooperative_return_feedback.py](test/test_cooperative_return_feedback.py) | 验证192仅影响归位及结束交接，归位前与冻结189逐周期一致；不连接硬件。 |
| [test_copyright.py](test/test_copyright.py) | 检查功能包源文件的版权声明格式，不启动节点或连接设备. |
| [test_feedback_adapter.py](test/test_feedback_adapter.py) | 离线验证反馈适配器，所有测试均禁止连接实际设备. |
| [test_feedback_audit.py](test/test_feedback_audit.py) | 离线验证反馈验收，ROS 消息使用合成数据，不连接设备. |
| [test_feedback_node.py](test/test_feedback_node.py) | 在真实 ROS 通信中验证只读节点，设备连接始终用替身隔离. |
| [test_flake8.py](test/test_flake8.py) | 仅检查当前功能包的 Python 样式，并处理检查器初始化失败. |
| [test_grasp_import.py](test/test_grasp_import.py) | 检查协同事件、连续时间轴、命令分发与失败保留旧转换，不连接真实设备。 |
| [test_grasp_smoothing.py](test/test_grasp_smoothing.py) | 检查协同样条的共同时间、位置边界、事件保持时长和故障说明，不连接真实设备。 |
| [test_grasp_task.py](test/test_grasp_task.py) | 检查协同入口许可、输入文件绑定及夹爪连续时序，不创建SDK连接。 |
| [test_gripper_adapter.py](test/test_gripper_adapter.py) | 以SDK替身验证夹爪发送准入、过期取消、端口隔离和制动，不连接设备。 |
| [test_gripper_sdk.py](test/test_gripper_sdk.py) | 运行官方Hands周期并截获全部传输，核对真实编码与BRAKE，不创建网络API。 |
| [test_joint_trial.py](test/test_joint_trial.py) | 离线验证试动文件、普通双臂计划校验，不连接SDK或驱动机械臂. |
| [test_operator_client.py](test/test_operator_client.py) | 离线验证显式操作客户端，服务后端仅为内存替身. |
| [test_pep257.py](test/test_pep257.py) | 从任意工作目录运行时，都只检查当前功能包的文档字符串. |
| [test_position_control.py](test/test_position_control.py) | 离线验证位置控制，所有发送均被替身接收，不连接实际设备. |
| [test_position_node.py](test/test_position_node.py) | 通过真实 ROS 服务和线程验证位置节点，设备始终使用离线替身. |
| [test_sdk_cycle.py](test/test_sdk_cycle.py) | 以内存传输运行真实 SDK 周期，验证位置、取消、过期和退出，不连接设备. |
| [test_sdk_position.py](test/test_sdk_position.py) | 使用真实 SDK 离线验证双臂历史隔离和目标拒改，不建立网络连接. |
| [test_task_file.py](test/test_task_file.py) | 验证整轮任务文件核对及篡改拒绝，不连接设备或执行动作. |
| [test_task_geometry.py](test/test_task_geometry.py) | 验证离线接近、连续路径加密采样及碰撞拒绝，不连接实机. |

## 数据与接口规则

转换默认写入archer_results_trans/原任务目录名，画圆tracking.json+tracking_report.json，协同state.json+state_report.json。同一输出目录成功转换时替换同名文件，写第二文件失败会回滚；失败不发布半套数据。

JSON包含左右六关节位置、时间和连续轨迹导数/停止节点；报告保存输出与输入SHA、限位配置、时间映射、阶段、夹爪事件和检查指标。不是“任何CSV读取后就立即运动”：转换只写文件，显式执行入口才接入机器人。

- 左臂CAN0/8439，右臂CAN1/9439，共同控制器192.168.1.100。软件共同参考时间不等于CAN总线硬件同步。
- 官方SDK250Hz意味着约4ms一次周期取样；原CSV无需逐行与4ms对应，执行器在连续曲线上取样。
- 整轮画圆使用0.05rad、协同0.045rad的统一预检/运行界限，中间点连续通过，不逐点等到位。
- POSITION用于轨迹目标；gripper_adapter管理GP80夹爪事件。双J2偏移与搬运双J3修正有各自幅度/速度/加速度及平滑窗口，不是第二个SDK发送线程。
- 新鲜实际初值与计划初值不同，由有界接近段衔接；正常末段回本轮初始目标。SDK初始化/必要校准、指定一次API超时清除沿用既有规则，运行中故障不自动清除重试。
- 本项目SDK仍来自hex_device；删除厂商SDK不能继续接入真机。封装用于集中处理版本、状态、新鲜度和关闭，不是复制厂商实现。

## 底层联调入口

`hardware_feedback`是只读反馈节点；`hardware_position`是显式位置控制服务节点，默认不连接。完整任务launch不需要它们另行启动。
`hardware_operator`的status/connect/enable/move/recover-limits/halt/disconnect/task-run用于高级联调，涉及连接/使能/运动，不能当成查看帮助自动执行。audit读取已发布反馈生成开发报告；trial只预览单关节/入限候选，不自动发运动。具体参数使用子命令 `--help`。

底层发布 `/hardware/position/status`，左右 `/hardware/position/...` 状态由驱动提供；软件halt取消活动请求但本身不等于退出连接。正常退出使用运行任务的前台终端Ctrl+C，公共流程取消、尽力BRAKE并关闭，不发送零力矩、MIT或故障归位。GUI窗口退出不等同于关闭另一终端的真机驱动。

## 算法与辅助模块为何分开

cooperative_feedback是运行时反馈，cooperative_timing是离线时间规划，两者属于同一211版本。J6/分臂提速已写入当前转换JSON，运行时不重新搜索时间最优解。通用grasp-import并未自动升级为对所有新CSV应用211的特定输入策略。

circle_task和grasp_task保持独立，公共流程已在file_task共享；它们的文件类型、许可、夹爪与后端不同。FeedbackCooperativeAdapter、ReturnFeedbackAdapter和CarryFeedbackAdapter包含实质算法，归位故意绕过普通阶段的J2分支，不能机械换成super或折叠为重复修正。

## 输出与验证边界

正常完整执行只输出终端状态，默认不生成任务结果文件。内存观测用于关闭摘要及故障上下文，最大50Hz；正式协同缓存8000条，不会因为进程结束自动写入archer_results。开发验收数据放archer_validation或开发缓存。

旧~/.cache/archer_development实验入口绑定旧安装源码摘要，不与正式包混用。206历史归档保留；需要回退应恢复匹配源码/配置/轨迹，而不是强行修改报告哈希。维护者和许可证占位仍需由项目拥有者确认。
