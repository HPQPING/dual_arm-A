# Archer双臂任务工作空间

更新：2026-10-29。ROS 2 Humble / Ubuntu。项目有两个主任务：**双臂画圆**和**简化协同搬运**。相同五个功能包提供仿真、成功任务记录、CSV转换和真实机械臂连续位置回放。

## 五个功能包与职责

| 顺序 | 功能包 | 职责与说明 |
|---|---|---|
| 1 | [archer_description](src/archer_description/README.md) | 六关节机械臂、GP80刚性网格、左右安装和TCP模型。 |
| 2 | [archer_mujoco](src/archer_mujoco/README.md) | MuJoCo模型加载、共享双臂动力学、反馈与辅助验证台。 |
| 3 | [archer_control](src/archer_control/README.md) | 运动学、轨迹规划、两类仿真任务、成功记录与开发检查。 |
| 4 | [archer_hardware](src/archer_hardware/README.md) | CSV/JSON核对、官方SDK反馈、POSITION连续回放及真实夹爪时序。 |
| 5 | [archer_bringup](src/archer_bringup/README.md) | 一条launch组织仿真/显示或完整真机任务。 |

```text
archer_ws/
├── README.md
├── src/
│   ├── archer_description/
│   ├── archer_mujoco/
│   ├── archer_control/
│   ├── archer_hardware/
│   └── archer_bringup/
├── archer_results/          # 成功仿真原始CSV与JSON，每轮独立目录
├── archer_results_trans/    # 转换后轨迹JSON与报告，同名转换成功替换旧结果
├── archer_validation/       # 开发验收/诊断记录，不是正常任务输入
├── build/                  # colcon生成
├── install/                # ROS可执行入口与安装资源
└── log/                    # 构建和测试日志
```

## 环境与构建

Python解释器需要同时能导入ROS 2和MuJoCo。CSV无需Conda；不能因为读取CSV就切换到缺少rclpy的环境。真机通过Ubuntu上的厂商hex_device SDK接入，SDK不是项目随意替换的依赖。

```bash
cd ~/archer_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select archer_description archer_control archer_mujoco archer_hardware archer_bringup --symlink-install
source install/setup.bash
```

每个新终端先加载环境：

```bash
source /opt/ros/humble/setup.bash && source ~/archer_ws/install/setup.bash
```

常用数值/显示依赖包括NumPy、PyYAML、SciPy、MuJoCo、xacro、RViz2和robot_state_publisher。SciPy用于部分离线优化；当前机器的用户NumPy与系统SciPy存在版本混用风险。只在离线数值测试中按需要使用兼容隔离环境；不要把PYTHONNOUSERSITE=1盲目用于需要用户目录SDK的真机启动。

## 两个任务的完整流程

仿真启动后先等待READY，再调用对应开始服务。不要把画圆和搬运服务混用。

| 任务 | 启动仿真 | 另一个终端开始 |
|---|---|---|
| 画圆 | `ros2 launch archer_bringup dual_circle_sim.launch.py` | `ros2 service call /dual/task/start std_srvs/srv/Trigger '{}'` |
| 协同搬运 | `ros2 launch archer_bringup dual_grasp_sim.launch.py` | `ros2 service call /dual/grasp/start std_srvs/srv/Trigger '{}'` |

两者默认MuJoCo窗口和RViz均为true。无图形环境加 `show_viewer:=false show_rviz:=false`。画圆另有 `record:=false` 关闭自动记录。正常任务包含计划内归位；异常中断不会为了归位继续运动。

成功记录目录采用**任务开始时的日期_时间_微秒_任务名称**；失败或中止的临时采样不发布到archer_results。画圆通常有tracking.csv和summary.json；协同有state.csv、events.csv和summary.json。CSV保存轨迹/状态，events记录阶段，summary记录配置、完成状态与误差等摘要；不是视频、图片或控制程序。部分辅助实验可额外输出文件，见各包说明。

### 转换仿真结果（不连接真机）

```bash
ros2 run archer_hardware hardware_operator circle-import --input "/原始任务目录/tracking.csv" --profile circle
ros2 run archer_hardware hardware_operator grasp-import --input "/原始任务目录/state.csv"
```

默认输出到 `~/archer_ws/archer_results_trans/原任务目录名/`。画圆输出tracking.json和tracking_report.json；协同输出state.json和state_report.json。文件名随输入保持，成功二次转换替换同一对文件；失败保留旧转换。原始CSV、summary与协同events仍需保留，报告用SHA256绑定原始数据、配置和转换文件。

**当前211说明：** 正式协同入口包含沿用206实机认可的双J2/搬运双J3反馈修正；211另优化J6调姿和归位开始分臂的名义时间。当前211转换文件已独立生成并离线验收。普通grasp-import保留其原有通用转换链，不会默默将211的特定基准优化套到所有新CSV。离线优化接口是archer_hardware.cooperative_timing.optimize_transitions，需重新检查最终曲线后使用。

### 选择转换后文件并执行真机（会运动）

**每次必须显式给轨迹JSON地址**；不设置默认任务路径，不在功能包内保存轨迹副本。只传tracking.json或state.json，程序自动读取同名_report.json以及报告绑定的原始记录。

```bash
ros2 launch archer_bringup dual_circle_real.launch.py execute:=true trajectory_file:="/转换目录/tracking.json"
ros2 launch archer_bringup dual_grasp_real.launch.py execute:=true trajectory_file:="/转换目录/state.json"
```

单条launch已经负责完整调度，不另开hardware_position，不并行使用旧缓存脚本。省略execute:=true不会创建真机控制任务。协同文件可先只读本地核对：

```bash
ros2 run archer_hardware hardware_grasp --check-only --trajectory-file "/转换目录/state.json"
```

执行流程为：连接前核对文件和完整路径 → SDK双臂连接与读取反馈 → 官方SDK准备 → 重新取得本轮真实起点和归位目标 → 核对接近/归位衔接 → 按共用时间连续发送接近、主体和归位 → 尽力BRAKE并关闭连接。中间采样点连续通过，不逐点等待。机器人192.168.1.100，左CAN0:8439、右CAN1:9439；项目默认250Hz，速度上限0.5rad/s、加速度0.3rad/s²，整轮画圆误差边界0.05rad、协同0.045rad。它们是项目配置，不是厂商额定加速度声明。

真机起点来自新鲜反馈，不采用仿真固定角度代替。初值工程容差、目标限位及归位投影由现有规则处理，不通过篡改CSV或关闭保护解决。归位回本轮保存的初始目标；异常停止不故障归位、不发送零力矩、不自动重试。前台执行终端Ctrl+C取消两臂活动任务并尽力关闭；没有独立键盘监听终端。

## 当前验证状态与边界

- 画圆已有多轮完整真机播放及归位记录；206协同已获得用户实际观察认可。
- 正式包当前为211（含206已验证反馈逻辑与两处时间优化）；211本轮只完成离线几何、真实SDK内存编码、安装入口与回归检查，尚未获得该整合版真机平稳性结论。
- 仿真GP80是刚性网格，显示完整模型但无真实手指开合；协同工件运动使用明确的演示附着。真实夹爪由SDK控制5.65/5.5；不把仿真附着或电机位置数值当作夹持力验证。
- 当前使用官方SDK周期POSITION发送，尚未接入ros2_control/FollowJointTrajectory。joint_states是反馈，不是运动指令入口。
- 真机正常运行默认只向终端输出状态，不自动向archer_results写文件。archer_validation与开发缓存是诊断材料；删除不会产生新的控制能力，也不是重放输入。记录是否保留由开发需要决定，既有归档不在本轮清理范围。

## 开发回归

本机数值转换与SDK使用不同的既有Python依赖组合。功能回归分两组，避免把系统旧protobuf带入SDK，或让用户NumPy与系统SciPy混用；不修改全局依赖。

在工作空间根目录，先运行保留用户SDK环境的非数值子集：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest src/archer_bringup/test src/archer_description/test src/archer_control/test src/archer_mujoco/test src/archer_hardware/test -q --ignore-glob='*/test_flake8.py' --ignore-glob='*/test_pep257.py' --ignore-glob='*/test_copyright.py' --ignore=src/archer_control/test/test_continuous_trajectory.py --ignore=src/archer_control/test/test_transit_timing.py --ignore=src/archer_control/test/test_recorded_timing.py --ignore=src/archer_hardware/test/test_circle_import.py --ignore=src/archer_hardware/test/test_grasp_smoothing.py --ignore=src/archer_hardware/test/test_task_file.py
```

再仅对纯离线数值测试使用系统NumPy/SciPy组合：

```bash
PYTHONNOUSERSITE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /usr/bin/python3 -m pytest src/archer_control/test/test_continuous_trajectory.py src/archer_control/test/test_transit_timing.py src/archer_control/test/test_recorded_timing.py src/archer_hardware/test/test_circle_import.py src/archer_hardware/test/test_grasp_smoothing.py src/archer_hardware/test/test_task_file.py -q
```

如果本机离线转换也出现SciPy二进制不兼容，可将 `PYTHONNOUSERSITE=1` 仅加在CSV转换命令前；不要把它设置为真机运行的全局环境变量，因为SDK位于用户目录。以上是当前机器环境的具体处理方式，不是所有部署机器都必须拆分的算法要求。

完整ament检查还包括代码风格/文档风格及元数据检查：`colcon test --packages-select archer_description archer_control archer_mujoco archer_hardware archer_bringup`，随后 `colcon test-result --verbose`。不要把省略风格和版权项的功能回归称为全部检查通过。部分包维护者/许可证仍为占位，发布前由维护者确认，不能猜测填入许可证。

开发检查、静态姿态观察和回放见[control检查目录](src/archer_control/archer_control/checks/README.md)。两份主仿真launch不自动启动真机；辅助单臂和接触验证入口不增加第三个主任务。
