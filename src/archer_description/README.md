# archer_description：URDF/Xacro、网格与TCP

更新：2026-10-29。本文以当前源码、注册入口和实际安装检查为准。

## 模型内容

六转动关节机械臂、法兰、GP80刚性网格、TCP及左右世界安装关系。本包不运行控制循环、不发布关节指令，也不包含hex_device。真实GP80开合事件由硬件包处理；当前网格没有独立手指活动关节，因此RViz/MuJoCo显示不等于真实开合机构仿真。

## 每份模型与配置文件

| 文件 | 功能 |
|---|---|
| [urdf/archer_arm.xacro](urdf/archer_arm.xacro) | 单臂各link/joint、六轴范围、惯性/碰撞/可视几何宏；加入法兰参考坐标，供夹爪宏连接。 |
| [urdf/gp80.xacro](urdf/gp80.xacro) | GP80刚性网格、法兰到夹爪及夹爪到TCP的固定变换；无手指开合关节。 |
| [urdf/single_arm.xacro](urdf/single_arm.xacro) | 指定left/right前缀生成单臂模型，供显示、单臂及双臂加载器使用。 |
| [urdf/dual_arm.xacro](urdf/dual_arm.xacro) | 把两臂安装到world，间距、高度、旋转及夹爪偏移可由参数传入。 |
| [rviz/archer.rviz](rviz/archer.rviz) | 单臂显示布局与RobotModel/TF设置。 |

### 网格逐项说明

| 网格 | 对应内容 |
|---|---|
| [meshes/base_link.STL](meshes/base_link.STL) | 机械臂底座可视/碰撞网格。 |
| [meshes/link_1.STL](meshes/link_1.STL) | 第一运动link网格。 |
| [meshes/link_2.STL](meshes/link_2.STL) | 第二运动link网格。 |
| [meshes/link_3.STL](meshes/link_3.STL) | 第三运动link网格。 |
| [meshes/link_4.STL](meshes/link_4.STL) | 第四运动link网格。 |
| [meshes/link_5.STL](meshes/link_5.STL) | 第五运动link网格。 |
| [meshes/link_6.STL](meshes/link_6.STL) | 第六运动link网格。 |
| [meshes/gp80/base_link.STL](meshes/gp80/base_link.STL) | GP80整体静态网格，不可凭外观把其拆成可动手指。 |

| 文件/目录 | 用途 |
|---|---|
| [package.xml](package.xml) | ROS依赖、包名与ament构建类型。 |
| [CMakeLists.txt](CMakeLists.txt) | 安装urdf、meshes、rviz和README并注册模型回归。 |

## 单位、坐标和参数

位置用米，关节和rpy用弧度。默认双臂基座间距0.42m，左右X为-0.21/+0.21，两侧基座rpy默认(0,0,pi/2)。实际任务安装参数由bringup配置传入，不能只改RViz而不改规划/仿真。

主任务TCP位于夹爪坐标系+Z的0.147021696m；link6到法兰的固定厚度在模型中，不能再叠加一次。单臂独立示例允许通过launch使用0.147m参数，与其仿真/任务内部一致；不要把这个约21.7微米的示例差异误判为主任务模型错误。

URDF定义的关节范围用于规划/仿真核对，真实驱动还读取独立硬件配置。真机初值从反馈获得，不能从URDF零角推断。

## 调用方式

本包是资源包，通常不使用ros2 run直接启动：

```bash
ros2 launch archer_bringup single_arm_display.launch.py side:=left
ros2 launch archer_bringup dual_arm_view.launch.py
```

第一条提供单臂滑块显示；第二条读取现有关节反馈显示双臂，不会产生仿真反馈或让机械臂运动。主仿真launch默认已包含双臂显示，不需另开重复显示组。

需要展开模型时：

```bash
xacro "$(ros2 pkg prefix --share archer_description)/urdf/dual_arm.xacro" > /tmp/archer_dual_arm.urdf
```

只生成描述文件，不连接设备。修改网格/惯性/坐标后应重新构建description与依赖包，并核对规划和模型一致性。

## 测试文件

| 文件 | 基本功能 |
|---|---|
| [test_models.py](test/test_models.py) | 检查Xacro关节、工具坐标链和安装网格资源，不启动仿真或硬件. |

测试检查Xacro展开、关节和资源。STL均有真实模型引用，本轮保留；维护者/许可证占位不由程序猜测修补。
