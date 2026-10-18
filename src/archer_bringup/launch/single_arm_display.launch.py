"""启动单臂模型、关节滑块、TF与RViz，用于姿态显示，不连接真机."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    side = LaunchConfiguration('side')

    model_path = PathJoinSubstitution([
        FindPackageShare('archer_description'),
        'urdf',
        'single_arm.xacro',
    ])

    robot_description = ParameterValue(
        Command([
            FindExecutable(name='xacro'),
            ' ',
            model_path,
            ' side:=',
            side,
        ]),
        value_type=str,
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'side',
            default_value='left',
            choices=['left', 'right'],
            description='Select the left or right arm model.',
        ),

        # 根据模型和关节状态发布各链接之间的坐标变换。
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='archer_state_publisher',
            parameters=[{
                'robot_description': robot_description,
            }],
            output='screen',
        ),

        # 用滑块产生显示用关节状态，不发送实机指令。
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='archer_joint_state_gui',
            parameters=[{
                'robot_description': robot_description,
            }],
            output='screen',
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='archer_rviz',
            output='screen',
        ),
    ])