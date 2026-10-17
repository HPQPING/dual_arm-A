"""启动单臂 MuJoCo、坐标标记与可选 RViz，使用已注册的仿真入口。"""

from pathlib import Path
import math

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import Command, FindExecutable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _vector_argument(context, name):
    text = LaunchConfiguration(name).perform(context)
    try:
        values = [float(v) for v in text.split()]
    except ValueError as exc:
        raise ValueError(f'{name} must be three numbers separated by spaces') from exc
    if len(values) != 3 or not all(math.isfinite(v) for v in values):
        raise ValueError(f'{name} must contain three finite numbers')
    return values


def _nodes(context):
    side = LaunchConfiguration('side').perform(context)
    tcp = _vector_argument(context, 'tcp_xyz')
    mount_xyz = _vector_argument(context, 'mount_xyz')
    mount_rpy = _vector_argument(context, 'mount_rpy')
    description_share = Path(get_package_share_directory('archer_description'))
    simulation_share = Path(get_package_share_directory('archer_mujoco'))
    config = simulation_share / 'config/simulation.yaml'
    if not config.is_file():
        raise FileNotFoundError(f'Missing simulation configuration: {config}')
    description = ParameterValue(Command([
        FindExecutable(name='xacro'), ' ',
        str(description_share / 'urdf/single_arm.xacro'),
        ' side:=', side,
        ' show_tcp:=false',
        ' tcp_xyz:="', ' '.join(str(v) for v in tcp), '"',
        ' gripper_mount_xyz:="', ' '.join(str(v) for v in mount_xyz), '"',
        ' gripper_mount_rpy:="', ' '.join(str(v) for v in mount_rpy), '"',
    ]), value_type=str)
    simulation_node = Node(
            package='archer_mujoco', executable='sim_node',
            name='single_arm_sim', output='screen',
            parameters=[str(config), {
                'side': side, 'tcp_xyz': tcp, 'mount_xyz': mount_xyz, 'mount_rpy': mount_rpy,
                'use_sim_time': False,
                'show_viewer': ParameterValue(LaunchConfiguration('show_viewer'), value_type=bool),
            }],
        )
    return [
        RegisterEventHandler(OnProcessExit(
            target_action=simulation_node,
            on_exit=[EmitEvent(event=Shutdown(reason='Single-arm simulation exited'))],
        )),
        simulation_node,
        Node(
            package='robot_state_publisher', executable='robot_state_publisher',
            name='archer_state_publisher', output='screen',
            parameters=[{'robot_description': description, 'use_sim_time': False}],
            remappings=[('joint_states', '/' + side + '/joint_states')],
        ),
        Node(
            package='archer_mujoco', executable='gripper_markers',
            name='gripper_markers', output='screen',
            parameters=[{'side': side, 'tcp_xyz': tcp, 'initial_position': 0.0, 'use_sim_time': False}],
        ),
        Node(
            package='rviz2', executable='rviz2', name='archer_rviz', output='screen',
            arguments=['-d', str(description_share / 'rviz/archer.rviz')],
            condition=IfCondition(LaunchConfiguration('show_rviz')),
            parameters=[{'use_sim_time': False}],
        ),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('side', default_value='left', choices=['left', 'right']),
        DeclareLaunchArgument('show_viewer', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('show_rviz', default_value='true', choices=['true', 'false']),
        DeclareLaunchArgument('tcp_xyz', default_value='0 0 0.147'),
        DeclareLaunchArgument('mount_xyz', default_value='0 0 0'),
        DeclareLaunchArgument('mount_rpy', default_value='0 0 0'),
        OpaqueFunction(function=_nodes),
    ])
