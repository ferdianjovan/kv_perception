"""Start smoke detection for a drone's bridged RGB camera."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """Create the detector launch with independently configurable topics/range."""
    return LaunchDescription([
        DeclareLaunchArgument('drone_name', default_value='x500_depth'),
        DeclareLaunchArgument(
            'config',
            default_value=PathJoinSubstitution([
                FindPackageShare('kv_perception'), 'config', 'smoke_detector.yaml',
            ]),
        ),
        DeclareLaunchArgument('image_topic', default_value='rgb_camera/image'),
        DeclareLaunchArgument('camera_info_topic', default_value='rgb_camera/camera_info'),
        DeclareLaunchArgument(
            'assumed_distance_m', default_value='10.0',
            description='Assumed forward camera-to-smoke distance; zero disables metres.',
        ),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        Node(
            package='kv_perception',
            executable='smoke_detector.py',
            name='smoke_detector',
            namespace=LaunchConfiguration('drone_name'),
            output='screen',
            parameters=[
                LaunchConfiguration('config'),
                {
                    'image_topic': LaunchConfiguration('image_topic'),
                    'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                    'assumed_distance_m': ParameterValue(
                        LaunchConfiguration('assumed_distance_m'), value_type=float,
                    ),
                    'use_sim_time': ParameterValue(
                        LaunchConfiguration('use_sim_time'), value_type=bool,
                    ),
                },
            ],
        ),
    ])
