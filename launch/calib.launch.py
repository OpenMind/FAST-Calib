"""
FAST-Calib launch file: single-scene LiDAR-camera extrinsic calibration.

Reads a recorded ROS 2 bag + image, computes T_cam_lidar, and (when `rviz` is
true) opens RViz2 with the intermediate point clouds published for inspection.

Example:
    ros2 launch fast_calib calib.launch.py \\
        bag_path:=/path/to/bag image_path:=/path/to/image.jpg
"""

import os.path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """
    Generate the launch description for single-scene FAST-Calib.
    """
    package_path = get_package_share_directory("fast_calib")

    fast_calib_node = Node(
        package="fast_calib",
        executable="fast_calib",
        name="fast_calib",
        output="screen",
        parameters=[
            os.path.join(package_path, "config", "qr_params.yaml"),
            {
                "bag_path": LaunchConfiguration("bag_path"),
                "image_path": LaunchConfiguration("image_path"),
                "output_path": LaunchConfiguration("output_path"),
                "lidar_topic": LaunchConfiguration("lidar_topic"),
            },
        ],
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(package_path, "rviz_cfg", "fast_livo2.rviz")],
        condition=IfCondition(LaunchConfiguration("rviz")),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "rviz", default_value="true", description="Start RViz2"
            ),
            DeclareLaunchArgument(
                "bag_path",
                default_value=os.path.join(
                    package_path, "calib_data", "avia_multi_scene_33"
                ),
                description="Path to the ROS 2 bag directory (containing metadata.yaml) "
                "with the recorded LiDAR topic",
            ),
            DeclareLaunchArgument(
                "image_path",
                default_value=os.path.join(
                    package_path, "calib_data", "avia_multi_scene", "33.jpg"
                ),
                description="Path to the corresponding calibration image",
            ),
            DeclareLaunchArgument(
                "output_path",
                default_value=os.path.join(package_path, "output"),
                description="Directory to write calibration results to",
            ),
            DeclareLaunchArgument(
                "lidar_topic",
                default_value="/livox/lidar",
                description="LiDAR topic recorded in the bag",
            ),
            fast_calib_node,
            rviz_node,
        ]
    )
