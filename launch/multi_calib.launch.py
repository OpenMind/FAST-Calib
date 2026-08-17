"""
FAST-Calib launch file: multi-scene joint LiDAR-camera extrinsic calibration.

Combines the last three `circle_center_record.txt` entries written by
calib.launch.py (single-scene runs) into one joint extrinsic estimate.

Example:
    ros2 launch fast_calib multi_calib.launch.py
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
    Generate the launch description for multi-scene FAST-Calib.
    """
    package_path = get_package_share_directory("fast_calib")

    multi_fast_calib_node = Node(
        package="fast_calib",
        executable="multi_fast_calib",
        name="multi_fast_calib",
        output="screen",
        parameters=[
            os.path.join(package_path, "config", "qr_params.yaml"),
            {"output_path": LaunchConfiguration("output_path")},
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
                "rviz", default_value="false", description="Start RViz2"
            ),
            DeclareLaunchArgument(
                "output_path",
                default_value=os.path.join(package_path, "output"),
                description="Directory holding circle_center_record.txt "
                "(written by calib.launch.py) and where multi_calib_result.txt is saved",
            ),
            multi_fast_calib_node,
            rviz_node,
        ]
    )
