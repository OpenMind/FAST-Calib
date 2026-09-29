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
                "live_capture_seconds": LaunchConfiguration("live_capture_seconds"),
                "rtsp_warmup_frames": LaunchConfiguration("rtsp_warmup_frames"),
                "lidar_type": LaunchConfiguration("lidar_type"),
                "lidar_frame": LaunchConfiguration("lidar_frame"),
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
                default_value="",
                description="Path to a ROS 2 bag directory (containing metadata.yaml) "
                "with a recorded LiDAR topic. Leave empty (the default) to instead "
                "subscribe to lidar_topic live for live_capture_seconds.",
            ),
            DeclareLaunchArgument(
                "image_path",
                default_value="/camera/front/image_raw",
                description="ROS image topic to grab a live frame from (leading '/', "
                "e.g. /camera/front/image_raw), a path to a calibration image file, "
                "or an rtsp:// URL.",
            ),
            DeclareLaunchArgument(
                "output_path",
                default_value=os.path.join(package_path, "output"),
                description="Directory to write calibration results to",
            ),
            DeclareLaunchArgument(
                "lidar_topic",
                default_value="/lidar_points_front",
                description="LiDAR topic to read (from the bag, or subscribed to live "
                "when bag_path is empty)",
            ),
            DeclareLaunchArgument(
                "live_capture_seconds",
                default_value="3.0",
                description="How long to accumulate points from lidar_topic when "
                "capturing live (bag_path empty)",
            ),
            DeclareLaunchArgument(
                "lidar_type",
                default_value="grid",
                description="Circle detector: 'grid' (occupancy-grid holes; works for "
                "any scan pattern incl. MEMS/rosette scanners like this one), 'auto' "
                "(ring field -> mech), 'solid' (normal-based boundaries), or 'mech' "
                "(per-ring gap detection)",
            ),
            DeclareLaunchArgument(
                "lidar_frame",
                default_value="zfwd",
                description="Axis convention of the lidar frame: 'zfwd' (camera-style: "
                "z forward, x right, y up — matches /lidar_points_front) or 'xfwd' "
                "(ROS standard: x forward, y left, z up) or 'nyfwd' (ROS axes, lidar yawed 90 "
                "deg so the camera looks along its -Y)",
            ),
            DeclareLaunchArgument(
                "rtsp_warmup_frames",
                default_value="15",
                description="Frames to discard after opening the RTSP stream before "
                "keeping one, so the decoder is past any stale buffered frames",
            ),
            fast_calib_node,
            rviz_node,
        ]
    )
