#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
1) 自动检测 ROS 2 bag 中雷达点云类型：
   - sensor_msgs/msg/PointCloud2  (如 /hesai/pandar)
   - livox_ros_driver2/msg/CustomMsg (如 /livox/lidar)
2) 按各自的解析方式把点云导出成一个带 intensity 的 PCD 文件 (x y z intensity, ASCII)
3) 使用 Open3D 对该 PCD 进行交互选点（至少 4 个点），并根据 4 个点计算包围范围，
   保存为同名 .txt 文件。

依赖：
    - rosbag2_py, rclpy (随 ROS 2 安装)
    - sensor_msgs_py
    - open3d, numpy
    - livox_ros_driver2 (可选，仅用于 Livox CustomMsg 输入；未安装时自动跳过)

用法示例：
    python3 distance_filter_tool.py
    python3 distance_filter_tool.py /path/to/bag_dir /path/to/output_dir
"""

import os
import sys
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d

try:
    from livox_ros_driver2.msg import CustomMsg as LivoxCustomMsg
except ImportError:
    LivoxCustomMsg = None

# ===================== 通用：打开 bag，遍历消息 =====================


def open_bag_reader(bag_path):
    """打开一个 ROS 2 bag 目录（包含 metadata.yaml），返回 (reader, topic->type 字典)"""
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)
    topic_types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    return reader, topic_types


def iter_topic_messages(bag_path, topic_name, msg_type):
    """依次反序列化 bag 中某个 topic 的全部消息"""
    reader, _ = open_bag_reader(bag_path)
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic == topic_name:
            yield deserialize_message(data, msg_type)


# ===================== 通用：保存 PCD =====================


def save_pcd_with_intensity(points, intensities, output_path):
    """
    保存点云为带 intensity 字段的 PCD 文件 (ASCII 格式)
    points: list/ndarray of [x, y, z]
    intensities: list/ndarray of intensity
    """
    N = len(points)
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {N}
HEIGHT 1
POINTS {N}
DATA ascii
"""
    with open(output_path, 'w') as f:
        f.write(header)
        for (x, y, z), inten in zip(points, intensities):
            f.write(f"{x} {y} {z} {inten}\n")
    print(f"[PCD] 保存带 intensity 字段的点云到: {output_path}")

# ===================== 情况 1：PointCloud2 =====================


def find_intensity_field(msg):
    """在 PointCloud2 的 fields 中自动检测强度字段名称"""
    candidates = ["intensity", "reflectivity", "i", "ref"]
    for field in msg.fields:
        if field.name.lower() in candidates:
            return field.name
    return None


def convert_pointcloud2_bag_to_pcd(
    bag_path,
    output_dir,
    topic_name="/hesai/pandar",                        # 如有不同，可改成 topic 名称
    pcd_name="sensor_PointCloud2_inten_ascii.pcd"
):
    """
    将 ROS 2 bag 中 PointCloud2 类型的点云合并导出为一个 PCD 文件。
    保持原始雷达坐标，不做坐标变换。
    """
    print(f"[Bag] 打开 ROS 2 bag: {bag_path}")

    all_points = []
    all_intensities = []
    intensity_field = None

    print(f"[Bag] 开始从 topic '{topic_name}' 读取 PointCloud2 点云...")
    for msg in iter_topic_messages(bag_path, topic_name, PointCloud2):
        if intensity_field is None:
            intensity_field = find_intensity_field(msg)
            if intensity_field:
                print(f"[Bag] 检测到 intensity 字段: {intensity_field}")
            else:
                print("[ERROR] 未找到强度字段! 退出 PointCloud2 转换。", file=sys.stderr)
                return None

        pts = pc2.read_points(msg, field_names=["x", "y", "z", intensity_field], skip_nans=True)
        for x, y, z, inten in pts:
            all_points.append([float(x), float(y), float(z)])
            all_intensities.append(float(inten))

    if not all_points:
        print("[ERROR] 未找到 PointCloud2 点云数据！", file=sys.stderr)
        return None

    output_path = os.path.join(output_dir, pcd_name)
    save_pcd_with_intensity(all_points, all_intensities, output_path)
    return output_path

# ===================== 情况 2：Livox CustomMsg =====================


def parse_livox_custom_msg(msg):
    """
    从 livox_ros_driver2/msg/CustomMsg 中解析 x, y, z, reflectivity
    """
    points = []
    intensities = []

    for pt in msg.points:
        points.append([pt.x, pt.y, pt.z])
        intensities.append(pt.reflectivity)

    return points, intensities


def convert_livox_custom_bag_to_pcd(
    bag_path,
    output_dir,
    topic_name="/livox/lidar",                     # 如有不同，可改成 topic 名称
    pcd_name="livox_CustomMsg_inten_ascii.pcd"
):
    """
    将 ROS 2 bag 中 livox_ros_driver2/msg/CustomMsg 类型的点云合并导出为一个 PCD 文件。
    保持原始雷达坐标，不做坐标变换。
    """
    if LivoxCustomMsg is None:
        print("[ERROR] livox_ros_driver2 未安装/未 source，无法解析 CustomMsg。", file=sys.stderr)
        return None

    print(f"[Bag] 打开 ROS 2 bag: {bag_path}")
    print(f"[Bag] 开始从 topic '{topic_name}' 读取 CustomMsg 点云...")

    all_points = []
    all_intensities = []
    for msg in iter_topic_messages(bag_path, topic_name, LivoxCustomMsg):
        pts, intens = parse_livox_custom_msg(msg)
        all_points.extend(pts)
        all_intensities.extend(intens)

    if not all_points:
        print("[ERROR] 未找到 Livox CustomMsg 点云数据!", file=sys.stderr)
        return None

    output_path = os.path.join(output_dir, pcd_name)
    intensities = np.array(all_intensities, dtype=np.float32)
    save_pcd_with_intensity(all_points, intensities, output_path)
    return output_path

# ===================== 自动检测：这个 bag 用哪种方式 =====================


def find_topic_of_type(topic_types, msg_type):
    """返回 bag 中第一个匹配 msg_type 的 topic 名称（如有多个，打印提示并取第一个）"""
    matches = [name for name, t in topic_types.items() if t == msg_type]
    if len(matches) > 1:
        print(f"[Detect] 检测到多个 {msg_type} topic: {matches}，默认使用 {matches[0]}。")
    return matches[0] if matches else None


def detect_lidar_msg_type(bag_path):
    """
    读取 bag 的 topic 列表，检测是否有 PointCloud2 或 Livox CustomMsg，并返回其 topic 名称。
    返回：
        ("PointCloud2", topic_name), ("CustomMsg", topic_name), 或 (None, None)
    如果两种都有，默认优先 PointCloud2,并打印提示。
    """
    print(f"[Detect] 扫描 bag: {bag_path}")
    _reader, topic_types = open_bag_reader(bag_path)
    types = set(topic_types.values())

    has_pc2 = "sensor_msgs/msg/PointCloud2" in types
    has_livox = "livox_ros_driver2/msg/CustomMsg" in types

    if has_pc2 and has_livox:
        print("[Detect] 同时检测到 PointCloud2 和 Livox CustomMsg, 默认使用 PointCloud2。")
        return "PointCloud2", find_topic_of_type(topic_types, "sensor_msgs/msg/PointCloud2")
    elif has_pc2:
        topic_name = find_topic_of_type(topic_types, "sensor_msgs/msg/PointCloud2")
        print(f"[Detect] 检测到 PointCloud2 点云，topic: {topic_name}")
        return "PointCloud2", topic_name
    elif has_livox:
        topic_name = find_topic_of_type(topic_types, "livox_ros_driver2/msg/CustomMsg")
        print(f"[Detect] 检测到 Livox CustomMsg 点云，topic: {topic_name}")
        return "CustomMsg", topic_name
    else:
        print("[Detect] 未检测到 PointCloud2 或 Livox CustomMsg 点云。")
        return None, None

# ===================== Open3D 交互选点 & 保存范围 =====================


def select_and_save_points(pcd_folder, target_pcd_name):
    """
    在给定目录中读取指定 PCD 文件，用 Open3D 交互式选点并保存范围。
    """
    pcd_path = os.path.join(pcd_folder, target_pcd_name)
    if not os.path.isfile(pcd_path):
        print(f"[ERROR] 指定的 PCD 文件不存在: {pcd_path}", file=sys.stderr)
        return

    # 读取点云
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"[ERROR] {target_pcd_name} 中没有点云数据，已跳过", file=sys.stderr)
        return

    print(f"\n正在处理: {target_pcd_name}")
    print("请在可视化窗口中按住 Shift 用鼠标左键选择点(至少4个)，然后按 Q 键关闭窗口")

    # 创建可视化窗口并添加点云
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name=f"选择点 - {target_pcd_name}")
    vis.add_geometry(pcd)

    # 等待用户交互（Shift+左键选点, Q 退出）
    vis.run()
    vis.destroy_window()

    # 获取用户选择的点的索引
    selected_indices = vis.get_picked_points()

    if not selected_indices:
        print(f"[ERROR] 未选择任何点，{target_pcd_name} 没有保存文件", file=sys.stderr)
        return

    if len(selected_indices) < 4:
        print(f"[ERROR] 只选中了 {len(selected_indices)} 个点，少于 4 个，跳过该文件", file=sys.stderr)
        return

    # 只取前 4 个点
    selected_indices = selected_indices[:4]

    all_points = np.asarray(pcd.points)
    selected_points = all_points[selected_indices, :]  # 形状 (4, 3)

    # 计算四个点在各轴上的最小值和最大值
    mins = selected_points.min(axis=0)  # [x_min_raw, y_min_raw, z_min_raw]
    maxs = selected_points.max(axis=0)  # [x_max_raw, y_max_raw, z_max_raw]

    # 按你的定义扩展 0.2m
    x_min = mins[0] - 0.2
    x_max = maxs[0] + 0.2
    y_min = mins[1] - 0.2
    y_max = maxs[1] + 0.2
    z_min = mins[2] - 0.2
    z_max = maxs[2] + 0.2

    # 生成保存文件名 (与 PCD 文件同名，改为 txt)
    base_name = os.path.splitext(target_pcd_name)[0]
    save_file = os.path.join(pcd_folder, f"{base_name}.txt")

    with open(save_file, 'w') as f:
        f.write("# 4 selected points (x y z)\n")
        for p in selected_points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        f.write("# range values in order:\n")
        f.write(f"x_min: {x_min:.1f}\n")
        f.write(f"x_max: {x_max:.1f}\n")
        f.write(f"y_min: {y_min:.1f}\n")
        f.write(f"y_max: {y_max:.1f}\n")
        f.write(f"z_min: {z_min:.1f}\n")
        f.write(f"z_max: {z_max:.1f}\n")

    print(f"[Save] 已保存选点与范围到: {save_file}")
    print("点云处理完成。")

# ===================== main =====================


if __name__ == "__main__":
    # 1) 解析命令行参数：bag 目录 & 输出目录
    if len(sys.argv) > 1:
        bag_path = sys.argv[1]
    else:
        # 默认使用当前目录下的某个 bag，可以按需修改
        bag_path = os.path.join(os.getcwd(), "all_2025-11-17-18-22-27")
        print(f"未指定 bag 目录，默认使用: {bag_path}")

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = os.getcwd()
        print(f"未指定输出目录，使用当前目录: {output_dir}")

    if not os.path.isdir(bag_path):
        print(f"[ERROR] bag 目录 '{bag_path}' 不存在 (ROS 2 bag 是一个包含 metadata.yaml 的目录)", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f"[ERROR] 输出目录 '{output_dir}' 不存在", file=sys.stderr)
        sys.exit(1)

    # 不需要 rclpy.init()，完全离线工具

    # 3) 自动检测 bag 中点云类型及其 topic 名称
    msg_type, topic_name = detect_lidar_msg_type(bag_path)
    if msg_type is None:
        print("[ERROR] 未检测到支持的雷达消息类型，退出。", file=sys.stderr)
        sys.exit(1)

    # 4) 根据类型做对应的 PCD 转换
    if msg_type == "PointCloud2":
        pcd_path = convert_pointcloud2_bag_to_pcd(
            bag_path=bag_path,
            output_dir=output_dir,
            topic_name=topic_name,
            pcd_name="sensor_PointCloud2_inten_ascii.pcd"
        )
    else:  # "CustomMsg"
        pcd_path = convert_livox_custom_bag_to_pcd(
            bag_path=bag_path,
            output_dir=output_dir,
            topic_name=topic_name,
            pcd_name="livox_CustomMsg_inten_ascii.pcd"
        )

    if pcd_path is None:
        print("[ERROR] PCD 生成失败，退出。", file=sys.stderr)
        sys.exit(1)

    # 5) 对刚生成的这个 PCD 做交互式选点 + 范围保存
    select_and_save_points(
        pcd_folder=output_dir,
        target_pcd_name=os.path.basename(pcd_path)
    )
