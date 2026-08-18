#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Headless replacement for the Open3D point-picking step of distance_filter_tool.py.

Finds the calibration board automatically in a LiDAR point cloud (live topic or
recorded bag), verifies it by locating the four circular holes, and prints the
distance-filter box for config/qr_params.yaml. No display needed — safe to run
over SSH (e.g. on a Jetson/Thor).

Usage:
    # live topic (default /lidar_points_front, 3 s capture)
    python3 auto_filter_box.py
    python3 auto_filter_box.py --topic /lidar_points_front --seconds 5

    # from a recorded bag instead
    python3 auto_filter_box.py --bag /path/to/bag_dir

    # additionally patch config/qr_params.yaml in place (then colcon build!)
    python3 auto_filter_box.py --apply

Only numpy + the ROS 2 Python libraries are required.
"""

import argparse
import os
import re
import sys
from collections import deque

import numpy as np

AXES = {"x": 0, "y": 1, "z": 2}


# ---------------------------------------------------------------- input

def points_from_live_topic(topic, seconds):
    import time
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs_py.point_cloud2 as pc2

    rclpy.init()
    node = Node("auto_filter_box")
    chunks = []

    def cb(msg):
        pts = pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)
        chunks.append(np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32))

    node.create_subscription(PointCloud2, topic, cb, 10)
    print(f"[capture] subscribing to {topic} for {seconds:.1f} s ...")
    t0 = time.time()
    while rclpy.ok() and time.time() - t0 < seconds:
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()
    if not chunks:
        sys.exit(f"[ERROR] no messages received on {topic}")
    return np.vstack(chunks)


def points_from_bag(bag_path, topic):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs_py.point_cloud2 as pc2

    storage = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    conv = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, conv)
    topics = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topic is None:
        pc2_topics = [n for n, t in topics.items() if t == "sensor_msgs/msg/PointCloud2"]
        if not pc2_topics:
            sys.exit("[ERROR] no PointCloud2 topic in this bag")
        topic = pc2_topics[0]
        print(f"[bag] using topic {topic}")
    chunks = []
    while reader.has_next():
        name, data, _ = reader.read_next()
        if name != topic:
            continue
        msg = deserialize_message(data, PointCloud2)
        pts = pc2.read_points(msg, field_names=["x", "y", "z"], skip_nans=True)
        chunks.append(np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32))
    if not chunks:
        sys.exit(f"[ERROR] no PointCloud2 messages for {topic} in bag")
    return np.vstack(chunks)


# ---------------------------------------------------------------- grid helpers

def dilate(mask, iterations):
    m = mask.copy()
    for _ in range(iterations):
        n = m.copy()
        n[1:, :] |= m[:-1, :]
        n[:-1, :] |= m[1:, :]
        n[:, 1:] |= m[:, :-1]
        n[:, :-1] |= m[:, 1:]
        m = n
    return m


def binary_close(mask, iterations):
    return ~dilate(~dilate(mask, iterations), iterations)


def label_components(mask):
    """4-connectivity labeling. Returns (labels array, count)."""
    labels = np.zeros(mask.shape, dtype=np.int32)
    current = 0
    H, W = mask.shape
    for sy, sx in zip(*np.where(mask)):
        if labels[sy, sx]:
            continue
        current += 1
        q = deque([(sy, sx)])
        labels[sy, sx] = current
        while q:
            y, x = q.popleft()
            for ny, nx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
                if 0 <= ny < H and 0 <= nx < W and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = current
                    q.append((ny, nx))
    return labels, current


# ---------------------------------------------------------------- board search

def fit_plane(pts):
    """Least-squares plane through pts. Returns (unit normal, d) with n.p + d = 0."""
    c = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - c, full_matrices=False)
    n = vt[2]
    return n, -n.dot(c)


def best_rectangle(holes, width=0.5, height=0.4, tol=0.08):
    """Pick the 4 hole candidates whose mutual distances best match the target
    rectangle (width x height). Returns list of 4 or None."""
    from itertools import combinations
    expected = sorted([width, width, height, height,
                       np.hypot(width, height), np.hypot(width, height)])
    best, best_err = None, None
    for combo in combinations(range(len(holes)), 4):
        pts = [holes[i] for i in combo]
        dists = sorted(np.linalg.norm(pts[i] - pts[j])
                       for i, j in combinations(range(4), 2))
        err = max(abs(a - b) for a, b in zip(dists, expected))
        if err < tol and (best_err is None or err < best_err):
            best, best_err = pts, err
    return best


def find_board(points, fwd, circle_radius, rng):
    fa = AXES[fwd]
    lat = [a for a in range(3) if a != fa]

    crop = points[(points[:, fa] > rng[0]) & (points[:, fa] < rng[1])
                  & (np.abs(points[:, lat[0]]) < 4) & (np.abs(points[:, lat[1]]) < 4)]
    if len(crop) < 2000:
        sys.exit("[ERROR] too few points in the search range")

    # 1 cm voxel downsample
    keys = np.floor(crop / 0.01).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    cloud = crop[idx].astype(np.float64)
    print(f"[search] {len(crop)} pts in range, {len(cloud)} after downsample")

    rs = np.random.RandomState(0)
    remaining = cloud
    for attempt in range(10):
        if len(remaining) < 1000:
            break
        # RANSAC plane
        best_inl = None
        for _ in range(300):
            tri = remaining[rs.choice(len(remaining), 3, replace=False)]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            norm = np.linalg.norm(n)
            if norm < 1e-9:
                continue
            n /= norm
            d = -n.dot(tri[0])
            inl = np.abs(remaining @ n + d) < 0.015
            if best_inl is None or inl.sum() > best_inl.sum():
                best_inl = inl
        plane_pts = remaining[best_inl]
        n, d = fit_plane(plane_pts)
        inl = np.abs(remaining @ n + d) < 0.015
        n, d = fit_plane(remaining[inl])
        # fit tight, rasterize loose: this class of sensor has 2-3 cm noise at
        # a few meters, and a thin membership band leaves the occupancy grid
        # too sparse to enclose the holes
        member = np.abs(remaining @ n + d) < 0.04
        board = check_plane(remaining[member], n, circle_radius)
        if board is not None:
            return board
        remaining = remaining[~inl]

    return None


def check_plane(plane_pts, n, circle_radius):
    """Return (board points, hole centers 3D) if this plane contains the target."""
    centroid = plane_pts.mean(axis=0)
    view = centroid / np.linalg.norm(centroid)
    if abs(n.dot(view)) < 0.5:      # board roughly faces the sensor
        return None

    # 2D basis in the plane
    u = np.cross(n, view)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [0, 0, 1.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    rel = plane_pts - centroid
    uv = np.c_[rel @ u, rel @ v]

    res = 0.01
    pad = 4
    lo = uv.min(axis=0)
    span = uv.max(axis=0) - lo
    W, H = (span / res).astype(int) + 1 + 2 * pad
    if W < 20 or H < 20 or W > 3000 or H > 3000:
        return None
    ij = ((uv - lo) / res).astype(int) + pad
    occ = np.zeros((H, W), bool)
    occ[ij[:, 1], ij[:, 0]] = True

    closed = binary_close(occ, 4)
    lbl, cnt = label_components(closed)
    for i in range(1, cnt + 1):
        comp = lbl == i
        cells = comp.sum()
        if cells * res * res < 0.4:          # component area >= 0.4 m^2
            continue
        ys, xs = np.where(comp)
        ext_u = (xs.max() - xs.min()) * res
        ext_v = (ys.max() - ys.min()) * res
        if not (0.6 < ext_u < 2.5 and 0.6 < ext_v < 2.5):
            continue

        # hole candidates: empty cells enclosed by the closed occupancy
        empty = ~closed
        elbl, ecnt = label_components(empty)
        border = set(elbl[0, :]) | set(elbl[-1, :]) | set(elbl[:, 0]) | set(elbl[:, -1])
        holes_uv = []
        r_px = circle_radius / res
        for j in range(1, ecnt + 1):
            if j in border:
                continue
            hys, hxs = np.where(elbl == j)
            area = len(hys)
            if not (np.pi * (r_px - 5) ** 2 < area < np.pi * (r_px + 5) ** 2):
                continue
            w, h = hxs.max() - hxs.min(), hys.max() - hys.min()
            if abs(w - h) > 0.6 * r_px:
                continue
            holes_uv.append(np.array([(hxs.mean() - pad) * res + lo[0],
                                      (hys.mean() - pad) * res + lo[1]]))

        # keep only the 4 holes matching the target's rectangle geometry
        rect = best_rectangle(holes_uv) if len(holes_uv) >= 4 else None
        if rect is not None:
            holes_uv = rect
        holes3d = [centroid + cu * u + cv * v for cu, cv in holes_uv]

        if rect is not None:
            # board region from hole geometry: the hole rectangle gives the
            # board's in-plane orientation, so apply the CAD margins beyond the
            # hole centers along each board axis (1.2x0.9 board, 0.5x0.4 holes)
            huv = np.array(holes_uv)
            hc = huv.mean(axis=0)
            # width axis = direction of a long (~0.5 m) hole-pair side
            dists = [(np.linalg.norm(huv[a] - huv[b]), a, b)
                     for a in range(4) for b in range(a + 1, 4)]
            _, a, b = max((d, a, b) for d, a, b in dists if d < 0.55)
            e1 = (huv[a] - huv[b]) / np.linalg.norm(huv[a] - huv[b])
            e2 = np.array([-e1[1], e1[0]])
            d1 = np.abs((uv - hc) @ e1)
            d2 = np.abs((uv - hc) @ e2)
            keep = (d1 < 0.25 + 0.36) & (d2 < 0.20 + 0.26)
            return plane_pts[keep], holes3d

        member = comp[ij[:, 1], ij[:, 0]]
        return plane_pts[member], holes3d
    return None


# ---------------------------------------------------------------- output

YAML_KEYS = ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max")


def apply_to_yaml(config_path, box):
    with open(config_path) as f:
        text = f.read()
    for key, val in zip(YAML_KEYS, box):
        # replace only the uncommented occurrence of each key
        pattern = rf"(?m)^(\s*)({key}):\s*-?[\d.]+"
        new_text, n = re.subn(pattern, rf"\g<1>\g<2>: {val:.2f}", text)
        if n != 1:
            print(f"[WARN] expected exactly 1 active '{key}:' line in {config_path}, "
                  f"found {n} — not applied for this key")
            continue
        text = new_text
    with open(config_path, "w") as f:
        f.write(text)
    print(f"[apply] updated {config_path}")
    print("[apply] REMEMBER: colcon build --packages-select fast_calib")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--bag", help="ROS 2 bag directory (default: live topic)")
    src.add_argument("--topic", default="/lidar_points_front", help="live PointCloud2 topic")
    ap.add_argument("--seconds", type=float, default=3.0, help="live capture duration")
    ap.add_argument("--forward-axis", choices=["x", "z"], default="z",
                    help="depth axis of the lidar frame (this sensor: z)")
    ap.add_argument("--circle-radius", type=float, default=0.12)
    ap.add_argument("--min-range", type=float, default=0.8)
    ap.add_argument("--max-range", type=float, default=6.5)
    ap.add_argument("--apply", action="store_true",
                    help="patch config/qr_params.yaml next to this script")
    args = ap.parse_args()

    if args.bag:
        points = points_from_bag(args.bag, None)
    else:
        points = points_from_live_topic(args.topic, args.seconds)
    print(f"[capture] {len(points)} points total")

    found = find_board(points, args.forward_axis, args.circle_radius,
                       (args.min_range, args.max_range))
    if found is None:
        sys.exit("[ERROR] no board-like plane with circular holes found — is the "
                 "target in view and between --min-range and --max-range?")
    board_pts, holes = found

    print(f"\n[board] {len(board_pts)} points, {len(holes)} circular hole(s) found")
    for h in holes:
        print(f"[board]   hole center at ({h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f})")
    if len(holes) != 4:
        print("[WARN] expected 4 holes — box is still printed, but check the setup")

    lo, hi = board_pts.min(axis=0), board_pts.max(axis=0)
    # when the 4-hole rectangle was verified, board_pts already spans exactly the
    # board face, so only a small pad is needed; pad more when unverified
    pad_val = 0.1 if len(holes) == 4 else 0.2
    pad = np.full(3, pad_val)
    pad[AXES[args.forward_axis]] = pad_val + 0.1
    box = [lo[0]-pad[0], hi[0]+pad[0], lo[1]-pad[1], hi[1]+pad[1], lo[2]-pad[2], hi[2]+pad[2]]

    print("\n# paste into config/qr_params.yaml (then: colcon build --packages-select fast_calib)")
    for key, val in zip(YAML_KEYS, box):
        print(f"      {key}: {val:.2f}")

    if args.apply:
        config = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "..", "config", "qr_params.yaml")
        apply_to_yaml(os.path.normpath(config), box)


if __name__ == "__main__":
    main()
