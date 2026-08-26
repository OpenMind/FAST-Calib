#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto-tune the FAST-Calib distance filter from the live LiDAR topic.

Captures a few seconds of points, RANSAC-fits the dominant plane in front of
the sensor (frame convention: z forward, x right, y up), keeps the connected
board-sized patch around it, and prints a tight x/y/z box, both as yaml lines
and as ready-to-use `-p` overrides for `ros2 run fast_calib fast_calib`.

Usage:
    python3 auto_distance_filter.py [--topic /lidar_points_front] [--seconds 3]
"""
import argparse, time
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", default="/lidar_points_front")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--zmin", type=float, default=1.0, help="search depth min")
    ap.add_argument("--zmax", type=float, default=6.0, help="search depth max")
    ap.add_argument("--margin", type=float, default=0.06, help="box margin (m)")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs_py.point_cloud2 as pc2

    rclpy.init()
    node = Node("auto_distance_filter")
    buf = []
    node.create_subscription(
        PointCloud2, args.topic,
        lambda m: buf.append(pc2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True)),
        qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < args.seconds and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    if not buf:
        raise SystemExit(f"no points received on {args.topic}")
    p = np.vstack(buf)
    print(f"captured {len(p)} points")

    # keep a frontal cone in the search depth range
    p = p[(p[:, 2] > args.zmin) & (p[:, 2] < args.zmax)
          & (np.abs(p[:, 0]) < 0.9 * p[:, 2]) & (np.abs(p[:, 1]) < 0.9 * p[:, 2])]

    # RANSAC plane, favouring planes facing the sensor at board-like depth
    rng = np.random.default_rng(0)
    best = None
    for _ in range(800):
        a, b, c = p[rng.choice(len(p), 3, replace=False)]
        n = np.cross(b - a, c - a)
        L = np.linalg.norm(n)
        if L < 1e-6:
            continue
        n /= L
        if abs(n[2]) < 0.7:          # board faces the sensor -> normal mostly along z
            continue
        d = -n @ a
        cnt = (np.abs(p @ n + d) < 0.02).sum()
        if best is None or cnt > best[0]:
            best = (cnt, n, d)
    if best is None:
        raise SystemExit("no frontal plane found")
    cnt, n, d = best
    inl = p[np.abs(p @ n + d) < 0.025]

    # keep the connected patch nearest the sensor axis (drop far wall points in
    # the same plane): cluster on a coarse grid, take the component containing
    # the point closest to the line x=y=0
    from scipy import ndimage
    u = np.cross(n, [0, 1, 0]); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    o = inl.mean(0)
    U, V = (inl - o) @ u, (inl - o) @ v
    cs = 0.04
    gi = ((U - U.min()) / cs).astype(int)
    gj = ((V - V.min()) / cs).astype(int)
    occ = np.zeros((gi.max() + 1, gj.max() + 1), bool)
    occ[gi, gj] = True
    lab, _ = ndimage.label(ndimage.binary_closing(occ, np.ones((3, 3))))
    axis_i = np.argmin(inl[:, 0] ** 2 + inl[:, 1] ** 2)
    keep = lab[gi, gj] == lab[gi[axis_i], gj[axis_i]]
    board = inl[keep]

    ext_u = (board - o) @ u
    ext_v = (board - o) @ v
    print(f"plane inliers {len(inl)}, board patch {len(board)} pts, "
          f"extent {ext_u.max()-ext_u.min():.2f} x {ext_v.max()-ext_v.min():.2f} m, "
          f"depth ~{board[:, 2].mean():.2f} m, normal {n.round(2)}")

    m = args.margin
    lo = board.min(0) - m
    hi = board.max(0) + m
    print("\nyaml block:")
    for k, val in zip(("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
                      (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])):
        print(f"      {k}: {val:.2f}")
    print("\nros2 run overrides:")
    print(" ".join(f"-p {k}:={val:.2f}" for k, val in
                   zip(("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"),
                       (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]))))


if __name__ == "__main__":
    main()
