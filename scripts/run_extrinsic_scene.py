#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-scene LiDAR-camera extrinsic calibration driver for the openmind robot.

Wraps the fast_calib nodes so one command handles a whole scene:
capture the LiDAR cloud, find the calibration board and its four holes,
derive a tight distance-filter box automatically, run the single-scene
calibration, and file the outputs into one dated session folder:

    output/extrinsic_<YYYYMMDD>/
        scene1_qr_detect.png   camera detection overlay per scene
        scene1_raw.png         raw camera frame per scene
        circle_center_record.txt   accumulated hole-centre pairs
        extrinsic_result.txt   final multi-scene extrinsic (after 'solve')

Defaults match the robot: image /camera/front/image_raw (M20 fisheye,
undistorted internally via qr_params.yaml), LiDAR /lidar (base_link,
x-forward), grid hole detector.

Usage:
    python3 run_extrinsic_scene.py check      # board visible? holes found?
    python3 run_extrinsic_scene.py 1          # calibrate scene 1
    python3 run_extrinsic_scene.py 2          # ... after moving the board
    python3 run_extrinsic_scene.py 3
    python3 run_extrinsic_scene.py solve      # joint solve -> extrinsic_result.txt
"""

import argparse
import itertools
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import cv2

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def capture_cloud(topic, seconds):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import PointCloud2
    import sensor_msgs_py.point_cloud2 as pc2

    rclpy.init(args=['--ros-args', '--log-level', 'rmw_cyclonedds_cpp:=error'])
    node = Node("extrinsic_box_finder")
    buf = []
    node.create_subscription(
        PointCloud2, topic,
        lambda m: buf.append(pc2.read_points_numpy(m, field_names=("x", "y", "z"), skip_nans=True)),
        qos_profile_sensor_data)
    t0 = time.time()
    while time.time() - t0 < seconds and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    rclpy.shutdown()
    if not buf:
        sys.exit(f"no messages on {topic} -- is the robot's sensor stack running?")
    return np.vstack(buf)


def find_board_box(pts, depth_axis=0, d_min=1.2, d_max=6.0):
    """Find the board plane + its 4 holes; return (box dict, info string).

    depth_axis 0 = x-forward frame (/lidar in base_link).
    """
    d = pts[:, depth_axis]
    lat = [i for i in range(3) if i != depth_axis]
    s = pts[(d > d_min) & (d < d_max)
            & (np.abs(pts[:, lat[0]]) < 0.9 * d) & (np.abs(pts[:, lat[1]]) < 0.9 * d)]
    if len(s) < 5000:
        return None, "too few points in the search cone"

    rng = np.random.default_rng(0)
    best = None
    for _ in range(2000):
        a, b, c = s[rng.choice(len(s), 3, replace=False)]
        n = np.cross(b - a, c - a)
        L = np.linalg.norm(n)
        if L < 1e-6:
            continue
        n /= L
        if abs(n[depth_axis]) < 0.6:     # board roughly faces the sensor
            continue
        dd = -n @ a
        cnt = (np.abs(s @ n + dd) < 0.02).sum()
        if best is None or cnt > best[0]:
            best = (cnt, n, dd)
    if best is None:
        return None, "no frontal plane found"
    cnt, n, dd = best
    inl = s[np.abs(s @ n + dd) < 0.03]

    # in-plane 2D occupancy -> enclosed empty regions = holes
    u = np.cross(n, [0.0, 0.0, 1.0])
    if np.linalg.norm(u) < 1e-6:
        u = np.cross(n, [0.0, 1.0, 0.0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    o = inl.mean(0)
    U, V = (inl - o) @ u, (inl - o) @ v
    res = 0.01
    W = int((U.max() - U.min()) / res) + 9
    H = int((V.max() - V.min()) / res) + 9
    if W > 4000 or H > 4000:
        return None, "degenerate plane extent"
    occ = np.zeros((H, W), np.uint8)
    occ[((V - V.min()) / res).astype(int) + 4, ((U - U.min()) / res).astype(int) + 4] = 255
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
    empty = 255 - closed
    outside = empty.copy()
    cv2.floodFill(outside, None, (0, 0), 128)
    holes = ((empty == 255) & (outside != 128)).astype(np.uint8)
    ncomp, _, stats, cent = cv2.connectedComponentsWithStats(holes, 8, cv2.CV_32S)
    cands = []
    for i in range(1, ncomp):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < 80 or a > 1500:
            continue
        r = np.sqrt(a / np.pi) * res
        if 0.09 < r < 0.15:
            cu = (cent[i][0] - 4) * res + U.min()
            cv_ = (cent[i][1] - 4) * res + V.min()
            cands.append((cu, cv_))

    depth = inl[:, depth_axis].mean()
    info = (f"plane depth ~{depth:.2f} m, normal {n.round(2)}, "
            f"{len(cands)} plausible holes")
    if len(cands) < 4:
        return None, info

    # best 4-subset matching the 0.5 x 0.4 rectangle
    def rect_err(q):
        ds = sorted(np.hypot(a[0] - b[0], a[1] - b[1]) for a, b in itertools.combinations(q, 2))
        return sum(abs(m - t) for m, t in zip(ds, [0.4, 0.4, 0.5, 0.5, 0.6403, 0.6403]))
    e, q = min(((rect_err(q), q) for q in itertools.combinations(cands, 4)), key=lambda t: t[0])
    if e > 0.15:
        return None, info + f"; no valid rectangle (err {e:.3f})"

    pts3 = np.array([o + cu * u + cv_ * v for cu, cv_ in q])
    lo, hi = pts3.min(0), pts3.max(0)
    box = {}
    for ax, name in enumerate("xyz"):
        if ax == depth_axis:
            m = inl[np.all((inl[:, lat] > lo[lat] - 0.3) & (inl[:, lat] < hi[lat] + 0.3), axis=1)]
            box[f"{name}_min"] = float(m[:, ax].min() - 0.06)
            box[f"{name}_max"] = float(m[:, ax].max() + 0.06)
        else:
            box[f"{name}_min"] = float(lo[ax] - 0.24)
            box[f"{name}_max"] = float(hi[ax] + 0.24)
    return box, info + f"; rectangle OK (err {e:.3f})"


def run_node(exe, extra_params, session, timeout_s=90):
    cmd = ["ros2", "run", "fast_calib", exe, "--ros-args",
           "--log-level", "rmw_cyclonedds_cpp:=error",
           "--params-file", os.path.join(PKG, "config", "qr_params.yaml"),
           "-p", f"output_path:={session}/"]
    for k, val in extra_params.items():
        cmd += ["-p", f"{k}:={val}"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines, t0 = [], time.time()
    try:
        for line in proc.stdout:
            line = ANSI.sub("", line.rstrip())
            lines.append(line)
            if any(k in line for k in ("hole(s) detected", "RMSE", "[Record]", "[Result]", "WARN", "Error")):
                print("  " + line)
            # qr_detect.png is the LAST artifact the node writes; stop only
            # after it is on disk (or on timeout / after the multi-scene result).
            if ("qr_detect.png" in line or "Multi-scene calibration results saved" in line
                    or time.time() - t0 > timeout_s):
                break
    finally:
        proc.terminate()
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scene", choices=["check", "1", "2", "3", "solve"])
    ap.add_argument("--image-topic", default="/camera/front/image_raw")
    ap.add_argument("--lidar-topic", default="/lidar")
    ap.add_argument("--lidar-frame", default="xfwd",
                    help="'xfwd' for /lidar (base_link), 'zfwd' for /lidar_points_front")
    ap.add_argument("--session", default=None,
                    help="session folder (default output/extrinsic_<today>)")
    ap.add_argument("--capture-seconds", type=float, default=6.0)
    args = ap.parse_args()

    session = args.session or os.path.join(PKG, "output", "extrinsic_" + datetime.now().strftime("%Y%m%d"))
    os.makedirs(session, exist_ok=True)
    depth_axis = 0 if args.lidar_frame == "xfwd" else 2

    if args.scene == "solve":
        rec = os.path.join(session, "circle_center_record.txt")
        if not os.path.exists(rec):
            sys.exit(f"{rec} not found -- run scenes 1-3 first")
        blocks = open(rec).read().count("-\n")
        print(f"solving from {rec}")
        run_node("multi_fast_calib", {}, session, timeout_s=30)
        src = os.path.join(session, "multi_calib_result.txt")
        dst = os.path.join(session, "extrinsic_result.txt")
        if os.path.exists(src):
            shutil.copy(src, dst)
            print(f"\n=== {dst} ===")
            print(open(dst).read())
        else:
            sys.exit("multi_calib_result.txt was not produced -- check the log above")
        return

    print(f"session folder: {session}")
    print(f"capturing {args.lidar_topic} to locate the board ...")
    pts = capture_cloud(args.lidar_topic, 4.0)
    box, info = find_board_box(pts, depth_axis=depth_axis)
    print(info)
    if box is None:
        sys.exit("Board not usable: adjust placement (all 4 holes visible to the LiDAR, "
                 "~1 m clearance behind, 1.8-3.5 m distance) and retry.")
    print("filter box: " + "  ".join(f"{k}={v:.2f}" for k, v in box.items()))
    if args.scene == "check":
        print("check OK -- the board is calibratable from here")
        return

    n = args.scene
    print(f"\nrunning single-scene calibration for scene {n} ...")
    log = run_node("fast_calib", {
        "bag_path": "''",
        "image_path": args.image_topic,
        "lidar_topic": args.lidar_topic,
        "lidar_type": "grid",
        "lidar_frame": args.lidar_frame,
        "live_capture_seconds": args.capture_seconds,
        "plane_dist_threshold": 0.03,
        **{k: round(v, 2) for k, v in box.items()},
    }, session)

    ok = "4 circular hole(s) detected" in log and "[Record] Saved" in log
    for name_src, name_dst in (("qr_detect.png", f"scene{n}_qr_detect.png"),
                               ("topic_frame.png", f"scene{n}_raw.png")):
        src = os.path.join(session, name_src)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(session, name_dst))
    if not ok:
        sys.exit(f"\nscene {n} FAILED (see log above) -- fix the placement and rerun this command; "
                 "nothing was recorded, so the record file is still clean")
    m = re.search(r"RMSE: ([0-9.]+)", log)
    print(f"\nscene {n} OK (RMSE {m.group(1)} m). Images saved as scene{n}_*.png in {session}")
    if n != "3":
        print(f"Move the board (new angle AND new distance), then run: python3 {sys.argv[0]} {int(n)+1}")
    else:
        print(f"All scenes done. Run: python3 {sys.argv[0]} solve")


if __name__ == "__main__":
    main()
