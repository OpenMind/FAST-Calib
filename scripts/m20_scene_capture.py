#!/usr/bin/env python3
"""Capture one FAST-Calib scene on the M20: the camera picture and 5 s of the front LiDAR.

Before it records it checks that the robot and the gimbal are still and at the patrol
angle; after, that the board's markers are in the picture and its plane is in the LiDAR,
and it fits the distance filter around the board by itself. Each scene gets its own folder
with a params.yaml for FAST-Calib.

    source ~/calib_ws/install/setup.bash
    python3 scripts/m20_scene_capture.py scene1           # board facing the camera
    python3 scripts/m20_scene_capture.py scene2           # board turned ~25 deg one way
    python3 scripts/m20_scene_capture.py scene3           # ... and the other way
    python3 scripts/m20_calib_run.py                      # calibrate all scenes

Where to put the board: front-right of the robot, 2-3 m out, its centre about 0.5-1.5 m
ahead of the front LiDAR. The front LiDAR sees nothing behind the plane through it
across the robot (x_base < 0.32 m), and the camera at yaw 90 sees 40 deg either side of
straight right, so the board sits in the left half of the picture.
"""
import argparse
import itertools
import json
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2
from std_msgs.msg import Float32

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20_common import (BOARD_IDS, LIDAR_TOPIC, QR_PARAMS, REPO, RTSP_URL, Stream,  # noqa: E402
                        detect_markers, get_yaml_value, lookup_base_lidar, pointcloud2_xyz_ring,
                        set_yaml_values)

def ransac_plane(P, thr=0.02, iters=400, vertical=True, rng=np.random.default_rng(0)):
    """Best plane through P; with vertical=True only planes whose normal is near horizontal."""
    best, best_n = None, 0
    for _ in range(iters):
        a, b, c = P[rng.choice(len(P), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-6:
            continue
        n /= np.linalg.norm(n)
        if vertical and abs(n[2]) > 0.3:
            continue
        inl = np.abs((P - a) @ n) < thr
        if inl.sum() > best_n:
            best, best_n = inl, inl.sum()
    return best


def largest_cluster(P, cell=0.06):
    """Largest 3D-grid-connected cluster of P (indices)."""
    keys = np.floor(P / cell).astype(int)
    cells = {}
    for i, k in enumerate(map(tuple, keys)):
        cells.setdefault(k, []).append(i)
    seen, best = set(), []
    for k in cells:
        if k in seen:
            continue
        stack, comp = [k], []
        seen.add(k)
        while stack:
            c = stack.pop()
            comp += cells[c]
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        nb = (c[0] + dx, c[1] + dy, c[2] + dz)
                        if nb in cells and nb not in seen:
                            seen.add(nb)
                            stack.append(nb)
        if len(comp) > len(best):
            best = comp
    return np.array(best)


def find_holes(B, n, w_c, h_c, r_c, res=0.01):
    """find_holes_k with the smallest gap-closing size that gives the 4 holes (scan lines are
    further apart on a far or turned board)."""
    for k in (7, 9, 11, 13):
        cands, err, inferred = find_holes_k(B, n, w_c, h_c, r_c, res, k)
        if err is not None and err < 0.15:
            break
    return cands, err, inferred


def find_holes_k(B, n, w_c, h_c, r_c, res, k):
    """The board's holes from its plane points B (normal n): an in-plane occupancy image, closed
    over the gaps between scan lines; enclosed empty round regions are holes. Returns (centres
    in B's frame, error of the best 4 against the w_c x h_c rectangle, or None)."""
    u = np.cross(n, [0.0, 0.0, 1.0])
    u = u / np.linalg.norm(u) if np.linalg.norm(u) > 1e-6 else np.array([1.0, 0, 0])
    v = np.cross(n, u)
    o = B.mean(0)
    U, V = (B - o) @ u, (B - o) @ v
    W, H = int(np.ptp(U) / res) + 9, int(np.ptp(V) / res) + 9
    if W > 1000 or H > 1000:
        return [], None
    occ = np.zeros((H, W), np.uint8)
    occ[((V - V.min()) / res).astype(int) + 4, ((U - U.min()) / res).astype(int) + 4] = 255
    closed = cv2.morphologyEx(occ, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    # empty space inside the board's outline: a hole whose thin outer border has a gap in it
    # (board edge near the LiDAR's blind plane, sparse scan lines) still counts
    pts = np.column_stack(np.nonzero(closed)[::-1]).astype(np.int32)
    hull = np.zeros_like(closed)
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 255)
    holes = ((hull == 255) & (closed == 0)).astype(np.uint8)
    nc, _, stats, cent = cv2.connectedComponentsWithStats(holes, 8, cv2.CV_32S)
    cands = []
    for i in range(1, nc):
        r = np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi) * res
        bw, bh = stats[i, cv2.CC_STAT_WIDTH] * res, stats[i, cv2.CC_STAT_HEIGHT] * res
        # scan-line gaps and lost edge returns enlarge the holes (r ~0.18 for 0.12 on the M20): same
        # limits as FAST-Calib's grid detector (lidar_detect.hpp detect_grid_lidar)
        if 0.75 * r_c < r < 1.6 * r_c and max(bw, bh) < 3.4 * r_c:
            cu, cv_ = (cent[i][0] - 4) * res + U.min(), (cent[i][1] - 4) * res + V.min()
            cands.append(o + cu * u + cv_ * v)
    inferred = False
    if len(cands) == 3:  # one hole lost at a cut board edge: complete the L as the C++ does
        for i in range(3):
            P, A, B = cands[i], cands[(i + 1) % 3], cands[(i + 2) % 3]
            a, b = A - P, B - P
            la, lb = np.linalg.norm(a), np.linalg.norm(b)
            lw, lh = max(la, lb), min(la, lb)
            if abs(a @ b) / (la * lb) < 0.14 and abs(lw - w_c) / w_c < 0.08 and abs(lh - h_c) / h_c < 0.08:
                cands = cands + [A + B - P]
                inferred = True
                break
    if len(cands) < 4:
        return cands, None, inferred
    want = sorted([h_c, h_c, w_c, w_c, np.hypot(w_c, h_c), np.hypot(w_c, h_c)])
    err = min(sum(abs(a - b) for a, b in zip(sorted(np.linalg.norm(p - q) for p, q in itertools.combinations(c4, 2)), want))
              for c4 in itertools.combinations(cands, 4))
    return cands, err, inferred


def board_filter(Pl, ring, T_bl, box, check, board_dims=(0.5, 0.4, 0.12), say=print, info=None):
    """Find the board in one cloud (LiDAR frame) and return FAST-Calib's distance filter
    (the board's box + 0.15 m, in the LiDAR frame) and lidar_up, plus the ground height.
    `info`, a dict, receives the board's points (base_link), plane normal, holes and sizes."""
    info = {} if info is None else info
    Pb = Pl @ T_bl[:3, :3].T + T_bl[:3, 3]
    low = Pb[(np.hypot(Pb[:, 0], Pb[:, 1]) < 4) & (Pb[:, 2] < 0)][:, 2]
    hist, edges = np.histogram(low, bins=np.arange(-1.5, 0.0, 0.02))
    ground = edges[hist.argmax()] + 0.01 if hist.max() > 50 else -0.45
    x0, x1, y0, y1, z0, z1 = box
    sel = ((Pb[:, 0] > x0) & (Pb[:, 0] < x1) & (Pb[:, 1] > y0) & (Pb[:, 1] < y1)
           & (Pb[:, 2] > ground + z0) & (Pb[:, 2] < ground + z1))
    say(f"  ground at z_base {ground:.2f} m; {sel.sum()} points in the search box")
    up_l = T_bl[:3, :3].T @ np.array([0, 0, 1.0])
    vals = dict(lidar_up_x=float(up_l[0]), lidar_up_y=float(up_l[1]), lidar_up_z=float(up_l[2]))
    board = None
    if sel.sum() > 100:
        idx = np.where(sel)[0]
        inl = ransac_plane(Pb[idx])
        if inl is not None:
            board = idx[inl][largest_cluster(Pb[idx[inl]])]
    if board is None or len(board) < 100:
        check(False, "no board plane in the LiDAR search box (move it, or set --box)")
        return vals, ground
    B = Pb[board]
    lo, hi = B.min(0), B.max(0)
    n = np.linalg.svd(B - B.mean(0))[2][2]
    cands, herr, inferred = find_holes(B, n, *board_dims)
    seen = len(cands) - inferred
    check(herr is not None and herr < 0.15,
          f"LiDAR holes: {seen} seen" + (" + 1 inferred from the other 3 (move the board forward or lower to see all 4)" if inferred else "")
          + (f", rectangle error {herr:.3f} m" if herr is not None else ""))
    ext = hi - lo
    width = np.hypot(ext[0], ext[1])
    rings = len(np.unique(ring[board]))
    say(f"  board plane: {len(board)} points, {rings} rings, x {lo[0]:.2f}..{hi[0]:.2f} y {lo[1]:.2f}..{hi[1]:.2f} "
          f"z {lo[2] - ground:.2f}..{hi[2] - ground:.2f} above ground, {width:.2f} m wide, {ext[2]:.2f} m tall")
    check(1.0 < width < 1.8 and 0.7 < ext[2] < 1.4, "board-sized plane (largest flat thing in the box: if this fails it found a wall)")
    if herr is None or herr >= 0.15:
        # only as the likely reason for missing holes: with all holes found, something in front of the
        # LiDAR (a rod, the board's stand) reaching back past the line does not matter
        check(lo[0] > T_bl[0, 3] + 0.10, f"whole board ahead of the LiDAR's blind plane (x_base > {T_bl[0, 3] + 0.10:.2f})")
    dist = np.linalg.norm((lo + hi) / 2 - T_bl[:3, 3])
    info.update(B=B, n=n, holes=cands, hole_err=herr, holes_inferred=inferred, rings=rings, dist=dist, width=width, height=ext[2])
    check(dist < 3.5, f"board {dist:.1f} m from the LiDAR (closer = more rings across each hole; < 3.5)")
    check(rings >= 15, f"{rings} rings on the board (>= 15 gives 3+ across each 240 mm hole)")
    # the mount is a pure axis swap, so a box in base_link is a box in the LiDAR frame
    m = 0.15
    cb = np.array([[x, y, z] for x in (lo[0] - m, hi[0] + m) for y in (lo[1] - m, hi[1] + m) for z in (lo[2] - m, hi[2] + m)])
    cl = (cb - T_bl[:3, 3]) @ T_bl[:3, :3]
    bmin, bmax = cl.min(0), cl.max(0)
    vals.update(x_min=float(bmin[0]), x_max=float(bmax[0]), y_min=float(bmin[1]), y_max=float(bmax[1]),
                z_min=float(bmin[2]), z_max=float(bmax[2]))
    say(f"  distance filter (LiDAR frame): x {bmin[0]:.2f}..{bmax[0]:.2f}  y {bmin[1]:.2f}..{bmax[1]:.2f}  z {bmin[2]:.2f}..{bmax[2]:.2f}")
    return vals, ground


class Probe(Node):
    def __init__(self, topic):
        super().__init__("m20_scene_capture")
        self.imu, self.att, self.zoom, self.raw = [], [], [], []
        self.recording = False
        self.create_subscription(Imu, "/IMU", lambda m: self.imu.append(
            (time.monotonic(), m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z)), qos_profile_sensor_data)
        self.create_subscription(Vector3, "/zr10/gimbal/attitude",
                                 lambda m: self.att.append((time.monotonic(), m.x, m.y, m.z)), 10)
        self.create_subscription(Float32, "/zr10/gimbal/zoom_level", lambda m: self.zoom.append(m.data), 10)
        self.create_subscription(PointCloud2, topic, self._cloud, qos_profile_sensor_data, raw=True)

    def _cloud(self, data):
        if self.recording:
            self.raw.append((time.time_ns(), data))

    def spin_for(self, s):
        t0 = time.monotonic()
        while time.monotonic() - t0 < s:
            rclpy.spin_once(self, timeout_sec=0.02)


def still_report(p, window):
    now = time.monotonic()
    g = np.array([r[1:] for r in p.imu if r[0] > now - window])
    a = np.array([r[1:] for r in p.att if r[0] > now - window])
    gyro = np.linalg.norm(g, axis=1).max() if len(g) else np.nan
    return gyro, a


def write_bag(path, topic, msgs):
    import rosbag2_py
    w = rosbag2_py.SequentialWriter()
    w.open(rosbag2_py.StorageOptions(uri=path, storage_id="sqlite3"),  # FAST-Calib reads sqlite3 only
           rosbag2_py.ConverterOptions("cdr", "cdr"))
    try:
        meta = rosbag2_py.TopicMetadata(id=0, name=topic, type="sensor_msgs/msg/PointCloud2", serialization_format="cdr")
    except TypeError:
        meta = rosbag2_py.TopicMetadata(name=topic, type="sensor_msgs/msg/PointCloud2", serialization_format="cdr")
    w.create_topic(meta)
    for t, data in msgs:
        w.write(topic, data, t)
    del w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="scene name, e.g. scene1")
    ap.add_argument("--session", default=os.path.join(REPO, "calib_data", "m20_" + time.strftime("%Y%m%d")))
    ap.add_argument("--duration", type=float, default=5.0, help="LiDAR seconds to record")
    ap.add_argument("--topic", default=LIDAR_TOPIC)
    ap.add_argument("--url", default=RTSP_URL)
    ap.add_argument("--params", default=QR_PARAMS, help="base FAST-Calib params (intrinsics, board)")
    ap.add_argument("--yaw", type=float, default=90.0, help="patrol gimbal yaw")
    ap.add_argument("--pitch", type=float, default=0.0, help="patrol gimbal pitch")
    ap.add_argument("--box", type=float, nargs=6, default=[0.40, 3.5, -4.0, -0.8, 0.15, 2.5], metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"),
                    help="where to look for the board, base_link metres (z above the ground)")
    ap.add_argument("--force", action="store_true", help="record even if a check fails")
    a = ap.parse_args()
    out = os.path.join(a.session, a.name)
    if os.path.exists(os.path.join(out, "bag")):
        sys.exit(f"{out} already has a recording: delete it or pick another name")
    os.makedirs(out, exist_ok=True)
    problems = []

    def check(ok, msg):
        print(("  ok   " if ok else "  FAIL ") + msg)
        if not ok:
            problems.append(msg)

    rclpy.init()
    p = Probe(a.topic)
    T_bl, src = lookup_base_lidar(p)
    print(f"base_link <- {a.topic} from {src}")
    stream = Stream(a.url)

    print("\n1. robot and gimbal still, at the patrol angle")
    p.spin_for(3.0)
    gyro, att = still_report(p, 2.5)
    check(gyro < 0.05, f"body rotation {np.degrees(gyro):.2f} deg/s (max over 2.5 s, < 2.9)")
    if len(att):
        yaw, pitch = att[:, 0], att[:, 1]
        check(np.ptp(yaw) < 0.5 and np.ptp(pitch) < 0.5, f"gimbal steady: yaw range {np.ptp(yaw):.1f}, pitch range {np.ptp(pitch):.1f} deg")
        check(abs(np.median(yaw) - a.yaw) < 2.5 and abs(np.median(pitch) - a.pitch) < 2.5,
              f"gimbal at yaw {np.median(yaw):.1f} pitch {np.median(pitch):.1f} (patrol {a.yaw:.0f}/{a.pitch:.0f})")
    else:
        check(False, "no /zr10/gimbal/attitude")
    zoom = p.zoom[-1] if p.zoom else None
    print(f"  zoom_level {zoom}")
    if problems and not a.force:
        sys.exit("\nnot recording: fix the above (or --force)")

    print("\n2. picture")
    best, sharp = None, -1
    last = 0
    for _ in range(25):  # ~1 s: keep the sharpest frame
        f, last = stream.read(after=last)
        s = cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        if s > sharp:
            best, sharp = f, s
    stream.close()
    cv2.imwrite(os.path.join(out, "image.png"), best)
    corners, ids = detect_markers(cv2.cvtColor(best, cv2.COLOR_BGR2GRAY))
    vis = best.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, np.array(ids).reshape(-1, 1) if ids else None)
    cv2.imwrite(os.path.join(out, "markers.jpg"), vis[::2, ::2])
    found = sorted(set(ids) & set(BOARD_IDS))
    check(len(found) >= 3, f"board markers found {found} of [1, 2, 3, 4] (FAST-Calib needs 3+; 4 is better)")
    if corners:
        xs = np.concatenate([c.reshape(-1, 2) for c in corners])[:, 0]
        print(f"  markers span x {xs.min():.0f}..{xs.max():.0f} px of {best.shape[1]}")

    print(f"\n3. LiDAR, {a.duration:.0f} s")
    p.recording = True
    t0 = time.monotonic()
    while time.monotonic() - t0 < a.duration:
        rclpy.spin_once(p, timeout_sec=0.02)
    p.recording = False
    gyro, att2 = still_report(p, a.duration)
    check(gyro < 0.05, f"still while recording: body {np.degrees(gyro):.2f} deg/s")
    check(len(p.raw) >= 0.8 * a.duration * 10, f"{len(p.raw)} clouds")
    write_bag(os.path.join(out, "bag"), a.topic, p.raw)

    params_text = open(a.params).read()
    clouds = [pointcloud2_xyz_ring(deserialize_message(m, PointCloud2)) for _, m in p.raw[-5:]]
    Pl, ring = np.vstack([c[0] for c in clouds]), np.concatenate([c[1] for c in clouds])
    dims = tuple(get_yaml_value(params_text, k) for k in ("delta_width_circles", "delta_height_circles", "circle_radius"))
    vals, ground = board_filter(Pl, ring, T_bl, a.box, check, dims)
    vals["plane_dist_threshold"] = 0.02  # real LiDAR noise; the code's default 0.01 suits a Livox
    vals["lidar_type"] = "grid"  # the ring-edge detector finds too few hole edges on this LiDAR
    vals.update(bag_path=os.path.abspath(os.path.join(out, "bag")), image_path=os.path.abspath(os.path.join(out, "image.png")),
                output_path=os.path.abspath(os.path.join(out, "out")), lidar_topic=a.topic)
    os.makedirs(vals["output_path"], exist_ok=True)
    open(os.path.join(out, "params.yaml"), "w").write(set_yaml_values(params_text, vals))
    json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), gimbal_attitude_median=np.median(att2, 0).tolist() if len(att2) else None,
                   zoom_level=zoom, base_T_lidar=T_bl.tolist(), base_T_lidar_source=src, ground_z_base=float(ground),
                   marker_ids=found, image_sharpness=float(sharp), clouds=len(p.raw), problems=problems,
                   marker_size=get_yaml_value(params_text, "marker_size"), fx=get_yaml_value(params_text, "fx")),
              open(os.path.join(out, "meta.json"), "w"), indent=1)
    p.destroy_node()
    rclpy.shutdown()
    print(f"\n{out}: image.png, markers.jpg, bag/, params.yaml, meta.json")
    print("RESULT:", "OK" if not problems else "with problems:\n  - " + "\n  - ".join(problems))


if __name__ == "__main__":
    main()
