#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fisheye intrinsic calibration for the front camera, using the FAST-Calib
4-marker ArUco board as the target (geometry taken from config/qr_params.yaml,
so verify those values with a tape measure first).

Capture phase: subscribes to the image topic, auto-captures a view whenever the
whole board is detected AND it sits somewhere new in the frame (so waving the
board around naturally builds coverage). Every captured frame is saved as a PNG.
A live 3x3 coverage grid is printed -- aim to light up all nine cells, corners
especially, with varied tilt and distance.

Solve phase: cv2.fisheye.calibrate over all captured views, dropping
ill-conditioned ones, then prints per-view errors and a ready-to-paste yaml
block for config/qr_params.yaml.

Usage:
    # capture 45 views then solve (Ctrl+C also stops capture and solves):
    python3 calibrate_fisheye.py --frames 45

    # re-solve later from the saved images, no camera needed:
    python3 calibrate_fisheye.py --solve-only <image_dir>
"""

import argparse
import glob
import os
import re
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import yaml

DEFAULT_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "qr_params.yaml"
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "calib_data"
)


def load_board(yaml_path):
    with open(yaml_path) as f:
        p = yaml.safe_load(f)["/**"]["ros__parameters"]
    m = p["marker_size"]
    dw = p["delta_width_qr_center"]
    dh = p["delta_height_qr_center"]
    h = m / 2.0
    centre = {1: (-dw, dh), 2: (dw, dh), 4: (dw, -dh), 3: (-dw, -dh)}
    obj = {}
    for i, (cx, cy) in centre.items():
        obj[i] = np.array(
            [(cx - h, cy + h, 0), (cx + h, cy + h, 0),
             (cx + h, cy - h, 0), (cx - h, cy - h, 0)], np.float64)
    guess = np.array([[p["fisheye_fx"], 0, p["fisheye_cx"]],
                      [0, p["fisheye_fy"], p["fisheye_cy"]],
                      [0, 0, 1]]) if p.get("fisheye_enable") else None
    return obj, guess


def make_detector():
    """detectMarkers wrapper covering both the pre-4.7 and >=4.7 ArUco APIs."""
    if hasattr(cv2.aruco, "ArucoDetector"):
        d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        par = cv2.aruco.DetectorParameters()
        par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        det = cv2.aruco.ArucoDetector(d, par)
        return lambda img: det.detectMarkers(img)
    d = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    par = cv2.aruco.DetectorParameters_create()
    par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return lambda img: cv2.aruco.detectMarkers(img, d, parameters=par)


def board_points(img, detect, obj_by_id):
    """Return (objpts Nx3, imgpts Nx2) if all four markers are found, else None."""
    corners, ids, _ = detect(img)
    if ids is None:
        return None
    ids = ids.ravel()
    if set(int(i) for i in ids) != {1, 2, 3, 4}:
        return None
    obj, pix = [], []
    for c, i in zip(corners, ids):
        obj.append(obj_by_id[int(i)])
        pix.append(c.reshape(4, 2))
    return np.vstack(obj), np.vstack(pix).astype(np.float64)


def coverage_cell(imgp, shape):
    h, w = shape[:2]
    cx, cy = imgp.mean(0)
    return min(int(3 * cy / h), 2), min(int(3 * cx / w), 2)


def print_grid(grid):
    rows = ["    " + " ".join(f"{grid[r][c]:3d}" for c in range(3)) for r in range(3)]
    print("  coverage (views per image region):")
    print("\n".join(rows))


def solve(views, image_size, K_guess):
    """cv2.fisheye.calibrate with iterative removal of ill-conditioned views."""
    objp = [v["obj"].reshape(1, -1, 3) for v in views]
    imgp = [v["img"].reshape(1, -1, 2) for v in views]
    names = [v["name"] for v in views]

    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW |
             cv2.fisheye.CALIB_CHECK_COND)
    K = K_guess.copy() if K_guess is not None else np.eye(3)
    if K_guess is not None:
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    D = np.zeros((4, 1))

    while True:
        if len(objp) < 10:
            sys.exit(f"only {len(objp)} usable views left -- capture more data")
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objp, imgp, image_size, K, D, None, None, flags,
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8))
            break
        except cv2.error as e:
            m = re.search(r"input array (\d+)", str(e))
            if not m:
                raise
            k = int(m.group(1))
            print(f"  dropping ill-conditioned view {names[k]}")
            for lst in (objp, imgp, names):
                del lst[k]

    per_view = []
    for i in range(len(objp)):
        proj, _ = cv2.fisheye.projectPoints(objp[i], rvecs[i], tvecs[i], K, D)
        e = np.linalg.norm(proj.reshape(-1, 2) - imgp[i].reshape(-1, 2), axis=1)
        per_view.append((names[i], float(np.sqrt((e ** 2).mean())), float(e.max())))
    return rms, K, D, per_view


def report(rms, K, D, per_view):
    print(f"\n{'='*62}\nCalibration over {len(per_view)} views\n{'='*62}")
    print(f"Overall RMS reprojection error: {rms:.4f} px "
          f"({'GOOD' if rms < 0.5 else 'acceptable' if rms < 1.0 else 'POOR -- recapture with better coverage'})")
    worst = sorted(per_view, key=lambda t: -t[1])[:5]
    print("Worst views (rms px / max px):")
    for n, r, mx in worst:
        print(f"  {n}: {r:.3f} / {mx:.3f}")
    print("\nPaste into config/qr_params.yaml (fisheye block):")
    print(f"      fisheye_enable: true")
    print(f"      fisheye_fx: {K[0,0]:.5f}")
    print(f"      fisheye_fy: {K[1,1]:.5f}")
    print(f"      fisheye_cx: {K[0,2]:.5f}")
    print(f"      fisheye_cy: {K[1,2]:.5f}")
    print(f"      fisheye_k1: {D[0,0]:.8f}")
    print(f"      fisheye_k2: {D[1,0]:.8f}")
    print(f"      fisheye_k3: {D[2,0]:.8f}")
    print(f"      fisheye_k4: {D[3,0]:.8f}")
    print("\n(The virtual pinhole fx/fy/cx/cy block below it can stay as-is.)")


def solve_from_dir(img_dir, detect, obj_by_id, K_guess):
    files = sorted(glob.glob(os.path.join(img_dir, "*.png")))
    if not files:
        sys.exit(f"no .png images in {img_dir}")
    views, size = [], None
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        size = img.shape[1::-1]
        bp = board_points(img, detect, obj_by_id)
        if bp is None:
            print(f"  {os.path.basename(f)}: board not found, skipped")
            continue
        views.append({"obj": bp[0], "img": bp[1], "name": os.path.basename(f)})
    print(f"{len(views)}/{len(files)} images usable")
    report(*solve(views, size, K_guess))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--topic", default="/camera/front/image_raw")
    ap.add_argument("--frames", type=int, default=45, help="target number of views")
    ap.add_argument("--seconds", type=float, default=0.0,
                help="capture time limit in seconds; 0 (default) = no limit, stop with Ctrl+C or when --frames is reached")
    ap.add_argument("--min-move", type=float, default=60.0,
                    help="min px the board must move from every previous view")
    ap.add_argument("--out-dir", default=None, help="where to save captured images")
    ap.add_argument("--solve-only", metavar="IMAGE_DIR",
                    help="skip capture; solve from previously saved images")
    args = ap.parse_args()

    obj_by_id, K_guess = load_board(args.yaml)
    detect = make_detector()

    if args.solve_only:
        solve_from_dir(args.solve_only, detect, obj_by_id, K_guess)
        return

    out_dir = args.out_dir or os.path.join(
        DEFAULT_OUT, "intrinsic_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving captured views to {out_dir}")

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("calibrate_fisheye")
    views, centres = [], []
    grid = [[0] * 3 for _ in range(3)]
    state = {"seen": 0, "size": None, "last_msg": time.time()}

    def cb(msg):
        state["last_msg"] = time.time()
        state["seen"] += 1
        if state["seen"] % 2:
            return
        buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else buf.copy()
        state["size"] = bgr.shape[1::-1]
        bp = board_points(bgr, detect, obj_by_id)
        if bp is None:
            return
        c = bp[1].mean(0)
        if centres and min(np.linalg.norm(c - o) for o in centres) < args.min_move:
            return
        name = f"view_{len(views):03d}.png"
        cv2.imwrite(os.path.join(out_dir, name), bgr)
        views.append({"obj": bp[0], "img": bp[1], "name": name})
        centres.append(c)
        r, col = coverage_cell(bp[1], bgr.shape)
        grid[r][col] += 1
        print(f"\n[{len(views)}/{args.frames}] captured {name} (board centre {c[0]:.0f},{c[1]:.0f})")
        print_grid(grid)

    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)
    print(f"Move the board around: all 9 grid cells, corners especially, "
          f"tilt it +/-30-45 deg, vary distance 1.5-3 m.\n"
          f"Auto-captures when the board lands somewhere new (> {args.min_move:.0f} px). "
          f"Ctrl+C to stop early and solve.")
    t0 = time.time()
    try:
        while (len(views) < args.frames and rclpy.ok()
               and (args.seconds <= 0 or time.time() - t0 < args.seconds)):
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        print("\ncapture interrupted -- solving with what we have")
    if args.seconds > 0 and len(views) < args.frames and time.time() - t0 >= args.seconds:
        print(f"\ntime limit ({args.seconds:.0f}s) reached with {len(views)} views -- "
              "captures stall when the board stops visiting NEW places in the frame")

    if len(views) < 10:
        sys.exit(f"\nonly {len(views)} views captured -- need at least 10 (40+ recommended)")

    border = [grid[r][c] for r in range(3) for c in range(3) if r != 1 or c != 1]
    if sum(1 for v in border if v > 0) < 4:
        print("\nWARNING: the frame edges/corners were barely covered -- the distortion"
              "\ncoefficients will be extrapolated and unreliable there. For usable"
              "\nresults, hold the board ~2.5 m away and walk it to each corner of the"
              "\nIMAGE (watch the coverage grid) before trusting the solve.")

    print(f"\nSolving fisheye calibration from {len(views)} views ...")
    report(*solve(views, state["size"], K_guess))
    print(f"\nImages saved in {out_dir} -- re-solve any time with:\n"
          f"  python3 {os.path.abspath(__file__)} --solve-only {out_dir}")


if __name__ == "__main__":
    main()
