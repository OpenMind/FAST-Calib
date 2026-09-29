#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pinhole (plumb_bob) intrinsic calibration for the Jetson camera, using the
FAST-Calib 4-marker ArUco board as the target (geometry taken from
config/qr_params.yaml, so verify those values with a tape measure first).

Same capture flow as calibrate_fisheye.py, but solves the standard
radial-tangential model (k1, k2, p1, p2) that FAST-Calib's non-fisheye path
and FAST-LIVO's `cam_model: Pinhole` both use. k3 is fixed at 0 because
neither consumer reads it.

Capture phase: subscribes to the image topic, auto-captures a view whenever the
whole board is detected AND it sits somewhere new in the frame (so waving the
board around naturally builds coverage). Every captured frame is saved as a PNG.
A live 3x3 coverage grid is printed -- aim to light up all nine cells, corners
especially, with varied tilt and distance.

Solve phase: cv2.calibrateCamera over all captured views, dropping outlier
views, then prints per-view errors and ready-to-paste yaml blocks for
config/qr_params.yaml and FAST-LIVO's camera yaml.

Usage:
    # capture 45 views then solve (Ctrl+C also stops capture and solves):
    python3 calibrate_jetson_cam.py --topic /camera/image_raw --frames 45

    # re-solve later from the saved images, no camera needed:
    python3 calibrate_jetson_cam.py --solve-only <image_dir>
"""

import argparse
import glob
import os
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
             (cx + h, cy - h, 0), (cx - h, cy - h, 0)], np.float32)
    print(f"Board from {os.path.abspath(yaml_path)}: marker {m} m, "
          f"marker-centre spacing {2*dw:.3f} x {2*dh:.3f} m")
    return obj


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
    return np.vstack(obj), np.vstack(pix).astype(np.float32)


def coverage_cell(imgp, shape):
    h, w = shape[:2]
    cx, cy = imgp.mean(0)
    return min(int(3 * cy / h), 2), min(int(3 * cx / w), 2)


def print_grid(grid):
    rows = ["    " + " ".join(f"{grid[r][c]:3d}" for c in range(3)) for r in range(3)]
    print("  coverage (views per image region):")
    print("\n".join(rows))


def solve(views, image_size):
    """cv2.calibrateCamera (k1 k2 p1 p2), iteratively dropping outlier views."""
    views = list(views)
    flags = cv2.CALIB_FIX_K3
    while True:
        if len(views) < 10:
            sys.exit(f"only {len(views)} usable views left -- capture more data")
        rms, K, D, rvecs, tvecs, _, _, per_err = cv2.calibrateCameraExtended(
            [v["obj"] for v in views], [v["img"] for v in views], image_size,
            None, None, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8))
        per_err = per_err.ravel()
        k = int(np.argmax(per_err))
        # a view far worse than the rest is a bad detection / motion blur
        if per_err[k] > max(3.0 * np.median(per_err), 1.0):
            print(f"  dropping outlier view {views[k]['name']} ({per_err[k]:.2f} px)")
            del views[k]
            continue
        break

    per_view = []
    for i, v in enumerate(views):
        proj, _ = cv2.projectPoints(v["obj"], rvecs[i], tvecs[i], K, D)
        e = np.linalg.norm(proj.reshape(-1, 2) - v["img"].reshape(-1, 2), axis=1)
        per_view.append((v["name"], float(np.sqrt((e ** 2).mean())), float(e.max())))
    return rms, K, D.ravel(), per_view


def report(rms, K, D, per_view, image_size):
    w, h = image_size
    k1, k2, p1, p2 = D[:4]
    print(f"\n{'='*62}\nPinhole calibration over {len(per_view)} views ({w}x{h})\n{'='*62}")
    print(f"Overall RMS reprojection error: {rms:.4f} px "
          f"({'GOOD' if rms < 0.5 else 'acceptable' if rms < 1.0 else 'POOR -- recapture with better coverage'})")
    worst = sorted(per_view, key=lambda t: -t[1])[:5]
    print("Worst views (rms px / max px):")
    for n, r, mx in worst:
        print(f"  {n}: {r:.3f} / {mx:.3f}")
    if abs(K[0, 2] - w / 2) > 0.15 * w or abs(K[1, 2] - h / 2) > 0.15 * h:
        print("WARNING: principal point is far from the image centre -- "
              "coverage is probably too one-sided, recapture.")

    print("\nPaste into config/qr_params.yaml (replaces the camera block):")
    print(f"      fisheye_enable: false")
    print(f"      fx: {K[0,0]:.5f}")
    print(f"      fy: {K[1,1]:.5f}")
    print(f"      cx: {K[0,2]:.5f}")
    print(f"      cy: {K[1,2]:.5f}")
    print(f"      k1: {k1:.8f}")
    print(f"      k2: {k2:.8f}")
    print(f"      p1: {p1:.8f}")
    print(f"      p2: {p2:.8f}")

    print("\nFAST-LIVO camera yaml (e.g. fast_livo/config/camera_rover.yaml):")
    print("/**:\n  ros__parameters:")
    print("    cam_model: Pinhole")
    print(f"    cam_width: {w}")
    print(f"    cam_height: {h}")
    print("    scale: 1.0")
    print(f"    cam_fx: {K[0,0]:.5f}")
    print(f"    cam_fy: {K[1,1]:.5f}")
    print(f"    cam_cx: {K[0,2]:.5f}")
    print(f"    cam_cy: {K[1,2]:.5f}")
    print(f"    cam_d0: {k1:.8f}")
    print(f"    cam_d1: {k2:.8f}")
    print(f"    cam_d2: {p1:.8f}")
    print(f"    cam_d3: {p2:.8f}")


def solve_from_dir(img_dir, detect, obj_by_id):
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
    report(*solve(views, size), size)


def to_bgr(msg):
    buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step)
    buf = buf[:, :msg.width * (msg.step // msg.width)].reshape(msg.height, msg.width, -1)
    if msg.encoding == "rgb8":
        return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)
    if msg.encoding in ("mono8", "8UC1"):
        return cv2.cvtColor(buf, cv2.COLOR_GRAY2BGR)
    return buf.copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--topic", default="/camera/image_raw")
    ap.add_argument("--frames", type=int, default=45, help="target number of views")
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="capture time limit in seconds; 0 (default) = no limit, stop with Ctrl+C or when --frames is reached")
    ap.add_argument("--min-move", type=float, default=None,
                    help="min px the board must move from every previous view "
                         "(default: 5%% of the image width, e.g. 32 px at 640)")
    ap.add_argument("--out-dir", default=None, help="where to save captured images")
    ap.add_argument("--solve-only", metavar="IMAGE_DIR",
                    help="skip capture; solve from previously saved images")
    args = ap.parse_args()

    obj_by_id = load_board(args.yaml)
    detect = make_detector()

    if args.solve_only:
        solve_from_dir(args.solve_only, detect, obj_by_id)
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
    node = Node("calibrate_jetson_cam")
    views, centres = [], []
    grid = [[0] * 3 for _ in range(3)]
    state = {"seen": 0, "size": None, "min_move": args.min_move}

    def cb(msg):
        state["seen"] += 1
        if state["seen"] % 2:
            return
        bgr = to_bgr(msg)
        state["size"] = bgr.shape[1::-1]
        if state["min_move"] is None:
            state["min_move"] = 0.05 * bgr.shape[1]
            print(f"First frame {bgr.shape[1]}x{bgr.shape[0]} ({msg.encoding}); "
                  f"min-move {state['min_move']:.0f} px")
        bp = board_points(bgr, detect, obj_by_id)
        if bp is None:
            return
        c = bp[1].mean(0)
        if centres and min(np.linalg.norm(c - o) for o in centres) < state["min_move"]:
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
    print(f"Listening on {args.topic}. Move the board around: all 9 grid cells, "
          f"corners especially, tilt it +/-30-45 deg, vary distance 1-3 m.\n"
          f"Auto-captures when the board lands somewhere new. Ctrl+C to stop early and solve.")
    t0 = time.time()
    try:
        while (len(views) < args.frames and rclpy.ok()
               and (args.seconds <= 0 or time.time() - t0 < args.seconds)):
            rclpy.spin_once(node, timeout_sec=0.1)
            if state["seen"] == 0 and time.time() - t0 > 5.0:
                print(f"no images on {args.topic} yet -- is the camera running?")
                t0 = time.time()
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
              "\ncoefficients will be extrapolated and unreliable there. Walk the board"
              "\nto each corner of the IMAGE (watch the coverage grid) and recapture.")

    print(f"\nSolving pinhole calibration from {len(views)} views ...")
    report(*solve(views, state["size"]), state["size"])
    print(f"\nImages saved in {out_dir} -- re-solve any time with:\n"
          f"  python3 {os.path.abspath(__file__)} --solve-only {out_dir}")


if __name__ == "__main__":
    main()
