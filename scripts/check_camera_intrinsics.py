#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Check the fisheye intrinsics in config/qr_params.yaml against live camera frames.

Undistorts each frame to the virtual pinhole exactly like undistortFisheyeImage()
in common_lib.h, detects the 4-marker ArUco board, solves the board pose, and
reports the reprojection error. Wave the board around so it lands in different
parts of the frame -- especially the corners, where fisheye distortion error
shows up first.

Reads:
  - reprojection error overall and binned by distance from the image centre
  - measured marker-centre spacings vs the nominal values in the yaml

Interpreting the output:
  - mean reprojection error < 0.5 px, flat across radius  -> intrinsics are good
  - error growing towards the periphery                   -> distortion model is off,
                                                             recalibrate
  - flat error but spacings off by a constant %           -> scale error: either the
                                                             focal length or the
                                                             printed board size.
                                                             Measure the board first.

Usage:
    python3 check_camera_intrinsics.py [--topic /camera/front/image_raw] [--frames 40]
    python3 check_camera_intrinsics.py --image /path/to/frame.png
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import yaml

DEFAULT_YAML = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "config", "qr_params.yaml"
)


def load_params(path):
    with open(path) as f:
        doc = yaml.safe_load(f)
    p = doc["/**"]["ros__parameters"]
    K_f = np.array([[p["fisheye_fx"], 0, p["fisheye_cx"]],
                    [0, p["fisheye_fy"], p["fisheye_cy"]],
                    [0, 0, 1]])
    D_f = np.array([p["fisheye_k1"], p["fisheye_k2"], p["fisheye_k3"], p["fisheye_k4"]])
    K_p = np.array([[p["fx"], 0, p["cx"]], [0, p["fy"], p["cy"]], [0, 0, 1]])
    D_p = np.array([p["k1"], p["k2"], p["p1"], p["p2"], 0.0])
    return p, K_f, D_f, K_p, D_p


def make_detector():
    """detectMarkers wrapper covering both the pre-4.7 and the >=4.7 ArUco APIs."""
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


def board_object_points(p, ids):
    """3D corners of the detected markers in the board frame (ids 1,2,4,3 layout)."""
    m, dw, dh = p["marker_size"], p["delta_width_qr_center"], p["delta_height_qr_center"]
    h = m / 2.0
    centre = {1: (-dw, dh), 2: (dw, dh), 4: (dw, -dh), 3: (-dw, -dh)}
    obj = []
    for i in ids:
        cx, cy = centre[int(i)]
        obj += [(cx - h, cy + h, 0), (cx + h, cy + h, 0),
                (cx + h, cy - h, 0), (cx - h, cy - h, 0)]
    return np.array(obj, np.float32)


def analyse(und, p, K_p, D_p, detect, samples):
    corners, ids, _ = detect(und)
    if ids is None or len(ids) < 4:
        return False
    ids = ids.ravel()
    if set(int(i) for i in ids) != {1, 2, 3, 4}:
        return False

    obj = board_object_points(p, ids)
    imgp = np.array([c for cs in corners for c in cs.reshape(4, 2)], np.float32)

    ok, rvec, tvec = cv2.solvePnP(obj, imgp, K_p, D_p)
    if not ok:
        return False
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K_p, D_p)
    err = np.linalg.norm(proj.reshape(-1, 2) - imgp, axis=1)

    h, w = und.shape[:2]
    rad = np.linalg.norm(imgp - np.array([w / 2.0, h / 2.0]), axis=1)

    # independent per-marker pose -> metric spacings, free of the board PnP model
    m = p["marker_size"]
    sq = np.array([(-m/2, m/2, 0), (m/2, m/2, 0), (m/2, -m/2, 0), (-m/2, -m/2, 0)], np.float32)
    pos = {}
    for c, i in zip(corners, ids):
        _, _, tv = cv2.solvePnP(sq, c.reshape(4, 2), K_p, D_p, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        pos[int(i)] = tv.ravel()

    dw, dh = p["delta_width_qr_center"], p["delta_height_qr_center"]
    def dist(a, b):
        return float(np.linalg.norm(pos[a] - pos[b]))
    spacing = [
        (dist(1, 2), 2 * dw), (dist(3, 4), 2 * dw),
        (dist(1, 3), 2 * dh), (dist(2, 4), 2 * dh),
    ]

    samples["err"].append(err)
    samples["rad"].append(rad)
    samples["depth"].append(float(tvec[2].item()))
    samples["scale"] += [meas / nom for meas, nom in spacing]
    return True


def report(samples):
    if not samples["err"]:
        print("No usable board detections -- is the whole target visible?")
        return
    err = np.concatenate(samples["err"])
    rad = np.concatenate(samples["rad"])
    n = len(samples["err"])

    print(f"\n{'='*62}\n{n} usable view(s), {len(err)} corner observations\n{'='*62}")
    print(f"Reprojection error : mean {err.mean():.3f} px   median {np.median(err):.3f} px"
          f"   p95 {np.percentile(err,95):.3f} px   max {err.max():.3f} px")

    print("\nError vs distance from image centre (this is the important one):")
    edges = [0, 150, 300, 450, 600, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (rad >= lo) & (rad < hi)
        if m.sum() < 4:
            continue
        tag = f"{lo:4.0f}-{hi:4.0f} px" if hi < 1e9 else f"{lo:4.0f}+     px"
        print(f"  r {tag}: n={m.sum():5d}  mean {err[m].mean():6.3f} px  max {err[m].max():6.3f} px")

    sc = np.array(samples["scale"])
    print(f"\nMeasured / nominal board spacing: {sc.mean():.4f} "
          f"(+/- {sc.std():.4f})  ->  {100*(sc.mean()-1):+.2f}% scale error")
    print(f"Board depth range seen: {min(samples['depth']):.2f} - {max(samples['depth']):.2f} m")

    print("\nVerdict:")
    if err.mean() < 0.5:
        print("  Reprojection is tight -- the distortion model is fine.")
    elif err.mean() < 1.0:
        print("  Reprojection is acceptable but not great.")
    else:
        print("  Reprojection error is HIGH for a rigid planar target. Recalibrate,")
        print("  or check that the board geometry in the yaml matches the real board.")
    far = rad > 400
    if far.sum() > 10 and err[far].mean() > 2 * err[~far].mean():
        print("  Error grows sharply towards the periphery -> the fisheye distortion")
        print("  coefficients are the problem. Recalibrate with corner coverage.")
    if abs(sc.mean() - 1) > 0.005:
        print(f"  Systematic {100*(sc.mean()-1):+.2f}% scale error. Measure the printed board")
        print("  with a tape measure first: if the board matches the yaml, the focal")
        print("  length is wrong by that amount and must be recalibrated.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", default=DEFAULT_YAML)
    ap.add_argument("--topic", default="/camera/front/image_raw")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--image", help="analyse a single image file instead of a topic")
    args = ap.parse_args()

    p, K_f, D_f, K_p, D_p = load_params(args.yaml)
    detect = make_detector()
    samples = {"err": [], "rad": [], "depth": [], "scale": []}

    def handle(bgr):
        und = cv2.fisheye.undistortImage(bgr, K_f, D_f, Knew=K_p, new_size=bgr.shape[1::-1])
        return analyse(und, p, K_p, D_p, detect, samples)

    if args.image:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"cannot read {args.image}")
        print("board detected" if handle(img) else "board NOT detected")
        report(samples)
        return

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = Node("check_camera_intrinsics")
    state = {"n": 0, "seen": 0}

    def cb(msg):
        buf = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, -1)
        bgr = cv2.cvtColor(buf, cv2.COLOR_RGB2BGR) if msg.encoding == "rgb8" else buf.copy()
        state["seen"] += 1
        if state["seen"] % 3:          # subsample: consecutive frames are redundant
            return
        if handle(bgr):
            state["n"] += 1
            print(f"\r  captured {state['n']}/{args.frames} views", end="", flush=True)

    node.create_subscription(Image, args.topic, cb, qos_profile_sensor_data)
    print(f"Move the board around the frame (corners especially). "
          f"Collecting up to {args.frames} views / {args.seconds:.0f} s ...")
    t0 = time.time()
    while state["n"] < args.frames and time.time() - t0 < args.seconds and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.1)
    report(samples)


if __name__ == "__main__":
    main()
