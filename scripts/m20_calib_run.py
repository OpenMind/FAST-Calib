#!/usr/bin/env python3
"""Run FAST-Calib on every scene of a session, then all scenes together, and give the camera in base_link.

    source ~/calib_ws/install/setup.bash
    python3 scripts/m20_calib_run.py                       # today's session (calib_data/m20_YYYYMMDD)
    python3 scripts/m20_calib_run.py calib_data/m20_20261014 --scenes scene1 scene2 scene3

Each scene: `fast_calib` on its params.yaml (its RMSE and T_cam_lidar). Then each scene's
hole centres go into one record and `multi_fast_calib` solves them together. Writes into
<session>/out/:
  camera_extrinsics.yaml   T_cam_lidar, T_base_cam (camera in base_link), intrinsics, gimbal angle
  overlay_<scene>.jpg      the LiDAR projected onto each picture, coloured by distance
The RViz view of each single scene: add --rviz.
"""
import argparse
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20_common import REPO, get_yaml_value, pointcloud2_xyz_ring  # noqa: E402

SAMPLE_FX = 2478.48702288166  # the sample data's intrinsics still in qr_params.yaml


def run_node(exe, params, wait_for, timeout=60, rviz=False):
    """Run a FAST-Calib executable until `wait_for` shows up in its output (it spins forever after)."""
    cmd = ["ros2", "run", "fast_calib", exe, "--ros-args", "--params-file", params]
    pr = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, start_new_session=True)
    viz = None
    if rviz:
        share = subprocess.run(["ros2", "pkg", "prefix", "fast_calib"], capture_output=True, text=True).stdout.strip()
        viz = subprocess.Popen(["rviz2", "-d", f"{share}/share/fast_calib/rviz_cfg/fast_livo2.rviz"], start_new_session=True)
    out, t0 = [], time.monotonic()
    os.set_blocking(pr.stdout.fileno(), False)
    while time.monotonic() - t0 < timeout:
        line = pr.stdout.readline()
        if line:
            out.append(re.sub(r"\x1b\[[0-9;]*m", "", line))
            if wait_for in out[-1]:
                time.sleep(1.0)
                try:
                    out += [re.sub(r"\x1b\[[0-9;]*m", "", l) for l in pr.stdout.readlines()]
                except TypeError:
                    pass
                break
        elif pr.poll() is not None:
            break
        else:
            time.sleep(0.05)
    if rviz and viz:
        input("  RViz is open: look at the clouds, then press Enter ")
        os.killpg(viz.pid, signal.SIGINT)
    if pr.poll() is None:
        os.killpg(pr.pid, signal.SIGINT)
        pr.wait(5)
    return "".join(out)


def parse_matrix(text, after):
    i = text.rfind(after)
    if i < 0:
        return None
    nums = re.findall(r"-?\d+\.\d+(?:e-?\d+)?", text[i + len(after):])[:16]
    return np.array(nums, float).reshape(4, 4) if len(nums) == 16 else None


def rot_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def overlay(scene, T_cl, K, D, out_path):
    """Project the scene's LiDAR (all of it) onto its picture, coloured by distance."""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    r = rosbag2_py.SequentialReader()
    r.open(rosbag2_py.StorageOptions(uri=os.path.join(scene, "bag"), storage_id="sqlite3"), rosbag2_py.ConverterOptions("cdr", "cdr"))
    msgs = []
    while r.has_next():
        msgs.append(r.read_next()[1])
    P = np.vstack([pointcloud2_xyz_ring(deserialize_message(m, PointCloud2))[0] for m in msgs[-3:]])
    Pc = P @ T_cl[:3, :3].T + T_cl[:3, 3]
    Pc = Pc[Pc[:, 2] > 0.3]
    uv = cv2.projectPoints(Pc.astype(np.float64), np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    img = cv2.imread(os.path.join(scene, "image.png"))
    h, w = img.shape[:2]
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    d = np.linalg.norm(Pc[ok], axis=1)
    col = cv2.applyColorMap(np.clip((d - 1) / 9 * 255, 0, 255).astype(np.uint8).reshape(-1, 1), cv2.COLORMAP_TURBO).reshape(-1, 3)
    for (u, v), c in zip(uv[ok].astype(int), col):
        cv2.circle(img, (u, v), 2, tuple(int(x) for x in c), -1)
    cv2.imwrite(out_path, img[::2, ::2])
    return ok.sum()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session", nargs="?", default=os.path.join(REPO, "calib_data", "m20_" + time.strftime("%Y%m%d")))
    ap.add_argument("--scenes", nargs="*", help="default: every folder with a params.yaml")
    ap.add_argument("--rviz", action="store_true")
    a = ap.parse_args()
    scenes = [os.path.join(a.session, s) for s in a.scenes] if a.scenes else \
        sorted(os.path.dirname(p) for p in glob.glob(os.path.join(a.session, "scene*", "params.yaml")))
    if not scenes:
        sys.exit(f"no scenes in {a.session}")
    ptext = open(os.path.join(scenes[0], "params.yaml")).read()
    if abs(get_yaml_value(ptext, "fx", 0) - SAMPLE_FX) < 1e-3:
        sys.exit("params.yaml still has the sample data's intrinsics: run m20_intrinsics_calibrate.py --write, then re-capture")
    metas = [json.load(open(os.path.join(s, "meta.json"))) for s in scenes]
    zooms = {m["zoom_level"] for m in metas}
    if len(zooms) > 1:
        print(f"WARNING: zoom differs between scenes: {zooms}")

    results = {}
    for s in scenes:
        name = os.path.basename(s)
        rec = os.path.join(s, "out", "circle_center_record.txt")
        if os.path.exists(rec):
            os.remove(rec)  # the node appends: one block per run
        print(f"\n== {name}")
        log = run_node("fast_calib", os.path.join(s, "params.yaml"), "single_calib_result.txt", rviz=a.rviz)
        open(os.path.join(s, "out", "fast_calib.log"), "w").write(log)
        for key in ("Loaded", "Depth filtered", "Plane cloud", "edge points", "hole centers found", "completed", "RMSE", "Error", "not 4"):
            for line in log.splitlines():
                if key in line:
                    print("  ", line.strip()[-140:])
        T = parse_matrix(log, "T_cam_lidar =")
        m = re.search(r"RMSE:\s*([\d.]+)", log)
        if T is None or not os.path.exists(rec):
            print(f"  FAILED: no result (see {s}/out/fast_calib.log)")
            continue
        rmse = float(m.group(1)) if m else None
        print(f"  RMSE {rmse} m")
        if rmse is None or rmse > 0.03:
            # holes matched wrongly or a bad hole fit: 3-5 mm is normal
            print(f"  LEFT OUT: RMSE above 3 cm (look at {s}/out/qr_detect.png: red crosses should sit on the holes)")
            continue
        if rmse > 0.015:
            print("  WARNING: RMSE above 1.5 cm, check qr_detect.png")
        results[name] = dict(T=T, rmse=rmse, record=open(rec).read().strip().splitlines()[-3:])

    if not results:
        sys.exit("\nno scene calibrated")
    out = os.path.join(a.session, "out")
    os.makedirs(out, exist_ok=True)
    names = list(results)
    if len(names) > 3:  # multi_fast_calib solves the last 3 records: give it the 3 best scenes
        names = sorted(names, key=lambda n: results[n]["rmse"] or 9)[:3]
        print(f"\nusing the 3 scenes with the lowest RMSE: {names}")
    if len(results) >= 3:
        open(os.path.join(out, "circle_center_record.txt"), "w").write("\n".join(sum((results[n]["record"] for n in names), [])) + "\n")
        mparams = os.path.join(out, "params.yaml")
        open(mparams, "w").write(re.sub(r'(^\s*output_path:\s*)"[^"]*"', rf'\1"{os.path.abspath(out)}"', ptext, flags=re.M))
        print(f"\n== all {len(names)} scenes together")
        log = run_node("multi_fast_calib", mparams, "multi_calib_result.txt")
        open(os.path.join(out, "multi_fast_calib.log"), "w").write(log)
        txt = open(os.path.join(out, "multi_calib_result.txt")).read() if os.path.exists(os.path.join(out, "multi_calib_result.txt")) else ""
        nums = re.findall(r"-?\d+\.\d+", txt.split("Rcl:")[-1]) if "Rcl:" in txt else []
        if len(nums) >= 12:
            v = np.array(nums[:12], float)
            T_cl = np.eye(4)
            T_cl[:3, :3], T_cl[:3, 3] = v[:9].reshape(3, 3), v[9:12]
        else:
            print("  multi-scene failed, using the scene with the lowest RMSE")
            T_cl = None
        for line in log.splitlines():
            if "RMSE" in line or "rror" in line:
                print("  ", line.strip())
    else:
        print(f"\nonly {len(results)} scene(s) calibrated: FAST-Calib's joint solve needs 3; using the best single scene")
        T_cl = None
    if T_cl is None:
        best = min(names, key=lambda n: results[n]["rmse"] or 9)
        T_cl = results[best]["T"]

    print("\nscene agreement with the result (rotation deg, translation cm):")
    for n in names:
        D = np.linalg.inv(T_cl) @ results[n]["T"]
        print(f"  {n}: {rot_deg(D[:3, :3]):.2f} deg, {np.linalg.norm(D[:3, 3]) * 100:.1f} cm   (RMSE {results[n]['rmse']} m)")

    T_bl = np.array(metas[0]["base_T_lidar"])
    T_bc = T_bl @ np.linalg.inv(T_cl)  # base_link <- camera
    zc = T_bc[:3, 2]
    yaw = np.degrees(np.arctan2(zc[1], zc[0]))
    pitch = np.degrees(np.arcsin(-zc[2]))
    roll = np.degrees(np.arctan2(-T_bc[2, 0], -T_bc[2, 1]))  # 0 when the image's x axis is level
    print(f"\ncamera in base_link: at ({T_bc[0, 3]:.3f}, {T_bc[1, 3]:.3f}, {T_bc[2, 3]:.3f}) m, "
          f"looking yaw {yaw:.1f} pitch {pitch:+.1f} (+ = down) roll {roll:+.1f} deg")
    g = metas[0]["gimbal_attitude_median"]
    print(f"  (base_link yaw -90 = straight right; the gimbal reported yaw {g[0]:.1f} pitch {g[1]:.1f})" if g else "")
    # gimbal yaw +N (to the right) is base_link yaw -N; the mount is not exact, so allow 10 deg
    if g and (abs(yaw + g[0]) > 10 or abs(pitch - g[1]) > 10):
        print("  WARNING: that is not the gimbal's angle: check the result before using it")
    # the hole rectangle is 180-deg symmetric: holes matched the wrong way round fit just as well
    # and turn the camera upside down about its own axis
    if abs(roll) > 10:
        print(f"  WARNING: camera roll {roll:+.0f} deg: the LiDAR holes were matched to the wrong camera holes "
              "(check lidar_up_* and the qr_detect.png crosses)")

    K = np.array([[get_yaml_value(ptext, "fx"), 0, get_yaml_value(ptext, "cx")], [0, get_yaml_value(ptext, "fy"), get_yaml_value(ptext, "cy")], [0, 0, 1]])
    D = np.array([get_yaml_value(ptext, k) for k in ("k1", "k2", "p1", "p2")] + [0.0])
    for s in scenes:
        try:
            n = overlay(s, T_cl, K, D, os.path.join(out, f"overlay_{os.path.basename(s)}.jpg"))
            print(f"  overlay_{os.path.basename(s)}.jpg ({n} points in the picture)")
        except Exception as e:  # noqa: BLE001
            print(f"  overlay for {s} failed: {e}")

    f = lambda M: "[" + ", ".join(f"{x:.6f}" for x in M.ravel()) + "]"  # noqa: E731
    with open(os.path.join(out, "camera_extrinsics.yaml"), "w") as fo:
        fo.write(f"# M20 SIYI PTZ <-> front LiDAR, FAST-Calib {time.strftime('%Y-%m-%d %H:%M')}, scenes {names}\n")
        fo.write("# Valid only at this gimbal angle and zoom. Row-major 4x4.\n")
        fo.write(f"gimbal_attitude: {metas[0]['gimbal_attitude_median']}\nzoom_level: {metas[0]['zoom_level']}\n")
        fo.write(f"image_width: {cv2.imread(os.path.join(scenes[0], 'image.png')).shape[1]}\n")
        for k in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"):
            fo.write(f"{k}: {get_yaml_value(ptext, k)}\n")
        fo.write(f"T_cam_lidar_front: {f(T_cl)}   # p_cam = T * p_lidar\n")
        fo.write(f"T_base_lidar_front: {f(T_bl)}   # from TF ({metas[0]['base_T_lidar_source']})\n")
        fo.write(f"T_base_cam: {f(T_bc)}   # p_base = T * p_cam (camera optical frame: x right, y down, z forward)\n")
    print(f"\nwrote {out}/camera_extrinsics.yaml")


if __name__ == "__main__":
    main()
