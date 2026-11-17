#!/usr/bin/env python3
"""Guided camera-LiDAR calibration on a web page: live video, step-by-step prompts, automatic capture.

    source /opt/ros/jazzy/setup.bash && source ~/calib_ws/install/setup.bash
    python3 scripts/m20_ptz_set.py                 # the PTZ head straight ahead, level, zoom 1x
    python3 scripts/m20_extrinsic_guide.py
    # open http://<robot-ip>:8098 on a phone

For each of the three scenes the page says where the board should be and shows, live, what
the camera and the LiDAR see of it. When every check has passed for 3 s it records the scene
by itself (5 s, keep still). After the third scene it runs the calibration and shows the result.
A session already started (same --session) carries on from its next scene.
"""
import argparse
import collections
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Imu, PointCloud2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20_common import (LIDAR_TOPIC, LUMEN_URL, QR_PARAMS, REPO, RTSP_URL, LumenGimbal, Stream,  # noqa: E402
                        board_marker_corners, lumen_gimbal_zero, board_points, get_yaml_value, lookup_base_lidar,
                        pointcloud2_xyz_ring, set_yaml_values)
from m20_calib_run import run_node  # noqa: E402
from m20_scene_capture import board_filter, write_bag  # noqa: E402

HOLD_S = 3.0          # every check green this long before recording
VIDEO_FPS = 12        # the page's live picture, an MJPEG stream
VIDEO_W = 960         # its width, px
# Each scene: board angle to the camera's line of sight (deg), the board centre's offset across the picture
# (m, + = right, camera frame) and its distance from the camera (m). Different poses and places pin down
# depth and rotation together: one board pose alone leaves them trading off against each other.
SCENES = {
    1: dict(angle=(-10, 10), side=(-0.25, 0.25), dist=(2.4, 3.2), text="Board FACING the camera, in the MIDDLE of the picture"),
    2: dict(angle=(15, 40), side=(-0.4, 0.4), dist=(2.3, 3.3), text="Turn the board about 25 deg so the angle reads +25"),
    3: dict(angle=(-40, -15), side=(-0.4, 0.4), dist=(2.3, 3.3), text="Turn the board about 25 deg the other way, angle -25"),
    4: dict(angle=(-15, 15), side=(-0.75, -0.3), dist=(2.6, 3.4), text="Board facing the camera, on the LEFT side of the picture (all 4 markers in view)"),
    5: dict(angle=(-15, 15), side=(0.3, 0.75), dist=(2.6, 3.4), text="Board facing the camera, on the RIGHT side of the picture (all 4 markers in view)"),
    6: dict(angle=(-15, 15), side=(-0.3, 0.3), dist=(1.9, 2.35), text="Board facing the camera, NEARER: about 2.1 m, in the middle"),
}
N_SCENES = len(SCENES)
BOARD_HALF = (0.72, 0.48)       # m, the board's half width / height (1.44 x 0.96 measured by the LiDAR)


class Sensors(Node):
    def __init__(self, topic):
        super().__init__("m20_extrinsic_guide")
        self.imu = collections.deque(maxlen=2000)
        self.clouds = collections.deque(maxlen=6)
        self.rec = None  # list while recording
        self.create_subscription(Imu, "/IMU", lambda m: self.imu.append(
            (time.monotonic(), np.linalg.norm([m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z]))),
            qos_profile_sensor_data)
        self.create_subscription(PointCloud2, topic, self._cloud, qos_profile_sensor_data, raw=True)

    def _cloud(self, data):
        self.clouds.append(data)
        if self.rec is not None:
            self.rec.append((time.time_ns(), data))

    def gyro_max(self, s):
        now = time.monotonic()
        g = [v for t, v in list(self.imu) if t > now - s]
        return max(g) if g else None



class Guide:
    def __init__(self, a):
        self.a = a
        self.ptext = open(a.params).read()
        g = lambda k: get_yaml_value(self.ptext, k)  # noqa: E731
        self.K = np.array([[g("fx"), 0, g("cx")], [0, g("fy"), g("cy")], [0, 0, 1]])
        self.D = np.array([g("k1"), g("k2"), g("p1"), g("p2"), 0.0])
        self.half_fov = float(np.degrees(np.arctan(g("cx") / g("fx"))))   # deg, left edge of the picture
        self.obj_by_id = board_marker_corners(self.ptext)
        self.dims = (g("delta_width_circles"), g("delta_height_circles"), g("circle_radius"))
        self.lock = threading.Lock()
        self.frame_jpg = self.lidar_jpg = b""
        self.frame_seq = 0
        self.frame_cv = threading.Condition()
        self.state = "waiting"      # waiting | recording | calibrating | done
        self.scene = self.next_scene()
        self.last_ty = 0.0           # the board centre's height in the camera frame, kept for the target outline
        self.checks, self.message, self.log = [], "", []
        self.hold_since = None
        self.board_hist = collections.deque(maxlen=40)
        self.lidar = {"checks": [], "vals": None, "ground": None, "t": 0}
        self.angle = self.dist = None
        self.best = None
        self.results = {}          # scene -> (ok, text) from FAST-Calib run on it right after recording

    def check_scene(self, k):
        """FAST-Calib on a just-recorded scene: are the 4 holes seen directly, and the RMSE."""
        self.results[k] = (None, f"scene {k}: checking with FAST-Calib...")
        d = os.path.join(self.a.session, f"scene{k}")
        rec = os.path.join(d, "out", "circle_center_record.txt")
        if os.path.exists(rec):
            os.remove(rec)
        log = run_node("fast_calib", os.path.join(d, "params.yaml"), "single_calib_result.txt", timeout=40)
        open(os.path.join(d, "out", "fast_calib.log"), "w").write(log)
        m = __import__("re").search(r"RMSE:\s*([\d.]+)", log)
        pm = __import__("re").search(r"least empty ([\d.]+) m of free space, .*centroids within ([\d.]+) m", log)
        if not m or not os.path.exists(rec):
            self.results[k] = (False, f"scene {k}: FAST-Calib found no hole set: press Redo, move the board")
        elif pm:   # pattern: the scene's own RMSE is ~0 by construction (two exact rectangles), the hole fit is the check
            self.results[k] = (True, f"scene {k}: FAST-Calib found all 4 holes (free space {float(pm.group(1)) * 100:.0f} cm, "
                                     f"fit within {float(pm.group(2)) * 100:.1f} cm)")
        elif "completed" in log:
            self.results[k] = (False, f"scene {k}: only 3 holes seen, the 4th inferred (RMSE {float(m.group(1)) * 1000:.0f} mm): "
                                      "usable, but better to press Redo and move the board further forward")
        else:
            self.results[k] = (True, f"scene {k}: all 4 holes seen directly, RMSE {float(m.group(1)) * 1000:.0f} mm")

    def next_scene(self):
        for k in SCENES:
            if not os.path.exists(os.path.join(self.a.session, f"scene{k}", "params.yaml")):
                return k
        return N_SCENES + 1

    # ---- camera side: board pose from its 16 marker corners
    def target(self, k, ty):
        """The pose scene k wants, in the camera frame: (R, t), board centre at height ty (camera y)."""
        sc = SCENES[k]
        side, dist, ang = np.mean(sc["side"]), np.mean(sc["dist"]), np.mean(sc["angle"])
        t = np.array([side, ty, np.sqrt(max(dist ** 2 - side ** 2 - ty ** 2, 0.5))])
        face = np.diag([1.0, -1.0, -1.0])   # board x right, y up, normal towards the camera
        best = None
        for sign in (1, -1):
            a = np.radians(sign * ang)
            Ry = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
            Rb = Ry @ face
            n, los = Rb[:, 2], -t / np.linalg.norm(t)
            got = np.degrees(np.arctan2(los[2] * n[0] - los[0] * n[2], los[0] * n[0] + los[2] * n[2]))
            if best is None or abs(got - ang) < best[0]:
                best = (abs(got - ang), Rb)
        return best[1], t

    def hints(self, k):
        """What to change to reach scene k's pose, in words."""
        if self.dist is None:
            return []
        sc, out = SCENES[k], []
        (s0, s1), (d0, d1), (a0, a1) = sc["side"], sc["dist"], sc["angle"]
        if not s0 <= self.side <= s1:
            mv = np.mean(sc["side"]) - self.side
            out.append(f"move the board {abs(mv):.2f} m to the robot's {'RIGHT (picture right)' if mv > 0 else 'LEFT (picture left)'}")
        if not d0 <= self.dist <= d1:
            mv = np.mean(sc["dist"]) - self.dist
            out.append(f"move it {abs(mv):.1f} m {'FURTHER from' if mv > 0 else 'NEARER to'} the robot")
        if not a0 <= self.angle <= a1:
            out.append(f"turn it to {np.mean(sc['angle']):+.0f} deg (now {self.angle:+.0f})")
        return out

    def draw_target(self, vis, k, ty, now=None):
        """The board outline where scene k wants it (yellow), and an arrow from the board's centre to it."""
        if k not in SCENES:
            return
        Rb, t = self.target(k, ty)
        bw, bh = BOARD_HALF
        corners = np.array([[-bw, bh, 0], [bw, bh, 0], [bw, -bh, 0], [-bw, -bh, 0], [0, 0, 0]], np.float64)
        q = cv2.projectPoints(corners, cv2.Rodrigues(Rb)[0], t, self.K, self.D)[0].reshape(-1, 2) / 2
        cv2.polylines(vis, [q[:4].astype(np.int32)], True, (0, 230, 255), 3)
        cv2.putText(vis, "PUT THE BOARD HERE", tuple((q[0] + [6, -10]).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 230, 255), 2)
        if now is not None:
            c = cv2.projectPoints(np.zeros((1, 3)), now[0], now[1], self.K, self.D)[0].reshape(2) / 2
            if np.linalg.norm(c - q[4]) > 15:
                cv2.arrowedLine(vis, tuple(c.astype(int)), tuple(q[4].astype(int)), (0, 230, 255), 3, tipLength=0.08)

    def camera_check(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        vis = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        cv2.drawMarker(vis, (vis.shape[1] // 2, vis.shape[0] // 2), (0, 0, 255), cv2.MARKER_CROSS, 40, 2)   # picture centre
        r = board_points(gray, self.obj_by_id)
        if r is None:
            self.angle = self.dist = self.side = None
            self.holes_in_view = False
            self.board_hist.clear()
            self.draw_target(vis, self.scene, self.last_ty)
            return vis, gray, False
        obj, img = r
        ok, rv, tv = cv2.solvePnP(obj, img, self.K, self.D)
        R = cv2.Rodrigues(rv)[0]
        t = tv.ravel()
        n = R[:, 2]                          # board normal, towards the camera
        los = -t / np.linalg.norm(t)         # from the board to the camera
        self.angle = float(np.degrees(np.arctan2(los[2] * n[0] - los[0] * n[2], los[0] * n[0] + los[2] * n[2])))
        self.dist = float(np.linalg.norm(t))
        self.side = float(t[0])
        self.board_hist.append((time.monotonic(), t))
        for p in img:
            cv2.circle(vis, tuple((p / 2).astype(int)), 4, (0, 255, 0), -1)
        # the 4 holes where the markers put them: they must sit on the real holes, all inside the picture
        w_c, h_c, r_c = self.dims
        rim = np.array([[x + r_c * np.cos(a), y + r_c * np.sin(a), 0.0] for x in (-w_c / 2, w_c / 2) for y in (-h_c / 2, h_c / 2)
                        for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)], np.float32)
        q = cv2.projectPoints(rim, rv, tv, self.K, self.D)[0].reshape(4, 24, 2)
        H, W = frame.shape[:2]
        self.holes_in_view = bool(((q[..., 0] > 2) & (q[..., 0] < W - 2) & (q[..., 1] > 2) & (q[..., 1] < H - 2)).all())
        for poly in q:
            cv2.polylines(vis, [(poly / 2).astype(np.int32)], True, (0, 255, 0) if self.holes_in_view else (0, 0, 255), 2)
        cv2.drawFrameAxes(vis, self.K / np.array([[2], [2], [1]]), self.D, rv, tv, 0.3)
        self.last_ty = float(t[1])
        self.draw_target(vis, self.scene, self.last_ty, now=(rv, tv))
        return vis, gray, True

    def board_still(self, s=1.5):
        now = time.monotonic()
        ts = np.array([t for tt, t in self.board_hist if tt > now - s])
        return len(ts) > 5 and np.ptp(ts, axis=0).max() < 0.02

    # ---- LiDAR side: the board plane and its holes, a few times a second
    def lidar_check(self, T_bl, clouds):
        if not clouds:
            return
        c = [pointcloud2_xyz_ring(deserialize_message(m, PointCloud2)) for m in clouds[-3:]]
        Pl, ring = np.vstack([x[0] for x in c]), np.concatenate([x[1] for x in c])
        checks, info = [], {}
        vals, ground = board_filter(Pl, ring, T_bl, self.a.box, lambda ok, msg: checks.append((bool(ok), msg)),
                                    self.dims, say=lambda *_: None, info=info)
        if "B" in info:
            bx0 = info["B"][:, 0].min()
            holes_ok = info["hole_err"] is not None and info["hole_err"] < 0.15
            # only a hint for why holes are missing: with all 4 holes found a cut edge does not matter
            if not holes_ok and bx0 < T_bl[0, 3] + 0.25 and info["width"] < 1.3:  # the board is 1.4 m wide
                checks.insert(0, (False, "LiDAR sees only the FRONT PART of the board: move the board forward, "
                                         "its back edge ahead of the red line on the map"))
        top = self.top_view(Pl, T_bl, ground, info.get("B"))
        img = np.full((300, 420, 3), 30, np.uint8)
        if "B" in info:  # the board plane seen face-on, holes circled
            B, n = info["B"], info["n"]
            u = np.cross(n, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(n, u)
            o = B.mean(0)
            px = lambda p: (int(210 + (p - o) @ u * 200), int(150 - (p - o) @ v * 200))  # noqa: E731
            for p in B[::2]:
                cv2.circle(img, px(p), 1, (200, 200, 200), -1)
            if info.get("pole") is not None:   # the LiDAR's spin axis: a blind spot, no hole should sit on it
                q = px(info["pole"])
                cv2.drawMarker(img, q, (255, 120, 0), cv2.MARKER_CROSS, 22, 2)
                cv2.putText(img, "LiDAR centre", (q[0] + 8, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 120, 0), 1)
            good = info["hole_err"] is not None and info["hole_err"] < 0.15
            for i, h in enumerate(info["holes"]):
                inferred = info["holes_inferred"] and i == len(info["holes"]) - 1  # the completed one is last
                col = (0, 200, 255) if inferred else (0, 220, 0) if good else (0, 0, 255)
                cv2.circle(img, px(h), int(self.dims[2] * 200), col, 1 if inferred else 2)
            cv2.putText(img, f"{info['dist']:.1f} m, {info['rings']} rings, {len(info['holes']) - info['holes_inferred']} holes seen"
                        + (" + 1 inferred" if info["holes_inferred"] else ""),
                        (8, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        else:
            cv2.putText(img, "no board in the LiDAR", (8, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        with self.lock:
            self.lidar = {"checks": checks, "vals": vals if "x_min" in vals else None, "ground": ground, "t": time.monotonic()}
            self.lidar_jpg = cv2.imencode(".jpg", np.vstack([top, img]))[1].tobytes()

    def top_view(self, Pl, T_bl, ground, board):
        """Top-down map, forward up: the robot, the front LiDAR's points, its blind line, the camera's
        view, the zone the board goes in, and the board found (yellow)."""
        W, Hh, S = 420, 420, 70.0            # px, px per metre: 3 m left .. 3 m right, 1.3 m back .. 4.6 m ahead
        ox, oy = 210, 320
        px = lambda x, y: (int(ox - y * S), int(oy - x * S))  # noqa: E731
        img = np.full((Hh, W, 3), 245, np.uint8)
        Pb = Pl @ T_bl[:3, :3].T + T_bl[:3, 3]
        Pb = Pb[(Pb[:, 2] > ground + 0.15) & (Pb[:, 2] < ground + 2.0)]
        for x, y, _ in Pb[::4]:
            p = px(x, y)
            if 0 <= p[0] < W and 0 <= p[1] < Hh:
                img[p[1], p[0]] = (90, 90, 90)
        if self.scene in SCENES:   # this scene's zone, the camera taken at base_link: forward ~ distance, left = -side
            sc = SCENES[self.scene]
            (s0, s1), (d0, d1) = sc["side"], sc["dist"]
            x0, x1, y0, y1 = d0, d1, -s1, -s0
            cv2.rectangle(img, px(x1, y1), px(x0, y0), (0, 190, 0), 2)
            cv2.putText(img, "board here", (px(x1, y1)[0] + 4, px(x1, y1)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 0), 1)
            bx = self.target(self.scene, 0.0)[0][:, 0]   # the board's x axis in the camera frame, as on the picture
            d = np.array([bx[2], -bx[0]]) / np.hypot(bx[2], bx[0])   # in base_link: camera z = forward, camera x = right
            cx, cy, hw = (x0 + x1) / 2, (y0 + y1) / 2, BOARD_HALF[0]
            cv2.line(img, px(cx - hw * d[0], cy - hw * d[1]), px(cx + hw * d[0], cy + hw * d[1]), (0, 150, 0), 3)
        bl = T_bl[0, 3]
        cv2.line(img, px(bl, 2), px(bl, -6), (0, 0, 255), 1)
        cv2.putText(img, "LiDAR sees only above", (px(0, -0.5)[0], px(bl, 0)[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        for d, col in ((0, (60, 60, 60)), (-self.half_fov, (170, 170, 170)), (self.half_fov, (170, 170, 170))):
            r = np.radians(-self.a.yaw + d)
            cv2.line(img, px(0, 0), px(6 * np.cos(r), 6 * np.sin(r)), col, 1)
        cv2.rectangle(img, px(0.45, 0.2), px(-0.45, -0.2), (200, 120, 0), -1)   # the robot, ~0.9 x 0.4 m
        cv2.arrowedLine(img, px(0.2, 0), px(0.6, 0), (255, 255, 255), 2, tipLength=0.4)
        if board is not None:
            for x, y, _ in board[::2]:
                cv2.circle(img, px(x, y), 2, (0, 200, 255), -1)
        cv2.putText(img, "top view, forward = up, 1 m = 70 px", (4, Hh - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        return img

    # ---- saving a recorded scene in the layout m20_calib_run.py reads
    def save_scene(self, k, frame, raw, T_bl, src, att, zoom):
        out = os.path.join(self.a.session, f"scene{k}")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(os.path.join(out, "out"))
        cv2.imwrite(os.path.join(out, "image.png"), frame)
        write_bag(os.path.join(out, "bag"), self.a.topic, raw)
        c = [pointcloud2_xyz_ring(deserialize_message(m, PointCloud2)) for _, m in raw[-5:]]
        checks = []
        vals, ground = board_filter(np.vstack([x[0] for x in c]), np.concatenate([x[1] for x in c]), T_bl, self.a.box,
                                    lambda ok, msg: checks.append((bool(ok), msg)), self.dims, say=lambda *_: None)
        vals.update(bag_path=os.path.abspath(os.path.join(out, "bag")), image_path=os.path.abspath(os.path.join(out, "image.png")),
                    output_path=os.path.abspath(os.path.join(out, "out")), lidar_topic=self.a.topic, plane_dist_threshold=0.02,
                    lidar_type="pattern")
        open(os.path.join(out, "params.yaml"), "w").write(set_yaml_values(self.ptext, vals))
        gimbal = [float(np.median(att[:, 0])), float(np.median(att[:, 1])), 0.0] if att is not None else None
        json.dump(dict(time=time.strftime("%Y-%m-%d %H:%M:%S"), gimbal_attitude_median=gimbal,
                       gimbal_convention="lumen: yaw + right, pitch + up, zoom x", lumen_gimbal_zero=lumen_gimbal_zero(),
                       zoom_level=zoom, base_T_lidar=T_bl.tolist(), base_T_lidar_source=src, ground_z_base=float(ground),
                       board_angle=self.angle, board_dist=self.dist, clouds=len(raw),
                       problems=[m for ok, m in checks if not ok]), open(os.path.join(out, "meta.json"), "w"), indent=1)
        return [m for ok, m in checks if not ok]

    def show(self, vis):
        """Publish the picture for the page's video stream."""
        h, w = vis.shape[:2]
        small = cv2.resize(vis, (VIDEO_W, int(h * VIDEO_W / w)), interpolation=cv2.INTER_AREA)
        jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
        with self.frame_cv:
            self.frame_jpg, self.frame_seq = jpg, self.frame_seq + 1
            self.frame_cv.notify_all()

    def run_calibration(self):
        pr = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "m20_calib_run.py"), self.a.session],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in pr.stdout:
            if "rosbag2_storage" not in line:
                with self.lock:
                    self.log.append(line.rstrip())
        pr.wait()
        self.state = "done"


def page():
    return """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>M20 calibration</title><style>
body{margin:0;background:#111;color:#eee;font:15px system-ui,sans-serif}
.wrap{max-width:1000px;margin:auto;padding:8px}
img{width:100%;display:block;border-radius:6px}
h1{font-size:20px;margin:8px 0} .msg{font-size:17px;background:#223;padding:10px;border-radius:6px;margin:8px 0}
.row{display:flex;gap:8px;flex-wrap:wrap} .row>div{flex:1;min-width:280px}
ul{list-style:none;padding:0;margin:0} li{padding:3px 0}
.ok{color:#5d5}.no{color:#f66} .bar{height:10px;background:#333;border-radius:5px;overflow:hidden;margin:6px 0}
.bar>div{height:100%;background:#5d5} button{font-size:15px;padding:8px 12px;margin-top:8px}
pre{white-space:pre-wrap;font-size:12px;background:#000;padding:8px;border-radius:6px;max-height:400px;overflow:auto}
</style></head><body><div class=wrap>
<h1 id=title></h1><div class=msg id=msg></div><div class=bar><div id=bar style="width:0"></div></div>
<div class=row><div><img id=cam src="/video.mjpg"></div><div><img id=lid><ul id=checks></ul>
<button onclick="fetch('/redo',{method:'POST'})">Redo the last scene</button></div></div>
<div id=result></div><pre id=log></pre></div><script>
async function tick(){
 const s=await (await fetch('/status')).json();
 title.textContent=s.title; msg.textContent=s.message; bar.style.width=(100*s.progress)+'%';
 checks.innerHTML=s.checks.map(c=>`<li class=${c[0]?'ok':'no'}>${c[0]?'✓':'✗'} ${c[1]}</li>`).join('');
 log.textContent=s.log.join('\\n'); log.style.display=s.log.length?'block':'none';
 result.innerHTML=s.images.map(n=>`<p>${n}</p><img src="/file/${n}?${Date.now()}">`).join('');
 lid.src='/lidar.jpg?'+Date.now();
}
async function loop(){ try{ await tick(); }catch(e){} setTimeout(loop,700); }
loop();
// reconnect the video if the stream drops
cam.onerror=()=>setTimeout(()=>{cam.src='/video.mjpg?'+Date.now()},1000);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--yaw", type=float, default=0.0, help="gimbal yaw, + = right (set it with m20_ptz_set.py)")
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--session", default=os.path.join(REPO, "calib_data", "m20_" + time.strftime("%Y%m%d")))
    ap.add_argument("--params", default=QR_PARAMS)
    ap.add_argument("--topic", default=LIDAR_TOPIC)
    ap.add_argument("--lumen", default=LUMEN_URL, help="Lumen, which drives the PTZ head")
    ap.add_argument("--url", default=RTSP_URL)
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--port", type=int, default=8098)
    ap.add_argument("--box", type=float, nargs=6, default=[1.2, 4.0, -1.6, 1.6, 0.05, 2.5],
                    help="where to look for the board, base_link metres x0 x1 y0 y1 z0 z1 (z above the ground)")
    a = ap.parse_args()
    os.makedirs(a.session, exist_ok=True)
    g = Guide(a)
    if abs(g.K[0, 0] - 2478.487) < 1e-2:
        sys.exit("qr_params.yaml still has the sample intrinsics: run m20_intrinsics_calibrate.py --write first")

    rclpy.init()
    # rclpy takes over SIGTERM to shut ROS down without ending the program: end it too
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    node = Sensors(a.topic)
    gimbal = LumenGimbal(a.lumen)
    T_bl, src = lookup_base_lidar(node)
    def spin():
        try:
            rclpy.spin(node)
        except Exception:  # noqa: BLE001 - shutdown
            pass
    threading.Thread(target=spin, daemon=True).start()
    st = Stream(a.url)

    class H(BaseHTTPRequestHandler):
        def _send(self, body, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def video(self):
            """multipart/x-mixed-replace: the browser shows it as live video, no polling."""
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            seq = -1
            try:
                while True:
                    with g.frame_cv:
                        g.frame_cv.wait_for(lambda: g.frame_seq != seq, timeout=5.0)
                        jpg, seq = g.frame_jpg, g.frame_seq
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                         + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    time.sleep(1.0 / VIDEO_FPS)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_GET(self):
            if self.path.startswith("/video.mjpg"):
                return self.video()
            if self.path.startswith("/frame.jpg"):
                return self._send(g.frame_jpg, "image/jpeg")
            if self.path.startswith("/lidar.jpg"):
                return self._send(g.lidar_jpg, "image/jpeg")
            if self.path.startswith("/file/"):
                name = os.path.basename(self.path[6:].split("?")[0])
                p = os.path.join(a.session, "out", name)
                if os.path.exists(p):
                    return self._send(open(p, "rb").read(), "image/jpeg")
                self.send_response(404); self.end_headers(); return
            if self.path.startswith("/status"):
                with g.lock:
                    s = dict(title=g.title, message=g.message, checks=[(bool(o), str(m)) for o, m in g.checks], progress=float(g.progress),
                             log=g.log[-60:], images=g.images)
                return self._send(json.dumps(s).encode(), "application/json")
            return self._send(page().encode(), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path == "/redo" and g.state in ("waiting", "done"):
                k = g.next_scene() - 1
                if k >= 1:
                    shutil.rmtree(os.path.join(a.session, f"scene{k}"), ignore_errors=True)
                    g.scene, g.state, g.log, g.images = k, "waiting", [], []
            self.send_response(204); self.end_headers()

        def log_message(self, *x):
            pass

    g.title, g.progress, g.images = "", 0.0, []
    threading.Thread(target=ThreadingHTTPServer(("0.0.0.0", a.port), H).serve_forever, daemon=True).start()
    print(f"open http://<robot-ip>:{a.port}   (session {a.session})")

    def lidar_loop():  # off the video loop: a check takes a few hundred ms
        while True:
            if g.state == "waiting":
                try:
                    g.lidar_check(T_bl, list(node.clouds))
                except Exception as e:  # noqa: BLE001 - show it, keep going
                    with g.lock:
                        g.lidar = {"checks": [(False, f"LiDAR check error: {e}")], "vals": None, "ground": None, "t": time.monotonic()}
            time.sleep(0.3)
    threading.Thread(target=lidar_loop, daemon=True).start()
    if g.scene == N_SCENES + 1:  # every scene already recorded: calibrate
        g.state = "calibrating"
        threading.Thread(target=g.run_calibration, daemon=True).start()

    last, rec_t0, rec_best = 0, None, None
    try:
        while rclpy.ok():
            frame, last = st.read(after=last)
            vis, gray, seen = g.camera_check(frame)
            if g.state in ("calibrating", "done"):
                with g.lock:
                    g.title = "Calibration" + (" running..." if g.state == "calibrating" else " finished")
                    g.message = ("Solving, about a minute." if g.state == "calibrating" else
                                 "Check: each scene RMSE < 1-2 cm, red crosses on the holes, the overlays line up. "
                                 "Result: out/camera_extrinsics.yaml")
                    g.checks, g.progress = [], 1.0
                    od = os.path.join(a.session, "out")
                    if g.state == "done" and os.path.isdir(od):
                        g.images = sorted(f for f in os.listdir(od) if f.startswith("overlay_"))
                g.show(vis)
                continue

            k = g.scene
            sc = SCENES[k]
            (lo, hi), want = sc["angle"], sc["text"]
            gyro = node.gyro_max(2.0)
            gim = gimbal.gimbal(3.0)
            checks = [
                (gyro is not None and gyro < 0.05, "robot still" if gyro is None or gyro < 0.05 else f"robot moving ({np.degrees(gyro):.1f} deg/s)"),
                (gim is not None and np.ptp(gim[:, 0]) < 0.5 and abs(np.median(gim[:, 0]) - a.yaw) < 2.5 and abs(np.median(gim[:, 1]) - a.pitch) < 2.5,
                 f"no gimbal status from Lumen ({gimbal.error})" if gim is None else f"gimbal yaw {np.median(gim[:, 0]):.1f} pitch {np.median(gim[:, 1]):.1f} (want {a.yaw:.0f}/{a.pitch:.0f}, steady)"),
                (seen, "all 4 markers in the picture" if seen else "camera: needs all 4 markers in view"),
            ]
            if seen:
                checks.append((g.holes_in_view, "camera: all 4 holes inside the picture (green circles on the holes)" if g.holes_in_view
                               else "camera: a hole is cut by the picture's edge (red circles): move the board in"))
                checks.append((lo <= g.angle <= hi, f"board angle {g.angle:+.0f} deg (want {lo:+d} to {hi:+d})"))
                d0, d1 = sc["dist"]
                s0, s1 = sc["side"]
                checks.append((d0 <= g.dist <= d1, f"board {g.dist:.1f} m from the camera (want {d0}-{d1})"))
                checks.append((s0 <= g.side <= s1, f"board {abs(g.side):.2f} m {'right' if g.side >= 0 else 'left'} of the picture's centre "
                                                   f"(want {'left ' if s1 <= 0 else 'right ' if s0 >= 0 else ''}{s0:+.2f} to {s1:+.2f})"))
                checks.append((g.board_still(), "board held still" if g.board_still() else "hold the board still"))
            with g.lock:
                checks += g.lidar["checks"] or [(False, "LiDAR: waiting for the board")]
            all_ok = all(c[0] for c in checks)

            if g.state == "waiting":
                g.hold_since = (g.hold_since or time.monotonic()) if all_ok else None
                held = 0 if g.hold_since is None else time.monotonic() - g.hold_since
                g.progress = min(held / HOLD_S, 1.0)
                hint = g.hints(k)
                g.message = (f"{want}: " + "; ".join(hint) + " (yellow outline on the picture)." if hint and not all_ok else
                             f"{want}. Put it on the yellow outline in the picture (the green zone on the map), all 4 markers "
                             "in view, nothing touching the markers, 1 m clear behind it. Keep the LiDAR's centre (blue + on the board "
                             "view) between the 4 holes." if not all_ok else
                             f"All good: hold still, recording in {max(HOLD_S - held, 0):.0f} s")
                if held >= HOLD_S:
                    g.state, rec_t0, rec_best = "recording", time.monotonic(), (-1, None)
                    node.rec = []
            elif g.state == "recording":
                el = time.monotonic() - rec_t0
                g.progress = min(el / a.duration, 1.0)
                g.message = f"RECORDING scene {k}: do not move ({a.duration - el:.0f} s)"
                sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                if sharp > rec_best[0]:
                    rec_best = (sharp, frame.copy())
                moving = (node.gyro_max(1.0) or 0) > 0.05 or not seen or not g.board_still(1.0)
                if moving:
                    node.rec, g.state, g.hold_since = None, "waiting", None
                    g.message = "Something moved while recording: starting again"
                elif el >= a.duration:
                    raw, node.rec = node.rec, None
                    gim = gimbal.gimbal(a.duration)
                    problems = g.save_scene(k, rec_best[1], raw, T_bl, src, gim, gimbal.zoom)
                    print(f"scene{k} saved" + (f" with problems: {problems}" if problems else ""))
                    if k < N_SCENES:  # the last scene is checked by the full calibration that follows
                        threading.Thread(target=g.check_scene, args=(k,), daemon=True).start()
                    g.scene, g.state, g.hold_since = g.next_scene(), "waiting", None
                    if g.scene == N_SCENES + 1:
                        g.state = "calibrating"
                        threading.Thread(target=g.run_calibration, daemon=True).start()
            with g.lock:
                g.checks = [(bool(ok), t) for ok, t in g.results.values() if ok is not None and os.path.isdir(
                    os.path.join(a.session, t.split(":")[0].replace(" ", "")))] + checks
                g.title = f"Scene {k} of {N_SCENES}" if k <= N_SCENES else "Calibration"
            if seen:
                cv2.putText(vis, f"angle {g.angle:+.0f} deg  {g.dist:.1f} m", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            if g.state == "recording":
                cv2.circle(vis, (vis.shape[1] - 30, 30), 14, (0, 0, 255), -1)
            g.show(vis)
    except KeyboardInterrupt:
        pass
    finally:
        st.close()
        gimbal.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
