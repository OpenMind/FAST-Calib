"""Shared pieces of the M20 calibration scripts: the PTZ stream and gimbal, the LiDAR mount, YAML.

The PTZ camera is the HIKMICRO HM-TD5767T-7 dual-spectrum head. Its optical stream comes through the
robot's relay (om1_video_processor), and its gimbal is driven through Lumen's HTTP API, which owns the
camera's ISAPI connection and credentials: these scripts never talk to the camera directly.
"""
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from collections import deque

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")
import cv2
import numpy as np

RTSP_URL = "rtsp://127.0.0.1:8554/ptz_camera"   # the relay of the HIKMICRO optical main stream
STREAM_SIZE = (2688, 1520)
LUMEN_URL = "http://127.0.0.1:8999"              # Lumen: GET /v1/ptz/status, POST /v1/ptz/angle, /v1/ptz/zoom
HFOV_1X = 54.5  # deg, the optical lens at zoom 1x (fx 2608 px on 2688 px, the optical-thermal calibration)
LIDAR_TOPIC = "/lidar_points_front"
LIDAR_FRAME = "lidar_front"
# base_link <- lidar_front from the M20's TF (used when /tf_static cannot be read)
LIDAR_T_DEFAULT = (np.array([0.320, 0.0, -0.013]), np.array([0.707107, 0.0, 0.707107, 0.0]))  # xyz, quat xyzw
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
QR_PARAMS = os.path.join(REPO, "config", "qr_params.yaml")


class Stream:
    """Latest frame of the RTSP stream, read on a thread so frames never queue up."""

    def __init__(self, url=RTSP_URL, warmup_frames=60):
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {url}")
        self.frame, self.count, self.stamp = None, 0, 0.0
        self.warmup = warmup_frames  # HEVC frames before the first keyframe are grey garbage
        self.lock = threading.Lock()
        self.alive = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.alive:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame, self.count, self.stamp = f, self.count + 1, time.monotonic()
        # released here, by the thread that reads: releasing it from another thread while
        # read() is decoding aborts FFmpeg ("Assertion fctx->async_lock failed")
        self.cap.release()

    def read(self, after=0, timeout=5.0):
        """Return (frame, count) newer than frame number `after`, past the warm-up."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                if self.frame is not None and self.count > max(after, self.warmup):
                    return self.frame.copy(), self.count
            time.sleep(0.005)
        raise RuntimeError("no frame from the stream")

    def close(self):
        self.alive = False
        self.thread.join(timeout=3.0)


# ---- the CRYSOUND acoustic camera: a fixed camera, its clean picture from the REST API ----
ACOUSTIC_URL = os.environ.get("ACOUSTIC_CAM_URL")   # http://<camera-ip>:9999, the camera's REST API
ACOUSTIC_SIZE = (1920, 1080)
ACOUSTIC_PARAMS = os.path.join(REPO, "config", "acoustic_params.yaml")


class AcousticStream:
    """Latest clean frame of the CRYSOUND acoustic camera (GET /api/image/v1/imrData, cameraImageData).

    Not the RTSP video: that one carries the sound cloud, dB label and focus box drawn into the picture.
    About 3 frames a second. Address from ACOUSTIC_CAM_URL (http://<camera-ip>:9999), credentials from
    ACOUSTIC_CAM_USER (default admin) and ACOUSTIC_CAM_PASSWORD."""

    def __init__(self, url=ACOUSTIC_URL, user=None, password=None):
        if not url:
            raise RuntimeError("set ACOUSTIC_CAM_URL=http://<camera-ip>:9999 (or pass --url)")
        self.url = url.rstrip("/")
        self.user = user or os.environ.get("ACOUSTIC_CAM_USER", "admin")
        self.password = password or os.environ.get("ACOUSTIC_CAM_PASSWORD")
        if not self.password:
            raise RuntimeError("set ACOUSTIC_CAM_PASSWORD (the acoustic camera's web password)")
        self.cookie = None
        self.frame, self.count, self.error = None, 0, None
        self.lock = threading.Lock()
        self.alive = True
        self._login()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _login(self):
        import urllib.parse
        body = urllib.parse.urlencode({"user": self.user, "password": self.password}).encode()
        with urllib.request.urlopen(urllib.request.Request(self.url + "/api/register/v1/login", data=body), timeout=5) as r:
            j = json.loads(r.read())
        if j.get("result") != 0:
            raise RuntimeError(f"acoustic camera login failed: {j.get('info')}")
        self.cookie = j["data"]["cookie"].split(";")[0]

    def _grab(self):
        import base64
        req = urllib.request.Request(self.url + "/api/image/v1/imrData", method="GET",
                                     data=json.dumps({"cameraImage": True, "compositeImage": False}).encode(),
                                     headers={"Content-Type": "application/json", "CRY-Cookie": self.cookie})
        with urllib.request.urlopen(req, timeout=10) as r:
            j = json.loads(r.read())
        b64 = j["data"]["pictureData"]["cameraImageData"].split(",", 1)[-1]
        return cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)

    def _run(self):
        while self.alive:
            try:
                f = self._grab()
                with self.lock:
                    self.frame, self.count, self.error = f, self.count + 1, None
            except Exception as e:  # noqa: BLE001 - session expired (401) or a dropped request: log in again
                self.error = str(e)
                time.sleep(0.5)
                try:
                    self._login()
                except Exception:  # noqa: BLE001
                    time.sleep(1.0)

    def read(self, after=0, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self.lock:
                if self.frame is not None and self.count > after:
                    return self.frame.copy(), self.count
            time.sleep(0.01)
        raise RuntimeError(f"no frame from the acoustic camera ({self.error})")

    def close(self):
        self.alive = False


class FixedMount:
    """Stands in for the gimbal of a camera without one: always at yaw 0, pitch 0, zoom 1."""
    zoom, error = 1.0, None

    def gimbal(self, s):
        return np.zeros((1, 2))

    def close(self):
        pass


CAMERAS = {  # --camera: stream, picture size, default params file
    "ptz": dict(size=(2688, 1520), params=QR_PARAMS, url=RTSP_URL),
    "acoustic": dict(size=ACOUSTIC_SIZE, params=ACOUSTIC_PARAMS, url=ACOUSTIC_URL),
}


def open_camera(camera, url=None):
    url = url or CAMERAS[camera]["url"]
    return AcousticStream(url) if camera == "acoustic" else Stream(url)


class LumenGimbal:
    """The PTZ head through Lumen. Yaw 0 = robot front, + = right; pitch 0 = level, + = up; zoom in x.

    A thread polls the status a few times a second, so `gimbal(s)` gives the recent history like the
    ROS attitude topic of the SIYI driver did."""

    def __init__(self, url=LUMEN_URL, poll_s=0.2):
        self.url = url.rstrip("/")
        self.att = deque(maxlen=300)      # (monotonic time, yaw, pitch)
        self.zoom = None
        self.error = None
        self.alive = True
        self.poll_s = poll_s
        threading.Thread(target=self._run, daemon=True).start()

    def _call(self, path, body=None, timeout=5.0):
        req = urllib.request.Request(self.url + path, method="POST" if body is not None else "GET",
                                     data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read() or b"{}")

    def status(self):
        """{yaw, pitch, roll, zoom} now."""
        return self._call("/v1/ptz/status")

    def _run(self):
        while self.alive:
            try:
                st = self.status()
                self.att.append((time.monotonic(), float(st["yaw"]), float(st["pitch"])))
                self.zoom, self.error = float(st["zoom"]), None
            except Exception as e:  # noqa: BLE001 - shown by the callers' checks
                self.error = str(e)
            time.sleep(self.poll_s)

    def gimbal(self, s):
        """(yaw, pitch) rows of the last s seconds, or None."""
        now = time.monotonic()
        a = np.array([(y, p) for t, y, p in list(self.att) if t > now - s])
        return a if len(a) else None

    def set(self, yaw, pitch, zoom=1.0):
        self._call("/v1/ptz/zoom", {"level": float(zoom)})
        return self._call("/v1/ptz/angle", {"yaw": float(yaw), "pitch": float(pitch)})

    def close(self):
        self.alive = False


def lumen_gimbal_zero():
    """Lumen's current LUMEN_GIMBAL_FORWARD_AZIMUTH / LEVEL_ELEVATION (the camera azimuth and elevation
    it calls yaw 0 / pitch 0), from the container's environment; 0 when unset or unreadable.
    Only these two variables are read: the container's environment also holds credentials."""
    zero = {"forward_azimuth": 0.0, "level_elevation": 0.0, "source": "default"}
    try:
        env = subprocess.run(["docker", "inspect", "lumen", "--format", "{{range .Config.Env}}{{println .}}{{end}}"],
                             capture_output=True, text=True, timeout=5).stdout.splitlines()
    except Exception:  # noqa: BLE001
        return zero
    for line in env:
        for key in ("FORWARD_AZIMUTH", "LEVEL_ELEVATION"):
            if line.startswith(f"LUMEN_GIMBAL_{key}="):
                try:
                    zero[key.lower()] = float(line.split("=", 1)[1])
                    zero["source"] = "lumen container"
                except ValueError:
                    pass
    if env and zero["source"] == "default":
        zero["source"] = "lumen container (unset: 0)"
    return zero


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def T_from(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def lookup_base_lidar(node=None, timeout=3.0):
    """4x4 base_link <- lidar_front, from /tf_static when a node is given, else the default."""
    if node is not None:
        try:
            import rclpy
            import tf2_ros
            buf = tf2_ros.Buffer()
            tf2_ros.TransformListener(buf, node, spin_thread=False)
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                rclpy.spin_once(node, timeout_sec=0.1)
                if buf.can_transform("base_link", LIDAR_FRAME, rclpy.time.Time()):
                    tr = buf.lookup_transform("base_link", LIDAR_FRAME, rclpy.time.Time()).transform
                    q = [tr.rotation.x, tr.rotation.y, tr.rotation.z, tr.rotation.w]
                    return T_from(quat_to_R(q), [tr.translation.x, tr.translation.y, tr.translation.z]), "tf"
        except Exception as e:  # noqa: BLE001 - fall back below
            print(f"  TF lookup failed ({e}); using the built-in M20 mount")
    t, q = LIDAR_T_DEFAULT
    return T_from(quat_to_R(q), t), "default"


def pointcloud2_xyz_ring(msg):
    """(N,3) float32 xyz and (N,) ring from a PointCloud2 with x, y, z, ring fields."""
    off = {f.name: f.offset for f in msg.fields}
    dt = np.dtype({"names": ["x", "y", "z", "ring"], "formats": ["<f4", "<f4", "<f4", "<u2"],
                   "offsets": [off["x"], off["y"], off["z"], off["ring"]], "itemsize": msg.point_step})
    a = np.frombuffer(bytes(msg.data), dt, count=msg.width * msg.height)
    P = np.c_[a["x"], a["y"], a["z"]]
    ok = np.isfinite(P).all(1) & (np.abs(P).sum(1) > 1e-3)
    return P[ok], a["ring"][ok]


def set_yaml_values(text, values):
    """Replace `key: value` lines in a FAST-Calib params YAML, keeping comments and layout.
    Keys that are missing are added under ros__parameters."""
    for k, v in values.items():
        vs = f'"{v}"' if isinstance(v, str) else repr(float(v)) if isinstance(v, float) else str(v)
        pat = re.compile(rf"^(\s*){re.escape(k)}:\s*[^#\n]*(#.*)?$", re.M)
        if pat.search(text):
            text = pat.sub(lambda m: f"{m.group(1)}{k}: {vs}" + (f" {m.group(2)}" if m.group(2) else ""), text, count=1)
        else:
            text = re.sub(r"(ros__parameters:\n)", rf"\1      {k}: {vs}\n", text, count=1)
    return text


def get_yaml_value(text, key, default=None):
    m = re.search(rf"^\s*{re.escape(key)}:\s*([^#\n]+)", text, re.M)
    if not m:
        return default
    v = m.group(1).strip().strip('"')
    try:
        return float(v)
    except ValueError:
        return v


# ---- the FAST-Calib board's four ArUco markers as a calibration target ----
BOARD_IDS = (1, 2, 4, 3)  # top-left, top-right, bottom-right, bottom-left (qr_detect.hpp)


def detect_markers(gray):
    """ArUco DICT_6X6_250 with sub-pixel corners, on OpenCV before and after 4.7."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    if hasattr(cv2.aruco, "ArucoDetector"):
        par = cv2.aruco.DetectorParameters()
        par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.ArucoDetector(d, par).detectMarkers(gray)
    else:
        par = cv2.aruco.DetectorParameters_create()
        par.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        corners, ids, _ = cv2.aruco.detectMarkers(gray, d, parameters=par)
    return corners, ([] if ids is None else ids.ravel().tolist())


def board_marker_corners(params_text):
    """{id: 4x3 corners} in the board frame (x right, y up, metres), from qr_params.yaml's
    marker_size and delta_*_qr_center, the same layout FAST-Calib's qr_detect.hpp builds."""
    m = get_yaml_value(params_text, "marker_size")
    dw = get_yaml_value(params_text, "delta_width_qr_center")
    dh = get_yaml_value(params_text, "delta_height_qr_center")
    out = {}
    for i, mid in enumerate(BOARD_IDS):
        cx = (-1 if i % 3 == 0 else 1) * dw
        cy = (1 if i < 2 else -1) * dh
        out[mid] = np.array([[cx + sx * m / 2, cy + sy * m / 2, 0] for sx, sy in ((-1, 1), (1, 1), (1, -1), (-1, -1))], np.float32)
    return out


def board_points(gray, obj_by_id):
    """(obj Nx3, img Nx2) of the board's 16 marker corners when all four markers are seen, else None."""
    corners, ids = detect_markers(gray)
    if set(ids) != set(BOARD_IDS):
        return None
    by = dict(zip(ids, corners))
    obj = np.vstack([obj_by_id[i] for i in BOARD_IDS])
    img = np.vstack([by[i].reshape(4, 2) for i in BOARD_IDS]).astype(np.float32)
    return obj, img
