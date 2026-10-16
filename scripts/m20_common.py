"""Shared pieces of the M20 calibration scripts: the SIYI stream, the LiDAR mount, YAML."""
import os
import re
import threading
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay")
import cv2
import numpy as np

RTSP_URL = "rtsp://10.21.31.90:8554/main.264"
STREAM_SIZE = (1920, 1080)
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
