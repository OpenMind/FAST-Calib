"""The M20 PTZ calibration as functions: camera pose and intrinsics at any gimbal angle and zoom.

Reference implementation of the model in README.md, reading ptz_calibration.yaml and stereo_calib.yaml
from this folder. Needs numpy, pyyaml and (for the projections) opencv.

    from ptz_model import PTZ
    ptz = PTZ()                                             # or PTZ("/path/to/this/folder")
    uv, depth = ptz.lidar_to_pixel(P_lidar, yaw, pitch, zoom)   # LiDAR points (N x 3, lidar_front frame) -> picture
    d = ptz.depth_at_pixel(P_lidar, u, v, yaw, pitch, zoom)     # how far the thing at pixel (u, v) is, metres
    o, ray = ptz.pixel_ray(u, v, yaw, pitch, zoom)              # the pixel's ray in base_link
    X_thermal = ptz.optical_to_thermal(X_cam, zoom)             # camera frame (m) -> thermal camera frame (m)

Gimbal angles are Lumen's (GET /v1/ptz/status): yaw + = right, pitch + = up, zoom in x.
"""
import os

import numpy as np
import yaml


def _Rz(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])


def _Ry(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


class PTZ:
    def __init__(self, folder=None):
        folder = folder or os.path.dirname(os.path.abspath(__file__))
        c = yaml.safe_load(open(os.path.join(folder, "ptz_calibration.yaml")))
        self.stereo = yaml.safe_load(open(os.path.join(folder, "stereo_calib.yaml")))
        self.R_mount = np.array(c["gimbal"]["R_mount"], float)
        self.pivot = np.array(c["gimbal"]["pivot"], float)
        self.lens_ahead = float(c["gimbal"]["lens_ahead"])
        self.T_base_lidar = np.array(c["lidar"]["T_base_lidar"], float)
        self.size = tuple(c["optical_1x"]["size"])

    # ---- pose
    def camera_in_base(self, yaw, pitch):
        """4x4 base_link <- camera (optical frame: x right, y down, z forward) at a gimbal angle."""
        T = np.eye(4)
        T[:3, :3] = _Rz(-yaw) @ _Ry(-pitch) @ self.R_mount
        T[:3, 3] = self.pivot + self.lens_ahead * T[:3, 2]
        return T

    def camera_from_lidar(self, yaw, pitch):
        """4x4 camera <- lidar_front at a gimbal angle: p_cam = T @ p_lidar."""
        return np.linalg.inv(self.camera_in_base(yaw, pitch)) @ self.T_base_lidar

    # ---- intrinsics
    def magnification(self, zoom):
        """The picture's scale at a zoom, relative to 1x (not equal to the zoom number)."""
        zc = self.stereo["zoom_curve"]
        return float(np.interp(zoom, zc["zoom"], zc["scale"]))

    def intrinsics(self, zoom=1.0):
        """(K 3x3, dist k1 k2 p1 p2 k3) for the 2688x1520 picture at a zoom.

        At a calibrated zoom (1, 1.5, 2, 3) that zoom's own fit; elsewhere the nearest one magnified
        along the zoom curve about the zoom centre. Measured up to 6x; beyond, the curve is extrapolated."""
        opt = self.stereo["optical"]
        level = min(opt, key=lambda z: abs(float(z) - zoom))
        K = np.array(opt[level]["K"], float)
        D = np.array((list(opt[level]["dist"]) + [0.0] * 5)[:5], float)
        if abs(float(level) - zoom) > 0.01:
            m = self.magnification(zoom) / self.magnification(float(level))
            cx, cy = self.stereo["zoom_curve"]["centre"]
            K[0, 0] *= m
            K[1, 1] *= m
            K[0, 2] = cx + (K[0, 2] - cx) * m
            K[1, 2] = cy + (K[1, 2] - cy) * m
        return K, D

    def hfov(self, zoom=1.0):
        """Horizontal field of view in degrees at a zoom."""
        return float(np.degrees(2 * np.arctan(self.size[0] / 2 / self.intrinsics(zoom)[0][0, 0])))

    # ---- LiDAR <-> picture
    def lidar_to_pixel(self, P_lidar, yaw, pitch, zoom=1.0):
        """LiDAR points (N x 3, lidar_front frame) -> (pixels N x 2, depth N); depth <= 0 is behind the camera."""
        import cv2
        T = self.camera_from_lidar(yaw, pitch)
        Pc = np.asarray(P_lidar, float) @ T[:3, :3].T + T[:3, 3]
        K, D = self.intrinsics(zoom)
        front = Pc[:, 2] > 0.05
        uv = np.full((len(Pc), 2), np.nan)
        if front.any():
            uv[front] = cv2.projectPoints(Pc[front], np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
        return uv, Pc[:, 2]

    def depth_at_pixel(self, P_lidar, u, v, yaw, pitch, zoom=1.0, radius_px=None):
        """Distance along the optical axis (m) of what is at pixel (u, v): the median of the LiDAR points that
        land within radius_px of it (default: 1.5 deg of view), or None when none do. The front LiDAR's scan
        lines are ~5 cm apart at 2.5 m, so take a radius that covers the object."""
        uv, z = self.lidar_to_pixel(P_lidar, yaw, pitch, zoom)
        if radius_px is None:
            radius_px = np.tan(np.radians(1.5)) * self.intrinsics(zoom)[0][0, 0]
        near = (np.hypot(uv[:, 0] - u, uv[:, 1] - v) < radius_px) & (z > 0.05)
        return float(np.median(z[near])) if near.any() else None

    def pixel_ray(self, u, v, yaw, pitch, zoom=1.0):
        """(origin, unit direction) in base_link of the ray through pixel (u, v)."""
        import cv2
        K, D = self.intrinsics(zoom)
        x, y = cv2.undistortPoints(np.array([[[u, v]]], float), K, D).reshape(2)
        T = self.camera_in_base(yaw, pitch)
        d = T[:3, :3] @ np.array([x, y, 1.0])
        return T[:3, 3], d / np.linalg.norm(d)

    def pixel_to_base(self, u, v, depth, yaw, pitch, zoom=1.0):
        """The point in base_link at pixel (u, v) and a depth along the optical axis (from depth_at_pixel)."""
        import cv2
        K, D = self.intrinsics(zoom)
        x, y = cv2.undistortPoints(np.array([[[u, v]]], float), K, D).reshape(2)
        T = self.camera_in_base(yaw, pitch)
        return T[:3, :3] @ (np.array([x, y, 1.0]) * depth) + T[:3, 3]

    # ---- optical <-> thermal (the optical-thermal calibration; both lenses turn with the head)
    def optical_to_thermal(self, X_cam, zoom=1.0):
        """Points in the optical camera frame (m) -> the thermal camera frame (m): X_t = R X_o + T(zoom).
        T is calibrated at 1x, 1.5x, 2x and 3x (the zoom lens' entrance pupil moves) and held beyond 3x."""
        R = np.array(self.stereo["R"], float)
        levels = sorted(self.stereo["T_mm"], key=float)
        zs = [float(z) for z in levels]
        T = np.array([np.interp(zoom, zs, [self.stereo["T_mm"][z][i] for z in levels]) for i in range(3)]) / 1000.0
        return np.asarray(X_cam, float) @ R.T + T

    def thermal_intrinsics(self):
        """(K, dist) of the 640x512 thermal matrix."""
        t = self.stereo["thermal"]
        return np.array(t["K"], float), np.array((list(t["dist"]) + [0.0] * 5)[:5], float)
