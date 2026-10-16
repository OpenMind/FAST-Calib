#!/usr/bin/env python3
"""Capture calibration-target views from the SIYI stream for the camera intrinsics.

The target is the FAST-Calib board itself (its four ArUco markers, geometry from
config/qr_params.yaml -- measure the board first), or a printed checkerboard (--cols/--rows).
Camera at zoom 1x (intrinsics do not depend on the gimbal angle). Move the target slowly
over the whole picture: centre, all four corners and edges, near and far, tilted
left/right/up/down. A view is kept only when the target is still, sharp and in a new pose.
Watch the live page on a phone: http://<robot-ip>:8098

    python3 scripts/m20_intrinsics_capture.py                                   # the ArUco board
    python3 scripts/m20_intrinsics_capture.py --cols 9 --rows 6 --square 0.025  # a checkerboard

--cols/--rows are INNER corners (a 10x7-square board has 9x6). Stop with Ctrl-C, or it stops
by itself once --target views cover the picture. Then run m20_intrinsics_calibrate.py.
"""
import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20_common import (QR_PARAMS, REPO, RTSP_URL, STREAM_SIZE, Stream,  # noqa: E402
                        board_marker_corners, board_points)

GRID = (4, 3)  # coverage cells across, down


def quad_pose(tl, tr, br, bl, pts, w, h):
    """Descriptor of a view: centre, size, and the two perspective tilts of the target's outline."""
    area = cv2.contourArea(np.array([tl, tr, br, bl], np.float32))
    left, right = np.linalg.norm(bl - tl), np.linalg.norm(br - tr)
    top, bottom = np.linalg.norm(tr - tl), np.linalg.norm(br - bl)
    return np.array([pts[:, 0].mean() / w, pts[:, 1].mean() / h, np.sqrt(area) / w,
                     np.log(left / right), np.log(top / bottom)])


def coverage(saved, w, h):
    g = np.zeros(GRID[::-1], int)
    for q in saved:
        cells = set(zip(np.clip((q[:, 1] / h * GRID[1]).astype(int), 0, GRID[1] - 1),
                        np.clip((q[:, 0] / w * GRID[0]).astype(int), 0, GRID[0] - 1)))
        for cell in cells:
            g[cell] += 1
    return g


class Preview(BaseHTTPRequestHandler):
    jpeg = b""

    def do_GET(self):
        if self.path.startswith("/frame.jpg"):
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(Preview.jpeg)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(b"<html><body style='margin:0;background:#111'><img id=i style='width:100%'>"
                         b"<script>setInterval(()=>{i.src='/frame.jpg?'+Date.now()},300)</script></body></html>")

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, help="checkerboard inner corners across (omit for the ArUco board)")
    ap.add_argument("--rows", type=int, help="checkerboard inner corners down")
    ap.add_argument("--square", type=float, help="checkerboard square size, metres")
    ap.add_argument("--params", default=QR_PARAMS, help="ArUco board geometry (marker_size, delta_*_qr_center)")
    ap.add_argument("--out", default=os.path.join(REPO, "calib_data", "intrinsics"))
    ap.add_argument("--url", default=RTSP_URL)
    ap.add_argument("--target", type=int, default=45, help="views to keep")
    ap.add_argument("--novelty", type=float, default=0.08, help="min pose difference to keep a view")
    ap.add_argument("--port", type=int, default=8098, help="live preview page, 0 = off")
    a = ap.parse_args()
    chess = a.cols is not None
    if chess and not (a.rows and a.square):
        sys.exit("a checkerboard needs --cols, --rows and --square")
    os.makedirs(a.out, exist_ok=True)
    if chess:
        board = {"type": "chess", "cols": a.cols, "rows": a.rows, "square": a.square}
    else:
        obj_by_id = board_marker_corners(open(a.params).read())
        board = {"type": "aruco", "corners": {str(k): v.tolist() for k, v in obj_by_id.items()}}
        c2, c4 = obj_by_id[2].mean(0), obj_by_id[4].mean(0)
        print(f"ArUco board from {a.params}: marker {obj_by_id[1][1, 0] - obj_by_id[1][0, 0]:.3f} m, "
              f"centres {2 * c2[0]:.3f} x {c2[1] - c4[1]:.3f} m apart (check these against the tape measure)")
    board_file = os.path.join(a.out, "board.json")
    if os.path.exists(board_file) and json.load(open(board_file))["type"] != board["type"]:
        sys.exit(f"{a.out} holds views of another target: use another --out")
    json.dump(board, open(board_file, "w"))

    if a.port:
        srv = ThreadingHTTPServer(("0.0.0.0", a.port), Preview)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        print(f"live preview: http://<robot-ip>:{a.port}")

    st = Stream(a.url)
    w, h = STREAM_SIZE
    saved, poses, last, last_p = [], [], 0, None
    n_id = len([f for f in os.listdir(a.out) if f.endswith(".png")])
    if n_id:
        print(f"{n_id} views already in {a.out}: new ones are added")
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
    status = "move the target into view"
    try:
        while len(saved) < a.target or (coverage(saved, w, h) < 3).any():
            frame, last = st.read(after=last)
            h, w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = cv2.resize(frame, (w // 2, h // 2))
            pts = None
            if chess:
                found, c = cv2.findChessboardCorners(cv2.resize(gray, (w // 2, h // 2)), (a.cols, a.rows), flags)
                if found:
                    c = c * 2.0
                    cv2.cornerSubPix(gray, c, (11, 11), (-1, -1), crit)
                    pts = c.reshape(-1, 2)
                    outline = pts[0], pts[a.cols - 1], pts[-1], pts[-a.cols]
                    cv2.drawChessboardCorners(vis, (a.cols, a.rows), c / 2, True)
            else:
                r = board_points(gray, obj_by_id)
                if r is not None:
                    pts = r[1]
                    outline = pts[0], pts[5], pts[10], pts[15]  # outer corners of markers 1, 2, 4, 3
                    for p in pts:
                        cv2.circle(vis, tuple((p / 2).astype(int)), 3, (0, 255, 0), -1)
            if pts is not None:
                moved = np.inf if last_p is None or len(last_p) != len(pts) else np.abs(pts - last_p).max()
                last_p = pts
                pose = quad_pose(*outline, pts, w, h)
                x0, y0 = pts.min(0).astype(int)
                x1, y1 = pts.max(0).astype(int)
                sharp = cv2.Laplacian(gray[y0:y1, x0:x1], cv2.CV_64F).var()
                novel = min((np.linalg.norm(pose - p) for p in poses), default=np.inf)
                if moved > 2.0:
                    status = f"hold still ({moved:.0f} px)"
                elif sharp < 40:
                    status = f"blurry ({sharp:.0f})"
                elif novel < a.novelty:
                    status = "same pose as a saved view: move or tilt it"
                else:
                    name = f"{a.out}/view_{n_id:03d}.png"
                    cv2.imwrite(name, frame)
                    saved.append(pts); poses.append(pose); n_id += 1
                    status = f"SAVED {len(saved)}"
                    print(f"saved {name}  centre ({pose[0]:.2f},{pose[1]:.2f}) size {pose[2]:.2f} "
                          f"tilt ({pose[3]:+.2f},{pose[4]:+.2f})")
            else:
                last_p = None
                status = "no target" if chess else "needs all 4 markers in view"
            g = coverage(saved, w, h)  # cells with fewer than 3 views are red
            for r_ in range(GRID[1]):
                for q in range(GRID[0]):
                    p0 = (q * w // 2 // GRID[0], r_ * h // 2 // GRID[1])
                    p1 = ((q + 1) * w // 2 // GRID[0], (r_ + 1) * h // 2 // GRID[1])
                    col = (0, 200, 0) if g[r_, q] >= 3 else (0, 0, 255)
                    cv2.rectangle(vis, p0, p1, col, 1)
                    cv2.putText(vis, str(g[r_, q]), (p0[0] + 6, p0[1] + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            tilts = sum(1 for p in poses if max(abs(p[3]), abs(p[4])) > 0.12)
            cv2.putText(vis, f"{status} | {len(saved)}/{a.target} views, {tilts} tilted",
                        (10, h // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            Preview.jpeg = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 70])[1].tobytes()
    except KeyboardInterrupt:
        pass
    finally:
        st.close()
    print(f"\n{len(saved)} new views in {a.out}. Coverage (views per cell):\n{coverage(saved, w, h)}")
    print(f"next: python3 scripts/m20_intrinsics_calibrate.py {a.out}")


if __name__ == "__main__":
    main()
