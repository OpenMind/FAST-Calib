#!/usr/bin/env python3
"""Camera intrinsics from the views of m20_intrinsics_capture.py, in FAST-Calib's model.

Works on views of the FAST-Calib ArUco board (its 16 marker corners) or of a checkerboard.

FAST-Calib uses fx, fy, cx, cy and k1, k2, p1, p2 with k3 = 0, so k3 is fixed here too.
Views whose reprojection error is far above the rest are dropped and the fit repeated.

    python3 scripts/m20_intrinsics_calibrate.py calib_data/intrinsics            # report only
    python3 scripts/m20_intrinsics_calibrate.py calib_data/intrinsics --write    # also into config/qr_params.yaml

Writes <dir>/intrinsics.yaml and <dir>/undistorted.jpg (straight edges should look straight).
"""
import argparse
import glob
import json
import os
import shutil
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20_common import BOARD_IDS, QR_PARAMS, board_points, set_yaml_values  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--write", action="store_true", help=f"write fx..p2 into {QR_PARAMS}")
    ap.add_argument("--params", default=QR_PARAMS)
    a = ap.parse_args()

    b = json.load(open(os.path.join(a.dir, "board.json")))
    aruco = b.get("type", "chess") == "aruco"
    if aruco:
        obj_by_id = {int(k): np.array(v, np.float32) for k, v in b["corners"].items()}
        marker = float(np.linalg.norm(obj_by_id[1][1] - obj_by_id[1][0]))
    else:
        cols, rows, sq = b["cols"], b["rows"], b["square"]
        grid = np.zeros((cols * rows, 3), np.float32)
        grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * sq

    names, objpts, imgpts, size = [], [], [], None
    for f in sorted(glob.glob(os.path.join(a.dir, "view_*.png"))):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        size = img.shape[::-1]
        if aruco:
            r = board_points(img, obj_by_id)
            if r is None:
                print(f"  {os.path.basename(f)}: not all 4 markers, skipped")
                continue
            o, c = r
        else:
            # the sector-based detector at full resolution is more accurate than the capture's
            ok, c = cv2.findChessboardCornersSB(img, (cols, rows), cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE)
            if not ok:
                print(f"  {os.path.basename(f)}: board not found, skipped")
                continue
            o = grid
        names.append(os.path.basename(f)); objpts.append(o.astype(np.float32))
        imgpts.append(c.reshape(-1, 1, 2).astype(np.float32))
    if len(names) < 10:
        sys.exit(f"only {len(names)} usable views in {a.dir}: capture at least 30")
    print(f"{len(names)} views, {size[0]}x{size[1]}")

    flags = cv2.CALIB_FIX_K3
    keep = list(range(len(names)))
    for it in range(4):
        rms, K, D, rv, tv, _, _, per = cv2.calibrateCameraExtended(
            [objpts[i] for i in keep], [imgpts[i] for i in keep], size, None, None, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8))[:8]
        per = per.ravel()
        lim = max(3 * np.median(per), 0.5)
        bad = [keep[i] for i in np.where(per > lim)[0]]
        print(f"fit {it + 1}: RMS {rms:.3f} px over {len(keep)} views (worst {per.max():.2f}, limit {lim:.2f})")
        if not bad or len(keep) - len(bad) < 10 or it == 3:  # keep rv/tv matching `keep`
            break
        print("  dropping", ", ".join(names[i] for i in bad))
        keep = [i for i in keep if i not in bad]

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2 = D.ravel()[:4]
    hfov = np.degrees(2 * np.arctan(size[0] / 2 / fx))
    # how much of the picture the corners covered: views far from the edges leave distortion unknown there
    allp = np.vstack([imgpts[i].reshape(-1, 2) for i in keep])
    g = np.zeros((6, 8), bool)
    g[np.clip((allp[:, 1] / size[1] * 6).astype(int), 0, 5), np.clip((allp[:, 0] / size[0] * 8).astype(int), 0, 7)] = True
    print(f"\nfx {fx:.2f}  fy {fy:.2f}  cx {cx:.2f}  cy {cy:.2f}   (HFOV {hfov:.1f} deg)")
    print(f"k1 {k1:.5f}  k2 {k2:.5f}  p1 {p1:.6f}  p2 {p2:.6f}")
    print(f"picture covered by corners: {g.mean() * 100:.0f}% of an 8x6 grid")
    warn = []
    if rms > 0.6: warn.append(f"RMS {rms:.2f} px is high (aim < 0.5): blur, a bent board, or motion")
    if abs(fx - fy) / fx > 0.01: warn.append("fx and fy differ by more than 1%")
    if abs(cx - size[0] / 2) > 0.05 * size[0] or abs(cy - size[1] / 2) > 0.05 * size[1]:
        warn.append("principal point is far from the centre")
    if g.mean() < 0.75: warn.append("corners do not reach the picture's edges/corners: capture more views there")
    if not 70 < hfov < 95: warn.append(f"HFOV {hfov:.0f} deg is not the ~81 deg of zoom 1x: is the zoom at 1x?")
    # error against distance from the image centre: growing outwards = the distortion model is off
    rad, err = [], []
    for j, i in enumerate(keep):
        pr = cv2.projectPoints(objpts[i], rv[j], tv[j], K, D)[0].reshape(-1, 2)
        q = imgpts[i].reshape(-1, 2)
        rad.append(np.hypot(q[:, 0] - cx, q[:, 1] - cy) / np.hypot(size[0] / 2, size[1] / 2))
        err.append(np.linalg.norm(pr - q, axis=1))
    rad, err = np.concatenate(rad), np.concatenate(err)
    bins = [(0, 0.33), (0.33, 0.66), (0.66, 1.01)]
    by_r = [np.sqrt(np.mean(err[(rad >= lo) & (rad < hi)] ** 2)) if ((rad >= lo) & (rad < hi)).any() else np.nan for lo, hi in bins]
    print("error by distance from the centre (inner / middle / outer third): "
          + " / ".join(f"{e:.2f}" for e in by_r) + " px")
    if np.isnan(by_r[2]):
        warn.append("no corners in the outer third of the picture: distortion there is extrapolated")
    elif by_r[2] > 2 * max(by_r[0], 0.1):
        warn.append("error grows towards the edges: more views near the corners, or the lens needs another model")
    if aruco:
        # each marker's own pose (marker_size) gives the marker spacings independently of the
        # board model: a constant ratio off 1 means marker_size and delta_*_qr_center disagree
        s2 = marker / 2
        mobj = np.array([[-s2, s2, 0], [s2, s2, 0], [s2, -s2, 0], [-s2, -s2, 0]], np.float32)
        nominal = {(1, 2): None, (3, 4): None, (1, 3): None, (2, 4): None}
        for k in nominal:
            nominal[k] = float(np.linalg.norm(obj_by_id[k[0]].mean(0) - obj_by_id[k[1]].mean(0)))
        ratios = []
        for i in keep:
            q = imgpts[i].reshape(4, 4, 2)
            cen = {}
            for j, mid in enumerate(BOARD_IDS):
                ok, r_, t_ = cv2.solvePnP(mobj, q[j], K, D, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                cen[mid] = t_.ravel()
            ratios += [np.linalg.norm(cen[m1] - cen[m2]) / d for (m1, m2), d in nominal.items()]
        rr = np.median(ratios)
        print(f"marker spacing measured / nominal: {rr:.3f} (median over {len(keep)} views)")
        if abs(rr - 1) > 0.02:
            warn.append(f"marker spacings are {100 * (rr - 1):+.1f}% off the nominal: re-measure marker_size "
                        "and delta_*_qr_center (a scale error here becomes a translation error in the extrinsics)")
    for w_ in warn:
        print("WARNING:", w_)

    out = os.path.join(a.dir, "intrinsics.yaml")
    with open(out, "w") as f:
        f.write(f"# SIYI A8 mini, zoom 1x, {size[0]}x{size[1]} main stream; FAST-Calib model (k3 = 0)\n")
        f.write(f"image_width: {size[0]}\nimage_height: {size[1]}\n")
        for k, v in dict(fx=fx, fy=fy, cx=cx, cy=cy, k1=k1, k2=k2, p1=p1, p2=p2).items():
            f.write(f"{k}: {v:.6f}\n")
        f.write(f"rms_px: {rms:.4f}\nviews: {len(keep)}\n")
    img = cv2.imread(os.path.join(a.dir, names[keep[0]]))
    cv2.imwrite(os.path.join(a.dir, "undistorted.jpg"), np.hstack([img, cv2.undistort(img, K, D)])[::2, ::2])
    print(f"\nwrote {out} and undistorted.jpg (left raw, right undistorted)")

    if a.write:
        shutil.copy(a.params, a.params + ".bak")
        text = open(a.params).read()
        text = set_yaml_values(text, dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy),
                                          k1=float(k1), k2=float(k2), p1=float(p1), p2=float(p2)))
        open(a.params, "w").write(text)
        print(f"updated {a.params} (previous in .bak). Rebuild or use --params-file so ROS sees it.")


if __name__ == "__main__":
    main()
