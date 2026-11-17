#!/usr/bin/env python3
"""Put the PTZ camera's intrinsics from the optical-thermal calibration into config/qr_params.yaml.

The optical-thermal calibration of this camera unit (om1-patrol-vision, thermal_patrol/calib/
stereo_calib.yaml, 2026-11-10) fitted the optical lens at zoom 1x from 40 views at 2688x1520: RMS 0.19 px,
fx known to 0.26 %. That is better than a fresh m20_intrinsics_capture.py run, so it is used as is.
A copy is kept in calibration/m20_hikmicro/stereo_calib.yaml.

    python3 scripts/m20_intrinsics_from_stereo.py            # print what would be written
    python3 scripts/m20_intrinsics_from_stereo.py --write    # write fx..p2 into qr_params.yaml (.bak kept)

Re-run it (and the extrinsic calibration) when the camera is replaced or re-calibrated.
"""
import argparse
import math
import os
import shutil
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20_common import QR_PARAMS, REPO, STREAM_SIZE, set_yaml_values  # noqa: E402

DEFAULT = os.path.join(REPO, "calibration", "m20_hikmicro", "stereo_calib.yaml")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stereo_calib", nargs="?", default=DEFAULT)
    ap.add_argument("--zoom", default="1.00", help="which calibrated zoom (the extrinsics are made at 1x)")
    ap.add_argument("--params", default=QR_PARAMS)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    c = yaml.safe_load(open(a.stereo_calib))
    o = c["optical"][a.zoom]
    (fx, _, cx), (_, fy, cy), _ = o["K"]
    k1, k2, p1, p2, k3 = (list(o["dist"]) + [0.0] * 5)[:5]
    size = tuple(o["size"])
    hfov = math.degrees(2 * math.atan(size[0] / 2 / fx))
    print(f"{a.stereo_calib} ({c.get('created', '?')}), optical zoom {a.zoom}: {size[0]}x{size[1]}, "
          f"RMS {o['rms']:.2f} px over {o.get('n_views', '?')} views, HFOV {hfov:.1f} deg")
    vals = dict(fx=float(fx), fy=float(fy), cx=float(cx), cy=float(cy), k1=float(k1), k2=float(k2), p1=float(p1), p2=float(p2))
    for k, v in vals.items():
        print(f"  {k}: {v}")
    if size != STREAM_SIZE:
        sys.exit(f"the calibration is for {size}, the stream is {STREAM_SIZE}: scale it or recalibrate")
    if abs(k3) > 1e-9:
        sys.exit(f"k3 = {k3}: FAST-Calib's model has no k3")
    if not a.write:
        print("(dry run: add --write)")
        return
    shutil.copy(a.params, a.params + ".bak")
    text = set_yaml_values(open(a.params).read(), vals)   # read before opening for writing
    open(a.params, "w").write(text)
    print(f"updated {a.params} (previous in .bak). Rebuild or use --params-file so ROS sees it.")


if __name__ == "__main__":
    main()
