#!/usr/bin/env python3
"""Point the M20's HIKMICRO PTZ head through Lumen, always approaching from the same side.

Yaw 0 = the robot's front (+ = right), pitch 0 = level (+ = up), as Lumen defines them
(LUMEN_GIMBAL_FORWARD_AZIMUTH / LEVEL_ELEVATION). The camera-LiDAR calibration is made looking
straight ahead at zoom 1x, which is the default:

    python3 scripts/m20_ptz_set.py                 # yaw 0, pitch 0, zoom 1x
    python3 scripts/m20_ptz_set.py 0 0 --zoom 1

Nothing else may move the head meanwhile: no patrol and no gauge read running.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m20_common import LUMEN_URL, LumenGimbal  # noqa: E402

PRE_YAW, PRE_PITCH = 5.0, 3.0  # approach from below-left on both axes, so backlash is always taken up the same way


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaw", type=float, nargs="?", default=0.0, help="degrees, + = right")
    ap.add_argument("pitch", type=float, nargs="?", default=0.0, help="degrees, + = up")
    ap.add_argument("--zoom", type=float, default=1.0, help="zoom magnification (1 = widest)")
    ap.add_argument("--lumen", default=LUMEN_URL)
    a = ap.parse_args()

    g = LumenGimbal(a.lumen)
    try:
        g.set(a.yaw - PRE_YAW, a.pitch - PRE_PITCH, a.zoom)
        time.sleep(2.0)
        g.set(a.yaw, a.pitch, a.zoom)
        time.sleep(3.0)
        st = g.status()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Lumen did not move the head ({e}): is the lumen container running, and no patrol or gauge read busy?")
    finally:
        g.close()
    off = max(abs(st["yaw"] - a.yaw), abs(st["pitch"] - a.pitch))
    print(f"set yaw {a.yaw:.1f} pitch {a.pitch:.1f} zoom {a.zoom:g}: head reports yaw {st['yaw']:.1f} "
          f"pitch {st['pitch']:.1f} zoom {st['zoom']:g}" + ("" if off < 0.5 else "   WARNING: not there yet"))


if __name__ == "__main__":
    main()
