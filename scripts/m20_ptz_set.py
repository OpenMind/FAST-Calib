#!/usr/bin/env python3
"""Set the SIYI gimbal to an angle, always approaching from the same side.

The gimbal stops within about a degree of the target on whichever side it came from:
yaw 70 reached from 30 and from 110 differ by 1.1 deg in the picture. From the same side
it returns to within 0.15 deg. So this goes to (yaw - 10, pitch - 5) first, then to the target,
and reports where it settled. Use it for calibration and for patrols alike.

    python3 scripts/m20_ptz_set.py 70 0          # yaw 70 (right), pitch 0, zoom 1x
"""
import argparse
import time

import rclpy
from geometry_msgs.msg import Vector3
from rclpy.node import Node
from std_msgs.msg import Float32

PRE_YAW, PRE_PITCH = 10.0, 5.0  # approach from below on both axes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("yaw", type=float, help="degrees, + = right")
    ap.add_argument("pitch", type=float, help="degrees, + = up")
    ap.add_argument("--zoom", type=float, default=1.0, help="zoom magnification (1 = widest)")
    ap.add_argument("--ns", default="/zr10")
    a = ap.parse_args()

    rclpy.init()
    n = Node("m20_ptz_set")
    angle = n.create_publisher(Vector3, f"{a.ns}/gimbal/angle", 10)
    zoom = n.create_publisher(Float32, f"{a.ns}/gimbal/zoom", 10)
    att, zl = [], []
    n.create_subscription(Vector3, f"{a.ns}/gimbal/attitude", lambda m: att.append((m.x, m.y)), 10)
    n.create_subscription(Float32, f"{a.ns}/gimbal/zoom_level", lambda m: zl.append(m.data), 10)

    def spin(s):
        t0 = time.time()
        while time.time() - t0 < s:
            rclpy.spin_once(n, timeout_sec=0.05)

    def send(y, p, wait):
        for _ in range(3):
            angle.publish(Vector3(x=float(y), y=float(p), z=0.0))
            spin(0.1)
        spin(wait)

    spin(0.5)
    zoom.publish(Float32(data=a.zoom))
    send(a.yaw - PRE_YAW, a.pitch - PRE_PITCH, 3.0)
    send(a.yaw, a.pitch, 4.0)
    if att:
        y, p = att[-1]
        print(f"set yaw {a.yaw:.1f} pitch {a.pitch:.1f}: gimbal reports yaw {y:.1f} pitch {p:.1f}, "
              f"zoom_level {zl[-1] if zl else '?'} (10.0 = 1x)")
    else:
        print("no /gimbal/attitude: is the gimbal driver running?")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
