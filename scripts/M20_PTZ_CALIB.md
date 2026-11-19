# M20: HIKMICRO PTZ <-> front LiDAR

(The fixed CRYSOUND acoustic camera uses the same tools with `--camera acoustic`:
[`calibration/m20_acoustic/`](../calibration/m20_acoustic/README.md).)

**The current result, its parameters and how to use them at any gimbal angle and zoom:
[`calibration/m20_hikmicro/`](../calibration/m20_hikmicro/README.md).** This page is how to redo it.

The M20's PTZ head is the HIKMICRO HM-TD5767T-7 dual-spectrum camera. It is calibrated **looking straight
ahead, level, at zoom 1x** (Lumen's yaw 0 / pitch 0), against `/lidar_points_front`. Looking ahead, the
whole picture is inside the front LiDAR's view, so the board simply goes in front of the robot.

What the calibration gives:
- `T_cam_lidar_front` / `T_base_cam`: the LiDAR projected onto the picture, the camera's position on the robot.
- **Where Lumen's yaw 0 / pitch 0 really points.** The script prints the `LUMEN_GIMBAL_FORWARD_AZIMUTH` /
  `LUMEN_GIMBAL_LEVEL_ELEVATION` that make yaw 0 the robot's front and pitch 0 level, and the
  `/ptz/extrinsics` (x, y, z) the orchestrator uses to aim at map targets (gauges).

The LiDAR projection holds at yaw 0 / pitch 0 / zoom 1x only. Aiming elsewhere uses the corrected zero
and the camera position; the head's pivot sits a few cm behind the lens, which matters little beyond ~1.5 m.

Everything goes through the robot's services: the picture from the relay `rtsp://127.0.0.1:8554/ptz_camera`
(2688x1520), the head through Lumen (`http://127.0.0.1:8999/v1/ptz/...`). No camera credentials are needed.

```bash
source /opt/ros/jazzy/setup.bash && source ~/calib_ws_hikmicro/install/setup.bash
cd ~/Documents/GitHub/FAST-Calib-hikmicro          # branch m20-hikmicro
```

Build once: `mkdir -p ~/calib_ws_hikmicro/src && ln -sfn ~/Documents/GitHub/FAST-Calib-hikmicro ~/calib_ws_hikmicro/src/fast_calib && cd ~/calib_ws_hikmicro && colcon build --packages-select fast_calib`

## 0. Before anything

- Robot **off the dock**, standing still on flat ground, 4 m clear ahead of it.
- No patrol and no gauge read running (they move the head). Lumen itself keeps running: it drives the head.
- **Do not** use the camera's own "north", "zero azimuth" or "installation tilt" calibrations (web page,
  user manual 4.3.3 / 14.13): they make the angles relative to north or gravity, not to the robot.

```bash
python3 scripts/m20_ptz_set.py          # yaw 0, pitch 0, zoom 1x, approached from the same side every time
```

## 1. Intrinsics: already done

The optical-thermal calibration of this camera (om1-patrol-vision `thermal_patrol/calib/stereo_calib.yaml`,
2026-11-10, copy in `calibration/m20_hikmicro/stereo_calib.yaml`) fitted the optical lens at 1x from 40 views at
2688x1520 with RMS 0.19 px: better than a fresh capture. It is written into `config/qr_params.yaml`:

```bash
python3 scripts/m20_intrinsics_from_stereo.py --write     # fx 2608.2 fy 2610.5 cx 1328.0 cy 754.3, HFOV 54.5 deg
```

(`m20_intrinsics_capture.py` / `m20_intrinsics_calibrate.py` still work for a new camera: want HFOV ~54.5 deg.)

## 2. Six scenes

**Where the board goes.** Straight ahead, 2.2-2.6 m out, near the robot's centre line, its centre about
0.55 m above the ground, facing the robot, 1 m clear of anything behind it.

Why that height: the front LiDAR's spin axis points straight ahead, 0.54 m above the ground. Its scan lines
are circles around that point, and at the point itself there is a small disc with no returns that looks
like a hole. With the board centred there the point falls between the 4 holes (the blue + on the page's
board view). The camera sits higher, so **tilt it down** to see the board, ~5 deg at 2.3 m:

```bash
python3 scripts/m20_ptz_set.py 0 -5
python3 scripts/m20_extrinsic_guide.py --pitch -5
```

Any pitch works: the solve reads the head's angle and gives the zero for yaw 0 / pitch 0 all the same.
The camera sees 27 deg either side and 16 deg up and down: at 2.3 m about 2.4 x 1.3 m.

The six scenes differ in pose and place, which pins depth and rotation down together: (1) facing, in the
middle; (2) turned +25 deg; (3) turned -25 deg; (4) facing, on the left of the picture; (5) on the right;
(6) facing, nearer (~2.1 m). The page draws where the board goes for the current scene (yellow outline on the
picture, green zone on the map) and says how to move it. The joint solve uses every scene recorded.

**Easiest: the guide page.** Live video, what the LiDAR sees of the board, every check, and it records each
scene by itself once all checks have been green for 3 s; after scene 6 it runs the calibration (step 3) and
shows the result:

```bash
python3 scripts/m20_extrinsic_guide.py      # open http://<robot-ip>:8098 on a phone
```

"Redo the last scene" on the page deletes and re-records it. Stop with Ctrl-C; re-running carries on with the
next scene of today's session. Or one scene at a time from the terminal:

```bash
python3 scripts/m20_scene_capture.py scene1    # facing the camera
python3 scripts/m20_scene_capture.py scene2    # turned ~25 deg one way
python3 scripts/m20_scene_capture.py scene3    # turned ~25 deg the other way
```

Each checks: robot still, gimbal at --yaw/--pitch, the 4 markers in the picture, the board plane in the LiDAR
(size, rings, ahead of the blind plane, all 4 holes found), and sets the distance filter itself. Fix any FAIL and re-run
(delete the scene folder first). `markers.jpg` shows what the camera found.

## 3. Solve

```bash
python3 scripts/m20_calib_run.py          # add --rviz to look at each scene
```

Good: each scene's RMSE < 1 cm (over 3 cm is left out), the red crosses on `<scene>/out/qr_detect.png` on the holes, the scenes agree with the result within ~0.5 deg / 2 cm, the camera
in base_link looks where the gimbal points (base_link yaw = -gimbal yaw, within a few deg), and in `out/overlay_scene*.jpg` the LiDAR lines up with the
board's edges and holes and with things behind it. Result: `calib_data/m20_<date>/out/camera_extrinsics.yaml`
(`T_cam_lidar_front`, `T_base_cam`, `lumen_gimbal_zero_corrected`, `ptz_extrinsics`).

## 4. Apply

1. **Lumen's zero.** Set the printed `LUMEN_GIMBAL_FORWARD_AZIMUTH` / `LUMEN_GIMBAL_LEVEL_ELEVATION` in
   Lumen's environment and restart it. Check: `python3 scripts/m20_ptz_set.py` then run step 2 + 3 once
   more (or just look at a target straight ahead): the solve should now report about 0 / 0.
2. **Orchestrator.** Save the printed `/ptz/extrinsics` (x, y, z, `yaw_sign` -1, offsets **0**: the
   offset now lives in Lumen's zero; correcting it in both places counts it twice).
3. **Lumen's field of view.** `LUMEN_GAUGE_HFOV=54.5` (the default 53.6 is the old SIYI's).
4. Mark gauges again in the portal if they were marked before the zero changed.

## Changes to FAST-Calib for this

HIKMICRO (branch `m20-hikmicro`, from `m20-ptz`): the head through Lumen's HTTP API instead of the SIYI ROS
driver, the 2688x1520 relay stream, the board straight ahead (search box, zone, the 54.5 deg view), the
intrinsics from the optical-thermal calibration, and the solve's Lumen zero / orchestrator extrinsics.
The SIYI A8 mini (`m20-ptz`) notes below still apply to the LiDAR side.

LiDAR holes, `lidar_type: pattern` (C++ `detect_pattern_lidar`, Python `match_pattern`): looking straight
ahead, the front LiDAR's scan lines are circles around its spin axis; gaps between them, the empty disc at
the axis and cut edges look like small holes, and the hole-by-hole `grid` search kept picking those (3 holes
+ 1 spurious, or 3 + 1 inferred, fitted radius 0.15-0.16 m for 0.12). `pattern` fits the 4 holes as one
rigid rectangle over a free-space distance map, takes each hole's free-region centroid, and fits the rigid
pattern (no scale) to them, so the outward lean of the centroids (sparser lines outward) cancels instead of
reading as a nearer board. Live: all 4 holes in every single cloud, pattern centre repeatable to 1.4 mm.
With it each scene's own RMSE is ~0 by construction (both sides are exact rectangles): check the hole fit
(`[LiDAR/pattern] ... centroids within`, < 3 cm), the joint solve's RMSE and the scenes' agreement.

- ArUco on OpenCV >= 4.7 (JetPack's 4.8 has no `aruco` contrib module): `qr_detect.hpp`.
- LiDAR hole order for a tilted LiDAR / a board beside the robot: `lidar_up_*` params, set by the
  scene script from TF. The old fixed axis swap matched the holes wrongly (7 cm RMSE instead of 3 mm).
- The plane alignment's rotation axis normalised (turned boards found no holes).
- From the `m20-thor` / `rover-jetson` branches: debug clouds `out/dbg_*.pcd`, the LiDAR hole centres drawn
  on `qr_detect.png`, `plane_dist_threshold` as a parameter (0.02 here), the LiDAR hole check before
  recording, and the intrinsics checks (error by radius, marker spacing). Their 8-ordering search is
  replaced by the ordering from TF plus a roll check: the hole rectangle is 180-deg symmetric, so a
  flipped match fits as well and shows only as a camera rolled ~180 deg.
- LiDAR holes: the ring-edge detector (`detect_mech_lidar`) finds too few hole edges on the front LiDAR, whose
  scan lines cross the board nearly vertically ~5 cm apart. The scripts set `lidar_type: grid`: the `m20-thor`
  branch's occupancy-grid detector, changed so holes inside the board's outline count even when its thin
  border has a gap, holes enlarged by the scan gaps (r ~0.18 m for 0.12) are accepted, the closing size grows
  until 4 holes fit, and a 4th hole is completed from an L of 3 (board edge at the blind plane). First real
  scene: RMSE 5 mm, camera yaw within 1 deg of the gimbal's reading.
