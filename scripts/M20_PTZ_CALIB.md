# M20: SIYI PTZ camera <-> front LiDAR

The camera at its patrol angle (e.g. yaw 70, pitch 0, zoom 1x = `zoom_level` 10.0, 1920x1080) against `/lidar_points_front`.
Pass the yaw you patrol with to the scene script: `--yaw 70`.
The result holds only for that angle and zoom.

```bash
source /opt/ros/jazzy/setup.bash && source ~/calib_ws/install/setup.bash
cd ~/Documents/Github/FAST-Calib
```

Build once: `mkdir -p ~/calib_ws/src && ln -s ~/Documents/Github/FAST-Calib ~/calib_ws/src/ && cd ~/calib_ws && colcon build --packages-select fast_calib`

## 0. Before anything

Stop anything else that moves the gimbal (Lumen: `docker stop lumen`), then set the patrol angle and zoom 1x:

```bash
python3 scripts/m20_ptz_set.py 70 0
```

It always approaches from below: the gimbal stops within ~1 deg of the target on the side it came from
(yaw 70 from 30 vs from 110: 1.1 deg apart in the picture; from the same side: within 0.15 deg). Set the
patrol angle the same way before every patrol, not from the portal. Robot standing still on flat ground.

## 1. Intrinsics (once per camera + zoom + stream size; not per gimbal angle)

The SIYI at 1x (81 deg) is a normal lens: the pinhole model with k1 k2 p1 p2. (The fisheye model on the
`m20-thor` branch is for the M20's own `/camera/front`.) The target is the FAST-Calib board itself: its 4
markers, geometry from `config/qr_params.yaml` (`marker_size`, `delta_*_qr_center`, already measured).

```bash
python3 scripts/m20_intrinsics_capture.py          # --cols/--rows/--square for a checkerboard instead
```

Watch `http://<robot-ip>:8098` on a phone. Carry the board slowly over the whole picture (all 4 markers in
view), into every corner, near (1.5 m) and far (3 m), tilted ~30 deg each way; hold it still a moment in
each pose. It keeps 45 views and stops when every cell has 3. Then:

```bash
python3 scripts/m20_intrinsics_calibrate.py calib_data/intrinsics --write
```

Want RMS < 0.5 px, HFOV ~81 deg, the error flat from centre to edge, marker spacing ratio within 2% of
1.000, no warnings; look at `undistorted.jpg`. `--write` puts fx..p2 into `config/qr_params.yaml`.

## 2. Three scenes

**Where the board goes.** The front LiDAR sees nothing behind the plane across the robot through it
(0.32 m ahead of base_link), and the camera sees 40 deg either side of its yaw. They overlap
only front-right: at yaw 90 in the left ~43% of the picture, at yaw 70 in the left ~64%. Board: on the robot's right, 2.5-3 m out, its centre
~1.2 m ahead of base_link, at camera height, facing the robot, 1 m clear of anything behind it.

**Easiest: the guide page.** Live video, what the LiDAR sees of the board, every check, and it records each
scene by itself once all checks have been green for 3 s; after scene 3 it runs the calibration (step 3) and
shows the result:

```bash
python3 scripts/m20_extrinsic_guide.py --yaw 70      # open http://<robot-ip>:8098 on a phone
```

"Redo the last scene" on the page deletes and re-records it. Stop with Ctrl-C; re-running carries on with the
next scene of today's session. Or one scene at a time from the terminal:

```bash
python3 scripts/m20_scene_capture.py scene1 --yaw 70    # facing the camera
python3 scripts/m20_scene_capture.py scene2 --yaw 70    # turned ~25 deg one way
python3 scripts/m20_scene_capture.py scene3 --yaw 70    # turned ~25 deg the other way
```

Each checks: robot still, gimbal at --yaw/--pitch, the 4 markers in the picture, the board plane in the LiDAR
(size, rings, ahead of the blind plane, all 4 holes found), and sets the distance filter itself. Fix any FAIL and re-run
(delete the scene folder first). `markers.jpg` shows what the camera found.

## 3. Solve

```bash
python3 scripts/m20_calib_run.py          # add --rviz to look at each scene
```

Good: each scene's RMSE < 1 cm (over 3 cm is left out), the red crosses on `<scene>/out/qr_detect.png` on the holes, the scenes agree with the result within ~0.5 deg / 2 cm, the camera
in base_link looks where the gimbal points (base_link yaw = -gimbal yaw), and in `out/overlay_scene*.jpg` the LiDAR lines up with the
board's edges and holes and with things behind it. Result: `calib_data/m20_<date>/out/camera_extrinsics.yaml`
(`T_cam_lidar_front`, `T_base_cam`).

## Changes to FAST-Calib for this

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
