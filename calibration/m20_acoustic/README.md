# M20 acoustic camera (CRYSOUND) ↔ front LiDAR calibration

How to calibrate the M20's fixed CRYSOUND acoustic camera against the front LiDAR (`/lidar_points_front`):
the camera's **intrinsics** (K and distortion at 1920×1080) and its **mount** on the robot (x, y, z and
yaw / pitch / roll in `base_link`, plus `T_cam_lidar`). It takes about 40 minutes with one person moving the
board and the robot standing still.

**Status:** calibrated 2026-11-19 on one M20; the results are in this folder and in step 5. Whether other
M20s' acoustic camera mounts match has not been checked.

## 1. How it works

Two steps, both with the same FAST-Calib board used for the PTZ camera: 1.44 × 0.96 m, four ArUco markers
(DICT_6X6_250, ids 1–4, 0.20 m, centres 1.10 × 0.70 m apart) and four holes (r 0.12 m, centres 0.50 × 0.40 m apart).

**Step A: intrinsics.** The camera ships with none, so they are measured. Each view of the board gives the
16 marker corners. Views are kept only when the board is still, sharp and in a new pose, until every cell
of a 4×3 grid over the picture holds 3 views. OpenCV `calibrateCamera` then fits fx, fy, cx, cy, k1, k2, p1,
p2 (k3 fixed at 0, the model FAST-Calib uses), dropping views whose error is far above the rest. It also
checks the error by distance from the centre, the corner coverage and the marker spacing.

**Step B: camera ↔ LiDAR (FAST-Calib, several scenes, solved jointly).**
1. *Camera:* PnP on the 16 marker corners gives the board's pose. The 4 hole centres follow from the
   board's geometry.
2. *LiDAR:* the board's plane is found inside a search box ahead of the robot. On that plane the
   `pattern` detector fits the 4 holes as one rigid 0.50 × 0.40 m rectangle over a free-space map. A rim
   check rejects gaps between scan lines, and each hole's centre is the centroid of its free region.
3. *Solve:* every scene gives 4 matching hole centres (LiDAR ↔ camera). `multi_fast_calib` finds the one
   rigid transform `T_cam_lidar` that fits all of them (4 × 6 = 24 points). Scenes with the board facing,
   turned ±25°, left, right and nearer pin down rotation and depth together.
4. `T_base_cam = T_base_lidar · T_cam_lidar⁻¹`, with `T_base_lidar` from the robot's TF. The mount angles
   come from that.

The front LiDAR's spin axis points straight ahead, 0.54 m above the ground. Its scan lines are circles
around that point, and there is a small disc with no returns at the point itself, which looks like a hole.
Keep that point between the board's 4 holes; the page shows it as a blue +.

## 2. The camera

- CRYSOUND fixed acoustic camera on the robot's internal network: REST API on port 9999, user `admin`.
  The address and password are in the camera's documentation and Lumen's config. **Do not commit them:
  this repo is public.**
- The scripts take the **clean picture from the REST API** (`GET /api/image/v1/imrData`,
  `cameraImageData`), **1920×1080**, about 3 frames a second. Do not use the RTSP video: it has the sound
  cloud, dB label and focus box drawn into it.
- Fixed lens, no zoom, no gimbal: nothing to set before calibrating.
- After calibrating, do not change the video resolution (`/api/video/v1/config`), the image mirror
  (`/api/image/v1/imageMirrored`, both off now) or the camera's mount. Any of these invalidates the result.
- The camera's own "positioning calibration" (sound cloud → pixel) is factory-made and unrelated: leave it.
- The `focus` settings are the acoustic focus box, not the lens.

## 3. Setup (once)

```bash
git clone -b m20-acoustic https://github.com/OpenMind/FAST-Calib.git ~/Documents/GitHub/FAST-Calib-acoustic
mkdir -p ~/calib_ws_acoustic/src && ln -sfn ~/Documents/GitHub/FAST-Calib-acoustic ~/calib_ws_acoustic/src/fast_calib
cd ~/calib_ws_acoustic && source /opt/ros/jazzy/setup.bash && colcon build --packages-select fast_calib   # ~1.5 min
```

Every new terminal:

```bash
source /opt/ros/jazzy/setup.bash && source ~/calib_ws_acoustic/install/setup.bash
cd ~/Documents/GitHub/FAST-Calib-acoustic
export ACOUSTIC_CAM_URL=http://<camera-ip>:9999
export ACOUSTIC_CAM_PASSWORD='<camera password>'
```

Quick check that the picture comes through:

```bash
python3 -c "import sys,cv2;sys.path.insert(0,'scripts');from m20_common import AcousticStream as A;s=A();f,_=s.read();cv2.imwrite('/tmp/ac.jpg',f);print(f.shape);s.close()"
# (1080, 1920, 3)
```

Before starting:
- Robot **off the dock**, standing still on flat ground, about 4 m clear ahead.
- No patrol running.
- Port 8098 free: nothing else of these scripts running.
- Check the board with a tape measure: marker size 0.20 m, marker centres 1.10 × 0.70 m apart.

## 4. Steps

### 4.1 Intrinsics (about 10 min)

```bash
python3 scripts/m20_intrinsics_capture.py --camera acoustic
# open http://<robot-ip>:8098 on a phone
```

Move the board slowly in front of the robot. The robot stays still.
- Show **all 4 markers** each time and hold still about 1 s. The page says `SAVED n`.
- Cover the **whole picture**: middle, the four corners, the four edges. A grid cell turns green at 3 views.
- Distances 1.5–3 m. About a third of the views **tilted** 20–30° left, right, up or down.
- It stops by itself at 45 views with every cell green (or Ctrl-C). Views go to
  `calib_data/intrinsics_acoustic/`; running it again adds to them.

```bash
python3 scripts/m20_intrinsics_calibrate.py calib_data/intrinsics_acoustic            # report only
python3 scripts/m20_intrinsics_calibrate.py calib_data/intrinsics_acoustic --write    # into config/acoustic_params.yaml
```

Good: RMS < 0.5 px, corners cover ≥ 75 % of the picture, marker spacing measured/nominal within 2 %.
`undistorted.jpg` (left raw, right undistorted): straight edges look straight. This camera's pixels are not
square: its 4:3 picture is stretched to 1920×1080 (the board's round holes look wide), so fy ≈ 0.75·fx and
the "fx and fy differ" WARNING is expected. Expect a horizontal view near 64° (fx ≈ 1535 px) and a vertical
one near 50° (fy ≈ 1150 px); a value far off means a problem. If it fails, add views where the report says
(edges or corners) and solve again.

`--write` replaces the seed values in `config/acoustic_params.yaml` (old copy in `.bak`). The next step
refuses to start until it has run.

### 4.2 Camera ↔ LiDAR, six scenes (about 20 min)

```bash
python3 scripts/m20_extrinsic_guide.py --camera acoustic
# open http://<robot-ip>:8098 on a phone
```

The page shows live:
- the picture, with a **yellow outline: "PUT THE BOARD HERE"** and an arrow from the board to it;
- the green circles where the markers put the holes;
- a top-down map with the green zone for the board;
- the LiDAR's view of the board, with the holes it found and the LiDAR centre (blue +);
- every check.

Once every check has been green for 3 s it records the scene by itself (5 s, keep still). It then checks
the scene with FAST-Calib and moves on.

| Scene | Board |
|---|---|
| 1 | facing the camera, middle of the picture, 2.4–3.2 m |
| 2 | turned about +25° |
| 3 | turned about −25° |
| 4 | facing, on the left of the picture |
| 5 | facing, on the right of the picture |
| 6 | facing, nearer, about 2.1 m |

Tips:
- Board centre about 0.55 m above the ground (the LiDAR's height), 1 m clear of anything behind it.
- Nothing in front of the markers.
- If the LiDAR says only 3 holes, or the blue + sits on a hole, move the board a little up, down or sideways.
- "Redo the last scene" re-records it. Ctrl-C stops; running it again carries on from the next scene of
  today's session (`calib_data/m20_acoustic_YYYYMMDD/`).

After scene 6 it runs the solve (`m20_calib_run.py`) and shows the result on the page. To run it again by hand:

```bash
python3 scripts/m20_calib_run.py calib_data/m20_acoustic_YYYYMMDD
```

### 4.3 Is it good?

- Joint solve RMSE ≲ 2.5 cm. Each scene's own RMSE is ~0 by construction (`pattern`), so it says nothing.
- Each scene agrees with the result within about 2.5° and 7 cm. The PTZ's five scenes agreed
  within 0.8–2.2° and 2.7–6.3 cm. A scene far outside that is a bad scene: redo it.
- The mount is plausible: x, y, z match a tape measure from the robot's centre within a few cm, and the
  roll is near 0. A roll near ±180° means the holes were matched upside down.
- `out/overlay_scene*.jpg`: the LiDAR points (coloured by distance) line up with the board's edges and holes
  and with things behind it.

The result is `calib_data/m20_acoustic_YYYYMMDD/out/camera_extrinsics.yaml`, with these fields:
- `camera_mount`: x, y, z and yaw / pitch / roll in `base_link`;
- `T_cam_lidar_front`: p_cam = T · p_lidar;
- `T_base_cam`: camera optical frame x right, y down, z forward;
- `fx` … `p2`.

## 5. Deliver

Copy into this folder and push:
- `camera_extrinsics.yaml`
- `intrinsics.yaml` from `calib_data/intrinsics_acoustic/`
- `out/circle_center_record.txt`, renamed `hole_centres.txt`

Then fill in the numbers below. Do not commit pictures with people in them, or the camera's address or
password.

**Lumen's 1344-px picture.** Lumen shrinks the camera's frames to 1344 px wide (scale s = 1344 / 1920 = 0.7).
For that picture use fx·s, fy·s, cx·s, cy·s, with the distortion k1, k2, p1, p2 unchanged. To be exact
with OpenCV's pixel centres, use (cx + 0.5)·s − 0.5, which differs by 0.15 px. The extrinsics do not
depend on the picture size.

| Result | Value |
|---|---|
| Intrinsics 1920×1080 (fx, fy, cx, cy; k1, k2, p1, p2) | 1535.27, 1150.66, 1026.93, 536.09; 0.00228, −0.07415, 0.00255, 0.00040 (RMS 1.04 px, 44 views) |
| Intrinsics for Lumen's 1344×756 picture (fx, fy, cx, cy) | 1074.69, 805.46, 718.70, 375.11 |
| Mount in `base_link` (x, y, z m; yaw_left, pitch_down, roll deg) | 0.210, 0.001, 0.129; −0.52, −1.53, +0.75 |
| Joint RMSE; scene agreement | 1.32 cm over scenes 1, 2, 3, 5, 6; each within 0.5–1.7° and 2.4–8.3 cm |

Notes on this run:
- Intrinsics RMS is 1.0 px, above the 0.5 px aim, but the fit is stable: without the views nearer than
  1.8 m it gives fx, fy within 0.2 % and cx, cy within 3 px.
- Scene 4 is not in the solve: the first recording disagreed by 4.3° and the redo failed the LiDAR hole
  check (someone stood behind the board). Scene 5 agrees within 8.3 cm, a little over the 7 cm guide; it too
  was recorded with someone behind the board.
- An earlier six-scene run on the same day (scenes 2, 4 and 6 redone after it) gave a mount within 1.6 cm and
  0.5° of this one.

## 6. Files

| File | What |
|---|---|
| `config/acoustic_params.yaml` | This camera's intrinsics (seed until 4.1 `--write`), the board's geometry, the LiDAR topic |
| `scripts/m20_common.py` | `AcousticStream` (REST picture, re-login on expiry), `FixedMount`, `--camera` table |
| `scripts/m20_intrinsics_capture.py`, `m20_intrinsics_calibrate.py` | Step 4.1 (`--camera acoustic`) |
| `scripts/m20_extrinsic_guide.py` | Step 4.2, the guided page (`--camera acoustic`: no gimbal check) |
| `scripts/m20_scene_capture.py` | The LiDAR board / hole checks the guide uses (`pattern` detector) |
| `scripts/m20_calib_run.py` | The solve: every scene, then all jointly; writes `camera_mount` for a fixed camera |
| `src/lidar_detect.hpp` (`detect_pattern_lidar`), `src/multi_scene.cpp` | The C++ LiDAR hole detector and the joint solve |
| `calibration/m20_hikmicro/` | The PTZ camera's finished calibration: an example of the result and its accuracy |
