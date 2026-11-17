# M20 PTZ camera calibration (HIKMICRO HM-TD5767T-7)

The calibration of the M20's dual-spectrum PTZ head, for the **gauge-recognition** and **thermal
pipe-detection** pipelines: where the camera is on the robot, how its picture relates to the front
LiDAR and to the thermal matrix, at **any gimbal angle and zoom**.

**Status (2026-11-16):** calibrated and checked. Direction is good to about **0.5°** (2 cm at 2.5 m),
the camera's position to 2–4 cm. The values in section 2 are **not yet set on the robot** (the
orchestrator still holds the old SIYI mount).

It belongs to this camera unit as mounted on this robot. Recalibrate after the camera is remounted
or replaced (section 7).

## 1. Files

| File | What it is | Used for |
|---|---|---|
| `ptz_calibration.yaml` | **Start here.** Every parameter in one place: gimbal zero, pivot, LiDAR↔camera, what to set on the robot, 1× intrinsics, zoom table, accuracy. | everything |
| `stereo_calib.yaml` | The optical–thermal calibration (2026-11-10, from `om1-patrol-vision` `thermal_patrol/calib/`): optical intrinsics at 1×, 1.5×, 2×, 3×, the zoom curve 1–6×, thermal intrinsics, optical→thermal `R` and `T(zoom)`. | intrinsics at any zoom; optical ↔ thermal |
| `ptz_model.py` | The model as functions (reads the two files above): camera pose and intrinsics at any gimbal angle and zoom, LiDAR → pixel, pixel → distance / 3-D point, optical → thermal. | call it, or port it |
| `camera_extrinsics.yaml` | Raw output of the camera–LiDAR solve (5 board scenes). Reference only. | traceability |
| `hole_centres.txt` | The board's hole centres per scene as the LiDAR and the camera saw them (the solve's input). | re-solving, checking |

Raw captures (pictures, LiDAR bags) are on the robot, not in git:
`~/Documents/GitHub/FAST-Calib-hikmicro/calib_data/m20_20261116b/`.

## 2. Parameters to set on the robot

| Where | Value |
|---|---|
| Orchestrator PTZ mount, `POST /ptz/extrinsics/set` | `{"x": -0.026, "y": 0.122, "z": 0.337, "yaw_sign": -1, "yaw_offset_deg": -1.34, "pitch_offset_deg": -1.61}` |
| Lumen | `LUMEN_GAUGE_HFOV=54.5` (the default 53.6 is the old SIYI's) |

- `x, y, z` is the **gimbal's pivot** in `base_link` (2.6 cm behind its origin, 12.2 cm left, 33.7 cm
  up; the lens is ~0.9 m above the ground). The pivot, not the lens, is the right origin for aiming.
- The offsets carry the head's zero error: at Lumen yaw 0 / pitch 0 the camera really looks **1.34°
  right and 1.61° down**. Alternative: leave both offsets 0 and set Lumen's own zero
  (`LUMEN_GIMBAL_FORWARD_AZIMUTH=358.7`, `LUMEN_GIMBAL_LEVEL_ELEVATION=1.6`). **Never both.**
- Gauges marked in the portal before this change must be marked again.
- Do not use the camera's own "north", "zero azimuth" or "installation tilt" calibrations: they make
  the angles relative to north or gravity instead of the robot.

## 3. The model: any gimbal angle, any zoom

One camera–LiDAR calibration (head at yaw 0, pitch −5, zoom 1×) plus the head's angles and the zoom
curve give everything else. Gimbal angles are Lumen's (`GET /v1/ptz/status`): yaw + right, pitch + up,
zoom in × (= ISAPI `absoluteZoom` / 10).

**Pose** (camera optical frame: x right, y down, z forward; `base_link`: x forward, y left, z up):

```
R_base_cam(yaw, pitch) = Rz(-yaw) · Ry(-pitch) · R_mount
t_base_cam(yaw, pitch) = pivot + lens_ahead · (third column of R_base_cam)
p_cam = inv(T_base_cam(yaw, pitch)) · T_base_lidar · p_lidar
```

`R_mount`, `pivot`, `lens_ahead` (0.072 m: the lens sits ahead of the pivot and swings with the
head) and `T_base_lidar` are in `ptz_calibration.yaml`.

**Intrinsics at a zoom** (2688×1520): the fit at the nearest calibrated zoom (1, 1.5, 2, 3),
magnified along the zoom curve about the zoom centre (1336, 767) px. **The zoom number is not the
magnification:**

| Zoom | 1× | 1.5× | 2× | 3× | 4× | 5× | 6× |
|---|---|---|---|---|---|---|---|
| Magnification | 1.00 | 1.31 | 1.80 | 2.60 | 3.41 | 4.21 | 5.04 |
| Horizontal view | 54.5° | 43.5° | 34.5° | 23.1° | 17.8° | 14.4° | 12.1° |

**Optical → thermal** (both lenses turn with the head, so this does not depend on the gimbal angle):
`X_thermal = R · X_optical + T(zoom)`, thermal intrinsics in `stereo_calib.yaml`.

```python
from ptz_model import PTZ
ptz = PTZ()
T = ptz.camera_in_base(yaw, pitch)                           # 4x4 base_link <- camera
K, dist = ptz.intrinsics(zoom)
uv, depth = ptz.lidar_to_pixel(P_lidar, yaw, pitch, zoom)    # /lidar_points_front points -> picture
d = ptz.depth_at_pixel(P_lidar, u, v, yaw, pitch, zoom)      # distance of what is at pixel (u, v)
p = ptz.pixel_to_base(u, v, d, yaw, pitch, zoom)             # that point in base_link
X_t = ptz.optical_to_thermal(X_cam, zoom)                    # into the thermal camera's frame
```

## 4. Gauge recognition

- **Aiming at a marked gauge:** the orchestrator mount of section 2 (patrol node → Lumen `view`).
- **Pixels ↔ degrees** (centring a dial, the gate that decides which dial is the gauge): use
  `intrinsics(zoom)` / `hfov(zoom)`. Lumen's `hfovAt` assumes the magnification equals the zoom
  number, which over-turns by 10–20 % when zoomed (2× is 1.80, 5× is 4.21): replace it with the
  table above.
- **Which dial is the gauge / where is it on the map:** `depth_at_pixel` at the dial's centre with the
  gimbal's angles at that moment, then `pixel_to_base`, then the robot's map pose. Compare that 3-D
  position with the marked one instead of gating on angles alone. The same gives a gauge's position
  when marking it (no height to guess).

## 5. Thermal pipe detection

- **Optical mask → thermal matrix** needs the surface's distance `d` (the lenses are ~20 cm apart):
  `H(d) = K_t (R + T nᵀ/d) K_o⁻¹` (`StereoCalib` in `thermal_patrol/pipe_hotspot/stereo.py`). Take `d`
  from the LiDAR, `depth_at_pixel` at the pipe, instead of searching for it on thermal edges.
- **LiDAR point → thermal pixel:** `optical_to_thermal(p_cam, zoom)`, then project with
  `thermal_intrinsics()`.
- `T(zoom)` is calibrated at 1×, 1.5×, 2×, 3× (the entrance pupil moves ~3.7 cm back from 1× to 3×).

## 6. Accuracy and limits

| Check | Result |
|---|---|
| Camera–LiDAR joint solve, 5 scenes (20 hole centres) | RMSE 22 mm; LiDAR holes land 0.40° (18 px) from the camera's on average |
| Independent 3-scene calibration the same day | within 0.6° and 4 cm |
| Leaving one scene out | changes at most 0.85° and 2.8 cm |
| **Other gimbal angles and zoom** (board at 2.2 m; yaw −2…+15°, pitch −8…−13°, zoom 1× and 1.3×) | **0.5–0.65°** per pose, the same at every angle once `lens_ahead` is in the model |
| Looked at, not measured: yaw −10°, pitch −1°, zoom 1.5× and 1.8× | predicted holes on the real holes |
| Distance of the board's centre, LiDAR vs camera | 2.769 m vs 2.759 m |
| Lens position by tape | ~0.9 m above the ground (calibration: 0.88–0.94), left of the centre line (rough tape 9 cm; calibration 12 cm) |
| Optical–thermal | under 1 thermal pixel: held-out boards 0.18–0.69 px, hot object 0.13–0.35 px at 1×–5× |

Expect **about 0.5° in direction and 2–4 cm in position**. Enough to aim at a 10–15 cm dial and to
tell which object a pixel is; not a centimetre-grade survey.

Limits:
- Not measured beyond yaw +15° / −2°, pitch above −4°, or zoom above 1.3× against the LiDAR (the board
  no longer fits in the picture). The zoom intrinsics themselves are measured to 6×; above that the
  curve is extrapolated.
- The front LiDAR sees only ahead of the plane 0.32 m in front of `base_link`, in circles around its
  forward axis about 5 cm apart at 2.5 m: a small object may carry few points, and there are none when
  the head looks sideways or back.
- The scenes disagree by 1.5–3 cm sideways for boards at the picture's edges and for turned boards,
  cause not found (the LiDAR's hole edges are the suspect). It is inside the figures above.
- `lens_ahead` is for 1×; the pupil's move with zoom (~4 cm) matters little beyond 1.5 m.
- Use the gimbal angles and the LiDAR cloud of the same moment, robot standing still.

## 7. Redoing it

`scripts/M20_PTZ_CALIB.md`: `m20_ptz_set.py 0 -5`, then `m20_extrinsic_guide.py --pitch -5` (six
guided board scenes, about 20 minutes), then `m20_calib_run.py`. The optical–thermal calibration is
redone in `om1-patrol-vision` (`thermal_patrol/CALIBRATION.md`); copy its `stereo_calib.yaml` here.
`ptz_calibration.yaml` is then rebuilt from the new `camera_extrinsics.yaml`: `R_mount =
Ry(pitch₀) · Rz(yaw₀) · R_base_cam`, `pivot = t_base_cam − lens_ahead · z_cam`, with (yaw₀, pitch₀)
the gimbal angle of the calibration.
