# FAST-Calib

## FAST-Calib: LiDAR-Camera Extrinsic Calibration in One Second

FAST-Calib is an efficient target-based extrinsic calibration tool for LiDAR-camera systems (eg., [FAST-LIVO2](https://github.com/hku-mars/FAST-LIVO2)). 

**Key highlights include:** 

1. Support solid-state and mechanical LiDAR.
2. No need for any initial extrinsic parameters.
3. Achieve highly accurate calibration results **in just one seconds**.

In short, it makes extrinsic calibration as simple as intrinsic calibration.

**Related paper:** 

[FAST-Calib: LiDAR-Camera Extrinsic Calibration in One Second](https://www.arxiv.org/pdf/2507.17210)

📬 For further assistance or inquiries, please feel free to contact Chunran Zheng at zhengcr@connect.hku.hk.

<p align="center">
  <img src="./pics/calib.jpg" width="100%">
  <font color=#a0a0a0 size=2>Left: Example of Mid360 LiDAR calibration. Right: Point cloud colored with the calibrated extrinsics.</font>
</p>

<p align="center">
  <img src="./pics/all_lidar_type.jpg" width="100%">
  <font color=#a0a0a0 size=2>Circular hole extraction supports multiple LiDAR models.</font>
</p>

## 1. Prerequisites
ROS 2 (tested on Humble), PCL>=1.8, OpenCV>=4.0.

The Livox `CustomMsg` lidar input is optional: if [`livox_ros_driver2`](https://github.com/Livox-SDK/livox_ros_driver2) is built in your workspace it is picked up automatically; otherwise FAST-Calib still builds and works with any `sensor_msgs/PointCloud2` LiDAR.

## 2. Run our examples
1. Prepare the static acquisition data in the `calib_data` folder (see [Single-scene Calibration Sample Data](https://drive.google.com/drive/folders/1W87Dx3MUuPhTpCLvaavWqNUJZV03yU6L?usp=drive_link) from Mid360, Avia and Ouster, and [Multi-scene Calibration Sample Data](https://drive.google.com/drive/folders/1g__plgFqp5tsk-TY7Ioh4RXru62AdLmr?usp=drive_link) from Avia):
- a ROS 2 bag (directory containing `metadata.yaml`) with the point cloud topic, e.g. recorded with `ros2 bag record /livox/lidar`
- corresponding image

2. Run the single-scene calibration process:
```bash
ros2 launch fast_calib calib.launch.py bag_path:=/path/to/bag image_path:=/path/to/image.jpg
```

3. After completing Step 2 for at least three different scenes, you can perform multi-scene joint calibration:
```bash
ros2 launch fast_calib multi_calib.launch.py
```

## 3. Run on your own sensor suite
1. Customize the calibration target in the image below, with the CAD model available [here](https://drive.google.com/file/d/1hdC8xGCHNP47a-wSLPyjr_tpOeynNFEG/view?usp=sharing).
2. Collect data from three scenes, with placement illustrated below, and record them into the corresponding rosbags.
3. Provide the instrinsic matrix in `qr_params.yaml`.
4. Set distance filter in `qr_params.yaml` for board point cloud (extra points are acceptable).
5. Calibrate now!

💡 **Note:** You can run `scripts/distance_filter_tool.py` to quickly obtain suitable filter parameters.
<p align="center">
  <img src="./pics/calibration_target.jpg" width="100%">
  <font color=#a0a0a0 size=2>Left: Actual calibration target | Right: Technical drawing with annotated dimensions.</font>
</p>
<p align="center">
  <img src="./pics/multi-scene.jpg" width="100%">
  <font color=#a0a0a0 size=2>Placement of the calibration target for multi-scene data collection: (a) facing forward, (b) oriented to the right, (c) oriented to the left.</font>
</p>

### 3.1 Jetson rover (Livox Mid-360 + IMX219 pinhole camera)
The rover's camera is `/camera/image_raw` (640x480 pinhole, from gscam) and its LiDAR is `/livox/lidar` (`livox_frame`). The Livox is mounted yawed ~90° relative to the camera: the camera looks along the LiDAR's **-Y** axis, not +X.

**LiDAR frame conventions.** `lidar_frame` tells FAST-Calib which way the board lies in the LiDAR frame. It is used to order the four LiDAR hole centers so they match the camera's circle centers, and by `run_extrinsic_scene.py` to search for the board.

| `lidar_frame` | LiDAR axes | Camera looks along |
|---|---|---|
| `xfwd` (default) | x forward, y left, z up (ROS) | +X |
| `zfwd` | z forward, x right, y up (camera-style) | +Z |
| `nyfwd` | x forward, y left, z up (ROS), LiDAR yawed 90° | -Y (the rover) |

The `nyfwd` option required changes in three files:
- `include/common_lib.h`: `sortPatternCenters()` has a `nyfwd` branch in both the LiDAR->camera axis mapping and the mapping back.
- `scripts/run_extrinsic_scene.py`: `--lidar-frame nyfwd` searches for the board along -Y and maps the distance-filter box back to raw `livox_frame` coordinates.
- `launch/calib.launch.py`: the `lidar_frame` argument help text lists `nyfwd`.

With the wrong frame, `check` fails with `0 plausible holes`, or finds a board but gives a wrong extrinsic.

**1. Intrinsics** (pinhole k1, k2, p1, p2). First check that `marker_size` and `delta_*_qr_center` in `config/qr_params.yaml` match the physical board.
```bash
python3 scripts/calibrate_jetson_cam.py --topic /camera/image_raw --frames 45
```
Move the board over all 9 cells of the coverage grid, corners included. The script saves each view under `calib_data/intrinsic_<timestamp>/`. You can re-solve the saved views with `--solve-only <dir>`. It prints two blocks:
- A camera block for `config/qr_params.yaml` (`fisheye_enable: false`, `fx` … `p2`). Paste it in before the extrinsic step, because the extrinsic step reads it. `fx != fy` is expected: gscam squashes the 1280x720 capture to 640x480, so any change to that pipeline needs a new calibration.
- A FAST-LIVO camera yaml (`cam_model: Pinhole`) to save as `camera_rover.yaml`.

**2. Extrinsics** (three scenes, then a joint solve):
```bash
source /opt/ros/jazzy/setup.bash
source ~/Documents/GitHub/openmind/FAST-Calib/install/setup.bash
cd ~/Documents/GitHub/openmind/FAST-Calib
F="--image-topic /camera/image_raw --lidar-topic /livox/lidar --lidar-frame nyfwd"

python3 scripts/run_extrinsic_scene.py check $F
python3 scripts/run_extrinsic_scene.py 1 $F
python3 scripts/run_extrinsic_scene.py 2 $F     # after moving the board
python3 scripts/run_extrinsic_scene.py 3 $F     # after moving the board
python3 scripts/run_extrinsic_scene.py solve $F
```
- `check` doesn't record anything. Run it after every board move to confirm all four holes are found.
- Keep ~1 m of clearance behind the board and keep the board fully inside the camera image.
- The Mid-360 sees from about -7° to +52° vertically, so keep the board at or above LiDAR height.
- If a scene fails, nothing is recorded, so you can rerun that scene number.

Outputs go to `output/extrinsic_<YYYYMMDD>/`:

| File | Contents |
|---|---|
| `sceneN_raw.png`, `sceneN_qr_detect.png` | raw camera frame and detection overlay for each scene |
| `circle_center_record.txt` | hole-center pairs from all scenes (input to `solve`) |
| `single_calib_result.txt`, `colored_cloud.pcd`, `dbg_*.pcd` | results and clouds from the latest scene |
| `extrinsic_result.txt` | final multi-scene `Rcl` / `Pcl` (written by `solve`) |

The folder is named after the date each command runs on. If a session spans midnight, pass the same `--session <dir>` to every command.

**3. FAST-LIVO config.** Copy `Rcl` / `Pcl` from `extrinsic_result.txt` into the rover's FAST-LIVO config (`rover.yaml`), along with `lid_topic: /livox/lidar`, `imu_topic: /livox/imu` and `img_topic: /camera/image_raw`. The extrinsic is relative to `livox_frame`, which is the frame of `/livox/lidar`.

## 4. Appendix
The calibration target design is based on the [velo2cam_calibration](https://github.com/beltransen/velo2cam_calibration).

For further details on the algorithm workflow, see [this document](https://github.com/xuankuzcr/FAST-Calib/blob/main/workflow.md).
## 5. Acknowledgments

Special thanks to [Jiaming Xu](https://github.com/Xujiaming1) for his support, [Haotian Li](https://github.com/luo-xue) for the equipment, and the [velo2cam_calibration](https://github.com/beltransen/velo2cam_calibration) algorithm.
