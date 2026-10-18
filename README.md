# Stereo Vision Depth Estimation + Robot Arm

Two cameras locate an object (any of the 90 COCO classes, e.g. a bottle) in
3D, and a 3-servo arm on a Raspberry Pi moves to it.

## Project layout

```
stereoVision/
├── scripts/                  programs you run
│   ├── calibration_images.py     1. capture chessboard pairs
│   ├── stereo_calibration.py     2. compute calibration/stereoMap.xml
│   ├── stereoVision.py           3. live object distance
│   └── stereoVisionsecure.py     4. distance + move the robot arm
├── stereo_vision/            library code used by the scripts
│   ├── config.py                 all settings
│   ├── cameras.py                HTTP or USB cameras
│   ├── calibration.py            rectification
│   ├── detection.py              SSD MobileNet v3 + left/right matching
│   ├── triangulation.py          depth and 3D position
│   ├── pipeline.py               detect -> match -> triangulate -> draw
│   └── arm.py                    inverse kinematics and servos
├── models/                   detector files (+ unused best.tflite)
├── calibration/              stereoMap.xml and captured images/
├── tests/                    unit tests, no hardware needed
└── docs/                     screenshots
```

## Pipeline

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `calibration_images.py` | Live view of both cameras; press `s` to save a chessboard pair to `calibration/images/` (only saved when the board is seen by both). |
| 2 | `stereo_calibration.py` | Calibrates both cameras from the saved pairs and writes `calibration/stereoMap.xml`. Prints the reprojection error and the measured baseline. |
| 3 | `stereoVision.py` | Rectifies frames, detects objects, pairs them across views and shows each object's distance. |
| 4 | `stereoVisionsecure.py` | Same as 3, then moves the arm to the closest target once it has been steady for 5 frames. |

## Setup

```bash
pip install -r requirements.txt
# on the Raspberry Pi, for the servo board:
pip install adafruit-circuitpython-servokit
```

## Usage

```bash
Run everything from the repository root:

```bash
python scripts/calibration_images.py                    # capture 15-30 chessboard pairs
python scripts/stereo_calibration.py                    # writes calibration/stereoMap.xml
python scripts/stereoVision.py --objects bottle cup     # depth of bottles and cups
python scripts/stereoVisionsecure.py --object bottle --dry-run   # arm, printing moves only
python -m unittest discover tests                       # tests, no hardware needed
```

Press `q` or `Esc` to quit any live window.

## Configuration

All settings live in `stereo_vision/config.py`, and each can be overridden with an
environment variable of the same name, for example:

```bash
LEFT_CAMERA=http://192.168.1.10/640x480.jpg RIGHT_CAMERA=http://192.168.1.11/640x480.jpg python scripts/stereoVision.py
LEFT_CAMERA=0 RIGHT_CAMERA=2 python scripts/stereoVision.py      # USB cameras
```

Values to check on your rig:

- `BASELINE_CM`: distance between camera centres. `scripts/stereo_calibration.py` prints the measured value when `CHESSBOARD_SQUARE_CM` is set to the real square size.
- `FOV_H_DEG`: horizontal field of view of the cameras.
- `ARM_OFFSET_X/Y/Z_CM`: position of the arm's shoulder joint relative to the left camera (x right, y down, z forward). The arm moves to the wrong place until this is measured.
- `ARM_L1_CM`, `ARM_L2_CM`, `SERVO_*`: link lengths and PCA9685 channels.

The servo angle conventions are documented at the top of
`stereo_vision/arm.py`. If a servo is mounted mirrored, adjust its angle there.

`calibration/stereoMap.xml` belongs to the current camera pair. Recalibrate
whenever the cameras move. `models/best.tflite` is a separate model that the
scripts do not use.
