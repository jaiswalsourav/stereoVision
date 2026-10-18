"""Shared settings for the stereo vision project.

Every value can be overridden with an environment variable of the same name,
so the same code runs on the Raspberry Pi and on a desktop without edits.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # repository root
MODELS_DIR = BASE_DIR / "models"
CALIBRATION_DIR = BASE_DIR / "stereo_calibration"


def _env(name, default, cast=str):
    value = os.environ.get(name)
    return cast(value) if value is not None else default


# --- Cameras -----------------------------------------------------------------
# Snapshot URLs of the two network cameras (e.g. ESP32-CAM). A plain integer
# such as "0" is treated as a local USB camera index instead.
LEFT_CAMERA = _env("LEFT_CAMERA", "http://192.168.220.216/640x480.jpg")
RIGHT_CAMERA = _env("RIGHT_CAMERA", "http://192.168.220.24/640x480.jpg")
CAMERA_TIMEOUT_S = _env("CAMERA_TIMEOUT_S", 3.0, float)
FRAME_SIZE = (640, 480)  # (width, height) used for calibration

# --- Stereo geometry ------------------------------------------------------------
BASELINE_CM = _env("BASELINE_CM", 3.6, float)   # distance between the cameras
FOV_H_DEG = _env("FOV_H_DEG", 65.0, float)      # horizontal field of view

# --- Files -------------------------------------------------------------------------
# Stereo calibration (see stereo_calibration/README.md). Override the folder with
# CALIBRATION_OUT to keep several calibrations side by side.
CALIB_OUT_DIR = Path(_env("CALIBRATION_OUT", str(CALIBRATION_DIR)))
STEREO_MAP_FILE = Path(_env("STEREO_MAP_FILE", str(CALIB_OUT_DIR / "stereoMap.xml")))
LEFT_INTRINSICS_FILE = CALIB_OUT_DIR / "left_camera.yml"
RIGHT_INTRINSICS_FILE = CALIB_OUT_DIR / "right_camera.yml"
CALIB_DIR = Path(_env("CALIB_DIR", str(CALIB_OUT_DIR / "images")))
CALIB_LEFT_DIR = CALIB_DIR / "left"
CALIB_RIGHT_DIR = CALIB_DIR / "right"
CHESSBOARD_SIZE = (9, 6)  # inner corners per row, per column
CHESSBOARD_SQUARE_CM = _env("CHESSBOARD_SQUARE_CM", 2.5, float)

# --- Object detector -------------------------------------------------------------
CLASS_FILE = MODELS_DIR / "coco.names"
MODEL_CONFIG = MODELS_DIR / "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
MODEL_WEIGHTS = MODELS_DIR / "frozen_inference_graph.pb"
CONF_THRESHOLD = _env("CONF_THRESHOLD", 0.5, float)
NMS_THRESHOLD = _env("NMS_THRESHOLD", 0.3, float)
# Max vertical offset (pixels) between the same object in the two rectified views.
MAX_ROW_DIFF_PX = _env("MAX_ROW_DIFF_PX", 40.0, float)

# --- Robot arm -----------------------------------------------------------------------
ARM_L1_CM = _env("ARM_L1_CM", 25.0, float)  # shoulder -> elbow
ARM_L2_CM = _env("ARM_L2_CM", 17.0, float)  # elbow -> gripper tip

# Each joint is described in "logical" degrees (see stereo_vision/servos.py for
# the conventions). Per joint, these environment variables override defaults
# (JOINT is BASE, SHOULDER, ELBOW or GRIPPER):
#   SERVO_<JOINT>          PCA9685 channel
#   <JOINT>_MIN_DEG / _MAX_DEG   allowed logical range (safety limits)
#   <JOINT>_HOME_DEG       home position
#   <JOINT>_OFFSET_DEG     trim added to the servo command (fixes a horn mounted off by a few degrees)
#   <JOINT>_INVERTED=1     the servo is mounted mirrored (180 - angle)
GRIPPER_OPEN_DEG = _env("GRIPPER_OPEN_DEG", 30.0, float)
GRIPPER_CLOSED_DEG = _env("GRIPPER_CLOSED_DEG", 110.0, float)


def _joint(name, channel, home, lo=0.0, hi=180.0):
    key = name.upper()
    return {
        "channel": _env(f"SERVO_{key}", channel, int),
        "min_deg": _env(f"{key}_MIN_DEG", lo, float),
        "max_deg": _env(f"{key}_MAX_DEG", hi, float),
        "home_deg": _env(f"{key}_HOME_DEG", home, float),
        "offset_deg": _env(f"{key}_OFFSET_DEG", 0.0, float),
        "inverted": _env(f"{key}_INVERTED", "0") in ("1", "true", "yes"),
    }


JOINTS = {
    "base": _joint("base", 0, 90.0),
    "shoulder": _joint("shoulder", 1, 90.0),
    "elbow": _joint("elbow", 3, 90.0),
    "gripper": _joint("gripper", 2, GRIPPER_OPEN_DEG),
}
# Pulse range of the servos in microseconds (check the servo's datasheet;
# many hobby servos need 500..2500 to cover the full 180 degrees).
SERVO_MIN_PULSE_US = _env("SERVO_MIN_PULSE_US", 750, int)
SERVO_MAX_PULSE_US = _env("SERVO_MAX_PULSE_US", 2250, int)

# Motion: joint speed for normal moves, tip speed for straight-line moves,
# and how often servo positions are updated.
ARM_SPEED_DEG_S = _env("ARM_SPEED_DEG_S", 60.0, float)
ARM_LINEAR_SPEED_CM_S = _env("ARM_LINEAR_SPEED_CM_S", 5.0, float)
ARM_UPDATE_HZ = _env("ARM_UPDATE_HZ", 50.0, float)

# Camera -> arm transform. If ARM_TRANSFORM_FILE exists (written by the
# "fit" command of scripts/arm_control.py) it is used. Otherwise the arm's
# axes are assumed parallel to the left camera's (x right, y down, z forward)
# with the shoulder joint at ARM_OFFSET_*_CM from the camera.
ARM_TRANSFORM_FILE = Path(_env("ARM_TRANSFORM_FILE", str(CALIBRATION_DIR / "arm_transform.yml")))
ARM_OFFSET_CM = (
    _env("ARM_OFFSET_X_CM", 0.0, float),
    _env("ARM_OFFSET_Y_CM", 0.0, float),
    _env("ARM_OFFSET_Z_CM", 0.0, float),
)

# --- Fruit picking -----------------------------------------------------------------
# Fruits the COCO model can detect. A custom model is needed for others.
FRUIT_CLASSES = ["apple", "orange", "banana"]
APPROACH_CM = _env("APPROACH_CM", 6.0, float)   # stop this far before the fruit first
PULL_BACK_CM = _env("PULL_BACK_CM", 8.0, float)  # pull back this far to detach it
PULL_DOWN_CM = _env("PULL_DOWN_CM", 3.0, float)  # and this far down
TWIST_DEG = _env("TWIST_DEG", 10.0, float)       # base wiggle while gripping
TWIST_COUNT = _env("TWIST_COUNT", 2, int)
# Servo angles (base, shoulder, elbow) above the basket where fruit is dropped.
DROP_POSE = {
    "base": _env("DROP_BASE_DEG", 170.0, float),
    "shoulder": _env("DROP_SHOULDER_DEG", 90.0, float),
    "elbow": _env("DROP_ELBOW_DEG", 90.0, float),
}
