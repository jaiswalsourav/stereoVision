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
ARM_L2_CM = _env("ARM_L2_CM", 17.0, float)  # elbow -> gripper
SERVO_BASE = _env("SERVO_BASE", 0, int)
SERVO_SHOULDER = _env("SERVO_SHOULDER", 1, int)
SERVO_ELBOW = _env("SERVO_ELBOW", 3, int)
# Position of the arm's base relative to the left camera, in camera axes
# (x right, y down, z forward), in cm. Measure this on the real rig.
ARM_OFFSET_CM = (
    _env("ARM_OFFSET_X_CM", 0.0, float),
    _env("ARM_OFFSET_Y_CM", 0.0, float),
    _env("ARM_OFFSET_Z_CM", 0.0, float),
)
# Servo speed: degrees per step and delay per step (smaller = gentler).
SERVO_STEP_DEG = _env("SERVO_STEP_DEG", 2.0, float)
SERVO_STEP_DELAY_S = _env("SERVO_STEP_DELAY_S", 0.02, float)

# --- Gripper and fruit picking -------------------------------------------------------
SERVO_GRIPPER = _env("SERVO_GRIPPER", 2, int)
GRIPPER_OPEN_DEG = _env("GRIPPER_OPEN_DEG", 30.0, float)
GRIPPER_CLOSED_DEG = _env("GRIPPER_CLOSED_DEG", 110.0, float)
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
