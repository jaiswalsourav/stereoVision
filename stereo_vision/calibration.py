"""Undistort and rectify stereo frames using the maps in stereoMap.xml."""
import cv2

from . import config

_maps = None


def load_maps(path=config.STEREO_MAP_FILE):
    """Load the rectification maps written by stereo_calibration.py."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run scripts/calibration_images.py and "
            "scripts/stereo_calibration.py first."
        )
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    maps = {name: fs.getNode(name).mat() for name in
            ("stereoMapL_x", "stereoMapL_y", "stereoMapR_x", "stereoMapR_y")}
    fs.release()
    missing = [name for name, m in maps.items() if m is None]
    if missing:
        raise ValueError(f"{path} is missing {', '.join(missing)}")
    return maps


def rectify(frame_left, frame_right):
    """Return the undistorted, rectified (left, right) frames."""
    global _maps
    if _maps is None:
        _maps = load_maps()
    left = cv2.remap(frame_left, _maps["stereoMapL_x"], _maps["stereoMapL_y"],
                     cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    right = cv2.remap(frame_right, _maps["stereoMapR_x"], _maps["stereoMapR_y"],
                      cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    return left, right


def undistortRectify(frameR, frameL):
    """Backward-compatible wrapper: takes and returns (right, left)."""
    left, right = rectify(frameL, frameR)
    return right, left
