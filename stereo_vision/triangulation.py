"""Depth and 3D position from a matched point in rectified stereo frames.

Camera axes follow the OpenCV convention for the left camera:
x to the right, y down, z forward (away from the camera). Units follow the
baseline (cm).
"""
import numpy as np

from . import config


def focal_length_px(frame_width, fov_deg=config.FOV_H_DEG):
    """Focal length in pixels from the horizontal field of view."""
    return (frame_width * 0.5) / np.tan(np.radians(fov_deg) * 0.5)


def find_depth(right_point, left_point, frame_right, frame_left,
               baseline=config.BASELINE_CM, f=None, alpha=config.FOV_H_DEG):
    """Depth (same units as baseline) of a point seen in both frames.

    ``f`` (the lens focal length in mm) is accepted for backward
    compatibility but not needed: the pixel focal length comes from ``alpha``.
    Returns None when the disparity is zero or negative, which means the two
    points cannot belong to the same object in front of the cameras.
    """
    width_r = frame_right.shape[1]
    width_l = frame_left.shape[1]
    if width_r != width_l:
        raise ValueError(f"Frame widths differ: right={width_r}, left={width_l}")

    disparity = left_point[0] - right_point[0]
    if disparity <= 0:
        return None
    return baseline * focal_length_px(width_l, alpha) / disparity


def point_3d(left_point, right_point, frame_shape,
             baseline=config.BASELINE_CM, fov_deg=config.FOV_H_DEG):
    """3D position (x, y, z) of a matched point, relative to the left camera.

    Returns None when the disparity is zero or negative.
    """
    height, width = frame_shape[:2]
    disparity = left_point[0] - right_point[0]
    if disparity <= 0:
        return None
    f_px = focal_length_px(width, fov_deg)
    z = baseline * f_px / disparity
    cx, cy = width / 2.0, height / 2.0
    x = (left_point[0] - cx) * z / f_px
    y = (left_point[1] - cy) * z / f_px
    return np.array([x, y, z])
