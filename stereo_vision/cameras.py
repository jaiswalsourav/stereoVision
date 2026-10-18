"""Frame sources for the left and right cameras.

A source is either an HTTP snapshot URL (ESP32-CAM style) or a USB camera
index. Both are wrapped behind the same ``read()`` call.
"""
import urllib.request
from urllib.error import URLError

import cv2
import numpy as np

from . import config


class Camera:
    def __init__(self, source, timeout=config.CAMERA_TIMEOUT_S):
        self.source = str(source)
        self.timeout = timeout
        self._cap = None
        if self.source.isdigit():
            self._cap = cv2.VideoCapture(int(self.source))
            if not self._cap.isOpened():
                raise RuntimeError(f"Cannot open USB camera {self.source}")

    def read(self):
        """Return a BGR frame, or None if the camera did not deliver one."""
        if self._cap is not None:
            ok, frame = self._cap.read()
            return frame if ok else None
        try:
            with urllib.request.urlopen(self.source, timeout=self.timeout) as resp:
                data = np.frombuffer(resp.read(), dtype=np.uint8)
        except (URLError, TimeoutError, OSError) as exc:
            print(f"[camera] {self.source}: {exc}")
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)

    def release(self):
        if self._cap is not None:
            self._cap.release()


class StereoCamera:
    def __init__(self, left=config.LEFT_CAMERA, right=config.RIGHT_CAMERA):
        self.left = Camera(left)
        self.right = Camera(right)

    def read(self):
        """Return (left, right) frames, or (None, None) if either failed."""
        frame_l = self.left.read()
        frame_r = self.right.read()
        if frame_l is None or frame_r is None:
            return None, None
        return frame_l, frame_r

    def release(self):
        self.left.release()
        self.right.release()
