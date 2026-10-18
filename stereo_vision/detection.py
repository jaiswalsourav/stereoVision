"""SSD MobileNet v3 object detection and left/right matching."""
from dataclasses import dataclass

import cv2

from . import config


@dataclass
class Detection:
    class_name: str
    confidence: float
    box: tuple  # (x, y, w, h) in pixels

    @property
    def center(self):
        x, y, w, h = self.box
        return (x + w / 2.0, y + h / 2.0)


class ObjectDetector:
    def __init__(self, conf_threshold=config.CONF_THRESHOLD,
                 nms_threshold=config.NMS_THRESHOLD):
        with open(config.CLASS_FILE, "rt") as fh:
            self.class_names = fh.read().rstrip("\n").split("\n")
        self.net = cv2.dnn_DetectionModel(str(config.MODEL_WEIGHTS),
                                          str(config.MODEL_CONFIG))
        self.net.setInputSize(320, 320)
        self.net.setInputScale(1.0 / 127.5)
        self.net.setInputMean((127.5, 127.5, 127.5))
        self.net.setInputSwapRB(True)  # frames are BGR, the model wants RGB
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

    def detect(self, frame_bgr, objects=None):
        """Detect objects in a BGR frame.

        ``objects`` optionally restricts results to these class names.
        """
        class_ids, confs, boxes = self.net.detect(
            frame_bgr, confThreshold=self.conf_threshold,
            nmsThreshold=self.nms_threshold)
        detections = []
        if len(class_ids) == 0:
            return detections
        for class_id, conf, box in zip(class_ids.flatten(), confs.flatten(), boxes):
            index = int(class_id) - 1
            if not 0 <= index < len(self.class_names):
                continue
            name = self.class_names[index]
            if objects and name not in objects:
                continue
            detections.append(Detection(name, float(conf), tuple(int(v) for v in box)))
        return detections


def match_detections(left, right, max_row_diff=config.MAX_ROW_DIFF_PX):
    """Pair detections of the same object across the rectified views.

    After rectification the same object lies on (almost) the same image row
    in both frames and appears further right in the left frame. Pairs are
    chosen greedily by row difference and box-size similarity.
    """
    candidates = []
    for i, dl in enumerate(left):
        for j, dr in enumerate(right):
            if dl.class_name != dr.class_name:
                continue
            (xl, yl), (xr, yr) = dl.center, dr.center
            row_diff = abs(yl - yr)
            if row_diff > max_row_diff or xl - xr <= 0:
                continue
            size_diff = abs(dl.box[2] - dr.box[2]) + abs(dl.box[3] - dr.box[3])
            candidates.append((row_diff + 0.5 * size_diff, i, j))

    pairs, used_l, used_r = [], set(), set()
    for _, i, j in sorted(candidates):
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        pairs.append((left[i], right[j]))
    return pairs


def draw_detection(frame, det, label=None, color=(0, 255, 0)):
    x, y, w, h = det.box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    text = label or f"{det.class_name.upper()} {det.confidence * 100:.0f}%"
    cv2.putText(frame, text, (x + 5, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
