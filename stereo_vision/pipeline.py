"""Per-frame processing shared by the live scripts."""
import cv2
import numpy as np

from .detection import draw_detection, match_detections
from .triangulation import point_3d


def process_frame(detector, frame_l, frame_r, objects=None):
    """Detect, match and triangulate. Returns [(left_det, right_det, xyz)]."""
    dets_l = detector.detect(frame_l, objects)
    dets_r = detector.detect(frame_r, objects)
    results = []
    for dl, dr in match_detections(dets_l, dets_r):
        xyz = point_3d(dl.center, dr.center, frame_l.shape)
        if xyz is not None:
            results.append((dl, dr, xyz))
    return results


def annotate(frame_l, frame_r, results, fps):
    """Draw boxes, distances and FPS; return the side-by-side view."""
    for dl, dr, xyz in results:
        label = f"{dl.class_name} {xyz[2]:.1f} cm"
        draw_detection(frame_l, dl, label)
        draw_detection(frame_r, dr, label)
    if not results:
        for frame in (frame_l, frame_r):
            cv2.putText(frame, "TRACKING LOST", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    for frame in (frame_l, frame_r):
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    return np.hstack([frame_l, frame_r])
