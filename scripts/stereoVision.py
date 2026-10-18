"""Live stereo depth estimation of detected objects.

Grabs frames from both cameras, rectifies them, detects objects with
SSD MobileNet v3, pairs each object across the two views and shows its
distance. Press "q" or Esc to quit.

    python scripts/stereoVision.py                    # all COCO classes
    python scripts/stereoVision.py --objects bottle cup
"""
import argparse
import time

import cv2

import _paths  # noqa: F401  (puts the repo root on sys.path)
from stereo_vision import config, rectification
from stereo_vision.cameras import StereoCamera
from stereo_vision.detection import ObjectDetector
from stereo_vision.pipeline import annotate, process_frame


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="*", default=None,
                        help="only track these COCO class names")
    parser.add_argument("--left", default=config.LEFT_CAMERA, help="left camera URL or index")
    parser.add_argument("--right", default=config.RIGHT_CAMERA, help="right camera URL or index")
    args = parser.parse_args()

    detector = ObjectDetector()
    cams = StereoCamera(args.left, args.right)
    rectification.load_maps()  # fail early if calibration is missing

    try:
        while True:
            start = time.time()
            frame_l, frame_r = cams.read()
            if frame_l is None:
                if cv2.waitKey(500) & 0xFF in (ord("q"), 27):
                    break
                continue

            frame_l, frame_r = rectification.rectify(frame_l, frame_r)
            results = process_frame(detector, frame_l, frame_r, args.objects)
            for dl, _, xyz in results:
                print(f"{dl.class_name}: x={xyz[0]:.1f} y={xyz[1]:.1f} z={xyz[2]:.1f} cm")

            fps = 1.0 / max(time.time() - start, 1e-6)
            cv2.imshow("stereo (left | right)", annotate(frame_l, frame_r, results, fps))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cams.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
