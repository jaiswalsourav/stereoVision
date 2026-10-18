"""Locate an object with the stereo cameras and move the robot arm to it.

The arm only moves once the object has been seen at a steady position for
several frames, then returns home and waits for a cooldown. Press "q" or Esc
to quit.

    python scripts/stereoVisionsecure.py                  # track bottles
    python scripts/stereoVisionsecure.py --object cup --dry-run
"""
import argparse
import time

import cv2
import numpy as np

import _paths  # noqa: F401  (puts the repo root on sys.path)
from stereo_vision import calibration
from stereo_vision.arm import Arm, UnreachableError
from stereo_vision.cameras import StereoCamera
from stereo_vision.detection import ObjectDetector
from stereo_vision.pipeline import annotate, process_frame


class StableTarget:
    """Accepts a target once it has stayed within `tolerance` for `frames` frames."""

    def __init__(self, frames=5, tolerance_cm=2.0):
        self.frames = frames
        self.tolerance = tolerance_cm
        self.history = []

    def update(self, xyz):
        if xyz is None:
            self.history.clear()
            return None
        if self.history and np.linalg.norm(xyz - self.history[-1]) > self.tolerance:
            self.history.clear()
        self.history.append(xyz)
        if len(self.history) >= self.frames:
            target = np.mean(self.history, axis=0)
            self.history.clear()
            return target
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", default="bottle", help="COCO class to pick")
    parser.add_argument("--dry-run", action="store_true", help="print servo moves only")
    parser.add_argument("--hold", type=float, default=2.0, help="seconds to stay at the target")
    parser.add_argument("--cooldown", type=float, default=3.0, help="seconds before the next move")
    args = parser.parse_args()

    detector = ObjectDetector()
    cams = StereoCamera()
    calibration.load_maps()
    arm = Arm(dry_run=args.dry_run)
    arm.home()
    tracker = StableTarget()
    next_move = 0.0

    try:
        while True:
            start = time.time()
            frame_l, frame_r = cams.read()
            if frame_l is None:
                if cv2.waitKey(500) & 0xFF in (ord("q"), 27):
                    break
                continue

            frame_l, frame_r = calibration.rectify(frame_l, frame_r)
            results = process_frame(detector, frame_l, frame_r, [args.object])
            # Closest matched object is the target.
            xyz = min((r[2] for r in results), key=lambda p: p[2]) if results else None
            target = tracker.update(xyz)

            fps = 1.0 / max(time.time() - start, 1e-6)
            cv2.imshow("stereo (left | right)", annotate(frame_l, frame_r, results, fps))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            if target is not None and time.time() >= next_move:
                print(f"target at x={target[0]:.1f} y={target[1]:.1f} z={target[2]:.1f} cm")
                try:
                    angles = arm.reach(target)
                    print("reached:", {k: round(v, 1) for k, v in angles.items()})
                    time.sleep(args.hold)
                    arm.home()
                except UnreachableError as exc:
                    print(f"cannot reach target: {exc}")
                next_move = time.time() + args.cooldown
    finally:
        arm.home()
        cams.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
