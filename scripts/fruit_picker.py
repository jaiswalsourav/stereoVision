"""Detect fruit with the stereo cameras and pluck it with the robot arm.

Loop: find fruit in both views, pick the closest one the arm can reach,
wait until it is steady for a few frames, then approach, grip, twist,
pull it off and drop it in the basket. The camera view stays live while the
arm works. Press "q" or Esc to stop the arm and quit.

    python scripts/fruit_picker.py --dry-run              # print moves only
    python scripts/fruit_picker.py --fruits apple orange  # only these
    python scripts/fruit_picker.py --max-picks 5

Fruits the current COCO model knows: apple, orange, banana.
"""
import argparse
import time

import cv2

import _paths  # noqa: F401  (puts the repo root on sys.path)
from stereo_vision import config, rectification
from stereo_vision.arm import Arm, ArmStopped, UnreachableError
from stereo_vision.cameras import StereoCamera
from stereo_vision.detection import ObjectDetector
from stereo_vision.pipeline import annotate, process_frame
from stereo_vision.tracking import StableTarget


def choose_fruit(results, arm):
    """Closest fruit whose whole pluck motion is possible, or None."""
    for det, _, xyz in sorted(results, key=lambda r: r[2][2]):
        try:
            arm.plan(xyz)
            return det, xyz
        except UnreachableError:
            continue
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fruits", nargs="+", default=config.FRUIT_CLASSES,
                        help="fruit classes to pick (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="print servo moves only")
    parser.add_argument("--max-picks", type=int, default=0, help="stop after N fruits (0 = no limit)")
    parser.add_argument("--stable-frames", type=int, default=5,
                        help="frames a fruit must stay still before picking")
    args = parser.parse_args()

    unknown = set(args.fruits) - set(config.FRUIT_CLASSES)
    if unknown:
        print(f"warning: the COCO model does not know {sorted(unknown)}; "
              f"it detects {config.FRUIT_CLASSES}")

    detector = ObjectDetector()
    cams = StereoCamera()
    rectification.load_maps()
    arm = Arm(dry_run=args.dry_run)
    arm.home()
    tracker = StableTarget(frames=args.stable_frames)
    picked, job, job_name = 0, None, None

    try:
        while True:
            start = time.time()

            # Report a finished pick.
            if job is not None and job.done():
                if job.error is None:
                    picked += 1
                    print(f"{job_name} picked ({picked} total)")
                elif isinstance(job.error, (UnreachableError, ArmStopped)):
                    print(f"{job_name} skipped: {job.error}")
                else:
                    raise job.error
                job = None
                tracker.reset()
                if args.max_picks and picked >= args.max_picks:
                    print("pick limit reached")
                    break

            frame_l, frame_r = cams.read()
            if frame_l is None:
                if cv2.waitKey(500) & 0xFF in (ord("q"), 27):
                    break
                continue

            frame_l, frame_r = rectification.rectify(frame_l, frame_r)
            results = process_frame(detector, frame_l, frame_r, args.fruits)

            target = None
            if not arm.busy:
                choice = choose_fruit(results, arm)
                target = tracker.update(choice[1] if choice else None)

            fps = 1.0 / max(time.time() - start, 1e-6)
            view = annotate(frame_l, frame_r, results, fps)
            status = f"picked: {picked}" + ("  ARM BUSY" if arm.busy else "")
            cv2.putText(view, status, (view.shape[1] - 360, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imshow("fruit picker (left | right)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

            if target is not None:
                job_name = choice[0].class_name
                print(f"plucking {job_name} at x={target[0]:.1f} y={target[1]:.1f} "
                      f"z={target[2]:.1f} cm")
                job = arm.start(arm.pluck, target)
    finally:
        arm.shutdown()
        cams.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
