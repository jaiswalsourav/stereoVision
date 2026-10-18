"""Capture chessboard image pairs for stereo calibration.

Show a printed 9x6 chessboard to both cameras. Press "s" to save a pair
(only saved when the board is visible in both views), Esc or "q" to quit.
Aim for 20-30 pairs covering different positions, angles and distances.
Images go to stereo_calibration/images/left and images/right.

    python stereo_calibration/capture.py
"""
import argparse

import cv2
import numpy as np

import _paths  # noqa: F401  (puts the repo root on sys.path)
from stereo_vision import config
from stereo_vision.cameras import StereoCamera


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--left", default=config.LEFT_CAMERA, help="left camera URL or index")
    parser.add_argument("--right", default=config.RIGHT_CAMERA, help="right camera URL or index")
    args = parser.parse_args()

    config.CALIB_LEFT_DIR.mkdir(parents=True, exist_ok=True)
    config.CALIB_RIGHT_DIR.mkdir(parents=True, exist_ok=True)
    num = len(list(config.CALIB_LEFT_DIR.glob("*.png")))
    cams = StereoCamera(args.left, args.right)

    try:
        while True:
            frame_l, frame_r = cams.read()
            key = cv2.waitKey(5) & 0xFF
            if key in (27, ord("q")):
                break
            if frame_l is None:
                continue

            found = []
            preview = []
            for frame in (frame_l, frame_r):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ok, corners = cv2.findChessboardCorners(
                    gray, config.CHESSBOARD_SIZE, cv2.CALIB_CB_FAST_CHECK)
                found.append(ok)
                view = frame.copy()
                cv2.drawChessboardCorners(view, config.CHESSBOARD_SIZE, corners, ok)
                preview.append(view)

            if key == ord("s"):
                if all(found):
                    cv2.imwrite(str(config.CALIB_LEFT_DIR / f"imageL{num:03d}.png"), frame_l)
                    cv2.imwrite(str(config.CALIB_RIGHT_DIR / f"imageR{num:03d}.png"), frame_r)
                    num += 1
                    print(f"pair {num} saved")
                else:
                    print("chessboard not visible in both cameras, not saved")

            status = f"pairs: {num}  board L:{'ok' if found[0] else '--'} R:{'ok' if found[1] else '--'}"
            canvas = np.hstack(preview)
            cv2.putText(canvas, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("calibration (left | right)", canvas)
    finally:
        cams.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
