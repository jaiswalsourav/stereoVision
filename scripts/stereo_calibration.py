"""Compute stereo rectification maps from chessboard image pairs.

Reads the pairs saved by calibration_images.py and writes
calibration/stereoMap.xml, which stereo_vision/calibration.py loads at runtime.

    python scripts/stereo_calibration.py            # show detected corners
    python scripts/stereo_calibration.py --no-show
"""
import argparse

import cv2 as cv
import numpy as np

import _paths  # noqa: F401  (puts the repo root on sys.path)
from stereo_vision import config

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def find_corners(show=True):
    left_files = sorted(config.CALIB_LEFT_DIR.glob("*.png"))
    right_files = sorted(config.CALIB_RIGHT_DIR.glob("*.png"))
    if not left_files or len(left_files) != len(right_files):
        raise SystemExit(f"Need matching image pairs in {config.CALIB_LEFT_DIR} and "
                         f"{config.CALIB_RIGHT_DIR} (found {len(left_files)} / {len(right_files)})")

    cols, rows = config.CHESSBOARD_SIZE
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * config.CHESSBOARD_SQUARE_CM

    objpoints, imgpoints_l, imgpoints_r = [], [], []
    image_size = None
    for file_l, file_r in zip(left_files, right_files):
        gray_l = cv.imread(str(file_l), cv.IMREAD_GRAYSCALE)
        gray_r = cv.imread(str(file_r), cv.IMREAD_GRAYSCALE)
        if gray_l is None or gray_r is None:
            print(f"skip {file_l.name}: unreadable")
            continue
        image_size = gray_l.shape[::-1]
        ok_l, corners_l = cv.findChessboardCorners(gray_l, config.CHESSBOARD_SIZE, None)
        ok_r, corners_r = cv.findChessboardCorners(gray_r, config.CHESSBOARD_SIZE, None)
        if not (ok_l and ok_r):
            print(f"skip {file_l.name}: board not found in both images")
            continue

        objpoints.append(objp)
        imgpoints_l.append(cv.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), CRITERIA))
        imgpoints_r.append(cv.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), CRITERIA))

        if show:
            view_l = cv.cvtColor(gray_l, cv.COLOR_GRAY2BGR)
            view_r = cv.cvtColor(gray_r, cv.COLOR_GRAY2BGR)
            cv.drawChessboardCorners(view_l, config.CHESSBOARD_SIZE, corners_l, True)
            cv.drawChessboardCorners(view_r, config.CHESSBOARD_SIZE, corners_r, True)
            cv.imshow("corners (left | right)", np.hstack([view_l, view_r]))
            cv.waitKey(300)
    cv.destroyAllWindows()

    if len(objpoints) < 5:
        raise SystemExit(f"Only {len(objpoints)} usable pairs; capture at least 10.")
    print(f"using {len(objpoints)} of {len(left_files)} pairs")
    return objpoints, imgpoints_l, imgpoints_r, image_size


def calibrate(objpoints, imgpoints_l, imgpoints_r, image_size):
    err_l, mtx_l, dist_l, _, _ = cv.calibrateCamera(objpoints, imgpoints_l, image_size, None, None)
    err_r, mtx_r, dist_r, _, _ = cv.calibrateCamera(objpoints, imgpoints_r, image_size, None, None)
    print(f"reprojection error  left: {err_l:.3f}px  right: {err_r:.3f}px")

    err, mtx_l, dist_l, mtx_r, dist_r, rot, trans, _, _ = cv.stereoCalibrate(
        objpoints, imgpoints_l, imgpoints_r, mtx_l, dist_l, mtx_r, dist_r,
        image_size, criteria=CRITERIA, flags=cv.CALIB_FIX_INTRINSIC)
    print(f"stereo reprojection error: {err:.3f}px (below ~0.5px is good)")
    print(f"measured baseline: {np.linalg.norm(trans):.2f} cm "
          f"(config BASELINE_CM = {config.BASELINE_CM})")

    rect_l, rect_r, proj_l, proj_r, q, _, _ = cv.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, image_size, rot, trans, alpha=1)
    map_l = cv.initUndistortRectifyMap(mtx_l, dist_l, rect_l, proj_l, image_size, cv.CV_16SC2)
    map_r = cv.initUndistortRectifyMap(mtx_r, dist_r, rect_r, proj_r, image_size, cv.CV_16SC2)
    return map_l, map_r, q


def save(map_l, map_r, q, path=config.STEREO_MAP_FILE):
    fs = cv.FileStorage(str(path), cv.FILE_STORAGE_WRITE)
    fs.write("stereoMapL_x", map_l[0])
    fs.write("stereoMapL_y", map_l[1])
    fs.write("stereoMapR_x", map_r[0])
    fs.write("stereoMapR_y", map_r[1])
    fs.write("Q", q)
    fs.release()
    print(f"saved {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-show", action="store_true", help="do not display corners")
    args = parser.parse_args()
    save(*calibrate(*find_corners(show=not args.no_show)))


if __name__ == "__main__":
    main()
