"""Stereo camera calibration, in two separate stages.

Stage 1, mono: calibrate each camera on its own from every image where it
sees the chessboard. Writes left_camera.yml and right_camera.yml.

Stage 2, stereo: load those two files, use the image pairs where both
cameras see the board, and compute the rotation/translation between the
cameras plus the rectification maps. Writes stereoMap.xml, which the main
programs load.

    python stereo_calibration/calibrate.py                 # both stages
    python stereo_calibration/calibrate.py --stage mono    # cameras only
    python stereo_calibration/calibrate.py --stage stereo  # reuse saved camera files
    python stereo_calibration/calibrate.py --show          # display detected corners
"""
import argparse
import re

import cv2 as cv
import numpy as np

import _paths  # noqa: F401  (puts the repo root on sys.path)
from stereo_vision import config

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
GOOD_RMS_PX = 0.5


def board_points():
    cols, rows = config.CHESSBOARD_SIZE
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * config.CHESSBOARD_SQUARE_CM
    return objp


def find_corners(path, show=False):
    """Sub-pixel chessboard corners in one image, or None."""
    gray = cv.imread(str(path), cv.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  {path.name}: unreadable")
        return None, None
    ok, corners = cv.findChessboardCorners(
        gray, config.CHESSBOARD_SIZE, cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None, gray.shape[::-1]
    corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    if show:
        view = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
        cv.drawChessboardCorners(view, config.CHESSBOARD_SIZE, corners, True)
        cv.imshow("corners", view)
        cv.waitKey(200)
    return corners, gray.shape[::-1]


def rms_px(projected, measured):
    """RMS distance between two point sets of any OpenCV shape ((N,2) or (N,1,2))."""
    diff = np.reshape(projected, (-1, 2)) - np.reshape(measured, (-1, 2))
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def image_number(path):
    """Pair images by the number in their name: imageL007 <-> imageR007."""
    match = re.search(r"(\d+)$", path.stem)
    return int(match.group(1)) if match else path.stem


def load_images(folder):
    files = sorted(folder.glob("*.png")) + sorted(folder.glob("*.jpg"))
    return {image_number(p): p for p in files}


# --- Stage 1: each camera separately ---------------------------------------------------

def calibrate_mono(side, folder, out_file, show=False):
    images = load_images(folder)
    if not images:
        raise SystemExit(f"No images in {folder}. Capture some with capture.py first.")
    objp = board_points()
    objpoints, imgpoints, names, size = [], [], [], None
    for _, path in sorted(images.items()):
        corners, img_size = find_corners(path, show)
        if img_size is not None:
            if size is not None and img_size != size:
                raise SystemExit(f"{path.name} is {img_size}, others are {size}")
            size = img_size
        if corners is not None:
            objpoints.append(objp)
            imgpoints.append(corners)
            names.append(path.name)
    if len(objpoints) < 5:
        raise SystemExit(f"{side}: board found in only {len(objpoints)} images; need 10+.")

    rms, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, size, None, None)
    per_view = [rms_px(cv.projectPoints(o, rv, tv, mtx, dist)[0], i)
                for o, i, rv, tv in zip(objpoints, imgpoints, rvecs, tvecs)]

    fs = cv.FileStorage(str(out_file), cv.FILE_STORAGE_WRITE)
    fs.write("image_width", size[0])
    fs.write("image_height", size[1])
    fs.write("camera_matrix", mtx)
    fs.write("dist_coeffs", dist)
    fs.write("rms_error_px", rms)
    fs.write("images_used", len(objpoints))
    fs.release()

    fov = np.degrees(2 * np.arctan(size[0] / (2 * mtx[0, 0])))
    print(f"{side:5s}: {len(objpoints)}/{len(images)} images, RMS {rms:.3f}px "
          f"{'OK' if rms < GOOD_RMS_PX else 'HIGH - retake blurry/bad images'}, "
          f"focal {mtx[0, 0]:.1f}px, horizontal FOV {fov:.1f} deg")
    worst = sorted(zip(per_view, names), reverse=True)[:3]
    print("       worst images: " + ", ".join(f"{n} ({e:.2f}px)" for e, n in worst))
    print(f"       saved {out_file}")


def load_intrinsics(path):
    if not path.exists():
        raise SystemExit(f"{path} missing. Run with --stage mono first.")
    fs = cv.FileStorage(str(path), cv.FILE_STORAGE_READ)
    mtx = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("dist_coeffs").mat()
    size = (int(fs.getNode("image_width").real()), int(fs.getNode("image_height").real()))
    fs.release()
    return mtx, dist, size


# --- Stage 2: the pair together ---------------------------------------------------------

def calibrate_stereo(show=False):
    mtx_l, dist_l, size_l = load_intrinsics(config.LEFT_INTRINSICS_FILE)
    mtx_r, dist_r, size_r = load_intrinsics(config.RIGHT_INTRINSICS_FILE)
    if size_l != size_r:
        raise SystemExit(f"Image sizes differ: left {size_l}, right {size_r}")
    size = size_l

    left, right = load_images(config.CALIB_LEFT_DIR), load_images(config.CALIB_RIGHT_DIR)
    common = sorted(set(left) & set(right))
    objp = board_points()
    objpoints, pts_l, pts_r, names = [], [], [], []
    for key in common:
        corners_l, _ = find_corners(left[key], show)
        corners_r, _ = find_corners(right[key], show)
        if corners_l is not None and corners_r is not None:
            objpoints.append(objp)
            pts_l.append(corners_l)
            pts_r.append(corners_r)
            names.append(left[key].name)
    if len(objpoints) < 5:
        raise SystemExit(f"Board seen by both cameras in only {len(objpoints)} pairs; need 10+.")

    # The return tuple differs between OpenCV 4 and 5; the first 7 items are stable.
    result = cv.stereoCalibrate(
        objpoints, pts_l, pts_r, mtx_l, dist_l, mtx_r, dist_r, size,
        criteria=CRITERIA, flags=cv.CALIB_FIX_INTRINSIC)
    rms, mtx_l, dist_l, mtx_r, dist_r, rot, trans = result[:7]
    baseline = float(np.linalg.norm(trans))
    print(f"stereo: {len(objpoints)}/{len(common)} pairs, RMS {rms:.3f}px "
          f"{'OK' if rms < GOOD_RMS_PX else 'HIGH - check the worst pairs below'}")
    print(f"        measured baseline {baseline:.2f} cm (config BASELINE_CM = {config.BASELINE_CM})")

    # Per pair: locate the board from the left view, then predict where the
    # right camera should see it. Large errors point at bad pairs.
    per_view = []
    for o, il, ir in zip(objpoints, pts_l, pts_r):
        _, rv, tv = cv.solvePnP(o, il, mtx_l, dist_l)
        r_right = cv.Rodrigues(rot @ cv.Rodrigues(rv)[0])[0]
        t_right = rot @ tv + trans.reshape(3, 1)
        per_view.append(rms_px(cv.projectPoints(o, r_right, t_right, mtx_r, dist_r)[0], ir))
    worst = sorted(zip(per_view, names), reverse=True)[:3]
    print("        worst pairs: " + ", ".join(f"{n} ({e:.2f}px)" for e, n in worst))

    # alpha=0 keeps only valid pixels, so the rectified views have no black borders.
    rect_l, rect_r, proj_l, proj_r, q, roi_l, roi_r = cv.stereoRectify(
        mtx_l, dist_l, mtx_r, dist_r, size, rot, trans, alpha=0)
    map_l = cv.initUndistortRectifyMap(mtx_l, dist_l, rect_l, proj_l, size, cv.CV_16SC2)
    map_r = cv.initUndistortRectifyMap(mtx_r, dist_r, rect_r, proj_r, size, cv.CV_16SC2)

    fs = cv.FileStorage(str(config.STEREO_MAP_FILE), cv.FILE_STORAGE_WRITE)
    fs.write("stereoMapL_x", map_l[0])
    fs.write("stereoMapL_y", map_l[1])
    fs.write("stereoMapR_x", map_r[0])
    fs.write("stereoMapR_y", map_r[1])
    fs.write("Q", q)
    fs.write("P1", proj_l)
    fs.write("P2", proj_r)
    fs.write("R", rot)
    fs.write("T", trans)
    fs.write("baseline_cm", baseline)
    fs.write("rms_error_px", rms)
    fs.release()
    print(f"        saved {config.STEREO_MAP_FILE}")
    print("Next: python stereo_calibration/check.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=["mono", "stereo", "all"], default="all")
    parser.add_argument("--show", action="store_true", help="display detected corners")
    args = parser.parse_args()

    config.CALIB_OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.stage in ("mono", "all"):
        calibrate_mono("left", config.CALIB_LEFT_DIR, config.LEFT_INTRINSICS_FILE, args.show)
        calibrate_mono("right", config.CALIB_RIGHT_DIR, config.RIGHT_INTRINSICS_FILE, args.show)
    if args.stage in ("stereo", "all"):
        calibrate_stereo(args.show)
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()
