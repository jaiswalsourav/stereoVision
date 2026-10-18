"""Check that stereoMap.xml is a usable calibration.

1. Map health: how much of each rectified view is valid image, and whether
   the map folds the image over itself (a sign of a failed calibration).
2. Row alignment: after rectification the same chessboard corner must lie
   on the same row in both views. Mean error under ~1 px is good.
3. Visual check: rectified pair with horizontal guide lines; features should
   sit on the same line in both views.

    python stereo_calibration/check.py           # use the saved image pairs
    python stereo_calibration/check.py --live    # use the cameras
"""
import argparse

import cv2 as cv
import numpy as np

import _paths  # noqa: F401  (puts the repo root on sys.path)
from calibrate import load_images
from stereo_vision import config, rectification

MIN_VALID_FRACTION = 0.85
MAX_ROW_ERROR_PX = 1.0


def map_health(maps):
    """Return (ok, report lines) for the left and right maps."""
    ok, lines = True, []
    for side in ("L", "R"):
        xy = maps[f"stereoMap{side}_x"][..., :2].astype(np.int32)
        h, w = xy.shape[:2]
        valid = (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
        frac = valid.mean()
        # Along each row of the valid area, source x must keep increasing.
        dx = np.diff(xy[..., 0].astype(float), axis=1)
        both = valid[:, 1:] & valid[:, :-1]
        backwards = (dx[both] < 0).mean() if both.any() else 1.0
        side_ok = frac >= MIN_VALID_FRACTION and backwards < 0.01
        ok &= side_ok
        lines.append(f"{'left ' if side == 'L' else 'right'}: valid area {frac * 100:.1f}%, "
                     f"folded {backwards * 100:.1f}% -> {'OK' if side_ok else 'BAD'}")
    return ok, lines


def row_alignment(maps_ok_pairs):
    """Mean |row difference| of matching chessboard corners after rectification."""
    errors = []
    for name, left, right in maps_ok_pairs:
        rect_l, rect_r = rectification.rectify(left, right)
        found = []
        for img in (rect_l, rect_r):
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            ok, corners = cv.findChessboardCorners(gray, config.CHESSBOARD_SIZE, None)
            found.append(corners.reshape(-1, 2) if ok else None)
        if found[0] is None or found[1] is None:
            continue
        err = float(np.mean(np.abs(found[0][:, 1] - found[1][:, 1])))
        disp = float(np.mean(found[0][:, 0] - found[1][:, 0]))
        errors.append((err, disp, name))
    return errors


def guide_view(left, right):
    rect_l, rect_r = rectification.rectify(left, right)
    view = np.hstack([rect_l, rect_r])
    for y in range(0, view.shape[0], 30):
        cv.line(view, (0, y), (view.shape[1], y), (0, 255, 0), 1)
    return view


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="show the cameras rectified")
    parser.add_argument("--no-show", action="store_true", help="print the report only")
    args = parser.parse_args()

    maps = rectification.load_maps()
    healthy, lines = map_health(maps)
    print(f"Map health ({config.STEREO_MAP_FILE.name}):")
    for line in lines:
        print("  " + line)

    if args.live:
        from stereo_vision.cameras import StereoCamera
        cams = StereoCamera()
        print("Live view: features should sit on the same green line. q/Esc to quit.")
        while True:
            left, right = cams.read()
            if left is not None:
                cv.imshow("rectified (left | right)", guide_view(left, right))
            if cv.waitKey(30) & 0xFF in (ord("q"), 27):
                break
        cams.release()
        cv.destroyAllWindows()
        return

    left_imgs, right_imgs = load_images(config.CALIB_LEFT_DIR), load_images(config.CALIB_RIGHT_DIR)
    pairs = [(left_imgs[k].name, cv.imread(str(left_imgs[k])), cv.imread(str(right_imgs[k])))
             for k in sorted(set(left_imgs) & set(right_imgs))]
    aligned = True
    if pairs:
        errors = row_alignment(pairs)
        if errors:
            mean_err = np.mean([e for e, _, _ in errors])
            wrong_side = sum(d <= 0 for _, d, _ in errors)
            aligned = mean_err <= MAX_ROW_ERROR_PX and wrong_side == 0
            print(f"Row alignment over {len(errors)} pairs: mean {mean_err:.2f}px "
                  f"(worst {max(errors)[0]:.2f}px in {max(errors)[2]}) -> "
                  f"{'OK' if aligned else 'BAD'}")
            if wrong_side:
                print(f"  {wrong_side} pairs have negative disparity: left/right cameras swapped?")
        else:
            print("Row alignment: board not found in any rectified pair")
            aligned = False
        if not args.no_show:
            cv.imshow("rectified (left | right)", guide_view(pairs[0][1], pairs[0][2]))
            print("Showing the first pair; press any key to close.")
            cv.waitKey(0)
            cv.destroyAllWindows()
    else:
        print("Row alignment: no saved image pairs to test (use --live to look at the cameras)")

    verdict = healthy and aligned
    print("RESULT:", "calibration looks good" if verdict else
          "calibration is NOT usable - recapture images and run calibrate.py again")
    raise SystemExit(0 if verdict else 1)


if __name__ == "__main__":
    main()
