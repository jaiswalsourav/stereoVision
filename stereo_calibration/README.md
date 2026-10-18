# Stereo camera calibration

A standalone tool that produces `stereoMap.xml`, the only file the main
programs need from calibration. Recalibrate whenever a camera moves, rotates
or is refocused.

## Steps

Run from the repository root.

```bash
python stereo_calibration/capture.py     # 1. save 20-30 chessboard pairs
python stereo_calibration/calibrate.py   # 2. calibrate each camera, then the pair
python stereo_calibration/check.py       # 3. verify the result
```

1. **Capture.** Print a 9x6 inner-corner chessboard, glue it to something
   flat, and measure one square. Set `CHESSBOARD_SQUARE_CM` in
   `stereo_vision/config.py` to that size. Press `s` to save a pair when the
   status bar shows `L:ok R:ok`. Vary the board's position (centre and all
   corners of the view), distance and tilt. Hold it still to avoid blur.
2. **Calibrate.** Runs in two separate stages:
   - `--stage mono` calibrates each camera alone, using every image in which
     that camera sees the board. Results go to `left_camera.yml` and
     `right_camera.yml`, with the RMS error, focal length and field of view.
   - `--stage stereo` loads those files and computes how the cameras sit
     relative to each other, then writes `stereoMap.xml`. It prints the
     measured baseline; copy it into `BASELINE_CM` in the config.

   Both stages list their worst images. Delete those from `images/` and run
   again if the RMS error is above 0.5 px.
3. **Check.** Reports whether the maps are healthy and whether matching
   corners land on the same row in both rectified views. `--live` shows the
   rectified cameras with guide lines.

## Files

| File | Written by | Contents |
|------|-----------|----------|
| `images/left/`, `images/right/` | `capture.py` | chessboard pairs (`imageL007.png` pairs with `imageR007.png`) |
| `left_camera.yml`, `right_camera.yml` | `calibrate.py --stage mono` | each camera's lens parameters |
| `stereoMap.xml` | `calibrate.py --stage stereo` | rectification maps, Q matrix, baseline |

Set `CALIBRATION_OUT=some/folder` to write a calibration somewhere else, for
example to keep one per camera rig.
