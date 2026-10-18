"""Stereo vision depth estimation and robot-arm control.

Modules:
    config         all settings (overridable with environment variables)
    cameras        HTTP / USB stereo camera input
    rectification  applies stereo_calibration/stereoMap.xml to frames
    detection      SSD MobileNet v3 detector and left/right matching
    triangulation  depth and 3D position
    pipeline       per-frame detect -> match -> triangulate -> draw
    tracking       waits for a steady target
    arm            inverse kinematics, servos, gripper, pluck routine

Calibration itself is a separate tool in stereo_calibration/.
"""
