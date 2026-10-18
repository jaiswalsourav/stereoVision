"""Stereo vision depth estimation and robot-arm control.

Modules:
    config         all settings (overridable with environment variables)
    cameras        HTTP / USB stereo camera input
    calibration    rectification using calibration/stereoMap.xml
    detection      SSD MobileNet v3 detector and left/right matching
    triangulation  depth and 3D position
    pipeline       per-frame detect -> match -> triangulate -> draw
    arm            inverse kinematics and servo control
"""
