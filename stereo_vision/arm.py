"""Inverse kinematics and servo control for the 3-servo (base + 2-link) arm.

Arm frame: the same axes as the left camera (x right, y down, z forward),
shifted so the origin is the shoulder joint (``config.ARM_OFFSET_CM``).

Servo conventions (all in degrees, 0..180, 90 = home):
  base      90 = straight ahead, <90 turns right, >90 turns left
  shoulder   0 = upper arm horizontal, 90 = pointing straight up
  elbow    180 = arm fully stretched, smaller values fold the forearm
"""
import math
import time

import numpy as np

from . import config

HOME = {"base": 90.0, "shoulder": 90.0, "elbow": 90.0}


class UnreachableError(ValueError):
    pass


def camera_to_arm(xyz_camera, offset=config.ARM_OFFSET_CM):
    return np.asarray(xyz_camera, dtype=float) - np.asarray(offset, dtype=float)


def inverse_kinematics(x, y, z, l1=config.ARM_L1_CM, l2=config.ARM_L2_CM,
                       elbow_up=True):
    """Servo angles (base, shoulder, elbow) in degrees for a target point.

    Raises UnreachableError when the point is outside the arm's workspace
    or would need a servo outside 0..180 degrees.
    """
    base = math.degrees(math.atan2(z, x))  # 90 when the target is straight ahead
    r = math.hypot(x, z)                   # horizontal reach
    h = -y                                 # height above the shoulder
    d = math.hypot(r, h)
    if not abs(l1 - l2) <= d <= l1 + l2:
        raise UnreachableError(
            f"target {d:.1f} cm away, arm reaches {abs(l1 - l2):.1f}..{l1 + l2:.1f} cm")

    cos_t2 = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    t2 = math.acos(max(-1.0, min(1.0, cos_t2)))      # elbow bend, 0 = straight
    k = math.atan2(l2 * math.sin(t2), l1 + l2 * math.cos(t2))
    shoulder = math.atan2(h, r) + (k if elbow_up else -k)

    angles = {"base": base,
              "shoulder": math.degrees(shoulder),
              "elbow": 180.0 - math.degrees(t2)}
    bad = {name: a for name, a in angles.items() if not 0.0 <= a <= 180.0}
    if bad:
        raise UnreachableError(f"servo angles out of range: {bad}")
    return angles


def forward_kinematics(base, shoulder, elbow, l1=config.ARM_L1_CM, l2=config.ARM_L2_CM):
    """Inverse of inverse_kinematics (elbow-up). Useful for checking."""
    yaw, s = math.radians(base), math.radians(shoulder)
    t2 = math.radians(180.0 - elbow)
    r = l1 * math.cos(s) + l2 * math.cos(s - t2)
    h = l1 * math.sin(s) + l2 * math.sin(s - t2)
    return np.array([r * math.cos(yaw), -h, r * math.sin(yaw)])


class Arm:
    """Drives the servos through a PCA9685 board, or prints in dry-run mode."""

    def __init__(self, dry_run=False, settle_s=0.5):
        self.settle_s = settle_s
        self.channels = {"base": config.SERVO_BASE,
                         "shoulder": config.SERVO_SHOULDER,
                         "elbow": config.SERVO_ELBOW}
        self.kit = None
        if not dry_run:
            try:
                from adafruit_servokit import ServoKit
                self.kit = ServoKit(channels=16)
            except (ImportError, ValueError, RuntimeError, OSError) as exc:
                print(f"[arm] servo board unavailable ({exc}); running dry")

    def move(self, angles):
        for name, angle in angles.items():
            angle = max(0.0, min(180.0, angle))
            if self.kit is not None:
                self.kit.servo[self.channels[name]].angle = angle
            else:
                print(f"[arm] {name} (ch {self.channels[name]}) -> {angle:.1f} deg")
        time.sleep(self.settle_s)

    def home(self):
        self.move(HOME)

    def reach(self, xyz_camera):
        """Move to a point given in left-camera coordinates (cm)."""
        target = camera_to_arm(xyz_camera)
        angles = inverse_kinematics(*target)
        self.move(angles)
        return angles
