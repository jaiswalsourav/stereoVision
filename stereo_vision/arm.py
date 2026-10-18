"""Inverse kinematics, servo control and the fruit-plucking routine.

Arm frame: the same axes as the left camera (x right, y down, z forward),
shifted so the origin is the shoulder joint (``config.ARM_OFFSET_CM``).

Servo conventions (all in degrees, 0..180, 90 = home):
  base      90 = straight ahead, <90 turns right, >90 turns left
  shoulder   0 = upper arm horizontal, 90 = pointing straight up
  elbow    180 = arm fully stretched, smaller values fold the forearm
  gripper   config.GRIPPER_OPEN_DEG / config.GRIPPER_CLOSED_DEG
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


def plan_pluck(target_arm, approach_cm=config.APPROACH_CM,
               pull_back_cm=config.PULL_BACK_CM, pull_down_cm=config.PULL_DOWN_CM):
    """Plan the poses for plucking a fruit at ``target_arm`` (arm frame, cm).

    The gripper comes in horizontally from the shoulder's side: first to a
    point ``approach_cm`` short of the fruit, then onto it, and after
    gripping it pulls back and down to snap the stem.
    Returns {"approach", "grab", "pull"} servo-angle dicts, or raises
    UnreachableError if any pose is impossible (checked before moving).
    """
    t = np.asarray(target_arm, dtype=float)
    horizontal = np.array([t[0], 0.0, t[2]])
    norm = np.linalg.norm(horizontal)
    if norm < 1e-6:
        raise UnreachableError("fruit is directly above or below the shoulder")
    toward = horizontal / norm
    approach = t - toward * approach_cm
    pull = t - toward * pull_back_cm + np.array([0.0, pull_down_cm, 0.0])
    return {"approach": inverse_kinematics(*approach),
            "grab": inverse_kinematics(*t),
            "pull": inverse_kinematics(*pull)}


class Arm:
    """Drives the servos through a PCA9685 board, or prints in dry-run mode."""

    def __init__(self, dry_run=False, settle_s=0.3,
                 step_deg=config.SERVO_STEP_DEG, step_delay_s=config.SERVO_STEP_DELAY_S):
        self.settle_s = settle_s
        self.step_deg = step_deg
        self.step_delay_s = step_delay_s
        self.channels = {"base": config.SERVO_BASE,
                         "shoulder": config.SERVO_SHOULDER,
                         "elbow": config.SERVO_ELBOW,
                         "gripper": config.SERVO_GRIPPER}
        self.current = {}  # last commanded angle per joint
        self.kit = None
        if not dry_run:
            try:
                from adafruit_servokit import ServoKit
                self.kit = ServoKit(channels=16)
            except (ImportError, ValueError, RuntimeError, OSError) as exc:
                print(f"[arm] servo board unavailable ({exc}); running dry")

    def _write(self, name, angle):
        if self.kit is not None:
            self.kit.servo[self.channels[name]].angle = angle
        self.current[name] = angle

    def move(self, angles, smooth=True):
        """Move joints to ``angles`` together, ramping in small steps."""
        target = {name: max(0.0, min(180.0, a)) for name, a in angles.items()}
        start = {name: self.current.get(name, a) for name, a in target.items()}
        span = max(abs(target[n] - start[n]) for n in target)
        steps = max(1, math.ceil(span / self.step_deg)) if smooth else 1
        for i in range(1, steps + 1):
            for name in target:
                self._write(name, start[name] + (target[name] - start[name]) * i / steps)
            if self.kit is not None and i < steps:
                time.sleep(self.step_delay_s)
        if self.kit is None:
            print("[arm] " + "  ".join(f"{n}={a:.1f}" for n, a in target.items()))
        time.sleep(self.settle_s)

    def home(self):
        self.move(HOME)

    def open_gripper(self):
        self.move({"gripper": config.GRIPPER_OPEN_DEG})

    def close_gripper(self):
        self.move({"gripper": config.GRIPPER_CLOSED_DEG})

    def reach(self, xyz_camera):
        """Move to a point given in left-camera coordinates (cm)."""
        angles = inverse_kinematics(*camera_to_arm(xyz_camera))
        self.move(angles)
        return angles

    def pluck(self, xyz_camera, drop_pose=config.DROP_POSE,
              twist_deg=config.TWIST_DEG, twist_count=config.TWIST_COUNT):
        """Pick the fruit at a left-camera point and drop it in the basket.

        Raises UnreachableError before moving if any pose is impossible.
        """
        plan = plan_pluck(camera_to_arm(xyz_camera))
        self.open_gripper()
        self.move(plan["approach"])
        self.move(plan["grab"])
        self.close_gripper()
        base = plan["grab"]["base"]
        for _ in range(twist_count):  # small wiggle helps break the stem
            self.move({"base": min(180.0, base + twist_deg)})
            self.move({"base": max(0.0, base - twist_deg)})
        self.move({"base": base})
        self.move(plan["pull"])
        self.move(HOME)
        self.move(drop_pose)
        self.open_gripper()
        self.home()
        return plan
