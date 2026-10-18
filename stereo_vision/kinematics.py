"""Arm geometry: forward/inverse kinematics and the camera-to-arm transform.

Pure maths, no hardware. Angles are "logical" joint angles in degrees:

  base      90 = pointing straight ahead (+z), 0 = right (+x), 180 = left
  shoulder   0 = upper arm horizontal, 90 = pointing straight up
  elbow    180 = forearm in line with the upper arm, smaller = folded

Arm frame (cm), origin at the shoulder joint: x right, y down, z forward.
"""
import math

import numpy as np

from . import config


class UnreachableError(ValueError):
    """The requested point or pose is outside what the arm can do."""


ARM_JOINTS = ("base", "shoulder", "elbow")


def forward_kinematics(base, shoulder, elbow, l1=config.ARM_L1_CM, l2=config.ARM_L2_CM):
    """Gripper tip position (x, y, z) in the arm frame for the given angles."""
    yaw, s = math.radians(base), math.radians(shoulder)
    bend = math.radians(180.0 - elbow)
    r = l1 * math.cos(s) + l2 * math.cos(s - bend)
    h = l1 * math.sin(s) + l2 * math.sin(s - bend)
    return np.array([r * math.cos(yaw), -h, r * math.sin(yaw)])


def within_limits(angles, joints=config.JOINTS, tol=1e-6):
    """Names of joints whose angle is outside its configured range."""
    return [n for n, a in angles.items()
            if not joints[n]["min_deg"] - tol <= a <= joints[n]["max_deg"] + tol]


def ik_solutions(x, y, z, l1=config.ARM_L1_CM, l2=config.ARM_L2_CM):
    """Both geometric solutions (elbow up, elbow down), ignoring joint limits."""
    r = math.hypot(x, z)   # horizontal reach
    h = -y                 # height above the shoulder
    d = math.hypot(r, h)
    if r < 1e-6:
        raise UnreachableError("point is directly above or below the shoulder")
    if not abs(l1 - l2) - 1e-9 <= d <= l1 + l2 + 1e-9:
        raise UnreachableError(
            f"point is {d:.1f} cm from the shoulder; arm reaches "
            f"{abs(l1 - l2):.1f}..{l1 + l2:.1f} cm")

    base = math.degrees(math.atan2(z, x))
    cos_bend = (d * d - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    bend = math.acos(max(-1.0, min(1.0, cos_bend)))       # 0 = straight
    k = math.atan2(l2 * math.sin(bend), l1 + l2 * math.cos(bend))
    phi = math.atan2(h, r)
    up = {"base": base, "shoulder": math.degrees(phi + k), "elbow": 180.0 - math.degrees(bend)}
    down = {"base": base, "shoulder": math.degrees(phi - k), "elbow": 180.0 + math.degrees(bend)}
    return [up, down]


def inverse_kinematics(x, y, z, joints=config.JOINTS, near=None,
                       l1=config.ARM_L1_CM, l2=config.ARM_L2_CM):
    """Joint angles reaching (x, y, z) within the joints' limits.

    If both elbow solutions are allowed, the one closest to ``near`` (a pose
    dict, usually the current pose) wins, so the arm never flips mid-motion.
    Raises UnreachableError if no solution respects the limits.
    """
    candidates = ik_solutions(x, y, z, l1, l2)
    valid = [c for c in candidates if not within_limits(c, joints)]
    if not valid:
        bad = within_limits(candidates[0], joints)
        detail = ", ".join(f"{n} {candidates[0][n]:.0f} deg" for n in bad)
        raise UnreachableError(f"needs {detail}, outside the joint limits")
    if near and len(valid) > 1:
        valid.sort(key=lambda c: sum(abs(c[n] - near.get(n, c[n])) for n in ARM_JOINTS))
    return valid[0]


def linear_path(start, end, step_cm, joints=config.JOINTS, near=None,
                l1=config.ARM_L1_CM, l2=config.ARM_L2_CM, ease=None):
    """Joint poses along the straight line start -> end, about one every ``step_cm``.

    ``ease`` optionally maps progress 0..1 to distance 0..1 (for smooth
    acceleration). Every point is solved up front, so an impossible path
    fails before the arm moves. Returns a list of pose dicts; the last one
    reaches ``end`` exactly.
    """
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = max(1, math.ceil(np.linalg.norm(end - start) / max(step_cm, 1e-3)))
    poses, prev = [], near
    for i in range(1, n + 1):
        f = i / n
        point = start + (end - start) * (ease(f) if ease else f)
        try:
            prev = inverse_kinematics(*point, joints=joints, near=prev, l1=l1, l2=l2)
        except UnreachableError as exc:
            raise UnreachableError(f"straight path leaves the workspace at "
                                   f"({point[0]:.1f}, {point[1]:.1f}, {point[2]:.1f}): {exc}")
        poses.append(prev)
    return poses


class CameraToArm:
    """Rigid transform from left-camera coordinates to arm coordinates (cm)."""

    def __init__(self, rotation=None, translation=None):
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
        self.translation = np.zeros(3) if translation is None else np.asarray(translation, float).ravel()

    def __call__(self, xyz_camera):
        return self.rotation @ np.asarray(xyz_camera, float) + self.translation

    @classmethod
    def from_offset(cls, offset_cm):
        """Axes parallel to the camera, shoulder at ``offset_cm`` from the camera."""
        return cls(np.eye(3), -np.asarray(offset_cm, float))

    @classmethod
    def fit(cls, camera_points, arm_points):
        """Best rigid transform mapping camera_points onto arm_points (Kabsch).

        Needs at least 3 non-collinear point pairs; 6+ spread through the
        workspace gives a good result. Returns (transform, rms_error_cm).
        """
        p = np.asarray(camera_points, float)
        q = np.asarray(arm_points, float)
        if len(p) < 3 or p.shape != q.shape:
            raise ValueError("need at least 3 matching camera/arm point pairs")
        pc, qc = p.mean(axis=0), q.mean(axis=0)
        u, _, vt = np.linalg.svd((p - pc).T @ (q - qc))
        d = np.sign(np.linalg.det(vt.T @ u.T))
        rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
        tf = cls(rot, qc - rot @ pc)
        mapped = p @ tf.rotation.T + tf.translation
        rms = float(np.sqrt(np.mean(np.sum((mapped - q) ** 2, axis=1))))
        return tf, rms

    def save(self, path):
        import cv2
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        fs.write("rotation", self.rotation)
        fs.write("translation", self.translation.reshape(3, 1))
        fs.release()

    @classmethod
    def load(cls, path):
        import cv2
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        rot, trans = fs.getNode("rotation").mat(), fs.getNode("translation").mat()
        fs.release()
        if rot is None or trans is None:
            raise ValueError(f"{path} is not a camera-to-arm transform file")
        return cls(rot, trans)

    @classmethod
    def default(cls):
        """The saved transform if there is one, else the configured offset."""
        if config.ARM_TRANSFORM_FILE.exists():
            return cls.load(config.ARM_TRANSFORM_FILE)
        return cls.from_offset(config.ARM_OFFSET_CM)
