"""High-level robot arm control.

- Smooth joint moves: all joints start and arrive together, with gentle
  acceleration and deceleration, at a set speed in degrees per second.
- Straight-line moves of the gripper tip, used to approach and pull fruit
  without swinging through the plant.
- Every target is checked against the joint limits before anything moves.
- Moves can run in the background (``start``) so the camera view stays live,
  and can be interrupted (``stop``) or the servos released (``release``).

Angle conventions and the arm frame are described in kinematics.py; servo
trim, direction and limits in servos.py and config.JOINTS.
"""
import math
import threading
import time
from contextlib import contextmanager

import numpy as np

from . import config
from .kinematics import (ARM_JOINTS, CameraToArm, UnreachableError,  # noqa: F401 (re-exported)
                         forward_kinematics, inverse_kinematics, linear_path,
                         within_limits)
from .servos import ServoBus, load_joints


class ArmStopped(RuntimeError):
    """A motion was interrupted by Arm.stop()."""


def ease(t):
    """Smooth 0..1 -> 0..1 profile: starts and ends with zero speed."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def plan_pluck(target_arm, joints=config.JOINTS, start_pose=None,
               approach_cm=config.APPROACH_CM, pull_back_cm=config.PULL_BACK_CM,
               pull_down_cm=config.PULL_DOWN_CM, drop_pose=None, step_cm=1.0):
    """Check and plan plucking a fruit at ``target_arm`` (arm frame, cm).

    The gripper first goes to a point ``approach_cm`` short of the fruit
    (coming from the shoulder's side), then moves in a straight line onto
    it, and after gripping pulls back and down in a straight line.
    Raises UnreachableError if any part is impossible, before anything moves.
    """
    drop_pose = config.DROP_POSE if drop_pose is None else drop_pose
    t = np.asarray(target_arm, dtype=float)
    horizontal = np.array([t[0], 0.0, t[2]])
    if np.linalg.norm(horizontal) < 1e-6:
        raise UnreachableError("fruit is directly above or below the shoulder")
    toward = horizontal / np.linalg.norm(horizontal)
    approach = t - toward * approach_cm
    pull = t - toward * pull_back_cm + np.array([0.0, pull_down_cm, 0.0])

    bad = within_limits(drop_pose, joints)
    if bad:
        raise UnreachableError(f"basket (drop) pose is outside the limits of {bad}")
    approach_pose = inverse_kinematics(*approach, joints=joints, near=start_pose)
    to_fruit = linear_path(approach, t, step_cm, joints, near=approach_pose)
    pull_out = linear_path(t, pull, step_cm, joints, near=to_fruit[-1])
    return {"approach": approach_pose, "grab": to_fruit[-1], "pull": pull_out[-1],
            "points": {"approach": approach, "grab": t, "pull": pull}}


_default_transform = None


def camera_to_arm(xyz_camera):
    """Convert a left-camera point to the arm frame with the default transform."""
    global _default_transform
    if _default_transform is None:
        _default_transform = CameraToArm.default()
    return _default_transform(xyz_camera)


class Job:
    """A command running in a background thread."""

    def __init__(self, fn, *args, **kwargs):
        self.result = None
        self.error = None
        self._thread = threading.Thread(target=self._run, args=(fn, args, kwargs), daemon=True)
        self._thread.start()

    def _run(self, fn, args, kwargs):
        try:
            self.result = fn(*args, **kwargs)
        except BaseException as exc:  # reported to the caller via .error
            self.error = exc

    def done(self):
        return not self._thread.is_alive()

    def wait(self, timeout=None):
        self._thread.join(timeout)
        return self.done()


class Arm:
    """The robot arm. Use ``dry_run=True`` to run without the servo board."""

    def __init__(self, dry_run=False, joints=None, transform=None, kit=None,
                 speed_deg_s=config.ARM_SPEED_DEG_S,
                 linear_speed_cm_s=config.ARM_LINEAR_SPEED_CM_S,
                 update_hz=config.ARM_UPDATE_HZ, verbose=None, realtime=None):
        self.joints = load_joints(joints)
        self.limits = {n: {"min_deg": j.min_deg, "max_deg": j.max_deg}
                       for n, j in self.joints.items()}
        self.bus = ServoBus(self.joints, dry_run=dry_run, kit=kit)
        self.transform = transform if transform is not None else CameraToArm.default()
        self.speed_deg_s = speed_deg_s
        self.linear_speed_cm_s = linear_speed_cm_s
        self.update_hz = update_hz
        # Without hardware, print each move and skip the real-time waits.
        self.verbose = (not self.bus.is_hardware) if verbose is None else verbose
        self.realtime = self.bus.is_hardware if realtime is None else realtime
        self.pose = {}  # last commanded logical angle per joint
        self._lock = threading.RLock()
        self._depth = 0
        self._stop = threading.Event()
        self._job = None

    # --- internals -----------------------------------------------------------------

    @contextmanager
    def _command(self):
        """Serialise commands; a new top-level command clears an old stop."""
        with self._lock:
            if self._depth == 0:
                self._stop.clear()
            self._depth += 1
            try:
                yield
            finally:
                self._depth -= 1

    def _log(self, text):
        if self.verbose:
            print(f"[arm] {text}")

    def _send(self, frame):
        for name, angle in frame.items():
            self.pose[name] = self.bus.write(name, angle)

    def _stream(self, frames):
        dt = 1.0 / self.update_hz
        next_t = time.monotonic()
        for frame in frames:
            if self._stop.is_set():
                raise ArmStopped("motion stopped")
            self._send(frame)
            if self.realtime:
                next_t += dt
                delay = next_t - time.monotonic()
                if delay > 0:
                    time.sleep(delay)

    def _require_pose(self):
        missing = [n for n in ARM_JOINTS if n not in self.pose]
        if missing:
            raise UnreachableError("arm position unknown; call home() first")

    # --- state ---------------------------------------------------------------------

    def position(self):
        """Current gripper tip position in the arm frame (cm)."""
        self._require_pose()
        return forward_kinematics(*(self.pose[n] for n in ARM_JOINTS))

    @property
    def busy(self):
        return self._job is not None and not self._job.done()

    # --- motions -------------------------------------------------------------------

    def move_joints(self, target, speed_deg_s=None):
        """Move the given joints to logical angles, smoothly and together."""
        with self._command():
            target = {n: float(a) for n, a in target.items()}
            unknown_names = [n for n in target if n not in self.joints]
            if unknown_names:
                raise ValueError(f"unknown joint {unknown_names[0]!r}; "
                                 f"joints are {', '.join(self.joints)}")
            bad = within_limits(target, self.limits)
            if bad:
                raise UnreachableError(
                    "outside joint limits: " + ", ".join(f"{n}={target[n]:.1f}" for n in bad))
            unknown = {n: a for n, a in target.items() if n not in self.pose}
            if unknown:  # first command for a joint: its position is unknown
                self._send(unknown)
            start = {n: self.pose[n] for n in target}
            span = max(abs(target[n] - start[n]) for n in target)
            if span > 1e-9:
                duration = span / (speed_deg_s or self.speed_deg_s)
                steps = max(1, math.ceil(duration * self.update_hz))
                self._stream({n: start[n] + (target[n] - start[n]) * ease(i / steps)
                              for n in target} for i in range(1, steps + 1))
            self._log("  ".join(f"{n}={a:.1f}" for n, a in target.items()))
            return dict(target)

    def move_to(self, xyz_arm, speed_deg_s=None):
        """Move the tip to a point (arm frame, cm) with a joint-space move."""
        with self._command():
            angles = inverse_kinematics(*xyz_arm, joints=self.limits, near=self.pose)
            return self.move_joints(angles, speed_deg_s)

    def move_linear(self, xyz_arm, speed_cm_s=None):
        """Move the tip in a straight line to a point (arm frame, cm)."""
        with self._command():
            start = self.position()
            end = np.asarray(xyz_arm, float)
            dist = float(np.linalg.norm(end - start))
            if dist < 1e-6:
                return dict(self.pose)
            duration = dist / (speed_cm_s or self.linear_speed_cm_s)
            steps = max(1, math.ceil(duration * self.update_hz))
            path = linear_path(start, end, dist / steps, joints=self.limits,
                               near=self.pose, ease=ease)
            self._stream(path)
            self._log(f"line to ({end[0]:.1f}, {end[1]:.1f}, {end[2]:.1f}) cm")
            return dict(self.pose)

    def home(self):
        """All joints, including the gripper, to their home angles."""
        return self.move_joints({n: j.home_deg for n, j in self.joints.items()})

    def home_arm(self):
        """Arm joints home, gripper unchanged (keeps hold of a fruit)."""
        return self.move_joints({n: self.joints[n].home_deg for n in ARM_JOINTS})

    def set_gripper(self, angle):
        return self.move_joints({"gripper": angle})

    def open_gripper(self):
        return self.set_gripper(config.GRIPPER_OPEN_DEG)

    def close_gripper(self):
        return self.set_gripper(config.GRIPPER_CLOSED_DEG)

    # --- tasks ---------------------------------------------------------------------

    def reach(self, xyz_camera):
        """Move the tip to a point seen by the camera (left-camera frame, cm)."""
        return self.move_to(self.transform(xyz_camera))

    def plan(self, xyz_camera):
        """Plan a pluck for a camera point without moving (raises if impossible)."""
        return plan_pluck(self.transform(xyz_camera), joints=self.limits)

    def pluck(self, xyz_camera):
        """Pluck the fruit at a left-camera point and drop it in the basket."""
        return self.pluck_at(self.transform(xyz_camera))

    def pluck_at(self, xyz_arm, drop_pose=None, twist_deg=config.TWIST_DEG,
                 twist_count=config.TWIST_COUNT):
        """Pluck the fruit at an arm-frame point and drop it in the basket.

        Everything is planned and checked first; an impossible pluck raises
        UnreachableError without moving.
        """
        drop_pose = config.DROP_POSE if drop_pose is None else drop_pose
        with self._command():
            plan = plan_pluck(xyz_arm, joints=self.limits, start_pose=self.pose or None,
                              drop_pose=drop_pose)
            base_joint = self.joints["base"]
            self.open_gripper()
            self.move_joints(plan["approach"])
            self.move_linear(plan["points"]["grab"])
            self.close_gripper()
            base = self.pose["base"]
            for _ in range(twist_count):  # a small wiggle helps break the stem
                self.move_joints({"base": base_joint.clamp(base + twist_deg)})
                self.move_joints({"base": base_joint.clamp(base - twist_deg)})
            self.move_joints({"base": base})
            self.move_linear(plan["points"]["pull"])
            self.home_arm()
            self.move_joints(drop_pose)
            self.open_gripper()
            self.home()
            return plan

    # --- background jobs and safety -------------------------------------------------------

    def start(self, fn, *args, **kwargs):
        """Run an arm command in the background, e.g. ``arm.start(arm.pluck, xyz)``."""
        if self.busy:
            raise RuntimeError("arm is busy")
        self._job = Job(fn, *args, **kwargs)
        return self._job

    def wait(self, timeout=None):
        return self._job.wait(timeout) if self._job else True

    def stop(self):
        """Interrupt the current motion; the arm holds where it is."""
        self._stop.set()

    def release(self):
        """Stop, then switch off the servo signals so the arm goes limp."""
        self.stop()
        self.wait(2.0)
        self.bus.release()
        self.pose.clear()
        self._log("servos released")

    def shutdown(self):
        """Stop any job and park the arm at home. Call before exiting."""
        self.stop()
        self.wait(5.0)
        try:
            self.home()
        except (ArmStopped, UnreachableError) as exc:
            self._log(f"could not park: {exc}")
