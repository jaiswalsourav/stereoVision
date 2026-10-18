"""Tests for the arm: servo mapping, limits, smooth motion, stopping, transforms.

    python -m unittest discover tests
"""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fakes import make_arm  # noqa: E402
from stereo_vision import config  # noqa: E402
from stereo_vision.arm import ArmStopped, UnreachableError, ease  # noqa: E402
from stereo_vision.kinematics import (CameraToArm, forward_kinematics,  # noqa: E402
                                      ik_solutions, inverse_kinematics)
from stereo_vision.servos import Joint  # noqa: E402


def joints_with(**changes):
    """config.JOINTS with some fields changed, e.g. elbow={"max_deg": 270}."""
    out = {n: dict(v) for n, v in config.JOINTS.items()}
    for name, fields in changes.items():
        out[name].update(fields)
    return out


class ServoMappingTest(unittest.TestCase):
    def test_offset_and_inversion(self):
        j = Joint("base", 0, offset_deg=4.0)
        self.assertEqual(j.to_servo(90), 94.0)
        j = Joint("base", 0, inverted=True)
        self.assertEqual(j.to_servo(30), 150.0)
        j = Joint("base", 0, min_deg=20, max_deg=160)
        self.assertEqual(j.to_servo(5), 20.0)       # clamped to the limit
        self.assertEqual(Joint("b", 0, offset_deg=10).to_servo(175), 180.0)

    def test_inverted_joint_reaches_board_mirrored(self):
        arm, kit = make_arm(joints=joints_with(base={"inverted": True}))
        arm.move_joints({"base": 60.0})
        self.assertAlmostEqual(kit.servo[arm.joints["base"].channel].angle, 120.0)
        self.assertAlmostEqual(arm.pose["base"], 60.0)  # logical angle unchanged

    def test_pulse_range_is_set(self):
        _, kit = make_arm()
        self.assertEqual(kit.servo[0].pulse_range,
                         (config.SERVO_MIN_PULSE_US, config.SERVO_MAX_PULSE_US))


class LimitsTest(unittest.TestCase):
    def test_out_of_limit_target_is_refused(self):
        arm, kit = make_arm(joints=joints_with(shoulder={"max_deg": 120}))
        with self.assertRaises(UnreachableError):
            arm.move_joints({"shoulder": 150})
        self.assertEqual(kit.log, [])

    def test_ik_respects_limits(self):
        target = (0.0, -10.0, 30.0)
        normal = inverse_kinematics(*target)
        with self.assertRaises(UnreachableError):
            inverse_kinematics(*target, joints=joints_with(
                shoulder={"max_deg": normal["shoulder"] - 5}))

    def test_elbow_down_used_when_allowed(self):
        target = (0.0, -10.0, 30.0)
        up, down = ik_solutions(*target)
        for sol in (up, down):
            np.testing.assert_allclose(forward_kinematics(**sol), target, atol=1e-9)
        # Elbow-up is the only option with a 0..180 elbow...
        self.assertEqual(inverse_kinematics(*target), up)
        # ...but if elbow-up is blocked and elbow-down allowed, use elbow-down.
        joints = joints_with(elbow={"min_deg": 180, "max_deg": 270},
                             shoulder={"min_deg": -90})
        self.assertEqual(inverse_kinematics(*target, joints=joints), down)


class MotionTest(unittest.TestCase):
    def test_smooth_and_together(self):
        arm, kit = make_arm(update_hz=50, speed_deg_s=60)
        arm.move_joints({"base": 90.0, "shoulder": 90.0})
        kit.log.clear()
        arm.move_joints({"base": 150.0, "shoulder": 60.0})
        base = kit.angles(arm.joints["base"].channel)
        shoulder = kit.angles(arm.joints["shoulder"].channel)
        self.assertEqual(len(base), 50)              # 60 deg at 60 deg/s, 50 Hz -> 1 s
        self.assertEqual(len(base), len(shoulder))   # both joints arrive together
        steps = np.diff([90.0] + base)
        self.assertLess(steps[0], steps[len(steps) // 2])   # starts slowly
        self.assertLess(steps[-1], steps[len(steps) // 2])  # ends slowly
        self.assertAlmostEqual(base[-1], 150.0)

    def test_ease(self):
        self.assertEqual(ease(0), 0)
        self.assertAlmostEqual(ease(1), 1)
        self.assertAlmostEqual(ease(0.5), 0.5)

    def test_linear_needs_known_pose(self):
        arm, _ = make_arm()
        with self.assertRaises(UnreachableError):
            arm.move_linear((0, -5, 28))

    def test_home_arm_keeps_gripper(self):
        arm, _ = make_arm()
        arm.home()
        arm.close_gripper()
        arm.move_to((0, -5, 28))
        arm.home_arm()
        self.assertEqual(arm.pose["gripper"], config.GRIPPER_CLOSED_DEG)
        self.assertEqual(arm.pose["base"], config.JOINTS["base"]["home_deg"])


class StopTest(unittest.TestCase):
    def test_stop_interrupts_background_move(self):
        arm, kit = make_arm(realtime=True, update_hz=100, speed_deg_s=30)
        arm.move_joints({"base": 30.0})
        job = arm.start(arm.move_joints, {"base": 150.0})  # takes 4 s
        self.assertTrue(arm.busy)
        job.wait(0.3)
        arm.stop()
        self.assertTrue(job.wait(1.0))
        self.assertIsInstance(job.error, ArmStopped)
        self.assertFalse(arm.busy)
        self.assertLess(arm.pose["base"], 100.0)  # stopped part way
        # A new command works after a stop.
        arm.move_joints({"base": 90.0}, speed_deg_s=1000)
        self.assertEqual(arm.pose["base"], 90.0)

    def test_busy_arm_refuses_second_job(self):
        arm, _ = make_arm(realtime=True, update_hz=100, speed_deg_s=30)
        arm.move_joints({"base": 30.0})
        arm.start(arm.move_joints, {"base": 150.0})
        with self.assertRaises(RuntimeError):
            arm.start(arm.home)
        arm.stop()
        arm.wait(1.0)

    def test_release_turns_servos_off(self):
        arm, kit = make_arm()
        arm.home()
        arm.release()
        self.assertTrue(all(kit.servo[j.channel].angle is None for j in arm.joints.values()))
        self.assertEqual(arm.pose, {})


class TransformTest(unittest.TestCase):
    def test_fit_recovers_rotation_and_offset(self):
        rng = np.random.default_rng(0)
        rot = CameraToArm.fit(rng.normal(size=(3, 3)), rng.normal(size=(3, 3)))[0].rotation
        true = CameraToArm(rot, [3.0, -12.0, 5.0])
        cam = rng.uniform(-20, 20, size=(8, 3))
        arm_pts = cam @ true.rotation.T + true.translation
        fitted, rms = CameraToArm.fit(cam, arm_pts)
        self.assertLess(rms, 1e-9)
        np.testing.assert_allclose(fitted.rotation, true.rotation, atol=1e-9)
        np.testing.assert_allclose(fitted([1, 2, 3]), true([1, 2, 3]), atol=1e-9)

    def test_offset_transform_matches_old_behaviour(self):
        tf = CameraToArm.from_offset((2.0, -3.0, 10.0))
        np.testing.assert_allclose(tf([2.0, -3.0, 40.0]), [0, 0, 30.0])

    def test_save_and_load(self):
        tf = CameraToArm(np.eye(3)[[1, 0, 2]] * [1, 1, 1], [1.0, 2.0, 3.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.yml"
            tf.save(path)
            loaded = CameraToArm.load(path)
        np.testing.assert_allclose(loaded.rotation, tf.rotation)
        np.testing.assert_allclose(loaded.translation, tf.translation)


if __name__ == "__main__":
    unittest.main()
