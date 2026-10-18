"""Tests for fruit plucking: motion planning and the servo sequence.

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stereo_vision import config  # noqa: E402
from stereo_vision.arm import (Arm, UnreachableError, forward_kinematics,  # noqa: E402
                               plan_pluck)
from stereo_vision.detection import Detection  # noqa: E402
from stereo_vision.tracking import StableTarget  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from fruit_picker import choose_fruit  # noqa: E402


class FakeServo:
    def __init__(self, log, channel):
        self.log, self.channel = log, channel

    @property
    def angle(self):
        return None

    @angle.setter
    def angle(self, value):
        assert 0.0 <= value <= 180.0, value
        self.log.append((self.channel, value))


class FakeKit:
    def __init__(self):
        self.log = []
        self.servo = [FakeServo(self.log, ch) for ch in range(16)]


def make_arm():
    arm = Arm(dry_run=True, settle_s=0, step_delay_s=0)
    arm.kit = FakeKit()
    return arm


class PlanTest(unittest.TestCase):
    def test_poses_hit_expected_points(self):
        fruit = np.array([5.0, -5.0, 28.0])
        plan = plan_pluck(fruit)
        np.testing.assert_allclose(forward_kinematics(**plan["grab"]), fruit, atol=1e-6)
        toward = np.array([5.0, 0, 28.0]) / np.hypot(5.0, 28.0)
        np.testing.assert_allclose(forward_kinematics(**plan["approach"]),
                                   fruit - toward * config.APPROACH_CM, atol=1e-6)
        pull = forward_kinematics(**plan["pull"])
        self.assertLess(np.hypot(pull[0], pull[2]), np.hypot(fruit[0], fruit[2]))  # pulled back
        self.assertGreater(pull[1], fruit[1])                                      # and down

    def test_out_of_reach(self):
        with self.assertRaises(UnreachableError):
            plan_pluck([0.0, 0.0, 60.0])


class SequenceTest(unittest.TestCase):
    def test_pluck_sequence(self):
        arm = make_arm()
        arm.pluck(np.array([0.0, -5.0, 28.0]))
        grip = [v for ch, v in arm.kit.log if ch == config.SERVO_GRIPPER]
        # opens, closes on the fruit, opens again over the basket
        self.assertEqual(grip[0], config.GRIPPER_OPEN_DEG)
        self.assertIn(config.GRIPPER_CLOSED_DEG, grip)
        self.assertEqual(grip[-1], config.GRIPPER_OPEN_DEG)
        closed_at = arm.kit.log.index((config.SERVO_GRIPPER, config.GRIPPER_CLOSED_DEG))
        base_after_grip = [v for ch, v in arm.kit.log[closed_at:] if ch == config.SERVO_BASE]
        self.assertIn(config.DROP_POSE["base"], base_after_grip)  # visits the basket
        self.assertEqual(arm.current["base"], 90.0)               # ends at home

    def test_unreachable_does_not_move(self):
        arm = make_arm()
        with self.assertRaises(UnreachableError):
            arm.pluck(np.array([0.0, 0.0, 80.0]))
        self.assertEqual(arm.kit.log, [])

    def test_moves_are_ramped(self):
        arm = make_arm()
        arm.move({"base": 90.0})
        arm.kit.log.clear()
        arm.move({"base": 150.0})
        steps = [v for _, v in arm.kit.log]
        self.assertGreater(len(steps), 10)
        self.assertLessEqual(max(np.diff([90.0] + steps)), config.SERVO_STEP_DEG + 1e-9)


class ChooseFruitTest(unittest.TestCase):
    def test_picks_closest_reachable(self):
        far = (Detection("apple", 0.9, (0, 0, 10, 10)), None, np.array([0.0, 0.0, 90.0]))
        near = (Detection("orange", 0.8, (0, 0, 10, 10)), None, np.array([0.0, -5.0, 28.0]))
        too_close = (Detection("apple", 0.9, (0, 0, 10, 10)), None, np.array([0.0, 0.0, 3.0]))
        det, xyz = choose_fruit([far, too_close, near])
        self.assertEqual(det.class_name, "orange")
        self.assertIsNone(choose_fruit([far]))


class StableTargetTest(unittest.TestCase):
    def test_needs_steady_frames(self):
        t = StableTarget(frames=3, tolerance_cm=1.0)
        p = np.array([0.0, 0.0, 30.0])
        self.assertIsNone(t.update(p))
        self.assertIsNone(t.update(p + 5))   # jumped: restart
        self.assertIsNone(t.update(p + 5))
        np.testing.assert_allclose(t.update(p + 5), p + 5)


if __name__ == "__main__":
    unittest.main()
