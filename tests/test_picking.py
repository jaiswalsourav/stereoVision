"""Tests for fruit plucking: motion planning, the servo sequence, target choice.

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from fakes import make_arm  # noqa: E402
from fruit_picker import choose_fruit  # noqa: E402
from stereo_vision import config  # noqa: E402
from stereo_vision.arm import UnreachableError, forward_kinematics, plan_pluck  # noqa: E402
from stereo_vision.detection import Detection  # noqa: E402
from stereo_vision.kinematics import CameraToArm  # noqa: E402
from stereo_vision.tracking import StableTarget  # noqa: E402

FRUIT = np.array([0.0, -5.0, 28.0])


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
        arm, kit = make_arm(transform=CameraToArm())
        arm.home()
        kit.log.clear()
        arm.pluck(FRUIT)
        ch = {n: j.channel for n, j in arm.joints.items()}
        grip = kit.angles(ch["gripper"])
        closed = [i for i, a in enumerate(grip) if abs(a - config.GRIPPER_CLOSED_DEG) < 1e-9]
        self.assertTrue(closed, "gripper never closed")
        self.assertAlmostEqual(grip[-1], config.GRIPPER_OPEN_DEG)  # released at the end
        # The gripper stays closed from grabbing until over the basket.
        closed_at = kit.log.index((ch["gripper"], config.GRIPPER_CLOSED_DEG))
        after = kit.log[closed_at:]
        reopen = next(i for i, (c, v) in enumerate(after)
                      if c == ch["gripper"] and v < config.GRIPPER_CLOSED_DEG)
        bases_while_closed = [v for c, v in after[:reopen] if c == ch["base"]]
        self.assertAlmostEqual(bases_while_closed[-1], config.DROP_POSE["base"])
        self.assertEqual(arm.pose["base"], 90.0)  # ends at home

    def test_unreachable_does_not_move(self):
        arm, kit = make_arm(transform=CameraToArm())
        arm.home()
        kit.log.clear()
        with self.assertRaises(UnreachableError):
            arm.pluck(np.array([0.0, 0.0, 80.0]))
        self.assertEqual(kit.log, [])

    def test_approach_is_straight(self):
        arm, kit = make_arm(transform=CameraToArm())
        arm.home()
        plan = plan_pluck(FRUIT)
        arm.move_joints(plan["approach"])
        start = arm.position()
        tips = []
        original = arm._send

        def record(frame):
            original(frame)
            tips.append(arm.position())
        arm._send = record
        arm.move_linear(FRUIT)
        line = FRUIT - start
        for tip in tips:  # distance of every intermediate tip position from the line
            offset = (tip - start) - line * np.dot(tip - start, line) / np.dot(line, line)
            self.assertLess(np.linalg.norm(offset), 1e-6)
        np.testing.assert_allclose(tips[-1], FRUIT, atol=1e-6)


class ChooseFruitTest(unittest.TestCase):
    def test_picks_closest_reachable(self):
        arm, _ = make_arm(transform=CameraToArm())
        det = lambda name: Detection(name, 0.9, (0, 0, 10, 10))  # noqa: E731
        far = (det("apple"), None, np.array([0.0, 0.0, 90.0]))
        near = (det("orange"), None, FRUIT)
        too_close = (det("apple"), None, np.array([0.0, 0.0, 3.0]))
        chosen, _ = choose_fruit([far, too_close, near], arm)
        self.assertEqual(chosen.class_name, "orange")
        self.assertIsNone(choose_fruit([far], arm))


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
