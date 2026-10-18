"""Tests for triangulation, matching and arm kinematics (no cameras needed).

    python -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stereo_vision.arm import UnreachableError, forward_kinematics, inverse_kinematics  # noqa: E402
from stereo_vision.detection import Detection, match_detections  # noqa: E402
from stereo_vision.triangulation import find_depth, focal_length_px, point_3d  # noqa: E402

FRAME = np.zeros((480, 640, 3), np.uint8)


def project(xyz, baseline=3.6):
    """Project a camera-frame point into the left and right images."""
    f = focal_length_px(640)
    x, y, z = xyz
    left = (320 + f * x / z, 240 + f * y / z)
    right = (320 + f * (x - baseline) / z, 240 + f * y / z)
    return left, right


class TriangulationTest(unittest.TestCase):
    def test_round_trip(self):
        for xyz in [(0, 0, 50), (10, -5, 80), (-7, 4, 30)]:
            left, right = project(xyz)
            np.testing.assert_allclose(point_3d(left, right, FRAME.shape), xyz, atol=1e-6)
            self.assertAlmostEqual(find_depth(right, left, FRAME, FRAME), xyz[2], places=6)

    def test_bad_disparity(self):
        self.assertIsNone(find_depth((100, 0), (100, 0), FRAME, FRAME))
        self.assertIsNone(point_3d((90, 0), (100, 0), FRAME.shape))


class MatchingTest(unittest.TestCase):
    def test_pairs_same_object(self):
        left = [Detection("bottle", 0.9, (300, 100, 40, 120)),
                Detection("cup", 0.8, (100, 300, 50, 50))]
        right = [Detection("cup", 0.8, (80, 302, 50, 50)),
                 Detection("bottle", 0.9, (270, 101, 40, 118))]
        pairs = match_detections(left, right)
        self.assertEqual({(l.class_name, r.class_name) for l, r in pairs},
                         {("bottle", "bottle"), ("cup", "cup")})

    def test_rejects_wrong_row_or_side(self):
        left = [Detection("bottle", 0.9, (300, 100, 40, 120))]
        self.assertEqual(match_detections(left, [Detection("bottle", 0.9, (270, 300, 40, 120))]), [])
        self.assertEqual(match_detections(left, [Detection("bottle", 0.9, (330, 100, 40, 120))]), [])


class ArmTest(unittest.TestCase):
    def test_ik_fk_round_trip(self):
        for target in [(0, -10, 30), (10, 0, 25), (-8, -15, 20), (5, 5, 30)]:
            angles = inverse_kinematics(*target)
            np.testing.assert_allclose(forward_kinematics(**angles), target, atol=1e-6)

    def test_straight_ahead_is_base_90(self):
        self.assertAlmostEqual(inverse_kinematics(0, -10, 30)["base"], 90.0)

    def test_fully_stretched(self):
        angles = inverse_kinematics(0, 0, 42)
        self.assertAlmostEqual(angles["elbow"], 180.0, places=3)
        self.assertAlmostEqual(angles["shoulder"], 0.0, places=3)

    def test_unreachable(self):
        with self.assertRaises(UnreachableError):
            inverse_kinematics(0, 0, 100)
        with self.assertRaises(UnreachableError):
            inverse_kinematics(0, 0, 2)


if __name__ == "__main__":
    unittest.main()
