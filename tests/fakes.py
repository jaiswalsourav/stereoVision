"""Test doubles: a fake PCA9685 ServoKit that records every command."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stereo_vision.arm import Arm  # noqa: E402


class FakeServo:
    def __init__(self, log, channel):
        self.log, self.channel = log, channel
        self.pulse_range = None
        self._angle = None

    def set_pulse_width_range(self, lo, hi):
        self.pulse_range = (lo, hi)

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        assert value is None or 0.0 <= value <= 180.0, value
        self._angle = value
        self.log.append((self.channel, value))


class FakeKit:
    def __init__(self):
        self.log = []
        self.servo = [FakeServo(self.log, ch) for ch in range(16)]

    def angles(self, channel):
        return [v for ch, v in self.log if ch == channel]


def make_arm(**kwargs):
    """Arm on a fake servo board, without real-time waits or printing."""
    kit = FakeKit()
    kwargs.setdefault("realtime", False)
    kwargs.setdefault("verbose", False)
    arm = Arm(kit=kit, **kwargs)
    return arm, kit
