"""Servo output: logical joint angles -> PCA9685 servo commands.

Kinematics works in logical angles (conventions in kinematics.py). Real
servos are mounted with small errors and sometimes mirrored, so each joint
has its own settings in config.JOINTS:

    servo command = (180 - angle if inverted else angle) + offset_deg

Commands are clamped to the joint's min/max (in logical degrees) and to the
servo's 0..180 range, so a bug elsewhere can never drive a joint past its
safety limits.
"""
from dataclasses import dataclass

from . import config


@dataclass
class Joint:
    name: str
    channel: int
    min_deg: float = 0.0
    max_deg: float = 180.0
    home_deg: float = 90.0
    offset_deg: float = 0.0
    inverted: bool = False

    def clamp(self, angle):
        return max(self.min_deg, min(self.max_deg, angle))

    def to_servo(self, angle):
        """Servo command (0..180) for a logical angle."""
        a = self.clamp(angle)
        a = 180.0 - a if self.inverted else a
        return max(0.0, min(180.0, a + self.offset_deg))


def load_joints(settings=None):
    settings = config.JOINTS if settings is None else settings
    return {name: Joint(name, **values) for name, values in settings.items()}


class ServoBus:
    """Writes joint angles to a PCA9685 board, or just records them (dry run)."""

    def __init__(self, joints, dry_run=False, kit=None):
        self.joints = joints
        self.kit = kit
        if self.kit is None and not dry_run:
            try:
                from adafruit_servokit import ServoKit
                self.kit = ServoKit(channels=16)
            except (ImportError, ValueError, RuntimeError, OSError) as exc:
                print(f"[arm] servo board unavailable ({exc}); running dry")
        if self.kit is not None:
            for joint in joints.values():
                self.kit.servo[joint.channel].set_pulse_width_range(
                    config.SERVO_MIN_PULSE_US, config.SERVO_MAX_PULSE_US)

    @property
    def is_hardware(self):
        return self.kit is not None

    def write(self, name, angle):
        """Command one joint; returns the (clamped) logical angle actually used."""
        joint = self.joints[name]
        angle = joint.clamp(angle)
        if self.kit is not None:
            self.kit.servo[joint.channel].angle = joint.to_servo(angle)
        return angle

    def release(self):
        """Stop sending pulses so the servos go limp (they can be moved by hand)."""
        if self.kit is not None:
            for joint in self.joints.values():
                self.kit.servo[joint.channel].angle = None
