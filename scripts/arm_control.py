"""Interactive robot-arm console. No cameras needed.

Use it to test the servos, find good gripper and basket angles, try moves
and plucks by coordinates, and measure the camera-to-arm transform.

    python scripts/arm_control.py              # real servos
    python scripts/arm_control.py --dry-run    # no hardware, just print

Commands (coordinates in cm, arm frame: x right, y down, z forward from the
shoulder joint; angles in degrees):

    home                      all joints to home
    where                     joint angles and gripper tip position
    joint <name> <deg>        set one joint (base, shoulder, elbow, gripper)
    jog <name> <+/-deg>       move one joint relative to where it is
    open | close | grip <deg> gripper
    goto <x> <y> <z>          move the tip to a point
    line <x> <y> <z>          move the tip there in a straight line
    pluck <x> <y> <z>         full pluck sequence for a fruit at that point
    check <x> <y> <z>         can the arm reach / pluck there? (no movement)
    speed <deg/s>             joint speed
    release                   switch servos off (arm goes limp)
    pair <cx> <cy> <cz>       record: camera sees the gripper tip at (cx, cy, cz)
    pairs | clearpairs        list / forget recorded pairs
    fit                       compute and save the camera-to-arm transform
    help | quit

Camera-to-arm transform: put a marker on the gripper tip, move the tip to 6+
spread-out positions, and at each run scripts/stereoVision.py to read the
marker's camera position, then type "pair cx cy cz". Finally type "fit".
"""
import argparse
import shlex

import _paths  # noqa: F401  (puts the repo root on sys.path)
from stereo_vision import config
from stereo_vision.arm import Arm, ArmStopped, CameraToArm, UnreachableError, plan_pluck
from stereo_vision.kinematics import inverse_kinematics


def fmt(xyz):
    return "(" + ", ".join(f"{v:.1f}" for v in xyz) + ")"


class Console:
    def __init__(self, arm):
        self.arm = arm
        self.pairs = []  # (camera_xyz, arm_xyz)

    def where(self):
        pose = "  ".join(f"{n}={a:.1f}" for n, a in self.arm.pose.items()) or "unknown"
        print(f"joints: {pose}")
        try:
            print(f"tip:    {fmt(self.arm.position())} cm")
        except UnreachableError:
            print("tip:    unknown (run 'home' first)")

    def run(self, words):
        cmd, args = words[0].lower(), words[1:]
        nums = [float(a) for a in args if _is_number(a)]
        arm = self.arm
        if cmd == "home":
            arm.home()
        elif cmd == "where":
            self.where()
        elif cmd == "joint":
            arm.move_joints({args[0]: float(args[1])})
        elif cmd == "jog":
            name = args[0]
            if name not in arm.pose:
                raise UnreachableError(f"{name} position unknown; run 'home' first")
            arm.move_joints({name: arm.pose[name] + float(args[1])})
        elif cmd == "open":
            arm.open_gripper()
        elif cmd == "close":
            arm.close_gripper()
        elif cmd == "grip":
            arm.set_gripper(nums[0])
        elif cmd == "goto":
            arm.move_to(nums[:3])
        elif cmd == "line":
            arm.move_linear(nums[:3])
        elif cmd == "pluck":
            arm.pluck_at(nums[:3])
        elif cmd == "check":
            angles = inverse_kinematics(*nums[:3], joints=arm.limits)
            print("reachable: " + "  ".join(f"{n}={a:.1f}" for n, a in angles.items()))
            plan_pluck(nums[:3], joints=arm.limits)
            print("pluck: possible")
        elif cmd == "speed":
            arm.speed_deg_s = nums[0]
            print(f"joint speed {nums[0]:.0f} deg/s")
        elif cmd == "release":
            arm.release()
        elif cmd == "pair":
            tip = arm.position()
            self.pairs.append((nums[:3], tip))
            print(f"pair {len(self.pairs)}: camera {fmt(nums[:3])} -> arm {fmt(tip)}")
        elif cmd == "pairs":
            for i, (cam, tip) in enumerate(self.pairs, 1):
                print(f"{i}: camera {fmt(cam)} -> arm {fmt(tip)}")
            if not self.pairs:
                print("no pairs recorded")
        elif cmd == "clearpairs":
            self.pairs.clear()
        elif cmd == "fit":
            tf, rms = CameraToArm.fit([c for c, _ in self.pairs], [a for _, a in self.pairs])
            tf.save(config.ARM_TRANSFORM_FILE)
            print(f"fit error {rms:.2f} cm over {len(self.pairs)} pairs "
                  f"({'good' if rms < 1.0 else 'high: add more, better-spread pairs'})")
            print(f"saved {config.ARM_TRANSFORM_FILE}")
        elif cmd == "help":
            print(__doc__.split("Commands", 1)[1])
        else:
            print(f"unknown command '{cmd}' (type help)")


def _is_number(text):
    try:
        float(text)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="no servo board, print moves")
    args = parser.parse_args()

    arm = Arm(dry_run=args.dry_run, verbose=True)
    console = Console(arm)
    print("Arm console. Type 'help' for commands, 'quit' to exit. Start with 'home'.")
    try:
        while True:
            try:
                line = input("arm> ").strip()
            except EOFError:
                break
            if not line or line.startswith("#"):
                continue
            words = shlex.split(line)
            if words[0].lower() in ("quit", "exit", "q"):
                break
            try:
                console.run(words)
            except (UnreachableError, ArmStopped, ValueError, IndexError, KeyError) as exc:
                print(f"error: {exc or 'missing or invalid argument'}")
    except KeyboardInterrupt:
        print()
    finally:
        arm.stop()


if __name__ == "__main__":
    main()
