"""Target selection helpers shared by the arm scripts."""
import numpy as np


class StableTarget:
    """Accepts a target once it has stayed within `tolerance_cm` for `frames` frames."""

    def __init__(self, frames=5, tolerance_cm=2.0):
        self.frames = frames
        self.tolerance = tolerance_cm
        self.history = []

    def reset(self):
        self.history.clear()

    def update(self, xyz):
        """Feed the latest position (or None). Returns the averaged target when stable."""
        if xyz is None:
            self.history.clear()
            return None
        if self.history and np.linalg.norm(xyz - self.history[-1]) > self.tolerance:
            self.history.clear()
        self.history.append(xyz)
        if len(self.history) >= self.frames:
            target = np.mean(self.history, axis=0)
            self.history.clear()
            return target
        return None
