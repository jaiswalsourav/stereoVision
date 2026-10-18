"""Makes the stereo_vision package importable when a script is run directly,
e.g. ``python scripts/stereoVision.py`` or the VS Code Run button."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
