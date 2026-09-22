"""Repository paths and domain constants."""

import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "model")
GRASPNESS_HOME = os.environ.get("GRASPNESS_HOME",
                                os.path.join(MODEL_DIR, "graspness_unofficial"))
VOXEL = 0.005
GRIP_HW_OPEN_M = 0.0694
GRIP_MAX_OPEN_M = 0.080
DEFAULT_PROMPT = "the object"
