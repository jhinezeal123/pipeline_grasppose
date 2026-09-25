"""Runtime configuration for the Jetson-oriented pipeline."""

import json
import os

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(HERE, "model")
RUNTIME_CONFIG = os.path.join(HERE, ".venv", "runtime.json")
RUNTIME_DIR = os.environ.get("GRASP_RUNTIME_DIR", os.path.join(HERE, ".runtime"))
WORKER_SOCKET = os.environ.get(
    "GRASP_WORKER_SOCKET", os.path.join(RUNTIME_DIR, "worker.sock"))
WORKER_PID = os.path.join(RUNTIME_DIR, "worker.pid")
WORKER_LOG = os.path.join(RUNTIME_DIR, "worker.log")


def _load_runtime_defaults():
    try:
        with open(RUNTIME_CONFIG, "r") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    return {
        key: value
        for key, value in data.items()
        if isinstance(key, str) and isinstance(value, str)
    }


_RUNTIME_DEFAULTS = _load_runtime_defaults()


def _runtime_path(name, fallback):
    return os.environ.get(name, _RUNTIME_DEFAULTS.get(name, fallback))


YOLOE_SOURCE_MODEL = os.environ.get(
    "YOLOE_SOURCE_MODEL",
    os.path.join(MODEL_DIR, "yoloe-26s-seg.pt"),
)
YOLOE_MODEL = _runtime_path(
    "YOLOE_MODEL",
    os.path.join(MODEL_DIR, "yoloe-26s-seg.engine"),
)
YOLOE_CLASSES = (
    "blue cube",
    "yellow ball",
    "blue cylinder",
)
YOLOE_IMGSZ = int(os.environ.get("YOLOE_IMGSZ", "640"))
YOLOE_CONF = float(os.environ.get("YOLOE_CONF", "0.20"))

LITEMONO_ONNX = _runtime_path(
    "LITEMONO_ONNX",
    os.path.join(MODEL_DIR, "lite-mono-tiny_192x640_op11.onnx"),
)
LITEMONO_ENGINE = _runtime_path(
    "LITEMONO_ENGINE",
    os.path.join(MODEL_DIR, "lite-mono-tiny_192x640_op11_fp16.engine"),
)
LITEMONO_TRT_LIBRARY = _runtime_path(
    "LITEMONO_TRT_LIBRARY",
    os.path.join(HERE, "build", "litemono_trt", "liblitemono_trt.so"),
)
LITEMONO_DEPTH_SCALE = float(os.environ.get("LITEMONO_DEPTH_SCALE", "1.0"))

TSDF_SIZE_M = float(os.environ.get("TSDF_SIZE_M", "0.30"))
TSDF_RESOLUTION = int(os.environ.get("TSDF_RESOLUTION", "40"))
TSDF_TRUNC_VOXELS = float(os.environ.get("TSDF_TRUNC_VOXELS", "4.0"))

VGN_ENGINE = _runtime_path(
    "VGN_ENGINE",
    os.path.join(MODEL_DIR, "vgn.engine"),
)
# Matches ethz-asl/vgn detection.select default threshold.
VGN_QUAL_THRESHOLD = float(os.environ.get("VGN_QUAL_THRESHOLD", "0.90"))

GRIP_HW_OPEN_M = 0.0694
GRIP_MAX_OPEN_M = 0.080
DEFAULT_PROMPT = YOLOE_CLASSES[0]
