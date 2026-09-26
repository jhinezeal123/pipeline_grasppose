"""Runtime settings for the Jetson-oriented pipeline."""

import os

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(HERE, "model"))
RUNTIME_DIR = os.environ.get("GRASP_RUNTIME_DIR", os.path.join(HERE, ".runtime"))

YOLOE_MODEL = os.environ.get("YOLOE_MODEL", os.path.join(MODEL_DIR, "yoloe-26s-seg.pt"))
YOLOE_TEXT_ENCODER = os.environ.get(
    "YOLOE_TEXT_ENCODER", os.path.join(HERE, "mobileclip2_b.ts"))
YOLOE_IMGSZ = int(os.environ.get("YOLOE_IMGSZ", "640"))
YOLOE_CONF = float(os.environ.get("YOLOE_CONF", "0.20"))
YOLOE_ARTIFACT_ROOT = os.environ.get(
    "YOLOE_ARTIFACT_ROOT", os.path.join(MODEL_DIR, "runtime", "yoloe"))
YOLOE_CURRENT_FILE = os.path.join(YOLOE_ARTIFACT_ROOT, "CURRENT")

LITEMONO_HOME = os.environ.get("LITEMONO_HOME", os.path.join(MODEL_DIR, "Lite-Mono"))
LITEMONO_WEIGHTS = os.environ.get("LITEMONO_WEIGHTS", os.path.join(MODEL_DIR, "lite-mono"))
LITEMONO_MODEL = os.environ.get("LITEMONO_MODEL", "lite-mono")
LITEMONO_DEPTH_SCALE = float(os.environ.get("LITEMONO_DEPTH_SCALE", "1.0"))
LITEMONO_ARTIFACT_ROOT = os.environ.get(
    "LITEMONO_ARTIFACT_ROOT", os.path.join(MODEL_DIR, "runtime", "lite-mono"))
LITEMONO_CURRENT_FILE = os.path.join(LITEMONO_ARTIFACT_ROOT, "CURRENT")

TSDF_SIZE_M = float(os.environ.get("TSDF_SIZE_M", "0.30"))
TSDF_RESOLUTION = int(os.environ.get("TSDF_RESOLUTION", "40"))
TSDF_TRUNC_VOXELS = float(os.environ.get("TSDF_TRUNC_VOXELS", "4.0"))

VGN_ENGINE = os.environ.get("VGN_ENGINE", os.path.join(MODEL_DIR, "vgn.engine"))
VGN_CHECKPOINT = os.environ.get("VGN_CHECKPOINT", os.path.join(MODEL_DIR, "vgn_conv.pth"))
VGN_MANIFEST = os.environ.get(
    "VGN_MANIFEST", os.path.join(MODEL_DIR, "runtime", "vgn.json"))
VGN_QUAL_THRESHOLD = float(os.environ.get("VGN_QUAL_THRESHOLD", "0.90"))

WORKER_SOCKET = os.environ.get(
    "GRASP_WORKER_SOCKET", os.path.join(RUNTIME_DIR, "worker.sock"))
WORKER_PID = os.path.join(RUNTIME_DIR, "worker.pid")
WORKER_LOG = os.path.join(RUNTIME_DIR, "worker.log")
