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

# Depth Anything 3 metric-large. Upstream checkpoint depth-anything/DA3METRIC-LARGE
# (Apache-2.0, monocular metric depth); the ONNX export is
# Heliosoph/da3metric-large-onnx. The model is a DINOv2 ViT-L with a single-channel
# DPT head plus a sky head: 0.35B parameters, 3157 graph nodes, opset 17.
DA3_MODEL_ID = "depth-anything/DA3METRIC-LARGE"
DA3_MODEL_DIR = os.environ.get(
    "DA3_MODEL_DIR", os.path.join(MODEL_DIR, "da3metric_large")
)
DA3_WEIGHTS = os.environ.get(
    "DA3_WEIGHTS", os.path.join(DA3_MODEL_DIR, "model_fp16.onnx")
)
DA3_WEIGHTS_SHA256 = (
    "aa3cd58f4033728a8078271b248a1f4622ffb0b64802d718514ffd8bddc54e0d"
)

# The ONNX export has a fixed trace resolution: only the batch axis is dynamic,
# because the ViT position-embedding interpolation baked the patch-token count
# into the graph. A different resolution requires re-running the exporter.
DA3_INPUT_SIZE = int(os.environ.get("DA3_INPUT_SIZE", "504"))

# CPU by default, and this is not a performance preference. The TensorRT and
# CUDA providers return a silent constant for this fp16 graph: every pixel
# 0.9741, identical for a real photograph, a simulation frame, uniform grey and
# pure noise, with no error raised. The CPU provider yields a real depth map.
DA3_PROVIDER = os.environ.get("DA3_PROVIDER", "CPUExecutionProvider")

# The upstream recipe converts canonical depth to metres as focal_px / 300.
# Measured against the simulation test bed (8.7 million ground-truth pixels over
# ten independent scenes) that overestimates depth, and this factor corrects it:
# the held-out cube then lands within 7.4 mm mean and 14.1 mm worst error against
# a 25 mm object, which is inside the graspable tolerance.
#
# VALIDATION STATUS: fitted on simulated renders only. It has NOT been checked
# against real-world ground truth, and the figure may absorb a domain gap in the
# simulation rather than express a true camera constant. Re-measure before
# trusting absolute distances on the physical cell. Set DA3_METRIC_SCALE=1.0 to
# recover the unmodified upstream formula.
DA3_METRIC_SCALE = float(os.environ.get("DA3_METRIC_SCALE", "0.38951"))

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
