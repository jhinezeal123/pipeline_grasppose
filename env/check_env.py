#!/usr/bin/env python3
"""Jetson AGX Xavier / L4T R35.6.4 (JetPack 5.1.6) compatibility preflight."""

import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# When executed as "python env/check_env.py", Python puts env/ rather than
# the repository root on sys.path. Add ROOT explicitly because this repository
# is intentionally not installed as a site-package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grasppose.config import (
    LITEMONO_ENGINE,
    LITEMONO_ONNX,
    LITEMONO_TRT_LIBRARY,
    VGN_ENGINE,
    YOLOE_MODEL,
)

ENV = ROOT / ".venv"


def _artifact_path(value):
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


YOLOE_ENGINE_PATH = _artifact_path(YOLOE_MODEL)
LITEMONO_ONNX_PATH = _artifact_path(LITEMONO_ONNX)
LITEMONO_ENGINE_PATH = _artifact_path(LITEMONO_ENGINE)
LITEMONO_TRT_LIBRARY_PATH = _artifact_path(LITEMONO_TRT_LIBRARY)
VGN_ENGINE_PATH = _artifact_path(VGN_ENGINE)

REQUIRED = (
    "numpy", "scipy", "torch", "torchvision", "cv2", "PIL",
    "ultralytics", "timm", "onnx", "gradio", "gdown", "tensorrt",
)


def _version_tuple(text):
    values = []
    for part in str(text).split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        values.append(int(digits))
    return tuple(values)


def _host_protected_versions():
    stamp = ENV / "host.json"
    if not stamp.is_file():
        return {}
    try:
        return json.loads(stamp.read_text()).get("protected", {})
    except Exception:
        return {}


def _mem_total_gib():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                kib = int(line.split()[1])
                return kib / (1024.0 * 1024.0)
    except Exception:
        pass
    return None


def main():
    problems = []
    print("Python %d.%d.%d" % sys.version_info[:3])
    print("Machine:", platform.machine())

    compatible_path = Path("/proc/device-tree/compatible")
    if compatible_path.is_file():
        compatible = compatible_path.read_bytes().replace(
            b"\x00", b" ").decode("ascii", "replace")
        print("Device tree:", compatible.strip())
        if "tegra194" not in compatible:
            problems.append(
                "expected Jetson Xavier tegra194; device tree is %s"
                % compatible.strip()
            )

    l4t_path = Path("/etc/nv_tegra_release")
    if l4t_path.is_file():
        l4t = l4t_path.read_text(errors="replace").splitlines()[0]
        print("L4T:", l4t)
        if "R35" not in l4t or "REVISION: 6.4" not in l4t:
            problems.append(
                "expected L4T R35.6.4 / JetPack 5.1.6; got %s"
                % l4t
            )

    if sys.version_info[:2] != (3, 8):
        problems.append(
            "JetPack 5.x target expects Python 3.8; got %d.%d"
            % sys.version_info[:2]
        )
    if platform.machine() != "aarch64":
        problems.append(
            "Jetson AGX Xavier target expects aarch64; got %s"
            % platform.machine()
        )

    modules = {}
    for name in REQUIRED:
        try:
            module = importlib.import_module(name)
            modules[name] = module
            print(
                "[OK] %-12s %s"
                % (name, getattr(module, "__version__", ""))
            )
        except Exception as exc:
            print("[--] %-12s %s" % (name, exc))
            problems.append("missing/broken %s: %s" % (name, exc))

    target_versions = {
        "ultralytics": "8.4.140",
        "opencv-python": "4.8.1.78",
        "Pillow": "10.4.0",
        "timm": "0.9.16",
        "gradio": "4.44.1",
        "gdown": "5.2.0",
        "onnx": "1.14.1",
        "polars": "0.20.31",
        "matplotlib": "3.7.5",
    }
    for dist_name, expected_version in target_versions.items():
        try:
            actual = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            problems.append("missing target package %s" % dist_name)
            continue
        if actual != expected_version:
            problems.append(
                "%s version mismatch: expected=%s actual=%s"
                % (dist_name, expected_version, actual)
            )

    expected = _host_protected_versions()
    protected_modules = {
        "torch": modules.get("torch"),
        "torchvision": modules.get("torchvision"),
        "numpy": modules.get("numpy"),
        "scipy": modules.get("scipy"),
    }
    for dist_name, module in protected_modules.items():
        expected_version = expected.get(dist_name)
        if not expected_version or module is None:
            continue
        actual = str(getattr(module, "__version__", "")).strip()
        if actual != expected_version:
            problems.append(
                "%s runtime was replaced inside .venv: "
                "host=%s runtime=%s"
                % (dist_name, expected_version, actual)
            )

    torch = modules.get("torch")
    torchvision = modules.get("torchvision")
    if torch is not None:
        cuda_ok = bool(torch.cuda.is_available())
        print("CUDA available:", cuda_ok)
        if not cuda_ok:
            problems.append("PyTorch cannot see CUDA")
        else:
            device_name = torch.cuda.get_device_name(0)
            capability = tuple(torch.cuda.get_device_capability(0))
            arch_list = list(torch.cuda.get_arch_list())
            print(
                "GPU:", device_name,
                "| capability:", capability,
                "| torch CUDA:", torch.version.cuda,
                "| arch list:", arch_list,
            )
            if capability != (7, 2):
                problems.append(
                    "expected Xavier sm_72 capability (7,2); got %r"
                    % (capability,)
                )
            if "sm_72" not in arch_list:
                problems.append(
                    "JetPack Torch build does not include sm_72"
                )
            if str(torch.version.cuda) != "11.4":
                problems.append(
                    "L4T R35.6.4 target expects Torch CUDA 11.4; got %s"
                    % torch.version.cuda
                )

            cudnn_version = torch.backends.cudnn.version()
            print("cuDNN:", cudnn_version)
            if cudnn_version != 8600:
                problems.append(
                    "L4T R35.6.4 target expects cuDNN 8.6.0 "
                    "(8600); got %s" % cudnn_version
                )

            torch_version = str(getattr(torch, "__version__", ""))
            if not torch_version.startswith("2.1.0a0+"):
                problems.append(
                    "expected NVIDIA JetPack Torch 2.1.0a0 build; got %s"
                    % torch_version
                )

            if torchvision is not None:
                tv_version = str(
                    getattr(torchvision, "__version__", ""))
                if not tv_version.startswith("0.16.1"):
                    problems.append(
                        "expected torchvision 0.16.1; got %s"
                        % tv_version
                    )

            # YOLOE commonly uses torchvision.ops.nms. A mismatched generic
            # torchvision wheel can import successfully but fail here.
            if torchvision is not None:
                try:
                    from torchvision.ops import nms
                    boxes = torch.tensor(
                        [[0.0, 0.0, 10.0, 10.0],
                         [1.0, 1.0, 9.0, 9.0]],
                        device="cuda",
                    )
                    scores = torch.tensor(
                        [0.9, 0.8], device="cuda")
                    keep = nms(boxes, scores, 0.5)
                    torch.cuda.synchronize()
                    print(
                        "[OK] torchvision CUDA NMS",
                        keep.detach().cpu().tolist(),
                    )
                except Exception as exc:
                    problems.append(
                        "torchvision CUDA ops are incompatible with "
                        "JetPack Torch: %s" % exc
                    )

    trt = modules.get("tensorrt")
    if trt is not None:
        version = getattr(trt, "__version__", "0")
        if _version_tuple(version) < (8, 5):
            problems.append(
                "TensorRT >=8.5 is required; got %s" % version
            )
        if not str(version).startswith("8.5."):
            problems.append(
                "L4T R35.6.4 target expects TensorRT 8.5.x; got %s"
                % version
            )

    trtexec = (
        shutil.which("trtexec")
        or (
            "/usr/src/tensorrt/bin/trtexec"
            if os.path.isfile("/usr/src/tensorrt/bin/trtexec")
            else None
        )
    )
    print("trtexec:", trtexec or "not found")
    if not trtexec:
        problems.append("trtexec is required to build TensorRT engines")

    total_gib = _mem_total_gib()
    if total_gib is not None:
        print("System RAM: %.1f GiB" % total_gib)
        if total_gib < 28.0:
            problems.append(
                "expected 32 GB Xavier SKU (>=28 GiB visible RAM)"
            )

    free_gib = shutil.disk_usage(str(ROOT)).free / (1024.0 ** 3)
    print("Free disk: %.1f GiB" % free_gib)
    if free_gib < 8.0:
        problems.append(
            "less than 8 GiB free disk; model/build artifacts may fail"
        )

    artifacts = (
        (YOLOE_MODEL, YOLOE_ENGINE_PATH),
        ("model/yoloe-26s-seg.classes.txt", ROOT / "model/yoloe-26s-seg.classes.txt"),
        (LITEMONO_ONNX, LITEMONO_ONNX_PATH),
        (LITEMONO_ENGINE, LITEMONO_ENGINE_PATH),
        (LITEMONO_TRT_LIBRARY, LITEMONO_TRT_LIBRARY_PATH),
        (VGN_ENGINE, VGN_ENGINE_PATH),
    )
    for label, path in artifacts:
        ok = path.is_file() and path.stat().st_size > 0
        print("[%s] %s" % ("OK" if ok else "--", label))
        if not ok:
            problems.append("missing artifact %s" % label)

    # Deserialize and execute the exported YOLOE TensorRT engine in a clean
    # subprocess. No text encoder or set_classes() call is allowed at runtime.
    if not problems:
        yoloe_smoke = r"""
import sys

import numpy as np

from grasppose.config import YOLOE_CLASSES
from grasppose.facade import DEFAULT_SERVICE

if "torch" in sys.modules:
    raise RuntimeError(
        "Torch was imported before YOLOE TensorRT load during service construction"
    )

vision = DEFAULT_SERVICE.core._vision
vision.load()
image = np.zeros((640, 640, 3), dtype=np.uint8)
for target in YOLOE_CLASSES:
    result = vision.predict(image, target)
    print(
        "YOLOE TensorRT smoke: target=%r boxes=%d"
        % (target, len(result.detection.boxes))
    )

try:
    vision.predict(image, "not baked")
except ValueError:
    pass
else:
    raise RuntimeError("YOLOE TensorRT accepted a prompt that was not baked")

vision.close()
"""
        completed = subprocess.run(
            [sys.executable, "-c", yoloe_smoke],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        if completed.stdout.strip():
            print(completed.stdout.strip())
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            problems.append(
                "YOLOE TensorRT smoke failed: %s"
                % detail
            )

    if not problems:
        try:
            from grasppose.adapters.lite_mono import LiteMonoDepth

            image = np.zeros((192, 640, 3), dtype=np.uint8)
            K = np.array(
                [[500.0, 0.0, 320.0],
                 [0.0, 500.0, 96.0],
                 [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            smoke = LiteMonoDepth()
            smoke.load()
            result = smoke.predict(image, camera_K=K)
            smoke.close()
            if result.depth.shape != (192, 640):
                raise RuntimeError(
                    "unexpected depth shape %r"
                    % (result.depth.shape,)
                )
            if not np.isfinite(result.depth).all():
                raise RuntimeError(
                    "Lite-Mono returned non-finite depth")
            print(
                "[OK] Lite-Mono Tiny TensorRT smoke inference | depth:",
                result.depth.shape,
            )
        except Exception as exc:
            problems.append(
                "Lite-Mono Tiny TensorRT smoke inference failed: %s: %s"
                % (type(exc).__name__, exc)
            )

    # Deserialize and execute the exact TensorRT engine once. This verifies
    # the 8.5.x API path, engine compatibility and CUDA execution on sm_72.
    if not problems:
        try:
            from grasppose.adapters.vgn_trt import VgnTensorRT
            from grasppose.domain.types import TSDFResult

            smoke = VgnTensorRT(str(VGN_ENGINE_PATH))
            smoke.load()
            tsdf = TSDFResult(
                grid=np.full(
                    (1, 40, 40, 40), 0.5, dtype=np.float32),
                voxel_size=0.0075,
                T_cam_volume=np.eye(4, dtype=np.float32),
                observed_voxels=40 ** 3,
            )
            result = smoke.predict(tsdf)
            smoke.close()
            print(
                "[OK] VGN TensorRT smoke inference | grasps:",
                len(result.graspgroup),
            )
        except Exception as exc:
            problems.append(
                "VGN TensorRT smoke inference failed: %s: %s"
                % (type(exc).__name__, exc)
            )

    if problems:
        print("Problems:")
        for problem in problems:
            print(" -", problem)
        return 1

    print("Jetson AGX Xavier environment looks compatible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
