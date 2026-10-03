#!/usr/bin/env python3
"""Chuẩn bị graph DA3 và wheel NVIDIA đã pin, không thay stack JetPack."""

import argparse
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grasppose.infrastructure.artifacts import verify_sha256
from grasppose.infrastructure.settings import (
    DA3_PROVIDER,
    DA3_WEIGHTS,
    DA3_WEIGHTS_SHA256,
)
from scripts.setup.fetch_prebuilt_trt import read_dependencies

MODEL_NAME = "da3metric-large-fp32"
ORT_NAME = "onnxruntime-gpu-jetson"


def da3_records(path=ROOT / "dependencies"):
    records = read_dependencies(path)
    model, wheel = records[MODEL_NAME], records[ORT_NAME]
    for record in (model, wheel):
        if not re.fullmatch(r"[a-f0-9]{64}", record.get("SHA256", "")):
            raise ValueError("DA3 dependency needs a valid SHA-256")
        if not record.get("URL", "").startswith("https://"):
            raise ValueError("DA3 dependency needs an HTTPS URL")
    if (
        model.get("KIND") != "onnx"
        or model.get("PRECISION") != "fp32"
        or model["SHA256"] != DA3_WEIGHTS_SHA256
    ):
        raise ValueError("DA3 graph must match the adapter's pinned FP32 weights")
    if (
        wheel.get("KIND") != "wheel"
        or wheel.get("PYTHON") != "3.8"
        or wheel.get("ARCH") != "aarch64"
        or wheel.get("VERSION") != "1.16.0"
        or wheel.get("FILENAME") != "onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl"
    ):
        raise ValueError("ORT wheel must be the pinned Python 3.8/aarch64 build")
    return model, wheel


def check_host():
    if platform.machine() != "aarch64" or sys.version_info[:2] != (3, 8):
        raise RuntimeError("DA3 preparation requires Jetson aarch64 / Python 3.8")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("run DA3 preparation with the checkout's .venv Python")


def fetch_verified(record, target):
    """Không kích hoạt file tải dở/sai hash; file cũ còn nguyên khi lỗi."""
    target = Path(target)
    if target.is_file():
        try:
            verify_sha256(str(target), record["SHA256"], record["NAME"])
            return target
        except RuntimeError:
            pass
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".download-", dir=str(target.parent)
    ) as temp:
        downloaded = Path(temp) / "artifact"
        subprocess.run(
            [
                "curl",
                "--silent",
                "--show-error",
                "--location",
                "--fail",
                "--retry",
                "3",
                "--connect-timeout",
                "20",
                "--max-time",
                "1800",
                record["URL"],
                "--output",
                str(downloaded),
            ],
            check=True,
        )
        verify_sha256(str(downloaded), record["SHA256"], record["NAME"])
        os.replace(str(downloaded), str(target))
    return target


def check_ort_runtime(ort, record, provider=DA3_PROVIDER):
    if ort.__version__ != record["VERSION"]:
        raise RuntimeError(
            "ONNX Runtime expected %s, got %s" % (record["VERSION"], ort.__version__)
        )
    if provider not in ort.get_available_providers():
        raise RuntimeError(
            "ONNX Runtime has no %s; got %s" % (provider, ort.get_available_providers())
        )


def install_ort(wheel_path, record):
    # This wheel declares NumPy>=1.24.4, but the JetPack stack is pinned at
    # 1.23.5. Keep it, as in Ultralytics' JP5 guide; mandatory real DA3 smoke
    # in check_env.py decides whether the native runtime works on this host.
    # Other wheel dependencies are explicitly installed via requirements.txt.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--force-reinstall",
            str(wheel_path),
        ],
        check=True,
    )
    # Use a clean interpreter after replacing a native extension.
    probe = (
        "import onnxruntime as ort; "
        "from scripts.setup.prepare_da3 import check_ort_runtime; "
        "check_ort_runtime(ort, %s, %r)"
        % (repr({"VERSION": record["VERSION"]}), DA3_PROVIDER)
    )
    subprocess.run([sys.executable, "-c", probe], cwd=str(ROOT), check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies", default=str(ROOT / "dependencies"))
    parser.add_argument("--weights", default=DA3_WEIGHTS)
    args = parser.parse_args(argv)
    model, wheel = da3_records(args.dependencies)
    check_host()
    wheel_path = fetch_verified(
        wheel, ROOT / ".runtime" / "downloads" / wheel["FILENAME"]
    )
    install_ort(wheel_path, wheel)
    model_path = fetch_verified(model, args.weights)
    print("DA3 FP32 ready:", model_path)
    print("Run env/check_env.py for real CUDA/depth smoke before starting the worker.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        subprocess.CalledProcessError,
    ) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
