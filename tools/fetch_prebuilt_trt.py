#!/usr/bin/env python3
"""Install the pinned Xavier TensorRT release bundles without rebuilding engines."""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model"
RELEASE = (
    "https://github.com/jhinezeal123/pipeline_grasppose/releases/download/"
    "jetson-xavier-pr7-fp16-bundle-v1/"
)
ASSETS = {
    "yoloe-26s-threeclass-trt-fp16": {
        "archive": "yoloe-26s-threeclass-fp16-xavier-r35.6.4-trt8.5.2.2.tar.gz",
        "files": {
            "yoloe-26s-seg.engine": "ENGINE_SHA256",
            "yoloe-26s-seg.classes.txt": "CLASSES_SHA256",
        },
    },
    "lite-mono-tiny-trt-fp16": {
        "archive": "lite-mono-tiny-fp16-xavier-r35.6.4-trt8.5.2.2.tar.gz",
        "files": {
            "lite-mono-tiny_192x640_op11_fp16.engine": "ENGINE_SHA256",
            "lite-mono-tiny_192x640_op11.onnx": "ONNX_SHA256",
        },
    },
    "vgn-trt-xavier": {
        "archive": "vgn-fp16-xavier-r35.6.4-trt8.5.2.2.tar.gz",
        "files": {
            "vgn.engine": "ENGINE_SHA256",
            "vgn_conv.pth": "CHECKPOINT_SHA256",
            "manifest.json": None,
        },
    },
}
HEX64 = re.compile(r"^[a-f0-9]{64}$")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_dependencies(path):
    records = {}
    current = {}
    for raw in path.read_text(encoding="utf-8").splitlines() + ["---"]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line == "---":
            if current:
                name = current.get("NAME")
                if not name or name in records:
                    raise ValueError("missing or duplicate dependency NAME: %r" % name)
                records[name] = current
                current = {}
            continue
        if "=" not in line:
            raise ValueError("invalid dependencies line: %r" % line)
        key, value = [part.strip() for part in line.split("=", 1)]
        if not key or not value or key in current:
            raise ValueError("invalid dependency field: %r" % line)
        current[key] = value
    return records


def require_record(name, record, spec):
    if record.get("KIND") != "tar.gz" or record.get("DEST") != "model":
        raise ValueError("%s must be a model tar.gz bundle" % name)
    if record.get("URL") != RELEASE + spec["archive"]:
        raise ValueError("%s URL must point to the pinned PR7 release" % name)
    if record.get("PRECISION") != "fp16":
        raise ValueError("%s must declare FP16 precision" % name)
    if (record.get("GPU"), record.get("L4T"), record.get("TENSORRT")) != (
            "tegra194", "R35.6.4", "8.5.2.2"):
        raise ValueError("%s has an incompatible Xavier build profile" % name)
    for field in ("SHA256",) + tuple(
            field for field in spec["files"].values() if field):
        if not HEX64.fullmatch(record.get(field, "")):
            raise ValueError("%s has invalid %s" % (name, field))


def check_host():
    if platform.machine() != "aarch64":
        raise RuntimeError("release engines require aarch64 AGX Xavier")
    compatible = Path("/proc/device-tree/compatible").read_bytes()
    if b"tegra194" not in compatible:
        raise RuntimeError("release engines require Xavier tegra194")
    l4t = Path("/etc/nv_tegra_release").read_text(errors="replace")
    if "R35" not in l4t or "REVISION: 6.4" not in l4t:
        raise RuntimeError("release engines require L4T R35.6.4")
    import tensorrt
    if tensorrt.__version__ != "8.5.2.2":
        raise RuntimeError(
            "release engines require TensorRT 8.5.2.2; got %s"
            % tensorrt.__version__)


def verify_files(name, record, spec, directory):
    for filename, field in spec["files"].items():
        path = directory / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("%s is missing %s" % (name, filename))
        if field and sha256_file(path) != record[field]:
            raise RuntimeError("%s SHA-256 mismatch: %s" % (name, filename))
    if name == "yoloe-26s-threeclass-trt-fp16":
        classes = (directory / "yoloe-26s-seg.classes.txt").read_text(
            encoding="utf-8").splitlines()
        if classes != ["blue cube", "yellow ball", "blue cylinder"]:
            raise RuntimeError("YOLOE release class order is wrong")
    if name == "vgn-trt-xavier":
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest.get("schema_version") != 1
                or manifest.get("engine", {}).get("sha256") != record["ENGINE_SHA256"]
                or manifest.get("checkpoint", {}).get("sha256") != record["CHECKPOINT_SHA256"]):
            raise RuntimeError("VGN release manifest does not match the engines")
        build = manifest.get("build", {})
        if (build.get("tensorrt") != "8.5.2.2"
                or build.get("l4t") != "R35.6.4"
                or build.get("gpu") != "tegra194"
                or build.get("trtexec_flag") != "--fp16"):
            raise RuntimeError("VGN release manifest has wrong build profile")


def install(name, record, spec):
    require_record(name, record, spec)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        verify_files(name, record, spec, MODEL_DIR)
        print("%s: reused verified files" % name)
        return
    except (OSError, RuntimeError, ValueError, KeyError):
        pass

    with tempfile.TemporaryDirectory(prefix=".download-trt-", dir=str(MODEL_DIR)) as temp:
        temp = Path(temp)
        archive_path = temp / spec["archive"]
        subprocess.run([
            "curl", "--silent", "--show-error", "--location", "--fail",
            "--retry", "3", "--connect-timeout", "20", "--max-time", "600",
            record["URL"], "--output", str(archive_path),
        ], check=True)
        if sha256_file(archive_path) != record["SHA256"]:
            raise RuntimeError("%s archive SHA-256 mismatch" % name)
        staging = temp / "staging"
        staging.mkdir()
        with tarfile.open(archive_path, "r:gz") as archive:
            members = archive.getmembers()
            if len(members) != len(spec["files"]) or {
                    member.name for member in members} != set(spec["files"]):
                raise RuntimeError("%s archive has unexpected members" % name)
            for member in members:
                if not member.isfile():
                    raise RuntimeError("%s archive contains a non-file" % name)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError("%s archive member cannot be read" % name)
                with source, (staging / member.name).open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        verify_files(name, record, spec, staging)
        for filename in spec["files"]:
            os.replace(str(staging / filename), str(MODEL_DIR / filename))
    print("%s: downloaded verified release bundle" % name)


def main():
    records = read_dependencies(ROOT / "dependencies")
    for name in ASSETS:
        if name not in records:
            raise RuntimeError("missing %s in dependencies" % name)
        require_record(name, records[name], ASSETS[name])
    check_host()
    for name, spec in ASSETS.items():
        install(name, records[name], spec)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        sys.exit("TensorRT bundle install failed: %s: %s" % (type(exc).__name__, exc))
