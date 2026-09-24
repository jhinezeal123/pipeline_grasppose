#!/usr/bin/env python3
"""Install pinned Jetson TensorRT bundles from the URLs in dependencies."""

import argparse
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
sys.path.insert(0, str(ROOT))

from grasppose.artifacts import sha256_file

BUNDLES = {
    "yoloe-26s-cube-trt-fp32": {
        "dest": "model/runtime/yoloe",
        "files": {"manifest.json", "prompts.npz", "yoloe_fp32.engine"},
        "engine": "yoloe_fp32.engine",
    },
    "lite-mono-trt-fp32": {
        "dest": "model/runtime/lite-mono",
        "files": {"manifest.json", "litemono_fp32.engine"},
        "engine": "litemono_fp32.engine",
    },
}
HEX64 = re.compile(r"^[a-f0-9]{64}$")


def read_dependencies(path):
    records = {}
    current = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines() + ["---"]:
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
            raise ValueError("invalid/duplicate dependency field: %r" % key)
        current[key] = value
    return records


def require_record(record, name):
    spec = BUNDLES[name]
    for field in ("URL", "SHA256", "ARTIFACT_ID", "TENSORRT", "L4T", "GPU"):
        if not record.get(field):
            raise ValueError("%s is missing %s in dependencies" % (name, field))
    if record.get("KIND") != "tar.gz" or record.get("DEST") != spec["dest"]:
        raise ValueError("%s has an invalid bundle kind or destination" % name)
    if record.get("PRECISION") != "fp32":
        raise ValueError("%s must use FP32" % name)
    if not record["URL"].startswith("https://github.com/jhinezeal123/pipeline_grasppose/releases/download/"):
        raise ValueError("%s has an unexpected release URL" % name)
    if not HEX64.fullmatch(record["SHA256"]) or not HEX64.fullmatch(record["ARTIFACT_ID"]):
        raise ValueError("%s has an invalid SHA-256/artifact ID" % name)


def check_host(record):
    if platform.machine() != "aarch64":
        raise RuntimeError("prebuilt TensorRT engines require aarch64 AGX Xavier")
    compatible = Path("/proc/device-tree/compatible").read_bytes()
    if record["GPU"].encode("ascii") not in compatible:
        raise RuntimeError("prebuilt TensorRT engine GPU does not match this device")
    l4t = Path("/etc/nv_tegra_release").read_text(errors="replace")
    expected = record["L4T"]
    major, revision = expected[1:].split(".", 1)
    if "R%s" % major not in l4t or "REVISION: %s" % revision not in l4t:
        raise RuntimeError("prebuilt TensorRT engine requires L4T %s" % expected)
    import tensorrt
    if tensorrt.__version__ != record["TENSORRT"]:
        raise RuntimeError(
            "prebuilt TensorRT engine requires TensorRT %s; got %s"
            % (record["TENSORRT"], tensorrt.__version__)
        )


def _verify_file(path, expected, label):
    if not HEX64.fullmatch(str(expected or "")) or not path.is_file():
        raise RuntimeError("%s is missing or lacks SHA-256: %s" % (label, path))
    actual = sha256_file(str(path))
    if actual != expected:
        raise RuntimeError("%s SHA-256 mismatch: %s" % (label, path))


def verify_artifact(name, record, artifact_dir, source_root):
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError("prebuilt manifest is missing: %s" % manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_id = record["ARTIFACT_ID"]
    engine = manifest.get("engine", {})
    spec = BUNDLES[name]
    if manifest.get("schema_version") != 1 or manifest.get("artifact_id") != artifact_id:
        raise RuntimeError("prebuilt artifact ID/schema mismatch")
    if (engine.get("precision") != "fp32"
            or engine.get("file") != spec["engine"]
            or manifest.get("build", {}).get("tensorrt") != record["TENSORRT"]):
        raise RuntimeError("prebuilt engine precision/name/TensorRT mismatch")
    _verify_file(artifact_dir / spec["engine"], engine.get("sha256"), "TensorRT engine")

    if name == "yoloe-26s-cube-trt-fp32":
        if manifest.get("precision_policy") != "fp32_only":
            raise RuntimeError("prebuilt YOLOE engine is not locked to FP32")
        prompts = manifest.get("prompts", [])
        prompt_ids = [item.get("id") for item in prompts]
        if prompt_ids != record.get("PROMPT_IDS", "").split(","):
            raise RuntimeError("prebuilt YOLOE prompts do not match dependencies")
        source = manifest.get("source", {})
        profile = source.get("profile", {})
        if profile.get("file") != "prompts.npz":
            raise RuntimeError("prebuilt YOLOE profile name mismatch")
        _verify_file(artifact_dir / "prompts.npz", profile.get("sha256"), "YOLOE prompt profile")
        _verify_file(source_root / "model/yoloe-26s-seg.pt",
                     source.get("checkpoint_sha256"), "YOLOE checkpoint")
        _verify_file(source_root / "mobileclip2_b.ts",
                     source.get("text_encoder_sha256"), "YOLOE text encoder")
        parity = manifest.get("full_pipeline_validation", {})
        if (parity.get("passed") is not True
                or parity.get("artifact_id") != artifact_id
                or parity.get("engine_sha256") != engine.get("sha256")
                or parity.get("prompt_ids") != prompt_ids):
            raise RuntimeError("prebuilt YOLOE engine lacks matching full-pipeline parity")
        identity = {
            "model_sha256": source["checkpoint_sha256"],
            "text_encoder_sha256": source["text_encoder_sha256"],
            "prompts": [{"id": item["id"], "text": item["text"]} for item in prompts],
            "validation_sha256": manifest["validation_images"],
            "imgsz": manifest["imgsz"],
            "conf": manifest["conf"],
            "ultralytics": manifest["build"]["ultralytics"],
            "tensorrt": manifest["build"]["tensorrt"],
        }
    else:
        if manifest.get("model", {}).get("name") != "lite-mono":
            raise RuntimeError("prebuilt Lite-Mono model name mismatch")
        if manifest["model"].get("source_revision") != record.get("SOURCE_REVISION"):
            raise RuntimeError("prebuilt Lite-Mono source revision mismatch")
        for filename in ("encoder.pth", "depth.pth"):
            _verify_file(source_root / "model/lite-mono" / filename,
                         manifest.get("weights", {}).get(filename),
                         "Lite-Mono %s" % filename)
        depth = manifest.get("depth", {})
        if float(depth.get("selected_validation_p95_relative_error", 1.0)) > 0.02:
            raise RuntimeError("prebuilt Lite-Mono engine failed depth parity")
        identity = {
            "weights": manifest["weights"],
            "model_name": manifest["model"]["name"],
            "validation": manifest["validation_images"],
            "tensorrt": manifest["build"]["tensorrt"],
            "graph": "static-192x640-v1",
        }
    calculated = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if calculated != artifact_id:
        raise RuntimeError("prebuilt manifest identity does not match artifact ID")
    return manifest


def _extract_archive(archive_path, artifact_id, staging, files):
    expected = {artifact_id + "/" + name for name in files}
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if {item.name for item in members} != expected:
            raise RuntimeError("prebuilt bundle has unexpected files")
        for member in members:
            if not member.isfile():
                raise RuntimeError("prebuilt bundle contains a non-file entry")
            stream = archive.extractfile(member)
            if stream is None:
                raise RuntimeError("cannot read prebuilt bundle entry")
            with stream, (staging / Path(member.name).name).open("wb") as output:
                shutil.copyfileobj(stream, output)


def _write_current(destination, artifact_id):
    temporary = destination / "CURRENT.tmp"
    temporary.write_text(artifact_id + "\n", encoding="ascii")
    os.replace(str(temporary), str(destination / "CURRENT"))


def install_bundle(name, record, install_root=ROOT, source_root=ROOT,
                   archive_path=None):
    require_record(record, name)
    spec = BUNDLES[name]
    destination = Path(install_root) / spec["dest"]
    destination.mkdir(parents=True, exist_ok=True)
    artifact_id = record["ARTIFACT_ID"]
    target = destination / artifact_id
    if target.is_dir():
        verify_artifact(name, record, target, Path(source_root))
        result = "reused"
    else:
        with tempfile.TemporaryDirectory(prefix=".download-", dir=str(destination)) as temporary:
            temporary = Path(temporary)
            bundle = Path(archive_path) if archive_path else temporary / "bundle.tar.gz"
            if archive_path is None:
                subprocess.run([
                    "curl", "--silent", "--show-error", "--location",
                    "--fail", "--retry", "3",
                    "--connect-timeout", "20", "--max-time", "600",
                    record["URL"], "--output", str(bundle),
                ], check=True)
            _verify_file(bundle, record["SHA256"], "release bundle")
            staging = temporary / "artifact"
            staging.mkdir()
            _extract_archive(bundle, artifact_id, staging, spec["files"])
            verify_artifact(name, record, staging, Path(source_root))
            os.replace(str(staging), str(target))
        result = "downloaded"
    current = destination / "CURRENT"
    existing = current.read_text(encoding="ascii").strip() if current.is_file() else ""
    if name != "yoloe-26s-cube-trt-fp32" or not existing or existing == artifact_id:
        _write_current(destination, artifact_id)
    print("%s: %s %s" % (name, result, target))
    return target


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dependencies", default=str(ROOT / "dependencies"))
    parser.add_argument("--install-root", default=str(ROOT))
    parser.add_argument("--archive-dir", default=None,
                        help="use local release bundles for offline verification")
    args = parser.parse_args(argv)
    records = read_dependencies(args.dependencies)
    for name in BUNDLES:
        record = records.get(name)
        if record is None:
            raise RuntimeError("missing %s from dependencies" % name)
        require_record(record, name)
        check_host(record)
        archive = (Path(args.archive_dir) / Path(record["URL"]).name
                   if args.archive_dir else None)
        install_bundle(name, record, args.install_root, ROOT, archive)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(1)
