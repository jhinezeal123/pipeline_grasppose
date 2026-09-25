#!/usr/bin/env python3
"""Require full-pipeline parity before enabling the fixed YOLOE FP32 engine."""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grasppose.artifacts import atomic_write_json
from grasppose.prompt_catalog import PromptCatalog


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts_json")
    parser.add_argument(
        "--validation-dir",
        default=os.environ.get(
            "YOLOE_VALIDATION_DIR",
            str(ROOT / "model" / "validation" / "yoloe"),
        ),
    )
    parser.add_argument(
        "--camera-k", nargs=4, type=float, required=True,
        metavar=("FX", "FY", "CX", "CY"),
    )
    parser.add_argument(
        "--camera-k-size", nargs=2, type=int,
        metavar=("WIDTH", "HEIGHT"),
    )
    args = parser.parse_args(argv)
    catalog = PromptCatalog.load(verify_engine=True)
    manifest_path = Path(catalog.artifact_dir) / "manifest.json"
    original = copy.deepcopy(catalog.manifest)
    trial = copy.deepcopy(original)
    trial.pop("full_pipeline_validation", None)
    atomic_write_json(str(manifest_path), trial)
    passed = False
    try:
        command = [
            sys.executable,
            str(ROOT / "tools" / "validate_trt_parity.py"),
            str(Path(args.prompts_json).resolve()),
            "--validation-dir", str(Path(args.validation_dir).resolve()),
            "--camera-k", *[str(value) for value in args.camera_k],
        ]
        if args.camera_k_size is not None:
            command.extend(
                ["--camera-k-size", *[str(value) for value in args.camera_k_size]])
        print("Checking full pipeline with YOLOE FP32 ...", flush=True)
        result = subprocess.run(command, cwd=str(ROOT))
        if result.returncode != 0:
            raise RuntimeError("YOLOE FP32 failed full-pipeline parity")
        trial["full_pipeline_validation"] = {
            "passed": True,
            "artifact_id": trial["artifact_id"],
            "engine_sha256": trial["engine"]["sha256"],
            "prompt_ids": [item["id"] for item in trial["prompts"]],
            "camera_k": list(args.camera_k),
            "camera_k_size": args.camera_k_size,
        }
        atomic_write_json(str(manifest_path), trial)
        passed = True
        print("YOLOE FP32 passed full-pipeline parity")
        return 0
    finally:
        if not passed:
            atomic_write_json(str(manifest_path), original)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(1)
