#!/usr/bin/env python3
"""Select the fastest YOLOE engine that passes full-pipeline parity."""

import argparse
import copy
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grasppose.artifacts import atomic_write_json, verify_sha256
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
    args = parser.parse_args(argv)
    catalog = PromptCatalog.load(verify_engine=True)
    manifest_path = Path(catalog.artifact_dir) / "manifest.json"
    original = copy.deepcopy(catalog.manifest)
    candidates = sorted(
        (item for item in original.get("candidates", []) if item.get("passed")),
        key=lambda item: float(item["median_predict_ms"]),
    )
    if not candidates:
        raise RuntimeError("no YOLOE engine passed the mask parity gate")

    selected = False
    try:
        for candidate in candidates:
            name = candidate.get("file")
            if not isinstance(name, str) or Path(name).name != name:
                raise RuntimeError("invalid YOLOE candidate engine path")
            engine_path = Path(catalog.artifact_dir) / name
            verify_sha256(
                str(engine_path), candidate.get("sha256"),
                "YOLOE candidate engine",
            )
            trial = copy.deepcopy(original)
            trial.pop("full_pipeline_validation", None)
            trial.pop("selection_reason", None)
            trial["engine"] = {
                "precision": candidate["precision"],
                "file": name,
                "sha256": candidate["sha256"],
                "imgsz": trial["imgsz"],
            }
            atomic_write_json(str(manifest_path), trial)
            command = [
                sys.executable,
                str(ROOT / "tools" / "validate_trt_parity.py"),
                str(Path(args.prompts_json).resolve()),
                "--validation-dir", str(Path(args.validation_dir).resolve()),
                "--camera-k", *[str(value) for value in args.camera_k],
            ]
            print("Checking full pipeline with YOLOE %s ..." % candidate["precision"], flush=True)
            result = subprocess.run(command, cwd=str(ROOT))
            if result.returncode != 0:
                print("YOLOE %s failed full-pipeline parity" % candidate["precision"], flush=True)
                continue
            trial["full_pipeline_validation"] = {
                "passed": True,
                "artifact_id": trial["artifact_id"],
                "engine_sha256": candidate["sha256"],
                "prompt_ids": [item["id"] for item in trial["prompts"]],
                "camera_k": list(args.camera_k),
            }
            trial["selection_reason"] = (
                "fastest engine passing mask, depth, and grasp parity"
            )
            atomic_write_json(str(manifest_path), trial)
            selected = True
            print("Selected YOLOE %s after full-pipeline parity" % candidate["precision"])
            return 0
        raise RuntimeError("no YOLOE engine passed full-pipeline parity")
    finally:
        if not selected:
            atomic_write_json(str(manifest_path), original)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(1)
