#!/usr/bin/env python3
"""Create or verify the VGN TensorRT/checkpoint checksum manifest."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grasppose.artifacts import atomic_write_json, sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "write"))
    parser.add_argument("--checkpoint", default=str(ROOT / "model" / "vgn_conv.pth"))
    parser.add_argument("--engine", default=str(ROOT / "model" / "vgn.engine"))
    parser.add_argument("--manifest", default=str(ROOT / "model" / "runtime" / "vgn.json"))
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).resolve()
    engine = Path(args.engine).resolve()
    manifest_path = Path(args.manifest).resolve()
    if not checkpoint.is_file() or not engine.is_file():
        raise RuntimeError("VGN checkpoint or TensorRT engine is missing")
    actual = {
        "schema_version": 1,
        "checkpoint_sha256": sha256_file(str(checkpoint)),
        "engine_sha256": sha256_file(str(engine)),
    }
    if args.action == "check":
        if not manifest_path.is_file():
            return 1
        try:
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            return 1
        return 0 if all(current.get(k) == v for k, v in actual.items()) else 1
    atomic_write_json(str(manifest_path), actual)
    print("Wrote VGN runtime manifest:", manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise
