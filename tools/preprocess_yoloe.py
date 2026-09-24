#!/usr/bin/env python3
"""Bake a closed prompt set into static YOLOE FP32/FP16 TensorRT engines."""

import argparse
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grasppose.artifacts import atomic_write_json, sha256_file
from grasppose.config import (
    YOLOE_CONF,
    YOLOE_IMGSZ,
    YOLOE_MODEL,
    YOLOE_TEXT_ENCODER,
)


PROMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def read_prompt_file(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(value, dict):
        value = value.get("prompts")
    if not isinstance(value, list) or not 1 <= len(value) <= 16:
        raise ValueError("prompts.json must contain an ordered list of 1 to 16 prompts")
    prompts = []
    seen = set()
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise ValueError("prompt %d must be an object with id and text" % index)
        prompt_id = item.get("id")
        text = item.get("text")
        if not isinstance(prompt_id, str) or not PROMPT_ID.fullmatch(prompt_id):
            raise ValueError(
                "prompt %d has an empty/invalid ID; use letters, digits, dot, '_' or '-'"
                % index
            )
        if prompt_id in seen:
            raise ValueError("duplicate prompt ID: %s" % prompt_id)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("prompt %s has empty text" % prompt_id)
        seen.add(prompt_id)
        prompts.append({
            "id": prompt_id,
            "text": text.strip(),
            "class_index": index,
        })
    return prompts


def validation_paths(prompts, directory):
    directory = Path(directory).resolve()
    paths = {}
    missing = []
    for prompt in prompts:
        candidates = [
            directory / (prompt["id"] + ext)
            for ext in IMAGE_EXTENSIONS
        ]
        found = next((path for path in candidates if path.is_file()), None)
        if found is None:
            missing.append(prompt["id"])
        else:
            paths[prompt["id"]] = found
    if missing:
        raise RuntimeError(
            "YOLOE parity gate needs one validation image per prompt ID in %s; "
            "missing: %s" % (directory, ", ".join(missing))
        )
    return paths


def bgr_image(path):
    rgb = np.asarray(Image.open(path).convert("RGB"))
    return np.ascontiguousarray(rgb[:, :, ::-1])


def configure_prompt_classes(model, prompts):
    """Bind safe prompt IDs to embeddings computed from natural-language text.

    Ultralytics YOLOE.set_classes() rejects spaces in class names, while prompt
    text is allowed to contain phrases. Compute phrase embeddings explicitly and
    use the stable prompt IDs as the exported class names.
    """
    class_names = [item["id"] for item in prompts]
    texts = [item["text"] for item in prompts]
    embeddings = model.get_text_pe(texts)
    model.set_classes(class_names, embeddings)


def best_mask(model, image, class_index, prompt_id):
    result = model.predict(
        source=image,
        imgsz=YOLOE_IMGSZ,
        conf=YOLOE_CONF,
        device=0,
        retina_masks=True,
        verbose=False,
    )[0]
    if result is None or result.boxes is None or len(result.boxes) == 0:
        return None
    classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    matches = np.flatnonzero(classes == int(class_index))
    if not len(matches) or result.masks is None or result.masks.data is None:
        return None
    best = int(matches[np.argmax(scores[matches])])
    mask = result.masks.data[best].detach().cpu().numpy()
    height, width = image.shape[:2]
    if mask.shape != (height, width):
        import cv2
        mask = cv2.resize(
            mask.astype(np.float32),
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    return mask > 0.5


def mask_iou(left, right):
    union = np.logical_or(left, right).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(left, right).sum()) / float(union)


def load_yoloe_api():
    """Import Ultralytics before Torch to preserve Xavier CUDA initialization."""
    from ultralytics import YOLO, YOLOE
    import torch
    return YOLO, YOLOE, torch


def export_candidate(checkpoint_copy, embeddings_path, staging, precision):
    _, YOLOE, torch = load_yoloe_api()

    model = YOLOE(str(checkpoint_copy))
    model.load_prompt_embeddings(str(embeddings_path))
    kwargs = {
        "format": "engine",
        "imgsz": YOLOE_IMGSZ,
        "batch": 1,
        "device": 0,
        "dynamic": False,
        "nms": False,
        "workspace": 4,
    }
    if precision == "fp16":
        kwargs["quantize"] = 16
    exported = Path(model.export(**kwargs)).resolve()
    candidate = staging / ("yoloe_%s.engine" % precision)
    if not exported.is_file():
        raise RuntimeError("Ultralytics export did not produce an engine")
    shutil.copy2(str(exported), str(candidate))
    del model
    gc.collect()
    torch.cuda.empty_cache()
    try:
        if exported.parent == staging and exported != candidate:
            exported.unlink()
    except OSError:
        pass
    return candidate


def validate_candidate(engine_path, prompts, validation, references):
    YOLO, _, torch = load_yoloe_api()

    model = YOLO(str(engine_path))
    images = {
        item["id"]: bgr_image(validation[item["id"]])
        for item in prompts
    }
    for prompt in prompts:
        model.predict(
            source=images[prompt["id"]],
            imgsz=YOLOE_IMGSZ,
            conf=YOLOE_CONF,
            device=0,
            retina_masks=True,
            verbose=False,
        )
    ious = []
    for prompt in prompts:
        image = images[prompt["id"]]
        current = best_mask(
            model, image, prompt["class_index"], prompt["id"])
        if current is None:
            ious.append(0.0)
        else:
            ious.append(mask_iou(references[prompt["id"]], current))
    timings = []
    for prompt in prompts:
        image = images[prompt["id"]]
        for _ in range(5):
            started = time.perf_counter()
            model.predict(
                source=image,
                imgsz=YOLOE_IMGSZ,
                conf=YOLOE_CONF,
                device=0,
                retina_masks=True,
                verbose=False,
            )
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000.0)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return min(ious), float(np.median(timings)), ious


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts_json")
    parser.add_argument("--validation-dir", default=os.environ.get(
        "YOLOE_VALIDATION_DIR",
        str(ROOT / "model" / "validation" / "yoloe")))
    parser.add_argument("--model", default=YOLOE_MODEL)
    parser.add_argument("--artifact-root", default=os.environ.get(
        "YOLOE_ARTIFACT_ROOT",
        str(ROOT / "model" / "runtime" / "yoloe")))
    args = parser.parse_args(argv)

    prompt_path = Path(args.prompts_json).resolve()
    if not prompt_path.is_file():
        raise RuntimeError("prompt file not found: %s" % prompt_path)
    model_path = Path(args.model).resolve()
    if not model_path.is_file():
        raise RuntimeError("YOLOE checkpoint is missing: %s" % model_path)
    prompts = read_prompt_file(prompt_path)
    validation = validation_paths(prompts, args.validation_dir)
    model_hash = sha256_file(str(model_path))
    text_encoder_path = Path(YOLOE_TEXT_ENCODER).resolve()
    if not text_encoder_path.is_file():
        raise RuntimeError("YOLOE text encoder is missing: %s" % text_encoder_path)
    text_encoder_hash = sha256_file(str(text_encoder_path))
    tensorrt_version = __import__("tensorrt").__version__
    validation_hashes = {
        item["id"]: sha256_file(str(validation[item["id"]]))
        for item in prompts
    }
    prompt_payload = [
        {"id": item["id"], "text": item["text"]}
        for item in prompts
    ]
    identity = {
        "model_sha256": model_hash,
        "text_encoder_sha256": text_encoder_hash,
        "prompts": prompt_payload,
        "validation_sha256": validation_hashes,
        "imgsz": YOLOE_IMGSZ,
        "conf": YOLOE_CONF,
        "ultralytics": __import__("ultralytics").__version__,
        "tensorrt": tensorrt_version,
    }
    artifact_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    artifact_root = Path(args.artifact_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    target = artifact_root / artifact_id
    current_path = artifact_root / "CURRENT"
    if target.is_dir():
        manifest_path = target / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            source = manifest.get("source", {})
            profile = source.get("profile", {})
            engine_meta = manifest.get("engine", {})
            profile_name = profile.get("file")
            engine_name = engine_meta.get("file")
            profile_path = target / profile_name if (
                isinstance(profile_name, str)
                and Path(profile_name).name == profile_name
            ) else None
            engine_path = target / engine_name if (
                isinstance(engine_name, str)
                and Path(engine_name).name == engine_name
            ) else None
            if (
                manifest.get("artifact_id") == artifact_id
                and manifest.get("conf") == YOLOE_CONF
                and manifest.get("build") == {
                    "ultralytics": __import__("ultralytics").__version__,
                    "tensorrt": tensorrt_version,
                }
                and source.get("checkpoint_sha256") == model_hash
                and source.get("text_encoder_sha256") == text_encoder_hash
                and profile_path is not None
                and profile_path.is_file()
                and sha256_file(str(profile_path)) == profile.get("sha256")
                and engine_path is not None
                and engine_path.is_file()
                and sha256_file(str(engine_path)) == engine_meta.get("sha256")
            ):
                temporary = current_path.with_suffix(".tmp")
                temporary.write_text(artifact_id + "\n", encoding="ascii")
                os.replace(str(temporary), str(current_path))
                print("YOLOE artifacts already prepared:", artifact_id)
                return 0
        raise RuntimeError(
            "existing YOLOE artifact is incomplete/checksum-mismatched: %s"
            % target
        )

    staging = Path(tempfile.mkdtemp(
        prefix=".yoloe-staging-", dir=str(artifact_root)))
    try:
        YOLO, YOLOE, torch = load_yoloe_api()

        text_model = YOLOE(str(model_path))
        configure_prompt_classes(text_model, prompts)
        embeddings_path = staging / "prompts.npz"
        text_model.save_prompt_embeddings(str(embeddings_path))
        del text_model
        gc.collect()
        torch.cuda.empty_cache()

        checkpoint_copy = staging / "yoloe-26s-seg.pt"
        shutil.copy2(str(model_path), str(checkpoint_copy))
        baseline = YOLOE(str(model_path))
        configure_prompt_classes(baseline, prompts)
        images = {
            item["id"]: bgr_image(validation[item["id"]])
            for item in prompts
        }
        references = {}
        for item in prompts:
            reference = best_mask(
                baseline, images[item["id"]], item["class_index"], item["id"])
            if reference is None or not reference.any():
                raise RuntimeError(
                    "PyTorch FP32 reference has no mask for prompt ID %s; "
                    "use a validation image where the target is visible"
                    % item["id"]
                )
            references[item["id"]] = reference
        results = []
        for precision in ("fp32", "fp16"):
            candidate = export_candidate(
                checkpoint_copy, embeddings_path, staging, precision)
            minimum_iou, median_ms, per_prompt_ious = validate_candidate(
                candidate, prompts, validation, references)
            passed = all(value >= 0.99 for value in per_prompt_ious)
            print(
                "%s: min mask IoU=%.5f, median=%.2f ms, gate=%s"
                % (precision, minimum_iou, median_ms,
                   "PASS" if passed else "FAIL")
            )
            results.append({
                "precision": precision,
                "file": candidate.name,
                "sha256": sha256_file(str(candidate)),
                "median_predict_ms": median_ms,
                "per_prompt_mask_iou": {
                    item["id"]: score
                    for item, score in zip(prompts, per_prompt_ious)
                },
                "passed": passed,
            })
        passing = [item for item in results if item["passed"]]
        if not passing:
            raise RuntimeError(
                "no YOLOE TensorRT candidate met mask IoU >= 0.99 "
                "for every prompt"
            )
        selected = min(passing, key=lambda item: item["median_predict_ms"])
        manifest = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "source": {
                "checkpoint": str(model_path),
                "checkpoint_sha256": model_hash,
                "text_encoder_sha256": text_encoder_hash,
                "profile": {
                    "file": embeddings_path.name,
                    "sha256": sha256_file(str(embeddings_path)),
                },
            },
            "prompts": prompts,
            "validation_images": validation_hashes,
            "imgsz": YOLOE_IMGSZ,
            "conf": YOLOE_CONF,
            "batch": 1,
            "build": {
                "ultralytics": __import__("ultralytics").__version__,
                "tensorrt": tensorrt_version,
            },
            "class_order": [item["id"] for item in prompts],
            "candidates": results,
            "engine": {
                "precision": selected["precision"],
                "file": selected["file"],
                "sha256": selected["sha256"],
                "imgsz": YOLOE_IMGSZ,
            },
            "accuracy_gate": {"mask_iou_min": 0.99},
        }
        checkpoint_copy.unlink()
        atomic_write_json(str(staging / "manifest.json"), manifest)
        os.replace(str(staging), str(target))
        temporary = current_path.with_suffix(".tmp")
        temporary.write_text(artifact_id + "\n", encoding="ascii")
        os.replace(str(temporary), str(current_path))
        print("Selected YOLOE TensorRT %s: %s"
              % (selected["precision"], target / selected["file"]))
    except Exception:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    finally:
        try:
            del baseline
        except UnboundLocalError:
            pass
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise
