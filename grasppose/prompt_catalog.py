"""Read the fixed prompt set baked into the active YOLOE engine."""

import hashlib
import json
import os
import re

from .artifacts import verify_sha256
from .config import (
    YOLOE_ARTIFACT_ROOT,
    YOLOE_CONF,
    YOLOE_CURRENT_FILE,
    YOLOE_MODEL,
)


_ARTIFACT_ID = re.compile(r"^[a-f0-9]{64}$")
_PROMPT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class PromptCatalog:
    def __init__(self, prompts, manifest, artifact_dir):
        self.prompts = tuple(prompts)
        self.manifest = manifest
        self.artifact_dir = artifact_dir
        self.by_id = {item["id"]: item for item in prompts}

    def require(self, prompt_id):
        prompt_id = str(prompt_id or "").strip()
        if prompt_id not in self.by_id:
            available = ", ".join(self.by_id) or "(none)"
            raise ValueError(
                "unknown prompt ID %r; available IDs: %s"
                % (prompt_id, available)
            )
        return self.by_id[prompt_id]

    @classmethod
    def load(cls, verify_engine=False, require_full_pipeline=False):
        if not os.path.isfile(YOLOE_CURRENT_FILE):
            raise RuntimeError(
                "prompt artifacts are not prepared; run "
                "preprocess_prompt.sh prompts.json"
            )
        with open(YOLOE_CURRENT_FILE, "r", encoding="utf-8") as handle:
            artifact_id = handle.read().strip()
        if not _ARTIFACT_ID.fullmatch(artifact_id):
            raise RuntimeError("invalid YOLOE CURRENT artifact pointer")
        artifact_dir = os.path.join(YOLOE_ARTIFACT_ROOT, artifact_id)
        manifest_path = os.path.join(artifact_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise RuntimeError("YOLOE manifest is missing: %s" % manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("schema_version") != 1:
            raise RuntimeError("unsupported YOLOE manifest schema")
        if float(manifest.get("conf", -1.0)) != YOLOE_CONF:
            raise RuntimeError("YOLOE confidence does not match artifact validation")
        if manifest.get("artifact_id") != artifact_id:
            raise RuntimeError("YOLOE CURRENT does not match manifest artifact ID")
        source = manifest.get("source")
        if not isinstance(source, dict):
            raise RuntimeError("YOLOE manifest is missing source checksums")
        verify_sha256(
            YOLOE_MODEL,
            source.get("checkpoint_sha256"),
            "YOLOE checkpoint",
        )
        profile = source.get("profile")
        if not isinstance(profile, dict):
            raise RuntimeError("YOLOE prompt embedding profile is missing")
        profile_name = profile.get("file")
        if not isinstance(profile_name, str) or os.path.basename(profile_name) != profile_name:
            raise RuntimeError("invalid YOLOE prompt profile path")
        verify_sha256(
            os.path.join(artifact_dir, profile_name),
            profile.get("sha256"),
            "YOLOE prompt embedding profile",
        )
        prompts = manifest.get("prompts")
        if not isinstance(prompts, list) or not 1 <= len(prompts) <= 16:
            raise RuntimeError("YOLOE manifest must contain 1 to 16 prompts")
        ids = set()
        for index, item in enumerate(prompts):
            prompt_id = item.get("id")
            text = item.get("text")
            if not isinstance(prompt_id, str) or not _PROMPT_ID.fullmatch(prompt_id):
                raise RuntimeError("invalid prompt ID in YOLOE manifest")
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError("empty prompt text in YOLOE manifest")
            if prompt_id in ids:
                raise RuntimeError("duplicate prompt ID in YOLOE manifest")
            if item.get("class_index") != index:
                raise RuntimeError("YOLOE prompt class ordering is invalid")
            ids.add(prompt_id)
        validation_hashes = manifest.get("validation_images")
        if not isinstance(validation_hashes, dict) or set(validation_hashes) != ids:
            raise RuntimeError("YOLOE validation image manifest is invalid")
        build = manifest.get("build")
        if not isinstance(build, dict):
            raise RuntimeError("YOLOE build metadata is missing")
        identity = {
            "model_sha256": source.get("checkpoint_sha256"),
            "text_encoder_sha256": source.get("text_encoder_sha256"),
            "prompts": [
                {"id": item["id"], "text": item["text"]}
                for item in prompts
            ],
            "validation_sha256": validation_hashes,
            "imgsz": manifest.get("imgsz"),
            "conf": manifest.get("conf"),
            "ultralytics": build.get("ultralytics"),
            "tensorrt": build.get("tensorrt"),
        }
        calculated_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if calculated_id != artifact_id:
            raise RuntimeError("YOLOE manifest identity does not match artifact ID")
        engine = manifest.get("engine", {})
        if manifest.get("imgsz") != int(engine.get("imgsz", -1)):
            raise RuntimeError("YOLOE engine image size does not match manifest")
        if manifest.get("class_order") != [item["id"] for item in prompts]:
            raise RuntimeError("YOLOE class ordering does not match prompt catalog")
        engine_name = engine.get("file")
        if not isinstance(engine_name, str) or os.path.basename(engine_name) != engine_name:
            raise RuntimeError("invalid YOLOE engine path in manifest")
        if verify_engine:
            verify_sha256(
                os.path.join(artifact_dir, engine_name),
                engine.get("sha256"),
                "YOLOE TensorRT engine",
            )
        if require_full_pipeline:
            validation = manifest.get("full_pipeline_validation")
            if (
                not isinstance(validation, dict)
                or validation.get("passed") is not True
                or validation.get("artifact_id") != artifact_id
                or validation.get("engine_sha256") != engine.get("sha256")
                or validation.get("prompt_ids") != [
                    item["id"] for item in prompts
                ]
            ):
                raise RuntimeError(
                    "YOLOE engine lacks passing full-pipeline validation; "
                    "run preprocess_prompt.sh with CAMERA_K configured"
                )
        return cls(prompts, manifest, artifact_dir)

    @staticmethod
    def valid_id(prompt_id):
        return isinstance(prompt_id, str) and bool(_PROMPT_ID.fullmatch(prompt_id))
