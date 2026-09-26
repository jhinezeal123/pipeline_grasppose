import builtins
import hashlib
import json
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch

from grasppose.modules.vision.prompt_catalog import PromptCatalog
import grasppose.modules.vision.prompt_catalog as prompt_catalog
from scripts.build.preprocess_yoloe import read_prompt_file


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PromptConfigurationTests(unittest.TestCase):
    def test_yoloe_api_imports_ultralytics_before_torch(self):
        from scripts.build.preprocess_yoloe import load_yoloe_api

        import_order = []
        original_import = builtins.__import__
        fake_ultralytics = SimpleNamespace(YOLO=object(), YOLOE=object())
        fake_torch = SimpleNamespace(cuda=object())

        def tracked_import(name, globals=None, locals=None,
                           fromlist=(), level=0):
            if name in ("ultralytics", "torch"):
                import_order.append(name)
                return {
                    "ultralytics": fake_ultralytics,
                    "torch": fake_torch,
                }[name]
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=tracked_import):
            loaded = load_yoloe_api()
        self.assertEqual(import_order, ["ultralytics", "torch"])
        self.assertIs(loaded[2], fake_torch)

    def test_multiword_text_uses_prompt_id_as_engine_class_name(self):
        from scripts.build.preprocess_yoloe import configure_prompt_classes

        class FakeYOLOE:
            def get_text_pe(self, texts):
                self.texts = list(texts)
                return "embedding"

            def set_classes(self, classes, embeddings):
                self.classes = list(classes)
                self.embeddings = embeddings

        model = FakeYOLOE()
        configure_prompt_classes(model, [{"id": "blue_cube", "text": "blue cube"}])
        self.assertEqual(model.texts, ["blue cube"])
        self.assertEqual(model.classes, ["blue_cube"])
        self.assertEqual(model.embeddings, "embedding")


class PromptCatalogTests(unittest.TestCase):
    def make_artifact(self, root):
        checkpoint = root / "yoloe.pt"
        text_encoder = root / "mobileclip.ts"
        checkpoint.write_bytes(b"checkpoint")
        text_encoder.write_bytes(b"text encoder")
        prompts = [
            {"id": "blue_cube", "text": "blue cube", "class_index": 0}
        ]
        validation_hashes = {"blue_cube": "b" * 64}
        build = {"ultralytics": "8.4.140", "tensorrt": "8.5.2.2"}
        identity = {
            "model_sha256": digest(checkpoint),
            "text_encoder_sha256": digest(text_encoder),
            "prompts": [{"id": "blue_cube", "text": "blue cube"}],
            "validation_sha256": validation_hashes,
            "imgsz": 640,
            "conf": 0.20,
            "ultralytics": build["ultralytics"],
            "tensorrt": build["tensorrt"],
        }
        artifact_id = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode("utf-8")
        ).hexdigest()
        artifact_dir = root / artifact_id
        artifact_dir.mkdir()
        profile = artifact_dir / "prompts.npz"
        engine = artifact_dir / "yoloe_fp32.engine"
        profile.write_bytes(b"prompt embeddings")
        engine.write_bytes(b"tensorrt engine")
        manifest = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "source": {
                "checkpoint_sha256": digest(checkpoint),
                "text_encoder_sha256": digest(text_encoder),
                "profile": {
                    "file": profile.name,
                    "sha256": digest(profile),
                },
            },
            "prompts": prompts,
            "validation_images": validation_hashes,
            "class_order": ["blue_cube"],
            "imgsz": 640,
            "conf": 0.20,
            "build": build,
            "precision_policy": "fp32_only",
            "engine": {
                "precision": "fp32",
                "file": engine.name,
                "imgsz": 640,
                "sha256": digest(engine),
            },
        }
        (artifact_dir / "manifest.json").write_text(json.dumps(manifest))
        current = root / "CURRENT"
        current.write_text(artifact_id + "\n")
        return artifact_id, artifact_dir, checkpoint, text_encoder, current

    def patched_catalog_paths(self, root, checkpoint, current):
        return (
            patch.object(prompt_catalog, "YOLOE_ARTIFACT_ROOT", str(root)),
            patch.object(prompt_catalog, "YOLOE_MODEL", str(checkpoint)),
            patch.object(prompt_catalog, "YOLOE_CURRENT_FILE", str(current)),
        )

    def test_prompt_file_preserves_order_and_validates_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "prompts.json"
            path.write_text(json.dumps({
                "prompts": [
                    {"id": "blue_cube", "text": " blue cube "},
                    {"id": "red_mug", "text": "red mug"},
                ]
            }))
            prompts = read_prompt_file(path)
            self.assertEqual(
                [(item["id"], item["text"], item["class_index"])
                 for item in prompts],
                [("blue_cube", "blue cube", 0), ("red_mug", "red mug", 1)],
            )
            path.write_text(json.dumps([
                {"id": "same", "text": "one"},
                {"id": "same", "text": "two"},
            ]))
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_prompt_file(path)
            path.write_text(json.dumps([{"id": "", "text": "empty"}]))
            with self.assertRaisesRegex(ValueError, "invalid"):
                read_prompt_file(path)
            path.write_text(json.dumps([{"id": "cube", "text": " "}]))
            with self.assertRaisesRegex(ValueError, "empty text"):
                read_prompt_file(path)

    def test_load_validates_profile_and_engine_checksums(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, checkpoint, text_encoder, current = self.make_artifact(root)
            text_encoder.unlink()
            contexts = self.patched_catalog_paths(
                root, checkpoint, current)
            with contexts[0], contexts[1], contexts[2]:
                catalog = PromptCatalog.load(verify_engine=True)
            self.assertEqual(catalog.require("blue_cube")["class_index"], 0)
            with self.assertRaisesRegex(ValueError, "unknown prompt ID"):
                catalog.require("cube")

    def test_worker_requires_parity_for_fp32_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_id, artifact_dir, checkpoint, _, current = (
                self.make_artifact(root)
            )
            contexts = self.patched_catalog_paths(root, checkpoint, current)
            with contexts[0], contexts[1], contexts[2]:
                with self.assertRaisesRegex(RuntimeError, "full-pipeline validation"):
                    PromptCatalog.load(
                        verify_engine=True, require_full_pipeline=True
                    )
                manifest_path = artifact_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["full_pipeline_validation"] = {
                    "passed": True,
                    "artifact_id": artifact_id,
                    "engine_sha256": manifest["engine"]["sha256"],
                    "prompt_ids": ["blue_cube"],
                    "camera_k": [957.0, 948.0, 636.0, 352.0],
                }
                manifest_path.write_text(json.dumps(manifest))
                PromptCatalog.load(
                    verify_engine=True, require_full_pipeline=True
                )
                manifest["full_pipeline_validation"]["engine_sha256"] = "0" * 64
                manifest_path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "full-pipeline validation"):
                    PromptCatalog.load(
                        verify_engine=True, require_full_pipeline=True
                    )

    def test_load_rejects_fp16_engine(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, artifact_dir, checkpoint, _, current = self.make_artifact(root)
            manifest_path = artifact_dir / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["engine"]["precision"] = "fp16"
            manifest_path.write_text(json.dumps(manifest))
            contexts = self.patched_catalog_paths(root, checkpoint, current)
            with contexts[0], contexts[1], contexts[2]:
                with self.assertRaisesRegex(RuntimeError, "requires the FP32 engine"):
                    PromptCatalog.load(verify_engine=True)

    def test_load_rejects_changed_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, checkpoint, text_encoder, current = self.make_artifact(root)
            checkpoint.write_bytes(b"changed checkpoint")
            contexts = self.patched_catalog_paths(
                root, checkpoint, current)
            with contexts[0], contexts[1], contexts[2]:
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    PromptCatalog.load(verify_engine=True)

    def test_load_rejects_current_pointer_manifest_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_id, artifact_dir, checkpoint, text_encoder, current = (
                self.make_artifact(root)
            )
            mismatch_dir = root / ("b" * 64)
            mismatch_dir.mkdir()
            (mismatch_dir / "manifest.json").write_bytes(
                (artifact_dir / "manifest.json").read_bytes()
            )
            current.write_text("b" * 64 + "\n")
            contexts = self.patched_catalog_paths(
                root, checkpoint, current)
            with contexts[0], contexts[1], contexts[2]:
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    PromptCatalog.load(verify_engine=True)


if __name__ == "__main__":
    unittest.main()
