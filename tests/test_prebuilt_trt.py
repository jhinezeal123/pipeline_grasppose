import hashlib
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from scripts.setup.fetch_prebuilt_trt import (
    BUNDLES,
    _extract_archive,
    _write_current,
    install_vgn_bundle,
    read_dependencies,
    require_record,
    require_vgn_record,
    VGN_BUNDLE_NAME,
    RELEASE_URL,
)

ROOT = Path(__file__).resolve().parents[1]


class PrebuiltTensorRTTests(unittest.TestCase):
    def test_dependencies_pin_both_fp32_bundles(self):
        records = read_dependencies(ROOT / "dependencies")
        for name in BUNDLES:
            self.assertIn(name, records)
            require_record(records[name], name)
            self.assertEqual(records[name]["TENSORRT"], "8.5.2.2")
            self.assertEqual(records[name]["L4T"], "R35.6.4")

    def test_all_engine_bundles_use_one_release(self):
        records = read_dependencies(ROOT / "dependencies")
        for name in list(BUNDLES) + [VGN_BUNDLE_NAME]:
            self.assertTrue(records[name]["URL"].startswith(RELEASE_URL))

    def test_bundle_rejects_unexpected_archive_entries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_path = root / "bundle.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                contents = b"untrusted"
                member = tarfile.TarInfo("../escape")
                member.size = len(contents)
                archive.addfile(member, io.BytesIO(contents))
            with self.assertRaisesRegex(RuntimeError, "unexpected files"):
                _extract_archive(
                    archive_path, "a" * 64, root, {"manifest.json"})
            self.assertFalse((root.parent / "escape").exists())

    def test_current_pointer_ends_with_newline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _write_current(root, "b" * 64)
            self.assertEqual(
                (root / "CURRENT").read_bytes(), ("b" * 64 + "\n").encode())

    def test_vgn_bundle_installs_into_empty_checkout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            engine = b"test VGN TensorRT engine"
            checkpoint = b"test VGN checkpoint"
            engine_hash = hashlib.sha256(engine).hexdigest()
            checkpoint_hash = hashlib.sha256(checkpoint).hexdigest()
            identity = {
                "engine_sha256": engine_hash,
                "checkpoint_sha256": checkpoint_hash,
                "tensorrt": "8.5.2.2",
                "l4t": "R35.6.4",
                "gpu": "tegra194",
                "build_flag": "--fp16",
            }
            artifact_id = hashlib.sha256(
                json.dumps(identity, sort_keys=True).encode("utf-8")
            ).hexdigest()
            manifest = {
                "schema_version": 1,
                "artifact_id": artifact_id,
                "engine": {
                    "file": "vgn.engine",
                    "sha256": engine_hash,
                    "precision": "fp16-enabled",
                },
                "checkpoint": {
                    "file": "vgn_conv.pth",
                    "sha256": checkpoint_hash,
                },
                "build": {
                    "tensorrt": "8.5.2.2",
                    "l4t": "R35.6.4",
                    "gpu": "tegra194",
                    "trtexec_flag": "--fp16",
                },
            }
            archive_path = root / "vgn.tar.gz"
            with tarfile.open(archive_path, "w:gz") as archive:
                for filename, payload in (
                        ("vgn.engine", engine),
                        ("vgn_conv.pth", checkpoint),
                        ("manifest.json", json.dumps(manifest).encode("utf-8"))):
                    member = tarfile.TarInfo(filename)
                    member.size = len(payload)
                    archive.addfile(member, io.BytesIO(payload))
            record = {
                "KIND": "tar.gz",
                "URL": "https://github.com/jhinezeal123/pipeline_grasppose/releases/download/jetson-xavier-trt-bundle-v2/vgn.tar.gz",
                "SHA256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "DEST": "model",
                "ARTIFACT_ID": artifact_id,
                "ENGINE_SHA256": engine_hash,
                "CHECKPOINT_SHA256": checkpoint_hash,
                "TENSORRT": "8.5.2.2",
                "L4T": "R35.6.4",
                "GPU": "tegra194",
                "PRECISION": "fp16-enabled",
            }
            installed = install_vgn_bundle(
                record, install_root=root, archive_path=archive_path)
            self.assertEqual(installed.read_bytes(), engine)
            self.assertEqual(
                (root / "model" / "vgn_conv.pth").read_bytes(), checkpoint)
            runtime = json.loads(
                (root / "model" / "runtime" / "vgn.json").read_text())
            self.assertEqual(runtime["engine_sha256"], engine_hash)
            self.assertEqual(runtime["checkpoint_sha256"], checkpoint_hash)
            install_vgn_bundle(
                record, install_root=root, archive_path=root / "absent.tar.gz")
            self.assertEqual(installed.read_bytes(), engine)

    def test_dependencies_pin_vgn_bundle(self):
        records = read_dependencies(ROOT / "dependencies")
        record = records[VGN_BUNDLE_NAME]
        require_vgn_record(record)
        self.assertEqual(record["TENSORRT"], "8.5.2.2")
        self.assertEqual(record["L4T"], "R35.6.4")


if __name__ == "__main__":
    unittest.main()
