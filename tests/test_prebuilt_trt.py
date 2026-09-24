import io
import tarfile
import tempfile
import unittest
from pathlib import Path

from tools.fetch_prebuilt_trt import (
    BUNDLES,
    _extract_archive,
    _write_current,
    read_dependencies,
    require_record,
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


if __name__ == "__main__":
    unittest.main()
