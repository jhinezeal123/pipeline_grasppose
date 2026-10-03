"""Khóa activation/checksum/native provider, không tải model hoặc cài wheel."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from grasppose.modules.depth.da3_metric import Da3MetricDepth
from scripts.setup import prepare_da3, fetch_prebuilt_trt


class Da3PreparationTests(unittest.TestCase):
    def test_prepare_shell_wires_only_the_selected_depth_backend(self):
        for selected in (None, "lite-mono"):
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                (root / "scripts").mkdir()
                shutil.copy2(prepare_da3.ROOT / "scripts/prepare.sh", root / "scripts")
                python = root / ".venv/bin/python"
                python.parent.mkdir(parents=True)
                python.write_text(
                    "#!" + sys.executable + "\n"
                    "import json, os, sys\n"
                    "with open(os.environ['PREPARATION_LOG'], 'a') as out:\n"
                    "    out.write(json.dumps(sys.argv[1:]) + '\\n')\n"
                )
                python.chmod(0o755)
                tools_dir = root / "tools"
                tools_dir.mkdir()
                git = tools_dir / "git"
                git.write_text("#!/bin/sh\nexit 0\n")
                git.chmod(0o755)
                for file in (
                    "model/yoloe-26s-seg.pt",
                    "mobileclip2_b.ts",
                    "model/lite-mono/encoder.pth",
                    "model/lite-mono/depth.pth",
                ):
                    path = root / file
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"fixture")
                log = root / "commands.jsonl"
                env = {
                    **os.environ,
                    "PYTHON": str(python),
                    "PREPARATION_LOG": str(log),
                    "PATH": str(tools_dir) + os.pathsep + os.environ["PATH"],
                }
                env.pop("GRASP_DEPTH_BACKEND", None)
                if selected is not None:
                    env["GRASP_DEPTH_BACKEND"] = selected
                result = subprocess.run(
                    ["bash", str(root / "scripts/prepare.sh")],
                    cwd=str(root),
                    env=env,
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                commands = [json.loads(line) for line in log.read_text().splitlines()]
                expected = selected or "da3"
                self.assertIn(
                    [
                        "scripts/setup/fetch_prebuilt_trt.py",
                        "--depth-backend",
                        expected,
                    ],
                    commands,
                )
                self.assertEqual(
                    ["scripts/setup/prepare_da3.py"] in commands, expected == "da3"
                )
                self.assertIn(["env/check_env.py"], commands)

    def record(self, content):
        return {
            "NAME": "fixture",
            "URL": "https://example.com/model.onnx",
            "SHA256": hashlib.sha256(content).hexdigest(),
        }

    def test_dependency_records_match_the_existing_fp32_adapter(self):
        model, wheel = prepare_da3.da3_records()
        self.assertEqual(model["SHA256"], prepare_da3.DA3_WEIGHTS_SHA256)
        self.assertIn(model["REVISION"], model["URL"])
        self.assertTrue(model["URL"].endswith("/model.onnx"))
        self.assertEqual(
            wheel["FILENAME"], "onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl"
        )

    def test_wrong_graph_checksum_and_wrong_wheel_abi_are_rejected(self):
        for name, key, value in (
            (prepare_da3.MODEL_NAME, "SHA256", "0" * 64),
            (
                prepare_da3.ORT_NAME,
                "FILENAME",
                "onnxruntime_gpu-1.16.0-cp38-cp38m-linux_aarch64.whl",
            ),
        ):
            records = fetch_prebuilt_trt.read_dependencies(
                prepare_da3.ROOT / "dependencies"
            )
            records[name][key] = value
            with patch.object(prepare_da3, "read_dependencies", return_value=records):
                with self.assertRaises(ValueError):
                    prepare_da3.da3_records()

    def test_clean_download_is_verified_before_atomic_activation(self):
        content = b"valid model"
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model" / "model.onnx"

            def download(args, **kwargs):
                Path(args[-1]).write_bytes(content)
                self.assertFalse(target.exists())

            with patch.object(prepare_da3.subprocess, "run", side_effect=download):
                self.assertEqual(
                    prepare_da3.fetch_verified(self.record(content), target), target
                )
            self.assertEqual(target.read_bytes(), content)
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_bad_download_does_not_replace_existing_file(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model.onnx"
            target.write_bytes(b"previous file")

            def bad_download(args, **kwargs):
                Path(args[-1]).write_bytes(b"HTML error or incomplete download")

            with patch.object(prepare_da3.subprocess, "run", side_effect=bad_download):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    prepare_da3.fetch_verified(self.record(b"valid model"), target)
            self.assertEqual(target.read_bytes(), b"previous file")
            self.assertEqual(list(Path(temp).iterdir()), [target])

    def test_interrupted_download_does_not_activate_a_file(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model.onnx"
            with patch.object(
                prepare_da3.subprocess,
                "run",
                side_effect=subprocess.CalledProcessError(1, "curl"),
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    prepare_da3.fetch_verified(self.record(b"valid"), target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_valid_existing_artifact_does_not_download(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "model.onnx"
            target.write_bytes(b"valid")
            with patch.object(prepare_da3.subprocess, "run") as run:
                prepare_da3.fetch_verified(self.record(b"valid"), target)
            run.assert_not_called()

    def test_wheel_install_never_resolves_or_replaces_host_dependencies(self):
        _, wheel = prepare_da3.da3_records()
        with patch.object(prepare_da3.subprocess, "run") as run:
            prepare_da3.install_ort(Path("/tmp") / wheel["FILENAME"], wheel)
        install, probe = run.call_args_list
        self.assertEqual(
            install.args[0][:6],
            [sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall"],
        )
        self.assertEqual(probe.args[0][:2], [sys.executable, "-c"])
        self.assertIn("check_ort_runtime", probe.args[0][2])

    def test_runtime_requires_pinned_version_and_requested_provider(self):
        _, wheel = prepare_da3.da3_records()
        for version, providers in (
            ("1.17.0", ["CUDAExecutionProvider"]),
            ("1.16.0", ["CPUExecutionProvider"]),
        ):
            module = SimpleNamespace(
                __version__=version, get_available_providers=lambda: providers
            )
            with self.assertRaises(RuntimeError):
                prepare_da3.check_ort_runtime(module, wheel, "CUDAExecutionProvider")
        good = SimpleNamespace(
            __version__="1.16.0",
            get_available_providers=lambda: ["CUDAExecutionProvider"],
        )
        prepare_da3.check_ort_runtime(good, wheel, "CUDAExecutionProvider")

    def test_native_session_cannot_silently_fall_back_to_cpu(self):
        session = SimpleNamespace(get_providers=lambda: ["CPUExecutionProvider"])
        module = SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            SessionOptions=SimpleNamespace,
            GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=1),
            InferenceSession=lambda *args, **kwargs: session,
        )
        depth = Da3MetricDepth(provider="CUDAExecutionProvider")
        with patch.dict(sys.modules, {"onnxruntime": module}), patch(
            "grasppose.modules.depth.da3_metric.os.path.isfile", return_value=True
        ), patch("grasppose.modules.depth.da3_metric.verify_sha256"):
            with self.assertRaisesRegex(RuntimeError, "did not activate"):
                depth.load()
        self.assertIsNone(depth._session)

    def test_host_guard_prevents_installing_a_jetson_wheel_on_desktop(self):
        with patch.object(prepare_da3.platform, "machine", return_value="x86_64"):
            with self.assertRaisesRegex(RuntimeError, "Jetson"):
                prepare_da3.check_host()

    def test_da3_bundle_path_does_not_require_lite_mono_artifacts(self):
        for selected, expected in (
            ("da3", ["yoloe-26s-cube-trt-fp32"]),
            ("lite-mono", ["yoloe-26s-cube-trt-fp32", "lite-mono-trt-fp32"]),
        ):
            with patch.object(fetch_prebuilt_trt, "check_host"), patch.object(
                fetch_prebuilt_trt, "install_bundle"
            ) as install, patch.object(fetch_prebuilt_trt, "install_vgn_bundle") as vgn:
                self.assertEqual(
                    fetch_prebuilt_trt.main(["--depth-backend", selected]), 0
                )
            self.assertEqual(
                [call.args[0] for call in install.call_args_list], expected
            )
            vgn.assert_called_once()


if __name__ == "__main__":
    unittest.main()
