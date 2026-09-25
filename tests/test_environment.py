import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "setup_env", ROOT / "env/setup_env.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


class EnvironmentTests(unittest.TestCase):
    def test_requires_jetpack_stack(self):
        with patch.object(
                setup.importlib, "import_module",
                side_effect=ImportError("missing")):
            with self.assertRaisesRegex(
                    RuntimeError, "Missing/broken host torch"):
                setup.protected_versions()

    def test_runtime_versions_win_over_stale_duplicate_metadata(self):
        runtime_versions = {
            "torch": "2.1.0a0+41361538.nv23.06",
            "torchvision": "0.16.1",
            "numpy": "1.23.5",
            "scipy": "1.10.1",
        }
        distributions = [
            Mock(metadata={"Name": "numpy"}, version="1.17.4"),
            Mock(metadata={"Name": "scipy"}, version="1.3.3"),
            Mock(metadata={"Name": "torch"}, version="1.13.0"),
            Mock(metadata={"Name": "gradio"}, version="1.0"),
            Mock(metadata={"Name": "ultralytics"}, version="1.0"),
        ]

        def import_module(name):
            module = Mock()
            module.__version__ = runtime_versions[name]
            return module

        with patch.object(
                setup.importlib, "import_module",
                side_effect=import_module), \
                patch.object(
                    setup.metadata, "distributions",
                    return_value=distributions):
            versions = setup.protected_versions()

        self.assertEqual(versions["numpy"], "1.23.5")
        self.assertEqual(versions["scipy"], "1.10.1")
        self.assertEqual(
            versions["torch"],
            "2.1.0a0+41361538.nv23.06",
        )
        self.assertEqual(versions["torchvision"], "0.16.1")
        self.assertNotIn("gradio", versions)
        self.assertNotIn("ultralytics", versions)

    def test_install_uses_overlay_python_and_constraints(self):
        versions = {
            "torch": "2.1.0a0+nv23.06",
            "torchvision": "0.16.1",
            "numpy": "1.23.5",
            "scipy": "1.10.1",
        }
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(
                    setup, "ENV", Path(temp) / ".venv"), \
                patch.object(
                    setup, "protected_versions",
                    return_value=versions), \
                patch.object(
                    setup, "_create_overlay") as create, \
                patch.object(
                    setup.subprocess, "run") as run, \
                patch.object(
                    setup.sys, "argv",
                    ["setup_env.py", "--install"]):

            def make_env():
                setup.ENV.mkdir(parents=True)
                (setup.ENV / "bin").mkdir()

            create.side_effect = make_env
            expected_python = setup.ENV / "bin" / "python"
            expected_constraints = setup.ENV / "host-constraints.txt"
            setup.main()
            pip_bootstrap = run.call_args_list[-2].args[0]
            install_cmd = run.call_args_list[-1].args[0]

            self.assertEqual(
                pip_bootstrap,
                [
                    str(expected_python), "-m", "pip",
                    "install", "--upgrade", "pip==25.0.1",
                ],
            )
            self.assertEqual(
                install_cmd[:3],
                [str(expected_python), "-m", "pip"],
            )
            self.assertIn(str(expected_constraints), install_cmd)
            self.assertIn(
                str(setup.ROOT / "requirements.txt"), install_cmd)

    def test_python38_requirements_match_xavier_host(self):
        lines = [
            line.strip()
            for line in (ROOT / "requirements.txt").read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        requirements = "\n".join(lines)
        self.assertIn(
            'numpy==1.23.5; python_version < "3.9"',
            requirements,
        )
        self.assertIn(
            'scipy==1.10.1; python_version < "3.9"',
            requirements,
        )
        self.assertIn(
            'onnx==1.14.1; python_version < "3.9"',
            requirements,
        )
        self.assertIn(
            'opencv-python==4.8.1.78; python_version < "3.9"',
            requirements,
        )
        self.assertFalse(any(
            line.startswith("opencv-python-headless")
            for line in lines
        ))
        self.assertFalse(any(
            line.startswith("pip")
            for line in lines
        ))

    def test_check_env_adds_repo_root_to_sys_path(self):
        check_path = ROOT / "env" / "check_env.py"
        source = check_path.read_text()
        self.assertIn(
            "sys.path.insert(0, str(ROOT))",
            source,
        )

    def test_infer_defaults_to_repo_local_output(self):
        source = (ROOT / "infer.sh").read_text()
        self.assertIn(
            'OUT="${OUTPUT_DIR:-$ROOT/output}"',
            source,
        )
        self.assertNotIn(
            "\nOUT=/output\n",
            source,
        )

    def test_prepare_does_not_run_global_pip_check(self):
        source = (ROOT / "prepare.sh").read_text()
        self.assertNotIn(
            '"$PYTHON" -m pip check',
            source,
        )

    def test_prepare_persists_runtime_artifact_paths(self):
        source = (ROOT / "prepare.sh").read_text()
        self.assertIn(
            'RUNTIME_CONFIG="$ROOT/.venv/runtime.json"',
            source,
        )
        for name in (
            "YOLOE_MODEL",
            "LITEMONO_ONNX",
            "LITEMONO_ENGINE",
            "LITEMONO_TRT_LIBRARY",
            "VGN_ENGINE",
        ):
            self.assertIn('"%s"' % name, source)
        self.assertIn("tmp.replace(dst)", source)

    def test_runtime_config_prefers_environment_over_persisted_paths(self):
        from grasppose import config as runtime_config

        with patch.object(
                runtime_config,
                "_RUNTIME_DEFAULTS",
                {"VGN_ENGINE": "/persisted/vgn.engine"}):
            with patch.dict(runtime_config.os.environ, {}, clear=True):
                self.assertEqual(
                    runtime_config._runtime_path(
                        "VGN_ENGINE", "/default/vgn.engine"),
                    "/persisted/vgn.engine",
                )
            with patch.dict(
                    runtime_config.os.environ,
                    {"VGN_ENGINE": "/explicit/vgn.engine"},
                    clear=True):
                self.assertEqual(
                    runtime_config._runtime_path(
                        "VGN_ENGINE", "/default/vgn.engine"),
                    "/explicit/vgn.engine",
                )

    def test_runtime_config_loads_persisted_json(self):
        from grasppose import config as runtime_config

        with tempfile.TemporaryDirectory() as temp:
            runtime_path = Path(temp) / "runtime.json"
            runtime_path.write_text(
                '{"VGN_ENGINE": "/persisted/vgn.engine"}'
            )
            with patch.object(
                    runtime_config, "RUNTIME_CONFIG", str(runtime_path)):
                self.assertEqual(
                    runtime_config._load_runtime_defaults(),
                    {"VGN_ENGINE": "/persisted/vgn.engine"},
                )

    def test_check_env_uses_runtime_config_artifact_paths(self):
        source = (ROOT / "env" / "check_env.py").read_text()
        self.assertIn("from grasppose.config import (", source)
        for line in (
            "YOLOE_ENGINE_PATH = _artifact_path(YOLOE_MODEL)",
            "LITEMONO_ONNX_PATH = _artifact_path(LITEMONO_ONNX)",
            "LITEMONO_ENGINE_PATH = _artifact_path(LITEMONO_ENGINE)",
            "LITEMONO_TRT_LIBRARY_PATH = _artifact_path(LITEMONO_TRT_LIBRARY)",
            "VGN_ENGINE_PATH = _artifact_path(VGN_ENGINE)",
            "(VGN_ENGINE, VGN_ENGINE_PATH)",
            "VgnTensorRT(str(VGN_ENGINE_PATH))",
        ):
            self.assertIn(line, source)
        self.assertNotIn(
            'VgnTensorRT(str(ROOT / "model/vgn.engine"))',
            source,
        )

    def test_check_env_compares_imported_runtime_versions(self):
        source = (ROOT / "env" / "check_env.py").read_text()
        self.assertIn(
            'protected_modules = {',
            source,
        )
        self.assertIn(
            'getattr(module, "__version__", "")',
            source,
        )
        self.assertIn(
            'actual = str(getattr(module, "__version__", "")).strip()',
            source,
        )

    def test_check_env_uses_fixed_class_yoloe_tensorrt_smoke(self):
        source = (ROOT / "env" / "check_env.py").read_text()
        self.assertIn(
            '"Torch was imported before YOLOE TensorRT load during service construction"',
            source,
        )
        self.assertIn(
            "for target in YOLOE_CLASSES:",
            source,
        )
        self.assertIn(
            'vision.predict(image, "not baked")',
            source,
        )
        self.assertIn(
            '"YOLOE TensorRT smoke failed: %s"',
            source,
        )

    def test_yoloe_export_bakes_exact_three_targets(self):
        source = (ROOT / "tools" / "export_yoloe_trt.py").read_text()
        for target in (
            '"blue cube"',
            '"yellow ball"',
            '"blue cylinder"',
        ):
            self.assertIn(target, source)
        self.assertIn('format="engine"', source)
        self.assertIn("quantize=16", source)
        self.assertIn("simplify=False", source)
        self.assertIn("model.set_classes(list(FIXED_CLASSES))", source)

    def test_prepare_installs_pinned_yoloe_release_without_export(self):
        prepare = (ROOT / "prepare.sh").read_text()
        dependencies = (ROOT / "dependencies").read_text()
        fetch = (ROOT / "tools" / "fetch_prebuilt_trt.py").read_text()
        self.assertIn('"$PYTHON" tools/fetch_prebuilt_trt.py', prepare)
        self.assertNotIn("tools/export_yoloe_trt.py", prepare)
        self.assertIn("jetson-xavier-pr7-fp16-bundle-v1", dependencies)
        self.assertIn("yoloe-26s-threeclass-trt-fp16", dependencies)
        self.assertIn("yoloe-26s-seg.engine", fetch)
        for target in ("blue cube", "yellow ball", "blue cylinder"):
            self.assertIn(target, fetch)

    def test_prepare_installs_pinned_litemono_and_vgn_release(self):
        prepare = (ROOT / "prepare.sh").read_text()
        dependencies = (ROOT / "dependencies").read_text()
        fetch = (ROOT / "tools" / "fetch_prebuilt_trt.py").read_text()
        self.assertIn("lite-mono-tiny-trt-fp16", dependencies)
        self.assertIn("vgn-trt-xavier", dependencies)
        self.assertIn("lite-mono-tiny_192x640_op11.onnx", fetch)
        self.assertIn("c4475f6389de466384149d0c09b785", dependencies)
        self.assertIn("82eefa90408d337a6b831204303efa", dependencies)
        self.assertIn("native/litemono_trt", prepare)
        self.assertNotIn("--saveEngine", prepare)


if __name__ == "__main__":
    unittest.main()
