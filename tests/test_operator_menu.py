"""Luồng người dùng qua terminal; không model thật, subprocess hoặc socket thật."""

import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from apps.operator.config import ProfileStore
from apps.operator.console import main
from apps.operator.tasks import Tasks
from apps.operator.terminal import Back, CommandRunner, Terminal

ROOT = Path(__file__).resolve().parents[1]
READY = {
    "state": "ready",
    "prompts": [{"id": "cube", "text": "Khối"}, {"id": "bag", "text": "Túi"}],
}


class OperatorMenuTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        python = self.root / ".venv/bin/python"
        python.parent.mkdir(parents=True)
        python.write_text("fake interpreter, never execute")
        python.chmod(0o755)
        self.store = ProfileStore(self.root)
        self.profile = replace(
            self.store.load(), camera_k="100 101 40 30", camera_k_size="80,60"
        )
        self.store.save(self.profile)
        self.image = self.root / "image with spaces.jpg"
        self.image.write_bytes(b"fixture")

    def tasks(self, answers="", dry=False):
        self.output = io.StringIO()
        view = Terminal(io.StringIO(answers), self.output)
        return Tasks(self.root, self.store, view, CommandRunner(self.root, view, dry))

    def response(self, render=False):
        return subprocess.CompletedProcess(
            [],
            0,
            "RUN_ID: "
            + "a" * 32
            + "\nGRASP_POSES: "
            + json.dumps(
                [
                    {"score": 0.9, "width_m": 0.03, "translation_m": [0.1, 0.2, 0.3]},
                ]
            )
            + "\n",
            "",
        )

    def test_root_catalog_works_outside_repo_without_site_packages(self):
        result = subprocess.run(
            [str(ROOT / "start"), "--list"],
            cwd="/tmp",
            env={**os.environ, "GRASP_CONSOLE_PYTHON": sys.executable},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Tìm pose gắp", result.stdout)
        code = "import sys; from apps.operator import console; assert not any(n in sys.modules for n in ('torch','numpy','gradio','onnxruntime')); console.main(['--list'])"
        cold = subprocess.run(
            [sys.executable, "-S", "-c", code],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(cold.returncode, 0, cold.stderr)

    def test_eof_and_bad_choice_do_not_start_any_task(self):
        out = io.StringIO()
        with patch("apps.operator.terminal.CommandRunner._execute") as run, patch(
            "apps.operator.tasks.request_worker"
        ) as rpc:
            self.assertEqual(
                main(
                    ["--config", str(self.store.path)],
                    Terminal(io.StringIO("bad\n0\n"), out),
                ),
                0,
            )
            self.assertEqual(
                main(
                    ["--config", str(self.store.path)], Terminal(io.StringIO(""), out)
                ),
                0,
            )
        run.assert_not_called()
        rpc.assert_not_called()
        self.assertIn("Chọn một số", out.getvalue())

    def test_profile_is_private_and_survives_reopening(self):
        selected = replace(
            self.profile, backend="lite-mono", confidence=0.05, ui_port=8090
        )
        self.store.save(selected)
        self.assertEqual(self.store.load(), selected)
        self.assertEqual(self.store.path.stat().st_mode & 0o777, 0o600)

    def test_atomic_save_failure_preserves_old_profile_and_cleans_temp(self):
        before = self.store.path.read_bytes()
        with patch(
            "apps.operator.config.os.replace", side_effect=OSError("disk error")
        ), self.assertRaises(OSError):
            self.store.save(replace(self.profile, confidence=0.05))
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertEqual(list(self.store.path.parent.iterdir()), [self.store.path])

    def test_invalid_camera_and_confidence_are_rejected_before_save(self):
        for changes in (
            {"camera_k": "nan 10 20 30"},
            {"camera_k": "0 10 20 30"},
            {"confidence": float("nan")},
            {"confidence": True},
            {"ui_port": True},
            {"socket": ""},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.store.save(replace(self.profile, **changes))

    def test_bad_profile_is_reported_without_overwriting_it(self):
        self.store.path.write_text('{"version":99}')
        out = io.StringIO()
        self.assertEqual(
            main(["--config", str(self.store.path)], Terminal(io.StringIO(""), out)), 1
        )
        self.assertIn("Không đọc được cấu hình", out.getvalue())
        self.assertEqual(self.store.path.read_text(), '{"version":99}')

    def test_guided_inference_selects_live_prompt_and_forwards_saved_camera(self):
        task = self.tasks(str(self.image) + "\n2\nn\n")
        with patch.object(task, "probe", return_value=READY), patch.object(
            task, "worker_environment", return_value=None
        ), patch(
            "apps.operator.terminal.CommandRunner._execute", return_value=self.response()
        ) as run:
            task.infer()
        (argv,) = run.call_args.args
        self.assertIn(str(self.image), argv)
        self.assertEqual(argv[-2:], ["--prompt-id", "bag"])
        self.assertEqual(run.call_args.kwargs["env"]["CAMERA_K"], "100 101 40 30")
        self.assertEqual(run.call_args.kwargs["env"]["CAMERA_K_SIZE"], "80 60")
        self.assertEqual(self.store.load().last_run, "a" * 32)
        self.assertIn("độ mở 30.0 mm", self.output.getvalue())

    def test_render_wait_is_outside_inference_and_reports_image_paths(self):
        task = self.tasks(str(self.image) + "\n1\ny\n")
        job = subprocess.CompletedProcess(
            [],
            0,
            json.dumps(
                {
                    "state": "done",
                    "files": ["box.png", "mask.png", "depth.png", "grasp.png"],
                }
            ),
            "",
        )
        with patch.object(task, "probe", return_value=READY), patch.object(
            task, "worker_environment", return_value=None
        ), patch(
            "apps.operator.terminal.CommandRunner._execute", side_effect=[self.response(), job]
        ) as run:
            task.infer()
        self.assertIn("--render", run.call_args_list[0].args[0])
        self.assertEqual(run.call_args_list[1].args[0][-2:], ["wait", "a" * 32])
        self.assertIn("grasp.png", self.output.getvalue())

    def test_dry_run_never_forks_probes_or_writes_profile(self):
        before = self.store.path.read_bytes()
        task = self.tasks(str(self.image) + "\ncube\nn\n", dry=True)
        with patch("apps.operator.terminal.CommandRunner._execute") as run, patch(
            "apps.operator.tasks.request_worker"
        ) as rpc:
            task.infer()
        run.assert_not_called()
        rpc.assert_not_called()
        self.assertEqual(self.store.path.read_bytes(), before)
        self.assertIn("Chỉ xem lệnh", self.output.getvalue())

    def test_missing_worker_can_be_declined_without_starting_process(self):
        task = self.tasks("n\n")
        with patch.object(task, "probe", return_value=None), patch(
            "apps.operator.terminal.CommandRunner._execute"
        ) as run, self.assertRaises(Back):
            task.ensure_worker()
        run.assert_not_called()

    def test_missing_worker_starts_with_the_selected_environment(self):
        task = self.tasks("y\n")
        with patch.object(task, "probe", side_effect=[None, READY]), patch.object(
            task, "worker_environment", return_value=None
        ), patch(
            "apps.operator.terminal.CommandRunner._execute",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            self.assertEqual(task.ensure_worker(), READY)
        self.assertEqual(run.call_args.args[0][-1], "start")
        self.assertEqual(
            run.call_args.kwargs["env"]["GRASP_WORKER_SOCKET"], self.profile.socket
        )

    def test_existing_worker_with_different_environment_requires_restart(self):
        task = self.tasks("y\n")
        with patch.object(task, "probe", return_value=READY), patch.object(
            task,
            "worker_environment",
            return_value={"YOLOE_CONF": "0.05", "GRASP_DEPTH_BACKEND": "da3"},
        ), patch(
            "apps.operator.terminal.CommandRunner._execute",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            task.ensure_worker()
        self.assertEqual(run.call_args.args[0][-1], "restart")
        self.assertEqual(run.call_args.kwargs["env"]["YOLOE_CONF"], "0.2")

    def test_refusing_restart_does_not_infer_under_mismatched_settings(self):
        task = self.tasks("n\n")
        with patch.object(task, "probe", return_value=READY), patch.object(
            task, "worker_environment", return_value={"YOLOE_CONF": "0.05"}
        ), patch("apps.operator.terminal.CommandRunner._execute") as run, self.assertRaises(
            Back
        ):
            task.ensure_worker()
        run.assert_not_called()

    def test_worker_command_environment_does_not_mutate_parent_defaults(self):
        task = self.tasks("2\n")
        task.profile = replace(task.profile, confidence=0.05)
        before = dict(os.environ)
        with patch(
            "apps.operator.terminal.CommandRunner._execute",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            task.worker()
        self.assertEqual(dict(os.environ), before)
        self.assertEqual(run.call_args.kwargs["env"]["YOLOE_CONF"], "0.05")

    def test_output_failure_and_child_exit_are_not_reported_as_success(self):
        task = self.tasks("a" * 32 + "\n")
        with patch(
            "apps.operator.terminal.CommandRunner._execute",
            return_value=subprocess.CompletedProcess([], 2, "", "snapshot expired"),
        ), self.assertRaisesRegex(RuntimeError, "mã 2"):
            task.output()
        self.assertIn("snapshot expired", self.output.getvalue())

    def test_benchmark_passes_calibration_resolution_and_minimum_runs(self):
        task = self.tasks(str(self.image) + "\n1\n20\n")
        with patch.object(task, "probe", return_value=READY), patch.object(
            task, "worker_environment", return_value=None
        ), patch(
            "apps.operator.terminal.CommandRunner._execute",
            return_value=subprocess.CompletedProcess([], 0),
        ) as run:
            task.benchmark()
        self.assertEqual(run.call_args.args[0][-3:], ["--camera-k-size", "80", "60"])
        self.assertIn("20", run.call_args.args[0])

    def test_direct_action_propagates_runtime_failure(self):
        out = io.StringIO()
        with patch.object(Tasks, "status", side_effect=RuntimeError("worker error")):
            self.assertEqual(
                main(
                    ["status", "--config", str(self.store.path)],
                    Terminal(io.StringIO(""), out),
                ),
                1,
            )
        self.assertIn("worker error", out.getvalue())

    def test_root_back_exits_without_running_a_task(self):
        out = io.StringIO()
        with patch("apps.operator.terminal.CommandRunner._execute") as run, patch(
            "apps.operator.tasks.request_worker"
        ) as rpc:
            self.assertEqual(main(
                ["--config", str(self.store.path)], Terminal(io.StringIO(":q\n"), out)
            ), 0)
        run.assert_not_called()
        rpc.assert_not_called()
        self.assertIn("Đã thoát menu", out.getvalue())

    def test_status_dry_run_never_contacts_worker(self):
        task = self.tasks(dry=True)
        with patch("apps.operator.tasks.request_worker") as rpc:
            task.status()
            self.assertIsNone(task.probe())
        rpc.assert_not_called()
        self.assertIn("dry-run", self.output.getvalue())

    def test_readiness_labels_keep_their_priority(self):
        task = self.tasks()
        expected = {
            "status": "có thể mở", "infer": "cần worker", "ui": "cần worker",
            "prompts": "cần worker", "worker": "có thể mở", "configure": "có thể mở",
            "prepare": "cần Jetson Xavier", "output": "chưa có lượt chạy",
            "benchmark": "cần worker",
        }
        self.assertEqual({key: task.availability(key) for key in expected}, expected)
        task.profile = replace(task.profile, camera_k="", camera_k_size="")
        for key in ("infer", "ui", "benchmark"):
            expected[key] = "cần camera K"
        self.assertEqual({key: task.availability(key) for key in expected}, expected)
        (self.root / ".venv/bin/python").unlink()
        for key in ("infer", "ui", "prompts", "worker", "output", "benchmark"):
            expected[key] = "cần môi trường"
        self.assertEqual({key: task.availability(key) for key in expected}, expected)

    def test_child_interrupt_keeps_exit_130(self):
        out = io.StringIO()
        with patch.object(Tasks, "status", side_effect=KeyboardInterrupt):
            self.assertEqual(
                main(
                    ["status", "--config", str(self.store.path)],
                    Terminal(io.StringIO(""), out),
                ),
                130,
            )
        task = self.tasks()
        with patch(
            "apps.operator.terminal.CommandRunner._execute",
            return_value=subprocess.CompletedProcess([], 130, "", ""),
        ), self.assertRaises(KeyboardInterrupt):
            task.runner.run(["fake"])


if __name__ == "__main__":
    unittest.main()
