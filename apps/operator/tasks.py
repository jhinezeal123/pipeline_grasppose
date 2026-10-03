"""Các tác vụ perception gọi CLI/RPC hiện có; không nhân bản inference."""

import json
import os
import re
from dataclasses import replace
from pathlib import Path

from grasppose.infrastructure.worker.rpc import WorkerError, request_worker

from .terminal import Back


class Tasks:
    def __init__(self, root, store, view, runner):
        self.root, self.store, self.view, self.runner = root, store, view, runner
        self.profile = store.load()

    def availability(self, action):
        if action in ("status", "configure"):
            return "có thể mở"
        if action == "prepare":
            return "cần Jetson Xavier"
        if not os.access(str(self.root / ".venv/bin/python"), os.X_OK):
            return "cần môi trường"
        if action == "output":
            return "có RUN_ID" if self.profile.last_run else "chưa có lượt chạy"
        if action in ("infer", "ui", "benchmark") and not self.profile.camera_k:
            return "cần camera K"
        if (
            action in ("infer", "ui", "prompts", "benchmark")
            and not Path(self.profile.socket).exists()
        ):
            return "cần worker"
        return "có thể mở"

    def save(self, profile):
        profile.validate()
        if not self.runner.dry_run:
            self.store.save(profile)
        self.profile = profile
        self.view.say(
            "Cấu hình tạm cho dry-run; chưa ghi file."
            if self.runner.dry_run
            else "Đã lưu cấu hình; lần sau dùng lại."
        )

    def python(self):
        path = self.root / ".venv" / "bin" / "python"
        if not self.runner.dry_run and not os.access(str(path), os.X_OK):
            raise RuntimeError(
                "Thiếu Python của pipeline. Chọn mục Chuẩn bị môi trường và model."
            )
        return path

    def script(self, name, *args):
        self.python()
        return self.runner.run(
            ["bash", self.root / "scripts" / name, *args], self.profile.environment()
        )

    def probe(self):
        if self.runner.dry_run:
            return None
        try:
            return request_worker(
                {"op": "status"}, timeout=2, socket_path=self.profile.socket
            )
        except WorkerError:
            return None

    def worker_environment(self, status):
        """Chỉ đọc hai biến đã chọn, không xuất environment chứa credential."""
        try:
            pid = int(status["pid"])
            if pid <= 0:
                return None
            raw = Path(f"/proc/{pid}/environ").read_bytes()
            values = dict(
                item.split(b"=", 1) for item in raw.split(b"\0") if b"=" in item
            )
            return {
                name: values[name.encode()].decode()
                for name in ("GRASP_DEPTH_BACKEND", "YOLOE_CONF")
                if name.encode() in values
            }
        except (OSError, ValueError, KeyError, UnicodeError, TypeError):
            return None

    def status(self):
        self.view.say(
            "Python pipeline: %s"
            % (
                "đã có"
                if os.access(str(self.root / ".venv/bin/python"), os.X_OK)
                else "chưa có — chọn mục Chuẩn bị"
            )
        )
        self.view.say(f"Socket: {self.profile.socket}")
        self.view.say(f"Profile: {self.store.path}")
        if self.runner.dry_run:
            self.view.say("Worker: chưa kiểm tra trong dry-run.")
            return
        status = self.probe()
        if status is None:
            self.view.say("Worker chưa sẵn sàng — chọn Khởi động / dừng worker.")
            return
        self.view.say(
            "Worker: {} | PID: {}".format(
                status.get("state", "không rõ"), status.get("pid", "không rõ")
            )
        )
        actual = self.worker_environment(status)
        self.view.say(
            "Backend/confidence trong environment process: {} / {}".format(
                (actual or {}).get("GRASP_DEPTH_BACKEND", "không rõ"),
                (actual or {}).get("YOLOE_CONF", "không rõ"),
            )
        )
        self.view.say(
            "Đối tượng: "
            + ", ".join(str(p.get("text", p["id"])) for p in status.get("prompts", []))
        )

    def ensure_worker(self):
        if self.runner.dry_run:
            return None
        status = self.probe()
        if status is None:
            self.view.say("Worker chưa chạy. Cần nạp model trước khi xử lý ảnh.")
            if not self.view.confirm("Khởi động worker với cấu hình đã lưu?"):
                raise Back()
            self.script("worker.sh", "start")
            status = self.probe()
            if status is None:
                raise RuntimeError(
                    "Worker chưa sẵn sàng. Xem .runtime/worker.log hoặc chọn Kiểm tra trạng thái."
                )
        actual = self.worker_environment(status)
        if actual and (
            actual.get("GRASP_DEPTH_BACKEND", self.profile.backend)
            != self.profile.backend
            or float(actual.get("YOLOE_CONF", self.profile.confidence))
            != self.profile.confidence
        ):
            self.view.say(f"Environment worker khác profile: {actual}.")
            if not self.view.confirm("Khởi động lại worker để áp dụng profile?"):
                raise Back()
            self.script("worker.sh", "restart")
            status = self.probe()
            if status is None:
                raise RuntimeError("Worker chưa sẵn sàng sau restart.")
        return status

    def camera(self):
        if not self.profile.camera_k:
            self.view.say(
                "Cần K đã hiệu chuẩn của camera/ảnh: FX FY CX CY. Không dùng giá trị mẫu cho camera thật."
            )
            self.configure_camera()
        if not self.profile.camera_k:
            raise RuntimeError("Chưa có K. Chọn Thiết lập camera.")

    def image_and_prompt(self):
        image = (
            Path(self.view.ask("Đường dẫn ảnh", self.profile.last_image))
            .expanduser()
            .resolve()
        )
        if not image.is_file():
            raise ValueError(f"Không tìm thấy ảnh: {image}")
        status = self.ensure_worker()
        prompts = status.get("prompts", []) if status else []
        if prompts:
            choices = {
                str(i + 1): "{} [{}]".format(p["text"], p["id"])
                for i, p in enumerate(prompts)
            }
            selected = self.view.choose("Muốn tìm đối tượng nào?", choices)
            if selected is None:
                raise Back()
            prompt_id = prompts[int(selected) - 1]["id"]
        elif self.runner.dry_run:
            prompt_id = self.view.ask(
                "ID đã chuẩn bị (dry-run chưa kiểm worker)", "cube"
            )
        else:
            raise RuntimeError(
                "Worker chưa có đối tượng đã chuẩn bị. Chọn mục Đối tượng."
            )
        self.camera()
        self.save(replace(self.profile, last_image=str(image)))
        return image, prompt_id

    def infer(self):
        image, prompt_id = self.image_and_prompt()
        render = self.view.confirm(
            "Xuất bốn ảnh để xem detection, mask, depth và grasp?"
        )
        argv = [
            "bash",
            self.root / "scripts/infer.sh",
            str(image),
            "--prompt-id",
            prompt_id,
        ]
        if render:
            argv.append("--render")
        self.python()
        result = self.runner.run(argv, self.profile.environment(), capture=True)
        if self.runner.dry_run:
            return
        for line in result.stdout.splitlines():
            if line.startswith("RUN_ID: "):
                self.save(replace(self.profile, last_run=line.split(": ", 1)[1]))
            elif line.startswith("GRASP_POSES: "):
                grasps = json.loads(line.split(": ", 1)[1])
                self.view.say(f"Có {len(grasps)} pose gắp hợp lệ (khung camera).")
                for index, grasp in enumerate(grasps, 1):
                    xyz = ", ".join(f"{v * 1000:.1f}" for v in grasp["translation_m"])
                    self.view.say(
                        f"  {index}. Điểm {grasp['score']:.3f} | XYZ {xyz} mm | độ mở {grasp['width_m'] * 1000:.1f} mm"
                    )
            else:
                self.view.say(line)
        if render and self.profile.last_run:
            self.output(wait=True)

    def ui(self):
        self.camera()
        self.ensure_worker()
        port = self.profile.ui_port
        self.view.say(
            f"Mở trình duyệt tới http://<IP-server>:{port} . Ctrl-C để đóng UI."
        )
        self.view.say(
            f"Nếu dùng SSH tunnel: ssh -L {port}:localhost:{port} <server>, rồi mở http://localhost:{port}"
        )
        self.script("gradio.sh", "--port", str(port))

    def worker(self):
        selected = self.view.choose(
            "Vòng đời worker",
            {
                "1": "Khởi động nếu đang tắt (có thể dùng lại process đang chạy)",
                "2": "Khởi động lại, áp dụng backend/confidence đã lưu",
                "3": "Dừng worker",
                "4": "Xem trạng thái",
            },
        )
        if selected is None:
            return
        if selected == "4":
            return self.status()
        self.script("worker.sh", {"1": "start", "2": "restart", "3": "stop"}[selected])

    def configure_camera(self):
        matrix = self.view.ask(
            "Camera K: FX FY CX CY (để trống nếu chưa có)", self.profile.camera_k
        )
        size = self.view.ask(
            "Resolution hiệu chuẩn: WIDTH HEIGHT (tùy chọn)", self.profile.camera_k_size
        )
        self.save(replace(self.profile, camera_k=matrix, camera_k_size=size))

    def configure(self):
        selected = self.view.choose(
            "Thiết lập một lần, dùng lại",
            {"1": "Camera K / resolution", "2": "Runtime / socket / giao diện"},
        )
        if selected == "1":
            return self.configure_camera()
        if selected != "2":
            return
        backend = self.view.ask("Backend (da3 hoặc lite-mono)", self.profile.backend)
        confidence = float(
            self.view.ask(
                "Confidence 0..1 (preset mô phỏng dùng 0.05)", self.profile.confidence
            )
        )
        socket = self.view.ask("Socket worker", self.profile.socket)
        port = int(self.view.ask("Port giao diện", self.profile.ui_port))
        self.save(
            replace(
                self.profile,
                backend=backend,
                confidence=confidence,
                socket=socket,
                ui_port=port,
            )
        )
        self.view.say(
            "Cấu hình model chỉ có hiệu lực sau restart worker. Dùng mục Khởi động / dừng worker."
        )

    def prepare(self):
        self.view.say(
            "Chuẩn bị stack/artifact cho Jetson Xavier; có tải graph/model và cài dependency đã pin."
        )
        if self.view.confirm("Chạy chuẩn bị môi trường?"):
            self.runner.run(
                ["bash", self.root / "scripts/prepare.sh"], self.profile.environment()
            )

    def prompts(self):
        status = self.ensure_worker()
        for prompt in (status or {}).get("prompts", []):
            self.view.say("{} — {}".format(prompt["id"], prompt["text"]))
        if not self.view.confirm("Chuẩn bị bộ đối tượng mới từ file JSON (nâng cao)?"):
            return
        self.camera()
        path = (
            Path(self.view.ask("Tệp JSON chứa prompts: [{id, text}, ...]"))
            .expanduser()
            .resolve()
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        prompts = data.get("prompts") if isinstance(data, dict) else None
        if not isinstance(prompts, list) or not prompts:
            raise ValueError("File cần danh sách prompts có id và text.")
        self.view.say(
            "Cần ảnh validation của từng ID dưới model/validation/yoloe/. Build sẽ thay bộ nhận diện sau khi parity đạt."
        )
        if not self.view.confirm("Dừng worker và build/validate bộ mới?"):
            return
        self.script("worker.sh", "stop")
        self.script("preprocess_prompt.sh", str(path))
        self.view.say("Đã chuẩn bị bộ mới. Chọn Khởi động worker để dùng.")

    def output(self, wait=False):
        run_id = (
            self.profile.last_run
            if wait
            else self.view.ask("RUN_ID", self.profile.last_run)
        )
        if not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("RUN_ID cần 32 ký tự hex; lấy từ lượt infer đã chạy.")
        self.python()
        script = self.root / "scripts/output.sh"
        if not wait:
            self.runner.run(
                ["bash", script, run_id], self.profile.environment(), capture=True
            )
        result = self.runner.run(
            ["bash", script, "wait", run_id], self.profile.environment(), capture=True
        )
        if self.runner.dry_run:
            return
        job = json.loads(result.stdout)
        self.view.say("Ảnh chẩn đoán: {}".format(job.get("state", "không rõ")))
        for file in job.get("files", []):
            self.view.say("  " + str(file))
        self.view.say("Snapshot hết hạn hoặc restart worker thì cần infer lại.")

    def benchmark(self):
        image, prompt_id = self.image_and_prompt()
        runs = int(self.view.ask("Số lượt đo", "20"))
        if not 20 <= runs <= 1000:
            raise ValueError(
                "Số lượt đo phải trong 20..1000 để benchmark đạt contract nghiệm thu."
            )
        argv = [
            self.python(),
            self.root / "tests/performance/benchmark_infer.py",
            str(image),
            "--prompt-id",
            prompt_id,
            "--camera-k",
            *self.profile.camera_k.split(),
            "--runs",
            str(runs),
        ]
        if self.profile.camera_k_size:
            argv += [
                "--camera-k-size",
                *self.profile.camera_k_size.replace(",", " ").split(),
            ]
        self.runner.run(argv, self.profile.environment())
