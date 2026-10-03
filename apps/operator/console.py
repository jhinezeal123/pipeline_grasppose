"""Danh mục công việc là nguồn duy nhất cho menu và --list."""

import argparse
from dataclasses import replace
from pathlib import Path

from .config import ProfileStore
from .tasks import FEATURES, Tasks
from .terminal import Back, CommandRunner, Terminal


def main(argv=None, terminal=None):
    parser = argparse.ArgumentParser(
        description="Menu perception tiếng Việt. Chạy ./start; :q để quay lại khi nhập dữ liệu."
    )
    parser.add_argument("action", nargs="?", choices=list(FEATURES))
    parser.add_argument(
        "--list", action="store_true", help="xem toàn bộ tính năng, không nạp model"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="xem lệnh dự kiến, không chạy tác vụ"
    )
    parser.add_argument("--config", type=Path, help="file profile riêng của menu")
    parser.add_argument("--image", type=Path, help="ảnh gợi ý cho tác vụ infer")
    parser.add_argument("--socket", type=Path, help="socket do consumer chỉ định")
    args = parser.parse_args(argv)
    view = terminal or Terminal()
    root = Path(__file__).resolve().parents[2]
    if args.list:
        view.say("PIPELINE — các công việc có thể làm")
        for key, (title, summary, _) in FEATURES.items():
            view.say(f"./start {key} — {title}\n  {summary}")
        return 0
    try:
        store = ProfileStore(root, args.config)
        tasks = Tasks(root, store, view, CommandRunner(root, view, args.dry_run))
        if args.image:
            tasks.profile = replace(
                tasks.profile, last_image=str(args.image.expanduser().resolve())
            )
        if args.socket:
            tasks.profile = replace(
                tasks.profile, socket=str(args.socket.expanduser().resolve())
            )
        while True:
            view.say("\nPERCEPTION — ảnh → đối tượng → pose gắp (khung camera)")
            view.say(
                "Backend cấu hình: {} | confidence: {:g} | camera: {}".format(
                    tasks.profile.backend,
                    tasks.profile.confidence,
                    "đã có K" if tasks.profile.camera_k else "chưa có K",
                )
            )
            view.say("Nhập :q để quay lại; '-' để xóa giá trị đã lưu.")
            if args.action:
                key = args.action
            else:
                selected = view.choose(
                    "Bạn muốn làm gì?",
                    {
                        str(i + 1): f"{title} [{tasks.availability(action)}]"
                        for i, (action, (title, _, _)) in enumerate(FEATURES.items())
                    },
                )
                if selected is None:
                    return 0
                key = list(FEATURES)[int(selected) - 1]
            try:
                view.say(FEATURES[key][1])
                getattr(tasks, key)()
            except Back:
                view.say("Đã quay lại.")
            except (OSError, RuntimeError, ValueError) as exc:
                view.say(f"Chưa chạy được: {exc}")
                view.say(
                    "Dùng mục Thiết lập hoặc Kiểm tra trạng thái để xem bước còn thiếu."
                )
                if args.action:
                    return 1
            if args.action:
                return 0
    except (EOFError, Back):
        view.say("\nĐã thoát menu.")
        return 0
    except KeyboardInterrupt:
        view.say("\nĐã ngắt tác vụ.")
        return 130
    except (OSError, ValueError, TypeError) as exc:
        view.say(f"Không đọc được cấu hình: {exc}. Dùng --config để chọn file khác.")
        return 1
