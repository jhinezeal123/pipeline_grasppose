# Môi trường chạy

`bash run.sh img/anh.png` tạo `.venv` riêng với `--system-site-packages`,
để dùng lại Torch/CUDA/MinkowskiEngine của máy. Mọi gói bổ sung được cài vào
`.venv`, không vào Python của notebook. Không cần `ensurepip`: pip của Python
chủ quản cài qua `--python .venv/bin/python` (cần pip >=22.3).

## Điều kiện trước khi chạy

- Linux, Python >=3.10; môi trường cũ ghi nhận trong snapshot dùng Python 3.12.
- Torch, torchvision, NumPy và MinkowskiEngine phải được cài sẵn và import được.
- GPU CUDA dùng được với Torch; CUDA toolkit (`nvcc`) và compiler để build pointnet2.
- MinkowskiEngine phải khớp Python/Torch/CUDA của máy. Repo không cung cấp wheel
  dùng chung cho mọi máy. Nếu thiếu, setup dừng trước khi tải model và báo tên gói.
- Internet để tải dependencies và model lần đầu.

`requirements.txt` là danh sách runtime có giới hạn phiên bản, **không phải lock
đầy đủ**. MoGe được ghim commit; pip giải cả dependencies gián tiếp của MoGe.
Các phiên bản Torch, torchvision, NumPy, SciPy, MinkowskiEngine, Triton và NVIDIA
đang có trên máy được ghi vào `.venv/host-constraints.txt`. Nếu yêu cầu mới xung đột,
pip dừng thay vì tự đổi bộ CUDA. `.venv/host.json` phát hiện thay đổi môi trường chủ.

`requirements.lock.txt` là snapshot lịch sử do repo cung cấp, không dùng để
bootstrap: thiếu dependencies gián tiếp/MoGe và không xác định đầy đủ index CUDA.
Không xem nó là bằng chứng rằng máy mới đã được kiểm thử.

## Chạy và chẩn đoán

```bash
bash run.sh img/anh.png
bash run.sh --serve --port 8080
.venv/bin/python env/check_env.py
```

Nếu setup báo môi trường chủ thay đổi, di chuyển `.venv` cũ sang nơi khác rồi
chạy lại. Build lại pointnet2 với bộ Torch/CUDA mới; chỉ có file `.so` là chưa đủ,
`run.sh` còn kiểm tra import thực để phát hiện lỗi ABI.

Không thêm `env/lib` hoặc `model/moge_repo` cũ vào `PYTHONPATH`: chúng có thể che
khuất gói mới. Với shell từng export các đường dẫn này, mở shell mới trước khi chạy.
Gói bổ sung được cách ly nhưng thư viện native vẫn phụ thuộc môi trường chủ;
đây không phải môi trường hermetic hay image Docker đã kiểm thử.

## Kiểm thử không cần GPU

```bash
python3 -m unittest discover -s tests -v
python3 test_pipeline_mock.py
python3 test_app.py
```
