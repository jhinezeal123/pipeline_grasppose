# Môi trường chạy

`bash run.sh img/anh.png` tạo `.venv` riêng với `--system-site-packages`,
để dùng lại Torch/CUDA của máy. Mọi gói bổ sung được cài vào
`.venv`, không vào Python của notebook. Không cần `ensurepip`: pip của Python
chủ quản cài qua `--python .venv/bin/python` (cần pip >=22.3).

## Điều kiện trước khi chạy

- Linux, Python >=3.10; môi trường cũ ghi nhận trong snapshot dùng Python 3.12.
- Torch, torchvision và NumPy phải được cài sẵn và import được.
- MinkowskiEngine **không** bắt buộc có sẵn; repo cài ở giai đoạn 2.
- GPU CUDA dùng được với Torch; CUDA toolkit (`nvcc`) và compiler để build pointnet2.
- Nếu build MinkowskiEngine từ source: cần compiler C++, Python headers và
  OpenBLAS headers/library; `nvcc` phải cùng phiên bản major.minor với Torch CUDA.
- Internet để tải dependencies và model lần đầu.

`requirements.txt` là danh sách runtime có giới hạn phiên bản, **không phải lock
đầy đủ**. MoGe được ghim commit; pip giải cả dependencies gián tiếp của MoGe.
Các phiên bản Torch, torchvision, NumPy, SciPy, MinkowskiEngine, Triton và NVIDIA
đang có trên máy được ghi vào `.venv/host-constraints.txt`. Nếu yêu cầu mới xung đột,
pip dừng thay vì tự đổi bộ CUDA. `.venv/host.json` phát hiện thay đổi môi trường chủ.

`requirements.lock.txt` là snapshot lịch sử do repo cung cấp, không dùng để
bootstrap: thiếu dependencies gián tiếp/MoGe và không xác định đầy đủ index CUDA.
Không xem nó là bằng chứng rằng máy mới đã được kiểm thử.

## MinkowskiEngine: cài sau khi tải model

`run.sh` gọi `env/install_minkowski.py` sau `setup_env.py --install`:

1. Nếu đặt `MINKOWSKI_ENGINE_WHEEL`, cài đúng file đó vào `.venv` với `--no-deps`.
2. Nếu không đặt, thử bản đã cài bằng **CUDA sparse convolution thật**, không chỉ import.
3. Nếu chưa có package, build source NVIDIA ghim commit
   `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`, rồi cài wheel vào `.venv`.
4. Nếu package có nhưng lỗi ABI/CUDA, giữ traceback và yêu cầu wheel tương thích;
   không tự ghi đè bản lỗi bằng một source build khác.

Ví dụ dùng wheel đã kiểm chứng trên Kaggle (thay đường dẫn bằng file thật):

```bash
MINKOWSKI_ENGINE_WHEEL=/kaggle/input/your-dataset/MinkowskiEngine-0.5.4-cp312-cp312-linux_x86_64.whl bash run.sh --serve
```

Tên wheel chỉ mô tả Python/platform, **không chứng minh khớp Torch/CUDA**.
Installer luôn chạy CUDA smoke test sau khi cài. Không tự chọn wheel đầu tiên từ
`/kaggle/input`, vì có thể lấy nhầm bản ABI.

Nếu không có wheel, `bash run.sh --serve` tự thử build source. Chuẩn bị trên Ubuntu:

```bash
sudo apt-get install build-essential python3-dev libopenblas-dev
```

Script không tự chạy apt/sudo. Có thể đặt `CUDA_HOME`, `CXX`, `MAX_JOBS` (mặc định 2).
Log compile được giữ ở `.venv/minkowski-build.log`. Source build dùng interpreter
và Torch hiện tại, truyền `--force_cuda --blas=openblas`; bản sao setup.py được bỏ
lệnh upstream gọi `pip uninstall` để tránh chạm môi trường host.

**Chưa kiểm chứng build trên Kaggle Python 3.12 / Torch 2.10 / CUDA 12.8.**
Upstream cũ có thể không biên dịch được với bộ mới; khi đó cần wheel đã kiểm chứng
hoặc bản vá compiler riêng. Không coi source fallback là cam kết box trắng chạy
thành công. Kiểm thử tự động hiện kiểm tra luồng cài bằng mock, không thay GPU test.
Model đã tải được giữ nguyên nếu cài runtime hoặc build thất bại.

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
