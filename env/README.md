# Môi trường chạy

> Nhanh Jetson AGX Xavier: xem them `JETSON_XAVIER.md`. Stack muc tieu la
> Python 3.8 + Torch NVIDIA 2.1 + CUDA 11.4 + sm_72; khong nang cap cac goi host
> ABI-sensitive bang pip.

`bash run.sh img/anh.png` tạo `.venv` riêng với `--system-site-packages`,
để dùng lại Torch/CUDA của máy. Mọi gói bổ sung được cài vào
`.venv`, không vào Python của notebook. Không cần `ensurepip`: pip của Python
chủ quản cài qua `--python .venv/bin/python` (cần pip >=22.3).

## Điều kiện trước khi chạy

- Linux, Python >=3.8; profile Jetson muc tieu dung Python 3.8.10.
- Torch, torchvision và NumPy phải được cài sẵn và import được.
- MinkowskiEngine **không** bắt buộc có sẵn; repo cài ở giai đoạn 2.
- GPU CUDA dùng được với Torch; CUDA toolkit (`nvcc`) và compiler để build pointnet2.
- Nếu build MinkowskiEngine từ source: cần compiler C++, Python headers và
  OpenBLAS headers/library; `nvcc` phải cùng phiên bản major.minor với Torch CUDA.
- Internet để tải dependencies và model lần đầu.

`requirements.txt` la runtime core Python-3.8-compatible, **khong phai lock day du**.\nMoGe-2 va utils3d duoc clone tu revision da ghim trong `dependencies` thay vi pip-install;\n`requirements-ui.txt` chi duoc dung khi chay `--serve`.
Các phiên bản Torch, torchvision, torchaudio, NumPy, SciPy, Triton và NVIDIA
đang có trên máy được ghi vào `.venv/host-constraints.txt`. Nếu yêu cầu mới xung đột,
pip dừng thay vì tự đổi bộ CUDA. `.venv/host.json` phát hiện thay đổi môi trường chủ.

MinkowskiEngine **không** nằm trong constraints: installer riêng của nó
(`env/install_minkowski.py`) cũng truyền `-c host-constraints.txt`, nên nếu host
đang có 0.5.3 thì constraints sẽ ghim `minkowskiengine==0.5.3` và pip từ chối
bundled wheel 0.5.4 bằng `ResolutionImpossible` — tức là chính constraints phá cơ
chế fallback mà nó sinh ra để bảo vệ.

`requirements.lock.txt` là snapshot lịch sử do repo cung cấp, không dùng để
bootstrap: thiếu dependencies gián tiếp/MoGe và không xác định đầy đủ index CUDA.
Không xem nó là bằng chứng rằng máy mới đã được kiểm thử.

## Tải model: ghim revision

`run.sh` đọc `dependencies`. Mỗi block `KIND = hf` có `REVISION` là commit SHA
trên HuggingFace, truyền vào `snapshot_download(revision=...)`. Không ghim thì HF
trả bản mới nhất và nội dung có thể đổi bất cứ lúc nào — kết quả không tái lập được.
Các block `git` cũng ghim `REVISION`.

`run.sh` **luôn** gọi `snapshot_download`, không tự kiểm tra "đã tải xong chưa" rồi
bỏ qua. Đó không phải là bỏ sót tối ưu: mọi cách tự kiểm tra đều không thể biết repo
**cần** bao nhiêu file, vì nó chỉ nhìn thấy file đang có trên đĩa. Repo cần 10 file,
tải xong `config.json` rồi mất mạng → thư mục không rỗng, file đó có metadata hợp lệ
→ lần sau bỏ qua → model thiếu 9 file weights, lỗi chỉ lộ ra rất muộn.

Gọi lại mỗi lần là **đúng và rẻ**: `snapshot_download` liệt kê file của repo rồi gọi
`hf_hub_download` cho **từng** file (`_snapshot_download.py`), mà `hf_hub_download`
tự kiểm *"file đã có + metadata khớp commit_hash ⇒ trả luôn"* (`file_download.py`).
Nên nó chỉ tải file còn thiếu, và bản tải dở nằm ở `*.incomplete` nên lần sau tự
resume tiếp. Đổi lại, bỏ được một hàm kiểm tra tự viết và cái bug của nó.

## MinkowskiEngine: cài sau khi tải model

`run.sh` gọi `env/install_minkowski.py` sau `setup_env.py --install`:

1. Nếu đặt `MINKOWSKI_ENGINE_WHEEL`, cài đúng file đó vào `.venv` với `--no-deps`.
2. Nếu không đặt, kiểm tra bản đã cài bằng **CUDA sparse convolution thật**.
   Nếu chạy được, giữ nguyên bản đó; không cài đè dù repo có wheel.
3. Nếu kiểm tra thất bại, tìm wheel đi kèm trong `model/`, lọc theo Python/ABI/platform
   của **interpreter `.venv`**. Wheel `cp312` không được tự chọn trên Python 3.10/3.11.
   Nếu nhiều wheel cùng tương thích, yêu cầu đặt `MINKOWSKI_ENGINE_WHEEL` rõ ràng.
4. Nếu không có wheel phù hợp và package chưa cài, thử build source NVIDIA ghim
   commit `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228` như trước.
5. Nếu bản đã cài bị lỗi và không có wheel phù hợp, giữ traceback và báo lỗi;
   không che lỗi ABI bằng một source build khác.

Wheel tags chỉ kiểm tra Python/ABI/platform, **không xác nhận Torch/CUDA ABI**.
Sau mỗi lần cài vẫn bắt buộc chạy CUDA smoke test.

### Ghi chu lich su ve wheel Kaggle\n\nWheel bundled **đã kiểm chứng trên Kaggle Torch 2.10/CUDA 12.8, box trắng không
mount gì** (commit `30fd5e9`): Gradio lên sau 270s, `depth_m = 0.482` — khớp giá
trị đo được ở lần chạy có mount (`0.4824655055999756`). Interpreter `.venv` import
được `MinkowskiEngine` và qua CUDA smoke test thật trên Tesla T4.

Ví dụ ghi đè bằng wheel khác (thay đường dẫn bằng file thật):

```bash
MINKOWSKI_ENGINE_WHEEL=/kaggle/input/your-dataset/minkowskiengine-0.5.4-cp312-cp312-linux_x86_64.whl bash run.sh --serve
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

