# Jetson Xavier grasp-pose pipeline

Runtime:

```text
RGB
 └─ YOLOE-26s-seg -> bbox + instance mask
     └─ Lite-Mono -> depth map
         └─ depth + camera K + mask -> point cloud
             └─ projective TSDF 0.30 m / 40^3
                 └─ VGN TensorRT -> 6-DoF grasp poses
```

YOLOE, Lite-Mono và VGN TensorRT được nạp một lần và giữ resident trong suốt process. Mỗi frame chỉ chạy inference; model chỉ được giải phóng khi gọi `close_models()` hoặc service kết thúc.


## Hardware target

Nhánh này được khóa cho Jetson AGX Xavier 32 GB / L4T R35.6.4 (JetPack 5.1.6):

- Ubuntu 20.04 / L4T R35.6.4;
- aarch64 + Carmel CPU;
- Volta GPU compute capability 7.2 (`sm_72`);
- CUDA 11.4, cuDNN 8.6, TensorRT 8.5.x;
- Python 3.8;
- NVIDIA Torch 2.1.0a0 + torchvision 0.16.x;
- NumPy 1.23.5 / SciPy 1.10.1 từ host JetPack.

Lưu ý versioning NVIDIA: JetPack 5.1.4 gốc đi với L4T 35.6.0; target thực tế ở đây là L4T 35.6.4, tương ứng JetPack 5.1.6. Compute stack vẫn là CUDA 11.4 / cuDNN 8.6 / TensorRT 8.5.x.

`prepare.sh` giữ nguyên Torch/torchvision/NumPy/SciPy của host bằng
`--system-site-packages` + constraints. Python 3.8 dependencies có pin riêng
để tránh pip chọn wheel mới không còn hỗ trợ focal/aarch64.

YOLOE runtime dùng TensorRT engine tĩnh đã bake toàn bộ bộ prompt; request chỉ
gửi prompt ID và không gọi text encoder/CLIP. `preprocess_prompt.sh` tạo
embedding và export ứng viên FP32/FP16 trên Xavier, rồi chọn ứng viên nhanh
nhất vượt kiểm tra mask, depth và grasp so với pipeline PyTorch FP32 trên ảnh
validation của từng prompt. Model YOLOE, MobileCLIP, prompt profile và engine
được khóa bằng checksum trong manifest. Worker chỉ nạp artifact có bản ghi
full-pipeline parity đạt ngưỡng cho đúng engine đã chọn.

Lite-Mono cũng dùng TensorRT engine tĩnh 192x640 gồm encoder và decoder.
Exporter so sánh FP32/FP16 với checkpoint PyTorch và chỉ nhận ứng viên có
p95 relative depth error <= 2%. Torch vẫn được dùng để chuyển CUDA buffer và
hậu xử lý depth; không có PyTorch model fallback khi runtime thiếu engine.

Composition root không probe Torch/CUDA trong constructor. Worker nạp các
engine một lần sau khi kiểm tra manifest, chạy warmup rồi giữ chúng trong process
nền. CLI và Gradio gửi ảnh/prompt ID qua Unix socket cục bộ.

YOLOE-26 text prompting cần thêm `mobileclip2_b.ts`. `prepare.sh` tải artifact
này, kiểm tra SHA-256, cài Ultralytics CLIP ở revision đã pin và chạy
`set_classes(["object"])` một lần. Vì vậy `infer.sh` / `space.sh` không cần
tự cài package hay tải text encoder ở request đầu tiên.

VGN ONNX được export bằng `onnx==1.14.1` trên Python 3.8 và TensorRT engine
được build bằng `trtexec` ngay trên Xavier. Không reuse engine build trên T4
(sm_75) hay máy TensorRT khác.

`env/check_env.py` kiểm tra thêm:

- đúng aarch64 / Python 3.8 / CUDA 11.4 / `sm_72`;
- Torch/torchvision/NumPy/SciPy trong venv không bị thay khỏi host versions;
- CUDA torchvision NMS hoạt động;
- TensorRT >= 8.5 và `trtexec` có mặt;
- YOLOE checkpoint, MobileCLIP source, Lite-Mono TensorRT artifact và VGN engine/manifest đầy đủ;
- chạy YOLOE text-encoder preparation smoke trong subprocess sạch, dùng `ultralytics/assets/bus.jpg` + prompt `person` và bắt buộc có ít nhất một box;
- chạy Lite-Mono CUDA inference thật;
- deserialize và chạy một VGN TensorRT dummy inference thật.

Lưu ý: venv dùng wheel `opencv-python==4.8.1.78` vì Ultralytics yêu cầu
distribution này. OpenCV hệ thống 4.5.4 có GStreamer vẫn còn nguyên trên OS,
nhưng code chạy trong venv sẽ import bản pip; pipeline hiện nhận ảnh file/Gradio
nên không phụ thuộc GStreamer. Nếu sau này đọc camera qua GStreamer thì cần tách
camera I/O khỏi venv hoặc đổi chiến lược OpenCV.

## Kiến trúc OOP

```text
CLI / Gradio
    │
    ▼
facade.py
    │
    ▼
application/
    │ depends only on
    ▼
ports/ + domain/
    ▲
    │ implemented by
    │
adapters/
  ├─ yoloe.py
  ├─ lite_mono.py
  └─ vgn_trt.py
```

Cấu trúc chính:

```text
grasppose/
├── adapters/                 # code phụ thuộc framework/model
│   ├── yoloe.py
│   ├── lite_mono.py
│   └── vgn_trt.py
├── application/
│   └── grasp_pipeline.py     # orchestration/use-case
├── domain/                   # numpy/scipy, không biết model framework
│   ├── geometry.py
│   ├── tsdf.py
│   ├── types.py
│   └── vgn.py
├── ports/                    # interface/contract
│   ├── vision.py
│   ├── depth.py
│   ├── tsdf.py
│   └── grasp.py
├── presentation/
│   └── rendering.py
├── bootstrap.py              # composition root
├── facade.py                 # API chung cho CLI/UI
├── config.py
└── runtime.py
```

Các adapter chỉ giữ resource persistent như weights, encoder/decoder hoặc TensorRT engine. Dữ liệu theo frame không được lưu trong object; mỗi frame đi qua `predict(...)` và typed dataclass trong `domain/types.py`.

Các implementation Grounding-DINO, SAM, MoGe và GraspNess/MinkowskiEngine cũ đã được loại khỏi source tree để repository chỉ có một runtime architecture canonical.

## Calibration bắt buộc

Point cloud dùng K thật của camera:

```text
K = [[fx, 0, cx],
     [0, fy, cy],
     [0,  0,  1]]
```

Truyền camera matrix qua `--camera-k FX FY CX CY`, hoặc đặt
`CAMERA_K="FX FY CX CY"`. Có thể dùng `--fov-x DEGREES` cho ảnh synthetic
khi chưa có calibration thật. Lite-Mono là monocular depth nên scale metric
không tuyệt đối; đặt `LITEMONO_DEPTH_SCALE` sau khi hiệu chuẩn nếu cần.

## Chuẩn bị và chạy

Yêu cầu JetPack đã có CUDA, TensorRT và PyTorch/torchvision tương thích Jetson.

### 1. Chuẩn bị môi trường và engine depth/grasp

```bash
bash prepare.sh
```

Script tạo `.venv`, cài dependency tương thích JetPack, tải YOLOE/MobileCLIP
và Lite-Mono weights, rồi export + build Lite-Mono và VGN TensorRT ngay trên
Xavier. Nó chạy các preflight inference ở cuối. Không xóa hoặc thay Torch,
CUDA hay package hệ thống của JetPack.

### 2. Đóng bộ prompt thành YOLOE engine

Tạo `prompts.json` với 1–16 prompt có thứ tự:

```json
{
  "prompts": [
    {"id": "blue_cube", "text": "blue cube"},
    {"id": "red_mug", "text": "red mug"}
  ]
}
```

Đặt một ảnh validation cho từng ID trong `model/validation/yoloe/`, ví dụ
`model/validation/yoloe/blue_cube.png`. Ảnh phải có object của prompt đó.
Đặt `CAMERA_K` đúng với camera của các ảnh validation, rồi chạy khi worker
đã dừng:

```bash
export CAMERA_K="615.2 614.8 320.1 239.7"
bash preprocess_prompt.sh prompts.json
```

Lệnh lưu prompt embeddings, export engine 640x640 batch 1 theo FP32 và FP16,
kiểm tra mask IoU >= 0.99, rồi so sánh toàn pipeline trong các process CUDA
riêng. Nó chọn engine nhanh nhất có p95 relative depth error <= 2% và, khi
cả hai bản có grasp, tâm lệch <= 7.5 mm, hướng <= 10 độ, độ mở <= 5 mm.
Nếu không có ứng viên đạt ngưỡng, lệnh dừng và khôi phục con trỏ `CURRENT`
trước đó. Có thể chạy riêng `tools/validate_trt_parity.py` để xem lại số đo.

### 3. Cold start và quản lý worker

```bash
bash cold.sh start
bash cold.sh status
bash cold.sh restart
bash cold.sh stop
```

`start` nạp và warmup YOLOE, Lite-Mono, VGN một lần rồi giữ process nền qua
Unix socket riêng trên máy. Gọi `start` khi worker đã sẵn sàng sẽ dùng lại
process hiện tại. Sau khi tạo bộ prompt mới, chạy `cold.sh restart` để nạp
artifact mới. Worker từ chối prompt ID không có trong profile hoặc manifest
không khớp checksum.

### 4. Inference một ảnh

```bash
bash infer.sh img/frame.png \
  --prompt-id blue_cube \
  --camera-k 615.2 614.8 320.1 239.7
```

CLI chỉ nhận prompt ID đã bake, không nhận prompt text tự do. Mỗi request gửi
ảnh qua worker resident, rồi ghi bốn file:

```text
output/<stem>_box.png
output/<stem>_mask.png
output/<stem>_depthmap.png
output/<stem>_grasp.png
```

Có thể đặt output directory bằng `--out DIR` hoặc `OUTPUT_DIR`. Worker ghi
PNG lossless song song (4 writer) với nén mức 1 để giảm latency; có thể điều
chỉnh bằng `GRASP_PNG_WORKERS` và `GRASP_PNG_COMPRESSION_LEVEL` (0–9). Dùng
`GRASP_PROFILE_INFER=1` khi khởi động worker để xem thời gian từng stage/render/write.
Để đo yêu cầu latency sau cold start (ít nhất 20 lần, gồm IPC và bốn lần ghi PNG):

```bash
python tools/benchmark_infer.py img/frame.png \
  --prompt-id blue_cube \
  --camera-k 615.2 614.8 320.1 239.7 \
  --runs 20
```

Ngưỡng nghiệm thu là P95 dưới 1000 ms và mọi lượt phải có detection. Kết quả
chỉ đại diện cho prompt/ảnh validation đã đo.

### 5. Gradio UI

```bash
export CAMERA_K="615.2 614.8 320.1 239.7"
bash space.sh --host 0.0.0.0 --port 8080
```

UI dùng dropdown prompt ID và gửi request tới cùng worker với CLI. Khi đổi bộ
prompt rồi `cold.sh restart`, tải lại trang hoặc nhấn `Lam moi prompts` để lấy
danh sách ID từ worker mới mà không cần khởi động lại Gradio.

## Python API

```python
import numpy as np
import pipeline as P

K = np.array([
    [615.2, 0, 320.1],
    [0, 614.8, 239.7],
    [0, 0, 1.0],
])

P.load_models()

result = P.pipeline(
    rgb,
    prompt_id="blue_cube",
    camera_K=K,
    top=5,
)

# các frame tiếp theo tái sử dụng model resident
# P.close_models() khi shutdown
```

Có thể truyền `T_cam_volume` (4x4, volume -> OpenCV camera). Nếu bỏ trống, TSDF builder tạo volume camera-aligned 0.30 m quanh point cloud mục tiêu; đây là fallback, không thay thế extrinsic/table calibration cho robot thật.

## Test

Các test logic không yêu cầu GPU/model:

```bash
python -m unittest \
  tests.test_architecture \
  tests.test_pipeline_guards \
  tests.test_prompt_catalog \
  tests.test_environment -v

python test_pipeline_mock.py
python test_app.py
```
