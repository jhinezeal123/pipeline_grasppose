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

YOLOE-26 mặc định chạy FP32 trên Xavier. Adapter đi theo flow chính thức của
Ultralytics: không probe Torch/CUDA trong constructor, để `device=None` cho
Ultralytics chọn device ở bước predict, và để precision unset cho FP32 mặc
định. FP16 chỉ là opt-in bằng `YOLOE_HALF=1`, được truyền bằng
`quantize=16` (cờ `half` upstream đã deprecated). Khi chọn
`device="cpu"`, adapter không bật FP16.

Điều này cũng áp dụng cho composition root: constructor Lite-Mono không còn
gọi `torch.cuda.is_available()`. Nhờ vậy import `grasppose.facade` chỉ tạo
object graph, chưa import Torch; lần framework import đầu tiên trong production
path là khi `Yoloe26sVision.load()` import `ultralytics.YOLOE`.

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
- artifact YOLOE/MobileCLIP/Lite-Mono/VGN đầy đủ;
- chạy YOLOE semantic smoke trong subprocess sạch theo đúng import order production, dùng `ultralytics/assets/bus.jpg` + prompt `person` và bắt buộc có ít nhất một box;
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

Ví dụ:

```bash
bash infer.sh img/frame.png \
  --prompt "the mug" \
  --camera-k 615.2 614.8 320.1 239.7
```

Cho UI:

```bash
export CAMERA_K="615.2 614.8 320.1 239.7"
```

Lite-Mono là monocular depth nên scale metric không tuyệt đối. Cần hiệu chuẩn:

```bash
export LITEMONO_DEPTH_SCALE=0.73
```

Nếu không đặt, pipeline dùng `1.0` và log cảnh báo.

## Ba entrypoint

Yêu cầu JetPack đã có CUDA, TensorRT và PyTorch/torchvision tương thích Jetson.

### 1. Chuẩn bị

```bash
bash prepare.sh
```

`prepare.sh`:

- tạo `.venv` với `--system-site-packages`;
- bootstrap `pip==25.0.1` trước khi resolve dependencies; đây là bản cuối hỗ trợ Python 3.8 trong dòng pip 25.0 và nhận diện các wheel tag ARM64/manylinux mới hơn tốt hơn pip cũ đi kèm Ubuntu 20.04;
- cài Python dependencies mà không thay Torch/CUDA của JetPack;
- tải YOLOE;
- clone Lite-Mono và tải weights;
- tải checkpoint VGN chính thức nếu thiếu;
- export ONNX và build `model/vgn.engine` bằng TensorRT trên chính Jetson;
- chạy `env/check_env.py`.

Có thể override checkpoint VGN:

```bash
VGN_CHECKPOINT=/path/to/vgn_conv.pth bash prepare.sh
```

### 2. Inference một ảnh

```bash
bash infer.sh img/frame.png \
  --camera-k FX FY CX CY \
  --prompt "the object"
```

`infer.sh` không cài dependency. Mặc định nó ghi đúng một bộ vào thư mục `output/` trong repo (được `prepare.sh` tạo, nên user Jetson thông thường có quyền ghi):

```text
<repo>/output/<stem>_box.png
<repo>/output/<stem>_mask.png
<repo>/output/<stem>_depthmap.png
<repo>/output/<stem>_grasp.png
```

Nếu môi trường/container đã provision một thư mục tuyệt đối khác, có thể override:

```bash
OUTPUT_DIR=/output bash infer.sh img/frame.png --camera-k FX FY CX CY
```

### 3. UI Space

```bash
export CAMERA_K="FX FY CX CY"
bash space.sh --host 0.0.0.0 --port 8080
```

UI preload các model một lần rồi tái sử dụng cho mọi request.

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
    prompt="the mug",
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
  tests.test_environment -v

python test_pipeline_mock.py
python test_app.py
```
