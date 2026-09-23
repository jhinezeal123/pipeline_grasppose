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

`infer.sh` không cài dependency. Nó chạy `pipeline.py` và ghi đúng một bộ:

```text
/output/<stem>_box.png
/output/<stem>_mask.png
/output/<stem>_depthmap.png
/output/<stem>_grasp.png
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
