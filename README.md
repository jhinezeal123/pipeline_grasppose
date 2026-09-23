# Jetson Xavier grasp-pose pipeline

Pipeline mặc định của nhánh này:

```text
RGB
 └─ YOLOE-26s-seg -> bbox + instance mask
     └─ Lite-Mono -> depth map
         └─ depth + camera K + mask -> point cloud
             └─ projective TSDF 0.30 m / 40^3
                 └─ VGN TensorRT -> 6-DoF grasp poses
```

Grounding-DINO, SAM, MoGe và GraspNess/MinkowskiEngine không còn nằm trên đường chạy mặc định. YOLOE, Lite-Mono và VGN TensorRT được **nạp một lần và giữ resident** trong suốt process; mỗi frame chỉ chạy inference. `close_models()` mới giải phóng model khi process kết thúc. TSDF chạy bằng NumPy/CPU.

## Calibration bắt buộc

Point cloud dùng **K thật của camera**, không suy intrinsics từ Lite-Mono:

```text
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

```bash
bash infer.sh img/frame.png --prompt "the mug" --camera-k 615.2 614.8 320.1 239.7
# hoặc cho Web UI
export CAMERA_K="615.2 614.8 320.1 239.7"
```

Lite-Mono là monocular depth nên scale metric không tuyệt đối. Cần hiệu chuẩn trước khi dùng TSDF/VGN theo mét:

```bash
export LITEMONO_DEPTH_SCALE=0.73   # ví dụ; phải đo trên camera/scene thật
```

Nếu không đặt, pipeline dùng `1.0` và log cảnh báo.

## Ba entrypoint

Yêu cầu JetPack đã có CUDA, TensorRT và PyTorch/torchvision tương thích Jetson.

### 1. Chuẩn bị tài nguyên

```bash
bash prepare.sh
```

`prepare.sh` tạo `.venv` với `--system-site-packages`, cài Python dependencies nhưng không thay CUDA/PyTorch của JetPack, tải `yoloe-26s-seg.pt`, clone Lite-Mono và tải weight 640x192.

VGN TensorRT engine phải được build trên chính Jetson. Đặt checkpoint tại `model/vgn_conv.pth` hoặc truyền `VGN_CHECKPOINT=/path/to/vgn_conv.pth`; nếu chưa có `model/vgn.engine`, `prepare.sh` tự export ONNX và chạy `trtexec --fp16`.

### 2. Inference một ảnh

```bash
bash infer.sh img/frame.png --camera-k FX FY CX CY --prompt "the object"
```

`infer.sh` chỉ chạy `pipeline.py`, không bootstrap dependency. Mỗi lần chạy ghi một bộ 4 ảnh `*_box.png`, `*_mask.png`, `*_depthmap.png`, `*_grasp.png` vào thư mục tuyệt đối `/output`.

### 3. UI

```bash
export CAMERA_K="FX FY CX CY"
bash space.sh --host 0.0.0.0 --port 8080
```

`space.sh` khởi động Gradio UI. Các model được preload một lần khi service khởi động và giữ resident cho các request tiếp theo.

### Build VGN TensorRT

Lấy checkpoint gốc `vgn_conv.pth`, rồi:

```bash
.venv/bin/python tools/export_vgn_onnx.py --checkpoint /path/vgn_conv.pth --out model/vgn.onnx
trtexec --onnx=model/vgn.onnx --saveEngine=model/vgn.engine --fp16
```

VGN nhận TSDF `(1,40,40,40)`; TensorRT input sau batch là `(1,1,40,40,40)`.

## Python API

```python
import numpy as np
import pipeline as P
K=np.array([[615.2,0,320.1],[0,614.8,239.7],[0,0,1.]])
P.load_models()  # optional: preload before the first frame
result=P.pipeline(rgb,prompt="the mug",camera_K=K,top=5)
# loop frames: P.pipeline(...) reuses the same resident models
# P.close_models() only when shutting down
```

Có thể truyền `T_cam_volume` (4x4, volume -> OpenCV camera) để dùng task/world frame đã hiệu chuẩn. Nếu bỏ trống, TSDF builder tạo volume camera-aligned 0.30 m quanh point cloud mục tiêu; đây chỉ là fallback, không thay thế extrinsic/table calibration của robot.

## Test logic không cần GPU/model

```bash
python -m unittest tests.test_architecture tests.test_pipeline_guards tests.test_environment -v
python test_pipeline_mock.py
python test_app.py
```
