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

Grounding-DINO, SAM, MoGe và GraspNess/MinkowskiEngine không còn nằm trên đường chạy mặc định. YOLOE được release trước khi nạp Lite-Mono; Lite-Mono được release trước khi nạp VGN TensorRT. TSDF chạy bằng NumPy/CPU để giảm peak VRAM.

## Calibration bắt buộc

Point cloud dùng **K thật của camera**, không suy intrinsics từ Lite-Mono:

```text
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
```

```bash
bash run.sh img/frame.png --prompt "the mug" --camera-k 615.2 614.8 320.1 239.7
# hoặc cho Web UI
export CAMERA_K="615.2 614.8 320.1 239.7"
```

Lite-Mono là monocular depth nên scale metric không tuyệt đối. Cần hiệu chuẩn trước khi dùng TSDF/VGN theo mét:

```bash
export LITEMONO_DEPTH_SCALE=0.73   # ví dụ; phải đo trên camera/scene thật
```

Nếu không đặt, pipeline dùng `1.0` và log cảnh báo.

## Setup Xavier

Yêu cầu JetPack đã có CUDA, TensorRT và PyTorch/torchvision tương thích Jetson. `run.sh` không cài đè CUDA/PyTorch của JetPack.

```bash
bash run.sh img/frame.png --camera-k FX FY CX CY --prompt "the object"
```

Script tạo `.venv` với `--system-site-packages`, tải `yoloe-26s-seg.pt`, clone Lite-Mono + weight 640x192, và yêu cầu `model/vgn.engine` được build trên chính Jetson.

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
result=P.pipeline(rgb,prompt="the mug",camera_K=K,top=5)
```

Có thể truyền `T_cam_volume` (4x4, volume -> OpenCV camera) để dùng task/world frame đã hiệu chuẩn. Nếu bỏ trống, TSDF builder tạo volume camera-aligned 0.30 m quanh point cloud mục tiêu; đây chỉ là fallback, không thay thế extrinsic/table calibration của robot.

## Test logic không cần GPU/model

```bash
python -m unittest tests.test_architecture tests.test_pipeline_guards tests.test_environment -v
python test_pipeline_mock.py
python test_app.py
```
