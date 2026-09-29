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

`scripts/prepare.sh` giữ nguyên Torch/torchvision/NumPy/SciPy của host bằng
`--system-site-packages` + constraints. Python 3.8 dependencies có pin riêng
để tránh pip chọn wheel mới không còn hỗ trợ focal/aarch64.

YOLOE runtime dùng TensorRT engine tĩnh đã bake toàn bộ bộ prompt; request chỉ
gửi prompt ID và không gọi text encoder/CLIP. `scripts/preprocess_prompt.sh` tạo
embedding và export duy nhất YOLOE FP32 trên Xavier. Engine này phải vượt
kiểm tra mask, depth và grasp so với pipeline PyTorch FP32 trên ảnh validation
của từng prompt. Model YOLOE, MobileCLIP, prompt profile và engine được khóa
bằng checksum trong manifest. Worker từ chối mọi manifest YOLOE không chỉ định
FP32 hoặc thiếu bản ghi full-pipeline parity đạt ngưỡng.

Lite-Mono dùng TensorRT FP32 engine tĩnh 192x640 gồm encoder và decoder.
`scripts/prepare.sh` tải YOLOE `cube`, Lite-Mono và VGN đã build trên AGX Xavier
L4T R35.6.4 / TensorRT 8.5.2.2 từ các URL cùng SHA-256 trong `dependencies`.
Gói VGN chứa cả engine và checkpoint; cả ba được cài trong chính checkout đang
chạy. Nó kiểm tra checksum weights, profile, engine và manifest trước khi kích
hoạt; không build lại TensorRT engine khi chạy chuẩn bị. Torch vẫn được dùng
để chuyển CUDA buffer và hậu xử lý depth.

Composition root không probe Torch/CUDA trong constructor. Worker nạp các
engine một lần sau khi kiểm tra manifest, chạy warmup rồi giữ chúng trong process
nền. CLI và Gradio gửi ảnh/prompt ID qua Unix socket cục bộ.

YOLOE-26 text prompting cần thêm `mobileclip2_b.ts`. `scripts/prepare.sh` tải artifact
này, kiểm tra SHA-256, cài Ultralytics CLIP ở revision đã pin và chạy
`set_classes(["object"])` một lần. Vì vậy `scripts/infer.sh` / `scripts/gradio.sh` không cần
tự cài package hay tải text encoder ở request đầu tiên.

VGN engine trong release được build trên chính Xavier với `trtexec --fp16`.
`scripts/prepare.sh` tải nó về `model/vgn.engine` và không đọc engine từ checkout khác.
`scripts/build/export_vgn_onnx.py` vẫn có thể dùng để build thủ công khi đổi hardware
hoặc checkpoint; engine phát hành không dùng được trên T4 (sm_75) hay stack
TensorRT khác.

`env/check_env.py` kiểm tra thêm:

- đúng aarch64 / Python 3.8 / CUDA 11.4 / `sm_72`;
- Torch/torchvision/NumPy/SciPy trong venv không bị thay khỏi host versions;
- CUDA torchvision NMS hoạt động;
- TensorRT 8.5.2.2 khớp các engine đã đóng gói;
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
CLI / Gradio / Robot
    │
    ▼
GraspEstimator interface
    │
    ▼
application/
    │ depends only on
    ▼
feature modules
    ▲
    │ implemented by
    │
modules/
  ├─ vision/
  ├─ depth/
  ├─ tsdf/
  └─ grasp/
```

Cấu trúc chính:

```text
grasppose/
├── application/
│   ├── interface.py          # GraspEstimator contract
│   ├── service.py            # local estimator implementation
│   ├── grasp_pipeline.py     # internal orchestration/use-case
│   └── types.py              # public/internal result contracts
├── modules/
│   ├── vision/               # port + types + YOLOE + prompt catalog
│   ├── depth/                # port + types + geometry + Lite-Mono
│   ├── tsdf/                 # port + types + projective builder
│   └── grasp/                # port + types + VGN logic + TensorRT adapter
├── infrastructure/
│   ├── composition.py         # production composition root
│   ├── settings.py            # environment/runtime settings
│   ├── runtime.py             # logging + CUDA cleanup helpers
│   ├── artifacts.py           # checksum/manifest integrity helpers
│   ├── worker/                # Unix-socket estimator adapter/server
│   ├── output/                # snapshot/render job runtime
│   └── tensorrt/              # shared TensorRT execution runtime
├── presentation/
│   └── rendering.py
└── api.py                     # public Python interface
```

Các adapter chỉ giữ resource persistent như weights, encoder/decoder hoặc TensorRT engine. Dữ liệu theo frame không được lưu trong object; mỗi frame đi qua `predict(...)` và typed dataclass nằm cạnh từng feature trong `modules/*/types.py`.

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

K phải tương ứng với kích thước ảnh đưa vào pipeline. Nếu K được hiệu chuẩn
ở kích thước khác và ảnh chỉ được resize toàn khung, truyền thêm
`--camera-k-size WIDTH HEIGHT` (hoặc `CAMERA_K_SIZE="WIDTH HEIGHT"`).
Worker scale cả tiêu cự và tâm ảnh theo kích thước ảnh thực tế trước khi tạo
point cloud/TSDF. Ảnh crop cần hiệu chỉnh thêm tọa độ tâm ảnh.

## Chuẩn bị và chạy

Yêu cầu JetPack đã có CUDA, TensorRT và PyTorch/torchvision tương thích Jetson.

### 1. Chuẩn bị môi trường và engine depth/grasp

```bash
bash scripts/prepare.sh
```

Script tạo `.venv`, cài dependency tương thích JetPack, tải YOLOE/MobileCLIP
và Lite-Mono weights, rồi tải các bundle TensorRT FP32 đã kiểm tra từ GitHub
Release theo URL/SHA-256 trong `dependencies`. Bundle YOLOE chứa đúng prompt
ID/text `cube`/`cube`; Lite-Mono chứa encoder + decoder 192x640. Script chỉ
còn build VGN trên Xavier và chạy các preflight inference ở cuối. Không xóa
hoặc thay Torch, CUDA hay package hệ thống của JetPack. Bundle chỉ dùng được
trên AGX Xavier L4T R35.6.4 / TensorRT 8.5.2.2; môi trường khác cần build
engine riêng trên thiết bị đích.

### 2. Đóng bộ prompt thành YOLOE engine

`scripts/prepare.sh` đã cài sẵn engine FP32 cho prompt `cube`, nên có thể chạy ngay
`scripts/worker.sh` và `scripts/infer.sh --prompt-id cube`. Với bộ prompt khác, tạo `prompts.json`
gồm 1–16 prompt có thứ tự:

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
# Đặt CAMERA_K_SIZE="640 480" nếu ảnh validation khác kích thước hiệu chuẩn.
bash scripts/preprocess_prompt.sh prompts.json
```

Lệnh lưu prompt embeddings, export duy nhất engine FP32 640x640 batch 1,
kiểm tra mask IoU >= 0.99, rồi so sánh toàn pipeline trong process CUDA riêng.
Engine cần có p95 relative depth error <= 2% và, khi cả hai bản có grasp,
tâm lệch <= 7.5 mm, hướng <= 10 độ, độ mở <= 5 mm. Nếu FP32 không đạt,
lệnh dừng và khôi phục con trỏ `CURRENT` trước đó. Có thể chạy riêng
`scripts/build/validate_pipeline.py` để xem lại số đo.

### 3. Cold start và quản lý worker

```bash
bash scripts/worker.sh start
bash scripts/worker.sh status
bash scripts/worker.sh restart
bash scripts/worker.sh stop
```

`start` nạp và warmup YOLOE, Lite-Mono, VGN một lần rồi giữ process nền qua
Unix socket riêng trên máy. Gọi `start` khi worker đã sẵn sàng sẽ dùng lại
process hiện tại. Sau khi tạo bộ prompt mới, chạy `scripts/worker.sh restart` để nạp
artifact mới. Worker từ chối prompt ID không có trong profile hoặc manifest
không khớp checksum.

### 4. Inference một ảnh

```bash
bash scripts/infer.sh artifacts/examples/bag_input.png \
  --prompt-id blue_cube \
  --camera-k 615.2 614.8 320.1 239.7
```

Với `cam2.jpg` (1920x1080) và K hiệu chuẩn ở 1280x720, dùng:

```bash
bash scripts/infer.sh /path/to/cam2.jpg --prompt-id cube \
  --camera-k 957.746642 948.820235 636.883856 352.232764 \
  --camera-k-size 1280 720 --render
# Dùng RUN_ID vừa in ra để chờ bốn ảnh hoàn tất khi cần:
bash scripts/output.sh wait RUN_ID
```

CLI chỉ nhận prompt ID đã bake, không nhận prompt text tự do. Mỗi request gửi
ảnh qua worker resident và mặc định trả ngay `RUN_ID`, số detection/mask/grasp,
độ sâu, danh sách grasp pose và latency mà không render hay ghi ảnh. Dùng
`scripts/output.sh RUN_ID` để khởi chạy renderer riêng sau đó. `--render` tự xếp
việc xuất bốn ảnh vào một tiến trình riêng và trả pose ngay; dùng
`scripts/output.sh wait RUN_ID` khi cần chờ ảnh hoàn tất:

```text
artifacts/output/<RUN_ID>/box.png
artifacts/output/<RUN_ID>/mask.png
artifacts/output/<RUN_ID>/depthmap.png
artifacts/output/<RUN_ID>/grasp.png
```

Output renderer giữ snapshot có giới hạn (tối đa 8 frame / 128 MiB / 10 phút);
snapshot hết hạn khi worker khởi động lại. Xếp việc xuất ảnh theo `RUN_ID` bằng
`bash scripts/output.sh RUN_ID`; kiểm tra tiến trình bằng
`bash scripts/output.sh status RUN_ID` hoặc `bash scripts/output.sh wait RUN_ID`.
`--out DIR` hoặc `OUTPUT_DIR` chọn output directory. Renderer ghi PNG
lossless song song với
mức nén 1; có thể chỉnh qua `GRASP_PNG_COMPRESSION_LEVEL` (0–9). Dùng
`GRASP_PROFILE_INFER=1` khi khởi động worker để xem timing các stage.
Để đo latency inference sau cold start (không tính render PNG):

```bash
python tests/performance/benchmark_infer.py artifacts/examples/bag_input.png \
  --prompt-id blue_cube \
  --camera-k 615.2 614.8 320.1 239.7 \
  --runs 20
```

Ngưỡng nghiệm thu là P95 dưới 1000 ms và mọi lượt phải có detection. Kết quả
chỉ đại diện cho prompt/ảnh validation đã đo.

### 5. Gradio UI

```bash
export CAMERA_K="615.2 614.8 320.1 239.7"
bash scripts/gradio.sh --host 0.0.0.0 --port 8080
```

UI gửi inference tới worker, xếp việc xuất ảnh ở tiến trình riêng rồi chờ
bốn ảnh để hiển thị; worker có thể nhận frame tiếp theo trong lúc UI chờ.
UI dùng dropdown prompt ID và gửi request tới cùng worker với CLI. Khi đổi bộ
prompt rồi `scripts/worker.sh restart`, tải lại trang hoặc nhấn `Lam moi prompts` để lấy
danh sách ID từ worker mới mà không cần khởi động lại Gradio.

## Python API

```python
import numpy as np
from grasppose import api as P

K = np.array([
    [615.2, 0, 320.1],
    [0, 614.8, 239.7],
    [0, 0, 1.0],
])

P.load_models()

result = P.estimate(
    rgb,
    prompt_id="blue_cube",
    camera_K=K,
    top=5,
)

best = result.grasps[0]
print(best.translation_m, best.rotation, best.width_m)

# CLI, Gradio và robot đều giao tiếp qua GraspEstimator.
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

python -m tests.test_pipeline_mock
python -m tests.test_app
```
