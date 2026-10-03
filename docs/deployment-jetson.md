# Deployment trên Jetson AGX Xavier

Tài liệu này giữ lại kiến thức deployment từ README trước refactor
(`main@c417fd0`). Code và lệnh trong bước refactor vẫn giữ nguyên hành vi.

**Bước 2 hoàn thiện deployment:** giữ DA3 FP32/CUDA mặc định của #13 và bổ sung
graph/ONNX Runtime còn thiếu ở bootstrap. `GRASP_DEPTH_BACKEND=lite-mono` chọn
backend thử nghiệm, không phải fallback ngầm. Những bundle Lite-Mono dưới đây
chỉ được chuẩn bị khi chọn backend đó. Không suy diễn benchmark Lite-Mono
thành benchmark DA3. Native CUDA smoke vẫn cần chạy trên Jetson khi triển khai.

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
Khi chọn Lite-Mono, `scripts/prepare.sh` tải thêm bundle Lite-Mono. YOLOE `cube`
và VGN luôn dùng bundle đã build trên AGX Xavier
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
- YOLOE checkpoint, MobileCLIP source, artifact của depth đã chọn và VGN engine/manifest đầy đủ;
- chạy YOLOE text-encoder preparation smoke trong subprocess sạch, dùng `ultralytics/assets/bus.jpg` + prompt `person` và bắt buộc có ít nhất một box;
- chạy inference thật của depth adapter đang được composition chọn;
- deserialize và chạy một VGN TensorRT dummy inference thật.

Lưu ý: venv dùng wheel `opencv-python==4.8.1.78` vì Ultralytics yêu cầu
distribution này. OpenCV hệ thống 4.5.4 có GStreamer vẫn còn nguyên trên OS,
nhưng code chạy trong venv sẽ import bản pip; pipeline hiện nhận ảnh file/Gradio
nên không phụ thuộc GStreamer. Nếu sau này đọc camera qua GStreamer thì cần tách
camera I/O khỏi venv hoặc đổi chiến lược OpenCV.

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
và chuẩn bị depth đã chọn. Mặc định tải graph DA3 FP32 và wheel ONNX Runtime
NVIDIA từ URL/SHA-256 trong `dependencies`; chỉ khi chọn `lite-mono` mới tải
source/weights/bundle Lite-Mono. Bundle YOLOE chứa đúng prompt ID/text
`cube`/`cube`; Lite-Mono chứa encoder + decoder 192x640. Script không build lại
engine phát hành; nó kiểm tra VGN và chạy preflight inference ở cuối. Không xóa
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

`start` nạp và warmup YOLOE, depth được composition chọn và VGN một lần rồi giữ process nền qua
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

Mốc P95 dưới 1000 ms là mục tiêu lịch sử của runtime Lite-Mono, không phải
kết quả DA3. #13 ghi riêng depth DA3 khoảng 1.6–2.1 giây trên simulation gate;
phải đo lại toàn pipeline trên ảnh/prompt/camera triển khai. Mỗi lượt benchmark
cần có detection; kết quả chỉ đại diện cho tập ảnh/prompt đã đo.

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

## Volume, extrinsic và confidence

Có thể truyền `T_cam_volume` (4x4, volume -> OpenCV camera). Nếu bỏ trống, TSDF builder tạo volume camera-aligned 0.30 m quanh point cloud mục tiêu; đây là fallback, không thay thế extrinsic/table calibration cho robot thật.

**Volume phải gravity-aligned.** VGN được huấn luyện trên volume có trục Z trùng
phương trọng lực: simulator gốc dựng volume bằng
`Transform(Rotation.identity(), ...)` với gravity `[0, 0, -9.81]`, và mạng 3D CNN
không bất biến với phép quay. Đo trên cùng một cảnh với cùng depth chính xác:

| Volume | Hướng approach trong khung base | Sai vị trí |
| --- | --- | --- |
| camera-aligned (fallback) | nằm ngang | 115 mm |
| gravity-aligned | chúc thẳng xuống | 15 mm |

Robot biết pose camera trong khung base nên hãy dựng volume từ đó thay vì dựa vào
fallback:

```python
R = T_base_camera[:3, :3].T                     # trục volume = trục base, Z hướng lên
T_cam_volume = np.eye(4)
T_cam_volume[:3, :3] = R
T_cam_volume[:3, 3] = R @ (centre_base - size_m / 2 - T_base_camera[:3, 3])
```

`YOLOE_CONF` chọn ngưỡng confidence lúc chạy và được phép khác giá trị ghi trong
manifest, vì confidence chỉ áp dụng sau inference chứ không mô tả engine; `imgsz`
và sha256 của engine vẫn là ràng buộc cứng.

Để tái lập simulation gate 10/10 của 6DoF với depth do DA3 tính:

```bash
cd /workspace/6DoF_Grasp/grasp_pipeline_repo
GRASP_DEPTH_BACKEND=da3 YOLOE_CONF=0.05 bash scripts/worker.sh restart
cd /workspace/6DoF_Grasp
MUJOCO_GL=egl .venv/bin/python tools/sim_grasp_validation.py \
  --mode all --volume gravity --depth-source worker \
  --socket /workspace/6DoF_Grasp/grasp_pipeline_repo/.runtime/worker.sock \
  --photo .local_data/private/cam2.jpg
```

Worker đang tắt có thể dùng `start`; worker đã chạy cần `restart` để áp dụng
backend/confidence mới. Script không tự đặt ngưỡng `0.05`, nên cần giữ biến này
trong lệnh khởi động hoặc cấu hình service. Trên phép đối chứng do người dùng
chạy ở Jetson, mặc định `auto` cho 0/10; volume `gravity` với confidence `0.20`
cho 6/10; `gravity` với `0.05` cho 10/10 ở cả code deployed và code sau PR.
Không thay các mặc định này trong bản vá run book.

Harness dựng volume `gravity` từ pose camera và **tâm vật thể ground-truth**.
10/10 kiểm chuỗi perception → TSDF → VGN → IK → thực thi vật lý với volume đó;
chưa chứng minh robot thật tự định vị được volume. Ghi SHA hai checkout, biến
môi trường và tham số harness cùng report; pin consumer không thay cho SHA
của worker thật.


## Số đo depth và giới hạn kiểm chứng

Theo [PR #13](https://github.com/jhinezeal123/pipeline_grasppose/pull/13), trên
cùng simulation gate: Lite-Mono Pearson depth −0.43, e2e 3/10; DA3 FP32/CUDA
e2e 10/10, riêng depth 1.6–2.1 giây khi `cudnn_conv_algo_search=HEURISTIC`.
Đó là số đo của tác giả, chưa được chạy lại trong đợt refactor này.

Graph DA3 phải là FP32 `model.onnx`, SHA-256 trong `settings.py`; không dùng
`model_fp16.onnx` vì #13 ghi nhận CUDA trả depth hằng mà không báo lỗi.
`DA3_METRIC_SCALE=0.39378` chỉ fit simulation; chưa kiểm chứng trên camera thật.
Giữ nguyên scale trong refactor, đo lại ground truth trước khi dùng khoảng cách
để điều khiển robot. DA3 không thay K/extrinsic/hand-eye calibration.

Pipeline và 6DoF dùng hai environment khác nhau. Ghi SHA checkout của worker
thật cùng pin consumer và calibration; pin metadata của 6DoF không tự chứng
minh process worker đang chạy cùng SHA. Sau khi đổi code/model/artifact, restart
worker; biến môi trường ở CLI không thay worker resident đã chạy.

## DA3 graph và ONNX Runtime đã pin

| Artifact | Pin |
| --- | --- |
| DA3 graph FP32 | Hugging Face revision `159aa982e543db46b672e196d3658f1d39a258b6`, file `model.onnx`, 1.337.201.123 byte |
| Graph SHA-256 | `ba0fd3c613901450c60f29ca7be63512ff06d0b8cd1601a60f4918726ea1a9cb`, giữ nguyên từ #13 |
| ONNX Runtime | Wheel NVIDIA Jetson Zoo `onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl` |
| Wheel SHA-256 | `44c82c33c41a0670702c67af1f1425002d57f66d5efb971b35e86067d0e971d9` |
| Target | Python 3.8/aarch64, CUDA 11.x/cuDNN 8; toàn pipeline vẫn khóa Xavier L4T R35.6.4/CUDA 11.4/TensorRT 8.5.2.2 |

Nguồn wheel: [NVIDIA xác nhận wheel và checksum MD5](https://forums.developer.nvidia.com/t/cant-install-onnxruntime-on-jetpack-5-1-2/277379).
Đợt sửa này đã tải wheel thật, đối chiếu MD5 `a50941a48fbd216da67be82a5314ee5f`
và tính SHA-256 trên bytes tải về. Metadata có tag `cp38-cp38-linux_aarch64`;
CUDA library của wheel liên kết `libcudart.so.11.0`, `libcudnn.so.8` và
`libcublas.so.11`. Đây là kiểm tra file, chưa phải native execution trên Xavier.
Graph lấy đúng [revision nguồn](https://huggingface.co/Heliosoph/da3metric-large-onnx/tree/159aa982e543db46b672e196d3658f1d39a258b6);
LFS SHA/size khớp adapter, graph IR=8 và opset=17. Không tải graph FP16.

**NumPy:** wheel NVIDIA khai báo `numpy>=1.24.4` trong metadata, còn target của
repo giữ NumPy 1.23.5. Installer dùng `--no-deps` để không cho pip thay host,
và requirements pin các dependency khác của ORT. Đây là lựa chọn có chủ ý theo
[hướng dẫn JetPack 5 của Ultralytics](https://docs.ultralytics.com/guides/nvidia-jetson/#run-on-jetpack-512),
vốn yêu cầu đưa NumPy về 1.23.5 sau khi cài ORT. Không sửa metadata wheel hay
nâng NumPy/Torch âm thầm; `pip check` có thể báo xung đột NumPy này.
Import, version/provider và inference DA3 thật là **gate bắt buộc** trong
`env/check_env.py`; nếu native runtime không hoạt động, prepare báo lỗi, không
fallback sang CPU/Lite-Mono và không tuyên bố environment đã sẵn sàng.

`prepare_da3.py` kiểm hash cả file có sẵn. Download vào file tạm cùng filesystem,
chỉ rename sau khi hash đúng; lỗi download/hash không kích hoạt file tải dở.
`DA3_WEIGHTS` đổi vị trí lưu graph nhưng không đổi pin/hash. Wheel chỉ cài trong
venv Python 3.8/aarch64; không dùng wheel CUDA 12/cuDNN 9 hoặc package CPU từ
PyPI thay cho build Jetson này. Dependency khác nằm trong requirements với
constraints bảo vệ host Torch/torchvision/NumPy/SciPy.

Khi chuẩn bị và nghiệm thu trên Xavier:

```bash
bash scripts/prepare.sh
.venv/bin/python env/check_env.py
GRASP_DEPTH_BACKEND=da3 bash scripts/worker.sh restart
```

Preflight phải thấy ORT 1.16.0 và `CUDAExecutionProvider`. Adapter kiểm provider
thực đã được native session kích hoạt, không chỉ danh sách provider của package.
DA3 smoke kiểm shape, giá trị hữu hạn/dương và map không hằng; YOLOE/VGN giữ các
gate hiện có. Sau đó chạy lại simulation gate của 6DoF trên **depth-source worker**,
ảnh/K/extrinsic/calibration đúng, ghi SHA hai checkout và latency. Không dùng
unit test hoặc import/provider list để thay thế lần chạy này.
