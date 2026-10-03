# pipeline_grasppose — module perception

Repo này nhận ảnh RGB + camera calibration và trả các pose gắp 6DoF.
`6DoF_Grasp` là consumer; pipeline không điều khiển robot.

Đọc [bản đồ code](docs/kien-truc-vi.md) và
[review đủ 14 PR](docs/review-pr-vi.md) trước khi thay runtime.

## Luồng chạy

RGB → YOLOE mask → depth → cloud → TSDF → VGN → `EstimateResult`.
Composition ở `grasppose/infrastructure/composition.py`. Model resident được
nạp/warmup một lần trong worker; render ảnh chẩn đoán ở process riêng.

Ở nền `c417fd0`, depth mặc định là **DA3 metric-large**. Bootstrap hiện vẫn
chuẩn bị Lite-Mono, chưa tải DA3/cài ONNX Runtime. Refactor giữ nguyên sự lựa
chọn này; việc rollback mặc định depth nằm trong PR sau. Không coi test CPU
xanh là bằng chứng cài mới hoặc inference Jetson đã chạy được.

## Nơi tìm code

| Mục đích | Thư mục/file |
| --- | --- |
| API consumer, input, selection, workflow | `grasppose/application/`, `grasppose/api.py` |
| Port và model từng bước | `grasppose/modules/{vision,depth,tsdf,grasp}/` |
| Chọn backend/config | `grasppose/infrastructure/composition.py`, `settings.py` |
| Unix worker, xử lý inference, RPC client | `grasppose/infrastructure/worker/` |
| Snapshot/output/render | `infrastructure/output/`, `presentation/` |
| CLI và Gradio | `apps/cli/`, `apps/gradio/` |
| Chuẩn bị/build model | `scripts/`, `env/`, `dependencies` |

Không import model stack trong CLI RPC. Không đọc internal graspgroup từ repo
robot. Model mới implement port; policy mới được inject vào estimator.

## Hai môi trường trên cùng server

Pipeline dùng Python 3.8 của JetPack trong `.venv`, host Torch/CUDA/TensorRT.
6DoF dùng Conda Python 3.10 riêng và trao đổi qua Unix socket.
Engine phát hành chỉ dành cho AGX Xavier L4T R35.6.4 / TensorRT 8.5.2.2,
CUDA 11.4, compute capability `sm_72`; không chuyển engine sang GPU khác.

6DoF đang khai báo pin `666c7eb608c5315ea252fd02b3f5446c39198ee6`.
Pin và SHA checkout worker thực tế là hai thông tin khác nhau; ghi cả hai khi
triển khai. Đợt refactor không tự đổi pin hoặc server đang chạy.

## Worker và inference

Với checkout có đầy đủ dependency/artifact tương ứng:

```bash
bash scripts/worker.sh start
bash scripts/worker.sh status
bash scripts/infer.sh image.png --prompt-id cube --camera-k 615.2 614.8 320.1 239.7
# K hiệu chuẩn ở resolution khác: thêm --camera-k-size WIDTH HEIGHT.
bash scripts/infer.sh image.png --prompt-id cube --fov-x 60 --render
bash scripts/output.sh RUN_ID
bash scripts/worker.sh stop
```

`--fov-x/--fov-y` phù hợp ảnh synthetic; robot thật cần K và extrinsic thật.
`CAMERA_K="FX FY CX CY"`, `CAMERA_K_SIZE="WIDTH HEIGHT"` cũng được hỗ trợ.
Socket mặc định: `<checkout>/.runtime/worker.sock`.
Ảnh, output, cache và venv không commit vào git.

YOLOE dùng engine FP32 có bộ prompt cố định, không gọi text encoder mỗi frame.
Prompt mới: tạo `prompts.json` có `prompts: [{"id":"cube","text":"cube"}]`,
đặt ảnh validation cho từng ID trong `model/validation/yoloe/`, cấu hình K rồi:

```bash
bash scripts/preprocess_prompt.sh prompts.json
```

Manifest/checksum phải hợp lệ và full-pipeline parity phải đạt ngưỡng.
`YOLOE_CONF` chỉ đổi ngưỡng hậu xử lý, không thay shape/checksum engine.

## Dùng trong Python

```python
from grasppose.api import get_estimator

estimator = get_estimator()
estimator.load()
estimator.warmup()
try:
    result = estimator.estimate(
        "image.png", "cube", camera_K=[615.2, 614.8, 320.1, 239.7],
        max_width=0.069, top=5,
    )
    for grasp in result.grasps:
        print(grasp.translation_m, grasp.rotation, grasp.width_m)
finally:
    estimator.close()
```

Translation/width là mét trong camera OpenCV. `T_cam_volume` là transform
volume→camera. VGN cần volume gravity-aligned; auto camera-aligned volume là
fallback, không thay hand-eye/table calibration của robot thật.

## Kiểm thử và thay đổi

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
python -m tests.test_pipeline_mock
python -m tests.test_app
```

CI chạy Python 3.8/3.12, không cần GPU. Full GPU inference, TensorRT parity và
depth metric phải kiểm chứng trên Jetson. Refactor và thay hành vi nằm trong
**hai PR khác nhau**; chia commit trong một diff chưa đủ.
