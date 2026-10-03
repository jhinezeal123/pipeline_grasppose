# Đọc code pipeline từ đâu?

Luồng chính: `GraspEstimator.estimate()` → chuẩn hóa input →
`GraspPipeline.run()` → chọn grasp → `EstimateResult`.

| Việc cần hiểu/sửa | File sở hữu trách nhiệm |
| --- | --- |
| API dùng từ 6DoF hoặc ứng dụng khác | `grasppose/api.py`, `application/interface.py`, `application/types.py` |
| Đọc ảnh, kiểm tra request, scale K/FOV | `application/input.py` — `EstimateInput` |
| Trình tự vision → depth → TSDF → grasp | `application/grasp_pipeline.py` — `GraspPipeline` |
| Lifecycle và kết quả public | `application/service.py` — `LocalGraspEstimator` |
| Lọc width, sắp score, chọn top | `application/selection.py` — `GraspSelector` |
| Chọn implementation thật | `infrastructure/composition.py` |
| Vision/depth/TSDF/grasp và port tương ứng | `modules/vision`, `modules/depth`, `modules/tsdf`, `modules/grasp` |
| Nhận một request, gọi estimator, giữ snapshot, tạo JSON | `infrastructure/worker/inference.py` — `WorkerInference` |
| Unix socket, framing, status/stop/snapshot, lifecycle process | `infrastructure/worker/server.py` |
| RPC nhẹ cho CLI | `infrastructure/worker/rpc.py` |
| Client typed cho Python | `infrastructure/worker/client.py` |
| Render ảnh chẩn đoán ở process riêng | `infrastructure/output`, `presentation/rendering.py` |

## Hợp đồng không được đoán

- Ảnh RGB `(H,W,3+)`; service giữ 3 kênh đầu. K tính theo pixel ảnh thực.
- Depth là mét theo camera-Z. Cloud và translation của grasp ở khung camera
  OpenCV: X phải, Y xuống, Z ra trước.
- `T_cam_volume` biến tọa độ volume thành tọa độ camera, không phải chiều ngược.
- Grasp public: score, width mét, translation mét, ma trận rotation 3×3.
- Internal graspgroup `(N,17)` float64 chỉ dùng trong pipeline; 6DoF không đọc nó.
- `EstimateResult.grasp_count` là số grasp trước lọc width/top.
- Socket `infer` trả `grasps`, `depth_m`, counts, `run_id`, snapshot flag và
  `server_ms`. Tên field, thứ tự pose, đơn vị và lỗi không đổi trong refactor.
- Model load/warmup một lần, resident qua nhiều frame; frame không lưu vào model.

## Mở rộng ít chạm code

Model mới implement port tương ứng rồi được chọn tại composition root. Không
import framework model vào application. `build_default_pipeline(depth=...)`
cho phép harness truyền depth thật từ simulator mà không monkeypatch adapter.

Policy chọn grasp mới được truyền qua
`LocalGraspEstimator(pipeline, selector=policy)`. Contract của `policy.select`
nhận mảng float64 `(N,17)`, width mét và top; trả tuple `GraspPose`.
Policy mặc định giữ nguyên toán học/sort của service cũ. Không cần thêm registry,
plugin framework hoặc lớp kế thừa chỉ để đổi vài dòng.

Input, policy, orchestration, transport và render có chủ sở hữu riêng. Dùng
composition và port sẵn có; không gom mọi thứ vào một `utils.py`.

## Quy tắc hai bước

PR refactor chỉ di chuyển/tách code và khóa output bằng test. Việc đổi model,
ngưỡng, scale, tọa độ hay wire protocol phải nằm trong PR sau, dựa trên refactor.
Một PR chứa cả hai vẫn là thay đổi trộn, kể cả có chia commit.
