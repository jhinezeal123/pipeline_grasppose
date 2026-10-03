# Review PR — pipeline_grasppose

Rà soát ngày 03/10/2026, tại `main@c417fd0ef8abac754636cb4f442c369dc6a2b91f`.
Đây là kết luận từ mã nguồn, diff, lịch sử merge và CI; chưa đọc checkout thực tế
trên server, chưa chạy GPU Jetson hay robot thật. Kết quả đo ghi trong PR là bằng
chứng do tác giả cung cấp, không phải một lần đo mới của đợt review này.

## Quyết định từng PR

| PR | Trạng thái | Quyết định | Lý do |
| --- | --- | --- | --- |
| [#1](https://github.com/jhinezeal123/pipeline_grasppose/pull/1) | Đã merge | Giữ | Cô lập venv, constraints và nguồn model. Runtime Kaggle này đã được thay bởi #5; revert sẽ kéo lại bootstrap cũ. |
| [#2](https://github.com/jhinezeal123/pipeline_grasppose/pull/2) | Đã merge | Giữ trong lịch sử | Tách bootstrap/download/install và xử lý Minkowski. Không còn là đường chạy Jetson hiện tại. |
| [#3](https://github.com/jhinezeal123/pipeline_grasppose/pull/3) | Đã merge | Giữ trong lịch sử | Wheel là giải pháp cho môi trường cũ; #5 đã bỏ Minkowski và wheel. Không đưa wheel cp312/x86 trở lại Jetson. |
| [#4](https://github.com/jhinezeal123/pipeline_grasppose/pull/4) | Đã merge | Giữ | Sửa lỗi depth/cloud rỗng, render, cleanup và coverage. Không có lý do revert các guard. |
| [#5](https://github.com/jhinezeal123/pipeline_grasppose/pull/5) | Đã merge | Giữ nền kiến trúc | Ports và model resident hữu ích. PR đồng thời thay cả model/runtime: vi phạm quy tắc refactor riêng, change riêng; không dùng làm mẫu cho PR mới. |
| [#6](https://github.com/jhinezeal123/pipeline_grasppose/pull/6) | Đóng, chưa merge | Bỏ nhánh này | Nhánh port trên stack Kaggle cũ, trước #5. Revert không áp dụng vì chưa merge. |
| [#7](https://github.com/jhinezeal123/pipeline_grasppose/pull/7) | Đóng, chưa merge | Bỏ nhánh này | Chồng lấn worker/TensorRT sau đó được triển khai trong #8. Không merge lại toàn nhánh để tránh đưa kiến trúc cũ trở lại. |
| [#8](https://github.com/jhinezeal123/pipeline_grasppose/pull/8) | Đã merge | Giữ | Worker resident, checksum/manifest, fixed prompt và warmup là nền giao tiếp với 6DoF. |
| [#9](https://github.com/jhinezeal123/pipeline_grasppose/pull/9) | Đã merge | Giữ | Render sang process riêng, cache snapshot có giới hạn, scale K theo resolution; vẫn cần benchmark GPU khi đổi triển khai. |
| [#10](https://github.com/jhinezeal123/pipeline_grasppose/pull/10) | Đã merge | Giữ API/ports; sửa cấu trúc tiếp | Feature modules hữu ích nhưng PR cũng đổi entry point/API và làm CLI import model stack; #11 sửa phần khởi động. Revert cả PR sẽ phá consumer. |
| [#11](https://github.com/jhinezeal123/pipeline_grasppose/pull/11) | Đã merge | Giữ | RPC dùng standard library, CLI không import NumPy/Torch/model. Có regression test khởi động nhẹ. Đây là pin đang được 6DoF khai báo. |
| [#12](https://github.com/jhinezeal123/pipeline_grasppose/pull/12) | Đã merge | Giữ | Confidence là hậu xử lý; engine/checksum vẫn bị ràng buộc. Hướng dẫn gravity-aligned volume cần giữ. |
| [#13](https://github.com/jhinezeal123/pipeline_grasppose/pull/13) | Đã merge | Giữ DA3 mặc định; hoàn thiện deployment riêng | Trên cùng simulation gate, tác giả ghi DA3 FP32/CUDA 10/10, Lite-Mono 3/10 và Pearson depth −0.43. Bootstrap thiếu DA3 là bug deployment, không đủ căn cứ chọn lại Lite-Mono. Scale `0.39378` vẫn cần đo trên camera thật. Giữ depth injection seam. |
| [#14](https://github.com/jhinezeal123/pipeline_grasppose/pull/14) | Đang mở | Chưa merge; nên bỏ cách refine hiện tại | Dịch grasp tới TSDF=0.5 nhưng giữ orientation, width và score của voxel cũ. Chưa đánh giá lại collision/grasp quality hay lift success. Các test plane/sphere chỉ chứng minh phép chiếu, không chứng minh grasp tốt hơn. |

## Những lỗi cần hành động

**P1 — #13 làm đường cài mới không đầy đủ.**
`infrastructure/composition.py` tạo `Da3MetricDepth()`; `scripts/prepare.sh`,
`dependencies` và `requirements.txt` không tải graph DA3/cài ONNX Runtime.
`env/check_env.py` lại bắt buộc graph đó. Checkout sạch làm theo README sẽ không
đủ tài nguyên để chạy. Sửa riêng bootstrap: tải graph FP32 đã pin/SHA-256,
cài ONNX Runtime GPU đúng Jetson và kiểm tra provider/inference thực. Giữ DA3
mặc định theo số đo #13; Lite-Mono chỉ là lựa chọn thử nghiệm explicit.
Khuyến nghị rollback ban đầu đã được sửa sau khi đối chiếu lại gate 10/10 so
với 3/10. Các số đo đó là kết quả simulation của tác giả, không phải bảo đảm
thành công trên camera/robot thật.

**P1 — #14 thay pose mà score không được đánh giá lại.**
Giới hạn 4 voxel có thể dịch tới 30 mm trên grid hiện tại, không chỉ sửa một
sai số lượng tử dưới 1 voxel. Với plane ngay trong test, một grasp dịch hơn
2 voxel mà score vẫn giữ nguyên. VGN gắn quality/orientation/width với vị trí
gripper tại voxel; pose origin không mặc nhiên là điểm tiếp xúc trên mặt vật.
Không thể lấy sai số tới surface giảm làm bằng chứng grasp thành công.
Nguồn: [VGN, mục 3–4 và hình 2c](https://proceedings.mlr.press/v155/breyer21a/breyer21a.pdf),
[mã nguồn gốc](https://github.com/ethz-asl/vgn).
Ngoài ra, helper chỉ kiểm tra bước dịch và gradient; không kiểm tra zero crossing
được bracket hay kiểm tra lại field tại điểm mới như mô tả trong PR.

**P2 — provenance giữa hai repo đang dễ gây hiểu nhầm.**
6DoF khai báo pin `666c7eb`, còn `main` ở đây là `c417fd0`. Protocol vẫn tương
thích, nhưng pin trong metadata không chứng minh worker đang chạy đúng SHA.
Cần ghi SHA checkout worker thực tế trong lần triển khai; thay pin/protocol là
PR chức năng riêng. Đợt refactor này không sửa pin, model hoặc calibration.

## Kiểm chứng refactor

- CI gốc của `c417fd0`: [success](https://github.com/jhinezeal123/pipeline_grasppose/actions/runs/36998911646).
- Test nền: 57 passed, 2 lỗi do host chặn tạo Unix socket.
- 6 characterization tests được chạy xanh trước khi tách code.
- Sau refactor: 63 passed; chỉ deselect đúng 2 ca socket bị chặn trên host này.
- Không thêm skip/xfail vào repo; GitHub CI vẫn chạy đầy đủ các test socket.
- Model/inference GPU, parity TensorRT, depth thực và thời gian chạy trên Jetson
  chưa được kiểm chứng lại. Unit test không thay thế các kiểm chứng đó.

## Trình tự áp dụng

1. Merge PR refactor sau khi full CI xanh: input/selection/service và
   inference/socket được tách, hành vi mặc định giữ nguyên.
2. Review PR hoàn thiện bootstrap DA3 riêng, trên nền refactor; DA3 vẫn mặc định.
3. Giữ 6DoF ở pin hiện tại cho tới khi chạy integration trên server.
4. Với DA3/refinement: replay cùng ảnh, K, extrinsic và robot calibration;
   so sánh depth, pose, collision và lift success trước khi chọn cho robot thật.
