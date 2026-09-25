# Môi trường Jetson

`prepare.sh` giữ nguyên CUDA stack của JetPack. `.venv` dùng `--system-site-packages`; `env/setup_env.py` ghi version host vào `.venv/host-constraints.txt` để pip không tự thay Torch/CUDA packages.

Yêu cầu: Jetson Xavier + JetPack/CUDA/TensorRT, Python >=3.8, host `torch`, `torchvision`, `numpy` hoạt động và `torch.cuda.is_available()` là `True`.

Không còn bước build MinkowskiEngine hoặc pointnet2. VGN chạy qua TensorRT engine đã đóng gói cho AGX Xavier/L4T R35.6.4/TensorRT 8.5.2.2 và được tải vào checkout hiện tại. Nếu JetPack hoặc TensorRT thay đổi, cần build và phát hành engine mới cho thiết bị đó.
