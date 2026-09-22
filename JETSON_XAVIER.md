# Jetson AGX Xavier / JetPack 5.1.4 port

Nhanh nay danh cho **Jetson AGX Xavier (tegra194, Volta sm_72)** voi:

- Ubuntu 20.04 / Python 3.8
- JetPack 5.1.4 / CUDA 11.4 / cuDNN 8.6
- NVIDIA PyTorch 2.1.0 + torchvision 0.16.1
- NumPy 1.23.5 / SciPy 1.10.1 / OpenCV JetPack 4.5.4

## Khac voi ban Kaggle

- **Khong nang cap Torch/CUDA/NumPy/OpenCV cua host.** `.venv` dung
  `--system-site-packages` va constraints de giu ABI JetPack.
- MoGe-3 ViT-L duoc thay bang **MoGe-2 ViT-S Normal** (metric scale, 35M params,
  model ~141 MB), pin source/model revision.
- Xavier chay MoGe va Grounding-DINO **tuan tu** thay vi hai CUDA workload song song.
- MoGe mac dinh `MOGE_RESOLUTION_LEVEL=4` va FP16.
- Bo Open3D khoi runtime; wireframe gripper duoc chieu bang NumPy + OpenCV.
- Gradio la tuy chon, chi cai khi dung `--serve`.
- MinkowskiEngine va pointnet2 build tai may cho `sm_72`.

## Chuan bi host mot lan

MinkowskiEngine can compiler, Python headers va OpenBLAS:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libopenblas-dev
```

Khong cai Torch bang pip. Phai giu ban NVIDIA dang co san trong JetPack.

## Chay

```bash
git switch port/jetson-agx-xavier-jp514

# Neu da tung chay nhanh Kaggle tren cung checkout:
rm -rf .venv

# Kiem tra host truoc:
python3 env/check_env.py   # MinkowskiEngine co the bao thieu truoc lan bootstrap dau

# Batch:
bash run.sh img/anh.png --prompt "the object"

# Web UI (cai them Gradio 4.x):
bash run.sh --serve --port 8080
```

Lan dau `run.sh` tai model/source da pin, build MinkowskiEngine va pointnet2.
Cac lan sau tai su dung model/extension neu ABI host khong doi.

## Tuning Xavier

`run.sh` tu nhan `aarch64 + tegra194` va dat:

```bash
TORCH_CUDA_ARCH_LIST=7.2
MAX_JOBS=2
PIPELINE_SERIAL_GPU=1
MOGE_RESOLUTION_LEVEL=4
```

Neu can nhanh hon, thu `MOGE_RESOLUTION_LEVEL=2` hoac `3`. Neu can chi tiet
depth cao hon va chap nhan cham hon, tang len `5`/ `6`.

## Trang thai kiem thu

CI kiem tra shell, logic pipeline/mock va Python 3.8 tren x86_64. Native CUDA
extensions va full inference van phai smoke-test tren **Jetson AGX Xavier that**,
vi GitHub-hosted CI khong co tegra194/CUDA 11.4.
