# env/ — phu thuoc moi truong

Thu muc nay tra loi cau hoi: **"may nay co chay duoc pipeline khong?"**

## Vi sao khong chua thu vien o day

Cach dong goi chuyen nghiep la **repo chua cong thuc, khong chua ket qua**. Copy
thu vien vao repo (vendoring) hau nhu khong ai lam, vi:

- torch + CUDA ~2.5 GB, MinkowskiEngine + phan con lai ~0.5 GB;
- `MinkowskiEngine` va `pointnet2._ext` la **C++/CUDA extension**, bien dich cho
  dung ABI cua mot ban torch + mot ban CUDA + mot ban Python. Copy sang may khac
  **khong chay duoc** — no khong mang lai tinh dong goi, chi mang lai dung luong;
- `git diff` tro nen vo dung, va moi lan nang cap phai copy lai tu dau.

Thu can tai lap khong phai *thu vien*, ma la **to hop CUDA + torch +
MinkowskiEngine + _ext**. To hop do duoc ghim trong `requirements.lock.txt`.

## Noi dung

| File | Vai tro |
|---|---|
| `check_env.py` | Kiem tra may nay co du dieu kien khong. Chay TRUOC `run.sh`. |
| `../requirements.lock.txt` | Ban ghim chinh xac moi thu da chay duoc. |

## Dung

```bash
python env/check_env.py
```

Tra ve 0 neu du dieu kien, 1 neu thieu. Kiem tra: Python, thu vien (kem doi chieu
phien ban da kiem chung), GPU + compute capability so voi arch torch ho tro, `nvcc`
de bien dich `_ext`, va `_ext` da build chua.

## To hop da kiem chung that

Do bang `importlib` tren may Kaggle dang chay, khong phai suy doan:

| | |
|---|---|
| Python | 3.12.13 |
| torch | 2.10.0+cu128 |
| CUDA | 12.8 (`nvcc` V12.8.93) |
| transformers | 5.0.0 |
| numpy | 2.0.2 |
| GPU | Tesla T4 (sm_75) |
| MinkowskiEngine | 0.5.4 |
| gcc | 11.4.0 |

## Ba thu KHONG cai duoc bang pip

1. **MinkowskiEngine** — khong co wheel tren PyPI cho cau hinh nay. Kaggle lay tu
   dataset ngoai; may khac phai tu build (rat kho, xem repo NVIDIA/MinkowskiEngine).
2. **pointnet2 `_ext`** — upstream chi co ma nguon. `run.sh` tu bien dich bang
   `setup.py`, build **vao trong repo** (`model/graspness_unofficial_build/`).
3. **graspnetAPI** — ban tren PyPI bi hong (con import `setuptools.extern.six`, da
   bi Python moi xoa). `run.sh` `git clone` vao `model/graspnetAPI_repo`.

## Vi sao `venv` khong dung duoc o day

Da thu that tren Kaggle va **that bai**: `python3 -m venv` bao
`ModuleNotFoundError: No module named 'ensurepip'`, va venv tao ra con **mat luon
`torch`** cua he thong. Nen khong the co lap bang venv theo cach thong thuong.
