# grasp_pipeline_repo

Pipeline: **text prompt -> grasp pose**, chay tren may Linux GPU (Kaggle).
Tu mot cau prompt (vi du "cai coc") + mot anh RGB -> vi tri dat tay de gap vat the:

1. **Grounding DINO** (tiny) - prompt -> **bounding box** cua vat the.
2. **SAM** (vit-base) - box -> **mask** chinh xac cua vat the.
3. **MoGe v3** (ViT-L) - anh -> **depth map** + point cloud + intrinsics (3D).
4. **GraspNet / Graspness** - point cloud + mask -> **grasp pose** (6-DoF).

Co **2 che do chay**:

- **Batch** (mac dinh): `bash run.sh img/anh.png` -> 4 anh trong `output/`. Muc 3.
- **Web UI** kieu HuggingFace Space: `bash run.sh --serve` -> mo trinh duyet, tai
  anh len, bam Submit -> 4 anh + **do sau cua vat (met)**. Muc 7.

## Cau truc thu muc

```
grasp_pipeline_repo/
  run.sh              # chay tat ca: tai model -> cai thu vien -> chay pipeline
  dependencies        # danh sach model can tai (nguon duy nhat, run.sh parse file nay)
  requirements.txt    # thu vien Python, dung dinh dang HF Spaces (xem muc 7)
  hf_token            # token HuggingFace, de TRONG cung duoc
  pipeline.py         # CHAY CHINH: dinh nghia 4 lop cu the + noi pipeline + pipeline(img)
  app.py              # WEB UI (gradio): anh -> 4 anh + do sau (muc 7)
  Object_Detection.py # \  KHAI BAO giao dien 4 module (lop truu tuong).
  Segmentation.py     #  | Chi co chu ky prepare()/inference()/release() va
  Depth_Estimate.py   #  | thuoc tinh model_path. Khong dinh nghia gi.
  GraspNess.py        # /  Doi model = thay lop cu the trong pipeline.py.
  test_pipeline_mock.py # kiem thu khong can GPU (model gia)
  test_app.py           # kiem thu web UI khong can gradio (muc 6)
  img/                # anh dau vao (.png/.jpg/.jpeg)
  output/             # anh ket qua
  model/              # trong so model duoc tai ve day
```

### Thao/lap model khac

4 file module la **hop dong**: chung chi khai bao lop truu tuong
(`model_path`, `prepare()`, `inference()`, `release()`). Toan bo phan dinh nghia
cu the nam trong `pipeline.py`. Muon doi model (vi du thay Grounding DINO bang
OWL-ViT) thi chi viet mot lop con moi cua `Object_Detection` trong `pipeline.py`
va doi cho khoi tao — khong phai sua file module nao.

## 1. Dien hf_token

Mo file `hf_token`, dan token HuggingFace vao (1 dong, khong xuong dong thua).
De trong cung chay duoc, chi la tai cham hon. Token chi duoc export thanh bien moi
truong `HF_TOKEN`, khong bao gio bi in ra.

## 2. Dien URL checkpoint graspness

Checkpoint `graspness_realsense.pth` (epoch=10) cua GraspNet **khong phai cong khai**.
Mo file `dependencies`, tim block `NAME = graspness`, thay dong `URL = PASTE_URL_HERE`
bang link tai cua ban (Kaggle dataset / Google Drive direct link).
De nguyen placeholder thi `run.sh` dung lai va bao loi ro rang.
Cach khac: tu copy file vao `model/graspness_reckpt.pth` truoc khi chay.

## 3. Chay

```bash
bash run.sh                                  # tu lay anh dau tien trong img/
bash run.sh img/anh-cua-ban.png              # chi dinh anh
bash run.sh img/anh.png --prompt "cai coc"   # flag them duoc chuyen tiep
bash run.sh --serve                          # WEB UI thay vi chay 1 anh (muc 7)
```

`run.sh` chay lai nhieu lan khong tai lai model (idempotent). `--serve` va `--port`
duoc `run.sh` tach ra truoc, phan tham so con lai moi chuyen cho `pipeline.py`.

## 4. Ket qua trong output/

Ten file la `<ten-anh>_<loai>.png`, gom 4 loai:
- **box** - anh goc ve bounding box tim duoc tu prompt.
- **mask** - anh goc + mask cua vat the (SAM cat theo box).
- **depthmap** - ban do do sau tu MoGe.
- **grasp** - anh goc + grasp pose (vi tri + huong dat tay).

Anh mau trong `example/` (prompt "a little bag", anh 1280x960):

| | |
|---|---|
| `bag_input.png` | anh goc |
| `bag_box.png` | DINO: 1 hop `bag` score 0.265 |
| `bag_mask.png` | SAM: 136358 px (11.1%) IoU 0.988 |
| `bag_depthmap.png` | MoGe: fov_x 74.42 do, depth 0.403..1.150 m |
| `bag_grasp.png` | 191 tu the, 114 vua khe kep that 69 mm |

Tong thoi gian chay: **~51 s** tren Tesla T4 (MoGe 11 s, DINO 5 s, SAM CPU ~9 s,
GraspNess phan con lai).

### `depth_m` - con so do sau cua vat

`pipeline()` tra ve them khoa `"depth_m"`: **trung vi do sau MoGe tren cac pixel
nam trong mask SAM**, don vi met. `None` khi mask rong hoac khong co pixel hop le.

- Trung vi chu khong phai trung binh: pixel nhieu o ria mask se keo lech trung binh.
- La do sau cua **VAT**, khong phai cua ca anh. Tren anh mau: ca anh 0.403..1.150 m
  nhung trong mask chi 0.444..0.811 m.
- CLI in ra dong `DO SAU VAT: ... m` o cuoi.

### Vi sao gripper trong anh trong nhu mot thanh mong

Khong phai loi ve. `plot_gripper_pro_max` cua upstream (graspnetAPI) dung ngon
day `height = 0.004` (4 mm) va `finger_width = 0.004`. Khi truc tiep can nam gan
mat phang anh, ta nhin nghieng tam 4 mm -> no ra mot vet mong.

`o3d.visualization.draw_geometries` (duong ve goc cua upstream) khong chay duoc
o day: `OffscreenRenderer` bao `Failed to load vulkan library`. Nen mesh duoc
chieu va to tam giac bang tay, co them do bong Lambert cho ra khoi 3D.
**Hinh hoc va mau (R=score, G=0, B=1-score) van lay nguyen tu upstream.**

## 5. Chien luoc VRAM 3 pha

Khong bao gio giu 2 model nang cung luc tren VRAM. Model nao chay xong thi bi
day ra ngay.

- **Pha 1**: nap **MoGe + Grounding DINO cung luc**, cho chung chay **song song**
  (2 thread). Cai nao xong truoc thi `release()` truoc. Day la cap duy nhat chay
  dong thoi, vi chung doc cung mot anh dau vao va khong phu thuoc nhau.
- **Pha 2**: DINO da nha VRAM -> nap **SAM** -> cat mask theo box -> nha.
  (SAM phai cho DINO vi no can box; trong khi do MoGe co the van dang chay.)
- **Pha 3**: tat ca da nha -> nap **GraspNet/Graspness** -> chay -> nha.
  (Can ca mask lan point cloud, nen phai cho ca 2 pha tren xong.)

Thu tu nay duoc `test_pipeline_mock.py` kiem chung bang day goi thuc te, khong
phai bang doc code.

## 6. Kiem thu khong can GPU

```bash
python test_pipeline_mock.py
python test_app.py
```

`test_pipeline_mock.py` thay 4 model bang lop gia roi chay `pipeline()` that. Kiem
tra: cong thuc intrinsics, `depth_to_cloud` loc theo mask, NMS, thu tu 3 pha,
`depth_m` dung dinh nghia (trung vi trong mask, khac trung vi ca anh), va 4 anh
dau ra. Chay duoc tren may khong co GPU (chi can numpy + opencv + open3d).

`test_app.py` kiem tra phan loi cua web UI ma **khong can cai gradio**: `app.py`
co y khong import gradio o cap module, nen thay `pipeline.pipeline()` bang ham gia
la test duoc. Kiem tra: tra ve dung thu tu, prompt rong -> dung mac dinh, pipeline
nem loi -> hien chu LOI chu khong crash, va `run_one()` **khong bao gio raise**
ke ca voi dau vao ki quac.

## 7. Web UI (kieu HuggingFace Space)

```bash
bash run.sh --serve                # cong 8080
bash run.sh --serve --port 7860    # doi cong
```

Mo `http://<may-chu>:<cong>/`. Giao dien gom:

- o tai **anh** len, o **prompt** (dien san `the object`), nut **Submit**;
- 4 anh ket qua: box -> mask -> depthmap -> grasp pose;
- o **Do sau vat (m)** va dong trang thai.

Thiet ke dang chu y:

- **Moi lan Submit nap lai ca 4 model, mat khoang 50 giay.** Do la he qua truc
  tiep cua chien luoc VRAM 3 pha (muc 5): model dung xong bi day ra ngay, nen lan
  sau phai nap lai. Doi lay viec chay duoc tren GPU 16 GB.
- **Chay song song bi gioi han ve 1** (mac dinh cua Gradio, khong phai cho dep): 2
  lan chay cung luc se OOM T4 16 GB, vi thiet ke 3 pha chi giai phong VRAM khi chay
  tuan tu. `app.py` co y **KHONG** viet `queue(concurrency_limit=...)`: tham so do
  khong ton tai (da bi loi that tren Kaggle: `Blocks.queue() got an unexpected
  keyword argument`), va mac dinh cua Gradio da la 1.
- Prompt rong -> dung `the object`. DINO khong tim thay vat -> **van tra 4 anh**
  kem ly do ghi truc tiep tren anh, o do sau de trong, app khong crash.
- `app.py` tach phan loi (`run_one`) khoi gradio de test duoc (muc 6). Import
  gradio o dau file se khien `import app` that bai tren may khong co gradio.

### Ve `requirements.txt`

File nay viet theo dung dinh dang HF Spaces doc duoc, nhung **khong du de dung mot
HF Space that**: hai thu bat buoc khong the cai bang pip —

1. **graspnetAPI** — goi tren PyPI bi hong (con import `setuptools.extern.six` da
   bi Python moi xoa). Phai `git clone` roi them vao `PYTHONPATH`; `pipeline.py`
   lam viec do qua `_load_graspnetapi()`. `run.sh` clone vao `model/graspnetAPI_repo`.
2. **Checkpoint graspness** — khong phai cong khai, khong co link chinh thuc (muc 2).

Vi vay **Kaggle la duong chay chinh thuc**. `requirements.txt` chi de repo hop
chuan va de cai nhanh phan UI.
