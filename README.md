# grasp_pipeline_repo

> **Jetson AGX Xavier / JetPack 5.1.4:** xem [JETSON_XAVIER.md](JETSON_XAVIER.md). Nhanh nay giu Torch/CUDA cua JetPack, dung MoGe-2 ViT-S va khong can Open3D.

Pipeline: **text prompt -> grasp pose**. Ban goc ho tro Kaggle; nhanh nay bo sung profile Jetson AGX Xavier.
Tu mot cau prompt (vi du "cai coc") + mot anh RGB -> vi tri dat tay de gap vat the:

1. **Grounding DINO** (tiny) - prompt -> **bounding box** cua vat the.
2. **SAM** (vit-base) - box -> **mask** chinh xac cua vat the.
3. **MoGe-2** (ViT-S Normal tren nhanh Jetson) - anh -> **depth map** + point cloud + intrinsics (3D).
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
  requirements.lock.txt # snapshot lịch sử, không dùng để bootstrap
  hf_token            # token HuggingFace, de TRONG cung duoc
  pipeline.py         # CHAY CHINH: dinh nghia 4 lop cu the + noi pipeline + pipeline(img)
  app.py              # WEB UI (gradio): anh -> 4 anh + do sau (muc 7)
  Object_Detection.py # \  KHAI BAO giao dien 4 module (lop truu tuong).
  Segmentation.py     #  | Chi co chu ky prepare()/inference()/release() va
  Depth_Estimate.py   #  | thuoc tinh model_path. Khong dinh nghia gi.
  GraspNess.py        # /  Doi model = thay lop cu the trong pipeline.py.
  test_pipeline_mock.py # kiem thu khong can GPU (model gia)
  test_app.py           # kiem thu web UI khong can gradio (muc 6)
  env/                # phu thuoc moi truong: check_env.py + ghi chu (muc 8)
  .venv/            # thu vien run.sh cai rieng cho repo (sinh ra, KHONG commit)
  example/            # anh mau de thu ngay
  img/                # anh dau vao (.png/.jpg/.jpeg)
  output/             # anh ket qua
  model/              # trong so model duoc tai ve day
```

`.venv/` va `model/` la **sinh ra**, khong nam trong git — `run.sh` tu tao lai.

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

## 2. Checkpoint graspness — `run.sh` tu tai, khong can lam gi

Block `NAME = graspness` trong `dependencies` da tro san vao mot dataset Kaggle
cong khai:

```
URL = https://www.kaggle.com/api/v1/datasets/download/bbucxi/graspness-realsense-ckpt
MD5 = f2c14a02cf024de789f324ba2da76277
```

`run.sh` tai ve, tu giai nen (Kaggle tra ve **zip** chua `.pth`), roi doi chieu
md5. Lech md5 thi xoa file va dung ngay — tai hong (nhan phai trang HTML, file
bi cat ngan) bi bat o day thay vi chet mo ho o buoc GraspNess sau nay.

Da do thuc te: tai xong trong **4.1 giay**, md5 khop, `epoch=10`, 216 tham so.

### Nguon goc

Checkpoint `graspness_realsense.pth` (epoch=10) cua GraspNet. Link chinh thuc
nam o muc "Model Weights" trong README cua
[graspnet/graspness_unofficial](https://github.com/graspnet/graspness_unofficial),
duoi dang Google Drive, file goc ten `minkuresunet_realsense.tar`.

**Khong dung duoc link Google Drive lam URL mac dinh**: `.../file/d/<id>/view`
chi la trang xem chu khong phai link tai, file >100 MB bi chan them buoc
"Virus scan warning", va rat hay gap "Quota exceeded". Dataset Kaggle o tren la
ban sao cua dung file do — da doi chieu noi dung, khong phai suy doan:

| | |
|---|---|
| `epoch` | 10 — khop `--checkpoint_path logs/log_kn/minkresunet_epoch10.tar` trong `command_test.sh` cua upstream |
| So tham so | 216, gom 4 nhanh `graspable` / `rotation` / `crop` / `swad` |
| Bon ten nhanh do | **chi** xuat hien trong `models/graspnet.py` cua `graspness_unofficial` |
| md5 | `f2c14a02cf024de789f324ba2da76277`, 184429769 byte |

Muon doi sang nguon khac: sua `URL` (va `MD5` neu co) trong `dependencies`.
De `URL` trong thi `run.sh` dung lai va in huong dan. Cung co the copy tay file
vao `model/graspness_reckpt.pth` truoc khi chay.

## 2b. `pointnet2._ext` — `run.sh` tu bien dich, khong can lam tay

`graspness_unofficial/pointnet2/pointnet2_utils.py` co dong:

```python
import pointnet2._ext as _ext
```

`pointnet2/_ext` la mot **extension CUDA**, va trong repo upstream **chi co ma
nguon** (`_ext_src/`), khong co ban dung san o bat ky dau. Thieu no thi
`from models.graspnet import GraspNet` nem:

```
ImportError: Could not import _ext module.
```

va **GraspNess khong nap duoc** — hau qua la khong ra tu the nao ca. Trieu chung
nay rat de chan doan nham thanh "loc qua chat" hoac "mask rong", nen `run.sh`
(BUOC 5d) tu lo:

- thu ghi that vao `pointnet2/`; ghi duoc thi build tai cho, khong thi **copy
  sang `model/graspness_unofficial_build/`** roi tro `GRASPNESS_HOME` vao do
  (tren Kaggle `model/` la symlink vao `/kaggle/input` chi doc);
  phep thu la mot lan `touch` that, **khong** dung `[ -w ]`: chay bang root thi
  `[ -w ]` tra ve dung ca tren mount chi doc;
- copy bang **`cp -rL`** chu khong phai `cp -r`. `model/graspness_unofficial`
  thuong **la mot symlink** vao `/kaggle/input`, ma `cp -r` mac dinh **copy chinh
  symlink do**, nen "ban sao" van tro vao cho chi doc va build chet bang
  `error: could not create '...': Read-only file system`;
- them `/usr/local/cuda/bin` vao `PATH` (nvcc co san nhung khong nam trong PATH);
- dat `TORCH_CUDA_ARCH_LIST` theo GPU that, mac dinh `7.5`. Khong dat thi torch
  tu do arch, va tren may khong thay GPU se ra danh sach rong roi build chet voi
  `IndexError` o `_get_cuda_arch_flags`;
- `setup.py build_ext --inplace` hay hong o buoc **copy cuoi cung** (no giai ma
  ten goi thanh `pointnet2/` tuong doi voi CWD, ma CWD da la `pointnet2/`, nen doi
  thu muc `pointnet2/pointnet2/` khong ton tai) — nhung file `.so` **da duoc sinh
  ra**, nen script chep thang no vao cho ma `import pointnet2._ext` tim.

### `set -euo pipefail` o dau `run.sh` va moi lenh co the that bai

`run.sh` chay duoi `set -euo pipefail`. Nghia la **mot lenh that bai bat ky se
giet ca script ngay lap tuc** — va vi `run.sh` la thu duy nhat ghi ra man hinh,
nguoi dung chi thay kernel Kaggle bao `ERROR` ma **khong co log nao**.

Ba cho trong BUOC 5d tung bi dung loi nay:

```bash
EXT_SO="$(find ... | head -1)"          # find loi, hoac head dong ong som
                                        # -> pipefail tra ve khac 0
TORCH_CUDA_ARCH_LIST=$(python3 -c ...)  # python3 loi -> khac 0, va gia tri
                                        # mac dinh 7.5 khong bao gio duoc dat
cp -r ... / chmod -R ...                # loi -> khac 0
```

Nay ca ba deu duoc chan (`|| true` cho phep gan, `if` cho `cp`). Khi sua BUOC 5d,
**phai chay `vla_test/_test_runsh_build.sh`** — no trich dung khoi nay tu `run.sh`
va chay duoi `set -euo pipefail` y nhu that.

Mat khoang 2-4 phut, chi chay mot lan. Build that bai **khong** lam chet script:
GraspNess se bao ro ly do ngay tren anh ket qua (xem `grasp_empty_msg`).

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

### Cach ve gripper trong anh

Mesh gripper cua upstream gom **4 hop** — ngon trai, ngon phai, thanh noi phia
truoc, va duoi — nhin thang ra **hinh chu U**.

`GraspGroup.to_open3d_geometry_list()` tra ve **`LineSet`**, khong phai
`TriangleMesh`. Ban dau code doc `np.asarray(mesh.triangles)`; `LineSet` khong co
thuoc tinh do nen numpy tra ve **mang rong**, vong lap ve canh khong chay dong
nao, va thu duy nhat hien len la cac vach do `fillPoly` sinh ra — anh trong nhu
"may cai que". Nay doc `geom.lines` va ve bang `cv2.line`, du **12 canh moi hop**
dung nhu upstream.

`o3d.visualization.draw_geometries` (duong ve goc cua upstream) khong chay duoc
o day: `OffscreenRenderer` bao `Failed to load vulkan library`. Nen mesh duoc
chieu xuong anh roi ve tung doan thang.

Da thu loc bot canh cho do roi (chi giu duong cheo mat) nhung bo di: o goc nhin
thay doi, khong phan biet duoc "canh song song truc" voi "duong cheo mat", nen
cach loc do lam mat net that cua hinh. Trung thanh voi upstream quan trong hon.

Hinh hoc va mau (R=score, G=0, B=1-score) lay nguyen tu upstream.

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
- **`share=False`** trong `launch()` la co y: chay trong notebook thi gradio tu
  bat `share=True` va mo mot duong cong khai ra Internet toi may dang chay GPU,
  khong xac thuc gi. Ta da co duong ham rieng nen khong can.

### So do that, do tren Kaggle T4 (khong phai suy doan)

Anh `example/bag_input.png`, prompt `"a little bag"`, qua dung duong web UI
(`/gradio_api/call/run_one` tren cong 8080):

| Buoc | Ket qua |
|---|---|
| Khoi dong (da gom build `_ext`) | 234-306 s, trong do build ~140 s |
| MoGe | `fov_x=74.42 do`, depth toan anh 0.403..1.150 m |
| Grounding-DINO | 1 hop `bag`, score 0.265 |
| SAM | mask 136358 px (11.1%) |
| Cloud tu mask | bbox 461 x 218 x 367 mm |
| GraspNess | 196 tu the, 109 vua khe kep 69 mm |
| **`depth_m`** | **0.482 m**, lap lai y nguyen qua 3 lan chay doc lap |
| Anh grasp | 55999 byte — rut ra 5 tu the, rong 42.2 / 54.2 / 65.2 / 68.3 / 77.7 mm |
| Ca am (prompt `"xyzzynotathing"`) | `depth_m = None`, 4 anh van tra ve, ghi ro ly do |

Lam lai: `vla_test/_remote_e2e.py` goi Gradio API **tu trong may Kaggle**
(127.0.0.1:8080) nen khong phu thuoc tunnel; `vla_test/_fetch_space_out.py` chay
no qua SSH roi keo 4 anh ve.

### Ve `requirements.txt`

File nay viet theo dung dinh dang HF Spaces doc duoc, nhung **khong du de dung mot
HF Space that**: hai thu bat buoc khong the cai bang pip —

1. **graspnetAPI** — goi tren PyPI bi hong (con import `setuptools.extern.six` da
   bi Python moi xoa). Phai `git clone` roi them vao `PYTHONPATH`; `pipeline.py`
   lam viec do qua `_load_graspnetapi()`. `run.sh` clone vao `model/graspnetAPI_repo`.
2. **pointnet2 `_ext`** — extension CUDA, upstream chi co ma nguon, phai bien dich
   bang `setup.py` (muc 2b). Tren Kaggle `run.sh` lo viec nay; tren HF Spaces thi
   khong co GPU san va khong chay duoc `nvcc` theo cach tuong tu.

Vi vay **Kaggle la duong chay chinh thuc**. `requirements.txt` chi de repo hop
chuan va de cai nhanh phan UI.

## 8. Phu thuoc moi truong (CUDA, GPU, va cach ly)

`run.sh` tạo `.venv` riêng, dùng lại bộ Torch/CUDA trên máy và
cài dependencies bổ sung bằng pip resolver. Phiên bản các thư viện ABI của máy
được giữ bằng constraints; MoGe, GraspNetAPI và GraspNess source được ghim commit.
Nếu thiếu native prerequisites hoặc extension lỗi ABI, setup dừng với thông báo.

Xem [hướng dẫn môi trường](env/README.md) để chuẩn bị máy, chạy kiểm tra và xử lý
môi trường cũ. `requirements.lock.txt` chỉ là snapshot lịch sử, không phải lock
cài được trên mọi máy. Bản sửa bootstrap được kiểm thử bằng mock; vẫn cần kiểm
chứng cài mới và suy luận trên GPU thật.

MinkowskiEngine được cài sau bước tải model: dùng wheel chỉ định qua
`MINKOWSKI_ENGINE_WHEEL`, dùng bản sẵn có nếu qua CUDA smoke test, hoặc thử build
source ghim commit. Xem `env/README.md` cho native prerequisites và giới hạn
chưa kiểm chứng trên Kaggle mới. Không yêu cầu host cài MinkowskiEngine từ trước.
