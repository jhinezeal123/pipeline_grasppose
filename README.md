# grasp_pipeline_repo

Pipeline: **text prompt -> grasp pose**, chay tren may Linux GPU (Kaggle).
Tu mot cau prompt (vi du "cai coc") + mot anh RGB -> vi tri dat tay de gap vat the:

1. **Grounding DINO** (tiny) - prompt -> **bounding box** cua vat the.
2. **SAM** (vit-base) - box -> **mask** chinh xac cua vat the.
3. **MoGe v3** (ViT-L) - anh -> **depth map** + point cloud + intrinsics (3D).
4. **GraspNet / Graspness** - point cloud + mask -> **grasp pose** (6-DoF).

## Cau truc thu muc

```
grasp_pipeline_repo/
  run.sh              # chay tat ca: tai model -> cai thu vien -> chay pipeline
  dependencies        # danh sach model can tai (nguon duy nhat, run.sh parse file nay)
  hf_token            # token HuggingFace, de TRONG cung duoc
  pipeline.py         # CHAY CHINH: dinh nghia 4 lop cu the + noi pipeline + pipeline(img)
  Object_Detection.py # \  KHAI BAO giao dien 4 module (lop truu tuong).
  Segmentation.py     #  | Chi co chu ky prepare()/inference()/release() va
  Depth_Estimate.py   #  | thuoc tinh model_path. Khong dinh nghia gi.
  GraspNess.py        # /  Doi model = thay lop cu the trong pipeline.py.
  test_pipeline_mock.py # kiem thu khong can GPU (model gia)
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
```

`run.sh` chay lai nhieu lan khong tai lai model (idempotent).

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
```

Thay 4 model bang lop gia roi chay `pipeline()` that. Kiem tra: cong thuc
intrinsics, `depth_to_cloud` loc theo mask, NMS, thu tu 3 pha, va 4 anh dau ra.
Chay duoc tren may khong co GPU (chi can numpy + opencv + open3d).
