#!/usr/bin/env python3
"""pipeline.py — Pipeline hoan chinh: anh tho -> 4 anh ket qua.

    raw image
        |-- Grounding-DINO --> box
        |                          `--> SAM --> mask
        `-- MoGe --> depth map + point cloud
                                   |
                        mask chon phan cloud can thiet
                                   |
                              GraspNess --> grasp pose

Dau ra: 4 anh (box / mask / depthmap / grasp pose).

Chien luoc bo nho — 3 phase:
    phase 1: MoGe + Grounding-DINO; Jetson Xavier chay TUAN TU.
             May GPU manh co the bo PIPELINE_SERIAL_GPU de chay song song.
    phase 2: DINO da nha -> nap SAM -> chay -> nha.
    phase 3: tat ca da nha -> nap GraspNess -> chay -> nha.

Chay:
    python3 pipeline.py --img img/bag.png --out output
    python3 pipeline.py --img img/bag.png --prompt "a little bag"
    python3 pipeline.py --img render.png --fov-x 77.4    # anh render MoJoCo
"""

import argparse
import gc
import os
import sys
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(HERE, "model")

# Trong so GraspNess la checkpoint cua GraspNet; repo graspness_unofficial
# KHONG nam trong `dependencies` vi no khong build duoc MinkowskiEngine o day.
# Dat no canh repo nay hoac tro qua bien moi truong GRASPNESS_HOME.
GRASPNESS_HOME = os.environ.get("GRASPNESS_HOME",
                                os.path.join(MODEL_DIR, "graspness_unofficial"))

#: Kich thuoc voxel cua GraspNet (met) — giong graspnet_dataset.py
VOXEL = 0.005

#: Khe mo TOI DA THAT cua kep myArm M750 (met).
#: Lay tu myarm_m750_mujoco.xml: left/right_gripper_joint range="0 0.0345",
#: hai ngon mo nguoc chieu nhau nen khe ho = 2 x 0.0345 = 0.069 m.
GRIP_HW_OPEN_M = 0.0694

#: Khe mo dung lam NGUONG MAC DINH khi loc tu the gap (met).
#: = 80 mm theo yeu cau nguoi dung ("tang gioi han kep len 8cm"). Rong hon
#: GRIP_HW_OPEN_M thi kep that KHONG mo toi, nhung van ve ra de nhin thay.
#: Muon dung cho robot that: --max-width 0.0694
GRIP_MAX_OPEN_M = 0.080

#: Mac dinh cho cau lenh van ban neu nguoi dung khong truyen --prompt
DEFAULT_PROMPT = "the object"


# ===========================================================================
# 0. NAP graspnetAPI MA KHONG CHAY __init__.py
# ===========================================================================
# graspnetAPI/__init__.py keo theo graspnet_eval -> eval_utils -> dexnet ->
# autolab_core / skimage / ... Day la chuoi phan DANH GIA (cham diem grasp tren
# dataset GraspNet-1Billion), minh khong dung. No keo theo 4-5 dependency nang
# va de vo.
#
# Cach chua: dang ky san module 'graspnetAPI' trong sys.modules voi __path__ tro
# vao thu muc that. Khi do `import graspnetAPI.grasp` tim thay file qua __path__
# va KHONG chay __init__.py nua -> khong keo theo dexnet.
def _load_graspnetapi():
    """Tra ve module graspnetAPI.grasp. Raise RuntimeError neu khong co."""
    import importlib
    import types

    if "graspnetAPI" not in sys.modules:
        repo = os.path.join(MODEL_DIR, "graspnetAPI_repo")
        pkg_dir = os.path.join(repo, "graspnetAPI")
        if not os.path.isdir(pkg_dir):
            raise RuntimeError(
                "khong thay graspnetAPI tai %s — chay run.sh de clone ve" % repo)
        pkg = types.ModuleType("graspnetAPI")
        pkg.__path__ = [pkg_dir]          # >> khong chay __init__.py <<
        pkg.__file__ = os.path.join(pkg_dir, "__init__.py")
        sys.modules["graspnetAPI"] = pkg
        if repo not in sys.path:
            sys.path.insert(0, repo)
    return importlib.import_module("graspnetAPI.grasp")


def _add_sys_path():
    for p in (HERE,
              os.path.join(MODEL_DIR, "moge_repo"),
              os.path.join(MODEL_DIR, "utils3d_repo"),
              os.path.join(MODEL_DIR, "graspnetAPI_repo")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


# ===========================================================================
# 1. TIEN ICH CHUNG
# ===========================================================================
def _log(msg):
    print("[pipeline] %s" % msg, flush=True)


def _vram(tag=""):
    """In muc VRAM dang dung — de nhin ro phase nao con giu model."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        a = torch.cuda.memory_allocated() / 2**20
        r = torch.cuda.memory_reserved() / 2**20
        _log("  VRAM%s: da cap phat %.0f MB / da giu %.0f MB" % (tag, a, r))
    except Exception:
        pass


def _free(obj, *attrs):
    """Nha model khoi VRAM: xoa attribute -> gc -> empty_cache.

    Phai xoa attribute TRUOC khi gc/empty_cache, khong phai sau. Ban cu lam
    `del o` tren tung doi so — nhung do chi xoa ten cuc bo trong vong lap, con
    tuple tham so VA chinh `self.model` van giu reference. Nen gc.collect() va
    empty_cache() chay luc model CHUA duoc giai phong, tuc la khong thu hoi duoc
    gi; chi den khi ham return va dong `self.model = ... = None` chay sau do thi
    moi nha — nhung luc do da khong con empty_cache nua.

    Doi sang nhan (obj, ten_attr...): dat attribute ve None ngay tai day, roi moi
    gc + empty_cache. `obj` giu ten trong suot ham la khong sao — quan trong la
    attribute (tham chieu THAT toi model) da bi cat.
    """
    for name in attrs:
        setattr(obj, name, None)
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        # synchronize truoc: empty_cache() can moi kernel dung model da chay xong,
        # neu khong thi bo nho co the chua kip duoc tra ve allocator.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def K_from_fovy(fovy_deg, w, h):
    """Ma tran intrinsics tu truong nhin doc (do) — dung cho anh RENDER.

    fx = fy = (h/2) / tan(fovy/2)   (pixel vuong)
    """
    fy = (h / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, w / 2.0],
                     [0.0, fy, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def fov_x_from_fovy(fovy_deg, w, h):
    """fov_x = 2*atan(tan(fovy/2) * w/h) — tham so MoGe can cho anh render."""
    return float(2.0 * np.degrees(np.arctan(
        np.tan(np.radians(fovy_deg) / 2.0) * float(w) / float(h))))


def depth_range_str(depth, zmin=0.05):
    """Mo ta khoang do sau de IN LOG, chiu duoc mang KHONG co pixel hop le.

    MoGe khong do duoc gi thi tra ve toan so 0 (hoac toan inf/nan). Loc roi moi
    lay min/max la sai: mang rong thi `.min()` nem
    "zero-size array to reduction operation minimum which has no identity",
    va ca pipeline chet ngay o dong log chan doan — dung luc can log nhat.
    O day tra ve CHUOI noi ro la khong do duoc, khong tra so bia.
    """
    d = np.asarray(depth, np.float64)
    fin = d[np.isfinite(d) & (d > zmin)]
    if fin.size == 0:
        return "khong do duoc (0 px hop le)"
    return "%.3f..%.3f m (%d px hop le)" % (fin.min(), fin.max(), fin.size)


def depth_to_cloud(depth, K, mask=None):
    """(H,W) met + K -> point cloud (N,3) he camera OpenCV (x phai, y xuong, z toi).

    mask: neu truyen, CHI lay pixel thuoc mask. Day la duong dung.
          KHONG dung hinh bao (bbox) mo rong: ban tung lam vay va diem cua vat
          the khac trong canh lot vao cloud -> sinh grasp nam ngoai vat.
    """
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05)
    if mask is not None:
        ok &= np.asarray(mask).astype(bool)
    zz = z[ok]
    px = (xs[ok] - K[0, 2]) * zz / K[0, 0]
    py = (ys[ok] - K[1, 2]) * zz / K[1, 1]
    return np.stack([px, py, zz], -1).astype(np.float32)


def nms_grasps(gg, t_th=0.03, r_th=np.pi / 6):
    """NMS cho grasp group: loai grasp trung tam <3cm VA goc xoay <30 do."""
    if len(gg) == 0:
        return gg
    order = np.argsort(-gg[:, 0])
    T = gg[:, 13:16]
    R = gg[:, 4:13].reshape(-1, 3, 3)
    keep = []
    for i in order:
        if all(not (np.linalg.norm(T[i] - T[k]) < t_th and
                    np.arccos(np.clip((np.trace(R[i].T @ R[k]) - 1) / 2, -1, 1)) < r_th)
               for k in keep):
            keep.append(i)
    return gg[keep]


# ===========================================================================
# 2. HIEN THUC CU THE CUA 4 MODULE
# ===========================================================================
from Object_Detection import Object_Detection          # noqa: E402
from Segmentation import Segmentation                  # noqa: E402
from Depth_Estimate import Depth_Estimate              # noqa: E402
from GraspNess import GraspNess                        # noqa: E402


class GroundingDinoDetector(Object_Detection):
    """Grounding-DINO — van ban -> hop bao vat the."""

    def __init__(self, model_path=None, device=None, threshold=0.20,
                 text_threshold=0.20):
        self.model_path = model_path or os.path.join(MODEL_DIR, "grounding-dino-tiny")
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.threshold = float(threshold)
        self.text_threshold = float(text_threshold)
        self.model = None
        self.proc = None
        self._image = None
        self._prompt = None
        self._out = None

    def prepare(self, image, prompt):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        self._image = np.asarray(image)
        self._prompt = str(prompt).lower().strip()
        if self.model is None:
            t0 = time.time()
            self.proc = AutoProcessor.from_pretrained(self.model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_path).to(self.device).eval()
            _log("DINO nap xong trong %.1fs (%s)" % (time.time() - t0, self.device))
        inp = self.proc(images=_to_pil(self._image), text=self._prompt,
                        return_tensors="pt").to(self.device)
        with torch.no_grad():
            self._out = self.model(**inp)
        self._inp = inp
        return self

    def inference(self):
        try:
            res = self.proc.post_process_grounded_object_detection(
                self._out, self._inp.input_ids,
                threshold=self.threshold, text_threshold=self.text_threshold,
                target_sizes=[_to_pil(self._image).size[::-1]])[0]
            boxes = res["boxes"].detach().cpu().numpy().astype(np.float32)
            scores = res["scores"].detach().cpu().numpy().astype(np.float32)
            labels = [str(s) for s in res.get("labels", [])]
        except Exception as e:
            return {"boxes": np.zeros((0, 4), np.float32), "scores": np.zeros(0, np.float32),
                    "labels": [], "reason": "post_process loi: %s: %s" % (type(e).__name__, e)}
        if len(boxes) == 0:
            return {"boxes": boxes, "scores": scores, "labels": labels,
                    "reason": "khong co hop nao qua nguong %.2f cho prompt %r"
                              % (self.threshold, self._prompt)}
        # Loc theo cau lenh: DINO doi khi tra nhan khac (vi du "bag" khi hoi
        # "a little bag"), va doi khi tra manh WordPiece ("##rmos").
        best = _phrase_match(labels, self._prompt)
        if best is not None:
            boxes, scores = boxes[best:best + 1], scores[best:best + 1]
            labels = [labels[best]]
        order = np.argsort(-scores)
        return {"boxes": boxes[order], "scores": scores[order],
                "labels": [labels[i] for i in order], "reason": None}

    def release(self):
        # _free() dat attribute ve None roi moi gc + empty_cache — xem docstring.
        # PHAI gom ca '_inp': prepare() luu input da .to(self.device), nen tren
        # CUDA day la tensor GPU that. De no lai thi empty_cache() van chay khi
        # con tham chieu GPU — dung cai loi ordering ma _free() sinh ra de tranh.
        _free(self, 'model', 'proc', '_out', '_inp')


class SamSegmenter(Segmentation):
    """SAM — hop -> mat na pixel.

    Chay tren CPU fp32: fp16 tren GPU lam suy bien logits, con fp32 tren GPU
    thi het VRAM tren T4.
    """

    def __init__(self, model_path=None, device="cpu"):
        self.model_path = model_path or os.path.join(MODEL_DIR, "sam-vit-base")
        self.device = device
        self.model = None
        self.proc = None
        self._boxes = None
        self._image = None

    def prepare(self, image, boxes):
        import torch
        from transformers import SamModel, SamProcessor
        self._image = np.asarray(image)
        self._boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        if self.model is None:
            t0 = time.time()
            self.proc = SamProcessor.from_pretrained(self.model_path)
            self.model = SamModel.from_pretrained(self.model_path).to(self.device).eval()
            _log("SAM nap xong trong %.1fs (%s fp32)" % (time.time() - t0, self.device))
        return self

    def inference(self):
        import torch
        if len(self._boxes) == 0:
            h, w = self._image.shape[:2]
            return {"mask": np.zeros((h, w), bool), "iou": np.zeros(0, np.float32),
                    "best": -1, "n_pred": 0,
                    "reason": "khong co hop nao de dua vao SAM"}
        h, w = self._image.shape[:2]
        try:
            inp = self.proc(_to_pil(self._image),
                            input_boxes=[self._boxes.tolist()],
                            return_tensors="pt").to(self.device)
            with torch.no_grad():
                out = self.model(**inp)
            masks = self.proc.image_processor.post_process_masks(
                out.pred_masks.detach().cpu(), inp["original_sizes"].cpu(),
                inp["reshaped_input_sizes"].cpu())[0]
            iou = out.iou_scores.detach().cpu().numpy().reshape(-1,
                                                               masks.shape[1] if masks.dim() > 3 else 1)
        except Exception as e:
            return {"mask": np.zeros((h, w), bool), "iou": np.zeros(0, np.float32),
                    "best": -1, "n_pred": 0,
                    "reason": "SAM loi: %s: %s" % (type(e).__name__, e)}
        # masks: (1, n_pred, H, W) hoac (n_pred, H, W)
        mm = masks.numpy() if hasattr(masks, "numpy") else np.asarray(masks)
        mm = mm.reshape(-1, h, w)
        iou_flat = np.asarray(iou).reshape(-1)[:len(mm)]
        if len(mm) == 0:
            return {"mask": np.zeros((h, w), bool), "iou": iou_flat, "best": -1,
                    "n_pred": 0, "reason": "SAM khong tra ung vien nao"}
        best = int(np.argmax(iou_flat)) if len(iou_flat) else 0
        return {"mask": mm[best].astype(bool), "iou": iou_flat.astype(np.float32),
                "best": best, "n_pred": len(mm), "reason": None}

    def release(self):
        _free(self, 'model', 'proc')


class MogeDepth(Depth_Estimate):
    """MoGe-2 ViT-S — anh RGB -> metric depth + point cloud.

    KHONG nap them dinov2_vitl14_pretrain.pth: from_pretrained() da nap encoder
    da fine-tune, nap chong len se GHI DE va lam sai ty le do sau.
    """

    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or os.path.join(MODEL_DIR, "moge-2-vits-normal")
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.model = None
        self._fov_x = None
        self._out = None
        self._h = self._w = 0

    def _resolve(self):
        """MoGeModel.from_pretrained() chi nhan FILE.

        Nhung run.sh tai bang huggingface snapshot_download(local_dir=...) nen
        thuong ra mot THU MUC chua model.pt. Nhan ca hai cho khoi vo:
          model/moge-2-vits-normal          (thu muc) -> .../model.pt
          model/moge-2-vits-normal/model.pt (file)    -> dung nguyen
          Ruicheng/moge-2-vits-normal        (repo id) -> de nguyen, HF tu tai
        """
        p = self.model_path
        if os.path.isdir(p):
            for cand in ("model.pt", "model.safetensors", "pytorch_model.bin"):
                if os.path.isfile(os.path.join(p, cand)):
                    return os.path.join(p, cand)
            raise RuntimeError(
                "thu muc %r khong chua model.pt / model.safetensors — "
                "xoa thu muc do roi chay lai run.sh" % p)
        if os.path.sep in p and not os.path.isfile(p):
            raise RuntimeError("khong thay trong so MoGe tai %r — chay run.sh" % p)
        return p                                  # repo id -> HF tu tai

    def prepare(self, image, fov_x=None):
        import torch
        _add_sys_path()
        from moge.model.v2 import MoGeModel
        self._fov_x = None if fov_x is None else float(fov_x)
        rgb = np.asarray(image)
        self._h, self._w = rgb.shape[:2]
        if self.model is None:
            t0 = time.time()
            self.model = MoGeModel.from_pretrained(self._resolve()).to(self.device).eval()
            _log("MoGe nap xong trong %.1fs (%s)" % (time.time() - t0, self.device))
        t = _to_tensor_chw(rgb, self.device)
        kw = {} if self._fov_x is None else {"fov_x": self._fov_x}
        # Xavier defaults to level 4 via run.sh. MoGe-2 level 9 is materially
        # heavier and not a good default for an 8-SM Volta GPU.
        level = int(os.environ.get("MOGE_RESOLUTION_LEVEL", "9"))
        level = max(0, min(9, level))
        with torch.no_grad():
            self._out = self.model.infer(
                t, resolution_level=level,
                use_fp16=str(self.device).startswith("cuda"), **kw)
        _log("MoGe-2 resolution_level=%d" % level)
        return self

    def inference(self):
        import torch
        if self._out is None:
            h, w = self._h, self._w
            return {"depth": np.zeros((h, w), np.float32),
                    "points": np.zeros((h, w, 3), np.float32),
                    "intrinsics": np.eye(3, dtype=np.float32), "fov_x_deg": 0.0,
                    "reason": "chua chay prepare()"}
        h, w = self._h, self._w
        o = self._out
        pts = o["points"].squeeze(0).detach().cpu().numpy().astype(np.float32)
        # MoGe mac dinh apply_mask=True -> pixel khong hop le thanh inf.
        # Phai nan_to_num TRUOC khi dung, neu khong moi phep tinh sau deu NaN.
        pts = np.nan_to_num(pts, nan=0.0, posinf=0.0, neginf=0.0)
        dep = o["depth"].squeeze(0).detach().float().cpu().numpy()
        dep = np.nan_to_num(dep, nan=0.0, posinf=0.0, neginf=0.0)
        dep = np.clip(dep, 0.0, 10.0).astype(np.float32)
        Ki = o["intrinsics"].squeeze(0).detach().cpu().numpy().astype(np.float64).copy()
        if Ki.max() <= 1.5:                          # con dang chuan hoa
            Ki[0, :] *= w
            Ki[1, :] *= h
        fx = float(Ki[0, 0])
        fovx = float(2.0 * np.degrees(np.arctan(w / 2.0 / max(fx, 1e-6))))
        return {"depth": dep, "points": pts, "intrinsics": Ki.astype(np.float32),
                "fov_x_deg": fovx, "reason": None}

    def release(self):
        _free(self, 'model', '_out')


class GraspnessModel(GraspNess):
    """GraspNet + Graspness — point cloud -> tu the gap.

    Can repo `graspness_unofficial` (khong public tren dependencies vi
    MinkowskiEngine khong build duoc tu dong).
    """

    def __init__(self, model_path=None, home=None, device=None):
        self.model_path = model_path or os.path.join(MODEL_DIR, "graspness_reckpt.pth")
        self.home = home or GRASPNESS_HOME
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.net = None
        self.ME = None
        self.pred_decode = None
        self._pts = None
        self._reason = None

    def prepare(self, points, num_point=15000):
        import torch
        self._pts = np.asarray(points, np.float32).reshape(-1, 3)
        if self.net is None:
            self._load()
        if self.net is None:
            return self
        # Lay mau dung num_point diem — giong infer_vis_grasp.py cua upstream
        p = self._pts
        n = len(p)
        if n == 0:
            return self
        if n >= num_point:
            idx = np.random.choice(n, num_point, replace=False)
        else:
            idx = np.concatenate([np.arange(n),
                                  np.random.choice(n, num_point - n, replace=True)])
        self._pts = np.ascontiguousarray(p[idx], np.float32)
        return self

    def _load(self):
        import types
        import torch
        import MinkowskiEngine as ME
        if not os.path.isdir(self.home):
            self._reason = ("khong thay graspness_unofficial tai %r — dat repo "
                            "vao model/ hoac tro GRASPNESS_HOME" % self.home)
            return
        if not os.path.isfile(self.model_path):
            self._reason = "khong thay checkpoint %r" % self.model_path
            return
        # sys.path giong test.py cua upstream: root + models + dataset + utils
        # + pointnet2 + knn (pointnet2_utils.py co bare "import pytorch_utils")
        for sub in ("", "models", "dataset", "utils", "pointnet2", "knn"):
            q = os.path.join(self.home, sub)
            if os.path.isdir(q) and q not in sys.path:
                sys.path.insert(0, q)
        # models/graspnet.py -> label_generation -> knn.knn_modules -> knn_pytorch
        # (CUDA ext, chi dung cho training labels). Che module knn di.
        if "knn" not in sys.modules:
            km = types.ModuleType("knn")
            kp = types.ModuleType("knn.knn_modules")

            def _knn_stub(*_a, **_k):
                raise NotImplementedError("knn chi dung cho training labels")

            kp.knn = _knn_stub
            km.knn_modules = kp
            sys.modules["knn"] = km
            sys.modules["knn.knn_modules"] = kp
        from models.graspnet import GraspNet, pred_decode
        dev = torch.device(self.device)
        net = GraspNet(seed_feat_dim=512, is_training=False).to(dev).eval()
        try:
            ck = torch.load(self.model_path, map_location=dev)
        except Exception:            # torch >= 2.6 doi weights_only=True
            ck = torch.load(self.model_path, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model_state_dict"])
        self.net, self.ME, self.pred_decode = net, ME, pred_decode
        _log("GraspNess nap xong: %s (epoch=%s)"
             % (os.path.basename(self.model_path), ck.get("epoch")))

    def inference(self):
        import torch
        if self.net is None:
            return {"graspgroup": np.zeros((0, 17), np.float64),
                    "reason": self._reason or "GraspNess chua nap duoc"}
        ME = self.ME
        cloud = np.ascontiguousarray(np.asarray(self._pts, np.float32).reshape(-1, 3))
        if len(cloud) == 0:
            return {"graspgroup": np.zeros((0, 17), np.float64),
                    "reason": "point cloud rong"}
        # Sao chep nhanh phan single-sample cua minkowski_collate_fn (verbatim)
        coors, feats = ME.utils.sparse_collate([cloud / VOXEL],
                                               [np.ones_like(cloud, np.float32)])
        coors = np.ascontiguousarray(coors, dtype=np.int32)
        coors, feats, _, q2o = ME.utils.sparse_quantize(
            coors, feats, return_index=True, return_inverse=True)
        dev = torch.device(self.device)
        batch = {
            "coors": torch.from_numpy(np.ascontiguousarray(coors)).to(dev),
            "feats": torch.from_numpy(np.ascontiguousarray(feats)).float().to(dev),
            "quantize2original": torch.from_numpy(
                np.ascontiguousarray(q2o)).long().to(dev),
            "point_clouds": torch.from_numpy(cloud).unsqueeze(0).to(dev),
        }
        with torch.no_grad():
            preds = self.pred_decode(self.net(batch))
        gg = np.asarray(preds[0].detach().cpu().numpy(), np.float64).reshape(-1, 17)
        return {"graspgroup": nms_grasps(gg), "reason": None}

    def release(self):
        # pred_decode la FUNCTION (import tu models.graspnet, goi o
        # `self.pred_decode(self.net(batch))`), khong phai tensor. Dat ve None
        # ngay trong _free() cho nhat quan: moi tham chieu tới model/function cua
        # upstream deu duoc cat TRUOC khi gc + empty_cache chay.
        _free(self, 'net', 'ME', 'pred_decode')


# ===========================================================================
# 3. TIEN ICH ANH
# ===========================================================================
def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _to_pil(rgb_uint8_hwc):
    from PIL import Image
    a = np.asarray(rgb_uint8_hwc)
    if a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    return Image.fromarray(a[:, :, :3])


def _to_tensor_chw(rgb_uint8_hwc, device):
    import torch
    a = np.asarray(rgb_uint8_hwc)[:, :, :3]
    t = torch.from_numpy(np.ascontiguousarray(a)).to(device)
    return t.permute(2, 0, 1)[None].float() / 255.0


def _phrase_match(labels, prompt):
    """Tim nhan khop voi cau lenh. Tra chi so hoac None.

    DINO doi khi tra manh WordPiece (vi du "##rmos" cho "thermos") — manh nay
    khong khop bang cach nao, va do la ly do that bai that da gap.
    """
    pb = str(prompt).lower().strip(" .,\"'")
    toks = [t for t in pb.replace(",", " ").split() if len(t) >= 3]
    for i, lab in enumerate(labels):
        lb = str(lab).lower().strip(" .,\"'")
        if not lb:
            continue
        if lb in pb or pb in lb:
            return i
        if any(t in lb for t in toks):
            return i
    # Khong nhan nao khop: neu chi co DUNG MOT hop thi van dung no.
    # ponytail: truong hop nay xay ra that (nhan "##rmos"), va bo qua no lam
    # mat ket qua dung. Giu lai vi loi ich ro rang.
    if len(labels) == 1:
        return 0
    return None


# ===========================================================================
# 4. VE 4 ANH KET QUA
# ===========================================================================
def draw_box(image, det):
    """Anh 1/4: anh goc + hop cua Grounding-DINO."""
    import cv2
    im = np.ascontiguousarray(np.asarray(image)[:, :, ::-1].copy())
    for i, (b, s) in enumerate(zip(det["boxes"], det["scores"])):
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 140, 255), 4)
        lab = det["labels"][i] if i < len(det["labels"]) else "?"
        _put(im, "DINO: '%s'  score %.3f" % (lab, s))
    if len(det["boxes"]) == 0:
        _put(im, "DINO: khong co hop  (%s)" % (det.get("reason") or "?"))
    return im[:, :, ::-1]


def draw_mask(image, seg):
    """Anh 2/4: anh goc + mat na cua SAM phu len."""
    import cv2
    im = np.ascontiguousarray(np.asarray(image)[:, :, ::-1].copy())
    m = np.asarray(seg["mask"]).astype(bool)
    if m.any():
        a = im.astype(np.float32)
        a[m] = a[m] * 0.45 + np.array([120.0, 255.0, 0.0]) * 0.55
        im = a.astype(np.uint8)
        ys, xs = np.where(m)
        cv2.rectangle(im, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 255), 3)
        _put(im, "SAM mask: %d px (%.1f%%)  IoU %.3f"
             % (m.sum(), 100.0 * m.mean(),
                seg["iou"][seg["best"]] if len(seg["iou"]) else 0.0),
             bg=(90, 180, 0))
    else:
        _put(im, "SAM: khong co mat na  (%s)" % (seg.get("reason") or "?"))
    return im[:, :, ::-1]


def draw_depth(dep):
    """Anh 3/4: DEPTH GOC cua MoGe tren TOAN ANH (khong mask).

    Chuan hoa theo percentile cua TOAN ANH. Chuan hoa theo mask la SAI: vung
    ngoai mask bi bao hoa va anh tro nen khong doc duoc.
    """
    import cv2
    d = np.asarray(dep["depth"], np.float32)
    fin = np.isfinite(d) & (d > 0)
    out = np.zeros((*d.shape, 3), np.uint8)
    if fin.sum() < 10:
        _put(out, "MoGe: khong co do sau hop le")
        return out[:, :, ::-1]
    v0, v1 = np.percentile(d[fin], 2), np.percentile(d[fin], 98)
    dn = np.clip((d - v0) / max(v1 - v0, 1e-6), 0, 1)
    cm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm[~fin] = (40, 40, 40)
    h, w = d.shape
    bar = np.zeros((58, w, 3), np.uint8)
    bar[12:46] = cv2.applyColorMap(
        np.tile(np.linspace(0, 255, w, dtype=np.uint8), (34, 1)), cv2.COLORMAP_TURBO)
    for i in range(6):
        x = int(i * (w - 1) / 5)
        cv2.line(bar, (x, 42), (x, 50), (255, 255, 255), 1)
        cv2.putText(bar, "%.2f" % (v0 + (v1 - v0) * i / 5),
                    (min(x, max(0, w - 56)), 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1)
    cv2.putText(bar, "m", (w - 22, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1)
    cv2.putText(bar, "MoGe depth GOC toan anh | xam = khong hop le",
                (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([cm, bar])[:, :, ::-1]


def grasp_empty_msg(n_raw, reason, max_width):
    """Cau bao khi khong ve duoc tu the nao.

    Tach rieng khoi draw_grasp de TEST duoc: ba nguyen nhan duoi day khac han
    nhau, gop lai lam mot cau (nhu truoc) thi khong phan biet duoc "model nap
    hong" voi "loc qua chat" — da lam mat thoi gian dung mot lan.
    """
    if reason:
        return "grasp: KHONG CHAY DUOC - %s" % reason
    if n_raw == 0:
        return "grasp: khong co tu the nao (mask rong?)"
    return ("grasp: %d tu the nhung TAT CA rong hon %.0f mm"
            % (n_raw, max_width * 1000))


def hw_open_note(width, hw_open=GRIP_HW_OPEN_M):
    """Chu thich them cho mot tu the: no co lot vao KHE MO THAT cua kep khong.

    Nhanh cu trong draw_grasp hoi `g[1] > max_width`, nhung `g` lay tu `sel` da
    loc `<= max_width` — dieu kien do KHONG BAO GIO dung, nen chu "VUOT" la code
    chet. Doi sang so sanh voi GRIP_HW_OPEN_M moi co nghia: `max_width` (mac dinh
    80 mm) la nguong LOC de VE, con GRIP_HW_OPEN_M (69.4 mm) la khe mo THAT cua
    myArm M750. Tu the rong hon 69.4 mm van duoc ve ra cho nguoi dung nhin thay
    (dung y do o comment dau file), nhung can noi ro la kep that khong mo toi.
    """
    if width > hw_open:
        return "  VUOT khe mo that %.0f mm" % (hw_open * 1000)
    return ""


def _box_wireframe(size_xyz, offset_xyz):
    """8 vertices + 12 canh cua mot hop; khong can Open3D."""
    sx, sy, sz = [float(v) for v in size_xyz]
    ox, oy, oz = [float(v) for v in offset_xyz]
    v = np.array([[0, 0, 0], [sx, 0, 0], [0, 0, sz], [sx, 0, sz],
                  [0, sy, 0], [sx, sy, 0], [0, sy, sz], [sx, sy, sz]],
                 dtype=np.float64)
    v += np.array([ox, oy, oz], dtype=np.float64)
    e = np.array([[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
                  [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]],
                 dtype=np.int32)
    return v, e


def _gripper_wireframe(g):
    """Hinh hoc tuong duong plot_gripper_pro_max cua graspnetAPI, bang NumPy."""
    g = np.asarray(g, np.float64).reshape(17)
    score, width, depth = float(g[0]), float(g[1]), float(g[3])
    R = g[4:13].reshape(3, 3)
    center = g[13:16]
    height = 0.004
    finger = 0.004
    tail = 0.04
    base = 0.02
    specs = [
        ((depth + base + finger, finger, height), (-base - finger, -width / 2 - finger, -height / 2)),
        ((depth + base + finger, finger, height), (-base - finger,  width / 2,          -height / 2)),
        ((finger, width, height),                 (-finger - base, -width / 2,          -height / 2)),
        ((tail, finger, height),                  (-tail - finger - base, -finger / 2,  -height / 2)),
    ]
    vertices, edges = [], []
    off = 0
    for size, pos in specs:
        v, e = _box_wireframe(size, pos)
        vertices.append(v)
        edges.append(e + off)
        off += len(v)
    V = np.concatenate(vertices, axis=0)
    E = np.concatenate(edges, axis=0)
    V = np.dot(R, V.T).T + center
    color = np.array([np.clip(score, 0, 1), 0.0, 1.0 - np.clip(score, 0, 1)])
    C = np.repeat(color[None, :], len(V), axis=0)
    return V, E, C


def draw_grasp(image, gg, K, max_width=GRIP_MAX_OPEN_M, top=1, min_sep=0.080,
               reason=None):
    """Anh 4/4: chieu wireframe gripper truc tiep, khong phu thuoc Open3D."""
    import cv2

    im = np.ascontiguousarray(np.asarray(image)[:, :, ::-1].copy())
    raw = np.asarray(gg, np.float64).reshape(-1, 17)
    sel = raw[raw[:, 1] <= float(max_width)] if len(raw) else raw
    if len(sel) == 0:
        _put(im, grasp_empty_msg(len(raw), reason, max_width))
        return im[:, :, ::-1]

    pick = []
    for i in np.argsort(-sel[:, 0]):
        c = sel[i, 13:16]
        if all(np.linalg.norm(c - sel[j, 13:16]) > min_sep for j in pick):
            pick.append(int(i))
        if len(pick) == top:
            break

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def proj(Pw):
        z = np.maximum(Pw[:, 2], 1e-6)
        return np.stack([fx * Pw[:, 0] / z + cx, fy * Pw[:, 1] / z + cy], -1)

    lines = []
    for rank, gi in enumerate(pick):
        g = sel[gi]
        V, E, C = _gripper_wireframe(g)
        uv = proj(V)
        fin = np.isfinite(uv).all(1) & (V[:, 2] > 1e-6)
        E = E[fin[E[:, 0]] & fin[E[:, 1]]]
        if len(E):
            order = np.argsort(-V[E].mean(1)[:, 2])
            E = E[order]
            c = np.clip(C[E].mean(1), 0, 1)
            for (a, b), cc in zip(E, c):
                col = (int(min(cc[2] * 1.35, 1) * 255),
                       int(cc[1] * 255),
                       int(min(cc[0] * 1.35, 1) * 255))
                cv2.line(im, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                         col, 2, cv2.LINE_AA)
        span = (uv[fin].max(0) - uv[fin].min(0)) if fin.any() else np.zeros(2)
        zc = float(g[15])
        lines.append("#%d score %.4f  width %.1f mm  z=%.2fm  %dx%d px%s"
                     % (rank + 1, g[0], g[1] * 1000, zc,
                        span[0], span[1], hw_open_note(g[1])))
    _put(im, lines)
    return im[:, :, ::-1]

def _put(im, text, bg=(255, 255, 255), fg=(0, 0, 0)):
    """Ghi chu o goc tren trai, co nen trang de doc duoc.

    `text` nhan str hoac list[str]; list thi ve thanh nhieu dong.
    """
    import cv2
    lines = [text] if isinstance(text, str) else list(text)
    hgt = 24
    cv2.rectangle(im, (0, 0), (min(im.shape[1], 900), 8 + hgt * len(lines)), bg, -1)
    for i, s in enumerate(lines):
        cv2.putText(im, s, (8, 24 + i * hgt), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, fg, 1, cv2.LINE_AA)


# ===========================================================================
# 5. PIPELINE — 3 PHASE
# ===========================================================================
def run_phases(image, prompt, fov_x=None, detector=None, segmenter=None,
               depther=None, grasper=None):
    """Chay 4 model theo 3 phase.

    PHASE 1 co the chay song song tren GPU manh; tren Jetson Xavier run.sh dat
    PIPELINE_SERIAL_GPU=1 de chay MoGe roi DINO tuan tu, giam contention va peak
    unified memory. Tu PHASE 2 tro di moi phase chi giu DUNG MOT model nang:
    SAM xong moi den GraspNess. Ly do la
    GraspNess + MinkowskiEngine an VRAM lon, khong the dung chung voi model khac.

    Vi vay moi lan release() xong deu duoc kiem tra: neu khong nha duoc thi ham
    dung ngay, vi VRAM luc do khong con dang tin de nap tiep.

    Tra ve dict: {"det":..., "seg":..., "dep":..., "grasp":..., "cloud":..., "K":...}
    """
    image = np.asarray(image)
    h, w = image.shape[:2]
    out = {}

    # ---------------- PHASE 1: MoGe + DINO ----------------
    serial_gpu = os.environ.get("PIPELINE_SERIAL_GPU", "0").lower() not in ("", "0", "false", "no")
    _log("PHASE 1: MoGe + Grounding-DINO (%s)" % ("tuan tu" if serial_gpu else "song song"))
    depther = depther or MogeDepth()
    detector = detector or GroundingDinoDetector()
    errs = {}
    # Loi release() phai de RIENG, khong gop vao errs: errs chi duoc doc khi
    # thieu ket qua ("dep" not in out). Neu inference THANH CONG roi release moi
    # nem, out["dep"] van ton tai nen loi trong errs se khong bao gio duoc doc.
    # Ma release that bai nghia la VRAM chua duoc nha — invariant "DINO/MoGe da
    # nha truoc SAM" khong con dung, phase sau co the OOM.
    rel_errs = {}

    def _depth_job():
        try:
            depther.prepare(image, fov_x=fov_x)
            out["dep"] = depther.inference()
        except Exception as e:
            errs["dep"] = "%s: %s" % (type(e).__name__, e)
        finally:
            # Exception trong luong phu KHONG lam job that bai — no chi in
            # traceback ra stderr roi bien mat. Bat tai day de khong bi che.
            try:
                depther.release()                # xong la nha ngay
            except Exception as e:
                rel_errs["dep"] = "%s: %s" % (type(e).__name__, e)
            _vram(" sau khi MoGe nha")

    def _det_job():
        try:
            detector.prepare(image, prompt)
            out["det"] = detector.inference()
        except Exception as e:
            errs["det"] = "%s: %s" % (type(e).__name__, e)
        finally:
            try:
                detector.release()
            except Exception as e:
                rel_errs["det"] = "%s: %s" % (type(e).__name__, e)
            _vram(" sau khi DINO nha")

    t0 = time.time()
    if serial_gpu:
        # Jetson Xavier: both jobs target the same 8-SM GPU and unified memory.
        # Serial execution avoids contention and lowers peak memory pressure.
        _depth_job()
        _det_job()
    else:
        ths = [threading.Thread(target=_depth_job, name="moge"),
               threading.Thread(target=_det_job, name="dino")]
        for t in ths:
            t.start()
        for t in ths:
            t.join()
    _log("PHASE 1 xong trong %.1fs" % (time.time() - t0))

    # VRAM chua duoc nha => dung NGAY, dung di tiep sang SAM/GraspNess. Chay tiep
    # thi model truoc van chiem VRAM, va loi OOM o phase sau se mang thong bao
    # khong lien quan gi toi nguyen nhan that.
    if rel_errs:
        raise RuntimeError(
            "khong nha duoc model phase 1 (VRAM chua duoc giai phong): %s"
            % "; ".join("%s -> %s" % kv for kv in sorted(rel_errs.items())))

    if "dep" not in out:
        raise RuntimeError("MoGe that bai: %s" % errs.get("dep", "?"))
    out.setdefault("det", {"boxes": np.zeros((0, 4), np.float32),
                           "scores": np.zeros(0, np.float32), "labels": [],
                           "reason": errs.get("det", "?")})
    dep = out["dep"]
    K = np.asarray(dep["intrinsics"], np.float64)
    _log("  MoGe: fov_x=%.2f do | depth toan anh %s"
         % (dep["fov_x_deg"], depth_range_str(dep["depth"])))
    _log("  DINO: %d hop | %s" % (len(out["det"]["boxes"]),
                                  out["det"].get("reason") or "OK"))

    # ---------------- PHASE 2: SAM ----------------
    _log("PHASE 2: nap SAM (DINO da nha)")
    segmenter = segmenter or SamSegmenter()
    try:
        segmenter.prepare(image, out["det"]["boxes"])
        out["seg"] = segmenter.inference()
    except Exception as e:
        out["seg"] = {"mask": np.zeros((h, w), bool), "iou": np.zeros(0, np.float32),
                      "best": -1, "n_pred": 0,
                      "reason": "%s: %s" % (type(e).__name__, e)}
    finally:
        # release() co the nem (vi du thieu torch). O day no nam trong finally
        # nen neu de no thoat ra thi no DE LUON ca out["seg"] vua tinh xong —
        # mask tot van bi vut di va run_phases chet. Bat lai, ghi vao rel_errs.
        try:
            segmenter.release()
        except Exception as e:
            rel_errs["seg"] = "%s: %s" % (type(e).__name__, e)
        _vram(" sau khi SAM nha")
    _log("  SAM: mask %d px (%.1f%%) | %s"
         % (out["seg"]["mask"].sum(), 100.0 * out["seg"]["mask"].mean(),
            out["seg"].get("reason") or "OK"))

    # Dung NGAY tai day, khong doi toi cuoi ham. SAM khong nha duoc thi VRAM
    # khong con dang tin, ma Phase 3 se nap them GraspNess — dung luc de nhat de
    # OOM, va thong bao OOM se khong lien quan gi toi nguyen nhan that.
    # Raise o cuoi ham la qua muon: GraspNess da prepare/inference xong roi.
    if "seg" in rel_errs:
        raise RuntimeError("khong nha duoc model phase 2 (VRAM chua duoc giai "
                           "phong): seg -> %s" % rel_errs["seg"])

    # ---------------- DO SAU CUA VAT (1 con so, met) ----------------
    # Trung vi do sau cua cac pixel NAM TRONG MASK SAM -> do sau cua VAT, khong
    # phai cua ca anh. Dung trung vi (khong phai trung binh) de pixel nhieu o ria
    # mask khong keo lech. Loc > 0 vi MoGe tra 0 o vung khong do duoc.
    mask = np.asarray(out["seg"]["mask"]).astype(bool)
    _dm = dep["depth"][mask]
    _dm = _dm[np.isfinite(_dm) & (_dm > 0)]
    out["depth_m"] = float(np.median(_dm)) if _dm.size else None
    _log("  do sau vat: %s"
         % ("khong co (mask rong)" if out["depth_m"] is None
            else "%.3f m (trung vi tren %d px mask)" % (out["depth_m"], _dm.size)))

    # ---------------- PHASE 3: GraspNess ----------------
    _log("PHASE 3: nap GraspNess (tat ca da nha)")
    if not mask.any():
        out["cloud"] = np.zeros((0, 3), np.float32)
        out["grasp"] = {"graspgroup": np.zeros((0, 17), np.float64),
                        "reason": "mask rong nen khong co cloud"}
        _log("  BO QUA: mask rong")
    else:
        # CLOUD = MASK THUAN (khong phai bbox mo rong)
        cloud = depth_to_cloud(dep["depth"], K, mask=mask)
        out["cloud"] = cloud
        if len(cloud) == 0:
            # mask.any() la True nhung KHONG pixel nao co depth hop le (MoGe
            # tra 0 trong vung mask) -> cloud rong. Neu cu di tiep thi
            # cloud.max(0) nem "zero-size array to reduction operation maximum",
            # va goi GraspNess voi cloud rong cung vo nghia.
            # Xu ly Y HET nhanh mask rong ngay tren: cung cloud rong, cung ly do
            # noi ro, cung BO QUA GraspNess -> hai duong ra ket qua nhat quan.
            out["grasp"] = {"graspgroup": np.zeros((0, 17), np.float64),
                            "reason": "mask co pixel nhung khong pixel nao co "
                                      "depth hop le nen cloud rong"}
            _log("  BO QUA: mask co %d px nhung cloud rong (0 px depth hop le)"
                 % int(mask.sum()))
        else:
            _log("  cloud tu mask: %d diem | bbox %.0f x %.0f x %.0f mm"
                 % (len(cloud), *((cloud.max(0) - cloud.min(0)) * 1000)))
            grasper = grasper or GraspnessModel()
            try:
                grasper.prepare(cloud)
                out["grasp"] = grasper.inference()
            except Exception as e:
                out["grasp"] = {"graspgroup": np.zeros((0, 17), np.float64),
                                "reason": "%s: %s" % (type(e).__name__, e)}
            finally:
                try:
                    grasper.release()
                except Exception as e:
                    rel_errs["grasp"] = "%s: %s" % (type(e).__name__, e)
                _vram(" sau khi GraspNess nha")
            gg = out["grasp"]["graspgroup"]
            n_ok = int((gg[:, 1] <= GRIP_HW_OPEN_M).sum()) if len(gg) else 0
            _log("  GraspNess: %d tu the | %d vua khe kep THAT %.0f mm"
                 % (len(gg), n_ok, GRIP_HW_OPEN_M * 1000))

    # VRAM chua duoc nha o BAT KY phase nao => dung NGAY, dung tra ket qua nhu
    # khong co gi. Ket qua co the van dung, nhung model truoc van chiem VRAM va
    # lan chay sau (hoac phase sau) se OOM voi thong bao khong lien quan.
    if rel_errs:
        raise RuntimeError(
            "khong nha duoc model (VRAM chua duoc giai phong): %s"
            % "; ".join("%s -> %s" % kv for kv in sorted(rel_errs.items())))

    out["K"] = K
    return out


# ===========================================================================
# 6. API CHINH
# ===========================================================================
def pipeline(img, prompt=DEFAULT_PROMPT, fov_x=None,
             max_width=GRIP_MAX_OPEN_M, top=1):
    """Anh tho -> bo 4 anh: box, mask, depthmap, grasp pose.

    Tham so:
        img       : numpy uint8 (H, W, 3) RGB, hoac duong dan file anh
        prompt    : str, cau lenh van ban, vi du "a little bag"
        fov_x     : float hoac None. Anh THAT de None; anh RENDER tu MoJoCo
                    truyen vao (vi du fov_x_from_fovy(62, 640, 480) = 77.40).
        max_width : float, khe mo toi da cua kep (met)
        top       : int, so tu the gap ve len anh grasp pose

    Tra ve:
        dict 4 anh RGB numpy uint8:
            "box"      : anh goc + hop Grounding-DINO
            "mask"     : anh goc + mat na SAM phu len
            "depthmap" : do sau GOC cua MoGe tren toan anh
            "grasp"    : anh goc + tu the gap (mesh upstream)
        va 1 con so:
            "depth_m"  : float hoac None. Do sau cua VAT tinh bang met = TRUNG VI
                         do sau MoGe tren cac pixel nam trong mask SAM (khong phai
                         do sau ca anh). None khi mask rong / khong co pixel hop le.

    Khong raise khi model khong tim thay vat: anh tuong ung se ghi ro ly do.
    """
    if isinstance(img, str):
        from PIL import Image
        img = np.array(Image.open(img).convert("RGB"))
    img = np.asarray(img)[:, :, :3]

    r = run_phases(img, prompt, fov_x=fov_x)
    return {
        "box": draw_box(img, r["det"]),
        "mask": draw_mask(img, r["seg"]),
        "depthmap": draw_depth(r["dep"]),
        "grasp": draw_grasp(img, r["grasp"]["graspgroup"], r["K"],
                            max_width=max_width, top=top,
                            reason=r["grasp"].get("reason")),
        "depth_m": r["depth_m"],
    }


# ===========================================================================
# 7. CLI
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Anh tho -> 4 anh: box, mask, depthmap, grasp pose")
    ap.add_argument("--img", required=True, help="duong dan anh dau vao")
    ap.add_argument("--out", default=os.path.join(HERE, "output"),
                    help="thu muc ghi ket qua")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT,
                    help="cau lenh van ban (mac dinh %r)" % DEFAULT_PROMPT)
    ap.add_argument("--fov-x", type=float, default=None,
                    help="truong nhin ngang (do) — CHI dung cho anh RENDER")
    ap.add_argument("--fov-y", type=float, default=None,
                    help="truong nhin doc (do) cua camera render; tu doi sang fov_x")
    ap.add_argument("--max-width", type=float, default=GRIP_MAX_OPEN_M,
                    help="khe mo toi da cua kep (met), mac dinh %.4f" % GRIP_MAX_OPEN_M)
    ap.add_argument("--top", type=int, default=1,
                    help="so tu the gap ve len anh grasp pose")
    args = ap.parse_args()

    _add_sys_path()

    from PIL import Image
    img = np.array(Image.open(args.img).convert("RGB"))
    h, w = img.shape[:2]
    fov_x = args.fov_x
    if fov_x is None and args.fov_y is not None:
        fov_x = fov_x_from_fovy(args.fov_y, w, h)
        _log("fov_y=%.2f do, %dx%d -> fov_x=%.4f do" % (args.fov_y, w, h, fov_x))

    _log("anh: %s (%dx%d) | prompt=%r | fov_x=%s"
         % (args.img, w, h, args.prompt,
            "tu uoc luong" if fov_x is None else "%.3f do" % fov_x))

    t0 = time.time()
    res = pipeline(img, prompt=args.prompt, fov_x=fov_x,
                   max_width=args.max_width, top=args.top)
    _log("TONG THOI GIAN: %.1fs" % (time.time() - t0))

    os.makedirs(args.out, exist_ok=True)
    stem = os.path.splitext(os.path.basename(args.img))[0]
    saved = []
    for key in ("box", "mask", "depthmap", "grasp"):
        p = os.path.join(args.out, "%s_%s.png" % (stem, key))
        Image.fromarray(res[key]).save(p)
        saved.append(p)
        _log("  ghi %s  (%dx%d)" % (p, res[key].shape[1], res[key].shape[0]))

    if res["depth_m"] is None:
        _log("DO SAU VAT: khong xac dinh duoc (mask rong / DINO khong thay vat)")
    else:
        _log("DO SAU VAT: %.3f m" % res["depth_m"])

    print("\n".join(saved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
