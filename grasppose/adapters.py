"""Concrete model adapters.

Each adapter implements one interface and owns all framework/model-specific state.
The orchestrator depends on the interfaces, not on transformers/MoGe/Minkowski details.
"""

import os
import sys
import time

import numpy as np

from Object_Detection import Object_Detection
from Segmentation import Segmentation
from Depth_Estimate import Depth_Estimate
from GraspNess import GraspNess

from .config import MODEL_DIR, GRASPNESS_HOME, VOXEL
from .runtime import _cuda, _free, _log
from .geometry import nms_grasps

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
    """MoGe-3 — anh RGB -> do sau + point cloud.

    KHONG nap them dinov2_vitl14_pretrain.pth: from_pretrained() da nap encoder
    da fine-tune, nap chong len se GHI DE va lam sai ty le do sau.
    """

    def __init__(self, model_path=None, device=None):
        self.model_path = model_path or os.path.join(MODEL_DIR, "moge-3-vitl")
        self.device = device or ("cuda" if _cuda() else "cpu")
        self.model = None
        self._fov_x = None
        self._out = None
        self._h = self._w = 0

    def _resolve(self):
        """MoGeModel.from_pretrained() chi nhan FILE.

        Nhung run.sh tai bang huggingface snapshot_download(local_dir=...) nen
        thuong ra mot THU MUC chua model.pt. Nhan ca hai cho khoi vo:
          model/moge-3-vitl          (thu muc)  -> model/moge-3-vitl/model.pt
          model/moge-3-vitl/model.pt (file)     -> dung nguyen
          Ruicheng/moge-3-vitl       (repo id)  -> de nguyen, HF tu tai
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
        from moge.model.v3 import MoGeModel          # v3 — PyPI `moge` la v2
        self._fov_x = None if fov_x is None else float(fov_x)
        rgb = np.asarray(image)
        self._h, self._w = rgb.shape[:2]
        if self.model is None:
            t0 = time.time()
            self.model = MoGeModel.from_pretrained(self._resolve()).to(self.device).eval()
            _log("MoGe nap xong trong %.1fs (%s)" % (time.time() - t0, self.device))
        t = _to_tensor_chw(rgb, self.device)
        kw = {} if self._fov_x is None else {"fov_x": self._fov_x}
        with torch.no_grad():
            self._out = self.model.infer(t, **kw)
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


