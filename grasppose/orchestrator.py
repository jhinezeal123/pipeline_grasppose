"""Pipeline orchestration.

Only this layer knows execution order and the VRAM phase invariant. Concrete models are
injected through factories, which keeps orchestration independently testable.
"""

import threading
import time

import numpy as np

from .adapters import GroundingDinoDetector, SamSegmenter, MogeDepth, GraspnessModel
from .config import GRIP_HW_OPEN_M
from .geometry import depth_range_str, depth_to_cloud
from .runtime import _log, _vram


class GraspPipeline:
    """Coordinate detector/depth/segmenter/grasp modules through the 3 VRAM phases."""

    def __init__(self, detector_factory=GroundingDinoDetector,
                 segmenter_factory=SamSegmenter,
                 depth_factory=MogeDepth,
                 grasper_factory=GraspnessModel):
        self.detector_factory = detector_factory
        self.segmenter_factory = segmenter_factory
        self.depth_factory = depth_factory
        self.grasper_factory = grasper_factory

    def run(self, image, prompt, fov_x=None, detector=None, segmenter=None,
                   depther=None, grasper=None):
        """Chay 4 model theo 3 phase.
    
        PHASE 1 co y cho MoGe + Grounding-DINO chay SONG SONG (hai model nhe nhat,
        tong VRAM van lot T4), doi lai giam gan mot nua thoi gian nap. Tu PHASE 2 tro
        di moi phase chi giu DUNG MOT model nang: SAM xong moi den GraspNess. Ly do la
        GraspNess + MinkowskiEngine an VRAM lon, khong the dung chung voi model khac.
    
        Vi vay moi lan release() xong deu duoc kiem tra: neu khong nha duoc thi ham
        dung ngay, vi VRAM luc do khong con dang tin de nap tiep.
    
        Tra ve dict: {"det":..., "seg":..., "dep":..., "grasp":..., "cloud":..., "K":...}
        """
        image = np.asarray(image)
        h, w = image.shape[:2]
        out = {}
    
        # ---------------- PHASE 1: MoGe + DINO song song ----------------
        _log("PHASE 1: nap MoGe + Grounding-DINO cung luc, chay song song")
        depther = depther or self.depth_factory()
        detector = detector or self.detector_factory()
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
        segmenter = segmenter or self.segmenter_factory()
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
                grasper = grasper or self.grasper_factory()
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
    
    
    


DEFAULT_PIPELINE = GraspPipeline()


def run_phases(image, prompt, fov_x=None, detector=None, segmenter=None,
               depther=None, grasper=None):
    """Compatibility function delegating to the default orchestrator."""
    return DEFAULT_PIPELINE.run(image, prompt, fov_x=fov_x,
                                detector=detector, segmenter=segmenter,
                                depther=depther, grasper=grasper)
