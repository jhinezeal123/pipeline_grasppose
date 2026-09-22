#!/usr/bin/env python3
"""Compatibility facade and CLI for the grasp-pose pipeline.

The implementation is split by responsibility under the grasppose package. Existing
callers can keep import pipeline as P while new code can depend on narrower modules.
"""

import argparse
import os
import sys
import time

import numpy as np

from Object_Detection import Object_Detection
from Segmentation import Segmentation
from Depth_Estimate import Depth_Estimate
from GraspNess import GraspNess

from grasppose.config import (HERE, MODEL_DIR, GRASPNESS_HOME, VOXEL,
                              GRIP_HW_OPEN_M, GRIP_MAX_OPEN_M, DEFAULT_PROMPT)
from grasppose.integrations import _load_graspnetapi, _add_sys_path
from grasppose.runtime import _log, _vram, _free, _cuda
from grasppose.geometry import (K_from_fovy, fov_x_from_fovy, depth_range_str,
                                depth_to_cloud, nms_grasps)
from grasppose.adapters import (GroundingDinoDetector, SamSegmenter, MogeDepth,
                                GraspnessModel, _to_pil, _to_tensor_chw,
                                _phrase_match)
from grasppose.rendering import (draw_box, draw_mask, draw_depth, draw_grasp,
                                 grasp_empty_msg, hw_open_note, _put)
from grasppose.orchestrator import GraspPipeline


DEFAULT_PIPELINE = GraspPipeline(
    detector_factory=GroundingDinoDetector,
    segmenter_factory=SamSegmenter,
    depth_factory=MogeDepth,
    grasper_factory=GraspnessModel,
)


def run_phases(image, prompt, fov_x=None, detector=None, segmenter=None,
               depther=None, grasper=None):
    """Compatibility wrapper; dependency injection remains available to tests/callers."""
    return DEFAULT_PIPELINE.run(image, prompt, fov_x=fov_x,
                                detector=detector, segmenter=segmenter,
                                depther=depther, grasper=grasper)

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
