#!/usr/bin/env python3
"""Jetson Xavier grasp pipeline.

YOLOE-26s -> Lite-Mono -> point cloud(depth + K) -> 40^3 TSDF -> VGN TensorRT.
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
from grasppose.config import DEFAULT_PROMPT, GRIP_MAX_OPEN_M, HERE, TSDF_RESOLUTION, TSDF_SIZE_M, TSDF_TRUNC_VOXELS
from grasppose.geometry import K_from_fovy, depth_range_str, depth_to_cloud, fov_x_from_fovy, nms_grasps
from grasppose.edge_adapters import LiteMonoDepth, VgnTensorRT, Yoloe26sVision
from grasppose.orchestrator import GraspPipeline
from grasppose.rendering import _put, draw_box, draw_depth, draw_grasp, draw_mask, grasp_empty_msg, hw_open_note
from grasppose.runtime import _cuda, _free, _log, _vram
from grasppose.tsdf import ProjectiveTSDFBuilder

DEFAULT_PIPELINE = GraspPipeline(
    vision_factory=Yoloe26sVision,
    depth_factory=LiteMonoDepth,
    tsdf_factory=lambda: ProjectiveTSDFBuilder(size_m=TSDF_SIZE_M,
        resolution=TSDF_RESOLUTION, trunc_voxels=TSDF_TRUNC_VOXELS),
    grasper_factory=VgnTensorRT,
)


def run_phases(image, prompt, camera_K=None, fov_x=None, T_cam_volume=None,
               vision=None, depther=None, tsdf_builder=None, grasper=None):
    return DEFAULT_PIPELINE.run(image, prompt, camera_K=camera_K, fov_x=fov_x,
        T_cam_volume=T_cam_volume, vision=vision, depther=depther,
        tsdf_builder=tsdf_builder, grasper=grasper)


def pipeline(img, prompt=DEFAULT_PROMPT, camera_K=None, fov_x=None,
             T_cam_volume=None, max_width=GRIP_MAX_OPEN_M, top=1):
    if isinstance(img, str):
        from PIL import Image
        img = np.array(Image.open(img).convert("RGB"))
    img = np.asarray(img)[:, :, :3]
    r = run_phases(img, prompt, camera_K=camera_K, fov_x=fov_x,
                   T_cam_volume=T_cam_volume)
    return {"box": draw_box(img, r["det"]), "mask": draw_mask(img, r["seg"]),
            "depthmap": draw_depth(r["dep"]),
            "grasp": draw_grasp(img, r["grasp"]["graspgroup"], r["K"],
                max_width=max_width, top=top, reason=r["grasp"].get("reason")),
            "depth_m": r["depth_m"]}


def _camera_K_from_args(args):
    if args.camera_k is None: return None
    fx, fy, cx, cy = map(float, args.camera_k)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], np.float64)


def main():
    ap = argparse.ArgumentParser(description="YOLOE-26s -> Lite-Mono -> TSDF -> VGN TensorRT")
    ap.add_argument("--img", required=True); ap.add_argument("--out", default=os.path.join(HERE, "output"))
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--camera-k", nargs=4, type=float, metavar=("FX", "FY", "CX", "CY"),
                    help="camera intrinsics in pixels; preferred for real cameras")
    ap.add_argument("--fov-x", type=float, default=None); ap.add_argument("--fov-y", type=float, default=None)
    ap.add_argument("--max-width", type=float, default=GRIP_MAX_OPEN_M); ap.add_argument("--top", type=int, default=1)
    args = ap.parse_args()
    from PIL import Image
    img = np.array(Image.open(args.img).convert("RGB")); h, w = img.shape[:2]
    fov_x = args.fov_x
    if fov_x is None and args.fov_y is not None: fov_x = fov_x_from_fovy(args.fov_y, w, h)
    K = _camera_K_from_args(args)
    if K is None and fov_x is None and not os.environ.get("CAMERA_K"):
        ap.error("real-camera pipeline needs --camera-k FX FY CX CY (or CAMERA_K env)")
    t0 = time.time(); res = pipeline(img, prompt=args.prompt, camera_K=K, fov_x=fov_x,
                                     max_width=args.max_width, top=args.top)
    _log("TOTAL: %.2fs" % (time.time() - t0))
    os.makedirs(args.out, exist_ok=True); stem = os.path.splitext(os.path.basename(args.img))[0]; saved=[]
    for key in ("box", "mask", "depthmap", "grasp"):
        path=os.path.join(args.out, "%s_%s.png" % (stem, key)); Image.fromarray(res[key]).save(path); saved.append(path)
    if res["depth_m"] is not None: _log("target depth: %.3f m" % res["depth_m"])
    print("\n".join(saved)); return 0


if __name__ == "__main__": sys.exit(main())
