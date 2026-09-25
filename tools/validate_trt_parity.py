#!/usr/bin/env python3
"""Compare the TensorRT pipeline with PyTorch YOLOE/Lite-Mono references."""

import argparse
import gc
import math
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from grasppose.config import (  # noqa: E402
    LITEMONO_HOME,
    LITEMONO_MODEL,
    LITEMONO_WEIGHTS,
    YOLOE_CONF,
    YOLOE_IMGSZ,
    YOLOE_MODEL,
)
from grasppose.domain.geometry import depth_to_cloud  # noqa: E402
from grasppose.prompt_catalog import PromptCatalog  # noqa: E402
from tools.preprocess_yoloe import (  # noqa: E402
    best_mask,
    bgr_image,
    configure_prompt_classes,
    load_yoloe_api,
    mask_iou,
    read_prompt_file,
    validation_paths,
)
from tools.export_litemono_trt import (  # noqa: E402
    depth_from_disp,
    image_tensor,
    load_models,
)


def camera_matrix(values):
    fx, fy, cx, cy = values
    return np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def infer_reference_depth(graph, path, feed_h, feed_w, scale):
    import torch

    with torch.inference_mode():
        tensor = image_tensor(path, feed_h, feed_w, "cuda")
        height, width = tuple(Image.open(path).size[::-1])
        disparity = graph(tensor)
        depth = depth_from_disp(disparity, (height, width))
    depth = depth.squeeze().detach().cpu().numpy().astype(np.float32)
    depth *= float(scale)
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[(depth < 0.05) | (depth > 10.0)] = 0.0
    return depth


def grasp_rows(grasps, limit=5):
    rows = []
    for grasp in np.asarray(grasps).reshape(-1, 17)[:int(limit)]:
        rows.append({
            "score": round(float(grasp[0]), 6),
            "width_mm": round(float(grasp[1]) * 1000.0, 2),
            "center_mm": [round(float(value) * 1000.0, 2)
                          for value in grasp[13:16]],
        })
    return rows


def grasp_errors(candidate, reference):
    if len(candidate) == 0 and len(reference) == 0:
        return None
    if len(candidate) == 0 or len(reference) == 0:
        return {"center_m": math.inf, "rotation_deg": math.inf, "width_m": math.inf}
    left = candidate[0]
    right = reference[0]
    center = float(np.linalg.norm(left[13:16] - right[13:16]))
    rotation_left = left[4:13].reshape(3, 3)
    rotation_right = right[4:13].reshape(3, 3)
    cosine = (np.trace(rotation_left.T @ rotation_right) - 1.0) / 2.0
    rotation_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    width = float(abs(left[1] - right[1]))
    return {
        "center_m": center,
        "rotation_deg": rotation_deg,
        "width_m": width,
    }


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("prompts_json")
    parser.add_argument(
        "--validation-dir",
        default=os.environ.get(
            "YOLOE_VALIDATION_DIR",
            str(ROOT / "model" / "validation" / "yoloe"),
        ),
    )
    parser.add_argument(
        "--camera-k", nargs=4, type=float, required=True,
        metavar=("FX", "FY", "CX", "CY"),
    )
    parser.add_argument("--model", default=YOLOE_MODEL)
    parser.add_argument("--litemono-home", default=LITEMONO_HOME)
    parser.add_argument("--litemono-weights", default=LITEMONO_WEIGHTS)
    parser.add_argument("--litemono-model", default=LITEMONO_MODEL)
    args = parser.parse_args(argv)

    prompts = read_prompt_file(args.prompts_json)
    paths = validation_paths(prompts, args.validation_dir)
    if Path(args.model).resolve() != Path(YOLOE_MODEL).resolve():
        raise RuntimeError(
            "--model must match the checkpoint configured for the active artifact"
        )
    catalog = PromptCatalog.load(verify_engine=True)
    if [(p["id"], p["text"]) for p in catalog.prompts] != [
        (p["id"], p["text"]) for p in prompts
    ]:
        raise RuntimeError(
            "active YOLOE artifact does not match prompts.json; "
            "run preprocess_prompt.sh first"
        )
    K = camera_matrix(args.camera_k)

    from grasppose.config import LITEMONO_DEPTH_SCALE
    from grasppose.facade import DEFAULT_SERVICE

    core = DEFAULT_SERVICE.core
    core.load()
    candidate_data = {}
    for prompt in prompts:
        image = np.asarray(Image.open(paths[prompt["id"]]).convert("RGB"))
        vision = core._vision.predict(image, prompt["id"])
        mask = np.asarray(vision.segmentation.mask, bool)
        if not mask.any():
            raise RuntimeError(
                "TensorRT YOLOE found no mask for prompt ID %s" % prompt["id"]
            )
        depth = core._depth.predict(image, camera_K=K).depth
        cloud = depth_to_cloud(depth, K, mask=mask)
        if not len(cloud):
            raise RuntimeError(
                "TensorRT Lite-Mono produced no target cloud for %s"
                % prompt["id"]
            )
        tsdf = core._tsdf_builder.build(
            depth, K, mask=mask, cloud=cloud)
        candidate_data[prompt["id"]] = {
            "mask": mask.copy(),
            "depth": np.asarray(depth, np.float32).copy(),
            "tsdf": tsdf,
        }

    # Release candidate engines before loading the larger PyTorch references.
    core._vision.close()
    core._depth.close()
    core._grasper.close()
    gc.collect()
    _, YOLOE, torch = load_yoloe_api()
    torch.cuda.empty_cache()

    reference_masks = {}
    reference_model = YOLOE(args.model)
    configure_prompt_classes(reference_model, prompts)
    try:
        for prompt in prompts:
            reference = best_mask(
                reference_model,
                bgr_image(paths[prompt["id"]]),
                prompt["class_index"],
                prompt["id"],
            )
            if reference is None or not reference.any():
                raise RuntimeError(
                    "PyTorch FP32 YOLOE found no mask for prompt ID %s"
                    % prompt["id"]
                )
            reference_masks[prompt["id"]] = reference
    finally:
        del reference_model
        gc.collect()
        torch.cuda.empty_cache()

    graph, feed_h, feed_w = load_models(
        Path(args.litemono_home).resolve(),
        Path(args.litemono_weights).resolve(),
        args.litemono_model,
    )
    graph = graph.to("cuda")
    reference_depths = {}
    try:
        for prompt in prompts:
            reference_depths[prompt["id"]] = infer_reference_depth(
                graph,
                paths[prompt["id"]],
                feed_h,
                feed_w,
                LITEMONO_DEPTH_SCALE,
            )
    finally:
        del graph
        gc.collect()
        torch.cuda.empty_cache()

    from grasppose.domain.types import TSDFResult

    reference_tsdfs = {}
    for prompt in prompts:
        prompt_id = prompt["id"]
        depth = reference_depths[prompt_id]
        mask = reference_masks[prompt_id]
        cloud = depth_to_cloud(depth, K, mask=mask)
        if not len(cloud):
            raise RuntimeError(
                "PyTorch depth produced no target cloud for %s" % prompt_id
            )
        reference_tsdfs[prompt_id] = core._tsdf_builder.build(
            depth, K, mask=mask, cloud=cloud)

    core._grasper.load()
    errors = []
    for prompt in prompts:
        prompt_id = prompt["id"]
        data = candidate_data[prompt_id]
        candidate_mask = data["mask"]
        reference_mask = reference_masks[prompt_id]
        iou = mask_iou(reference_mask, candidate_mask)

        reference_depth_map = reference_depths[prompt_id]
        valid = np.isfinite(reference_depth_map) & (reference_depth_map > 0.05)
        if not valid.any():
            raise RuntimeError(
                "PyTorch depth has no valid pixels for %s" % prompt_id
            )
        rel = np.abs(data["depth"][valid] - reference_depth_map[valid]) / np.maximum(
            np.abs(reference_depth_map[valid]), 1e-6)
        depth_p95 = float(np.quantile(rel, 0.95))

        candidate_grasp = core._grasper.predict(data["tsdf"]).graspgroup
        reference_grasp = core._grasper.predict(
            reference_tsdfs[prompt_id]).graspgroup
        grasp = grasp_errors(candidate_grasp, reference_grasp)
        passed = iou >= 0.99 and depth_p95 <= 0.02
        if grasp is not None:
            passed = passed and (
                grasp["center_m"] <= 0.0075
                and grasp["rotation_deg"] <= 10.0
                and grasp["width_m"] <= 0.005
            )
        errors.append(not passed)
        if grasp is None:
            grasp_summary = "both no grasp"
        else:
            grasp_summary = (
                "center=%.2fmm rotation=%.2fdeg width=%.2fmm"
                % (
                    grasp["center_m"] * 1000.0,
                    grasp["rotation_deg"],
                    grasp["width_m"] * 1000.0,
                )
            )
        print(
            "%s: mask_iou=%.5f depth_p95=%.3f%% %s %s"
            % (
                prompt_id,
                iou,
                depth_p95 * 100.0,
                grasp_summary,
                "PASS" if passed else "FAIL",
            )
        )
        if not passed:
            print("  candidate grasps:", grasp_rows(candidate_grasp))
            print("  reference grasps:", grasp_rows(reference_grasp))
            print("  candidate volume origin:",
                  data["tsdf"].T_cam_volume[:3, 3].tolist())
            print("  reference volume origin:",
                  reference_tsdfs[prompt_id].T_cam_volume[:3, 3].tolist())

    core._grasper.close()
    return 1 if any(errors) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(1)
