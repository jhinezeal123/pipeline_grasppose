#!/usr/bin/env python3
"""Export, build, validate, and select a static Lite-Mono TensorRT engine."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from grasppose.artifacts import atomic_write_json, sha256_file
from grasppose.config import LITEMONO_DEPTH_SCALE
from grasppose.trt_engine import TensorRTEngine


class DepthGraph:
    def __init__(self, encoder, decoder):
        import torch.nn as nn
        self.module = nn
        super().__init__()

    def __new__(cls, encoder, decoder):
        import torch.nn as nn

        class _Graph(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = encoder
                self.decoder = decoder

            def forward(self, image):
                return self.decoder(self.encoder(image))[("disp", 0)]

        return _Graph()


def load_models(home, weights_dir, model_name):
    import torch
    sys.path.insert(0, str(home))
    networks = __import__("networks")
    encoder_checkpoint = torch.load(
        str(weights_dir / "encoder.pth"), map_location="cpu",
        weights_only=False)
    decoder_checkpoint = torch.load(
        str(weights_dir / "depth.pth"), map_location="cpu",
        weights_only=False)
    feed_h = int(encoder_checkpoint["height"])
    feed_w = int(encoder_checkpoint["width"])
    encoder = networks.LiteMono(
        model=model_name, height=feed_h, width=feed_w)
    state = encoder.state_dict()
    encoder.load_state_dict(
        {key: value for key, value in encoder_checkpoint.items()
         if key in state},
        strict=True,
    )
    decoder = networks.DepthDecoder(encoder.num_ch_enc, scales=range(3))
    state = decoder.state_dict()
    decoder.load_state_dict(
        {key: value for key, value in decoder_checkpoint.items()
         if key in state},
        strict=True,
    )
    return DepthGraph(encoder.eval(), decoder.eval()), feed_h, feed_w


def image_tensor(path, height, width, device):
    import torch
    resized = Image.open(path).convert("RGB").resize(
        (width, height), Image.LANCZOS)
    array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1)
    tensor = torch.from_numpy(np.ascontiguousarray(array[None] / 255.0))
    return tensor.to(device)


def depth_from_disp(disparity, out_hw=None):
    import torch.nn.functional as F
    if out_hw:
        disparity = F.interpolate(
            disparity, out_hw, mode="bilinear", align_corners=False)
    min_disp = 1.0 / 100.0
    max_disp = 1.0 / 0.1
    return 1.0 / (min_disp + (max_disp - min_disp) * disparity)


def locate_trtexec():
    configured = os.environ.get("TRTEXEC")
    choices = [
        configured,
        shutil.which("trtexec"),
        "/usr/src/tensorrt/bin/trtexec",
    ]
    for candidate in choices:
        if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("trtexec is required to build Lite-Mono TensorRT engines")


def run_trtexec(trtexec, onnx, engine, use_fp16, log_path, avg_timing=8):
    command = [
        trtexec,
        "--onnx=%s" % onnx,
        "--saveEngine=%s" % engine,
        "--workspace=4096",
        "--avgTiming=%d" % int(avg_timing),
        "--verbose",
    ]
    if use_fp16:
        command.append("--fp16")
    with open(log_path, "w", encoding="utf-8") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if completed.returncode != 0 or not os.path.isfile(engine):
        try:
            tail = Path(log_path).read_text(errors="replace")[-8000:]
        except OSError:
            tail = ""
        raise RuntimeError(
            "trtexec failed for Lite-Mono %s candidate (exit=%s):\n%s"
            % ("FP16" if use_fp16 else "FP32",
               completed.returncode, tail)
        )


def validate_candidate(engine_path, validation_images, graph, feed_h, feed_w):
    import torch
    graph = graph.to("cuda")
    engine = TensorRTEngine(engine_path, label="Lite-Mono candidate").load()
    inputs = [
        image_tensor(path, feed_h, feed_w, "cuda")
        for path in validation_images
    ]
    errors = []
    with torch.inference_mode():
        for path, tensor in zip(validation_images, inputs):
            reference_disp = graph(tensor)
            out_hw = tuple(Image.open(path).size[::-1])
            reference_depth = (
                depth_from_disp(reference_disp, out_hw) * LITEMONO_DEPTH_SCALE
            )
            candidate_disp = engine.infer_cuda(
                tensor.detach().cpu().numpy()
            )["disp"].float()
            candidate_depth = (
                depth_from_disp(candidate_disp, out_hw) * LITEMONO_DEPTH_SCALE
            )
            torch.cuda.synchronize()
            ref = reference_depth.squeeze().float().cpu().numpy()
            got = candidate_depth.squeeze().float().cpu().numpy()
            ref = np.nan_to_num(ref, nan=0.0, posinf=0.0, neginf=0.0)
            got = np.nan_to_num(got, nan=0.0, posinf=0.0, neginf=0.0)
            ref[(ref < 0.05) | (ref > 10.0)] = 0.0
            got[(got < 0.05) | (got > 10.0)] = 0.0
            valid = ref > 0.05
            if not valid.any():
                raise RuntimeError("Lite-Mono validation depth is empty")
            rel = np.abs(got[valid] - ref[valid]) / np.maximum(
                np.abs(ref[valid]), 1e-6)
            image_p95 = float(np.quantile(rel, 0.95))
            errors.append(image_p95)
            print(
                "  %s: p95=%.4f%% valid=%d ref=[%.4f,%.4f]m candidate=[%.4f,%.4f]m"
                % (
                    Path(path).name,
                    image_p95 * 100.0,
                    int(valid.sum()),
                    float(ref[valid].min()),
                    float(ref[valid].max()),
                    float(got[valid].min()),
                    float(got[valid].max()),
                )
            )
    for _ in range(3):
        engine.infer_cuda(inputs[0].detach().cpu().numpy())
    torch.cuda.synchronize()
    timings = []
    for _ in range(20):
        started = time.perf_counter()
        engine.infer_cuda(inputs[0].detach().cpu().numpy())
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - started) * 1000.0)
    engine.close()
    graph.cpu()
    torch.cuda.empty_cache()
    return max(errors), float(np.median(timings))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", default=os.environ.get(
        "LITEMONO_HOME", str(ROOT / "model" / "Lite-Mono")))
    parser.add_argument("--weights", default=os.environ.get(
        "LITEMONO_WEIGHTS", str(ROOT / "model" / "lite-mono")))
    parser.add_argument("--model-name", default=os.environ.get(
        "LITEMONO_MODEL", "lite-mono"))
    parser.add_argument("--artifact-root", default=str(
        ROOT / "model" / "runtime" / "lite-mono"))
    parser.add_argument("--validation-image", action="append", default=None)
    parser.add_argument(
        "--reuse-engine", action="append", default=[], metavar="PRECISION=PATH",
        help="validate existing FP32/FP16 engines instead of rebuilding them",
    )
    args = parser.parse_args(argv)
    reused_engines = {}
    for item in args.reuse_engine:
        try:
            precision, path = item.split("=", 1)
        except ValueError:
            parser.error("--reuse-engine must be PRECISION=PATH")
        if precision not in ("fp32", "fp16") or precision in reused_engines:
            parser.error("--reuse-engine precision must be unique fp32/fp16")
        reused_engines[precision] = Path(path).resolve()
    home = Path(args.home).resolve()
    weights = Path(args.weights).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    validation_images = args.validation_image
    if validation_images is None:
        configured = os.environ.get("LITEMONO_VALIDATION_IMAGES")
        validation_images = (
            configured.split(os.pathsep) if configured
            else [str(ROOT / "artifacts" / "examples" / "bag_input.png")]
        )
    validation_images = [Path(item).resolve() for item in validation_images]
    missing = [str(path) for path in validation_images if not path.is_file()]
    if missing:
        raise RuntimeError("Lite-Mono validation images missing: %s" %
                           ", ".join(missing))
    for name in ("encoder.pth", "depth.pth"):
        if not (weights / name).is_file():
            raise RuntimeError("Lite-Mono checkpoint is missing: %s" %
                               (weights / name))

    weight_hashes = {
        name: sha256_file(str(weights / name))
        for name in ("encoder.pth", "depth.pth")
    }
    validation_hashes = {
        str(path): sha256_file(str(path)) for path in validation_images
    }
    tensorrt_version = __import__("tensorrt").__version__
    identity = {
        "weights": weight_hashes,
        "model_name": args.model_name,
        "validation": validation_hashes,
        "tensorrt": tensorrt_version,
        "graph": "static-192x640-v1",
    }
    artifact_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()
    artifact_root.mkdir(parents=True, exist_ok=True)
    target = artifact_root / artifact_id
    current_path = artifact_root / "CURRENT"
    if target.is_dir():
        manifest_path = target / "manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text())
            engine = target / manifest["engine"]["file"]
            if (manifest.get("artifact_id") == artifact_id
                    and engine.is_file()
                    and sha256_file(str(engine)) == manifest["engine"]["sha256"]):
                temporary = current_path.with_suffix(".tmp")
                temporary.write_text(artifact_id + "\n")
                os.replace(str(temporary), str(current_path))
                print("Lite-Mono TensorRT artifact already prepared:", artifact_id)
                return 0
        raise RuntimeError(
            "existing Lite-Mono artifact is incomplete or checksum-mismatched: %s"
            % target
        )

    staging = Path(tempfile.mkdtemp(
        prefix=".litemono-staging-", dir=str(artifact_root)))
    try:
        import torch
        graph, feed_h, feed_w = load_models(
            home, weights, args.model_name)
        if (feed_h, feed_w) != (192, 640):
            raise RuntimeError(
                "this TensorRT graph is fixed at 192x640, checkpoint is %sx%s"
                % (feed_h, feed_w)
            )
        for name, source in reused_engines.items():
            if not source.is_file():
                raise RuntimeError(
                    "reused %s TensorRT engine is missing: %s"
                    % (name, source)
                )
        reuse_all = set(reused_engines) == {"fp32", "fp16"}
        if not reuse_all:
            dummy = torch.zeros((1, 3, feed_h, feed_w), dtype=torch.float32)
            onnx_path = staging / "litemono.onnx"
            torch.onnx.export(
                graph,
                dummy,
                str(onnx_path),
                opset_version=13,
                input_names=["images"],
                output_names=["disp"],
                dynamic_axes=None,
                do_constant_folding=True,
            )
            trtexec = locate_trtexec()
        else:
            trtexec = None
        candidates = []
        candidate_records = []
        for name, fp16 in (("fp32", False), ("fp16", True)):
            engine_path = staging / ("litemono_%s.engine" % name)
            if name in reused_engines:
                shutil.copy2(str(reused_engines[name]), str(engine_path))
            else:
                run_trtexec(
                    trtexec, str(onnx_path), str(engine_path), fp16,
                    str(staging / ("trtexec_%s.log" % name)),
                    avg_timing=1 if reused_engines else 8,
                )
            p95_error, median_ms = validate_candidate(
                str(engine_path), validation_images, graph, feed_h, feed_w)
            print("%s: depth p95 relative error=%.4f%%, median=%.2f ms"
                  % (name, p95_error * 100.0, median_ms))
            passed = p95_error <= 0.02
            candidate_records.append({
                "precision": name,
                "file": engine_path.name,
                "sha256": sha256_file(str(engine_path)),
                "median_engine_ms": median_ms,
                "passed": passed,
                "p95_relative_error": p95_error,
            })
            if passed:
                candidates.append((median_ms, name, p95_error, engine_path))
        if not candidates:
            raise RuntimeError(
                "no Lite-Mono TensorRT precision met p95 depth error <= 2%%"
            )
        candidates.sort(key=lambda item: item[0])
        _, selected_name, selected_error, selected_path = candidates[0]
        selected_hash = sha256_file(str(selected_path))
        manifest = {
            "schema_version": 1,
            "artifact_id": artifact_id,
            "model": {
                "name": args.model_name,
                "source_revision": _git_revision(home),
            },
            "build": {"tensorrt": tensorrt_version},
            "weights": weight_hashes,
            "validation_images": validation_hashes,
            "input": {"shape": [1, 3, feed_h, feed_w],
                      "hw": [feed_h, feed_w], "color": "RGB"},
            "depth": {
                "min_disp": 0.01,
                "max_disp": 10.0,
                "scale": LITEMONO_DEPTH_SCALE,
                "resize": "PIL Lanczos",
                "depth_resize": "bilinear align_corners=False",
                "max_validation_p95_relative_error": 0.02,
                "selected_validation_p95_relative_error": selected_error,
            },
            "candidates": candidate_records,
            "engine": {
                "precision": selected_name,
                "file": selected_path.name,
                "sha256": selected_hash,
            },
        }
        for temporary_name in ("litemono.onnx", "trtexec_fp32.log",
                               "trtexec_fp16.log"):
            try:
                (staging / temporary_name).unlink()
            except OSError:
                pass
        atomic_write_json(str(staging / "manifest.json"), manifest)
        os.replace(str(staging), str(target))
        temporary = current_path.with_suffix(".tmp")
        temporary.write_text(artifact_id + "\n")
        os.replace(str(temporary), str(current_path))
        print("Selected Lite-Mono TensorRT %s: %s"
              % (selected_name, target / selected_path.name))
    except Exception:
        shutil.rmtree(str(staging), ignore_errors=True)
        raise
    return 0


def _git_revision(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise
