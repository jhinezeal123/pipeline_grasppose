# Jetson Xavier grasp pipeline (PR7 FP16)

Pipeline: YOLOE-26s TensorRT segmentation → Lite-Mono Tiny TensorRT depth →
TSDF → VGN TensorRT grasp poses. The YOLOE engine contains three fixed classes
in this order: `blue cube`, `yellow ball`, `blue cylinder`.

## Target hardware and artifacts

This checkout targets AGX Xavier (`tegra194`), L4T R35.6.4, Python 3.8 and
TensorRT 8.5.2.2. TensorRT engines are tied to this software and device
profile. The three archives in
[the PR7 FP16 release](https://github.com/jhinezeal123/pipeline_grasppose/releases/tag/jetson-xavier-pr7-fp16-bundle-v1)
contain:

- YOLOE-26s FP16 engine at 640×640, batch 1, and its class-order stamp;
- Lite-Mono Tiny FP16 engine at 192×640 and its pinned opset-11 ONNX source;
- the VGN FP16-enabled engine, checkpoint and manifest copied unchanged from
  the [PR8 release](https://github.com/jhinezeal123/pipeline_grasppose/releases/tag/jetson-xavier-trt-bundle-v2).

`dependencies` pins each release URL, the archive SHA-256 and extracted engine
hashes. `prepare.sh` checks the Xavier/TensorRT profile, verifies those hashes,
and reuses files already present with the expected hashes. It builds only the
small Lite-Mono C++ runtime shim. It does not export or build TensorRT engines.

## Prepare, start and infer

```bash
bash prepare.sh
bash cold.sh
bash infer.sh img/frame.png \
  --prompt "blue cube" \
  --camera-k FX FY CX CY
# Copy RUN_ID from the inference response:
bash get_output.sh RUN_ID
bash get_output.sh wait RUN_ID
```

`prepare.sh` creates `.venv` while retaining the JetPack Torch, torchvision,
NumPy and TensorRT packages. It writes the selected artifact paths to
`.venv/runtime.json`. Explicit environment variables still override these
paths; for example, `VGN_ENGINE=/path/to/vgn.engine bash prepare.sh` checks
and remembers the selected VGN engine for subsequent inference.

`cold.sh` starts a local Unix-socket worker, loads all three models and runs
one warmup through each inference path before it reports ready. It is safe to
call `cold.sh` again while the worker is running. Lifecycle commands:

```bash
bash cold.sh status
bash cold.sh stop
bash cold.sh restart
```

Each `infer.sh` call reuses that process and immediately returns a `RUN_ID`,
detection/depth metadata, selected grasp poses and client/worker time. It does
not draw or save images. `get_output.sh RUN_ID` launches a separate renderer
and returns immediately. Use `get_output.sh status RUN_ID` or `wait RUN_ID` to
see when `output/RUN_ID/{box,mask,depthmap,grasp}.png` is ready. The output
renderer does not load any model. `OUTPUT_DIR` can override the repo's `output/`
directory for a different deployment. An unknown prompt, expired `RUN_ID` or absent worker produces a
clear error. Inference does not load a text encoder or call `set_classes()`.
Use `get_output.sh retry RUN_ID` after a failed render job while its snapshot
is still cached.

The worker keeps up to eight recent frame snapshots in RAM, capped at 128 MiB
and ten minutes. Gradio's **Xem bon anh** button uses the same on-demand
renderer. Restarting the worker clears its snapshot cache.

The camera matrix must describe the input image:

```text
K = [[FX, 0, CX], [0, FY, CY], [0, 0, 1]]
```

For the UI, set `CAMERA_K="FX FY CX CY"` before starting the worker. Lite-Mono
is monocular and its metric scale needs camera calibration; set
`LITEMONO_DEPTH_SCALE` before `cold.sh` when needed. After changing model paths
or runtime configuration, run `cold.sh restart`.

## Gradio

```bash
export CAMERA_K="FX FY CX CY"
bash space.sh --host 0.0.0.0 --port 8080
```

Gradio uses the same local worker as `infer.sh` and offers only the three
classes baked into the engine. It displays the grasp metadata first and
generates the four diagnostic images only when requested. `space.sh` starts
the worker if necessary.

## Timing

`cold.sh` pays model loading and first-use initialization once. The `TOTAL`
printed by `infer.sh` starts when the client sends a request and includes
worker communication, image decoding and the full grasp pipeline. Rendering
and writing four PNGs are measured separately by `get_output.sh`.
Compare runs on the same image and prompt after `cold.sh` reports ready;
zero-detection images skip depth and grasp and do not represent full-pipeline
latency.

Before on-demand rendering, on `ktmt` (AGX Xavier, 2026-09-25), 20 consecutive
warm runs on the same `cam2.jpg` with `--prompt "blue cube"` and calibrated K
all returned 2 boxes and a 100,449-pixel mask. Full `infer.sh` process wall
time was 1,563 ms median / 1,655 ms P95.

With on-demand rendering and the faster geometry/client path, 30 warm runs on
the same image returned 2 boxes, a 100,449-pixel mask and 6 grasps every time.
The full `infer.sh` process wall time was 695 ms median / 749 ms P95; the
slowest of those 30 calls took 1,026 ms. When an output renderer was active
concurrently, 10 additional infer calls measured 744 ms median / 784 ms P95.
All four on-demand PNGs matched the old files byte-for-byte on the reference
image. The image renderer finished in about 1.2 seconds separately from
inference.

The Python `pipeline.py` API remains available for direct in-process use and
does not send requests to the worker. Use `infer.sh` or `space.sh` for the
resident worker path.
