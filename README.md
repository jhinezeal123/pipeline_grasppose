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

Each `infer.sh` call reuses that process and prints the four image paths plus
client and worker time. Outputs are written to `output/<image-stem>_{box,mask,
depthmap,grasp}.png`; set `OUTPUT_DIR` to change the directory. An unknown
prompt or an absent worker produces an error. Inference does not load a text
encoder or call `set_classes()`.

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
classes baked into the engine. `space.sh` starts the worker if necessary.

## Timing

`cold.sh` pays model loading and first-use initialization once. The `TOTAL`
printed by `infer.sh` starts when the client sends a request and includes
worker communication, image decoding, the full pipeline and writing four PNGs.
Compare runs on the same image and prompt after `cold.sh` reports ready;
zero-detection images skip depth and grasp and do not represent full-pipeline
latency.

The Python `pipeline.py` API remains available for direct in-process use and
does not send requests to the worker. Use `infer.sh` or `space.sh` for the
resident worker path.
