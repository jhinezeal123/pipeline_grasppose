#!/usr/bin/env python3
"""Gradio UI over the shared grasp service."""

import argparse
import os
import sys
import tempfile

import numpy as np

from grasppose.config import DEFAULT_PROMPT, RUNTIME_DIR, YOLOE_CLASSES
from grasppose.worker_client import infer_image, request_worker

TOP_GRASPS = 5
PORT_DEFAULT = 8080
ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT, "output"))


class WorkerService:
    """Keep Gradio light: the model instances live in cold.sh's worker."""

    def load(self):
        status = request_worker({"op": "status"}, timeout=2)
        if not status.get("ok"):
            raise RuntimeError("inference worker is not ready; run cold.sh")
        return self

    def infer(self, image, prompt, top):
        from PIL import Image

        incoming = os.path.join(RUNTIME_DIR, "incoming")
        os.makedirs(incoming, exist_ok=True)
        fd, image_path = tempfile.mkstemp(
            prefix="gradio-", suffix=".png", dir=incoming)
        os.close(fd)
        try:
            rgb = np.asarray(image)[:, :, :3].astype(np.uint8)
            Image.fromarray(rgb).save(image_path, format="PNG")
            camera_k = os.environ.get("CAMERA_K", "").split()
            if camera_k and len(camera_k) != 4:
                raise ValueError("CAMERA_K must contain FX FY CX CY")
            response = infer_image(
                image_path,
                prompt,
                camera_k=[float(value) for value in camera_k]
                if camera_k else None,
                output_dir=OUTPUT_DIR,
                top=top,
            )
            images = []
            for path in response["files"]:
                with Image.open(path) as rendered:
                    images.append(np.asarray(rendered.convert("RGB")))
            return dict(zip(
                ("box", "mask", "depthmap", "grasp"), images),
                depth_m=response.get("depth_m"),
            )
        finally:
            try:
                os.unlink(image_path)
            except OSError:
                pass


SERVICE = WorkerService()


def run_one(image, prompt):
    if image is None:
        return (
            None, None, None, None, None,
            "Chua co anh.",
        )

    prompt = str(prompt or "").strip() or DEFAULT_PROMPT
    try:
        result = SERVICE.infer(
            image,
            prompt=prompt,
            top=TOP_GRASPS,
        )
    except Exception as exc:
        return (
            None, None, None, None, None,
            "LOI: %s: %s" % (
                type(exc).__name__, exc),
        )

    depth_m = result["depth_m"]
    if depth_m is not None:
        status = "Do sau vat: %.3f m | prompt: %r" % (
            depth_m, prompt)
    else:
        status = (
            "Chay xong nhung KHONG co so do sau hop le "
            "cho vat %r." % prompt
        )

    return (
        result["box"],
        result["mask"],
        result["depthmap"],
        result["grasp"],
        depth_m,
        status,
    )


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Jetson grasp pipeline") as demo:
        gr.Markdown(
            "# YOLOE-26s TensorRT → Lite-Mono TensorRT → TSDF → VGN TensorRT\n"
            "Dat `CAMERA_K=\"fx fy cx cy\"` va hieu chuan "
            "`LITEMONO_DEPTH_SCALE` truoc khi dung depth/grasp "
            "theo don vi met."
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    type="numpy", label="Anh dau vao")
                input_prompt = gr.Dropdown(
                    choices=list(YOLOE_CLASSES),
                    value=DEFAULT_PROMPT,
                    label="Target",
                )
                submit = gr.Button(
                    "Submit", variant="primary")
            with gr.Column():
                output_depth = gr.Number(
                    label="Do sau vat (m)", precision=3)
                output_status = gr.Textbox(
                    label="Trang thai",
                    interactive=False,
                    lines=3,
                )

        with gr.Row():
            output_box = gr.Image(
                label="1. YOLOE box", interactive=False)
            output_mask = gr.Image(
                label="2. YOLOE mask", interactive=False)
            output_depthmap = gr.Image(
                label="3. Lite-Mono depth",
                interactive=False,
            )
            output_grasp = gr.Image(
                label="4. VGN grasp", interactive=False)

        submit.click(
            run_one,
            inputs=[input_image, input_prompt],
            outputs=[
                output_box,
                output_mask,
                output_depthmap,
                output_grasp,
                output_depth,
                output_status,
            ],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port", type=int, default=PORT_DEFAULT)
    parser.add_argument(
        "--host", default="0.0.0.0")
    args = parser.parse_args()

    SERVICE.load()
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
