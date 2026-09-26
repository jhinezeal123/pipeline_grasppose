#!/usr/bin/env python3
"""Gradio frontend that sends prompt IDs to the resident local worker."""

import argparse
import os
import sys

import numpy as np
from PIL import Image

from grasppose.infrastructure.worker.client import WorkerGraspEstimator, request_worker
from grasppose.infrastructure.output.control import start as start_output, wait as wait_output

ESTIMATOR = WorkerGraspEstimator()

TOP_GRASPS = 5
PORT_DEFAULT = 8080
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(ROOT, "artifacts", "output"))


def _camera_k():
    values = os.environ.get("CAMERA_K", "").split()
    if not values:
        return None
    if len(values) != 4:
        raise ValueError("CAMERA_K must contain FX FY CX CY")
    fx, fy, cx, cy = map(float, values)
    return [fx, fy, cx, cy]


def _camera_k_size():
    values = os.environ.get("CAMERA_K_SIZE", "").replace(",", " ").split()
    if not values:
        return None
    if len(values) != 2:
        raise ValueError("CAMERA_K_SIZE must contain WIDTH HEIGHT")
    return [int(value) for value in values]


def prompt_choices():
    status = request_worker({"op": "status"}, timeout=2)
    prompts = status.get("prompts", [])
    if not prompts:
        raise RuntimeError("worker has no prepared prompts")
    return [
        ("%s  [%s]" % (item["text"], item["id"]), item["id"])
        for item in prompts
    ]


def run_one(image, prompt_id):
    if image is None:
        return None, None, None, None, None, "Chua co anh."
    if not prompt_id:
        return None, None, None, None, None, "Hay chon prompt ID."
    try:
        estimate = ESTIMATOR.estimate(
            image,
            str(prompt_id),
            camera_K=_camera_k(),
            camera_K_size=_camera_k_size(),
            top=TOP_GRASPS,
        )
        if not estimate.snapshot_available or not estimate.request_id:
            raise RuntimeError("output snapshot was not retained")
        start_output(estimate.request_id, output_dir=OUTPUT_DIR)
        job = wait_output(estimate.request_id)
        if job.get("state") != "done":
            raise RuntimeError(job.get("error", "output render failed"))
        files = job.get("files", [])
        if len(files) != 4:
            raise RuntimeError("output renderer did not produce four images")
        outputs = [
            np.asarray(Image.open(path).convert("RGB"))
            for path in files
        ]
        depth_m = estimate.depth_m
        status = "Prompt ID: %s | infer %.1f ms | render %.1f ms" % (
            prompt_id, estimate.latency_ms or 0.0, job["render_ms"])
        if depth_m is not None:
            status = "Do sau vat: %.3f m | %s" % (depth_m, status)
        else:
            status = "Khong co so do sau hop le | %s" % status
        return outputs[0], outputs[1], outputs[2], outputs[3], depth_m, status
    except Exception as exc:
        return None, None, None, None, None, "LOI: %s: %s" % (
            type(exc).__name__, exc)



def build_ui():
    import gradio as gr

    choices = prompt_choices()

    def refresh_prompt_dropdown():
        current = prompt_choices()
        return gr.update(choices=current, value=current[0][1])
    with gr.Blocks(title="Jetson grasp pipeline") as demo:
        gr.Markdown(
            "# YOLOE TensorRT → Lite-Mono TensorRT → TSDF → VGN TensorRT\n"
            "Chọn prompt ID đã được chuẩn bị trước; inference không nhận prompt tự do."
        )
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="numpy", label="Anh dau vao")
                input_prompt = gr.Dropdown(
                    choices=choices,
                    value=choices[0][1],
                    label="Prompt ID",
                )
                refresh = gr.Button("Lam moi prompts")
                submit = gr.Button("Submit", variant="primary")
            with gr.Column():
                output_depth = gr.Number(
                    label="Do sau vat (m)", precision=3)
                output_status = gr.Textbox(
                    label="Trang thai", interactive=False, lines=3)

        with gr.Row():
            output_box = gr.Image(label="1. YOLOE box", interactive=False)
            output_mask = gr.Image(label="2. YOLOE mask", interactive=False)
            output_depthmap = gr.Image(
                label="3. Lite-Mono depth", interactive=False)
            output_grasp = gr.Image(label="4. VGN grasp", interactive=False)

        demo.load(refresh_prompt_dropdown, outputs=input_prompt)
        refresh.click(refresh_prompt_dropdown, outputs=input_prompt)
        submit.click(
            run_one,
            inputs=[input_image, input_prompt],
            outputs=[
                output_box, output_mask, output_depthmap, output_grasp,
                output_depth, output_status,
            ],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()
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
