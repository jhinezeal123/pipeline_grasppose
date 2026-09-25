#!/usr/bin/env python3
"""Gradio interface for fast inference and optional image rendering."""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

from grasppose.config import DEFAULT_PROMPT, YOLOE_CLASSES
from grasppose.worker_client import infer_image, request_worker


TOP_GRASPS = 5
PORT_DEFAULT = 8080
ROOT = os.path.dirname(os.path.abspath(__file__))


class WorkerService:
    """Models stay in cold.sh's worker; rendering runs only when requested."""

    def load(self):
        status = request_worker({"op": "status"}, timeout=2)
        if not status.get("ok"):
            raise RuntimeError("inference worker is not ready; run cold.sh")
        return self

    def infer(self, image_path, prompt, top):
        if not isinstance(image_path, str) or not os.path.isfile(image_path):
            raise ValueError("input image path is unavailable")
        camera_k = os.environ.get("CAMERA_K", "").split()
        if camera_k and len(camera_k) != 4:
            raise ValueError("CAMERA_K must contain FX FY CX CY")
        return infer_image(
            image_path,
            prompt,
            camera_k=[float(value) for value in camera_k]
            if camera_k else None,
            top=top,
        )

    def output(self, run_id):
        from PIL import Image

        script = os.path.join(ROOT, "get_output.sh")
        subprocess.run(["bash", script, run_id], cwd=ROOT,
                       check=True, capture_output=True, text=True)
        completed = subprocess.run(
            ["bash", script, "wait", run_id], cwd=ROOT,
            check=True, capture_output=True, text=True, timeout=130,
        )
        result = json.loads(completed.stdout)
        if result.get("state") != "done":
            raise RuntimeError(result.get("error", "output render failed"))
        images = []
        for path in result["files"]:
            with Image.open(path) as rendered:
                images.append(np.asarray(rendered.convert("RGB")))
        return images


SERVICE = WorkerService()


def run_one(image_path, prompt):
    if image_path is None:
        return (None, "", None, "", "Chua co anh.",
                None, None, None, None)

    prompt = str(prompt or "").strip() or DEFAULT_PROMPT
    try:
        result = SERVICE.infer(image_path, prompt=prompt, top=TOP_GRASPS)
        run_id = result["run_id"]
        depth_m = result.get("depth_m")
        status = (
            "Xong: %d box, %d grasp | prompt: %r | RUN_ID: %s"
            % (result.get("detection_count", 0),
               result.get("grasp_count", 0), prompt, run_id)
        )
        return (
            run_id, run_id, depth_m,
            json.dumps(result.get("grasps", []), ensure_ascii=False),
            status, None, None, None, None,
        )
    except Exception as exc:
        return (
            None, "", None, "",
            "LOI: %s: %s" % (type(exc).__name__, exc),
            None, None, None, None,
        )


def show_output(run_id):
    if not run_id:
        return (None, None, None, None, "Chua co RUN_ID de xuat anh.")
    try:
        images = SERVICE.output(run_id)
        output_root = os.environ.get("OUTPUT_DIR", os.path.join(ROOT, "output"))
        return tuple(images) + (
            "Da xuat anh vao %s" % os.path.join(output_root, run_id),)
    except Exception as exc:
        return (None, None, None, None,
                "LOI xuat anh: %s: %s" % (type(exc).__name__, exc))


def build_ui():
    import gradio as gr

    with gr.Blocks(title="Jetson grasp pipeline") as demo:
        gr.Markdown(
            "# YOLOE TensorRT → Lite-Mono TensorRT → TSDF → VGN TensorRT\n"
            "Suy luan tra ket qua truoc; bon anh chi duoc tao khi bam "
            "**Xem bon anh**."
        )
        run_state = gr.State()
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(type="filepath", label="Anh dau vao")
                input_prompt = gr.Dropdown(
                    choices=list(YOLOE_CLASSES), value=DEFAULT_PROMPT,
                    label="Target",
                )
                submit = gr.Button("Infer", variant="primary")
                show = gr.Button("Xem bon anh")
            with gr.Column():
                output_run_id = gr.Textbox(label="RUN_ID", interactive=False)
                output_depth = gr.Number(label="Do sau vat (m)", precision=3)
                output_grasps = gr.Textbox(
                    label="Grasp poses", interactive=False, lines=5)
                output_status = gr.Textbox(
                    label="Trang thai", interactive=False, lines=3)

        with gr.Row():
            output_box = gr.Image(label="1. YOLOE box", interactive=False)
            output_mask = gr.Image(label="2. YOLOE mask", interactive=False)
            output_depthmap = gr.Image(
                label="3. Lite-Mono depth", interactive=False)
            output_grasp = gr.Image(label="4. VGN grasp", interactive=False)

        submit.click(
            run_one,
            inputs=[input_image, input_prompt],
            outputs=[run_state, output_run_id, output_depth,
                     output_grasps, output_status,
                     output_box, output_mask, output_depthmap, output_grasp],
        )
        show.click(
            show_output, inputs=[run_state],
            outputs=[output_box, output_mask, output_depthmap,
                     output_grasp, output_status],
        )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=PORT_DEFAULT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    SERVICE.load()
    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=args.host, server_port=args.port, share=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
