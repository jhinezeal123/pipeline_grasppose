#!/usr/bin/env python3
"""Small Gradio UI for the Jetson pipeline."""
import argparse
import os
import sys
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
import pipeline as P
TOP_GRASPS=5; PORT_DEFAULT=8080


def run_one(image, prompt):
    if image is None: return (None,None,None,None,None,"Chua co anh.")
    prompt=str(prompt or "").strip() or P.DEFAULT_PROMPT
    try: r=P.pipeline(image,prompt=prompt,top=TOP_GRASPS)
    except Exception as e:
        return (None,None,None,None,None,"LOI: %s: %s" % (type(e).__name__,e))
    dm=r["depth_m"]
    status=("Do sau vat: %.3f m | prompt: %r" % (dm,prompt) if dm is not None
            else "Chay xong nhung KHONG co so do sau hop le cho vat %r." % prompt)
    return r["box"],r["mask"],r["depthmap"],r["grasp"],dm,status


def build_ui():
    import gradio as gr
    with gr.Blocks(title="Jetson grasp pipeline") as demo:
        gr.Markdown("# YOLOE-26s → Lite-Mono → TSDF → VGN TensorRT\nDat `CAMERA_K=\"fx fy cx cy\"` va hieu chuan `LITEMONO_DEPTH_SCALE` truoc khi dung depth/grasp theo don vi met.")
        with gr.Row():
            with gr.Column():
                inp_img=gr.Image(type="numpy",label="Anh dau vao"); inp_prompt=gr.Textbox(value=P.DEFAULT_PROMPT,label="Prompt"); btn=gr.Button("Submit",variant="primary")
            with gr.Column():
                out_depth=gr.Number(label="Do sau vat (m)",precision=3); out_stat=gr.Textbox(label="Trang thai",interactive=False,lines=3)
        with gr.Row():
            out_box=gr.Image(label="1. YOLOE box",interactive=False); out_mask=gr.Image(label="2. YOLOE mask",interactive=False)
            out_depthmap=gr.Image(label="3. Lite-Mono depth",interactive=False); out_grasp=gr.Image(label="4. VGN grasp",interactive=False)
        btn.click(run_one,inputs=[inp_img,inp_prompt],outputs=[out_box,out_mask,out_depthmap,out_grasp,out_depth,out_stat])
    return demo


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--port",type=int,default=PORT_DEFAULT); ap.add_argument("--host",default="0.0.0.0"); args=ap.parse_args()
    # Pay model startup cost once; every Submit reuses the same resident instances.
    P.load_models()
    demo=build_ui(); demo.queue(); demo.launch(server_name=args.host,server_port=args.port,share=False); return 0


if __name__ == "__main__": sys.exit(main())
