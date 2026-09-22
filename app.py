#!/usr/bin/env python3
"""app.py - web UI (Gradio) cho pipeline: anh -> 4 anh + 1 con so do sau (met).

Cach dung:
    python3 app.py                 # mo cong 8080
    python3 app.py --port 7860

Tren Kaggle / may chu:
    bash run.sh --serve --port 8080

THIET KE: phan loi (run_one) KHONG import gradio. Nho vay test_app.py kiem tra
duoc toan bo logic ma khong can cai gradio, va gradio chi duoc nap dung luc mo
giao dien. Import gradio o dau file se khien `import app` that bai tren may
khong co gradio -> khong test duoc.

Moi lan bam Submit nap lai ca 4 model trong ~50 giay (thiet ke 3 pha: nap, dung
xong nha ngay, de khong bao gio giu 2 model nang cung luc tren GPU 16 GB).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pipeline as P                                            # noqa: E402

# So tu the gap ve len anh grasp pose. Co dinh (khong lam slider) cho UI gon.
TOP_GRASPS = 5

PORT_DEFAULT = 8080


def run_one(image, prompt):
    """Chay pipeline cho 1 anh. KHONG BAO GIO raise.

    Tham so:
        image  : numpy uint8 (H, W, 3) RGB (gradio tra ve), hoac None
        prompt : str, co the rong

    Tra ve 6 phan tu, dung thu tu nay:
        (anh_box, anh_mask, anh_depthmap, anh_grasp, depth_m, trang_thai)

    Moi anh la numpy uint8 hoac None; depth_m la float hoac None.
    """
    if image is None:
        return (None, None, None, None, None,
                "Chua co anh. Hay tai 1 anh len roi bam Submit.")

    # str() truoc .strip(): prompt co the khong phai chuoi (so, list...) tuy
    # gradio luon gui str - ham nay cam ket KHONG BAO GIO raise nen phai ep kieu.
    prompt = str(prompt or "").strip() or P.DEFAULT_PROMPT

    try:
        r = P.pipeline(image, prompt=prompt, top=TOP_GRASPS)
    except Exception as e:
        # Bat moi thu: thieu trong so, het VRAM, anh hong... UI phai song tiep.
        return (None, None, None, None, None,
                "LOI: %s: %s" % (type(e).__name__, e))

    dm = r["depth_m"]
    if dm is None:
        trang_thai = ("Chay xong nhung KHONG co so do sau: Grounding-DINO khong "
                      "tim thay vat khop prompt %r, hoac mask SAM rong. Xem ly do "
                      "ghi truc tiep tren tung anh." % prompt)
    else:
        trang_thai = "Do sau vat: %.3f m   |   prompt: %r" % (dm, prompt)

    return r["box"], r["mask"], r["depthmap"], r["grasp"], dm, trang_thai


def build_ui():
    """Dung giao dien Gradio. Import gradio o TRONG ham nay (xem docstring dau file)."""
    import gradio as gr

    with gr.Blocks(title="grasp pipeline") as demo:
        gr.Markdown(
            "# Anh -> grasp pose\n"
            "Tai 1 anh len, bam **Submit**. Ket qua gom 4 anh va do sau cua vat "
            "(met).\n\n"
            "Moi lan chay nap lai 4 model, mat khoang **50 giay**."
        )

        with gr.Row():
            with gr.Column():
                inp_img = gr.Image(type="numpy", label="Anh dau vao")
                inp_prompt = gr.Textbox(
                    value=P.DEFAULT_PROMPT, label="Prompt (mo ta vat can gap)")
                btn = gr.Button("Submit", variant="primary")
            with gr.Column():
                out_depth = gr.Number(label="Do sau vat (m)", precision=3)
                out_stat = gr.Textbox(label="Trang thai", interactive=False,
                                      lines=3)

        with gr.Row():
            out_box = gr.Image(label="1. Box (Grounding-DINO)", interactive=False)
            out_mask = gr.Image(label="2. Mask (SAM)", interactive=False)
            out_depthmap = gr.Image(label="3. Depthmap (MoGe)", interactive=False)
            out_grasp = gr.Image(label="4. Grasp pose", interactive=False)

        # Thu tu outputs PHAI khop thu tu tra ve cua run_one().
        btn.click(
            run_one,
            inputs=[inp_img, inp_prompt],
            outputs=[out_box, out_mask, out_depthmap, out_grasp,
                     out_depth, out_stat],
        )
    return demo


def main():
    ap = argparse.ArgumentParser(
        description="Web UI: anh -> 4 anh + do sau cua vat (met)")
    ap.add_argument("--port", type=int, default=PORT_DEFAULT,
                    help="cong de mo (mac dinh %d)" % PORT_DEFAULT)
    ap.add_argument("--host", default="0.0.0.0",
                    help="dia chi lang nghe (mac dinh 0.0.0.0)")
    args = ap.parse_args()

    demo = build_ui()
    # demo.queue() KHONG nhan tham so 'concurrency_limit' — da bi loi that tren
    # Kaggle: "Blocks.queue() got an unexpected keyword argument 'concurrency_limit'".
    # Khong can dat gi ca: tai lieu Gradio noi ro gioi han chay song song MAC DINH
    # cua moi event listener LA 1 ("Defaults to 1 if not set otherwise"), dung y ta.
    # Do la thu BAT BUOC phai co: 2 lan chay song song se OOM T4 16 GB, vi thiet ke
    # 3 pha chi giai phong VRAM khi chay tuan tu.
    demo.queue()
    # share=False la CO Y, khong phai mac dinh: khi chay trong notebook (Kaggle
    # dung %run) gradio tu doan la notebook roi TU BAT share=True, mo mot duong
    # cong khai ra Internet toi may dang chay GPU ma khong co xac thuc nao.
    # Ta da co duong ham rieng (tailcat) nen khong can.
    demo.launch(server_name=args.host, server_port=args.port, share=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
