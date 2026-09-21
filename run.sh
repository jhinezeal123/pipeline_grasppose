#!/usr/bin/env bash
# =============================================================================
# run.sh - "text prompt -> grasp" tren may Linux GPU (Kaggle).
#
# Script nay lam 4 viec, theo dung thu tu:
#   1) Tai cac model trong file 'dependencies' (idempotent: chay lai khong tai lai).
#   2) Cai thu vien python + cai MoGe/GraspNetAPI tu source (git clone).
#   3) Chon anh dau vao (tu img/ hoac tu tham so dong lenh).
#   4) Chay pipeline.py va in ra 4 file ket qua trong output/.
#
# Cach dung:
#   bash run.sh                      # tu lay anh dau tien trong img/
#   bash run.sh img/anh-cua-ban.png  # chi dinh anh
#   bash run.sh img/anh.png --prompt "cai coc"   # cac flag them duoc chuyen tiep
#   bash run.sh --serve              # mo WEB UI (gradio) thay vi chay 1 anh
#   bash run.sh --serve --port 7860  # doi cong (mac dinh 8080)
#
# KHONG co duong dan tuyet doi nao bi hardcode: moi thu tinh theo thu muc script.
# =============================================================================
set -euo pipefail

# -----------------------------------------------------------------------------
# BUOC 1: tim thu muc chua chinh script nay roi cd vao do.
# (nho vay chay tu dau cung dung: bash /duong/dan/run.sh)
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Tren Kaggle, driver NVIDIA nam o /usr/local/nvidia/lib64 va KHONG co trong
# ld.so.conf -> thieu bien nay thi torch.cuda.is_available() tra ve False.
if [ -d /usr/local/nvidia/lib64 ]; then
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
fi

# -----------------------------------------------------------------------------
# BUOC 1b: tach cac co RIENG cua run.sh ra khoi tham so se chuyen cho pipeline.py.
# Phai lam TRUOC buoc 6, vi o do "$1" duoc coi la duong dan anh — neu khong tach
# thi "--serve" se bi hieu la ten file anh.
# -----------------------------------------------------------------------------
SERVE=0
PORT=8080
_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --serve)   SERVE=1; shift ;;
    --port)    PORT="${2:?--port can 1 gia tri}"; shift 2 ;;
    --port=*)  PORT="${1#--port=}"; shift ;;
    *)         _ARGS+=("$1"); shift ;;
  esac
done
# Tra lai phan tham so con lai, de "${1:-...}" va "${@:2}" ben duoi chay nhu cu.
set -- ${_ARGS[@]+"${_ARGS[@]}"}

case "$PORT" in
  ''|*[!0-9]*)
    echo "LOI: --port phai la so nguyen, nhan duoc '$PORT'" >&2
    exit 1 ;;
esac

# -----------------------------------------------------------------------------
# BUOC 2: in header cho nguoi dung biet dang chay o dau.
# -----------------------------------------------------------------------------
echo "=============================================================="
echo " grasp_pipeline_repo - text prompt -> grasp"
echo "--------------------------------------------------------------"
echo "  Script     : $SCRIPT_DIR/run.sh"
echo "  Thu muc    : $SCRIPT_DIR"
echo "  Anh vao    : $SCRIPT_DIR/img"
echo "  Model      : $SCRIPT_DIR/model"
echo "  Ket qua    : $SCRIPT_DIR/output"
echo "  Python     : $(command -v python3 || echo 'KHONG THAY python3')"
echo "  Che do     : $( [ "$SERVE" = "1" ] && echo "SERVE - web UI o cong $PORT" || echo "BATCH - chay 1 anh")"
echo "=============================================================="

# -----------------------------------------------------------------------------
# BUOC 3: doc hf_token (neu co) -> export HF_TOKEN.
# File hf_token de TRONG cung chay duoc, nhung model tren HuggingFace se tai
# cham hon / co the bi gioi han toc do. Token KHONG bao gio bi in ra man hinh.
# -----------------------------------------------------------------------------
HF_TOKEN=""
if [ -s hf_token ]; then
  # -s = file ton tai VA co noi dung (> 0 byte). Xoa khoang trang / xuong dong thua.
  HF_TOKEN="$(tr -d ' \t\r\n' < hf_token)"
fi

if [ -n "$HF_TOKEN" ]; then
  export HF_TOKEN
  echo "[3/8] hf_token: CO"
else
  echo "[3/8] hf_token: KHONG (tai khong dung token)"
fi

# -----------------------------------------------------------------------------
# BUOC 4: doc 'dependencies' bang python3 (khong dung awk/sed cho chac chan)
# va tai tung model. Chia 3 loai theo KIND:
#   hf  -> huggingface snapshot_download
#   git -> git clone --depth 1
#   url -> curl/wget 1 file
# -----------------------------------------------------------------------------
echo "[4/8] Kiem tra / tai model theo file 'dependencies' ..."
echo "--------------------------------------------------------------"

# Parser in ra TSV: KIND <TAB> NAME <TAB> SRC(REPO hoac URL) <TAB> DEST
# Neu parse loi thi set -e se dung script ngay (khong tai thieu model).
MODELS_TSV="$(python3 - <<'PY'
import os
import sys

path = 'dependencies'
if not os.path.isfile(path):
    sys.exit("LOI: khong tim thay file 'dependencies' canh run.sh")

blocks = []
cur = {}
with open(path, encoding='utf-8') as fh:
    for raw in fh:
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        if line == '---':
            if cur:
                blocks.append(cur)
            cur = {}
            continue
        if '=' not in line:
            sys.exit("LOI: dong khong dung dinh dang KEY = VALUE: %r" % raw)
        key, val = line.split('=', 1)
        cur[key.strip().upper()] = val.strip()
if cur:
    blocks.append(cur)

if not blocks:
    sys.exit("LOI: file 'dependencies' khong co block nao")

for b in blocks:
    kind = b.get('KIND', '').lower()
    name = b.get('NAME', '')
    dest = b.get('DEST', '')
    src = b.get('REPO', '') if kind == 'hf' else b.get('URL', '')
    if not (kind and name and dest and src):
        sys.exit("LOI: block thieu KEY bat buoc: %r" % b)
    print('\t'.join([kind, name, src, dest]))
PY
)"

# Bo moi ky tu CR: neu 'dependencies' bi luu kieu Windows (CRLF) hoac python tren
# Windows in ra \r\n thi DEST/SRC se dinh '\r' o cuoi -> sai duong dan, khong skip duoc.
MODELS_TSV="${MODELS_TSV//$'\r'/}"

while IFS=$'\t' read -r KIND NAME SRC DEST; do
  case "$KIND" in
    # ------------------------- KIND = hf -------------------------
    hf)
      # Da co san va co it nhat 1 file -> bo qua (idempotent).
      if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null || true)" ]; then
        echo "     DA CO $NAME -> bo qua"
      else
        echo "     [hf]  $NAME: snapshot_download $SRC -> $DEST"
        # huggingface_hub chac chan phai co truoc khi tai (buoc 5 moi cai hang loat).
        python3 -c "import huggingface_hub" 2>/dev/null || pip install -q huggingface_hub
        # Token truyen sang python qua argv (khong nhung vao chuoi lenh).
        python3 -c '
import sys
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=sys.argv[1],
    local_dir=sys.argv[2],
    token=(sys.argv[3] or None),
)
' "$SRC" "$DEST" "${HF_TOKEN:-}"
        echo "     XONG $NAME"
      fi
      ;;
    # ------------------------- KIND = git ------------------------
    git)
      if [ -d "$DEST/.git" ]; then
        echo "     DA CO $NAME -> bo qua"
      else
        echo "     [git] $NAME: clone $SRC -> $DEST"
        git clone --depth 1 "$SRC" "$DEST"
        echo "     XONG $NAME"
      fi
      ;;
    # ------------------------- KIND = url ------------------------
    url)
      if [ -s "$DEST" ]; then
        # -s = ton tai va > 0 byte (file tai do dang bi cat ngan => tai lai).
        echo "     DA CO $NAME -> bo qua"
      elif [ "$SRC" = "PASTE_URL_HERE" ] || [ -z "$SRC" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! CANH BAO: $NAME chua co URL."
        echo "!! Checkpoint 'graspness_realsense.pth' (epoch=10) cua GraspNet"
        echo "!! KHONG phai tai lieu cong khai, khong co link chinh thuc."
        echo "!! Ban phai tu mo file 'dependencies' va thay dong:"
        echo "!!     URL = PASTE_URL_HERE"
        echo "!! bang link tai cua ban (Kaggle dataset / Google Drive direct link),"
        echo "!! hoac copy tay file vao: $DEST"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
      elif command -v curl >/dev/null 2>&1; then
        echo "     [url] $NAME: curl -> $DEST"
        curl -L --fail -o "$DEST" "$SRC"
        echo "     XONG $NAME"
      else
        echo "     [url] $NAME: wget (khong co curl) -> $DEST"
        wget -O "$DEST" "$SRC"
        echo "     XONG $NAME"
      fi
      ;;
    *)
      echo "LOI: KIND khong hop le '$KIND' (chi nhan hf | git | url)"
      exit 1
      ;;
  esac
done <<< "$MODELS_TSV"

echo "--------------------------------------------------------------"
echo "     TAT CA MODEL DA SAN SANG."

# -----------------------------------------------------------------------------
# BUOC 5: cai thu vien python.
# -----------------------------------------------------------------------------
echo "[5/8] Cai dat thu vien python ..."
echo "--------------------------------------------------------------"

# 5a) MoGe: cai tu source (khong phai ban PyPI) roi kiem tra import that.
echo "     [5a] pip install -e model/moge_repo"
pip install -q -e model/moge_repo

# Cho python tim thay source cua MoGe va GraspNetAPI truoc khi verify / chay pipeline.
export PYTHONPATH="$SCRIPT_DIR/model/moge_repo:$SCRIPT_DIR/model/graspnetAPI_repo:${PYTHONPATH:-}"

echo "     [5a] verify: from moge.model.v3 import MoGeModel"
if ! python3 -c "from moge.model.v3 import MoGeModel; print('moge v3 OK')"; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!! LOI: import MoGe v3 that bai."
  echo "!! Kiem tra: model/moge_repo da clone chua? torch da cai chua?"
  echo "!! Thu chay tay:  PYTHONPATH=model/moge_repo python3 -c \\"
  echo "!!     \"from moge.model.v3 import MoGeModel; print('ok')\""
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 1
fi

# 5b) Cac thu vien con lai. gradio CHI cai khi chay --serve, de duong chay batch
# khong phai keo them ~150 MB (fastapi/uvicorn/pydantic) ma no khong dung den.
GRADIO_PKG=()
if [ "$SERVE" = "1" ]; then
  GRADIO_PKG=(gradio)
  echo "     [5b] (che do --serve nen cai them gradio)"
fi
echo "     [5b] pip install transformers/torch/open3d/opencv/transforms3d/..."
# transforms3d BAT BUOC: graspnetAPI/utils/utils.py can no ngay dong import dau
# ("from transforms3d.euler import euler2mat"). Thieu no thi viec ve tu the gap
# bang mesh upstream se vo luc import, du moi thu khac chay tot.
pip install -q "transformers>=4.40" "torch" open3d opencv-python-headless pillow numpy scipy huggingface_hub timm transforms3d ${GRADIO_PKG[@]+"${GRADIO_PKG[@]}"}

# 5c) Ghi chu co y (de nguoi sau khong mat thoi gian go loi):
echo "     GHI CHU: goi 'graspnetAPI' tren PyPI bi HONG (loi setuptools.extern.six"
echo "              da bi Python moi xoa) -> script nay dung ban git clone trong"
echo "              model/graspnetAPI_repo va them no vao PYTHONPATH."
echo "     GHI CHU: KHONG can cai 'autolab_core' / 'scikit-image' / DexNet."
echo "              pipeline.py nap thang graspnetAPI.grasp va bo qua __init__.py,"
echo "              nen ca chuoi phan DANH GIA (graspnet_eval -> dexnet) khong bi keo theo."

# -----------------------------------------------------------------------------
# BUOC 5d: pointnet2._ext — extension CUDA BAT BUOC phai duoc bien dich.
#
# 'graspness_unofficial/pointnet2/pointnet2_utils.py' co dong:
#     import pointnet2._ext as _ext
# va pointnet2/ chi co MA NGUON (_ext_src/), KHONG co ban dung san o bat ky dau.
# Thieu no thi 'from models.graspnet import GraspNet' nem:
#     ImportError: Could not import _ext module.
# khien GraspNess khong nap duoc => khong ra tu the nao ca. Rat de chan doan
# nham thanh "loc qua chat" hoac "mask rong", nen buoc nay phai nam trong script
# chu khong the la thao tac tay.
#
# model/ thuong la symlink vao /kaggle/input (CHI DOC) nen phai build trong mot
# ban sao ghi duoc, roi tro GRASPNESS_HOME vao ban sao do.
# -----------------------------------------------------------------------------
GRASPNESS_DIR="${GRASPNESS_HOME:-$SCRIPT_DIR/model/graspness_unofficial}"
if [ ! -d "$GRASPNESS_DIR/pointnet2" ]; then
  echo "     [5d] khong thay $GRASPNESS_DIR/pointnet2 -> bo qua"
  echo "          (GraspNess se tu bao ro ly do nay tren anh ket qua)"
elif ls "$GRASPNESS_DIR"/pointnet2/_ext*.so >/dev/null 2>&1; then
  echo "     [5d] pointnet2._ext: DA CO -> bo qua"
else
  # KHONG dung `[ -w ... ]`: tien trinh chay bang root nen phep thu do bao "ghi
  # duoc" ngay ca tren mount chi doc (/kaggle/input). Phai THU GHI THAT.
  if touch "$GRASPNESS_DIR/pointnet2/.write_probe" 2>/dev/null; then
    rm -f "$GRASPNESS_DIR/pointnet2/.write_probe"
    BUILD_DIR="$GRASPNESS_DIR"
  else
    BUILD_DIR="$SCRIPT_DIR/model/graspness_unofficial_build"
    if [ ! -d "$BUILD_DIR/pointnet2" ]; then
      echo "     [5d] $GRASPNESS_DIR chi doc -> copy sang $BUILD_DIR"
      rm -rf "$BUILD_DIR"
      cp -r "$GRASPNESS_DIR" "$BUILD_DIR"
      chmod -R u+w "$BUILD_DIR"
    fi
    # pipeline.py doc GRASPNESS_HOME luc import -> phai export truoc khi goi python.
    export GRASPNESS_HOME="$BUILD_DIR"
  fi
  echo "     [5d] pointnet2._ext CHUA CO -> bien dich tai $BUILD_DIR (2-4 phut)"
  # nvcc co san o /usr/local/cuda/bin nhung KHONG nam trong PATH mac dinh.
  export PATH="/usr/local/cuda/bin:${PATH}"
  # Khong dat thi torch tu do arch; gap may khong thay GPU se ra danh sach rong
  # va build chet voi IndexError o _get_cuda_arch_flags.
  if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    TORCH_CUDA_ARCH_LIST=$(python3 -c "
import torch
print('%d.%d' % torch.cuda.get_device_capability(0)
      if torch.cuda.is_available() else '7.5')
" 2>/dev/null)
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"
  fi
  echo "     [5d] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  ( cd "$BUILD_DIR/pointnet2" && python3 setup.py build_ext --inplace ) \
      >"$SCRIPT_DIR/model/_ext_build.log" 2>&1 || true
  # 'build_ext --inplace' hay hong o buoc copy cuoi: no giai ma ten goi thanh
  # 'pointnet2/' tuong doi voi CWD (da la pointnet2/) nen doi thu muc
  # pointnet2/pointnet2/ khong ton tai. Nhung file .so DA duoc sinh ra roi ->
  # chep thang vao cho ma 'import pointnet2._ext' tim.
  EXT_SO=$(find "$BUILD_DIR/pointnet2/build" -name "_ext*.so" 2>/dev/null | head -1)
  if [ -n "$EXT_SO" ]; then
    cp "$EXT_SO" "$BUILD_DIR/pointnet2/"
    echo "     [5d] XONG: $(basename "$EXT_SO")"
  else
    echo "     CANH BAO: khong bien dich duoc pointnet2._ext."
    echo "              Xem log: model/_ext_build.log"
    echo "              GraspNess se khong nap duoc nhung se bao RO ly do tren anh."
  fi
fi

# -----------------------------------------------------------------------------
# BUOC 5c: NEU la --serve thi mo web UI roi DUNG (khong chay 1 anh nao).
# Dat ngay sau khi model + thu vien da san sang, truoc buoc 6 (chon anh) — vi
# che do nay khong can anh dau vao.
# -----------------------------------------------------------------------------
if [ "$SERVE" = "1" ]; then
  echo "--------------------------------------------------------------"
  echo "[6/8] CHE DO --serve: mo web UI (khong chay 1 anh)."
  echo "     http://<may-chu>:$PORT/"
  echo "     Moi lan bam Submit nap lai 4 model, mat khoang 50 giay."
  echo "     Ctrl-C de dung."
  echo "--------------------------------------------------------------"
  # exec: thay the shell bang app.py -> Ctrl-C di thang toi server, khong de lai
  # tien trinh mo côi.
  exec python3 app.py --port "$PORT"
fi

# -----------------------------------------------------------------------------
# BUOC 6: chon anh dau vao.
# Uu tien tham so 1 cua dong lenh; neu khong co thi lay anh dau tien trong img/.
# -----------------------------------------------------------------------------
IMG="${1:-$(ls img/*.{png,jpg,jpeg} 2>/dev/null | head -1 || true)}"

if [ -z "${IMG:-}" ]; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!! LOI: khong tim thay anh dau vao nao."
  echo "!! Hay copy 1 anh (.png/.jpg/.jpeg) vao: $SCRIPT_DIR/img/"
  echo "!! Hoac chi dinh ro:  bash run.sh img/anh-cua-ban.png"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 1
fi

# -----------------------------------------------------------------------------
# BUOC 7: chay pipeline.
# "$@" : tham so 1 la anh (da tach ra), phan con lai ("${@:2}") la cac flag
#        nhu --prompt ... va duoc chuyen tiep nguyen ven cho pipeline.py.
# -----------------------------------------------------------------------------
mkdir -p output

echo "[6/8] Anh dau vao : $IMG"
echo "[7/8] Chay: python3 pipeline.py --img \"$IMG\" --out output ..."
echo "--------------------------------------------------------------"

# HF_TOKEN da duoc export o buoc 3 nen huggingface_hub tu doc duoc. Ngoai ra, neu
# pipeline.py CO ho tro flag --token thi truyen them cho chac — grep truoc de khong
# truyen bua mot flag ma pipeline.py khong hieu (argparse se bao loi va dung script).
TOKEN_ARGS=()
if [ -n "$HF_TOKEN" ] && grep -q -- "--token" pipeline.py 2>/dev/null; then
  TOKEN_ARGS=(--token "$HF_TOKEN")
fi

python3 pipeline.py --img "$IMG" --out output "${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}" "${@:2}"

# -----------------------------------------------------------------------------
# BUOC 8: bao ket qua - 4 anh mong doi trong output/.
# pipeline.py ghi theo ten: <ten-anh>_box.png / _mask.png / _depthmap.png / _grasp.png
# -----------------------------------------------------------------------------
echo "--------------------------------------------------------------"
echo "[8/8] XONG. Ket qua trong $SCRIPT_DIR/output :"
for kind in box mask depthmap grasp; do
  hit="$(ls -1 output/*${kind}* 2>/dev/null | head -1 || true)"
  if [ -n "$hit" ]; then
    echo "     [ok] $SCRIPT_DIR/$hit"
  else
    echo "     [--] khong thay file '$kind*' trong output/"
  fi
done
echo "     (tat ca file trong output/:)"
ls -1 output/ | sed 's|^|       output/|'
