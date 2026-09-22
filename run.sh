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

# Bootstrap GIAI DOAN 1: kiem tra host (torch/torchvision/numpy/CUDA) va tao .venv.
# Chi tai huggingface_hub neu thieu. Phat hien loi host truoc khi tai model.
# Giai doan 2 (cai requirements.txt) nam SAU buoc tai model — xem 'BUOC 5b'.
HOST_PYTHON="$(command -v "${PYTHON:-python3}")"
"$HOST_PYTHON" "$SCRIPT_DIR/env/setup_env.py"
PYTHON="$SCRIPT_DIR/.venv/bin/python"
export PATH="$SCRIPT_DIR/.venv/bin:/usr/local/cuda/bin:$PATH"

# -----------------------------------------------------------------------------
# BUOC 3: doc hf_token (neu co) -> export HF_TOKEN.
# File hf_token de TRONG cung chay duoc, nhung model tren HuggingFace se tai
# cham hon / co the bi gioi han toc do. Token KHONG bao gio bi in ra man hinh.
# -----------------------------------------------------------------------------
HF_TOKEN="${HF_TOKEN:-}"
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
MODELS_TSV="$("$PYTHON" - <<'PY'
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
    # MD5 tuy chon: co thi phai khop, khong thi bo qua (truong rong).
    md5 = b.get('MD5', '')
    revision = b.get('REVISION', '-')
    print('\t'.join([kind, name, src, dest, md5 or '-', revision]))
PY
)"

# Bo moi ky tu CR: neu 'dependencies' bi luu kieu Windows (CRLF) hoac python tren
# Windows in ra \r\n thi DEST/SRC se dinh '\r' o cuoi -> sai duong dan, khong skip duoc.
MODELS_TSV="${MODELS_TSV//$'\r'/}"

while IFS=$'\t' read -r KIND NAME SRC DEST MD5 REVISION; do
  [ "$MD5" = "-" ] && MD5=""
  mkdir -p "$(dirname "$DEST")"
  case "$KIND" in
    # ------------------------- KIND = hf -------------------------
    hf)
      # Da co san va co it nhat 1 file -> bo qua (idempotent).
      if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null || true)" ]; then
        echo "     DA CO $NAME -> bo qua"
      else
        echo "     [hf]  $NAME: snapshot_download $SRC -> $DEST"
        # huggingface_hub co san tu host (qua system_site_packages) hoac da duoc
        # cai rieng o giai doan 1 — xem env/setup_env.py.
        # Token truyen sang python qua argv (khong nhung vao chuoi lenh).
        "$PYTHON" -c '
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
      if [ "$REVISION" != "-" ]; then
        CURRENT="$(git -C "$DEST" rev-parse HEAD)"
        if [ "$CURRENT" != "$REVISION" ]; then
          if [ -n "$(git -C "$DEST" status --porcelain)" ]; then
            echo "LOI: $DEST co thay doi local; khong ghi de." >&2
            exit 1
          fi
          git -C "$DEST" fetch --depth 1 origin "$REVISION"
          git -C "$DEST" checkout --detach "$REVISION"
        fi
      fi
      ;;
    # ------------------------- KIND = url ------------------------
    url)
      # URL Kaggle (/api/v1/datasets/download/...) tra ve mot file ZIP chua
      # .pth, nen sau khi tai phai giai nen. Cac URL khac tra ve file tran.
      if [ -s "$DEST" ]; then
        # -s = ton tai va > 0 byte (file tai do dang bi cat ngan => tai lai).
        echo "     DA CO $NAME -> bo qua"
      elif [ "$SRC" = "PASTE_URL_HERE" ] || [ -z "$SRC" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "!! CANH BAO: $NAME chua co URL."
        echo "!! Checkpoint RealSense cua GraspNet CO link chinh thuc, nam o muc"
        echo "!! 'Model Weights' trong README cua:"
        echo "!!     https://github.com/graspnet/graspness_unofficial"
        echo "!! File goc ten 'minkuresunet_realsense.tar' (176 MB, chua .pth ben trong)."
        echo "!!"
        echo "!! Link Google Drive KHONG tai truc tiep duoc:"
        echo "!!   - .../file/d/<id>/view chi la trang xem, khong phai link tai;"
        echo "!!   - file > 100 MB con bi chan them buoc 'Virus scan warning';"
        echo "!!   - va rat hay gap 'Quota exceeded' khi nhieu nguoi tai."
        echo "!! Cach gon nhat: tai bang trinh duyet, giai nen, roi copy file"
        echo "!! .pth vao: $DEST"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1
      else
        TMP="$(mktemp -d)"
        RAW="$TMP/download"
        if command -v curl >/dev/null 2>&1; then
          echo "     [url] $NAME: curl -> $DEST"
          curl -L --fail -o "$RAW" "$SRC"
        else
          echo "     [url] $NAME: wget (khong co curl) -> $DEST"
          wget -O "$RAW" "$SRC"
        fi

        # Giai nen neu la zip; neu khong thi dung luon file vua tai.
        if unzip -o -q "$RAW" -d "$TMP/x" 2>/dev/null; then
          FOUND="$(find "$TMP/x" -name '*.pth' -type f | head -1 || true)"
          if [ -z "$FOUND" ]; then
            echo "!! LOI: giai nen $NAME nhung khong thay file .pth nao."
            echo "!!   noi dung: $(ls "$TMP/x" || true)"
            rm -rf "$TMP"; exit 1
          fi
          echo "     [url] giai nen -> $(basename "$FOUND")"
          cp "$FOUND" "$DEST"
        else
          cp "$RAW" "$DEST"
        fi
        rm -rf "$TMP"

        # Kiem md5 neu 'dependencies' co khai bao. Tai hong (HTML thay vi file,
        # file cut ngan, ban ghi sai) se bi bat o day thay vi chet mo ho sau nay.
        if [ -n "${MD5:-}" ]; then
          GOT="$(md5sum "$DEST" | cut -d' ' -f1)"
          if [ "$GOT" != "$MD5" ]; then
            echo "!! LOI: $NAME sai md5."
            echo "!!   mong doi: $MD5"
            echo "!!   nhan duoc: $GOT  ($(wc -c < "$DEST") byte)"
            rm -f "$DEST"; exit 1
          fi
          echo "     [url] md5 khop: $GOT"
        fi
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

# 5b) Bootstrap GIAI DOAN 2: giai phu thuoc requirements.txt vao .venv.
#
# Dat O DAY (sau buoc 4 tai model) chu khong phai luc khoi dong, vi day la buoc
# duy nhat tai hang tram MB tu PyPI. Neu mang/PyPI loi thi `set -e` dung script
# — nhung model da tai xong o buoc 4 van con nguyen, lan chay sau chi viec cai
# tiep. Cai runtime that bai khong xoa cac model da tai.
#
# pip tu bo qua nhung gi da dung -> chay lai la re.
echo "     [5b] pip install -r requirements.txt -> .venv (co host-constraints)"
"$HOST_PYTHON" "$SCRIPT_DIR/env/setup_env.py" --install

# MinkowskiEngine belongs to the repo setup, not the host prerequisites.
"$HOST_PYTHON" "$SCRIPT_DIR/env/install_minkowski.py"

# Kiem chung THAT sau khi cai: hai goi kho nhat phai import duoc. Loi o day thi
# bao ngay, khong de den luc chay inference moi vo.
export PYTHONPATH="$SCRIPT_DIR/model/graspnetAPI_repo:${PYTHONPATH:-}"
"$PYTHON" -c 'from moge.model.v3 import MoGeModel; import MinkowskiEngine; print("MoGe / MinkowskiEngine OK")'

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
  exit 1
elif ls "$GRASPNESS_DIR"/pointnet2/_ext*.so >/dev/null 2>&1; then
  echo "     [5d] pointnet2._ext: DA CO -> bo qua"
else
  # KHONG dung `[ -w ... ]`: tien trinh chay bang root nen phep thu do bao "ghi
  # duoc" ngay ca tren mount chi doc (/kaggle/input). Phai THU GHI THAT.
  if touch "$GRASPNESS_DIR/pointnet2/.write_probe" 2>/dev/null; then
    rm -f "$GRASPNESS_DIR/pointnet2/.write_probe"
    BUILD_DIR="$GRASPNESS_DIR"
  else
    BUILD_DIR="$SCRIPT_DIR/.venv/native/graspness_unofficial"
    mkdir -p "$(dirname "$BUILD_DIR")"
    # [ -L ] : neu lan chay truoc da de lai mot SYMLINK hong thi phai lam lai.
    if [ ! -d "$BUILD_DIR/pointnet2" ] || [ -L "$BUILD_DIR" ]; then
      echo "     [5d] $GRASPNESS_DIR chi doc -> copy sang $BUILD_DIR"
      rm -rf "$BUILD_DIR"
      # '-L' la BAT BUOC, khong phai cho chac: $GRASPNESS_DIR thuong LA MOT
      # SYMLINK (model/graspness_unofficial -> /kaggle/input/...), va 'cp -r'
      # mac dinh COPY CHINH SYMLINK DO chu khong copy noi dung. Ban "sao" khi do
      # van tro vao cho chi doc, va build that bai bang:
      #     error: could not create '...': Read-only file system
      # 'set -e' o dau script: moi lenh co the that bai deu phai duoc chan, neu
      # khong ca script thoat ngay va KHONG con co hoi bao ly do tren anh.
      if cp -rL "$GRASPNESS_DIR" "$BUILD_DIR" 2>/dev/null; then
        chmod -R u+w "$BUILD_DIR" 2>/dev/null || true
      else
        echo "     CANH BAO: copy that bai -> build tai cho (se hong neu chi doc)"
        BUILD_DIR="$GRASPNESS_DIR"
      fi
    fi
    # pipeline.py doc GRASPNESS_HOME luc import -> phai export truoc khi goi python.
    export GRASPNESS_HOME="$BUILD_DIR"
  fi
  echo "     [5d] pointnet2._ext CHUA CO -> bien dich tai $BUILD_DIR (2-4 phut)"
  # nvcc co san o /usr/local/cuda/bin nhung KHONG nam trong PATH mac dinh.
  export PATH="/usr/local/cuda/bin:${PATH}"
  # Khong dat thi torch tu do arch; gap may khong thay GPU se ra danh sach rong
  # va build chet voi IndexError o _get_cuda_arch_flags.
  # '|| true' la bat buoc: duoi 'set -e' thi mot phep gan that bai se giet script
  # ngay tai day (va TORCH_CUDA_ARCH_LIST se khong bao gio duoc dat mac dinh).
  if [ -z "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    DETECTED_ARCH="$("$PYTHON" -c "
import torch
print('%d.%d' % torch.cuda.get_device_capability(0)
      if torch.cuda.is_available() else '7.5')
" 2>/dev/null || true)"
    export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH:-7.5}"
  fi
  echo "     [5d] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  ( cd "$BUILD_DIR/pointnet2" && "$PYTHON" setup.py build_ext --inplace ) \
      >"$SCRIPT_DIR/model/_ext_build.log" 2>&1 || true
  # 'build_ext --inplace' hay hong o buoc copy cuoi: no giai ma ten goi thanh
  # 'pointnet2/' tuong doi voi CWD (da la pointnet2/) nen doi thu muc
  # pointnet2/pointnet2/ khong ton tai. Nhung file .so DA duoc sinh ra roi ->
  # chep thang vao cho ma 'import pointnet2._ext' tim.
  # '|| true' cho CA duong ong: 'find' loi hoac 'head' dong ong som deu lam
  # pipefail tra ve khac 0 du find da tim ra file.
  EXT_SO="$(find "$BUILD_DIR/pointnet2/build" -name "_ext*.so" 2>/dev/null \
            | head -1 || true)"
  if [ -n "$EXT_SO" ] && cp "$EXT_SO" "$BUILD_DIR/pointnet2/" 2>/dev/null; then
    echo "     [5d] XONG: $(basename "$EXT_SO")"
  else
    echo "     CANH BAO: khong bien dich duoc pointnet2._ext."
    echo "              Xem log: model/_ext_build.log"
    exit 1
  fi
fi

"$PYTHON" - "$GRASPNESS_DIR" <<'PY_EXT'
import os
import sys
import torch
sys.path.insert(0, os.environ.get('GRASPNESS_HOME', sys.argv[1]))
try:
    import pointnet2._ext
except (ImportError, OSError) as exc:
    sys.exit(f'pointnet2 ABI/import error: {exc}. Rebuild pointnet2 with the current torch/CUDA.')
PY_EXT

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
  exec "$PYTHON" app.py --port "$PORT"
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
echo "[7/8] Chay: $PYTHON pipeline.py --img \"$IMG\" --out output ..."
echo "--------------------------------------------------------------"

# HF_TOKEN da duoc export o buoc 3 nen huggingface_hub tu doc duoc. Ngoai ra, neu
# pipeline.py CO ho tro flag --token thi truyen them cho chac — grep truoc de khong
# truyen bua mot flag ma pipeline.py khong hieu (argparse se bao loi va dung script).
TOKEN_ARGS=()
if [ -n "$HF_TOKEN" ] && grep -q -- "--token" pipeline.py 2>/dev/null; then
  TOKEN_ARGS=(--token "$HF_TOKEN")
fi

"$PYTHON" pipeline.py --img "$IMG" --out output "${TOKEN_ARGS[@]+"${TOKEN_ARGS[@]}"}" "${@:2}"

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
