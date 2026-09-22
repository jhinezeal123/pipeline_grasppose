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
    # MD5 tuy chon: co thi phai khop, khong thi bo qua (truong rong).
    md5 = b.get('MD5', '')
    print('\t'.join([kind, name, src, dest, md5]))
PY
)"

# Bo moi ky tu CR: neu 'dependencies' bi luu kieu Windows (CRLF) hoac python tren
# Windows in ra \r\n thi DEST/SRC se dinh '\r' o cuoi -> sai duong dan, khong skip duoc.
MODELS_TSV="${MODELS_TSV//$'\r'/}"

while IFS=$'\t' read -r KIND NAME SRC DEST MD5; do
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

# 5.0) LD_LIBRARY_PATH — BAT BUOC, va phai dat TRUOC moi lenh `import` kiem tra.
#
# Tren Kaggle driver nam o /usr/local/nvidia/lib64 nhung thu muc do KHONG nam
# trong ldconfig mac dinh. Thieu bien nay thi:
#   - `torch.cuda.is_available()` -> False du may CO T4;
#   - `import open3d` THAT BAI (no can libcuda), va vi ta kiem tra bang
#     `2>/dev/null` nen no im lang -> bi hieu nham thanh "chua cai open3d",
#     roi tai ve 400 MB vo ich va van khong import duoc.
# Da gap dung the nay: log bao 'THIEU : open3d' trong khi Kaggle co san open3d.
if [ -d /usr/local/nvidia/lib64 ]; then
  export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
  echo "     [5.0] LD_LIBRARY_PATH=/usr/local/nvidia/lib64 (bat buoc cho open3d/torch)"
fi
# CUDA toolkit tren Kaggle nam ngoai PATH mac dinh; can cho buoc 5d (build _ext).
export PATH="/usr/local/cuda/bin:${PATH}"

# 5a) MoGe: cai tu source (khong phai ban PyPI) roi kiem tra import that.
#
# KHONG dung --no-deps tran: da thu va no lam VO MoGe. Bang chung tu log:
#     moge/model/v3.py line 8:  import utils3d_moge as utils3d
#     ModuleNotFoundError: No module named 'utils3d_moge'
#     ... line 10:              import utils3d
#     ModuleNotFoundError: No module named 'utils3d'
# Ba goi duoi day den tu git, KHONG co tren PyPI, nen --no-deps chan luon chung.
#
# Nhung cung KHONG the de pip tu giai phu thuoc: pyproject.toml cua MoGe khai
# bao torch>=2.4 / torchvision>=0.19 / starlette / gradio>=6.0, va trong
# [tool.uv.sources] tro torch vao index "pytorch-cu130" (CUDA 13.0). De pip tu do
# thi no thay torch cu128 cua Kaggle bang ban khac -> driver khong khop.
#
# Cach dung: cai moge voi --no-deps (chi lay chinh no), roi cai TAY dung ba goi
# git ma no can, cung bang --no-deps de khong keo torch moi.
echo "     [5a] pip install -e model/moge_repo --no-deps"
pip install -e model/moge_repo --no-deps 2>&1 | tail -4

# Ba goi git ma moge can, da DOC NGUON de xac nhan chu khong doan:
#   moge/model/v3.py            : import utils3d_moge as utils3d
#   moge/model/modules/sparse_unet.py : from flex_gemm.ops import NeighborCache
#   (goi 'pipeline' khong duoc import truc tiep, nhung nam trong danh sach
#    dependencies cua pyproject.toml — de lai cho day du, cai thieu con hon.)
# Ghim dung commit trong pyproject.toml cua MoGe de ket qua lap lai duoc.
MOGE_GIT_DEPS=(
  "utils3d_moge|utils3d_moge @ git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37"
  "flex_gemm|flex-gemm @ git+https://github.com/JeffreyXiang/FlexGEMM.git@b2fadb29d41846c7981ade6801ffc689fae119cf"
  "pipeline|pipeline @ git+https://github.com/EasternJournalist/pipeline.git@1c511390d90226c00c101f34b84df26a0f8789b4"
)
echo "     [5a] cai 3 goi git ma MoGe can (utils3d_moge / flex_gemm / pipeline)"
for entry in "${MOGE_GIT_DEPS[@]}"; do
  mod="${entry%%|*}"
  dep="${entry#*|}"
  # Da co roi thi bo qua — moi goi nay mat 30-60s de build tu source.
  if python3 -c "import $mod" 2>/dev/null; then
    echo "     [5a]   co san: $mod"
  else
    echo "     [5a]   cai   : $mod"
    pip install --no-deps "$dep" 2>&1 | tail -3
  fi
done

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

# 5b) Cac thu vien con lai, cai vao TRONG REPO (env/lib), KHONG dung he thong.
#
# VI SAO: cach cu dung `pip install` tran da PHA moi truong cua chinh notebook
# chua no. Bang chung do duoc tu log:
#     ERROR: pip's dependency resolver ...
#     google-adk 1.29.0 requires starlette<1.0.0,>=0.49.1,
#     but you have starlette 1.6.0 which is incompatible
# Nen bat ky setup thi nghiem nao khac trong cung session deu bi anh huong.
#
# Cach lam: `pip install --target env/lib` roi dua env/lib len dau PYTHONPATH.
# Da do thuc te: goi nap tu env/lib, con ban he thong van nguyen ven.
#
# CANH BAO da do duoc: --target VAN keo theo phu thuoc moi (cai transforms3d
# cung keo numpy 2.5.3, trong khi he thong co numpy 2.0.2 da kiem chung), va
# gay xung dot "numba requires numpy<2.1, but you have numpy 2.5.3". Vi vay o
# day chi cai nhung goi HE THONG CHUA CO, va ghim version lay tu
# requirements.lock.txt. Goi nao he thong da co thi dung nguyen ban cua no —
# nho vay torch/CUDA/MinkowskiEngine (nang, va phai khop ABI) khong bi dung toi.
ENV_DIR="$SCRIPT_DIR/env/lib"
echo "     [5b] cai thu vien vao env/lib (khong dung he thong)"
mkdir -p "$ENV_DIR"

# Danh sach goi BAT BUOC phai co. Goi nao import duoc roi thi bo qua.
# (ten import | ten goi pip | version da kiem chung hoac rong)
NEED="
numpy|numpy|2.0.2
scipy|scipy|1.16.3
torch|torch|
transformers|transformers|5.0.0
PIL|pillow|11.3.0
cv2|opencv-python-headless|4.13.0.88
open3d|open3d|0.20.0
huggingface_hub|huggingface_hub|1.32.0
transforms3d|transforms3d|0.4.2
"
if [ "$SERVE" = "1" ]; then
  NEED="$NEED
gradio|gradio|6.28.0"
fi

# Loc ra nhung goi con thieu (import that, khong tin danh sach pip).
# KHONG dung 2>/dev/null: nuốt loi that se khien "import loi" bi hieu nham thanh
# "chua cai", roi tai ve 400 MB vo ich. Giu lai thong bao loi de con chan doan.
MISSING=""
while IFS='|' read -r IMP Pkg Ver; do
  [ -z "$IMP" ] && continue
  ERR="$(python3 -c "import $IMP" 2>&1)"
  if [ -z "$ERR" ]; then
    echo "     [5b]   co san: $Pkg"
  else
    if [ -n "$Ver" ]; then
      MISSING="$MISSING $Pkg==$Ver"
    else
      MISSING="$MISSING $Pkg"
    fi
    echo "     [5b]   THIEU : $Pkg"
    # In dong loi dau tien — neu la loi CUDA/thu vien chu khong phai thieu goi
    # thi nhin la biet ngay, khong phai doan.
    echo "$ERR" | tail -3 | sed 's/^/     [5b]     | /'
  fi
done <<< "$NEED"

if [ -n "$MISSING" ]; then
  echo "     [5b] pip install --target env/lib:$MISSING"
  # --no-deps: tranh keo theo ban phu thuoc moi de len ban he thong da kiem chung
  # (da do: cai transforms3d keo numpy 2.5.3, lam vo numba/torch).
  pip install -q --target "$ENV_DIR" --no-deps $MISSING
else
  echo "     [5b] moi thu da co san, khong cai gi"
fi

# env/lib len DAU PYTHONPATH de goi thieu duoc lay tu do; goi he thong van thay
# duoc o phia sau nen torch/CUDA khong bi anh huong.
export PYTHONPATH="$ENV_DIR:$SCRIPT_DIR/model/moge_repo:$SCRIPT_DIR/model/graspnetAPI_repo:${PYTHONPATH:-}"

# Kiem tra that: moi goi bat buoc phai import duoc.
BAD=""
while IFS='|' read -r IMP Pkg Ver; do
  [ -z "$IMP" ] && continue
  python3 -c "import $IMP" 2>/dev/null || BAD="$BAD $Pkg"
done <<< "$NEED"
if [ -n "$BAD" ]; then
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  echo "!! LOI: van khong import duoc:$BAD"
  echo "!! Thu chay tay de xem loi that:"
  echo "!!   PYTHONPATH=$ENV_DIR python3 -c 'import ${BAD## }'"
  echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
  exit 1
fi
echo "     [5b] tat ca thu vien bat buoc: OK"

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
    DETECTED_ARCH="$(python3 -c "
import torch
print('%d.%d' % torch.cuda.get_device_capability(0)
      if torch.cuda.is_available() else '7.5')
" 2>/dev/null || true)"
    export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH:-7.5}"
  fi
  echo "     [5d] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
  ( cd "$BUILD_DIR/pointnet2" && python3 setup.py build_ext --inplace ) \
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
