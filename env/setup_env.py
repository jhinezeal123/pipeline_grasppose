#!/usr/bin/env python3
"""Create a repo-local overlay while preserving the host CUDA ABI stack.

Hai giai doan, tach roi co chu y:

    python env/setup_env.py            # giai doan 1: tao .venv + constraints
    python env/setup_env.py --install       # giai doan 2: cai core runtime\n    python env/setup_env.py --install --ui  # core + Gradio web UI

Giai doan 1 tao venv; chi can mang neu thieu huggingface_hub.
Giai doan 2 moi cai runtime day du.
Ly do tach: run.sh goi giai doan 1 TRUOC khi tai model (de phat hien som may
hong), roi tai model, ROI MOI goi giai doan 2. Nho vay mot lan cai that bai vi
mang/PyPI khong lam mat model da tai xong.

`--install` chay lai duoc nhieu lan: pip tu bo qua nhung gi da dung.
"""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / '.venv'


def protected_versions():
    # KHONG ghim minkowskiengine vao host-constraints.txt. MinkowskiEngine co
    # installer rieng (env/install_minkowski.py) va installer do cung truyen
    # '-c host-constraints.txt'; neu host dang co 0.5.3 thi constraints se ghim
    # minkowskiengine==0.5.3, va pip se tu choi bundled wheel 0.5.4 bang
    # ResolutionImpossible — tuc la chinh constraints pha co che fallback.
    # Cac goi con lai van giu vi chung thuoc ABI stack cua host.
    versions = {}
    for dist in metadata.distributions():
        name = dist.metadata['Name'].lower().replace('_', '-')
        if name in {'torch', 'torchvision', 'torchaudio', 'numpy', 'scipy',
                    'triton'} or name.startswith('nvidia-'):
            versions[name] = dist.version
    for name in ('torch', 'torchvision', 'numpy'):
        if name not in versions:
            raise RuntimeError(f'Missing host {name}. Install the matching CUDA stack first; see env/README.md.')
    return versions


def main():
    # --- Giai doan 1: kiem tra host + tao .venv. KHONG dung mang. ---
    versions = protected_versions()
    # Fail before downloads if the native stack itself is broken.
    subprocess.run([sys.executable, '-c',
                    'import torch, torchvision, numpy; '
                    'assert torch.cuda.is_available(), "CUDA GPU is unavailable"'], check=True)
    fingerprint = {'python': sys.version, 'executable': sys.executable,
                   'protected': versions}
    stamp = ENV / 'host.json'
    if ENV.exists():
        if not stamp.exists() or json.loads(stamp.read_text()) != fingerprint:
            raise RuntimeError('Host Python/CUDA packages changed or .venv is unmanaged. '
                               'Move .venv aside and rerun; rebuild pointnet2 for the new host.')
    else:
        # No ensurepip needed, and the notebook's CUDA extensions remain visible.
        venv.EnvBuilder(system_site_packages=True, with_pip=False).create(ENV)
        stamp.write_text(json.dumps(fingerprint, indent=2) + '\n')
    constraints = ENV / 'host-constraints.txt'
    constraints.write_text(''.join(f'{k}=={v}\n' for k, v in sorted(versions.items())))

    if '--install' not in sys.argv:
        # huggingface_hub phai co TRUOC buoc tai model, ma requirements.txt chi
        # duoc cai o giai doan 2 (sau khi tai xong). Tren Kaggle no thuong co san
        # trong host va system_site_packages=True lam no hien ra qua .venv; neu
        # host KHONG co thi tai model se chet ngay dong dau. Cai rieng mot goi
        # nhe nay (~500 KB) ngay bay gio, ghim theo constraints.
        try:
            subprocess.run([str(ENV / 'bin/python'), '-c', 'import huggingface_hub'],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError:
            print('huggingface_hub thieu trong host -> cai vao .venv truoc khi tai model')
            subprocess.run([sys.executable, '-m', 'pip', '--python',
                            str(ENV / 'bin/python'), 'install', '-c', str(constraints),
                            'huggingface_hub==0.24.7'], check=True)
        print(f'.venv ready ({ENV}); run with --install to resolve requirements')
        return

    # --- Giai doan 2: cai core runtime; Gradio chi khi --ui. ---
    # Host pip drives installation into the overlay, never the host interpreter.
    req = ROOT / ('requirements-ui.txt' if '--ui' in sys.argv else 'requirements.txt')
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(ENV / 'bin/python'),
                    'install', '-c', str(constraints), '-r', str(req)],
                   check=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f'Environment setup failed: {exc}')
