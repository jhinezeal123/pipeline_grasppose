#!/usr/bin/env python3
"""Create a repo-local overlay while preserving the host CUDA ABI stack."""
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / '.venv'


def protected_versions():
    versions = {}
    for dist in metadata.distributions():
        name = dist.metadata['Name'].lower().replace('_', '-')
        if name in {'torch', 'torchvision', 'torchaudio', 'numpy', 'scipy',
                    'minkowskiengine', 'triton'} or name.startswith('nvidia-'):
            versions[name] = dist.version
    for name in ('torch', 'torchvision', 'numpy', 'minkowskiengine'):
        if name not in versions:
            raise RuntimeError(f'Missing host {name}. Install the matching CUDA stack first; see env/README.md.')
    return versions


def main():
    versions = protected_versions()
    # Fail before downloads if the native stack itself is broken.
    subprocess.run([sys.executable, '-c',
                    'import torch, torchvision, numpy, MinkowskiEngine; '
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
    # The host pip drives installation into the overlay; it does not install into
    # the host interpreter. Requires pip >=22.3 (--python).
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(ENV / 'bin/python'),
                    'install', '-c', str(constraints), '-r', str(ROOT / 'requirements.txt')],
                   check=True)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        sys.exit(f'Environment setup failed: {exc}')
