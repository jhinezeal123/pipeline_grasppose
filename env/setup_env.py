#!/usr/bin/env python3
"""Create a repo-local overlay without replacing the JetPack CUDA stack."""

import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv"


def protected_versions():
    versions = {}
    for dist in metadata.distributions():
        name = dist.metadata["Name"].lower().replace("_", "-")
        if name in {"torch", "torchvision", "torchaudio", "numpy", "scipy", "triton"} or name.startswith("nvidia-"):
            versions[name] = dist.version
    for name in ("torch", "torchvision", "numpy"):
        if name not in versions:
            raise RuntimeError("Missing host %s. Install the JetPack-compatible package first." % name)
    return versions


def _create_overlay():
    try:
        venv.EnvBuilder(system_site_packages=True, with_pip=True).create(ENV)
    except Exception as exc:
        raise RuntimeError("Could not create .venv with pip. Install the OS python3-venv package: %s" % exc)


def main():
    versions = protected_versions()
    subprocess.run([sys.executable, "-c", "import torch, torchvision, numpy; assert torch.cuda.is_available(), 'CUDA GPU is unavailable'"], check=True)
    fingerprint = {"python": sys.version, "executable": sys.executable, "protected": versions}
    stamp = ENV / "host.json"
    if ENV.exists():
        if not stamp.exists() or json.loads(stamp.read_text()) != fingerprint:
            raise RuntimeError("Host Python/CUDA packages changed or .venv is unmanaged. Remove/rename .venv and rerun.")
    else:
        _create_overlay(); stamp.write_text(json.dumps(fingerprint, indent=2) + "\n")
    constraints = ENV / "host-constraints.txt"
    constraints.write_text("".join("%s==%s\n" % (k, v) for k, v in sorted(versions.items())))
    if "--install" not in sys.argv:
        print(".venv ready (%s)" % ENV); return
    python = ENV / "bin" / "python"
    subprocess.run([str(python), "-m", "pip", "install", "-c", str(constraints), "-r", str(ROOT / "requirements.txt")], check=True)


if __name__ == "__main__":
    try: main()
    except (RuntimeError, subprocess.CalledProcessError) as exc: sys.exit("Environment setup failed: %s" % exc)
