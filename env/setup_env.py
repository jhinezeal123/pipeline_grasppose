#!/usr/bin/env python3
"""Create a repo-local overlay without replacing the JetPack CUDA stack."""

import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import subprocess
import sys
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / ".venv"
PIP_VERSION = "25.0.1"


RUNTIME_PROTECTED = ("torch", "torchvision", "numpy", "scipy")


def _runtime_version(name):
    """Return the version of the module this interpreter actually imports."""
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        raise RuntimeError(
            "Missing/broken host %s. Install the JetPack-compatible "
            "package first: %s" % (name, exc)
        )
    version = str(getattr(module, "__version__", "")).strip()
    if not version:
        raise RuntimeError(
            "Host %s does not expose __version__" % name)
    return version


def protected_versions():
    # JetPack/Ubuntu may expose duplicate dist-info metadata from /usr/lib
    # and /usr/local. Protect the versions Python actually imports, rather
    # than whichever duplicate metadata entry happens to be iterated last.
    versions = {
        name: _runtime_version(name)
        for name in RUNTIME_PROTECTED
    }

    # Preserve optional CUDA/runtime packages when present, but never let
    # duplicate metadata override the imported core stack above.
    for dist in metadata.distributions():
        raw_name = dist.metadata.get("Name")
        if not raw_name:
            continue
        name = raw_name.lower().replace("_", "-")
        if name in versions:
            continue
        if name in {"torchaudio", "triton"} or name.startswith("nvidia-"):
            versions[name] = dist.version
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
    # Ubuntu 20.04's venv can start with an old pip that does not recognize
    # newer manylinux/aarch64 wheel tags. Upgrade pip before dependency
    # resolution so compatible ARM64 wheels are selected instead of sdists.
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade",
         "pip==%s" % PIP_VERSION],
        check=True,
    )
    subprocess.run(
        [str(python), "-m", "pip", "install",
         "-c", str(constraints),
         "-r", str(ROOT / "requirements.txt")],
        check=True,
    )


if __name__ == "__main__":
    try: main()
    except (RuntimeError, subprocess.CalledProcessError) as exc: sys.exit("Environment setup failed: %s" % exc)
