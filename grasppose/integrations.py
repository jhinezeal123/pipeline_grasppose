"""External import/path integration kept behind one module boundary."""

import importlib
import os
import sys
import types

from .config import HERE, MODEL_DIR

def _load_graspnetapi():
    """Tra ve module graspnetAPI.grasp. Raise RuntimeError neu khong co."""
    import importlib
    import types

    if "graspnetAPI" not in sys.modules:
        repo = os.path.join(MODEL_DIR, "graspnetAPI_repo")
        pkg_dir = os.path.join(repo, "graspnetAPI")
        if not os.path.isdir(pkg_dir):
            raise RuntimeError(
                "khong thay graspnetAPI tai %s — chay run.sh de clone ve" % repo)
        pkg = types.ModuleType("graspnetAPI")
        pkg.__path__ = [pkg_dir]          # >> khong chay __init__.py <<
        pkg.__file__ = os.path.join(pkg_dir, "__init__.py")
        sys.modules["graspnetAPI"] = pkg
        if repo not in sys.path:
            sys.path.insert(0, repo)
    return importlib.import_module("graspnetAPI.grasp")


def _add_sys_path():
    for p in (HERE,
              os.path.join(MODEL_DIR, "graspnetAPI_repo")):
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


