"""Runtime helpers for logging and explicit CUDA resource cleanup."""

import gc
import traceback


def log(message):
    print("[pipeline] %s" % message, flush=True)


def log_exception(message):
    """Log the active exception with traceback without swallowing it."""
    log(message)
    traceback.print_exc()


def log_vram(tag=""):
    try:
        import torch
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / 2**20
        reserved = torch.cuda.memory_reserved() / 2**20
        log("VRAM%s: allocated %.0f MB / reserved %.0f MB" % (
            tag, allocated, reserved))
    except Exception:
        pass


def release_attributes(obj, *attribute_names):
    """Drop owned model references, then release cached CUDA memory."""
    for name in attribute_names:
        setattr(obj, name, None)
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def cuda_available():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False
