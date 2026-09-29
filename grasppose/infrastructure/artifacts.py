"""Checksum helpers for generated runtime artifacts."""

import hashlib
import json
import os
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path, expected, label):
    if not os.path.isfile(path):
        raise RuntimeError("%s is missing: %s" % (label, path))
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(
            "%s checksum mismatch for %s: expected %s, got %s"
            % (label, path, expected, actual)
        )
    return actual


def atomic_write_json(path, value):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
