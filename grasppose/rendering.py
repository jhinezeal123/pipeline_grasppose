"""Lightweight OpenCV-only visualization helpers."""

import numpy as np
from .config import GRIP_HW_OPEN_M, GRIP_MAX_OPEN_M


def _put(img, text, xy=(8, 24), color=(255, 80, 80), scale=0.55):
    import cv2
    out = np.asarray(img).copy()
    cv2.putText(out, str(text), tuple(map(int, xy)), cv2.FONT_HERSHEY_SIMPLEX,
                float(scale), tuple(map(int, color)), 1, cv2.LINE_AA)
    return out


def draw_box(image, det):
    import cv2
    out = np.asarray(image).copy()
    for box, score, label in zip(det.get("boxes", []), det.get("scores", []), det.get("labels", [])):
        x1, y1, x2, y2 = map(int, np.rint(box))
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 220, 0), 2)
        cv2.putText(out, "%s %.2f" % (label, score), (x1, max(16, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 1, cv2.LINE_AA)
    if len(det.get("boxes", [])) == 0:
        out = _put(out, det.get("reason") or "no detection")
    return out.astype(np.uint8)


def draw_mask(image, seg):
    out = np.asarray(image).copy().astype(np.float32)
    mask = np.asarray(seg.get("mask", np.zeros(out.shape[:2], bool)), bool)
    if mask.any():
        overlay = np.zeros_like(out); overlay[:, :, 1] = 255
        out[mask] = 0.55 * out[mask] + 0.45 * overlay[mask]
    else:
        out = _put(out.astype(np.uint8), seg.get("reason") or "empty mask").astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_depth(dep):
    import cv2
    d = np.asarray(dep.get("depth"), np.float32)
    valid = np.isfinite(d) & (d > 0.05)
    if not valid.any():
        return _put(np.zeros((*d.shape, 3), np.uint8), dep.get("reason") or "no valid depth")
    lo, hi = np.percentile(d[valid], [2, 98]); hi = max(float(hi), float(lo) + 1e-6)
    x = np.zeros_like(d, np.uint8)
    x[valid] = np.clip((d[valid] - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(255 - x, cv2.COLORMAP_TURBO)[:, :, ::-1]


def grasp_empty_msg(n_total, reason, max_width=GRIP_MAX_OPEN_M):
    if reason: return "VGN did not run: %s" % reason
    if n_total == 0: return "VGN found no grasp pose"
    return "%d grasps, none <= %.0f mm" % (n_total, max_width * 1000)


def hw_open_note(width, hw_open=GRIP_HW_OPEN_M):
    return " > hardware %.1f mm" % (hw_open * 1000) if width > hw_open else ""


def draw_grasp(image, graspgroup, K, max_width=GRIP_MAX_OPEN_M, top=1, reason=None):
    import cv2
    out = np.asarray(image).copy(); gg = np.asarray(graspgroup, np.float64).reshape(-1, 17)
    if len(gg) == 0: return _put(out, grasp_empty_msg(0, reason, max_width))
    valid = gg[gg[:, 1] <= float(max_width)]
    if len(valid) == 0: return _put(out, grasp_empty_msg(len(gg), reason, max_width))
    valid = valid[np.argsort(-valid[:, 0])[:int(top)]]
    for g in valid:
        width, depth = float(g[1]), max(float(g[3]), 0.03)
        R, t = g[4:13].reshape(3, 3), g[13:16]
        local = np.array([[-depth, -width/2, 0.0], [-depth, width/2, 0.0],
                          [ depth, -width/2, 0.0], [ depth, width/2, 0.0]])
        uv = _project(local @ R.T + t, K)
        if uv is None: continue
        p0, p1, p2, p3 = [tuple(map(int, p)) for p in uv]
        cv2.line(out, p0, p1, (255, 80, 255), 2, cv2.LINE_AA)
        cv2.line(out, p0, p2, (255, 80, 255), 2, cv2.LINE_AA)
        cv2.line(out, p1, p3, (255, 80, 255), 2, cv2.LINE_AA)
        center = _project(t[None], K)
        if center is not None: cv2.circle(out, tuple(map(int, center[0])), 3, (255, 255, 0), -1)
    return out.astype(np.uint8)


def _project(points, K):
    p = np.asarray(points, np.float64); z = p[:, 2]
    if np.any(z <= 1e-6): return None
    K = np.asarray(K, np.float64).reshape(3, 3)
    u = K[0, 0] * p[:, 0] / z + K[0, 2]; v = K[1, 1] * p[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)
