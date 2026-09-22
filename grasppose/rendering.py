"""Rendering/presentation layer.

This module turns inference results into display images. It does not load or run ML models.
"""

import numpy as np

from .config import GRIP_HW_OPEN_M, GRIP_MAX_OPEN_M
from .integrations import _add_sys_path, _load_graspnetapi

def draw_box(image, det):
    """Anh 1/4: anh goc + hop cua Grounding-DINO."""
    import cv2
    im = np.ascontiguousarray(np.asarray(image)[:, :, ::-1].copy())
    for i, (b, s) in enumerate(zip(det["boxes"], det["scores"])):
        x0, y0, x1, y1 = [int(round(v)) for v in b]
        cv2.rectangle(im, (x0, y0), (x1, y1), (0, 140, 255), 4)
        lab = det["labels"][i] if i < len(det["labels"]) else "?"
        _put(im, "DINO: '%s'  score %.3f" % (lab, s))
    if len(det["boxes"]) == 0:
        _put(im, "DINO: khong co hop  (%s)" % (det.get("reason") or "?"))
    return im[:, :, ::-1]


def draw_mask(image, seg):
    """Anh 2/4: anh goc + mat na cua SAM phu len."""
    import cv2
    im = np.ascontiguousarray(np.asarray(image)[:, :, ::-1].copy())
    m = np.asarray(seg["mask"]).astype(bool)
    if m.any():
        a = im.astype(np.float32)
        a[m] = a[m] * 0.45 + np.array([120.0, 255.0, 0.0]) * 0.55
        im = a.astype(np.uint8)
        ys, xs = np.where(m)
        cv2.rectangle(im, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 255), 3)
        _put(im, "SAM mask: %d px (%.1f%%)  IoU %.3f"
             % (m.sum(), 100.0 * m.mean(),
                seg["iou"][seg["best"]] if len(seg["iou"]) else 0.0),
             bg=(90, 180, 0))
    else:
        _put(im, "SAM: khong co mat na  (%s)" % (seg.get("reason") or "?"))
    return im[:, :, ::-1]


def draw_depth(dep):
    """Anh 3/4: DEPTH GOC cua MoGe tren TOAN ANH (khong mask).

    Chuan hoa theo percentile cua TOAN ANH. Chuan hoa theo mask la SAI: vung
    ngoai mask bi bao hoa va anh tro nen khong doc duoc.
    """
    import cv2
    d = np.asarray(dep["depth"], np.float32)
    fin = np.isfinite(d) & (d > 0)
    out = np.zeros((*d.shape, 3), np.uint8)
    if fin.sum() < 10:
        _put(out, "MoGe: khong co do sau hop le")
        return out[:, :, ::-1]
    v0, v1 = np.percentile(d[fin], 2), np.percentile(d[fin], 98)
    dn = np.clip((d - v0) / max(v1 - v0, 1e-6), 0, 1)
    cm = cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    cm[~fin] = (40, 40, 40)
    h, w = d.shape
    bar = np.zeros((58, w, 3), np.uint8)
    bar[12:46] = cv2.applyColorMap(
        np.tile(np.linspace(0, 255, w, dtype=np.uint8), (34, 1)), cv2.COLORMAP_TURBO)
    for i in range(6):
        x = int(i * (w - 1) / 5)
        cv2.line(bar, (x, 42), (x, 50), (255, 255, 255), 1)
        cv2.putText(bar, "%.2f" % (v0 + (v1 - v0) * i / 5),
                    (min(x, max(0, w - 56)), 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (255, 255, 255), 1)
    cv2.putText(bar, "m", (w - 22, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (255, 255, 255), 1)
    cv2.putText(bar, "MoGe depth GOC toan anh | xam = khong hop le",
                (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return np.vstack([cm, bar])[:, :, ::-1]


def grasp_empty_msg(n_raw, reason, max_width):
    """Cau bao khi khong ve duoc tu the nao.

    Tach rieng khoi draw_grasp de TEST duoc: ba nguyen nhan duoi day khac han
    nhau, gop lai lam mot cau (nhu truoc) thi khong phan biet duoc "model nap
    hong" voi "loc qua chat" — da lam mat thoi gian dung mot lan.
    """
    if reason:
        return "grasp: KHONG CHAY DUOC - %s" % reason
    if n_raw == 0:
        return "grasp: khong co tu the nao (mask rong?)"
    return ("grasp: %d tu the nhung TAT CA rong hon %.0f mm"
            % (n_raw, max_width * 1000))


def hw_open_note(width, hw_open=GRIP_HW_OPEN_M):
    """Chu thich them cho mot tu the: no co lot vao KHE MO THAT cua kep khong.

    Nhanh cu trong draw_grasp hoi `g[1] > max_width`, nhung `g` lay tu `sel` da
    loc `<= max_width` — dieu kien do KHONG BAO GIO dung, nen chu "VUOT" la code
    chet. Doi sang so sanh voi GRIP_HW_OPEN_M moi co nghia: `max_width` (mac dinh
    80 mm) la nguong LOC de VE, con GRIP_HW_OPEN_M (69.4 mm) la khe mo THAT cua
    myArm M750. Tu the rong hon 69.4 mm van duoc ve ra cho nguoi dung nhin thay
    (dung y do o comment dau file), nhung can noi ro la kep that khong mo toi.
    """
    if width > hw_open:
        return "  VUOT khe mo that %.0f mm" % (hw_open * 1000)
    return ""


def draw_grasp(image, gg, K, max_width=GRIP_MAX_OPEN_M, top=1, min_sep=0.080,
               reason=None):
    """Anh 4/4: anh goc + tu the gap, ve bang CHINH mesh cua upstream.

    Dung graspnetAPI: GraspGroup.to_open3d_geometry_list() -> plot_gripper_pro_max_wo_side
    (4 hop RONG: ngon trai, ngon phai, thanh noi truoc, duoi -> nhin thang ra chu U;
    mau R=score, G=0, B=1-score).

    Tra ve LineSet chu KHONG phai TriangleMesh, nen phai ve tung CANH bang
    cv2.line. (Ban truoc doc nham geom.triangles cua LineSet -> numpy tra mang
    rong -> chi con lai may vach roi rac trong nhu "cai que".)

    OffscreenRenderer cua Open3D KHONG chay duoc o day (thieu Vulkan/X11), nen
    hinh duoc chieu bang tay. Hinh hoc va mau la cua upstream.
    """
    import cv2

    im = np.ascontiguousarray(np.asarray(image)[:, :, ::-1].copy())
    raw = np.asarray(gg, np.float64).reshape(-1, 17)
    sel = raw[raw[:, 1] <= float(max_width)] if len(raw) else raw
    if len(sel) == 0:
        # Duong THOAT SOM: chi can cv2 de ve chu. KHONG import open3d va KHONG
        # _load_graspnetapi() o day — hai thu do chi can khi that su ve gripper.
        # Truoc day chung chay TRUOC phep kiem tra nay, nen khi GraspNess da fail
        # (khong co tu the nao) ma graspnetAPI/open3d cung thieu thi ham raise
        # them mot lan nua — mat luon anh 4/4, trong khi dang le chi can ve chu.
        _put(im, grasp_empty_msg(len(raw), reason, max_width))
        return im[:, :, ::-1]

    import open3d as o3d                                        # noqa: F401
    _add_sys_path()
    GraspGroup = _load_graspnetapi().GraspGroup
    pick = []
    for i in np.argsort(-sel[:, 0]):
        c = sel[i, 13:16]
        if all(np.linalg.norm(c - sel[j, 13:16]) > min_sep for j in pick):
            pick.append(int(i))
        if len(pick) == top:
            break
    gl = GraspGroup(np.ascontiguousarray(sel)).to_open3d_geometry_list()
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def proj(Pw):
        z = np.maximum(Pw[:, 2], 1e-6)
        return np.stack([fx * Pw[:, 0] / z + cx, fy * Pw[:, 1] / z + cy], -1)

    lines = []
    for rank, gi in enumerate(pick):
        geom = gl[gi]
        V = np.asarray(geom.vertices)
        C = np.asarray(geom.vertex_colors)
        uv = proj(V)
        fin = np.isfinite(uv).all(1)
        # Mesh gripper cua upstream gom 4 hop: ngon trai, ngon phai, thanh noi
        # phia truoc, va duoi -> nhin thang ra CHU U.
        #
        # to_open3d_geometry_list() tra ve LineSet (ban 'wo_side': 4 hop RONG,
        # moi hop 8 dinh / 12 canh), KHONG phai TriangleMesh. Vi vay phai doc
        # geom.lines. Tung ban sua truoc day to np.asarray(geom.triangles) —
        # LineSet khong co truong do, numpy tra ve mang RONG, nen vong lap ve
        # canh khong chay dong nao; thu duy nhat hien len la cac vach do fillPoly
        # sinh ra, va ket qua trong nhu may cai que.
        #
        # Ve DU 12 canh moi hop, dung nhu upstream. Da thu loc bot cho do roi
        # (chi giu duong cheo mat) nhung bo di: o goc nhin nay khong phan biet
        # duoc "canh song song truc" voi "duong cheo mat", nen cach loc do lam
        # mat net that cua hinh. Trung thanh voi upstream quan trong hon.
        if hasattr(geom, "lines") and len(np.asarray(geom.lines)):
            E = np.asarray(geom.lines).reshape(-1, 2)
        else:                                   # phong khi la TriangleMesh
            T = np.asarray(geom.triangles).reshape(-1, 3)
            E = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
        ok = fin[E[:, 0]] & fin[E[:, 1]]
        E = E[ok]
        if len(E):
            # Ve canh XA truoc, GAN sau — de canh gan de len canh xa.
            order = np.argsort(-V[E].mean(1)[:, 2])
            E = E[order]
            # Mau theo score giong upstream (R=score, B=1-score), lam sang hon
            # vi nen anh that khong toi nhu khung mau cua o3d.
            c = np.clip(C[E].mean(1), 0, 1)
            for (a, b), cc in zip(E, c):
                col = (int(min(cc[2] * 1.35, 1) * 255),
                       int(cc[1] * 255),
                       int(min(cc[0] * 1.35, 1) * 255))
                cv2.line(im, tuple(uv[a].astype(int)), tuple(uv[b].astype(int)),
                         col, 2, cv2.LINE_AA)
        g = sel[gi]
        # Kich thuoc chieu ra pixel — de DOI CHIEU bang so, khong doan bang mat.
        span = (uv[fin].max(0) - uv[fin].min(0)) if fin.any() else np.zeros(2)
        zc = float(g[15])
        lines.append("#%d score %.4f  width %.1f mm  z=%.2fm  %dx%d px%s"
                     % (rank + 1, g[0], g[1] * 1000, zc,
                        span[0], span[1],
                        hw_open_note(g[1])))
    # Ghi nhan SAU khi ve xong: _put() xoa dai tren-trai, goi trong vong lap thi
    # nhan sau de nhan truoc, cuoi cung chi con dong cuoi.
    _put(im, lines)
    return im[:, :, ::-1]


def _put(im, text, bg=(255, 255, 255), fg=(0, 0, 0)):
    """Ghi chu o goc tren trai, co nen trang de doc duoc.

    `text` nhan str hoac list[str]; list thi ve thanh nhieu dong.
    """
    import cv2
    lines = [text] if isinstance(text, str) else list(text)
    hgt = 24
    cv2.rectangle(im, (0, 0), (min(im.shape[1], 900), 8 + hgt * len(lines)), bg, -1)
    for i, s in enumerate(lines):
        cv2.putText(im, s, (8, 24 + i * hgt), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, fg, 1, cv2.LINE_AA)


