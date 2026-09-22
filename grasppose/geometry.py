"""Pure geometry helpers with no model/framework dependencies."""

import numpy as np

def K_from_fovy(fovy_deg, w, h):
    """Ma tran intrinsics tu truong nhin doc (do) — dung cho anh RENDER.

    fx = fy = (h/2) / tan(fovy/2)   (pixel vuong)
    """
    fy = (h / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array([[fy, 0.0, w / 2.0],
                     [0.0, fy, h / 2.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def fov_x_from_fovy(fovy_deg, w, h):
    """fov_x = 2*atan(tan(fovy/2) * w/h) — tham so MoGe can cho anh render."""
    return float(2.0 * np.degrees(np.arctan(
        np.tan(np.radians(fovy_deg) / 2.0) * float(w) / float(h))))


def depth_range_str(depth, zmin=0.05):
    """Mo ta khoang do sau de IN LOG, chiu duoc mang KHONG co pixel hop le.

    MoGe khong do duoc gi thi tra ve toan so 0 (hoac toan inf/nan). Loc roi moi
    lay min/max la sai: mang rong thi `.min()` nem
    "zero-size array to reduction operation minimum which has no identity",
    va ca pipeline chet ngay o dong log chan doan — dung luc can log nhat.
    O day tra ve CHUOI noi ro la khong do duoc, khong tra so bia.
    """
    d = np.asarray(depth, np.float64)
    fin = d[np.isfinite(d) & (d > zmin)]
    if fin.size == 0:
        return "khong do duoc (0 px hop le)"
    return "%.3f..%.3f m (%d px hop le)" % (fin.min(), fin.max(), fin.size)


def depth_to_cloud(depth, K, mask=None):
    """(H,W) met + K -> point cloud (N,3) he camera OpenCV (x phai, y xuong, z toi).

    mask: neu truyen, CHI lay pixel thuoc mask. Day la duong dung.
          KHONG dung hinh bao (bbox) mo rong: ban tung lam vay va diem cua vat
          the khac trong canh lot vao cloud -> sinh grasp nam ngoai vat.
    """
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05)
    if mask is not None:
        ok &= np.asarray(mask).astype(bool)
    zz = z[ok]
    px = (xs[ok] - K[0, 2]) * zz / K[0, 0]
    py = (ys[ok] - K[1, 2]) * zz / K[1, 1]
    return np.stack([px, py, zz], -1).astype(np.float32)


def nms_grasps(gg, t_th=0.03, r_th=np.pi / 6):
    """NMS cho grasp group: loai grasp trung tam <3cm VA goc xoay <30 do."""
    if len(gg) == 0:
        return gg
    order = np.argsort(-gg[:, 0])
    T = gg[:, 13:16]
    R = gg[:, 4:13].reshape(-1, 3, 3)
    keep = []
    for i in order:
        if all(not (np.linalg.norm(T[i] - T[k]) < t_th and
                    np.arccos(np.clip((np.trace(R[i].T @ R[k]) - 1) / 2, -1, 1)) < r_th)
               for k in keep):
            keep.append(i)
    return gg[keep]


