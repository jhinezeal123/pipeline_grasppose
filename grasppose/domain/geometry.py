"""Pure geometry helpers with no model/framework dependencies."""

import os

import numpy as np


def K_from_fovy(fovy_deg, width, height):
    fy = (height / 2.0) / np.tan(np.radians(fovy_deg) / 2.0)
    return np.array(
        [[fy, 0.0, width / 2.0],
         [0.0, fy, height / 2.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def fov_x_from_fovy(fovy_deg, width, height):
    return float(2.0 * np.degrees(np.arctan(
        np.tan(np.radians(fovy_deg) / 2.0) * float(width) / float(height)
    )))


def resolve_camera_intrinsics(camera_K, fov_x, width, height):
    """Resolve a pixel-space 3x3 camera matrix.

    A calibrated camera_K is preferred. CAMERA_K='fx fy cx cy' is the runtime
    fallback; fov_x is kept only for rendered/synthetic images.
    """
    if camera_K is not None:
        K = np.asarray(camera_K, np.float64).reshape(3, 3)
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("camera_K must have positive fx/fy")
        return K

    env = os.environ.get("CAMERA_K", "").strip()
    if env:
        vals = [float(x) for x in env.replace(",", " ").split()]
        if len(vals) != 4:
            raise ValueError("CAMERA_K must be 'fx fy cx cy'")
        fx, fy, cx, cy = vals
        if fx <= 0 or fy <= 0:
            raise ValueError("CAMERA_K must have positive fx/fy")
        return np.array(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            np.float64,
        )

    if fov_x is not None:
        fx = width / (2.0 * np.tan(np.radians(float(fov_x)) / 2.0))
        return np.array(
            [[fx, 0.0, width / 2.0],
             [0.0, fx, height / 2.0],
             [0.0, 0.0, 1.0]],
            np.float64,
        )

    raise ValueError(
        "camera intrinsics are required: pass camera_K, set "
        "CAMERA_K='fx fy cx cy', or pass fov_x"
    )


def depth_range_str(depth, zmin=0.05):
    d = np.asarray(depth, np.float64)
    finite = d[np.isfinite(d) & (d > zmin)]
    if finite.size == 0:
        return "no valid depth (0 px)"
    return "%.3f..%.3f m (%d valid px)" % (
        finite.min(), finite.max(), finite.size
    )


def depth_to_cloud(depth, K, mask=None):
    """Convert metric depth + K to an OpenCV-camera-frame point cloud."""
    depth = np.asarray(depth)
    K = np.asarray(K, np.float64).reshape(3, 3)
    height, width = depth.shape
    ys, xs = np.mgrid[0:height, 0:width]
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0.05)
    if mask is not None:
        valid &= np.asarray(mask, bool)

    zz = z[valid]
    px = (xs[valid] - K[0, 2]) * zz / K[0, 0]
    py = (ys[valid] - K[1, 2]) * zz / K[1, 1]
    return np.stack([px, py, zz], axis=-1).astype(np.float32)


def nms_grasps(graspgroup, translation_threshold=0.03,
               rotation_threshold=np.pi / 6):
    gg = np.asarray(graspgroup)
    if len(gg) == 0:
        return gg
    order = np.argsort(-gg[:, 0])
    translations = gg[:, 13:16]
    rotations = gg[:, 4:13].reshape(-1, 3, 3)
    keep = []
    for index in order:
        duplicate = False
        for kept in keep:
            distance = np.linalg.norm(
                translations[index] - translations[kept])
            angle = np.arccos(np.clip(
                (np.trace(rotations[index].T @ rotations[kept]) - 1) / 2,
                -1, 1,
            ))
            if (distance < translation_threshold and
                    angle < rotation_threshold):
                duplicate = True
                break
        if not duplicate:
            keep.append(index)
    return gg[keep]
