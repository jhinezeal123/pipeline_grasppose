"""VGN output post-processing independent of TensorRT."""

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation


def process_vgn(tsdf, quality, rotation, width, gaussian_sigma=1.0,
                min_width_vox=1.33, max_width_vox=9.33):
    tsdf = np.asarray(tsdf, np.float32).squeeze()
    quality = np.asarray(quality, np.float32).squeeze().copy()
    rotation = np.asarray(rotation, np.float32).squeeze()
    width = np.asarray(width, np.float32).squeeze()

    quality = ndimage.gaussian_filter(
        quality, sigma=gaussian_sigma, mode="nearest")
    outside = tsdf > 0.5
    inside = (tsdf > 1e-3) & (tsdf < 0.5)
    valid = ndimage.binary_dilation(
        outside, iterations=2, mask=~inside)
    quality[~valid] = 0.0
    quality[(width < min_width_vox) | (width > max_width_vox)] = 0.0
    return quality, rotation, width


def refine_to_surface(tsdf, index, voxel_size, max_shift_voxels=4.0):
    """Move a decoded voxel corner onto the TSDF zero crossing.

    ``ProjectiveTSDFBuilder`` stores ``0.5 * (clip(signed / truncation, -1, 1)
    + 1)`` and leaves unobserved voxels at 0.0, so 0.5 is the observed surface,
    values above it are free space in front of the surface, values below it are
    behind it, and 0.0 is never a surface sample. One Gauss-Newton step on that
    field, with central differences for the gradient, puts the point where the
    trilinear field equals 0.5.

    ``max_shift_voxels`` bounds the step by the truncation band: the field is
    proportional to the signed distance only within ``TSDF_TRUNC_VOXELS``
    (4.0) voxels of the surface and saturates outside it, so a longer step is
    not bracketed. The plain voxel corner is returned whenever the crossing is
    not bracketed this close, the gradient is degenerate, or a sample is
    unobserved or non-finite.
    """
    grid = np.asarray(tsdf, np.float32).squeeze()
    index = np.asarray(index, np.int64).reshape(3)
    corner = index.astype(np.float64) * float(voxel_size)
    if np.any(index < 1) or np.any(index + 2 > grid.shape):
        return corner
    i, j, k = index
    patch = np.asarray(
        grid[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2], np.float64)
    if not np.all(np.isfinite(patch) & (patch > 0.0)):
        return corner
    centre = patch[1, 1, 1]
    gradient = 0.5 * np.array([
        patch[2, 1, 1] - patch[0, 1, 1],
        patch[1, 2, 1] - patch[1, 0, 1],
        patch[1, 1, 2] - patch[1, 1, 0],
    ])
    norm_sq = float(gradient @ gradient)
    if norm_sq < 1e-12:
        return corner
    step = (0.5 - centre) / norm_sq * gradient
    shift = float(np.linalg.norm(step))
    if not np.all(np.isfinite(step)) or shift > max_shift_voxels:
        return corner
    return corner + step * float(voxel_size)


def vgn_to_graspgroup(tsdf, quality, rotation, width, voxel_size,
                      T_cam_volume, threshold=0.90,
                      max_filter_size=4, max_grasps=128,
                      gripper_height=0.02, grasp_depth=0.04,
                      refine_subvoxel=False):
    quality, rotation, width = process_vgn(
        tsdf, quality, rotation, width)
    quality[quality < float(threshold)] = 0.0
    maxima = ndimage.maximum_filter(
        quality, size=int(max_filter_size))
    mask = (quality > 0.0) & (quality == maxima)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return np.zeros((0, 17), np.float64)

    scores = quality[tuple(indices.T)]
    order = np.argsort(-scores)[:int(max_grasps)]
    indices, scores = indices[order], scores[order]

    transform = np.asarray(
        T_cam_volume, np.float64).reshape(4, 4)
    R_cam_volume = transform[:3, :3]
    t_cam_volume = transform[:3, 3]

    graspgroup = np.zeros((len(indices), 17), np.float64)
    for n, (i, j, k) in enumerate(indices):
        quaternion = np.asarray(
            rotation[:, i, j, k], np.float64)
        norm = np.linalg.norm(quaternion)
        R_volume = (
            np.eye(3) if norm < 1e-8
            else Rotation.from_quat(
                quaternion / norm).as_matrix()
        )
        point_volume = np.array(
            [i, j, k], np.float64) * float(voxel_size)
        if refine_subvoxel:
            point_volume = refine_to_surface(
                tsdf, (i, j, k), voxel_size)

        graspgroup[n, 0] = float(scores[n])
        graspgroup[n, 1] = (
            float(width[i, j, k]) * float(voxel_size))
        graspgroup[n, 2] = float(gripper_height)
        graspgroup[n, 3] = float(grasp_depth)
        graspgroup[n, 4:13] = (
            R_cam_volume @ R_volume).reshape(-1)
        graspgroup[n, 13:16] = (
            R_cam_volume @ point_volume + t_cam_volume)
        graspgroup[n, 16] = -1.0

    return graspgroup
