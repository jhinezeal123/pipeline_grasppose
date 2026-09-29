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


def vgn_to_graspgroup(tsdf, quality, rotation, width, voxel_size,
                      T_cam_volume, threshold=0.90,
                      max_filter_size=4, max_grasps=128,
                      gripper_height=0.02, grasp_depth=0.04):
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
