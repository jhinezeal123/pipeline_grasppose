"""VGN output post-processing independent of TensorRT."""

import numpy as np
from scipy import ndimage
from scipy.spatial.transform import Rotation


def process_vgn(tsdf, qual, rot, width, gaussian_sigma=1.0,
                min_width_vox=1.33, max_width_vox=9.33):
    tsdf = np.asarray(tsdf, np.float32).squeeze()
    qual = np.asarray(qual, np.float32).squeeze().copy()
    rot = np.asarray(rot, np.float32).squeeze()
    width = np.asarray(width, np.float32).squeeze()
    qual = ndimage.gaussian_filter(qual, sigma=gaussian_sigma, mode="nearest")
    outside = tsdf > 0.5
    inside = (tsdf > 1e-3) & (tsdf < 0.5)
    valid = ndimage.binary_dilation(outside, iterations=2, mask=~inside)
    qual[~valid] = 0.0
    qual[(width < min_width_vox) | (width > max_width_vox)] = 0.0
    return qual, rot, width


def vgn_to_graspgroup(tsdf, qual, rot, width, voxel_size, T_cam_volume,
                      threshold=0.90, max_filter_size=4, max_grasps=128,
                      gripper_height=0.02, grasp_depth=0.04):
    qual, rot, width = process_vgn(tsdf, qual, rot, width)
    qual[qual < float(threshold)] = 0.0
    maxima = ndimage.maximum_filter(qual, size=int(max_filter_size))
    mask = (qual > 0.0) & (qual == maxima)
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return np.zeros((0, 17), np.float64)
    scores = qual[tuple(indices.T)]
    order = np.argsort(-scores)[:int(max_grasps)]
    indices, scores = indices[order], scores[order]
    T = np.asarray(T_cam_volume, np.float64).reshape(4, 4)
    R_cam_vol, t_cam_vol = T[:3, :3], T[:3, 3]
    gg = np.zeros((len(indices), 17), np.float64)
    for n, (i, j, k) in enumerate(indices):
        q = np.asarray(rot[:, i, j, k], np.float64)
        norm = np.linalg.norm(q)
        R_vol = np.eye(3) if norm < 1e-8 else Rotation.from_quat(q / norm).as_matrix()
        p_vol = np.array([i, j, k], np.float64) * float(voxel_size)
        gg[n, 0] = float(scores[n])
        gg[n, 1] = float(width[i, j, k]) * float(voxel_size)
        gg[n, 2] = float(gripper_height)
        gg[n, 3] = float(grasp_depth)
        gg[n, 4:13] = (R_cam_vol @ R_vol).reshape(-1)
        gg[n, 13:16] = R_cam_vol @ p_vol + t_cam_vol
        gg[n, 16] = -1.0
    return gg
