"""Projective single-view TSDF construction used by VGN.

The grid uses the same encoding as the original VGN/Open3D pipeline:
0 means unobserved, observed signed distances are mapped from [-1, 1] to [0, 1],
therefore the observed surface is near 0.5.
"""

import numpy as np

from .geometry import depth_to_cloud


class ProjectiveTSDFBuilder:
    def __init__(self, size_m=0.30, resolution=40, trunc_voxels=4.0):
        self.size_m = float(size_m)
        self.resolution = int(resolution)
        self.voxel_size = self.size_m / self.resolution
        self.truncation = float(trunc_voxels) * self.voxel_size

    def build(self, depth, K, mask=None, cloud=None, T_cam_volume=None):
        depth = np.asarray(depth, np.float32)
        K = np.asarray(K, np.float64).reshape(3, 3)
        h, w = depth.shape
        mask_bool = None if mask is None else np.asarray(mask, bool)

        if cloud is None:
            cloud = depth_to_cloud(depth, K, mask=mask_bool)
        cloud = np.asarray(cloud, np.float32).reshape(-1, 3)
        if len(cloud) == 0:
            raise ValueError("cannot build TSDF from an empty point cloud")

        if T_cam_volume is None:
            T_cam_volume = self._auto_pose(cloud)
        T_cam_volume = np.asarray(T_cam_volume, np.float64).reshape(4, 4)

        r = self.resolution
        idx = np.indices((r, r, r), dtype=np.float32).reshape(3, -1).T
        p_vol = idx * self.voxel_size
        R = T_cam_volume[:3, :3]
        t = T_cam_volume[:3, 3]
        p_cam = p_vol @ R.T + t

        z = p_cam[:, 2]
        valid = z > 1e-6
        u = np.zeros_like(z)
        v = np.zeros_like(z)
        u[valid] = K[0, 0] * p_cam[valid, 0] / z[valid] + K[0, 2]
        v[valid] = K[1, 1] * p_cam[valid, 1] / z[valid] + K[1, 2]
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid &= (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)

        sampled = np.zeros_like(z, dtype=np.float32)
        ids = np.flatnonzero(valid)
        sampled[ids] = depth[vi[ids], ui[ids]]
        valid &= np.isfinite(sampled) & (sampled > 0.05)
        if mask_bool is not None:
            ids = np.flatnonzero(valid)
            keep = np.zeros_like(valid)
            keep[ids] = mask_bool[vi[ids], ui[ids]]
            valid &= keep

        signed = sampled - z.astype(np.float32)
        encoded = np.zeros_like(signed, dtype=np.float32)
        encoded[valid] = 0.5 * (
            np.clip(signed[valid] / self.truncation, -1.0, 1.0) + 1.0
        )
        grid = encoded.reshape(r, r, r)[None]

        return {
            "grid": np.ascontiguousarray(grid, dtype=np.float32),
            "voxel_size": float(self.voxel_size),
            "T_cam_volume": T_cam_volume.astype(np.float32),
            "observed_voxels": int(valid.sum()),
        }

    def _auto_pose(self, cloud):
        lo = np.percentile(cloud, 5.0, axis=0)
        hi = np.percentile(cloud, 95.0, axis=0)
        center = 0.5 * (lo + hi)
        origin = center - self.size_m * 0.5
        T = np.eye(4, dtype=np.float64)
        T[:3, 3] = origin
        return T
