"""Projective single-view TSDF construction used by VGN."""

import numpy as np

from .geometry import depth_to_cloud
from .types import TSDFResult


class ProjectiveTSDFBuilder:
    """Build a 40^3-style VGN TSDF without depending on Open3D."""

    def __init__(self, size_m=0.30, resolution=40, trunc_voxels=4.0):
        self.size_m = float(size_m)
        self.resolution = int(resolution)
        self.voxel_size = self.size_m / self.resolution
        self.truncation = float(trunc_voxels) * self.voxel_size

    def build(self, depth, K, mask=None, cloud=None, T_cam_volume=None):
        depth = np.asarray(depth, np.float32)
        K = np.asarray(K, np.float64).reshape(3, 3)
        height, width = depth.shape
        mask_bool = None if mask is None else np.asarray(mask, bool)

        if cloud is None:
            cloud = depth_to_cloud(depth, K, mask=mask_bool)
        cloud = np.asarray(cloud, np.float32).reshape(-1, 3)
        if len(cloud) == 0:
            raise ValueError("cannot build TSDF from an empty point cloud")

        if T_cam_volume is None:
            T_cam_volume = self._auto_pose(cloud)
        T_cam_volume = np.asarray(
            T_cam_volume, np.float64).reshape(4, 4)

        resolution = self.resolution
        indices = np.indices(
            (resolution, resolution, resolution),
            dtype=np.float32,
        ).reshape(3, -1).T
        points_volume = indices * self.voxel_size
        rotation = T_cam_volume[:3, :3]
        translation = T_cam_volume[:3, 3]
        points_camera = points_volume @ rotation.T + translation

        z = points_camera[:, 2]
        valid = z > 1e-6
        u = np.zeros_like(z)
        v = np.zeros_like(z)
        u[valid] = (
            K[0, 0] * points_camera[valid, 0] / z[valid] + K[0, 2]
        )
        v[valid] = (
            K[1, 1] * points_camera[valid, 1] / z[valid] + K[1, 2]
        )
        ui = np.rint(u).astype(np.int32)
        vi = np.rint(v).astype(np.int32)
        valid &= (
            (ui >= 0) & (ui < width) &
            (vi >= 0) & (vi < height)
        )

        sampled = np.zeros_like(z, dtype=np.float32)
        visible_ids = np.flatnonzero(valid)
        sampled[visible_ids] = depth[vi[visible_ids], ui[visible_ids]]
        valid &= np.isfinite(sampled) & (sampled > 0.05)

        if mask_bool is not None:
            visible_ids = np.flatnonzero(valid)
            keep = np.zeros_like(valid)
            keep[visible_ids] = mask_bool[
                vi[visible_ids], ui[visible_ids]
            ]
            valid &= keep

        signed = sampled - z.astype(np.float32)
        encoded = np.zeros_like(signed, dtype=np.float32)
        encoded[valid] = 0.5 * (
            np.clip(
                signed[valid] / self.truncation, -1.0, 1.0
            ) + 1.0
        )
        grid = encoded.reshape(
            resolution, resolution, resolution)[None]

        return TSDFResult(
            grid=np.ascontiguousarray(grid, dtype=np.float32),
            voxel_size=float(self.voxel_size),
            T_cam_volume=T_cam_volume.astype(np.float32),
            observed_voxels=int(valid.sum()),
        )

    def _auto_pose(self, cloud):
        low = np.percentile(cloud, 5.0, axis=0)
        high = np.percentile(cloud, 95.0, axis=0)
        center = 0.5 * (low + high)
        origin = center - self.size_m * 0.5
        transform = np.eye(4, dtype=np.float64)
        transform[:3, 3] = origin
        return transform
