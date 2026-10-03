"""Chọn grasp theo width/score; không biết camera, model hay giao thức socket."""

import numpy as np

from .types import GraspPose


class GraspSelector:
    """Policy mặc định. Có thể inject policy khác vào LocalGraspEstimator.

    select() nhận mảng float64 (N,17), trả tuple GraspPose.
    Width tính bằng mét, score cao đứng trước.
    """

    def select(self, graspgroup, max_width, top):
        grasps = graspgroup
        valid = grasps[grasps[:, 1] <= float(max_width)]
        valid = valid[np.argsort(-valid[:, 0])[: int(top)]]
        poses = tuple(
            GraspPose(
                score=float(row[0]),
                width_m=float(row[1]),
                translation_m=tuple(float(value) for value in row[13:16]),
                rotation=tuple(
                    tuple(float(value) for value in axis)
                    for axis in row[4:13].reshape(3, 3)
                ),
            )
            for row in valid
        )

        return poses
