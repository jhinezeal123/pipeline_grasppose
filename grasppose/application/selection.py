"""Chọn grasp theo width/score; không biết camera, model hay giao thức socket."""

import numpy as np

from .types import GraspPose


def select_grasps(graspgroup, max_width, top):
    valid = graspgroup[graspgroup[:, 1] <= float(max_width)]
    valid = valid[np.argsort(-valid[:, 0])[: int(top)]]
    return tuple(
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


class GraspSelector:
    """Compatibility policy; có thể inject object có select() như trước."""

    def select(self, graspgroup, max_width, top):
        return select_grasps(graspgroup, max_width, top)
