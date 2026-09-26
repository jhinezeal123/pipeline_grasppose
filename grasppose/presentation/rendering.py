"""OpenCV-only visualization over typed domain results."""

import numpy as np

from ..modules.grasp.constants import GRIP_HW_OPEN_M, GRIP_MAX_OPEN_M


def _put(image, text, xy=(8, 24), color=(255, 80, 80),
         scale=0.55):
    import cv2
    output = np.asarray(image).copy()
    cv2.putText(
        output,
        str(text),
        tuple(map(int, xy)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        tuple(map(int, color)),
        1,
        cv2.LINE_AA,
    )
    return output


def draw_box(image, detection):
    import cv2
    output = np.asarray(image).copy()
    for box, score, label in zip(
            detection.boxes,
            detection.scores,
            detection.labels):
        x1, y1, x2, y2 = map(int, np.rint(box))
        cv2.rectangle(
            output, (x1, y1), (x2, y2),
            (255, 220, 0), 2)
        cv2.putText(
            output,
            "%s %.2f" % (label, score),
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 220, 0),
            1,
            cv2.LINE_AA,
        )
    if len(detection.boxes) == 0:
        output = _put(
            output, detection.reason or "no detection")
    return output.astype(np.uint8)


def draw_mask(image, segmentation):
    output = np.asarray(image).copy().astype(np.float32)
    mask = np.asarray(segmentation.mask, bool)
    if mask.any():
        overlay = np.zeros_like(output)
        overlay[:, :, 1] = 255
        output[mask] = (
            0.55 * output[mask] +
            0.45 * overlay[mask]
        )
    else:
        output = _put(
            output.astype(np.uint8),
            segmentation.reason or "empty mask",
        ).astype(np.float32)
    return np.clip(output, 0, 255).astype(np.uint8)


def draw_depth(depth_result):
    import cv2
    depth = np.asarray(depth_result.depth, np.float32)
    valid = np.isfinite(depth) & (depth > 0.05)
    if not valid.any():
        return _put(
            np.zeros((*depth.shape, 3), np.uint8),
            depth_result.reason or "no valid depth",
        )

    low, high = np.percentile(depth[valid], [2, 98])
    high = max(float(high), float(low) + 1e-6)
    normalized = np.zeros_like(depth, np.uint8)
    normalized[valid] = np.clip(
        (depth[valid] - low) * 255.0 / (high - low),
        0, 255,
    ).astype(np.uint8)
    return cv2.applyColorMap(
        255 - normalized,
        cv2.COLORMAP_TURBO,
    )[:, :, ::-1]


def grasp_empty_msg(total_count, reason,
                    max_width=GRIP_MAX_OPEN_M):
    if reason:
        return "VGN did not run: %s" % reason
    if total_count == 0:
        return "VGN found no grasp pose"
    return "%d grasps, none <= %.0f mm" % (
        total_count, max_width * 1000)


def hw_open_note(width, hw_open=GRIP_HW_OPEN_M):
    if width > hw_open:
        return "%.1f mm > HW %.1f mm" % (
            width * 1000, hw_open * 1000)
    return ""


def draw_grasp(image, grasp_result, K,
               max_width=GRIP_MAX_OPEN_M, top=1):
    import cv2
    output = np.asarray(image).copy()
    graspgroup = np.asarray(
        grasp_result.graspgroup,
        np.float64,
    ).reshape(-1, 17)

    if len(graspgroup) == 0:
        return _put(
            output,
            grasp_empty_msg(
                0, grasp_result.reason, max_width),
        )

    valid = graspgroup[
        graspgroup[:, 1] <= float(max_width)]
    if len(valid) == 0:
        return _put(
            output,
            grasp_empty_msg(
                len(graspgroup),
                grasp_result.reason,
                max_width,
            ),
        )

    valid = valid[
        np.argsort(-valid[:, 0])[:int(top)]
    ]
    for grasp in valid:
        width = float(grasp[1])
        depth = max(float(grasp[3]), 0.03)
        rotation = grasp[4:13].reshape(3, 3)
        translation = grasp[13:16]
        # VGN frame: jaw width is local Y, fingers reach along local +Z.
        local = np.array([
            [0.0, -width / 2, 0.0],
            [0.0, width / 2, 0.0],
            [0.0, -width / 2, depth],
            [0.0, width / 2, depth],
        ])
        projected = _project(
            local @ rotation.T + translation, K)
        if projected is None:
            continue

        p0, p1, p2, p3 = [
            tuple(map(int, point))
            for point in projected
        ]
        # Draw a parallel-jaw gripper as a readable U: the palm is the
        # crossbar and the two fingers extend toward the open end. A dark
        # outline keeps the silhouette visible over both the object and mask.
        fingers = ((p0, p2), (p1, p3))
        for start, end in ((p0, p1),) + fingers:
            cv2.line(
                output, start, end,
                (18, 18, 18), 10, cv2.LINE_AA)
        cv2.line(
            output, p0, p1,
            (60, 235, 255), 6, cv2.LINE_AA)
        for start, end in fingers:
            cv2.line(
                output, start, end,
                (255, 60, 230), 5, cv2.LINE_AA)
        for tip in (p2, p3):
            cv2.circle(
                output, tip, 7,
                (18, 18, 18), -1, cv2.LINE_AA)
            cv2.circle(
                output, tip, 3,
                (255, 255, 255), -1, cv2.LINE_AA)

        center = _project(
            translation[None], K)
        if center is not None:
            center_xy = tuple(map(int, center[0]))
            cv2.circle(
                output,
                center_xy,
                3,
                (255, 255, 0),
                -1,
            )
            note = hw_open_note(width)
            if note:
                cv2.putText(
                    output,
                    note,
                    (center_xy[0] + 6, center_xy[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 80, 80),
                    1,
                    cv2.LINE_AA,
                )

    return output.astype(np.uint8)


def _project(points, K):
    points = np.asarray(points, np.float64)
    z = points[:, 2]
    if np.any(z <= 1e-6):
        return None

    K = np.asarray(K, np.float64).reshape(3, 3)
    u = K[0, 0] * points[:, 0] / z + K[0, 2]
    v = K[1, 1] * points[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)
