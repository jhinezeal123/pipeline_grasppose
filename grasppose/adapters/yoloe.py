"""TensorRT YOLOE-26s adapter implementing the vision port."""

import time

import numpy as np

from ..config import (
    YOLOE_CLASSES,
    YOLOE_CONF,
    YOLOE_IMGSZ,
    YOLOE_MODEL,
)
from ..domain.types import (
    DetectionResult,
    SegmentationResult,
    VisionResult,
)
from ..ports.vision import VisionPort
from ..runtime import log, release_attributes


class Yoloe26sVision(VisionPort):
    """Static three-class YOLOE-26s TensorRT segmentation runtime."""

    def __init__(self, model_path=None, device=None, conf=None, imgsz=None):
        self.model_path = model_path or YOLOE_MODEL
        self.device = device
        self.conf = YOLOE_CONF if conf is None else float(conf)
        self.imgsz = YOLOE_IMGSZ if imgsz is None else int(imgsz)
        self.classes = tuple(YOLOE_CLASSES)
        self._class_to_id = {
            name: index for index, name in enumerate(self.classes)
        }
        self._model = None

    def load(self):
        if self._model is None:
            from ultralytics import YOLO

            started = time.time()
            # The engine was exported from YOLOE after set_classes(), so it
            # behaves like a normal static Ultralytics segmentation model.
            self._model = YOLO(self.model_path, task="segment")
            log("YOLOE-26s TensorRT loaded in %.1fs | classes=%r" % (
                time.time() - started,
                self.classes,
            ))
        return self

    def _target_id(self, prompt):
        target = str(prompt).strip()
        if not target:
            target = self.classes[0]
        if target not in self._class_to_id:
            raise ValueError(
                "YOLOE TensorRT prompt must be one of %r; got %r"
                % (self.classes, target)
            )
        return target, self._class_to_id[target]

    def predict(self, image, prompt):
        target, target_id = self._target_id(prompt)
        self.load()

        rgb = np.asarray(image)[:, :, :3]
        height, width = rgb.shape[:2]
        # Ultralytics treats NumPy HWC inputs as OpenCV-style BGR.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])

        predict_kwargs = {
            "source": bgr,
            "conf": self.conf,
            "imgsz": self.imgsz,
            "retina_masks": True,
            "verbose": False,
        }
        if self.device is not None:
            predict_kwargs["device"] = self.device

        result = self._model.predict(**predict_kwargs)[0]

        if result is None or result.boxes is None or len(result.boxes) == 0:
            reason = "YOLOE TensorRT did not find %r" % target
            return VisionResult(
                detection=DetectionResult.empty(reason),
                segmentation=SegmentationResult.empty(
                    height, width, reason),
            )

        class_ids = (
            result.boxes.cls.detach().cpu().numpy()
            .astype(np.int64)
        )
        target_indices = np.flatnonzero(class_ids == target_id)
        if target_indices.size == 0:
            reason = "YOLOE TensorRT did not find %r" % target
            return VisionResult(
                detection=DetectionResult.empty(reason),
                segmentation=SegmentationResult.empty(
                    height, width, reason),
            )

        boxes_all = (
            result.boxes.xyxy.detach().cpu().numpy()
            .astype(np.float32)
        )
        scores_all = (
            result.boxes.conf.detach().cpu().numpy()
            .astype(np.float32)
        )
        target_scores = scores_all[target_indices]
        order = target_indices[np.argsort(-target_scores)]
        boxes = boxes_all[order]
        scores = scores_all[order]

        detection = DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=[target for _ in order],
        )

        if (result.masks is None or result.masks.data is None or
                len(result.masks.data) == 0):
            return VisionResult(
                detection=detection,
                segmentation=SegmentationResult.empty(
                    height, width,
                    "YOLOE TensorRT returned boxes but no instance mask",
                ),
            )

        masks_all = result.masks.data.detach().cpu().numpy()
        masks = masks_all[order]
        if masks.shape[-2:] != (height, width):
            import cv2
            masks = np.stack([
                cv2.resize(
                    mask.astype(np.float32),
                    (width, height),
                    interpolation=cv2.INTER_NEAREST,
                )
                for mask in masks
            ])

        segmentation = SegmentationResult(
            mask=masks[0] > 0.5,
            scores=scores.copy(),
            best_index=0,
            candidate_count=int(len(masks)),
        )
        return VisionResult(
            detection=detection,
            segmentation=segmentation,
        )

    def close(self):
        release_attributes(self, "_model")
