"""Ultralytics YOLOE adapter implementing the vision port."""

import time

import numpy as np

from ..config import YOLOE_CONF, YOLOE_IMGSZ, YOLOE_MODEL
from ..domain.types import (
    DetectionResult,
    SegmentationResult,
    VisionResult,
)
from ..ports.vision import VisionPort
from ..runtime import cuda_available, log, release_attributes


class Yoloe26sVision(VisionPort):
    """Open-vocabulary detection + segmentation in a single model."""

    def __init__(self, model_path=None, device=None, conf=None, imgsz=None):
        self.model_path = model_path or YOLOE_MODEL
        self.device = device if device is not None else (
            0 if cuda_available() else "cpu")
        self.conf = YOLOE_CONF if conf is None else float(conf)
        self.imgsz = YOLOE_IMGSZ if imgsz is None else int(imgsz)
        self._model = None
        self._classes_prompt = None

    def load(self):
        if self._model is None:
            from ultralytics import YOLOE
            started = time.time()
            self._model = YOLOE(self.model_path)
            log("YOLOE-26s loaded in %.1fs" % (
                time.time() - started))
        return self

    def predict(self, image, prompt):
        self.load()
        rgb = np.asarray(image)[:, :, :3]
        height, width = rgb.shape[:2]
        # Ultralytics treats NumPy HWC inputs as OpenCV-style BGR and flips
        # them to RGB in predictor.preprocess(). The pipeline contract is RGB,
        # so convert here exactly once before handing the array to Ultralytics.
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        prompt = str(prompt).strip() or "object"

        if self._classes_prompt != prompt:
            self._model.set_classes([prompt])
            self._classes_prompt = prompt

        result = self._model.predict(
            source=bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            device=self.device,
            half=bool(cuda_available()),
            retina_masks=True,
            verbose=False,
        )[0]

        if result is None or result.boxes is None or len(result.boxes) == 0:
            reason = "YOLOE did not find an object for prompt %r" % prompt
            return VisionResult(
                detection=DetectionResult.empty(reason),
                segmentation=SegmentationResult.empty(
                    height, width, reason),
            )

        boxes = (
            result.boxes.xyxy.detach().cpu().numpy()
            .astype(np.float32)
        )
        scores = (
            result.boxes.conf.detach().cpu().numpy()
            .astype(np.float32)
        )
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        detection = DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=[prompt for _ in order],
        )

        if (result.masks is None or result.masks.data is None or
                len(result.masks.data) == 0):
            return VisionResult(
                detection=detection,
                segmentation=SegmentationResult.empty(
                    height, width,
                    "YOLOE returned boxes but no instance mask",
                ),
            )

        masks = result.masks.data.detach().cpu().numpy()[order]
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
        self._classes_prompt = None
