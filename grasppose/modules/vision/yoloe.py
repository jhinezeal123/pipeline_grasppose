"""Static-prompt YOLOE TensorRT adapter."""

import time

import numpy as np

from ...infrastructure.settings import YOLOE_ARTIFACT_ROOT, YOLOE_CONF, YOLOE_IMGSZ, YOLOE_MODEL
from .types import DetectionResult, SegmentationResult, VisionResult
from .port import VisionPort
from .prompt_catalog import PromptCatalog
from ...infrastructure.runtime import log, release_attributes


class Yoloe26sVision(VisionPort):
    """Run a fixed YOLOE prompt profile exported to a TensorRT engine."""

    def __init__(self, engine_path=None, conf=None, imgsz=None):
        self.conf = YOLOE_CONF if conf is None else float(conf)
        self.imgsz = YOLOE_IMGSZ if imgsz is None else int(imgsz)
        self.engine_path = engine_path
        self._model = None
        self._catalog = None

    def load(self):
        if self._model is not None:
            return self
        self._catalog = PromptCatalog.load(verify_engine=True)
        engine = self._catalog.manifest["engine"]
        if self.imgsz != int(engine.get("imgsz", -1)):
            raise RuntimeError(
                "YOLOE engine is fixed at imgsz=%s; runtime requested %s"
                % (engine.get("imgsz"), self.imgsz)
            )
        if self.conf != float(self._catalog.manifest.get("conf", -1.0)):
            raise RuntimeError(
                "YOLOE confidence does not match artifact validation"
            )
        path = self.engine_path or (
            self._catalog.artifact_dir + "/" + engine["file"])
        started = time.time()
        # Ultralytics documents exported prompted files as standard YOLO
        # models; this path does not load CLIP or call set_classes().
        from ultralytics import YOLO
        self._model = YOLO(path)
        log("YOLOE TensorRT engine loaded in %.2fs" % (
            time.time() - started))
        return self

    def predict(self, image, prompt_id):
        self.load()
        prompt = self._catalog.require(prompt_id)
        rgb = np.asarray(image)[:, :, :3]
        height, width = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        result = self._model.predict(
            source=bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=0,
            retina_masks=True,
            verbose=False,
        )[0]
        if result is None or result.boxes is None or len(result.boxes) == 0:
            reason = "YOLOE found no object for prompt ID %r" % prompt_id
            return VisionResult(
                detection=DetectionResult.empty(reason),
                segmentation=SegmentationResult.empty(height, width, reason),
            )

        classes = result.boxes.cls.detach().cpu().numpy().astype(np.int32)
        scores_all = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
        keep = np.flatnonzero(classes == int(prompt["class_index"]))
        if not len(keep):
            reason = "YOLOE found no object for prompt ID %r" % prompt_id
            return VisionResult(
                detection=DetectionResult.empty(reason),
                segmentation=SegmentationResult.empty(height, width, reason),
            )
        boxes_all = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        order = keep[np.argsort(-scores_all[keep])]
        boxes = boxes_all[order]
        scores = scores_all[order]
        detection = DetectionResult(
            boxes=boxes,
            scores=scores,
            labels=[prompt["text"] for _ in order],
        )
        if result.masks is None or result.masks.data is None or len(result.masks.data) == 0:
            return VisionResult(
                detection=detection,
                segmentation=SegmentationResult.empty(
                    height, width, "YOLOE returned boxes but no instance mask"
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
        return VisionResult(
            detection=detection,
            segmentation=SegmentationResult(
                mask=masks[0] > 0.5,
                scores=scores.copy(),
                best_index=0,
                candidate_count=int(len(masks)),
            ),
        )

    def warmup(self):
        self.load()
        item = self._catalog.prompts[0]
        self.predict(np.zeros((640, 640, 3), np.uint8), item["id"])

    def close(self):
        release_attributes(self, "_model")
        self._catalog = None
