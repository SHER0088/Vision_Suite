"""
object_detector.py
---------------------
YOLOv8n object detector running via ONNX Runtime (CPU-optimized).

Lifecycle:
  1. Check models/object_detection/yolov8n.onnx -> use it if present
  2. Else check/download yolov8n.pt (weights_url in config)
  3. Export .pt -> .onnx once via ultralytics, cache permanently
  4. All future runs skip straight to step 1

Inference always runs on whatever frame is passed in (caller is
responsible for passing the inference_resolution frame, per our
display/inference resolution decoupling).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, NamedTuple

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from src.core.config_loader import AppConfig
from src.core.model_downloader import ensure_file
from src.core.logger import get_logger

logger = get_logger("object_detector")


class Detection(NamedTuple):
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


class ObjectDetector:
    def __init__(self, config: AppConfig):
        self.cfg = config.models.object_detection
        self.onnx_providers = config.models.onnx_providers
        self.models_dir = Path(config.paths.models_dir) / "object_detection"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.input_size = tuple(self.cfg.input_size)  # (w, h)
        self.conf_threshold = self.cfg.confidence_threshold
        self.iou_threshold = self.cfg.iou_threshold
        self.classes_filter = set(self.cfg.classes_filter) if self.cfg.classes_filter else None

        self.class_names = self._load_class_names()

        onnx_path = self.models_dir / self.cfg.onnx_filename
        self.session = self._load_or_build_onnx(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        logger.info(f"ObjectDetector ready. Input size={self.input_size}, "
                    f"classes_filter={self.classes_filter or 'ALL'}")

    # ------------------------------------------------------------------ #

    def _load_class_names(self) -> List[str]:
        class_file = Path(self.cfg.class_names_path)
        if not class_file.exists():
            raise FileNotFoundError(
                f"Class names file not found at {class_file} "
                f"(set via models.object_detection.class_names_path in config.yaml)"
            )
        with open(class_file, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return data["classes"]

    def _load_or_build_onnx(self, onnx_path: Path) -> ort.InferenceSession:
        if onnx_path.exists():
            logger.info(f"Loading cached ONNX model: {onnx_path}")
            return ort.InferenceSession(str(onnx_path), providers=self.onnx_providers)

        logger.info("Cached ONNX model not found. Building it now (one-time setup)...")

        pt_path = self.models_dir / self.cfg.weights_filename
        ensure_file(pt_path, self.cfg.weights_url, description="YOLOv8n weights (.pt)")

        logger.info("Exporting .pt weights to ONNX format...")
        from ultralytics import YOLO  # local import - only needed for one-time export

        model = YOLO(str(pt_path))
        exported_path = Path(model.export(format="onnx", imgsz=self.input_size[0], simplify=True))

        if exported_path.resolve() != onnx_path.resolve():
            exported_path.rename(onnx_path)

        logger.info(f"ONNX export complete. Cached at: {onnx_path}")
        return ort.InferenceSession(str(onnx_path), providers=self.onnx_providers)

    # ------------------------------------------------------------------ #

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, self.input_size)
        img = resized[:, :, ::-1].astype(np.float32) / 255.0  # BGR -> RGB, normalize
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def _postprocess(self, output: np.ndarray, orig_shape: tuple) -> List[Detection]:
        # YOLOv8 ONNX output: (1, 4+num_classes, num_boxes) -> transpose
        predictions = np.squeeze(output).T

        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = np.max(class_scores, axis=1)

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        if len(boxes_xywh) == 0:
            return []

        orig_h, orig_w = orig_shape[:2]
        scale_x = orig_w / self.input_size[0]
        scale_y = orig_h / self.input_size[1]

        boxes_for_nms = []
        for cx, cy, w, h in boxes_xywh:
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            w_scaled = w * scale_x
            h_scaled = h * scale_y
            boxes_for_nms.append([x1, y1, w_scaled, h_scaled])

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, confidences.tolist(), self.conf_threshold, self.iou_threshold
        )

        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = boxes_for_nms[i]
                class_id = int(class_ids[i])
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)

                if self.classes_filter and class_name not in self.classes_filter:
                    continue

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=float(confidences[i]),
                    bbox=(int(x), int(y), int(x + w), int(y + h)),
                ))
        return detections

    # ------------------------------------------------------------------ #

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Runs detection on a single frame, returns boxes in that frame's coordinate space."""
        input_tensor = self._preprocess(frame)
        output = self.session.run([self.output_name], {self.input_name: input_tensor})[0]
        return self._postprocess(output, frame.shape)