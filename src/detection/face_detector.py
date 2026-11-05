"""
Face detection using SCRFD via insightface.

A face-specific detector — outputs face bounding boxes only.
Model is auto-downloaded by insightface on first use and cached locally.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import numpy as np

from insightface.app import FaceAnalysis

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("face_detector")


class FaceDetection(NamedTuple):
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2 in frame coordinates
    embedding: Optional[np.ndarray] = None


class FaceDetector:
    """SCRFD face detection via insightface (ONNX Runtime CPU)."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.models.face_detection
        self.conf_threshold = float(self.cfg.confidence_threshold)
        raw_size = self.cfg.input_size
        self.input_size = int(raw_size) if isinstance(raw_size, (int, float)) else int(raw_size[0])

        self.app = FaceAnalysis(providers=list(config.models.onnx_providers))
        self.app.prepare(ctx_id=0, det_size=(self.input_size, self.input_size))

        logger.info(
            "FaceDetector ready | backend=insightface | input_size=%d | providers=%s",
            self.input_size, config.models.onnx_providers,
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> List[FaceDetection]:
        """Run face detection on a BGR frame. Returns list of FaceDetection."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        faces = self.app.get(frame_bgr)
        detections: List[FaceDetection] = []
        for face in faces:
            # insightface bbox: [x1, y1, x2, y2, score]
            bbox = face.bbox
            conf = float(bbox[4])
            if conf < self.conf_threshold:
                continue
            detections.append(
                FaceDetection(
                    confidence=conf,
                    bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
                )
            )
        return detections
