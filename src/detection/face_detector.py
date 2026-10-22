"""
face_detector.py
-------------------
Thin wrapper exposing only the face DETECTION responsibility. Loads the
shared InsightFace engine in detection-only mode (no recognition/landmark/
gender-age submodels) - this pipeline doesn't use embeddings, so skipping
those forward passes is a pure speed win.
"""
from __future__ import annotations

from typing import List, NamedTuple, Optional

import numpy as np

from src.core.config_loader import AppConfig
from src.detection.face_analysis_engine import get_face_analysis_engine
from src.core.logger import get_logger

logger = get_logger("face_detector")


class FaceDetection(NamedTuple):
    confidence: float
    bbox: tuple                        # (x1, y1, x2, y2)
    landmarks: np.ndarray              # 5-point landmarks (from SCRFD, always available)
    embedding: Optional[np.ndarray]    # None here - recognition module not loaded


class FaceDetector:
    def __init__(self, config: AppConfig):
        self.engine = get_face_analysis_engine(config, need_recognition=False)
        self.conf_threshold = config.models.face_detection.confidence_threshold
        logger.info("FaceDetector ready (SCRFD via InsightFace, detection-only).")

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        faces = self.engine.get(frame)
        results = []
        for face in faces:
            if face.det_score < self.conf_threshold:
                continue
            x1, y1, x2, y2 = face.bbox.astype(int)
            results.append(FaceDetection(
                confidence=float(face.det_score),
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                landmarks=face.kps,
                embedding=getattr(face, "normed_embedding", None),
            ))
        return results