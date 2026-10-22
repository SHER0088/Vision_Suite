"""
recognition/embedder.py
------------------------
Face embedding extraction using InsightFace ArcFace model.
Provides normalized 512-dimensional embeddings for face recognition.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from src.core.config_loader import AppConfig
from src.detection.face_analysis_engine import get_face_analysis_engine
from src.core.logger import get_logger

logger = get_logger("embedder")


class FaceEmbedder:
    """Extracts face embeddings using the ArcFace model from InsightFace buffalo_s pack."""

    def __init__(self, config: AppConfig):
        # Load the full InsightFace model (detection + recognition)
        self.engine = get_face_analysis_engine(config, need_recognition=True)
        logger.info("FaceEmbedder ready (ArcFace via InsightFace, recognition-enabled).")

    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract normalized embedding from a face image.

        Args:
            face_img: BGR face image (cropped from detection)

        Returns:
            Normalized 512-dimensional embedding vector, or None if no face detected
        """
        if face_img is None or face_img.size == 0:
            return None

        # Get face analysis results
        faces = self.engine.get(face_img)
        if not faces:
            return None

        # Use the largest face (by bounding box area)
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))

        # Extract normalized embedding
        embedding = getattr(largest_face, "normed_embedding", None)
        if embedding is None:
            # Fallback to raw embedding and normalize it
            raw_embedding = getattr(largest_face, "embedding", None)
            if raw_embedding is not None:
                norm = np.linalg.norm(raw_embedding)
                if norm > 0:
                    embedding = raw_embedding / norm
                else:
                    embedding = raw_embedding
            else:
                return None

        return embedding.astype(np.float32) if embedding is not None else None

    def get_embeddings(self, face_imgs: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """
        Extract embeddings from multiple face images.

        Args:
            face_imgs: List of BGR face images

        Returns:
            List of embedding vectors (or None for failed extractions)
        """
        return [self.get_embedding(img) for img in face_imgs]