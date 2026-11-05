"""
recognition/embedder.py
------------------------
Face embedding extraction using the fixed pretrained FaceNet
InceptionResnetV1 (VGGFace2) model.

Produces normalized 512-dimensional embeddings. The model is never
retrained or fine-tuned during enrollment or recognition — identity
data lives separately in the pickle database + FAISS index.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("embedder")


class FaceEmbedder:
    """Extracts face embeddings using fixed FaceNet InceptionResnetV1/VGGFace2.

    The detector supplies the face bounding box. The face is clipped,
    padded to a square, resized to FaceNet's required 160x160 RGB input,
    prewhitened, and passed through the frozen model to produce a
    normalized 512-dimensional embedding.
    """

    def __init__(self, config: AppConfig):
        self.model_name = getattr(
            config.models.face_recognition, "model_name", "facenet-inceptionresnetv1-vggface2"
        )
        self.embedding_size = int(getattr(config.models.face_recognition, "embedding_size", 512))
        self.device = torch.device(config.runtime.device)

        cache_dir = Path(config.paths.models_dir) / "facenet-cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache_dir))

        try:
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        except Exception as exc:
            raise RuntimeError(f"Unable to load pretrained FaceNet VGGFace2 weights: {exc}") from exc

        logger.info("FaceEmbedder ready (%s on %s).", self.model_name, self.device)

    def get_embedding(self, face_img: np.ndarray) -> Optional[np.ndarray]:
        """Extract a normalized embedding from a face image.

        Args:
            face_img: BGR face image (cropped from a detection box)

        Returns:
            Normalized 512-dimensional float32 embedding, or None on failure
        """
        if face_img is None or face_img.size == 0:
            return None

        try:
            face = self._preprocess(face_img)
            tensor = torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
            with torch.no_grad():
                vector = self._model(tensor).cpu().numpy()[0]
            values = self._normalize(vector)
            return values.astype(np.float32)
        except Exception as exc:
            logger.warning("FaceNet embedding extraction failed: %s", exc)
            return None

    def get_embeddings(self, face_imgs: List[np.ndarray]) -> List[Optional[np.ndarray]]:
        """Extract embeddings from multiple face images."""
        return [self.get_embedding(img) for img in face_imgs]

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """Preprocess a face crop to FaceNet's 160x160 RGB input."""
        rgb = cv2.cvtColor(
            cv2.resize(image, (160, 160), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB
        )
        return (rgb.astype(np.float32) - 127.5) / 128.0

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalize a vector."""
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(values))
        if norm == 0.0:
            raise ValueError("Cannot normalize a zero-length embedding")
        return values / norm
