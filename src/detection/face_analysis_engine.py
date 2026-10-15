"""
face_analysis_engine.py
---------------------------
Shared InsightFace engine (SCRFD detector + ArcFace recognizer bundled
as the 'buffalo_s' pack). Loaded ONCE as a singleton - both face_detector.py
and the future recognition/embedder.py pull from this same instance to
avoid loading the model pack twice and wasting CPU/RAM.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from insightface.app import FaceAnalysis

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("face_analysis_engine")

_engine_instance: Optional[FaceAnalysis] = None


def get_face_analysis_engine(config: AppConfig) -> FaceAnalysis:
    """
    Returns a cached singleton FaceAnalysis instance. Auto-downloads the
    configured pack on first use, cached under models/insightface/.
    """
    global _engine_instance
    if _engine_instance is not None:
        return _engine_instance

    pack_name = config.models.face_detection.pack_name
    insightface_root = Path(config.paths.models_dir) / "insightface"
    insightface_root.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading InsightFace pack '{pack_name}' (detection + recognition)...")
    logger.info("First run will auto-download the pack - this may take a minute.")

    app = FaceAnalysis(
        name=pack_name,
        root=str(insightface_root),
        providers=config.models.onnx_providers,
    )

    det_size = tuple(config.models.face_detection.input_size)
    app.prepare(ctx_id=-1, det_size=det_size)  # ctx_id=-1 forces CPU

    logger.info(f"InsightFace pack '{pack_name}' ready. Cached at: {insightface_root}")

    _engine_instance = app
    return _engine_instance