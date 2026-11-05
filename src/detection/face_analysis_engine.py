"""
face_analysis_engine.py
---------------------------
DEPRECATED — Legacy InsightFace engine module. Previously used by SCRFD-based face
detection. Face detection now uses YOLOv8n-Face directly in
src/detection/face_detector.py. Face recognition uses FaceNet in
src/recognition/embedder.py. This module is NOT imported by any active pipeline
and will be removed in a future release.
"""
from __future__ import annotations

import warnings
from importlib import import_module
from pathlib import Path
from typing import Any, Optional, Tuple

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("face_analysis_engine")

warnings.warn(
    "face_analysis_engine is deprecated and unused by active pipelines. "
    "Use src/detection/face_detector.py for detection and "
    "src/recognition/embedder.py for recognition.",
    DeprecationWarning,
    stacklevel=2,
)

_engine_instance: Optional[Any] = None
_engine_need_recognition: Optional[bool] = None


def _resolve_det_size(raw: Any) -> Tuple[int, int]:
    """Accepts int (640) or list/tuple ([640, 640]); always returns a 2-tuple."""
    if isinstance(raw, (list, tuple)):
        return (int(raw[0]), int(raw[1] if len(raw) > 1 else raw[0]))
    return (int(raw), int(raw))


def get_face_analysis_engine(config: AppConfig, need_recognition: bool = True) -> Any:
    """
    Returns a cached singleton FaceAnalysis instance. Auto-downloads the
    configured pack on first use, cached under models/insightface/.

    If the engine was previously loaded with need_recognition=False but a
    later request needs recognition (True), the engine is re-loaded with the
    full pack so ArcFace embeddings are available. This prevents the
    detection-only engine from silently blocking embedding extraction.
    """
    global _engine_instance, _engine_need_recognition
    if _engine_instance is not None:
        if _engine_need_recognition != need_recognition:
            if need_recognition and not _engine_need_recognition:
                # Upgrade from detection-only to full pack
                logger.warning(
                    "Re-loading InsightFace pack with recognition enabled "
                    "(previous load was detection-only)."
                )
                _engine_instance = None
                _engine_need_recognition = None
            else:
                logger.warning(
                    "Face engine already loaded with need_recognition=%s; "
                    "ignoring later request for need_recognition=%s in this process.",
                    _engine_need_recognition, need_recognition,
                )
                return _engine_instance

    # Import lazily so modules that do not use face analysis can still load
    # when InsightFace is not installed in the active environment.
    FaceAnalysis = import_module("insightface.app").FaceAnalysis

    pack_name = config.models.face_detection.pack_name
    insightface_root = Path(config.paths.models_dir) / "insightface"
    insightface_root.mkdir(parents=True, exist_ok=True)

    allowed_modules = None if need_recognition else ["detection"]

    logger.info(
        "Loading InsightFace pack '%s' (modules=%s)...",
        pack_name, allowed_modules or "all",
    )
    logger.info("First run will auto-download the pack - this may take a minute.")

    app = FaceAnalysis(
        name=pack_name,
        root=str(insightface_root),
        providers=config.models.onnx_providers,
        allowed_modules=allowed_modules,
    )

    det_size = _resolve_det_size(config.models.face_detection.input_size)
    app.prepare(ctx_id=-1, det_size=det_size)  # ctx_id=-1 forces CPU

    logger.info(f"InsightFace pack '{pack_name}' ready. Cached at: {insightface_root}")

    _engine_instance = app
    _engine_need_recognition = need_recognition
    return _engine_instance