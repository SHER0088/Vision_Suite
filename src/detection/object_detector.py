"""YOLOv8n object detector via ONNX Runtime (CPU)."""

from __future__ import annotations

from pathlib import Path
import importlib
from typing import Any, List, NamedTuple, Tuple

cv2 = importlib.import_module("cv2")
import numpy as np
import onnxruntime as ort
import yaml

from src.core.config_loader import AppConfig
from src.core.logger import get_logger
from src.core.model_downloader import ensure_file

logger = get_logger("object_detector")


class Detection(NamedTuple):
    class_id: int
    class_name: str
    confidence: float
    # x1, y1, x2, y2 in the coordinate space of the frame passed to detect()
    bbox: Tuple[float, float, float, float]


class ObjectDetector:
    """YOLOv8n → ONNX (one-time export) → ONNX Runtime inference."""

    def __init__(self, config: AppConfig) -> None:
        self.cfg = config.models.object_detection
        self.paths = config.paths
        self.onnx_providers = list(config.models.onnx_providers)

        self.models_dir = Path(self.paths.models_dir) / "object_detection"
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.conf_threshold = float(self.cfg.confidence_threshold)
        self.iou_threshold = float(self.cfg.iou_threshold)

        # Handle integer 640 or list [640, 640] safely
        raw_size = getattr(self.cfg, "input_size", 640)
        if isinstance(raw_size, (list, tuple)):
            self.input_size = int(raw_size[0])
        else:
            self.input_size = int(raw_size)

        self.class_names = self._load_class_names()

        onnx_path = self.models_dir / self.cfg.onnx_filename
        self.session = self._load_or_build_onnx(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        logger.info(
            "ObjectDetector ready | providers=%s | classes=%d | input_size=%d",
            self.session.get_providers(),
            len(self.class_names),
            self.input_size,
        )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _load_class_names(self) -> List[str]:
        path = Path(self.cfg.class_names_path)
        if not path.exists():
            raise FileNotFoundError(
                f"COCO class names file not found: {path}. "
                "Expected configs/coco_classes.yaml"
            )
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if isinstance(data, list):
            return [str(x) for x in data]

        if isinstance(data, dict):
            # Check for common wrapper keys like 'classes' or 'names'
            for key in ("classes", "names"):
                if key in data:
                    inner = data[key]
                    if isinstance(inner, list):
                        return [str(x) for x in inner]
                    if isinstance(inner, dict):
                        return [str(inner[k]) for k in sorted(inner, key=lambda x: int(x))]

            # Direct index dictionary mapping {0: 'person', 1: 'bicycle'}
            try:
                return [str(data[k]) for k in sorted(data, key=lambda x: int(x))]
            except (ValueError, TypeError):
                return [str(v) for v in data.values()]

        raise ValueError(f"Unrecognized class names format in {path}")

    def _load_or_build_onnx(self, onnx_path: Path) -> ort.InferenceSession:
        if onnx_path.exists():
            logger.info("Loading cached ONNX model: %s", onnx_path)
            try:
                return self._make_session(onnx_path)
            except Exception as exc:  # noqa: BLE001
                if not self._is_ir_or_model_load_error(exc):
                    raise
                logger.warning(
                    "Cached ONNX failed to load (%s). Deleting and re-exporting.",
                    exc,
                )
                try:
                    onnx_path.unlink(missing_ok=True)
                except OSError as unlink_exc:
                    logger.error("Could not delete bad ONNX cache: %s", unlink_exc)
                    raise exc from unlink_exc

        logger.info("Building ONNX model (one-time setup)...")
        self._export_onnx(onnx_path)
        return self._make_session(onnx_path)

    def _make_session(self, onnx_path: Path) -> ort.InferenceSession:
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return ort.InferenceSession(
            str(onnx_path),
            sess_options=so,
            providers=self.onnx_providers,
        )

    @staticmethod
    def _is_ir_or_model_load_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        needles = (
            "ir version",
            "unsupported model",
            "load model",
            "onnxruntime",
            "invalid protobuf",
            "failed to load",
        )
        return any(n in msg for n in needles)

    def _export_onnx(self, onnx_path: Path) -> None:
        """Download .pt if needed, export to ONNX with opset=12."""
        pt_name = getattr(self.cfg, "weights_filename", None) or getattr(
            self.cfg, "pt_filename", "yolov8n.pt"
        )
        pt_path = self.models_dir / pt_name

        if not pt_path.exists():
            url = getattr(self.cfg, "weights_url", None) or getattr(
                self.cfg, "pt_url", None
            )
            if url:
                # NOTE: ensure_file(local_path, url, description) - local path FIRST.
                ensure_file(pt_path, url, description=f"{pt_name} weights")
            else:
                pt_path = Path(getattr(self.cfg, "model_name", "yolov8n.pt"))

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is required for ONNX export. Install requirements.txt."
            ) from exc

        logger.info("Exporting %s → %s (opset=12)", pt_path, onnx_path)
        model = YOLO(str(pt_path))
        export_result = model.export(
            format="onnx",
            imgsz=self.input_size,
            opset=12,
            simplify=True,
            dynamic=False,
            half=False,
        )

        exported = Path(str(export_result))
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        if exported.resolve() != onnx_path.resolve():
            if onnx_path.exists():
                onnx_path.unlink()
            exported.replace(onnx_path)
            if not onnx_path.exists() and exported.exists():
                import shutil
                shutil.copy2(exported, onnx_path)

        if not onnx_path.exists():
            raise FileNotFoundError(
                f"ONNX export reported success but file missing: {onnx_path}"
            )
        logger.info("ONNX export complete: %s", onnx_path)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def detect(self, frame_bgr: np.ndarray) -> List[Detection]:
        """Run detection on a BGR frame (inference resolution)."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        h0, w0 = frame_bgr.shape[:2]
        inp, gain, pad = self._letterbox(frame_bgr, self.input_size)

        blob = inp[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[np.newaxis, ...])

        outputs = self.session.run([self.output_name], {self.input_name: blob})[0]
        return self._postprocess(outputs, gain, pad, w0, h0)

    def _letterbox(
        self, img: np.ndarray, new_size: int
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        h, w = img.shape[:2]
        r = min(new_size / h, new_size / w)
        nw, nh = int(round(w * r)), int(round(h * r))
        pad_w, pad_h = (new_size - nw) / 2, (new_size - nh) / 2

        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        out = cv2.copyMakeBorder(
            resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        return out, r, (pad_w, pad_h)

    def _postprocess(
        self,
        output: np.ndarray,
        gain: float,
        pad: Tuple[float, float],
        orig_w: int,
        orig_h: int,
    ) -> List[Detection]:
        pred = output[0]
        if pred.shape[0] == 4 + len(self.class_names) or pred.shape[0] < pred.shape[-1]:
            pred = pred.T

        boxes_xywh = pred[:, :4]
        scores_all = pred[:, 4:]
        class_ids = np.argmax(scores_all, axis=1)
        confidences = scores_all[np.arange(scores_all.shape[0]), class_ids]

        mask = confidences >= self.conf_threshold
        boxes_xywh = boxes_xywh[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]

        if len(confidences) == 0:
            return []

        pad_w, pad_h = pad
        x = (boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2 - pad_w) / gain
        y = (boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2 - pad_h) / gain
        x2 = (boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2 - pad_w) / gain
        y2 = (boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2 - pad_h) / gain

        x = np.clip(x, 0, orig_w)
        y = np.clip(y, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)

        boxes_cv = np.stack([x, y, x2 - x, y2 - y], axis=1).tolist()
        conf_list = confidences.tolist()
        cls_list = class_ids.tolist()

        indices = cv2.dnn.NMSBoxes(
            boxes_cv, conf_list, self.conf_threshold, self.iou_threshold
        )
        if indices is None or len(indices) == 0:
            return []

        indices = np.array(indices).reshape(-1).tolist()

        detections: List[Detection] = []
        for i in indices:
            cid = int(cls_list[i])
            name = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            bw, bh = boxes_cv[i][2], boxes_cv[i][3]
            detections.append(
                Detection(
                    class_id=cid,
                    class_name=name,
                    confidence=float(conf_list[i]),
                    bbox=(
                        float(boxes_cv[i][0]),
                        float(boxes_cv[i][1]),
                        float(boxes_cv[i][0] + bw),
                        float(boxes_cv[i][1] + bh),
                    ),
                )
            )
        return detections