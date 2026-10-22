"""Configuration loader and Pydantic validation schema for Vision_Suite.

The config.yaml is the single source of truth for all system values.
A new top-level `input` section replaces the old flat `cameras` list:

    input:
      type: "web_cam"          # video | web_cam | live
      camera_indices: [0, 1]   # used when type == "web_cam"
      video_paths: [...]       # used when type == "video"
      rtsp_urls: [...]         # used when type == "live"
      ...

Only the section matching `type` is read. Each source gets its own
capture thread, frame buffer, and preview window.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --- SCHEMAS ---

class BaseConfigModel(BaseModel):
    """Base model: disables protected namespaces warning, allows extra fields
    so the config can evolve without breaking validation."""
    model_config = ConfigDict(protected_namespaces=(), extra="allow")

    @staticmethod
    def _parse_input_size(v: Any) -> int:
        if isinstance(v, list) and len(v) > 0:
            return int(v[0])
        return int(v)


class InputSourceConfig(BaseConfigModel):
    """Single input section — one type, multiple sources of that type."""
    type: str = Field(
        default="web_cam",
        description="Input type: 'web_cam' (device indices), 'video' (file paths), or 'live' (RTSP URLs)",
    )
    camera_indices: List[int] = Field(
        default_factory=lambda: [0],
        description="Device indices for web_cam / usb_cam inputs",
    )
    video_paths: List[str] = Field(
        default_factory=list,
        description="Offline video file paths for video input",
    )
    rtsp_urls: List[str] = Field(
        default_factory=list,
        description="RTSP stream URLs for live input",
    )
    frame_skip: Optional[int] = 0
    reconnect_delay_seconds: Optional[int] = 5
    transport: Optional[str] = "tcp"
    display_resolution: Optional["ResolutionItem"] = None
    inference_resolution: Optional["ResolutionItem"] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"video", "web_cam", "live"}
        if v.lower() not in allowed:
            raise ValueError(f"input.type must be one of {sorted(allowed)}, got: {v!r}")
        return v.lower()

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v.lower() not in {"tcp", "udp"}:
            raise ValueError(f"input.transport must be 'tcp' or 'udp', got: {v!r}")
        return v.lower() if v else v

    def active_sources(self) -> List[Dict[str, Any]]:
        """Return the list of sources for the configured type.

        Each entry is a dict with keys: id, type, source (int or str),
        plus shared settings. This is what VideoSourceManager consumes.
        """
        t = self.type
        sources: List[Dict[str, Any]] = []

        if t == "web_cam":
            for idx, cam_idx in enumerate(self.camera_indices):
                sources.append({
                    "id": f"webcam_{cam_idx}",
                    "type": "live",
                    "source": cam_idx,
                    "enabled": True,
                    "frame_skip": self.frame_skip,
                    "reconnect_delay_seconds": self.reconnect_delay_seconds,
                    "transport": self.transport,
                    "display_resolution": self.display_resolution,
                    "inference_resolution": self.inference_resolution,
                })
        elif t == "video":
            for idx, path in enumerate(self.video_paths):
                sources.append({
                    "id": f"video_{idx}",
                    "type": "offline",
                    "source": path,
                    "enabled": True,
                    "frame_skip": self.frame_skip,
                    "reconnect_delay_seconds": self.reconnect_delay_seconds,
                    "transport": None,
                    "display_resolution": self.display_resolution,
                    "inference_resolution": self.inference_resolution,
                })
        elif t == "live":
            for idx, url in enumerate(self.rtsp_urls):
                sources.append({
                    "id": f"rtsp_{idx}",
                    "type": "rtsp",
                    "source": url,
                    "enabled": True,
                    "frame_skip": self.frame_skip,
                    "reconnect_delay_seconds": self.reconnect_delay_seconds,
                    "transport": self.transport,
                    "display_resolution": self.display_resolution,
                    "inference_resolution": self.inference_resolution,
                })

        return sources


class ProjectConfig(BaseConfigModel):
    name: str = "Vision_Suite"
    version: str = "0.1.0"
    device: str = "cpu"
    description: Optional[str] = "Vision Suite Pipeline"


class PathsConfig(BaseConfigModel):
    models_dir: str = "models"
    data_dir: str = "data"
    faces_dir: Optional[str] = "data/faces"
    embeddings_dir: Optional[str] = "data/embeddings"
    face_db_path: Optional[str] = "models/face_recognition/face_db.pkl"
    output_dir: str = "output"
    logs_dir: str = "output/logs"
    crops_dir: str = "output/crops"


class LoggingConfig(BaseConfigModel):
    system_log_level: Optional[str] = "INFO"
    system_log_file: Optional[str] = "output/logs/system.log"
    system_log_max_bytes: Optional[int] = 5242880
    system_log_backup_count: Optional[int] = 5
    event_log_cooldown_seconds: Optional[float] = 10.0
    log_cooldown_seconds: Optional[float] = 5.0
    events_csv_file: Optional[str] = "output/logs/events.csv"
    event_csv_columns: Optional[List[str]] = None
    rotation: Optional[str] = "1 day"
    retention: Optional[str] = "7 days"


class ObjectDetectionModelConfig(BaseConfigModel):
    name: str = "yolov8n"
    weights_url: str = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    weights_filename: str = "yolov8n.pt"
    onnx_filename: str = "yolov8n.onnx"
    class_names_path: str = "configs/coco_classes.yaml"
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    input_size: Union[int, List[int]] = 640
    classes_filter: Optional[List[Any]] = None

    @field_validator("input_size", mode="before")
    @classmethod
    def parse_input_size(cls, v: Any) -> int:
        return cls._parse_input_size(v)


class FaceDetectionModelConfig(BaseConfigModel):
    name: str = "scrfd"
    provider: str = "insightface"
    pack_name: str = "buffalo_s"
    confidence_threshold: float = 0.5
    input_size: Union[int, List[int]] = 640
    det_size: List[int] = [640, 640]

    @field_validator("input_size", mode="before")
    @classmethod
    def parse_input_size(cls, v: Any) -> int:
        return cls._parse_input_size(v)


class FaceRecognitionModelConfig(BaseConfigModel):
    name: str = "arcface"
    provider: str = "insightface"
    pack_name: str = "buffalo_s"
    similarity_threshold: float = 0.45
    embedding_size: int = 512
    metric: str = "cosine"


class ModelsConfig(BaseConfigModel):
    onnx_providers: List[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    object_detection: ObjectDetectionModelConfig
    face_detection: FaceDetectionModelConfig
    face_recognition: FaceRecognitionModelConfig


class TrackingConfig(BaseConfigModel):
    type: Optional[str] = "iou"
    iou_threshold: float = 0.3
    max_age: int = 30
    min_hits: int = 3


class ResolutionItem(BaseConfigModel):
    width: int
    height: int

    def as_tuple(self) -> tuple[int, int]:
        return (self.width, self.height)


class ResolutionConfig(BaseConfigModel):
    display: ResolutionItem
    inference: ResolutionItem


class DisplayConfig(BaseConfigModel):
    mode: str = "grid"
    grid_columns: int = 2
    window_name_prefix: str = "Vision_Suite"


class DetectionConfig(BaseConfigModel):
    enable_object_detection: bool = True
    enable_face_detection: bool = True


class CropsConfig(BaseConfigModel):
    save_object_crops: bool = True
    save_face_crops: bool = True
    crop_padding: int = 10
    min_crop_size: int = 40
    cooldown_seconds: float = 5.0
    object_crop_dir: str = "output/crops/objects"
    face_crop_dir: str = "output/crops/faces"
    unknown_crop_dir: str = "output/crops/unknown"


class QualityChecksConfig(BaseConfigModel):
    min_brightness: float = 40.0
    max_brightness: float = 220.0
    blur_threshold: float = 80.0
    min_face_size_px: int = 80


class DataCollectionConfig(BaseConfigModel):
    person_name: str = "John_Doe"
    overwrite_if_exists: bool = True
    details: Dict[str, Any] = Field(default_factory=dict)
    poses: List[str] = Field(default_factory=lambda: ["straight", "left", "right", "up", "down"])
    samples_per_pose: int = 15
    capture_interval_seconds: float = 0.5
    quality_checks: QualityChecksConfig = Field(default_factory=QualityChecksConfig)


class RecognitionConfig(BaseConfigModel):
    save_crops: bool = True
    cooldown_seconds: float = 5.0
    unknown_label: str = "unknown"
    draw_landmarks: bool = False


class SystemConfig(BaseConfigModel):
    enable_ear_mar: bool = False


class AppConfig(BaseConfigModel):
    project: ProjectConfig
    paths: PathsConfig
    logging: LoggingConfig
    models: ModelsConfig
    tracking: TrackingConfig
    resolution: ResolutionConfig
    display: DisplayConfig
    input: InputSourceConfig
    detection: DetectionConfig
    crops: CropsConfig
    data_collection: DataCollectionConfig
    recognition: RecognitionConfig
    system: SystemConfig


# --- CONFIG LOADER SINGLETON ---

_config_instance: Optional[AppConfig] = None


def find_config_path(explicit_path: Optional[str] = None) -> Path:
    """Find config.yaml at explicit path, CWD, or by walking up parent directories."""
    if explicit_path:
        p = Path(explicit_path)
        if p.exists():
            return p

    cwd_path = Path.cwd() / "config.yaml"
    if cwd_path.exists():
        return cwd_path

    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        config_path = parent / "config.yaml"
        if config_path.exists():
            return config_path

    checked_paths = [cwd_path]
    for i, parent in enumerate(current_path.parents):
        if i >= 5:
            break
        checked_paths.append(parent / "config.yaml")

    raise FileNotFoundError(
        f"config.yaml not found! Checked:\n"
        + "\n".join(f" - {p}" for p in checked_paths)
    )


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load config.yaml, validate using AppConfig Pydantic model."""
    global _config_instance

    target_path = find_config_path(config_path)

    try:
        with target_path.open("r", encoding="utf-8") as f:
            raw_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in config file {target_path}: {e}") from e
    except OSError as e:
        raise OSError(f"Cannot read config file {target_path}: {e}") from e

    if raw_dict is None:
        raise ValueError(f"Config file {target_path} is empty or contains only comments")

    _config_instance = AppConfig(**raw_dict)
    return _config_instance


def get_config() -> AppConfig:
    """Get loaded singleton config, or load default."""
    global _config_instance
    if _config_instance is None:
        _config_instance = load_config()
    return _config_instance


def ensure_runtime_directories(config: AppConfig) -> None:
    """Ensure runtime folders (output/, logs/, crops/, models/, data/) exist."""
    from src.core.logger import get_logger  # lazy import to avoid circular import

    logger = get_logger("config_loader")
    project_root = Path(__file__).resolve().parents[2]

    dirs_to_create = [
        project_root / config.paths.models_dir / "object_detection",
        project_root / config.paths.models_dir / "face_detection",
        project_root / config.paths.models_dir / "face_recognition",
        project_root / config.paths.data_dir / "faces",
        project_root / config.paths.data_dir / "embeddings",
        project_root / config.paths.output_dir / "logs",
        project_root / config.paths.output_dir / "crops" / "objects",
        project_root / config.paths.output_dir / "crops" / "faces",
        project_root / config.paths.output_dir / "crops" / "recognized",
    ]

    for directory in dirs_to_create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise
