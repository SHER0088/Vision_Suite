"""
Configuration loader and Pydantic validation schema for Vision_Suite.

config.yaml is the single source of truth. Every runtime value is read from
config.yaml — nothing is hardcoded in Python. All sections from the master
configuration blueprint are represented here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import yaml

from pydantic import BaseModel, ConfigDict


class BaseConfigModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=(), extra="allow")


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------
class ProjectConfig(BaseConfigModel):
    name: str = "Vision_Suite"
    version: str = "0.2.0"
    environment: str = "development"


# ---------------------------------------------------------------------------
# runtime
# ---------------------------------------------------------------------------
class RuntimeConfig(BaseConfigModel):
    device: str = "cpu"
    random_seed: int = 42
    deterministic: bool = True
    num_threads: int = 4
    multiprocessing: bool = False
    graceful_shutdown: bool = True


# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
class PathsConfig(BaseConfigModel):
    models_dir: str = "models"
    data_dir: str = "data"
    raw_faces_dir: str = "data/faces/raw"
    processed_faces_dir: str = "data/faces/processed"
    embeddings_dir: str = "data/embeddings"
    database_dir: str = "data/database"
    output_dir: str = "output"
    logs_dir: str = "output/logs"
    crops_dir: str = "output/crops"
    reports_dir: str = "output/reports"
    temporary_dir: str = "output/tmp"
    legacy_face_db_path: str = "models/face_recognition/face_db.pkl"


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------
class InputBufferConfig(BaseConfigModel):
    max_frames: int = 2
    drop_oldest: bool = True


class InputSourceConfig(BaseConfigModel):
    id: Optional[str] = None
    type: str = "web_cam"
    index: int = 0
    enabled: bool = True
    camera_settings: Optional[dict] = None


class CameraConfig(BaseConfigModel):
    brightness: int = 0
    contrast: int = 0
    saturation: int = 0
    exposure: int = 0
    focus: int = 0
    gain: int = 0


class InputConfig(BaseConfigModel):
    type: str = "video"
    camera: Optional[CameraConfig] = None
    camera_indices: List[int] = [0]
    video_paths: List[str] = []
    rtsp_urls: List[str] = []
    reconnect_delay_seconds: int = 5
    max_reconnect_attempts: int = 0
    open_timeout_seconds: int = 10
    read_timeout_seconds: int = 10
    transport: str = "tcp"
    buffer: InputBufferConfig = InputBufferConfig()
    display_resolution: Optional[Any] = None
    inference_resolution: Optional[Any] = None
    sources: List[InputSourceConfig] = []

    def active_sources(self) -> List[InputSourceConfig]:
        """Return only the enabled input sources."""
        return [source for source in self.sources if source.enabled]


# ---------------------------------------------------------------------------
# performance
# ---------------------------------------------------------------------------
class ProfilingConfig(BaseConfigModel):
    enabled: bool = False
    report_interval_seconds: int = 30
    measure_preprocessing: bool = True
    measure_inference: bool = True
    measure_postprocessing: bool = True


class PerformanceConfig(BaseConfigModel):
    frame_skip: int = 0
    inference_interval_ms: int = 0
    max_inference_fps: int = 0
    warmup_iterations: int = 3
    resize_before_inference: bool = True
    max_concurrent_inferences: int = 1
    batch_size: int = 1
    profiling: ProfilingConfig = ProfilingConfig()


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
class ObjectDetectionModelConfig(BaseConfigModel):
    enabled: bool = True
    backend: str = "ultralytics"
    name: str = "yolov8n"
    weights_path: str = "models/detection/yolov8n.pt"
    weights_url: str = (
        "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    )
    onnx_filename: str = "yolov8n.onnx"
    class_names_path: str = "configs/coco_classes.yaml"
    input_size: Any = [640, 640]
    confidence_threshold: float = 0.45
    iou_threshold: float = 0.45
    classes_filter: Optional[List[int]] = None


class FaceDetectionModelConfig(BaseConfigModel):
    enabled: bool = True
    backend: str = "insightface"
    name: str = "scrfd"
    pack_name: str = "buffalo_s"
    weights_filename: str = "yolov8n-face.onnx"
    weights_url: str = (
        "https://github.com/derronqi/yolov8-face/releases/download/v0.1/yolov8n-face.onnx"
    )
    confidence_threshold: float = 0.50
    nms_iou_threshold: float = 0.45
    input_size: Any = [640, 640]
    min_face_size_px: int = 40


class FaceRecognitionModelConfig(BaseConfigModel):
    enabled: bool = True
    backend: str = "insightface"
    model_name: str = "arcface"
    pack_name: str = "buffalo_s"
    embedding_size: int = 512
    normalize_embeddings: bool = True
    inference_provider: str = "CPUExecutionProvider"


class ModelsConfig(BaseConfigModel):
    download_policy: str = "if_missing"
    verify_downloads: bool = True
    cache_enabled: bool = True
    onnx_providers: List[str] = ["CPUExecutionProvider"]
    object_detection: ObjectDetectionModelConfig = ObjectDetectionModelConfig()
    face_detection: FaceDetectionModelConfig = FaceDetectionModelConfig()
    face_recognition: FaceRecognitionModelConfig = FaceRecognitionModelConfig()


# ---------------------------------------------------------------------------
# recognition
# ---------------------------------------------------------------------------
class RecognitionMatchingConfig(BaseConfigModel):
    top_k: int = 5
    minimum_margin: float = 0.05
    require_quality_check: bool = True


class RecognitionTemporalConfig(BaseConfigModel):
    enabled: bool = True
    min_consecutive_matches: int = 3
    voting_window: int = 5


class RecognitionConfig(BaseConfigModel):
    similarity_metric: str = "cosine"
    similarity_threshold: float = 0.45
    unknown_label: str = "unknown"
    matching: RecognitionMatchingConfig = RecognitionMatchingConfig()
    temporal_confirmation: RecognitionTemporalConfig = RecognitionTemporalConfig()
    duplicate_embedding_threshold: float = 0.98
    save_crops: bool = True
    crop_dir: str = "output/crops/recognized"
    recognize_crop_dir: str = "output/crops/recognized"
    cooldown_seconds: float = 5.0


# ---------------------------------------------------------------------------
# data_collection
# ---------------------------------------------------------------------------
class DataCollectionQualityChecksConfig(BaseConfigModel):
    enabled: bool = True
    min_brightness: int = 60
    max_brightness: int = 200
    blur_threshold: int = 100
    min_face_size_px: int = 80
    reject_multiple_faces: bool = True
    reject_duplicate_images: bool = True


class DataCollectionConfig(BaseConfigModel):
    output_root: str = "data/faces/raw"
    samples_per_pose: int = 15
    capture_interval_seconds: float = 0.3
    max_samples_per_person: int = 300
    allowed_image_extensions: List[str] = [".jpg", ".jpeg", ".png"]
    poses: List[str] = ["straight", "left", "right", "up", "down"]
    quality_checks: DataCollectionQualityChecksConfig = DataCollectionQualityChecksConfig()


# ---------------------------------------------------------------------------
# enrollment
# ---------------------------------------------------------------------------
class EnrollmentConfig(BaseConfigModel):
    auto_generate_identity_code: bool = True
    identity_code_prefix: str = "EMP"
    default_name: str = ""
    default_employee_id: str = ""
    require_unique_employee_id: bool = True
    overwrite_if_exists: bool = False
    require_confirmation_for_overwrite: bool = True
    require_minimum_samples: int = 10
    generate_embeddings_after_collection: bool = True
    allow_incremental_enrollment: bool = True
    metadata_fields: List[str] = ["employee_id", "department"]


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------
class EmbeddingIndexConfig(BaseConfigModel):
    backend: str = "faiss"
    index_type: str = "FlatIP"
    normalize_vectors: bool = True


class DatabasePersistenceConfig(BaseConfigModel):
    autosave: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24


class DatabaseConfig(BaseConfigModel):
    backend: str = "sqlite"
    sqlite_path: str = "data/database/faces.db"
    embedding_index: EmbeddingIndexConfig = EmbeddingIndexConfig()
    persistence: DatabasePersistenceConfig = DatabasePersistenceConfig()


# ---------------------------------------------------------------------------
# tracking
# ---------------------------------------------------------------------------
class TrackingIdentitySmoothingConfig(BaseConfigModel):
    enabled: bool = True
    history_size: int = 5
    minimum_votes: int = 3


class TrackingConfig(BaseConfigModel):
    enabled: bool = True
    type: str = "iou"
    iou_threshold: float = 0.30
    max_age: int = 30
    min_hits: int = 3
    identity_smoothing: TrackingIdentitySmoothingConfig = TrackingIdentitySmoothingConfig()


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------
class CropsConfig(BaseConfigModel):
    save_object_crops: bool = True
    save_face_crops: bool = True
    crop_padding: int = 10
    min_crop_size: int = 40
    cooldown_seconds: float = 5.0
    object_crop_dir: str = "output/crops/objects"
    face_crop_dir: str = "output/crops/faces"
    unknown_crop_dir: str = "output/crops/unknown"


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------
class OutputConfig(BaseConfigModel):
    save_detection_results: bool = True
    save_event_csv: bool = True
    save_annotated_video: bool = False
    image_format: str = "jpg"
    jpeg_quality: int = 90


# ---------------------------------------------------------------------------
# logging
# ---------------------------------------------------------------------------
class LoggingConfig(BaseConfigModel):
    level: str = "INFO"
    max_bytes: int = 5242880
    backup_count: int = 5
    console_enabled: bool = True
    file_enabled: bool = True
    structured_json: bool = False
    event_log_cooldown_seconds: int = 10


# ---------------------------------------------------------------------------
# events
# ---------------------------------------------------------------------------
class EventsConfig(BaseConfigModel):
    enabled: bool = True
    cooldown_seconds: int = 10
    include_unknown_faces: bool = True
    csv_columns: List[str] = [
        "timestamp",
        "event_type",
        "camera_id",
        "class_or_name",
        "confidence",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "crop_path",
        "track_id",
    ]


# ---------------------------------------------------------------------------
# resolution
# ---------------------------------------------------------------------------
class ResolutionItemConfig(BaseConfigModel):
    width: int = 640
    height: int = 480


class ResolutionConfig(BaseConfigModel):
    display: ResolutionItemConfig = ResolutionItemConfig(width=1280, height=720)
    inference: ResolutionItemConfig = ResolutionItemConfig(width=640, height=480)


# ---------------------------------------------------------------------------
# display
# ---------------------------------------------------------------------------
class DisplayConfig(BaseConfigModel):
    mode: str = "grid"
    grid_columns: int = 2
    window_name_prefix: str = "Vision_Suite"
    show_fps: bool = True
    show_confidence: bool = True
    show_track_id: bool = True
    show_unknown_label: bool = True
    bbox_thickness: int = 3
    tight_bbox: bool = True
    circle_overlay: bool = True
    show_landmarks: bool = True
    enrollment_visuals: bool = True


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------
class FeaturesConfig(BaseConfigModel):
    object_detection: bool = True
    face_detection: bool = True
    face_recognition: bool = True
    tracking: bool = True
    data_collection: bool = True
    event_logging: bool = True
    ear_mar: bool = False
    anti_spoofing: bool = False


# ---------------------------------------------------------------------------
# detection
# ---------------------------------------------------------------------------
class DetectionConfig(BaseConfigModel):
    enable_object_detection: bool = True
    enable_face_detection: bool = True


# ---------------------------------------------------------------------------
# system
# ---------------------------------------------------------------------------
class SystemConfig(BaseConfigModel):
    enable_ear_mar: bool = False


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
class EvaluationBenchmarkConfig(BaseConfigModel):
    enabled: bool = False
    warmup_iterations: int = 5
    measurement_iterations: int = 100
    report_cpu_memory: bool = True
    report_latency_percentiles: bool = True


class EvaluationConfig(BaseConfigModel):
    enabled: bool = False
    dataset_dir: str = "data/evaluation"
    output_dir: str = "output/reports"
    recognition_metrics: List[str] = [
        "precision",
        "recall",
        "false_acceptance_rate",
        "false_rejection_rate",
    ]
    benchmark: EvaluationBenchmarkConfig = EvaluationBenchmarkConfig()


# ---------------------------------------------------------------------------
# privacy
# ---------------------------------------------------------------------------
class PrivacyConfig(BaseConfigModel):
    delete_raw_faces_after_enrollment: bool = False
    unknown_crop_retention_days: int = 30
    event_retention_days: int = 90
    require_authorized_enrollment: bool = True


# ---------------------------------------------------------------------------
# AppConfig — top level
# ---------------------------------------------------------------------------
class AppConfig(BaseConfigModel):
    config_version: str = "2.0"
    project: ProjectConfig
    runtime: RuntimeConfig
    paths: PathsConfig
    input: InputConfig
    performance: PerformanceConfig
    models: ModelsConfig
    recognition: RecognitionConfig
    data_collection: DataCollectionConfig
    enrollment: EnrollmentConfig
    database: DatabaseConfig
    tracking: TrackingConfig
    crops: CropsConfig
    output: OutputConfig
    logging: LoggingConfig
    events: EventsConfig
    resolution: ResolutionConfig
    display: DisplayConfig
    features: FeaturesConfig
    evaluation: EvaluationConfig
    privacy: PrivacyConfig


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
        f"config.yaml not found! Checked:\n" + "\n".join(f" - {p}" for p in checked_paths)
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
    """Ensure runtime folders exist from config.yaml paths."""
    from src.core.logger import get_logger  # lazy import to avoid circular import

    logger = get_logger("config_loader")
    project_root = Path(__file__).resolve().parents[2]

    dirs_to_create = [
        project_root / config.paths.models_dir,
        project_root / config.paths.data_dir,
        project_root / config.paths.raw_faces_dir,
        project_root / config.paths.processed_faces_dir,
        project_root / config.paths.embeddings_dir,
        project_root / config.paths.database_dir,
        project_root / config.paths.output_dir,
        project_root / config.paths.logs_dir,
        project_root / config.paths.crops_dir,
        project_root / config.paths.reports_dir,
        project_root / config.paths.temporary_dir,
        # Model subdirectories
        project_root / config.paths.models_dir / "face_detection",
        project_root / config.paths.models_dir / "face_recognition",
        project_root / config.paths.models_dir / "object_detection",
        # Database parent
        Path(config.database.sqlite_path).parent,
        # Legacy face db parent
        project_root / Path(config.paths.legacy_face_db_path).parent,
    ]

    for directory in dirs_to_create:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise
