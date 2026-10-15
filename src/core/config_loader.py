"""
config_loader.py
-----------------
Single source of truth for loading, validating, and exposing config.yaml
throughout the entire application. Nothing in this project reads config.yaml
directly except this file.

Uses Pydantic for strict validation - bad config values fail LOUD and CLEAR
at startup instead of causing mysterious crashes mid-execution.
"""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import yaml

try:
    _pydantic = import_module("pydantic")
except ModuleNotFoundError as exc:
    if exc.name != "pydantic":
        raise
    raise ImportError(
        "Pydantic is required to load the application configuration. "
        "Install it with `python -m pip install pydantic`."
    ) from exc

BaseModel = _pydantic.BaseModel
Field = _pydantic.Field
field_validator = _pydantic.field_validator
model_validator = _pydantic.model_validator
ValidationError = _pydantic.ValidationError


# =============================================================================
# SCHEMA DEFINITIONS
# =============================================================================

class ProjectConfig(BaseModel):
    name: str
    version: str
    device: str = "cpu"


class PathsConfig(BaseModel):
    models_dir: str
    data_dir: str
    faces_dir: str
    embeddings_dir: str
    face_db_path: str
    output_dir: str
    logs_dir: str
    crops_dir: str


class LoggingConfig(BaseModel):
    system_log_level: str = "INFO"
    system_log_max_bytes: int = 5_242_880
    system_log_backup_count: int = 5
    event_log_cooldown_seconds: float = 10.0
    event_csv_columns: List[str]

    @field_validator("system_log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in allowed:
            raise ValueError(f"system_log_level must be one of {allowed}, got '{v}'")
        return v.upper()


class ObjectDetectionModelConfig(BaseModel):
    name: str
    weights_url: str
    weights_filename: str
    onnx_filename: str
    class_names_path: str
    input_size: List[int] = Field(min_length=2, max_length=2)
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    iou_threshold: float = Field(ge=0.0, le=1.0)
    classes_filter: Optional[List[str]] = None


class FaceDetectionModelConfig(BaseModel):
    name: str
    provider: str
    pack_name: str
    confidence_threshold: float = Field(ge=0.0, le=1.0)
    input_size: List[int] = Field(min_length=2, max_length=2)


class FaceRecognitionModelConfig(BaseModel):
    name: str
    provider: str
    pack_name: str
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    embedding_size: int


class ModelsConfig(BaseModel):
    object_detection: ObjectDetectionModelConfig
    face_detection: FaceDetectionModelConfig
    face_recognition: FaceRecognitionModelConfig
    onnx_providers: List[str]


class TrackingConfig(BaseModel):
    type: str
    iou_threshold: float = Field(ge=0.0, le=1.0)
    max_age: int = Field(gt=0)
    min_hits: int = Field(gt=0)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"iou", "bytetrack"}
        if v not in allowed:
            raise ValueError(f"tracking.type must be one of {allowed}, got '{v}'")
        return v


class ResolutionItem(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    def as_tuple(self) -> tuple:
        return (self.width, self.height)


class ResolutionConfig(BaseModel):
    display: ResolutionItem
    inference: ResolutionItem


class DisplayConfig(BaseModel):
    mode: str
    grid_columns: int = Field(gt=0)
    window_name_prefix: str = "Vision_Suite"

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"grid", "multi_window"}
        if v not in allowed:
            raise ValueError(f"display.mode must be one of {allowed}, got '{v}'")
        return v


class CameraConfig(BaseModel):
    id: str
    enabled: bool = True
    type: str
    source: Union[int, str]
    transport: Optional[str] = None
    frame_skip: int = Field(ge=0, default=0)
    reconnect_delay_seconds: float = Field(gt=0, default=5.0)
    display_resolution: Optional[List[int]] = None
    inference_resolution: Optional[List[int]] = None

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        allowed = {"offline", "live", "rtsp"}
        if v not in allowed:
            raise ValueError(f"camera.type must be one of {allowed}, got '{v}'")
        return v

    @field_validator("transport")
    @classmethod
    def validate_transport(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        allowed = {"tcp", "udp"}
        if v not in allowed:
            raise ValueError(f"camera.transport must be one of {allowed} or null, got '{v}'")
        return v

    @model_validator(mode="after")
    def validate_source_matches_type(self) -> "CameraConfig":
        if self.type == "live" and not isinstance(self.source, int):
            try:
                self.source = int(self.source)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Camera '{self.id}': type 'live' requires an integer camera index as source."
                )
        if self.type == "rtsp" and not isinstance(self.source, str):
            raise ValueError(f"Camera '{self.id}': type 'rtsp' requires a URL string as source.")
        if self.type == "offline" and not isinstance(self.source, str):
            raise ValueError(f"Camera '{self.id}': type 'offline' requires a file path string as source.")
        return self


class DetectionConfig(BaseModel):
    enable_object_detection: bool
    enable_face_detection: bool


class CropsConfig(BaseModel):
    save_object_crops: bool
    save_face_crops: bool
    crop_padding: int = Field(ge=0)
    min_crop_size: int = Field(gt=0)
    cooldown_seconds: float = Field(ge=0)
    object_crop_dir: str
    face_crop_dir: str
    unknown_crop_dir: str


class QualityChecksConfig(BaseModel):
    min_brightness: float = Field(ge=0, le=255)
    max_brightness: float = Field(ge=0, le=255)
    blur_threshold: float = Field(ge=0)
    min_face_size_px: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_brightness_range(self) -> "QualityChecksConfig":
        if self.min_brightness >= self.max_brightness:
            raise ValueError("min_brightness must be less than max_brightness")
        return self


class DataCollectionConfig(BaseModel):
    person_name: str
    overwrite_if_exists: bool
    details: Dict[str, Any] = {}
    poses: List[str] = Field(min_length=1)
    samples_per_pose: int = Field(gt=0)
    capture_interval_seconds: float = Field(ge=0)
    quality_checks: QualityChecksConfig

    @field_validator("person_name")
    @classmethod
    def validate_person_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("data_collection.person_name cannot be empty")
        # Keep it filesystem-safe since it becomes a folder name
        invalid_chars = set('<>:"/\\|?*')
        if any(ch in invalid_chars for ch in v):
            raise ValueError(f"person_name '{v}' contains invalid filesystem characters")
        return v


class RecognitionConfig(BaseModel):
    similarity_threshold: float = Field(ge=0.0, le=1.0)
    unknown_label: str
    save_crops: bool
    cooldown_seconds: float = Field(ge=0)
    recognize_crop_dir: str


class SystemConfig(BaseModel):
    enable_ear_mar: bool = False


class AppConfig(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    logging: LoggingConfig
    models: ModelsConfig
    tracking: TrackingConfig
    resolution: ResolutionConfig
    display: DisplayConfig
    cameras: List[CameraConfig]
    detection: DetectionConfig
    crops: CropsConfig
    data_collection: DataCollectionConfig
    recognition: RecognitionConfig
    system: SystemConfig

    @model_validator(mode="after")
    def validate_unique_camera_ids(self) -> "AppConfig":
        ids = [cam.id for cam in self.cameras]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate camera ids found in config: {duplicates}")
        return self

    def get_camera(self, camera_id: str) -> CameraConfig:
        for cam in self.cameras:
            if cam.id == camera_id:
                return cam
        raise KeyError(f"No camera with id '{camera_id}' found in config.yaml")

    def enabled_cameras(self) -> List[CameraConfig]:
        return [cam for cam in self.cameras if cam.enabled]


# =============================================================================
# LOADING LOGIC
# =============================================================================

_CONFIG_CACHE: Optional[AppConfig] = None


def load_config(config_path: Union[str, Path] = "config.yaml") -> AppConfig:
    """
    Loads, validates, and returns the full application config.
    Raises SystemExit with a clear, human-readable error on failure
    instead of letting a raw traceback confuse the user.
    """
    path = Path(config_path)

    if not path.exists():
        print(f"[CONFIG ERROR] config.yaml not found at: {path.resolve()}")
        sys.exit(1)

    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"[CONFIG ERROR] Failed to parse config.yaml - invalid YAML syntax:\n{e}")
        sys.exit(1)

    if raw is None:
        print("[CONFIG ERROR] config.yaml is empty.")
        sys.exit(1)

    try:
        config = AppConfig(**raw)
    except ValidationError as e:
        print("[CONFIG ERROR] config.yaml failed validation:\n")
        for err in e.errors():
            loc = " -> ".join(str(p) for p in err["loc"])
            print(f"  - {loc}: {err['msg']}")
        sys.exit(1)

    return config


def get_config(config_path: Union[str, Path] = "config.yaml", force_reload: bool = False) -> AppConfig:
    """
    Returns a cached singleton config instance so the file isn't re-read
    and re-validated every time a module needs it.
    """
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None or force_reload:
        _CONFIG_CACHE = load_config(config_path)
    return _CONFIG_CACHE


def ensure_runtime_directories(config: AppConfig) -> None:
    """
    Creates all required output/data/model directories at runtime if missing.
    Called once at the start of every CLI command.
    """
    dirs_to_create = [
        config.paths.models_dir,
        config.paths.data_dir,
        config.paths.faces_dir,
        config.paths.embeddings_dir,
        Path(config.paths.face_db_path).parent,
        config.paths.output_dir,
        config.paths.logs_dir,
        config.paths.crops_dir,
        config.crops.object_crop_dir,
        config.crops.face_crop_dir,
        config.crops.unknown_crop_dir,
        config.recognition.recognize_crop_dir,
    ]
    for d in dirs_to_create:
        Path(d).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Quick manual sanity check: run `python src/core/config_loader.py`
    cfg = load_config("config.yaml")
    ensure_runtime_directories(cfg)
    print("config.yaml loaded and validated successfully.")
    print(f"Project: {cfg.project.name} v{cfg.project.version}")
    print(f"Enabled cameras: {[c.id for c in cfg.enabled_cameras()]}")