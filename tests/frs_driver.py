"""Driver functions (Pipelines) for Vision_Suite commands."""

from __future__ import annotations

import time
import cv2
from rich.console import Console

from src.core.config_loader import ensure_runtime_directories, get_config
from src.core.logger import get_event_logger, get_logger, setup_system_logger
from src.core.tracker import Tracker
from src.core.video_source import VideoSourceManager
from src.detection.face_detector import FaceDetector
from src.detection.object_detector import ObjectDetector

logger = get_logger("frs_driver")
console = Console()


def run_detect_pipeline() -> None:
    """Pipeline logic for 'frs detect'."""
    from src.pipelines.detect_pipeline import run_detect_pipeline as _run_detect_pipeline
    _run_detect_pipeline()


def run_crops_pipeline() -> None:
    """Pipeline logic for 'frs crops'."""
    from src.pipelines.crops_pipeline import run_crops_pipeline as _run_crops_pipeline
    _run_crops_pipeline()


def run_collect_pipeline() -> None:
    """Pipeline logic for 'frs collect' (Fully config.yaml driven)."""
    from src.pipelines.collect_pipeline import run_collect_pipeline as _run_collect_pipeline
    _run_collect_pipeline()


def run_train_pipeline() -> None:
    """Pipeline logic for 'frs train'."""
    from src.pipelines.train_pipeline import run_train_pipeline as _run_train_pipeline
    _run_train_pipeline()


def run_recognize_pipeline() -> None:
    """Pipeline logic for 'frs recognize'."""
    from src.pipelines.recognize_pipeline import run_recognize_pipeline as _run_recognize_pipeline
    _run_recognize_pipeline()


def run_system_pipeline() -> None:
    """Pipeline logic for 'frs system'."""
    from src.pipelines.system_pipeline import run_system_pipeline as _run_system_pipeline
    _run_system_pipeline()


def run_list_persons() -> None:
    """List all enrolled face profiles."""
    from src.pipelines.list_persons import run_list_persons as _run_list_persons
    _run_list_persons()


def run_delete_person(person_name: str) -> None:
    """Delete a specific enrolled face profile."""
    from src.pipelines.delete_person import run_delete_person as _run_delete_person
    _run_delete_person(person_name)