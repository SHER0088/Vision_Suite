"""Driver functions for Vision_Suite CLI commands."""

from __future__ import annotations

from src.pipelines.collect_pipeline import (
    run_collect_pipeline as _run_collect,
    run_enroll_pipeline as _run_enroll,
)
from src.pipelines.detect_pipeline import run_detect_pipeline as _run_detect
from src.pipelines.crops_pipeline import run_crops_pipeline as _run_crops
from src.pipelines.recognize_pipeline import run_recognize_pipeline as _run_recognize
from src.pipelines.system_pipeline import run_system_pipeline as _run_system
from src.pipelines.list_persons import run_list_persons as _run_list
from src.pipelines.delete_person import run_delete_person as _run_delete
from src.pipelines.train_pipeline import run_train_pipeline as _run_train


def run_collect_pipeline(
    name: str | None = None,
    samples_per_pose: int | None = None,
) -> None:
    """Camera-based data collection with pose guidance."""
    _run_collect(name=name, samples_per_pose=samples_per_pose)


def run_enroll_pipeline(
    image_path: str | None = None,
    name: str | None = None,
    employee_id: str | None = None,
    metadata: str = "{}",
) -> None:
    """Enroll person(s) with reference photo(s). Auto-discovers from data/ if no path given."""
    _run_enroll(image_path, name=name, employee_id=employee_id, metadata=metadata)


def run_recognize_pipeline(source: str | None = None) -> None:
    """Recognize faces from camera or file."""
    _run_recognize(source=source)


def run_list_persons() -> None:
    """List all enrolled identities."""
    _run_list()


def run_delete_person(identity_code: str) -> None:
    """Delete a person by identity code."""
    _run_delete(identity_code)


def run_detect_pipeline() -> None:
    """Run live object & face detection across configured cameras."""
    _run_detect()


def run_crops_pipeline() -> None:
    """Detect objects/faces and save cropped images per class."""
    _run_crops()


def run_system_pipeline(source: str | None = None) -> None:
    """Run full production system pipeline (Object + Face + Recognition + Logging)."""
    _run_system(source=source)


def run_train_pipeline() -> None:
    """Rebuild FAISS index from SQLite (no model retraining)."""
    _run_train()
