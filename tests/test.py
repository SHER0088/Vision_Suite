"""CLI Entrypoint for Vision_Suite.

Handles argument parsing and routes commands to driver logic in test/frs_driver.py.
"""

from __future__ import annotations

import sys
from pathlib import Path
import typer

# Ensure project root is in Python path when executing test/test.py
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from frs_driver import (
    run_collect_pipeline,
    run_crops_pipeline,
    run_delete_person,
    run_detect_pipeline,
    run_list_persons,
    run_recognize_pipeline,
    run_system_pipeline,
    run_train_pipeline,
)

# Force -h and --help to work across all command levels
CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    name="vision",
    help="Vision_Suite: Combined Object Detection, Face Detection & Face Recognition CLI Tool",
    context_settings=CONTEXT_SETTINGS,
)

frs_app = typer.Typer(
    name="frs",
    help="Face & Object Suite Commands",
    context_settings=CONTEXT_SETTINGS,
)
app.add_typer(frs_app, name="frs")


@frs_app.command("detect", context_settings=CONTEXT_SETTINGS)
def detect() -> None:
    """Run live object & face detection across configured camera sources (No saving)."""
    run_detect_pipeline()


@frs_app.command("crops", context_settings=CONTEXT_SETTINGS)
def crops() -> None:
    """Detect objects/faces and save cropped images per class with cooldown logic."""
    run_crops_pipeline()


@frs_app.command("collect", context_settings=CONTEXT_SETTINGS)
def collect() -> None:
    """Collect face samples for a person based purely on config.yaml settings."""
    run_collect_pipeline()


@frs_app.command("train", context_settings=CONTEXT_SETTINGS)
def train() -> None:
    """Build/update the face recognition database (.pkl) from collected embeddings."""
    run_train_pipeline()


@frs_app.command("recognize", context_settings=CONTEXT_SETTINGS)
def recognize() -> None:
    """Recognize registered faces, draw labels/bboxes, and save crops with timestamps."""
    run_recognize_pipeline()


@frs_app.command("system", context_settings=CONTEXT_SETTINGS)
def system() -> None:
    """Run full production system pipeline (Object + Face + Recognition + Logging)."""
    run_system_pipeline()


@frs_app.command("list-persons", context_settings=CONTEXT_SETTINGS)
def list_persons() -> None:
    """List all registered/collected persons in the dataset."""
    run_list_persons()


@frs_app.command("delete-person", context_settings=CONTEXT_SETTINGS)
def delete_person(
    name: str = typer.Argument(..., help="Name of the person to delete from database and crops"),
) -> None:
    """Delete a registered person's embeddings and saved dataset."""
    run_delete_person(name)


if __name__ == "__main__":
    app()