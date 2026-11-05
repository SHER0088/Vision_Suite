"""CLI Entrypoint for Vision_Suite.

Commands:
  frs enroll [IMAGE_PATH] [--name NAME] [--employee-id EMP_ID]   Enroll person(s) (auto-discovers data/faces/raw/)
  frs recognize [source]                                         Recognize faces (camera or file)
  frs list-persons                                               List enrolled identities
  frs delete-person <identity_code>                              Delete a person
  frs detect                                                     Live object & face detection
  frs crops                                                      Detect and save cropped images
  frs train                                                      Rebuild FAISS index from SQLite
  frs system [source]                                            Full production pipeline (camera or file)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from frs_driver import (
    run_collect_pipeline,
    run_crops_pipeline,
    run_delete_person,
    run_detect_pipeline,
    run_enroll_pipeline,
    run_list_persons,
    run_recognize_pipeline,
    run_system_pipeline,
    run_train_pipeline,
)

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    name="vision",
    help="Vision_Suite: Minimal Face Recognition CLI",
    context_settings=CONTEXT_SETTINGS,
)

frs_app = typer.Typer(
    name="frs",
    help="Face Recognition Commands",
    context_settings=CONTEXT_SETTINGS,
)
app.add_typer(frs_app, name="frs")


def print_top_level_help() -> None:
    """Print custom-formatted help for the top-level CLI."""
    invoke = f"python {sys.argv[0]}"
    print(f"Usage: {invoke} frs <command> [ARGS...]")
    print()
    print("Commands:")
    commands = [
        (
            "frs enroll [IMAGE_PATH] [--name NAME] [--employee-id EMP_ID]",
            "Enroll person(s) (auto-discovers data/faces/raw/ if no path given)",
        ),
        (
            "frs recognize [source]",
            "Recognize faces from camera or image/video file",
        ),
        ("frs list-persons", "List all enrolled identities"),
        (
            "frs delete-person <identity_code>",
            "Delete a person by identity code",
        ),
        ("frs detect", "Live object & face detection across configured cameras"),
        ("frs crops", "Detect objects/faces and save cropped images per class"),
        ("frs collect", "Camera-based data collection with pose guidance"),
        ("frs train", "Train recognition model from collected face images"),
        (
            "frs system [source]",
            "Full production pipeline (camera or image/video file)",
        ),
    ]
    for cmd, desc in commands:
        print(f"  {invoke} {cmd:<48} -- {desc}")
    print()
    print(f"Use '{invoke} frs <command> --help' for more information on a command.")


@frs_app.command("enroll", context_settings=CONTEXT_SETTINGS)
def enroll(
    image_path: Optional[str] = typer.Argument(
        None, help="Path to reference photo(s). Auto-discovers from data/faces/raw/ if omitted"
    ),
    name: Optional[str] = typer.Option(
        None, help="Person's full name (prompts interactively if omitted)"
    ),
    employee_id: Optional[str] = typer.Option(
        None, help="Employee ID (prompts interactively if omitted)"
    ),
    metadata: str = typer.Option("{}", help="JSON metadata string"),
) -> None:
    """Enroll person(s) with reference photo(s).

    If no IMAGE_PATH is given, images are auto-discovered from data/faces/raw/.
    Missing name/employee_id are collected via interactive prompts.
    """
    run_enroll_pipeline(image_path, name=name, employee_id=employee_id, metadata=metadata)


@frs_app.command("recognize", context_settings=CONTEXT_SETTINGS)
def recognize(
    source: str = typer.Argument(None, help="Optional image or video file path"),
) -> None:
    """Recognize faces from camera or an image/video file."""
    run_recognize_pipeline(source=source)


@frs_app.command("list-persons", context_settings=CONTEXT_SETTINGS)
def list_persons() -> None:
    """List all enrolled identities."""
    run_list_persons()


@frs_app.command("delete-person", context_settings=CONTEXT_SETTINGS)
def delete_person(
    identity_code: str = typer.Argument(..., help="Identity code of the person to delete"),
) -> None:
    """Delete a person from SQLite + FAISS."""
    run_delete_person(identity_code)


@frs_app.command("collect", context_settings=CONTEXT_SETTINGS)
def collect(
    name: Optional[str] = typer.Option(
        None, help="Person's name (defaults to config enrollment.default_name)"
    ),
    samples_per_pose: Optional[int] = typer.Option(
        None, help="Samples per pose (defaults to config data_collection.samples_per_pose)"
    ),
) -> None:
    """Camera-based data collection with pose guidance.

    Captures multiple face images per pose (straight / left / right / up / down)
    and saves them to data/faces/raw/<name>/. Use 'frs train' afterwards.
    """
    run_collect_pipeline(name=name, samples_per_pose=samples_per_pose)


@frs_app.command("detect", context_settings=CONTEXT_SETTINGS)
def detect() -> None:
    """Run live object & face detection across configured cameras."""
    run_detect_pipeline()


@frs_app.command("crops", context_settings=CONTEXT_SETTINGS)
def crops() -> None:
    """Detect objects/faces and save cropped images per class."""
    run_crops_pipeline()


@frs_app.command("train", context_settings=CONTEXT_SETTINGS)
def train() -> None:
    """Rebuild FAISS index from SQLite (no model retraining)."""
    run_train_pipeline()


@frs_app.command("system", context_settings=CONTEXT_SETTINGS)
def system(
    source: str = typer.Argument(None, help="Optional image or video file path"),
) -> None:
    """Run full production pipeline (camera or image/video file)."""
    run_system_pipeline(source=source)


if __name__ == "__main__":
    # Intercept top-level help for custom formatting
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help"):
        print_top_level_help()
        sys.exit(0)
    app()
