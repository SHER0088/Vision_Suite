"""
pipelines/delete_person.py
--------------------------
Pipeline logic for 'frs delete-person' command.
Delete a specific enrolled person's profile and associated data.
"""

from __future__ import annotations

import shutil
import signal
from pathlib import Path

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_event_logger, get_logger, setup_system_logger
from rich.console import Console

logger = get_logger("delete_person")

def run_delete_person(person_name: str) -> None:
    """Delete a specific enrolled face profile."""
    config = get_config()
    ensure_runtime_directories(config)
    setup_system_logger(config)
    event_logger = get_event_logger(config)

    # Signal handling for graceful shutdown
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Signal %d received, initiating shutdown...", signum)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    console.print(f"[bold red][Management] Deleting person profile: {person_name}[/bold red]")

    # Paths
    faces_dir = Path(config.paths.data_dir) / config.paths.faces_dir
    embeddings_dir = Path(config.paths.data_dir) / config.paths.embeddings_dir
    face_db_path = Path(config.paths.face_db_path)
    recognize_crop_dir = Path(config.paths.output_dir) / config.recognition.recognize_crop_dir

    # Track what was deleted
    deleted_items = []

    # 1. Delete from faces directory
    if shutdown_requested:
        logger.info("Shutdown requested, aborting deletion.")
        return
    person_faces_dir = faces_dir / person_name
    if person_faces_dir.exists():
        try:
            shutil.rmtree(person_faces_dir)
            deleted_items.append(f"faces/{person_name}")
            logger.info("Deleted faces directory for person: %s", person_name)
        except Exception as exc:
            logger.error("Failed to delete faces directory for %s: %s", person_name, exc)
            console.print(f"[red]Error deleting faces directory: {exc}[/red]")
            return

    # 2. Delete from embeddings directory
    if shutdown_requested:
        logger.info("Shutdown requested, aborting deletion.")
        return
    person_embeddings_dir = embeddings_dir / person_name
    if person_embeddings_dir.exists():
        try:
            shutil.rmtree(person_embeddings_dir)
            deleted_items.append(f"embeddings/{person_name}")
            logger.info("Deleted embeddings directory for person: %s", person_name)
        except Exception as exc:
            logger.error("Failed to delete embeddings directory for %s: %s", person_name, exc)
            console.print(f"[red]Error deleting embeddings directory: {exc}[/red]")
            return

    # 3. Delete from face database
    if shutdown_requested:
        logger.info("Shutdown requested, aborting deletion.")
        return
    if face_db_path.exists():
        try:
            import pickle
            with open(face_db_path, "rb") as f:
                face_db = pickle.load(f)

            if person_name in face_db:
                del face_db[person_name]
                with open(face_db_path, "wb") as f:
                    pickle.dump(face_db, f)
                deleted_items.append(f"face database entry for {person_name}")
                logger.info("Deleted person %s from face database", person_name)
            else:
                logger.warning("Person %s not found in face database", person_name)
        except Exception as exc:
            logger.error("Failed to update face database: %s", exc)
            console.print(f"[red]Error updating face database: {exc}[/red]")
            return

    # 4. Delete recognized crops (optional - crops with this person's name)
    if shutdown_requested:
        logger.info("Shutdown requested, aborting deletion.")
        return
    if recognize_crop_dir.exists():
        deleted_crops = []
        try:
            for crop_file in recognize_crop_dir.glob(f"{person_name}_*"):
                if shutdown_requested:
                    logger.info("Shutdown requested, aborting deletion.")
                    return
                crop_file.unlink()
                deleted_crops.append(crop_file.name)

            if deleted_crops:
                deleted_items.append(f"{len(deleted_crops)} recognized crop(s)")
                logger.info("Deleted %d recognized crop(s) for person: %s", len(deleted_crops), person_name)
        except Exception as exc:
            logger.error("Failed to delete recognized crops for %s: %s", person_name, exc)
            console.print(f"[red]Error deleting recognized crops: {exc}[/red]")
            return

    # Report results
    if shutdown_requested:
        logger.info("Shutdown requested, aborting deletion.")
        return
    if deleted_items:
        console.print(f"[bold green]Successfully deleted profile for:[/bold green] [cyan]{person_name}[/cyan]")
        for item in deleted_items:
            console.print(f"  - [dim]{item}[/dim]")
        logger.info("Completed deletion of person profile: %s", person_name)
    else:
        console.print(f"[yellow]No data found for person: {person_name}[/yellow]")
        logger.warning("No data found for person: %s", person_name)