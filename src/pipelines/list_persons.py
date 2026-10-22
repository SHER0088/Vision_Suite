"""
pipelines/list_persons.py
-------------------------
Pipeline logic for 'frs list-persons' command.
List all enrolled face profiles.
"""

from __future__ import annotations

import signal
from pathlib import Path

from src.core.config_loader import get_config, ensure_runtime_directories
from src.core.logger import get_event_logger, get_logger, setup_system_logger
from rich.console import Console

logger = get_logger("list_persons")
console = Console()


def run_list_persons() -> None:
    """List all enrolled face profiles."""
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

    console.print("[bold cyan][Management] Listing enrolled persons...[/bold cyan]")

    # Paths
    faces_dir = Path(config.paths.data_dir) / config.paths.faces_dir
    embeddings_dir = Path(config.paths.data_dir) / config.paths.embeddings_dir
    face_db_path = Path(config.paths.face_db_path)

    if shutdown_requested:
        logger.info("Shutdown requested, aborting.")
        return

    if not faces_dir.exists():
        console.print("[yellow]No face data directory found.[/yellow]")
        return

    # Get list of persons from faces directory
    persons_from_faces = set()
    if faces_dir.exists():
        for person_dir in faces_dir.iterdir():
            if shutdown_requested:
                logger.info("Shutdown requested, aborting.")
                return
            if person_dir.is_dir():
                persons_from_faces.add(person_dir.name)

    # Get list of persons from embeddings directory
    persons_from_embeddings = set()
    if embeddings_dir.exists():
        for person_dir in embeddings_dir.iterdir():
            if shutdown_requested:
                logger.info("Shutdown requested, aborting.")
                return
            if person_dir.is_dir():
                persons_from_embeddings.add(person_dir.name)

    # Get list of persons from face database
    persons_from_db = set()
    if face_db_path.exists():
        try:
            import pickle
            with open(face_db_path, "rb") as f:
                face_db = pickle.load(f)
                persons_from_db = set(face_db.keys())
        except Exception as exc:
            logger.error("Failed to load face database: %s", exc)
            console.print(f"[red]Error loading face database: {exc}[/red]")
            if shutdown_requested:
                logger.info("Shutdown requested, aborting.")
                return

    # Combine all sources
    if shutdown_requested:
        logger.info("Shutdown requested, aborting.")
        return
    all_persons = persons_from_faces | persons_from_embeddings | persons_from_db

    if not all_persons:
        console.print("[yellow]No enrolled persons found.[/yellow]")
        console.print("[dim]Run 'frs collect' to enroll persons.[/dim]")
        return

    # Display results
    if shutdown_requested:
        logger.info("Shutdown requested, aborting.")
        return
    console.print(f"[bold green]Found {len(all_persons)} enrolled person(s):[/bold green]")
    for i, person in enumerate(sorted(all_persons), 1):
        if shutdown_requested:
            logger.info("Shutdown requested, aborting.")
            return
        sources = []
        if person in persons_from_faces:
            sources.append("faces")
        if person in persons_from_embeddings:
            sources.append("embeddings")
        if person in persons_from_db:
            sources.append("database")

        source_str = ", ".join(sources)
        console.print(f"  {i}. [cyan]{person}[/cyan] ([dim]{source_str}[/dim])")

    # Show detailed info for each person if requested
    if shutdown_requested:
        logger.info("Shutdown requested, aborting.")
        return
    console.print("\n[dim]For detailed information about a specific person, check:[/dim]")
    console.print(f"  - Faces: {faces_dir}/<person_name>/")
    console.print(f"  - Embeddings: {embeddings_dir}/<person_name>/")
    console.print(f"  - Face database: {face_db_path}")