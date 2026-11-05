"""
Pipeline logic for 'frs delete-person' command.
Delete a person from SQLite + FAISS. Not model retraining.
"""

from __future__ import annotations

from src.core.config_loader import ensure_runtime_directories, get_config
from src.core.logger import get_logger
from src.recognition.matcher import FaceMatcher
from rich.console import Console

logger = get_logger("delete_person")
console = Console()


def run_delete_person(identity_code: str) -> None:
    """Delete a person by identity code."""
    config = get_config()
    ensure_runtime_directories(config)

    matcher = FaceMatcher(config)
    success = matcher.delete_person(identity_code)

    if success:
        console.print(f"[bold green]Deleted identity: {identity_code}[/bold green]")
    else:
        console.print(f"[yellow]Identity not found: {identity_code}[/yellow]")
