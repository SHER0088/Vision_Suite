"""
Pipeline logic for 'frs list-persons' command.
List all enrolled identities from SQLite + FAISS.
"""

from __future__ import annotations

from src.core.config_loader import ensure_runtime_directories, get_config
from src.core.logger import get_logger
from src.recognition.matcher import FaceMatcher
from rich.console import Console

logger = get_logger("list_persons")
console = Console()


def run_list_persons() -> None:
    """List all enrolled face profiles."""
    config = get_config()
    ensure_runtime_directories(config)

    matcher = FaceMatcher(config)
    persons = matcher.list_persons()

    if not persons:
        console.print("[yellow]No enrolled persons found.[/yellow]")
        console.print("[dim]Run 'frs enroll' to add a person.[/dim]")
        return

    console.print(f"[bold green]Found {len(persons)} enrolled person(s):[/bold green]")
    for i, p in enumerate(persons, 1):
        console.print(
            f"  {i}. [cyan]{p['name']}[/cyan] "
            f"([dim]code={p['identity_code']}, id={p['id']}, "
            f"embeddings={p['embedding_count']}[/dim])"
        )
