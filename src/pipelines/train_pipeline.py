"""
Pipeline logic for 'frs train' command.

Scans data/faces/raw/<name>/ for collected face images, generates embeddings
for each, and enrolls every person in the SQLite + FAISS recognition database.
This enables multi-sample enrollment (60-80 images per person across multiple
poses) captured via 'frs collect'.
"""

from __future__ import annotations

import cv2
from pathlib import Path

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_logger
from src.detection.face_detector import FaceDetector
from src.recognition.embedder import FaceEmbedder
from src.recognition.matcher import FaceMatcher
from rich.console import Console

logger = get_logger("train_pipeline")
console = Console()

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def run_train_pipeline() -> None:
    """Train recognition model from collected raw face images.

    Scans ``data/faces/raw/<name>/`` for images, generates embeddings for
    each detected face, and enrolls every person in SQLite + FAISS with
    all their collected samples.
    """
    config = get_config()
    ensure_runtime_directories(config)

    raw_dir = Path(config.paths.raw_faces_dir)
    if not raw_dir.exists():
        console.print(f"[bold red]Raw faces directory not found: {raw_dir}[/bold red]")
        console.print("[yellow]Run 'frs collect' to capture images first.[/yellow]")
        return

    person_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not person_dirs:
        console.print(f"[bold red]No person directories found in {raw_dir}[/bold red]")
        console.print("[yellow]Run 'frs collect' to capture images first.[/yellow]")
        return

    face_detector = FaceDetector(config)
    embedder = FaceEmbedder(config)
    matcher = FaceMatcher(config)

    total_enrolled = 0
    total_images = 0

    for person_dir in person_dirs:
        name = person_dir.name
        image_paths = sorted(
            p for p in person_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )

        if not image_paths:
            console.print(f"[yellow]Skipping {name}: no images found[/yellow]")
            continue

        console.print(f"[cyan]Processing {name}: {len(image_paths)} images[/cyan]")

        embeddings: list = []
        valid_images: list[str] = []

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            face_dets = face_detector.detect(img)
            if len(face_dets) == 0:
                continue

            # If multiple faces, use the largest one
            if len(face_dets) > 1:
                face_dets.sort(
                    key=lambda d: (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1]),
                    reverse=True,
                )

            fd = face_dets[0]
            x1, y1, x2, y2 = fd.bbox
            face_img = img[y1:y2, x1:x2]
            if face_img.size == 0:
                continue

            embedding = embedder.get_embedding(face_img)
            if embedding is None:
                continue

            embeddings.append(embedding)
            valid_images.append(str(img_path))
            total_images += 1

        if not embeddings:
            console.print(f"[yellow]Skipping {name}: no valid faces detected[/yellow]")
            continue

        # Enroll with first embedding (creates identity)
        identity_id = matcher.enroll(
            None,  # auto-generate identity_code
            name,
            valid_images[0],
            embeddings[0],
            employee_id="",
            metadata="{}",
        )

        if identity_id is None:
            console.print(
                f"[yellow]Skipping {name}: enrollment failed "
                f"(duplicate identity_code or employee_id).[/yellow]"
            )
            continue

        # Add remaining embeddings to the same identity
        for emb, img_path in zip(embeddings[1:], valid_images[1:]):
            matcher.add_embedding(identity_id, emb, img_path)

        total_enrolled += 1
        console.print(
            f"[green]Enrolled {name} with {len(embeddings)} samples[/green]"
        )

    console.print(f"\n[bold green]Training complete![/bold green]")
    console.print(f"[dim]Persons enrolled: {total_enrolled}[/dim]")
    console.print(f"[dim]Total images processed: {total_images}[/dim]")

    info = matcher.get_database_info()
    console.print(f"[dim]Identities in DB: {info['total_identities']}[/dim]")
    console.print(f"[dim]Embeddings in DB: {info['total_embeddings']}[/dim]")
