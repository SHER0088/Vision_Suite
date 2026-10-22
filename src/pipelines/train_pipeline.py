"""
pipelines/train_pipeline.py
---------------------------
Pipeline logic for 'frs train' command.
Build/update the face recognition database (.pkl) from collected embeddings.
"""

from __future__ import annotations

import pickle
import cv2
from pathlib import Path
from typing import Dict, List
import numpy as np

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_event_logger, get_logger, setup_system_logger
from src.recognition.embedder import FaceEmbedder
from src.recognition.matcher import FaceMatcher
from rich.console import Console

logger = get_logger("train_pipeline")
console = Console()


def run_train_pipeline() -> None:
    """Pipeline logic for 'frs train'."""
    config = get_config()
    ensure_runtime_directories(config)
    setup_system_logger(config)
    event_logger = get_event_logger(config)

    console.print("[bold cyan][Pipeline] 'frs train' started.[/bold cyan]")

    # Initialize embedder and matcher
    embedder = FaceEmbedder(config)
    matcher = FaceMatcher(config)

    # Paths
    faces_dir = Path(config.paths.data_dir) / config.paths.faces_dir
    embeddings_dir = Path(config.paths.data_dir) / config.paths.embeddings_dir
    face_db_path = Path(config.paths.face_db_path)

    # Ensure directories exist
    faces_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    face_db_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Starting face recognition database training")
    logger.info("Faces directory: %s", faces_dir)
    logger.info("Embeddings directory: %s", embeddings_dir)
    logger.info("Face DB path: %s", face_db_path)

    # Process each person's face images
    person_embeddings: Dict[str, List[np.ndarray]] = {}
    total_images_processed = 0
    total_faces_detected = 0
    total_embeddings_extracted = 0

    if not faces_dir.exists():
        logger.warning("Faces directory does not exist: %s", faces_dir)
        console.print("[yellow]No face data found for training. Please run 'frs collect' first.[/yellow]")
        return

    # Iterate through each person's directory
    for person_dir in faces_dir.iterdir():
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        logger.info("Processing person: %s", person_name)

        # Collect all images for this person (including subdirectories like raw/)
        image_files = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
            image_files.extend(person_dir.rglob(ext))

        if not image_files:
            logger.warning("No image files found for person: %s", person_name)
            continue

        person_embeddings[person_name] = []
        logger.info("Found %d images for person: %s", len(image_files), person_name)

        # Process each image
        for image_path in image_files:
            total_images_processed += 1
            try:
                # Load image
                img = cv2.imread(str(image_path))
                if img is None:
                    logger.warning("Failed to load image: %s", image_path)
                    continue

                # Extract embedding
                embedding = embedder.get_embedding(img)
                if embedding is not None:
                    person_embeddings[person_name].append(embedding)
                    total_embeddings_extracted += 1
                    logger.debug("Extracted embedding from %s (shape: %s)", image_path.name, embedding.shape)
                else:
                    logger.warning("No face detected in image: %s", image_path)

            except Exception as exc:
                logger.error("Error processing image %s: %s", image_path, exc)

        total_faces_detected += len(person_embeddings[person_name])
        logger.info("Person %s: %d embeddings extracted from %d images",
                   person_name, len(person_embeddings[person_name]), len(image_files))

    # Calculate average embeddings for each person
    logger.info("Calculating average embeddings for %d persons", len(person_embeddings))
    face_db: Dict[str, np.ndarray] = {}

    for person_name, embeddings in person_embeddings.items():
        if embeddings:
            # Stack embeddings and calculate mean
            embeddings_array = np.array(embeddings)
            mean_embedding = np.mean(embeddings_array, axis=0)

            # Normalize the mean embedding
            norm = np.linalg.norm(mean_embedding)
            if norm > 0:
                mean_embedding = mean_embedding / norm

            face_db[person_name] = mean_embedding.astype(np.float32)
            logger.info("Person %s: averaged %d embeddings -> norm=%.4f",
                       person_name, len(embeddings), norm)
        else:
            logger.warning("No valid embeddings found for person: %s", person_name)

    # Save the face database
    if face_db:
        try:
            with open(face_db_path, "wb") as f:
                pickle.dump(face_db, f)

            logger.info("Face recognition database saved successfully to %s", face_db_path)
            logger.info("Database contains %d persons", len(face_db))

            # Also save individual embeddings for reference
            for person_name, embedding in face_db.items():
                person_embedding_dir = embeddings_dir / person_name
                person_embedding_dir.mkdir(parents=True, exist_ok=True)
                embedding_file = person_embedding_dir / f"{person_name}.npy"
                np.save(str(embedding_file), embedding)
                logger.debug("Saved individual embedding for %s to %s", person_name, embedding_file)

            console.print(f"[bold green]Training completed![/bold green]")
            console.print(f"[dim]Processed {total_images_processed} images[/dim]")
            console.print(f"[dim]Detected {total_faces_detected} faces[/dim]")
            console.print(f"[dim]Extracted {total_embeddings_extracted} embeddings[/dim]")
            console.print(f"[dim]Created database with {len(face_db)} persons[/dim]")
            console.print(f"[dim]Database saved to: {face_db_path}[/dim]")

        except Exception as exc:
            logger.error("Failed to save face database: %s", exc)
            console.print(f"[bold red]Error saving face database: {exc}[/bold red]")
    else:
        logger.warning("No face data processed - database not updated")
        console.print("[yellow]No face data found for training. Please run 'frs collect' first.[/yellow]")