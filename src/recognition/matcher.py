"""
recognition/matcher.py
----------------------
Face recognition matching using cosine similarity.
Compares face embeddings against a database of known identities.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("matcher")


class FaceMatcher:
    """Matches face embeddings against a known identities database using cosine similarity."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.similarity_threshold = config.recognition.similarity_threshold
        self.unknown_label = config.recognition.unknown_label
        self.face_db_path = Path(config.paths.face_db_path)

        # Ensure the directory exists
        self.face_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize the face database
        self.face_db: Dict[str, np.ndarray] = self._load_face_db()
        logger.info(
            "FaceMatcher ready | threshold=%.2f | unknown_label='%s' | db_size=%d",
            self.similarity_threshold,
            self.unknown_label,
            len(self.face_db),
        )

    def _load_face_db(self) -> Dict[str, np.ndarray]:
        """Load the face recognition database from disk."""
        if self.face_db_path.exists():
            try:
                with open(self.face_db_path, "rb") as f:
                    data = pickle.load(f)
                    logger.info("Loaded face database from %s (%d entries)", self.face_db_path, len(data))
                    return data
            except Exception as exc:
                logger.warning("Failed to load face database from %s: %s", self.face_db_path, exc)
                return {}
        else:
            logger.info("Face database not found at %s, starting with empty database", self.face_db_path)
            return {}

    def save_face_db(self) -> None:
        """Save the face recognition database to disk."""
        try:
            with open(self.face_db_path, "wb") as f:
                pickle.dump(self.face_db, f)
            logger.info("Saved face database to %s (%d entries)", self.face_db_path, len(self.face_db))
        except Exception as exc:
            logger.error("Failed to save face database to %s: %s", self.face_db_path, exc)

    def add_face(self, person_name: str, embedding: np.ndarray) -> None:
        """
        Add or update a person's face embedding in the database.

        Args:
            person_name: Identifier for the person
            embedding: Normalized face embedding vector
        """
        if embedding is None:
            logger.warning("Cannot add face with None embedding for person: %s", person_name)
            return

        # Ensure embedding is normalized
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized_embedding = embedding / norm
        else:
            normalized_embedding = embedding

        self.face_db[person_name] = normalized_embedding.astype(np.float32)
        logger.debug("Added/updated face for person: %s", person_name)

    def remove_face(self, person_name: str) -> bool:
        """
        Remove a person's face embedding from the database.

        Args:
            person_name: Identifier for the person to remove

        Returns:
            True if person was found and removed, False otherwise
        """
        if person_name in self.face_db:
            del self.face_db[person_name]
            logger.info("Removed face for person: %s", person_name)
            return True
        else:
            logger.warning("Person not found in database: %s", person_name)
            return False

    def match_face(self, embedding: np.ndarray) -> Tuple[str, float]:
        """
        Match a face embedding against the database.

        Args:
            embedding: Normalized face embedding vector to match

        Returns:
            Tuple of (person_name, similarity_score)
            person_name is the best match or unknown_label if below threshold
            similarity_score is the cosine similarity (0-1)
        """
        if embedding is None or len(self.face_db) == 0:
            return self.unknown_label, 0.0

        # Ensure query embedding is normalized
        query_norm = np.linalg.norm(embedding)
        if query_norm > 0:
            query_embedding = embedding / query_norm
        else:
            query_embedding = embedding

        best_match = self.unknown_label
        best_similarity = 0.0

        # Compare against all known faces
        for person_name, db_embedding in self.face_db.items():
            # Calculate cosine similarity
            similarity = float(np.dot(query_embedding, db_embedding))

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = person_name

        # Check if similarity meets threshold
        if best_similarity < self.similarity_threshold:
            return self.unknown_label, best_similarity
        else:
            return best_match, best_similarity

    def match_faces(self, embeddings: List[Optional[np.ndarray]]) -> List[Tuple[str, float]]:
        """
        Match multiple face embeddings against the database.

        Args:
            embeddings: List of face embedding vectors (or None)

        Returns:
            List of (person_name, similarity_score) tuples
        """
        results = []
        for embedding in embeddings:
            if embedding is None:
                results.append((self.unknown_label, 0.0))
            else:
                results.append(self.match_face(embedding))
        return results

    def get_database_info(self) -> Dict[str, int]:
        """Get information about the face database."""
        return {
            "total_persons": len(self.face_db),
            "database_path": str(self.face_db_path),
        }