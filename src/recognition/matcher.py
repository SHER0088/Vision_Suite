"""
Face recognition matching using FAISS + SQLite.

Identity metadata lives in SQLite (identities + face_images tables).
FAISS IndexFlatIP provides fast inner-product (≈ cosine) search over
L2-normalized embeddings. The FAISS index is rebuilt from SQLite on
startup and after add/delete operations.

Add/delete is a database + vector-index operation — the FaceNet model
is never retrained.
"""

from __future__ import annotations

import datetime
import pickle
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("matcher")


class FaceMatcher:
    """Matches face embeddings against known identities.

    Storage:
      - SQLite: identity metadata + embedding blobs + image paths
      - FAISS:  in-memory vector index for fast similarity search
    """

    def __init__(self, config: AppConfig):
        self.config = config
        self.similarity_threshold = float(config.recognition.similarity_threshold)
        self.unknown_label = config.recognition.unknown_label
        self.face_db_path = Path(config.paths.legacy_face_db_path)
        self.db_path = Path(config.database.sqlite_path)
        self.embedding_size = int(config.models.face_recognition.embedding_size)

        self.face_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory: identity_id -> {name, identity_code, embeddings: [np.ndarray, ...]}
        self.face_db: Dict[int, dict] = {}
        self._id_map: Dict[int, str] = {}  # FAISS vector ID -> identity name

        self._init_sqlite()
        self._load_from_sqlite()

        # FAISS index
        self._index = None
        self._faiss = None
        self._use_faiss = self._init_faiss()
        self._rebuild_faiss_index()

        logger.info(
            "FaceMatcher ready | threshold=%.2f | unknown='%s' | "
            "identities=%d | backend=%s",
            self.similarity_threshold, self.unknown_label,
            len(self.face_db), "faiss" if self._use_faiss else "numpy",
        )

    # ------------------------------------------------------------------
    # SQLite
    # ------------------------------------------------------------------

    def _init_sqlite(self) -> None:
        """Create SQLite tables if they don't exist."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS identities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_code TEXT UNIQUE NOT NULL,
                    employee_id TEXT DEFAULT '',
                    name TEXT NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL
                )
            """)
            # Migration: add employee_id column to existing tables
            try:
                conn.execute("ALTER TABLE identities ADD COLUMN employee_id TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS face_images (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    identity_id INTEGER NOT NULL,
                    image_path TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (identity_id) REFERENCES identities(id)
                )
            """)
            conn.commit()
        finally:
            conn.close()

    def _load_from_sqlite(self) -> None:
        """Load all active identities and embeddings from SQLite into memory."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            rows = conn.execute("""
                SELECT i.id, i.identity_code, i.employee_id, i.name, i.metadata, i.status,
                       fi.id, fi.image_path, fi.embedding, fi.status
                FROM identities i
                LEFT JOIN face_images fi ON fi.identity_id = i.id AND fi.status = 'active'
                WHERE i.status = 'active'
            """).fetchall()
        finally:
            conn.close()

        # Group by identity
        by_id: Dict[int, dict] = {}
        for row in rows:
            ident_id, identity_code, employee_id, name, metadata, ident_status, \
                img_id, image_path, embedding_blob, img_status = row

            if ident_id not in by_id:
                by_id[ident_id] = {
                    "identity_code": identity_code,
                    "employee_id": employee_id or "",
                    "name": name,
                    "metadata": metadata or "{}",
                    "embeddings": [],
                }

            if img_id is not None and embedding_blob is not None:
                try:
                    emb = np.frombuffer(embedding_blob, dtype=np.float32)
                    if emb.shape[0] == self.embedding_size:
                        by_id[ident_id]["embeddings"].append(emb)
                except ValueError:
                    continue

        self.face_db = by_id

    def _save_identity_to_sqlite(
        self, identity_code: str, employee_id: str, name: str, metadata: str, image_path: str, embedding: np.ndarray
    ) -> int:
        """Insert a new identity + face image + embedding into SQLite. Returns identity id."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "INSERT INTO identities (identity_code, employee_id, name, metadata, status, created_at) "
                "VALUES (?, ?, ?, ?, 'active', ?)",
                (identity_code, employee_id, name, metadata, datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )
            identity_id = cur.lastrowid

            emb_blob = np.asarray(embedding, dtype=np.float32).tobytes()
            conn.execute(
                "INSERT INTO face_images (identity_id, image_path, embedding, status, created_at) "
                "VALUES (?, ?, ?, 'active', ?)",
                (identity_id, image_path, emb_blob, datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )
            conn.commit()
            return int(identity_id)
        finally:
            conn.close()

    def _deactivate_identity_in_sqlite(self, identity_code: str) -> bool:
        """Soft-delete an identity and its images in SQLite."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            cur = conn.execute(
                "UPDATE identities SET status = 'deleted' WHERE identity_code = ?",
                (identity_code,),
            )
            conn.execute(
                "UPDATE face_images SET status = 'deleted' "
                "WHERE identity_id IN (SELECT id FROM identities WHERE identity_code = ?)",
                (identity_code,),
            )
            conn.commit()
            return cur.rowcount > 0
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # FAISS
    # ------------------------------------------------------------------

    def _init_faiss(self) -> bool:
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(self.embedding_size)
            return True
        except ImportError:
            self._index = None
            self._faiss = None
            return False

    def _rebuild_faiss_index(self) -> None:
        """Rebuild the FAISS index from the current in-memory face_db."""
        if self._use_faiss and self._index is not None:
            self._index = self._faiss.IndexFlatIP(self.embedding_size)
        self._id_map = {}
        vid = 0

        for ident_id, data in self.face_db.items():
            for emb in data["embeddings"]:
                values = np.asarray(emb, dtype=np.float32).reshape(1, -1)
                if values.shape[1] != self.embedding_size:
                    continue
                if self._use_faiss and self._index is not None:
                    self._index.add(values)
                self._id_map[vid] = data["name"]
                vid += 1

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enroll(
        self,
        identity_code: Optional[str],
        name: str,
        image_path: str,
        embedding: np.ndarray,
        employee_id: str = "",
        metadata: str = "{}",
    ) -> Optional[int]:
        """Enroll a new person with one face image and embedding.

        If identity_code is None and auto_generate_identity_code is enabled,
        a unique code is generated as <prefix>-<N> where N is the next
        available number. employee_id must be unique when require_unique_employee_id
        is enabled.

        Returns the new identity id, or None on failure.
        """
        # Auto-generate identity_code if enabled and not provided
        if identity_code is None:
            if self.config.enrollment.auto_generate_identity_code:
                prefix = self.config.enrollment.identity_code_prefix
                existing_codes = {
                    d["identity_code"] for d in self.face_db.values()
                }
                n = 1
                while f"{prefix}-{n:03d}" in existing_codes:
                    n += 1
                identity_code = f"{prefix}-{n:03d}"
            else:
                logger.error("identity_code is required when auto_generate_identity_code is disabled")
                return None

        # Check identity_code uniqueness
        if identity_code in {d["identity_code"] for d in self.face_db.values()}:
            logger.warning("Identity code already exists: %s", identity_code)
            return None

        # Check employee_id uniqueness
        if employee_id and self.config.enrollment.require_unique_employee_id:
            if employee_id in {d["employee_id"] for d in self.face_db.values() if d["employee_id"]}:
                logger.warning("Employee ID already exists: %s", employee_id)
                return None

        normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(normalized))
        if norm > 0:
            normalized = normalized / norm

        identity_id = self._save_identity_to_sqlite(
            identity_code, employee_id, name, metadata, image_path, normalized
        )

        # Update in-memory db
        self.face_db[identity_id] = {
            "identity_code": identity_code,
            "employee_id": employee_id,
            "name": name,
            "metadata": metadata,
            "embeddings": [normalized.copy()],
        }

        # Add to FAISS index
        values = normalized.reshape(1, -1).astype(np.float32)
        if self._use_faiss and self._index is not None:
            self._index.add(values)
        vid = len(self._id_map)
        self._id_map[vid] = name

        logger.info("Enrolled identity: %s (%s, employee_id=%s)", name, identity_code, employee_id)
        return identity_id

    def add_embedding(
        self, identity_id: int, embedding: np.ndarray, image_path: str = ""
    ) -> None:
        """Add an additional embedding to an existing identity.

        Useful for multi-sample enrollment (e.g. camera-based data collection
        capturing 60-80 images per person across multiple poses).
        """
        if identity_id not in self.face_db:
            logger.warning("Identity %d not found, cannot add embedding", identity_id)
            return

        normalized = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(normalized))
        if norm > 0:
            normalized = normalized / norm

        # Add to in-memory db
        self.face_db[identity_id]["embeddings"].append(normalized.copy())

        # Add to FAISS index
        values = normalized.reshape(1, -1).astype(np.float32)
        if self._use_faiss and self._index is not None:
            self._index.add(values)
        vid = len(self._id_map)
        self._id_map[vid] = self.face_db[identity_id]["name"]

        # Persist to SQLite
        conn = sqlite3.connect(str(self.db_path))
        try:
            emb_blob = normalized.tobytes()
            conn.execute(
                "INSERT INTO face_images (identity_id, image_path, embedding, status, created_at) "
                "VALUES (?, ?, ?, 'active', ?)",
                (identity_id, image_path, emb_blob, datetime.datetime.now(datetime.timezone.utc).isoformat()),
            )
            conn.commit()
        finally:
            conn.close()

        logger.debug("Added embedding for identity %d (%s)", identity_id, self.face_db[identity_id]["name"])

    def match_face(self, embedding: np.ndarray) -> Tuple[str, float]:
        """Match a face embedding against known identities.

        Returns (name, similarity). Below threshold -> (unknown_label, score).
        """
        if embedding is None or len(self.face_db) == 0:
            return self.unknown_label, 0.0

        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embedding_size:
            return self.unknown_label, 0.0

        q_norm = float(np.linalg.norm(query))
        if q_norm > 0:
            query = query / q_norm

        best_name = self.unknown_label
        best_similarity = 0.0

        if self._use_faiss and self._index is not None and self._index.ntotal > 0:
            scores, ids = self._index.search(query, min(1, self._index.ntotal))
            for score, vid in zip(scores[0], ids[0]):
                if vid == -1:
                    continue
                name = self._id_map.get(int(vid), self.unknown_label)
                sim = float(score)
                if sim > best_similarity:
                    best_similarity = sim
                    best_name = name
        else:
            for data in self.face_db.values():
                for emb in data["embeddings"]:
                    sim = float(np.dot(query[0], emb))
                    if sim > best_similarity:
                        best_similarity = sim
                        best_name = data["name"]

        if best_similarity < self.similarity_threshold:
            return self.unknown_label, best_similarity
        return best_name, best_similarity

    def delete_person(self, identity_code: str) -> bool:
        """Delete a person from SQLite and rebuild FAISS index.

        Not model retraining — just removes the vector entry.
        """
        if not self._deactivate_identity_in_sqlite(identity_code):
            logger.warning("Identity not found: %s", identity_code)
            return False

        # Rebuild in-memory db from SQLite
        self._load_from_sqlite()
        self._rebuild_faiss_index()
        logger.info("Deleted identity: %s", identity_code)
        return True

    def list_persons(self) -> List[dict]:
        """Return list of enrolled identities with metadata."""
        results = []
        for ident_id, data in self.face_db.items():
            results.append({
                "id": ident_id,
                "identity_code": data["identity_code"],
                "employee_id": data.get("employee_id", ""),
                "name": data["name"],
                "embedding_count": len(data["embeddings"]),
            })
        return sorted(results, key=lambda x: x["name"])

    def get_database_info(self) -> Dict[str, int | str]:
        total_embeddings = sum(len(d["embeddings"]) for d in self.face_db.values())
        return {
            "total_identities": len(self.face_db),
            "total_embeddings": total_embeddings,
            "db_path": str(self.db_path),
            "face_db_path": str(self.face_db_path),
            "backend": "faiss" if self._use_faiss else "numpy",
        }
