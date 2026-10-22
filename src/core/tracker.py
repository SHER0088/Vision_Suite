"""Lightweight IOU-based persistent Object & Face Tracker for CPU pipelines.

Assigns stable `track_id` values across frames, enabling:
1. Frame-to-frame bounding box continuity.
2. Per-object event logging cooldowns (no duplicate log rows every 33ms).
3. Per-object crop extraction cooldowns.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("tracker")


def compute_iou(boxA: Tuple[float, float, float, float], boxB: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IOU) between two bboxes (x1, y1, x2, y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_width = max(0.0, xB - xA)
    inter_height = max(0.0, yB - yA)
    inter_area = inter_width * inter_height

    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    union_area = boxA_area + boxB_area - inter_area
    if union_area <= 0:
        return 0.0

    return float(inter_area / union_area)


class TrackedItem:
    """Represents an actively tracked object or face across multiple frames."""

    def __init__(
        self,
        track_id: int,
        bbox: Tuple[float, float, float, float],
        class_name: str,
        confidence: float,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        self.track_id: int = track_id
        self.bbox: Tuple[float, float, float, float] = bbox
        self.class_name: str = class_name
        self.confidence: float = confidence
        self.embedding: Optional[np.ndarray] = embedding

        self.age: int = 1  # Total frames track has existed
        self.time_since_update: int = 0  # Consecutive frames without match
        self.hits: int = 1  # Number of matched frames

        # Timestamps for cooldown-based tasks (logs / crop saving)
        self.last_logged_timestamp: float = 0.0
        self.last_cropped_timestamp: float = 0.0
        self.assigned_person_name: str = "unknown"

    def update(
        self,
        bbox: Tuple[float, float, float, float],
        confidence: float,
        class_name: str,
        embedding: Optional[np.ndarray] = None,
    ) -> None:
        """Update track with fresh detection match."""
        self.bbox = bbox
        self.confidence = confidence
        self.class_name = class_name
        if embedding is not None:
            self.embedding = embedding
        self.hits += 1
        self.time_since_update = 0

    def mark_missed(self) -> None:
        """Increment counters when track is missed in current frame."""
        self.age += 1
        self.time_since_update += 1


class Tracker:
    """Multi-object / multi-face tracker relying on spatial overlap and class affinity."""

    def __init__(self, config: AppConfig) -> None:
        self.iou_threshold: float = float(config.tracking.iou_threshold)
        self.max_age: int = int(config.tracking.max_age)
        self.min_hits: int = int(config.tracking.min_hits)

        self.next_id: int = 1
        self.tracks: Dict[int, TrackedItem] = {}

    def update(
        self,
        detections: List[Tuple[Tuple[float, float, float, float], float, str, Optional[np.ndarray]]],
    ) -> List[TrackedItem]:
        """Update tracker state with current frame detections.

        Args:
            detections: List of tuples -> (bbox_xyxy, confidence, class_name, optional_embedding)

        Returns:
            List of confirmed TrackedItem objects visible in current frame.
        """
        # Input validation
        if not isinstance(detections, list):
            raise ValueError("detections must be a list")
        for i, det in enumerate(detections):
            if not isinstance(det, tuple) or len(det) != 4:
                raise ValueError(f"detection #{i} must be a tuple of length 4, got: {det}")
            bbox, confidence, class_name, embedding = det
            if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
                raise ValueError(f"detection #{i} bbox must be a tuple of length 4, got: {bbox}")
            if not all(isinstance(coord, (int, float)) for coord in bbox):
                raise ValueError(f"detection #{i} bbox coordinates must be numeric, got: {bbox}")
            if not isinstance(confidence, (int, float)):
                raise ValueError(f"detection #{i} confidence must be numeric, got: {confidence}")
            if not isinstance(class_name, str):
                raise ValueError(f"detection #{i} class_name must be a string, got: {class_name}")
            if embedding is not None and not isinstance(embedding, np.ndarray):
                raise ValueError(f"detection #{i} embedding must be a numpy array or None, got: {embedding}")

        # Step 1: Mark all existing tracks as missed initially
        for track in self.tracks.values():
            track.mark_missed()

        active_track_ids = list(self.tracks.keys())

        if len(active_track_ids) > 0 and len(detections) > 0:
            # Step 2: Build IOU Cost Matrix (Tracks x Detections)
            iou_matrix = np.zeros((len(active_track_ids), len(detections)), dtype=np.float32)

            for i, tid in enumerate(active_track_ids):
                track_box = self.tracks[tid].bbox
                track_cls = self.tracks[tid].class_name

                for j, (det_box, _, det_cls, _) in enumerate(detections):
                    # Only match if same category (e.g. face to face, person to person).
                    # Do NOT mix face and non-face categories.
                    if track_cls == det_cls:
                        iou_matrix[i, j] = compute_iou(track_box, det_box)
                    else:
                        iou_matrix[i, j] = 0.0

            # Greedy Hungarian-style matching
            matched_track_indices = set()
            matched_det_indices = set()

            # Iterate by highest IOU scores
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break

                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                tid = active_track_ids[i]

                # Assign match
                det_box, det_conf, det_cls, det_emb = detections[j]
                self.tracks[tid].update(det_box, det_conf, det_cls, det_emb)

                matched_track_indices.add(i)
                matched_det_indices.add(j)

                # Zero out row and column
                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

            # Create new tracks for unmatched detections
            for j, (det_box, det_conf, det_cls, det_emb) in enumerate(detections):
                if j not in matched_det_indices:
                    new_track = TrackedItem(
                        track_id=self.next_id,
                        bbox=det_box,
                        class_name=det_cls,
                        confidence=det_conf,
                        embedding=det_emb,
                    )
                    self.tracks[self.next_id] = new_track
                    self.next_id += 1

        elif len(detections) > 0:
            # No existing tracks: instantiate all detections as new tracks
            for det_box, det_conf, det_cls, det_emb in detections:
                new_track = TrackedItem(
                    track_id=self.next_id,
                    bbox=det_box,
                    class_name=det_cls,
                    confidence=det_conf,
                    embedding=det_emb,
                )
                self.tracks[self.next_id] = new_track
                self.next_id += 1

        # Purge stale tracks that haven't been seen for > max_age frames
        dead_ids = [
            tid for tid, track in self.tracks.items() if track.time_since_update > self.max_age
        ]
        for tid in dead_ids:
            del self.tracks[tid]

        # Return active tracks seen in current frame with sufficient hit count
        confirmed_active = [
            track
            for track in self.tracks.values()
            if track.time_since_update == 0 and track.hits >= self.min_hits
        ]

        return confirmed_active