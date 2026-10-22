"""
logger.py
-----------
Centralized logging system for Vision_Suite. Provides two independent
logging mechanisms:

1. System Logger  - standard Python logging, rotating file handler,
                     for operational messages (camera connect, errors, etc.)
2. Event Logger    - tabular CSV logger for detection/recognition events,
                     with cooldown-based throttling per track_id so we don't
                     flood the file with a row every single frame.

Both auto-create their target directories and auto-rotate by date.
"""

from __future__ import annotations

import csv
import logging
import logging.handlers
import threading
from datetime import datetime, date
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

# NOTE: AppConfig is imported lazily inside functions that need it at runtime.
# It's only used in type annotations (which are strings due to
# `from __future__ import annotations`), so no top-level import is needed.
# This avoids a circular import with config_loader.py.


# =============================================================================
# SYSTEM LOGGER (plain text, operational events)
# =============================================================================

_system_logger_configured = False
_system_logger_lock = threading.Lock()


def setup_system_logger(config: AppConfig) -> logging.Logger:
    """
    Configures and returns the root 'vision_suite' logger.
    Call this ONCE at application startup (each CLI command's entrypoint).
    Safe to call multiple times - will not duplicate handlers.
    """
    global _system_logger_configured

    logger = logging.getLogger("vision_suite")
    logger.setLevel(getattr(logging, config.logging.system_log_level))

    if _system_logger_configured:
        return logger

    with _system_logger_lock:
        if _system_logger_configured:
            return logger

        try:
            logs_dir = Path(config.paths.logs_dir)
            logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            raise RuntimeError(f"Failed to create logs directory {config.paths.logs_dir}: {e}") from e

        today_str = date.today().isoformat()
        log_file = logs_dir / f"system_{today_str}.log"

        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        try:
            # Rotating file handler - rolls over within the same day if it exceeds max_bytes
            file_handler = logging.handlers.RotatingFileHandler(
                filename=str(log_file),
                maxBytes=config.logging.system_log_max_bytes,
                backupCount=config.logging.system_log_backup_count,
                encoding="utf-8",
            )
            file_handler.setFormatter(formatter)
        except OSError as e:
            raise RuntimeError(f"Failed to create rotating file handler for {log_file}: {e}") from e

        # Also print to console so you see what's happening live during CLI runs
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        try:
            logger.addHandler(file_handler)
            logger.addHandler(console_handler)
        except Exception as e:
            raise RuntimeError(f"Failed to add handlers to logger: {e}") from e

        logger.propagate = False

        _system_logger_configured = True

        logger.info(f"System logger initialized. Writing to: {log_file}")
        return logger


def get_logger(name: str) -> logging.Logger:
    """
    Returns a child logger under the 'vision_suite' namespace.
    Use this in every module instead of logging.getLogger(__name__) directly,
    so everything funnels through the same configured handlers.

    Example: logger = get_logger("video_source")
    """
    return logging.getLogger(f"vision_suite.{name}")


# =============================================================================
# EVENT LOGGER (tabular CSV, cooldown-throttled)
# =============================================================================

class EventLogger:
    """
    Writes detection/recognition events to a daily CSV file.

    Cooldown logic: an event for the same (camera_id, track_id, event_type)
    combination is throttled so we don't flood the file with a row every
    single frame. Uses a bounded cooldown cache to prevent memory leaks.
    """

    def __init__(self, config: AppConfig) -> None:
        self.logger = logging.getLogger("vision_suite.logger")
        self.logs_dir = Path(config.paths.logs_dir)
        try:
            self.logs_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            self.logger.error(f"Failed to create logs directory {config.paths.logs_dir}: {e}")
            raise
        self.cooldown_seconds = float(config.logging.event_log_cooldown_seconds)
        self.columns = config.logging.event_csv_columns or [
            "timestamp", "event_type", "camera_id", "class_or_name",
            "confidence", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "crop_path", "track_id",
        ]
        self._lock = threading.Lock()
        self._last_logged: Dict[Tuple[str, Any, str], float] = {}
        self._current_date: Optional[str] = None
        self._current_path: Optional[Path] = None
        self._max_cooldown_cache = 10000  # prevent unbounded memory growth

    def _ensure_file_ready(self) -> None:
        """Opens/creates today's CSV file, writing header if it's new."""
        today_str = date.today().isoformat()

        if self._current_date == today_str:
            return  # already pointed at today's file

        self._current_date = today_str
        self._current_path = self.logs_dir / f"events_{today_str}.csv"

        file_exists = self._current_path.exists()
        if not file_exists:
            try:
                with open(self._current_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.columns)
            except OSError as e:
                self.logger.error(f"Failed to create event log file {self._current_path}: {e}")
                raise

    def _cooldown_key(self, camera_id: str, track_id: Any, event_type: str) -> Tuple[str, Any, str]:
        # Ensure track_id is hashable; if not, convert to string for use as dict key
        try:
            hash(track_id)
        except TypeError:
            self.logger.warning(
                f"track_id {track_id!r} is not hashable; converting to string for cooldown key"
            )
            track_id_for_key = str(track_id)
        else:
            track_id_for_key = track_id
        return (camera_id, track_id_for_key, event_type)

    def _is_on_cooldown(self, key: Tuple[str, Any, str], now: float) -> bool:
        last_time = self._last_logged.get(key)
        if last_time is None:
            return False
        return (now - last_time) < self.cooldown_seconds

    def _cleanup_old_cooldowns(self, now: float) -> None:
        """Remove cooldown entries older than 2x the cooldown period to bound memory."""
        if len(self._last_logged) <= self._max_cooldown_cache:
            return
        cutoff = now - (self.cooldown_seconds * 2)
        stale_keys = [k for k, t in self._last_logged.items() if t < cutoff]
        for k in stale_keys:
            del self._last_logged[k]

    # ------------------------------------------------------------------ #

    def log_event(
        self,
        event_type: str,
        camera_id: str,
        class_or_name: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        crop_path: str = "",
        track_id: Any = -1,
        force: bool = False,
    ) -> bool:
        """
        Logs one event row if not currently on cooldown for this
        (camera_id, track_id, event_type) combination.

        Returns True if the row was actually written, False if skipped
        due to cooldown.

        Set force=True to bypass cooldown (e.g., for one-off system events
        you still want in the CSV).
        """
        if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
            raise ValueError(f"bbox must be a 4-tuple (x1, y1, x2, y2), got: {bbox}")

        now = datetime.now().timestamp()
        key = self._cooldown_key(camera_id, track_id, event_type)

        with self._lock:
            self._ensure_file_ready()  # handles midnight rollover mid-session
            self._cleanup_old_cooldowns(now)

            if not force and self._is_on_cooldown(key, now):
                return False

            row = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
                "event_type": event_type,
                "camera_id": camera_id,
                "class_or_name": class_or_name,
                "confidence": round(float(confidence), 4),
                "bbox_x1": int(bbox[0]),
                "bbox_y1": int(bbox[1]),
                "bbox_x2": int(bbox[2]),
                "bbox_y2": int(bbox[3]),
                "crop_path": crop_path,
                "track_id": track_id,
            }

            if self._current_path is None:
                raise RuntimeError("Event log path is not initialized")

            try:
                with open(self._current_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=self.columns)
                    writer.writerow(row)
            except OSError as e:
                self.logger.error(f"Failed to write to event log file {self._current_path}: {e}")
                raise

            self._last_logged[key] = now
            return True


# =============================================================================
# SINGLETON ACCESSORS
# =============================================================================

_event_logger_instance: Optional[EventLogger] = None


def get_event_logger(config: AppConfig) -> EventLogger:
    """Returns a cached singleton EventLogger instance."""
    global _event_logger_instance
    if _event_logger_instance is None:
        _event_logger_instance = EventLogger(config)
    return _event_logger_instance