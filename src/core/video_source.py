"""
video_source.py
-----------------
Unified video input abstraction for offline files, live cameras, and RTSP
streams. This is THE critical file for solving the RTSP lag problem.

Design principles:
- Offline mode: sequential blocking read (every frame matters, no drops)
- Live/RTSP mode: dedicated background thread + single-slot "latest frame
  only" buffer. The inference loop never waits on a slow stream and never
  processes stale backlog frames - it always gets the newest frame available,
  or is told nothing new has arrived yet.
"""

from __future__ import annotations

import os
import time
import logging
import threading
from pathlib import Path
from typing import Optional, Tuple, Callable, List

import cv2
import numpy as np

from src.core.config_loader import CameraConfig

logger = logging.getLogger("vision_suite.video_source")


# =============================================================================
# FRAME BUFFER
# =============================================================================

class FrameBuffer:
    """Thread-safe, single-slot buffer. Always holds only the most recent frame."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = -1

    def update(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._frame_id += 1

    def get_latest(self) -> Tuple[Optional[np.ndarray], int]:
        with self._lock:
            if self._frame is None:
                return None, -1
            return self._frame.copy(), self._frame_id


# =============================================================================
# BACKGROUND CAPTURE THREAD (for live + rtsp)
# =============================================================================

class _CaptureThread(threading.Thread):
    """
    Continuously reads frames from a VideoCapture source in the background,
    independent of how fast/slow the inference pipeline consumes them.
    Handles automatic reconnection on stream drop (critical for RTSP).
    """

    def __init__(
        self,
        camera_id: str,
        open_fn: Callable[[], Optional[cv2.VideoCapture]],
        reconnect_delay: float,
        frame_skip: int,
    ):
        super().__init__(daemon=True, name=f"capture-{camera_id}")
        self.camera_id = camera_id
        self._open_fn = open_fn
        self.reconnect_delay = reconnect_delay
        self.frame_skip = max(0, frame_skip)

        self.buffer = FrameBuffer()
        self._stop_event = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self.connected = False

    def run(self) -> None:
        while not self._stop_event.is_set():
            if self._cap is None:
                self._cap = self._try_open()
                if self._cap is None:
                    time.sleep(self.reconnect_delay)
                    continue

            skip_counter = 0
            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    logger.warning(f"[{self.camera_id}] Stream lost. Attempting reconnect...")
                    self.connected = False
                    self._safe_release()
                    break

                self.connected = True

                if self.frame_skip > 0:
                    skip_counter += 1
                    if skip_counter <= self.frame_skip:
                        continue
                    skip_counter = 0

                self.buffer.update(frame)

            if not self._stop_event.is_set():
                time.sleep(self.reconnect_delay)

    def _try_open(self) -> Optional[cv2.VideoCapture]:
        try:
            cap = self._open_fn()
            if cap is None or not cap.isOpened():
                logger.error(f"[{self.camera_id}] Failed to open video source.")
                return None
            logger.info(f"[{self.camera_id}] Connected successfully.")
            return cap
        except Exception as e:
            logger.error(f"[{self.camera_id}] Exception while opening source: {e}")
            return None

    def _safe_release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def read_latest(self) -> Tuple[Optional[np.ndarray], int]:
        return self.buffer.get_latest()

    def stop(self) -> None:
        self._stop_event.set()
        self._safe_release()


# =============================================================================
# UNIFIED VIDEO SOURCE
# =============================================================================

class VideoSource:
    """
    Single interface for offline / live / rtsp input. Downstream pipelines
    call .read() and don't need to know or care which mode is active.
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        default_display_res: Tuple[int, int],
        default_inference_res: Tuple[int, int],
    ):
        self.id = camera_config.id
        self.type = camera_config.type
        self.source = camera_config.source
        self.transport = camera_config.transport
        self.frame_skip = camera_config.frame_skip
        self.reconnect_delay = camera_config.reconnect_delay_seconds

        self.display_res = tuple(camera_config.display_resolution) if camera_config.display_resolution else default_display_res
        self.inference_res = tuple(camera_config.inference_resolution) if camera_config.inference_resolution else default_inference_res

        self._offline_cap: Optional[cv2.VideoCapture] = None
        self._capture_thread: Optional[_CaptureThread] = None
        self._last_frame_id: int = -1

        if self.type == "offline":
            self._init_offline()
        else:
            self._init_streaming()

    # --------------------------- Open helpers -------------------------------

    def _open_offline(self) -> cv2.VideoCapture:
        path = Path(str(self.source))
        if not path.exists():
            raise FileNotFoundError(f"[{self.id}] Offline video file not found: {path}")
        return cv2.VideoCapture(str(path))

    def _open_live(self) -> cv2.VideoCapture:
        # CAP_DSHOW is the Windows-recommended backend for reliable camera access
        cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_res[1])
        return cap

    def _open_rtsp(self) -> cv2.VideoCapture:
        if self.transport:
            # Must be set before VideoCapture is constructed
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{self.transport}"
        cap = cv2.VideoCapture(str(self.source), cv2.CAP_FFMPEG)
        return cap

    # --------------------------- Init paths ----------------------------------

    def _init_offline(self) -> None:
        self._offline_cap = self._open_offline()
        logger.info(f"[{self.id}] Offline source opened: {self.source}")

    def _init_streaming(self) -> None:
        open_fn = self._open_live if self.type == "live" else self._open_rtsp
        self._capture_thread = _CaptureThread(
            camera_id=self.id,
            open_fn=open_fn,
            reconnect_delay=self.reconnect_delay,
            frame_skip=self.frame_skip,
        )
        self._capture_thread.start()

    # --------------------------- Public API -----------------------------------

    def read(self) -> Tuple[str, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Returns (status, display_frame, inference_frame)

        status values:
          "ok"            - new frame available, both frames returned
          "no_new_frame"  - stream alive but no new frame since last read (live/rtsp only)
          "end_of_stream" - offline file finished
          "disconnected"  - live/rtsp source currently unreachable
        """
        if self.type == "offline":
            ok, frame = self._offline_cap.read()
            if not ok:
                return "end_of_stream", None, None
            display_frame = cv2.resize(frame, self.display_res)
            inference_frame = cv2.resize(frame, self.inference_res)
            return "ok", display_frame, inference_frame

        frame, frame_id = self._capture_thread.read_latest()

        if frame is None:
            status = "disconnected" if not self._capture_thread.connected else "no_new_frame"
            return status, None, None

        if frame_id == self._last_frame_id:
            return "no_new_frame", None, None

        self._last_frame_id = frame_id
        display_frame = cv2.resize(frame, self.display_res)
        inference_frame = cv2.resize(frame, self.inference_res)
        return "ok", display_frame, inference_frame

    def is_offline(self) -> bool:
        return self.type == "offline"

    def release(self) -> None:
        if self.type == "offline" and self._offline_cap is not None:
            self._offline_cap.release()
        elif self._capture_thread is not None:
            self._capture_thread.stop()
            self._capture_thread.join(timeout=2)
        logger.info(f"[{self.id}] Released.")


# =============================================================================
# MULTI-CAMERA MANAGER (foundation for grid/multi_window display later)
# =============================================================================

class VideoSourceManager:
    """Builds and manages VideoSource instances for every enabled camera."""

    def __init__(self, config):
        self.config = config
        self.sources: dict[str, VideoSource] = {}
        self._build_sources()

    def _build_sources(self) -> None:
        display_res = self.config.resolution.display.as_tuple()
        inference_res = self.config.resolution.inference.as_tuple()

        for cam_cfg in self.config.enabled_cameras():
            try:
                self.sources[cam_cfg.id] = VideoSource(cam_cfg, display_res, inference_res)
            except Exception as e:
                logger.error(f"[{cam_cfg.id}] Failed to initialize: {e}")

    def read_all(self) -> dict:
        """Returns {camera_id: (status, display_frame, inference_frame)} for every active source."""
        results = {}
        for cam_id, source in self.sources.items():
            results[cam_id] = source.read()
        return results

    def release_all(self) -> None:
        for source in self.sources.values():
            source.release()