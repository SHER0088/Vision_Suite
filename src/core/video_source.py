"""Unified Multi-Input Video Source Abstraction (Live, RTSP, Offline).

Supports the new top-level `input` section of config.yaml:

    input:
      type: "web_cam"          # video | web_cam | live
      camera_indices: [0, 1]   # device indices (web_cam)
      video_paths: [...]       # file paths (video)
      rtsp_urls: [...]         # RTSP URLs (live)
      ...

Each source gets its own capture thread, single-slot frame buffer, and
preview window. The number of previews equals the number of sources listed
for the selected input type.
"""

from __future__ import annotations

import threading
import time
import platform
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("video_source")


def _resolve_res_tuple(res_obj: Any, default: Tuple[int, int]) -> Tuple[int, int]:
    """Helper to convert ResolutionItem, tuple, list, or dict into (width, height)."""
    if res_obj is None:
        return default
    if hasattr(res_obj, "as_tuple") and callable(res_obj.as_tuple):
        return res_obj.as_tuple()
    if hasattr(res_obj, "width") and hasattr(res_obj, "height"):
        return (int(res_obj.width), int(res_obj.height))
    if isinstance(res_obj, dict):
        return (int(res_obj.get("width", default[0])), int(res_obj.get("height", default[1])))
    if isinstance(res_obj, (list, tuple)) and len(res_obj) >= 2:
        return (int(res_obj[0]), int(res_obj[1]))
    return default


@dataclass
class FramePacket:
    status: str  # "ok", "no_new_frame", "end_of_stream", "disconnected"
    display_frame: Optional[np.ndarray] = None
    inference_frame: Optional[np.ndarray] = None
    timestamp: float = 0.0


class FrameBuffer:
    """Thread-safe single-slot frame buffer (latest frame only)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._packet: Optional[FramePacket] = None

    def put(self, packet: FramePacket) -> None:
        with self._lock:
            self._packet = packet

    def get(self) -> Optional[FramePacket]:
        with self._lock:
            return self._packet


class _CaptureThread(threading.Thread):
    """Background capture thread for a single video source."""

    def __init__(
        self,
        source_cfg: Dict[str, Any],
        display_res: Tuple[int, int],
        inference_res: Tuple[int, int],
        buffer: FrameBuffer,
    ) -> None:
        super().__init__(daemon=True, name=f"CaptureThread-{source_cfg['id']}")
        self.cfg = source_cfg
        self.camera_id = source_cfg["id"]
        self.source_val = source_cfg["source"]
        self.type = str(source_cfg["type"]).lower()
        self.display_res = display_res
        self.inference_res = inference_res
        self.buffer = buffer

        self.reconnect_delay = int(source_cfg.get("reconnect_delay_seconds", 5) or 5)
        self.frame_skip = int(source_cfg.get("frame_skip", 0) or 0)
        self.transport = source_cfg.get("transport", "tcp")

        self._stop_event = threading.Event()
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        frame_count = 0

        while not self._stop_event.is_set():
            if not self.is_connected or self.cap is None or not self.cap.isOpened():
                if not self._connect():
                    time.sleep(self.reconnect_delay)
                    continue

            ret, frame = self.cap.read()

            if not ret or frame is None or frame.size == 0:
                if self.type == "offline":
                    logger.info("[%s] End of offline video stream reached.", self.camera_id)
                    self.buffer.put(FramePacket(status="end_of_stream"))
                    break
                else:
                    logger.warning("[%s] Frame read failed. Reconnecting...", self.camera_id)
                    self.is_connected = False
                    self._release_cap()
                    self.buffer.put(FramePacket(status="disconnected"))
                    time.sleep(self.reconnect_delay)
                    continue

            frame_count += 1
            if self.frame_skip > 0 and (frame_count % (self.frame_skip + 1)) != 0:
                continue

            # Resize display & inference frames independently
            disp_frame = cv2.resize(
                frame, self.display_res, interpolation=cv2.INTER_LINEAR
            )
            inf_frame = cv2.resize(
                frame, self.inference_res, interpolation=cv2.INTER_LINEAR
            )

            packet = FramePacket(
                status="ok",
                display_frame=disp_frame,
                inference_frame=inf_frame,
                timestamp=time.time(),
            )
            self.buffer.put(packet)
            time.sleep(0.005)  # Yield CPU brief moment

        self._release_cap()

    def _connect(self) -> bool:
        if self.cap is not None:
            self._release_cap()

        logger.info("[%s] Connecting to source: %s...", self.camera_id, self.source_val)

        try:
            if self.type == "live":
                cam_idx = int(self.source_val)
                system = platform.system()
                if system == "Windows":
                    backend = cv2.CAP_DSHOW
                elif system == "Linux":
                    backend = cv2.CAP_V4L2
                elif system == "Darwin":  # macOS
                    backend = cv2.CAP_AVFOUNDATION
                else:
                    backend = cv2.CAP_ANY
                self.cap = cv2.VideoCapture(cam_idx, backend)
            elif self.type == "rtsp":
                source_str = str(self.source_val)
                old_env = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
                try:
                    if self.transport == "tcp":
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                    elif self.transport == "udp":
                        os.environ["OPENCF_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
                    else:
                        if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                            del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
                    self.cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
                finally:
                    if old_env is None:
                        if "OPENCV_FFMPEG_CAPTURE_OPTIONS" in os.environ:
                            del os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]
                    else:
                        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = old_env
            elif self.type == "offline":
                filepath = Path(str(self.source_val))
                if not filepath.exists():
                    logger.error("[%s] Offline video file missing: %s", self.camera_id, filepath)
                    return False
                self.cap = cv2.VideoCapture(str(filepath))
            else:
                logger.error("[%s] Unknown camera type: %s", self.camera_id, self.type)
                return False

            if self.cap is not None and self.cap.isOpened():
                # --- Apply camera properties for best quality & low latency ---
                self._apply_camera_properties()
                self.is_connected = True
                logger.info("[%s] Connected successfully.", self.camera_id)
                return True

            logger.error("[%s] Failed to open video capture source.", self.camera_id)
            return False

        except Exception as exc:
            logger.error("[%s] Connection exception: %s", self.camera_id, exc)
            return False

    def _apply_camera_properties(self) -> None:
        """Apply resolution, FPS, and buffer-size hints to the capture device.

        Best-effort: many USB/webcam drivers ignore some of these, but setting
        them improves quality and reduces latency on the ones that honor them.
        """
        if self.cap is None:
            return

        try:
            w, h = self.display_res
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(w))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(h))
            self.cap.set(cv2.CAP_PROP_FPS, 30.0)
            # Minimize internal buffering so we always get the freshest frame
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
            # Try autofocus (ignored if unsupported)
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1.0)
            except Exception:
                pass
        except Exception as exc:
            logger.warning(
                "[%s] Could not apply camera properties: %s", self.camera_id, exc
            )

    def _release_cap(self) -> None:
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self.is_connected = False


class VideoSource:
    """Wrapper managing the background capture thread for a single input source."""

    def __init__(
        self,
        source_cfg: Dict[str, Any],
        display_res: Tuple[int, int],
        inference_res: Tuple[int, int],
    ) -> None:
        self.camera_id = source_cfg["id"]
        self.cfg = source_cfg
        self.buffer = FrameBuffer()
        self.thread = _CaptureThread(source_cfg, display_res, inference_res, self.buffer)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.thread.stop()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def read(self) -> FramePacket:
        packet = self.buffer.get()
        if packet is None:
            return FramePacket(status="no_new_frame")
        return packet


class VideoSourceManager:
    """Manages all active video sources from config.input.

    Reads the `input` section of config.yaml, determines the input type,
    creates one VideoSource per source of that type, and provides a unified
    read_all() interface.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.sources: Dict[str, VideoSource] = {}
        self._build_sources()

    def _build_sources(self) -> None:
        default_display = _resolve_res_tuple(
            getattr(self.config.resolution, "display", None), (1280, 720)
        )
        default_inference = _resolve_res_tuple(
            getattr(self.config.resolution, "inference", None), (640, 480)
        )

        # Build source configs from the new input section
        source_configs = self.config.input.active_sources()

        for src_cfg in source_configs:
            # Apply per-source resolution overrides or fall back to defaults
            src_display_res = _resolve_res_tuple(
                src_cfg.get("display_resolution"), default_display
            )
            src_inference_res = _resolve_res_tuple(
                src_cfg.get("inference_resolution"), default_inference
            )
            src_cfg["display_resolution"] = src_display_res
            src_cfg["inference_resolution"] = src_inference_res

            source = VideoSource(src_cfg, src_display_res, src_inference_res)
            source.start()
            self.sources[source.camera_id] = source

        logger.info(
            "VideoSourceManager: %d source(s) active (type=%s)",
            len(self.sources),
            self.config.input.type,
        )

    def get_active_sources(self) -> List[VideoSource]:
        return list(self.sources.values())

    def get_source_count(self) -> int:
        """Return the number of active input sources (= preview count)."""
        return len(self.sources)

    def read_all(self) -> Dict[VideoSource, FramePacket]:
        results: Dict[VideoSource, FramePacket] = {}
        for source in self.sources.values():
            results[source] = source.read()
        return results

    def stop_all(self) -> None:
        for source in self.sources.values():
            source.stop()
        logger.info("All video capture sources stopped.")
