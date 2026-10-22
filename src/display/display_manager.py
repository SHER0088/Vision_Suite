"""
display/display_manager.py
--------------------------
Professional multi-camera display output for Vision_Suite.

Features:
- Dark theme with clean typography (not a "school project" look)
- Per-camera FPS counter and status indicator (LIVE / DISCONNECTED / END)
- Aspect-ratio-preserving grid cells (no squashed video)
- Timestamp overlay on every frame
- Multi-window mode with per-window camera titles
- Professional bounding-box & label rendering helpers
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.core.config_loader import AppConfig
from src.core.logger import get_logger

logger = get_logger("display_manager")


# ---------------------------------------------------------------------------
# Color palette (professional, color-blind friendly)
# ---------------------------------------------------------------------------

COLOR_BG = (25, 25, 25)          # dark charcoal background
COLOR_PANEL = (40, 40, 40)       # slightly lighter panel
COLOR_TEXT = (230, 230, 230)     # near-white text
COLOR_MUTED = (150, 150, 150)    # muted gray
COLOR_ACCENT = (0, 180, 255)     # cyan accent
COLOR_OK = (0, 255, 120)         # green for live
COLOR_WARN = (0, 165, 255)       # orange for warning
COLOR_ERR = (0, 0, 255)          # red for error/disconnected

# Class-based bounding box colors (distinct, professional)
CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "person": (0, 255, 0),        # green
    "face": (0, 180, 255),        # cyan
    "car": (0, 255, 255),         # yellow
    "truck": (0, 200, 200),       # teal
    "bus": (0, 160, 255),         # orange-blue
    "motorcycle": (255, 0, 255),  # magenta
    "bicycle": (255, 120, 0),     # azure
    "dog": (120, 255, 0),         # lime
    "cat": (255, 0, 120),         # rose
    "default": (0, 255, 0),       # green fallback
}


def class_color(class_name: str) -> Tuple[int, int, int]:
    """Return a consistent BGR color for a detection class."""
    return CLASS_COLORS.get(class_name.lower(), CLASS_COLORS["default"])


# ---------------------------------------------------------------------------
# FPS tracker
# ---------------------------------------------------------------------------

class FPSCounter:
    """Tracks smoothed FPS over a sliding window of frame timestamps."""

    def __init__(self, window: int = 30) -> None:
        self._stamps: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        """Record a frame and return the current smoothed FPS."""
        now = time.time()
        self._stamps.append(now)
        if len(self._stamps) < 2:
            return 0.0
        span = self._stamps[-1] - self._stamps[0]
        if span <= 0:
            return 0.0
        return (len(self._stamps) - 1) / span


@dataclass
class CameraStatus:
    """Runtime status for a single camera feed."""

    camera_id: str
    fps: FPSCounter = field(default_factory=FPSCounter)
    connected: bool = False
    frame_count: int = 0
    last_frame_time: Optional[float] = None
    error_message: Optional[str] = None

    def update(self, ok: bool, error: Optional[str] = None) -> None:
        self.frame_count += 1
        self.last_frame_time = time.time()
        if ok:
            self.connected = True
            self.fps.tick()
            self.error_message = None
        else:
            self.connected = False
            self.error_message = error

    def status_text(self) -> str:
        if not self.connected:
            if self.error_message:
                return f"DISCONNECTED ({self.error_message})"
            return "DISCONNECTED"
        return "LIVE"

    def status_color(self) -> Tuple[int, int, int]:
        if self.connected:
            return COLOR_OK
        return COLOR_ERR


# ---------------------------------------------------------------------------
# Display config
# ---------------------------------------------------------------------------

@dataclass
class DisplayConfig:
    mode: str  # "grid" or "multi_window"
    grid_columns: int
    window_name_prefix: str


# ---------------------------------------------------------------------------
# DisplayManager
# ---------------------------------------------------------------------------

class DisplayManager:
    """Professional multi-camera display manager."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.display_cfg = config.display
        self.window_names: Dict[str, str] = {}
        self.statuses: Dict[str, CameraStatus] = {}
        self._initialized = False
        self._start_time = time.time()

        logger.info(
            "DisplayManager initialized | mode=%s | grid_columns=%d",
            self.display_cfg.mode,
            self.display_cfg.grid_columns,
        )

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def initialize_windows(self, camera_ids: List[str]) -> None:
        """Create display windows (grid or per-camera) for the given IDs."""
        if self._initialized:
            return

        self._camera_ids = list(camera_ids)
        for cam_id in camera_ids:
            self.statuses[cam_id] = CameraStatus(camera_id=cam_id)

        if self.display_cfg.mode == "multi_window":
            for cam_id in camera_ids:
                window_name = f"{self.display_cfg.window_name_prefix} — {cam_id}"
                self.window_names[cam_id] = window_name
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                logger.debug("Created window: %s", window_name)

        elif self.display_cfg.mode == "grid":
            window_name = f"{self.display_cfg.window_name_prefix} — Grid View"
            self.window_names["grid"] = window_name
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            self._resize_grid_window(len(camera_ids))
            logger.debug("Created grid window: %s", window_name)

        self._initialized = True

    def _resize_grid_window(self, num_cameras: int) -> None:
        """Size the grid window based on camera count."""
        if num_cameras == 0:
            return
        cols = min(self.display_cfg.grid_columns, num_cameras)
        rows = (num_cameras + cols - 1) // cols
        cell_w, cell_h = 480, 360  # 4:3 professional cell size
        win_w = cols * cell_w + 20
        win_h = rows * cell_h + 110  # extra for title bar
        window_name = self.window_names.get("grid")
        if window_name:
            cv2.resizeWindow(window_name, win_w, win_h)

    # ------------------------------------------------------------------ #
    # Status helpers
    # ------------------------------------------------------------------ #

    def mark_frame(self, camera_id: str, ok: bool, error: Optional[str] = None) -> None:
        """Record that a frame (or lack thereof) was received for a camera."""
        if camera_id in self.statuses:
            self.statuses[camera_id].update(ok, error)

    # ------------------------------------------------------------------ #
    # Drawing helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _put_text_with_bg(
        img: np.ndarray,
        text: str,
        org: Tuple[int, int],
        font_scale: float = 0.6,
        color: Tuple[int, int, int] = COLOR_TEXT,
        bg_color: Tuple[int, int, int] = COLOR_BG,
        thickness: int = 2,
        pad: int = 5,
    ) -> None:
        """Draw text with a solid background rectangle for readability."""
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        x, y = org
        x = max(0, min(x, img.shape[1] - w - pad * 2))
        y = max(h + pad, min(y, img.shape[0] - pad))
        cv2.rectangle(
            img, (x - pad, y - h - pad), (x + w + pad, y + pad), bg_color, -1
        )
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    @staticmethod
    def draw_detection(
        img: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        color: Optional[Tuple[int, int, int]] = None,
        thickness: int = 3,
    ) -> None:
        """Draw a professional bounding box with label background."""
        x1, y1, x2, y2 = bbox
        color = color or CLASS_COLORS["default"]

        # Bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label background
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y = y1 - 10 if y1 - 10 > th + 10 else y1 + th + 10
        cv2.rectangle(img, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 6), color, -1)
        cv2.putText(
            img, label, (x1 + 5, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_BG, 2,
        )

    # ------------------------------------------------------------------ #
    # Per-frame overlay
    # ------------------------------------------------------------------ #

    def _overlay_camera_info(
        self,
        frame: np.ndarray,
        camera_id: str,
        detection_count: int = 0,
    ) -> None:
        """Burn FPS, status, timestamp, and camera ID onto a single frame."""
        status = self.statuses.get(camera_id)
        if status is None:
            return

        h, w = frame.shape[:2]

        # Top-left: camera ID + status
        status_text = f"{camera_id}  ●  {status.status_text()}"
        self._put_text_with_bg(
            frame, status_text, (15, 30),
            font_scale=0.7,
            color=COLOR_TEXT,
            bg_color=COLOR_PANEL,
            thickness=2,
        )
        # Status dot
        dot_color = status.status_color()
        cv2.circle(frame, (12, 24), 6, dot_color, -1)

        # Top-right: FPS + detection count
        fps_text = f"FPS: {status.fps.tick():.1f}" if status.connected else "FPS: --"
        det_text = f"Dets: {detection_count}"
        info_text = f"{fps_text}   {det_text}"
        (tw, _), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        self._put_text_with_bg(
            frame, info_text, (w - tw - 20, 30),
            font_scale=0.6,
            color=COLOR_TEXT,
            bg_color=COLOR_PANEL,
            thickness=2,
        )

        # Bottom-right: timestamp
        ts = time.strftime("%H:%M:%S")
        date_str = time.strftime("%Y-%m-%d")
        ts_text = f"{date_str}  {ts}"
        (tsw, _), _ = cv2.getTextSize(ts_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        self._put_text_with_bg(
            frame, ts_text, (w - tsw - 15, h - 15),
            font_scale=0.6,
            color=COLOR_MUTED,
            bg_color=COLOR_BG,
            thickness=1,
        )

        # Bottom-left: resolution
        res_text = f"{w}×{h}"
        (_, _), _ = cv2.getTextSize(res_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        self._put_text_with_bg(
            frame, res_text, (15, h - 15),
            font_scale=0.6,
            color=COLOR_MUTED,
            bg_color=COLOR_BG,
            thickness=1,
        )

    # ------------------------------------------------------------------ #
    # Grid composition
    # ------------------------------------------------------------------ #

    def _create_grid_image(
        self,
        frames: Dict[str, np.ndarray],
        _detection_counts: Dict[str, int],
    ) -> Optional[np.ndarray]:
        """Build a professional tiled grid from per-camera frames."""
        if not frames:
            return None

        cam_ids = list(frames.keys())
        num_cams = len(cam_ids)
        cols = min(self.display_cfg.grid_columns, num_cams)
        rows = (num_cams + cols - 1) // cols

        # Determine cell size from the first valid frame
        cell_w = cell_h = 0
        for fid in cam_ids:
            f = frames.get(fid)
            if f is not None and f.size > 0:
                cell_h, cell_w = f.shape[:2]
                break
        if cell_w == 0 or cell_h == 0:
            cell_w, cell_h = 640, 480

        # Cap cell size so the grid isn't enormous
        max_cell_w, max_cell_h = 640, 480
        scale = min(max_cell_w / cell_w, max_cell_h / cell_h, 1.0)
        cell_w = int(cell_w * scale)
        cell_h = int(cell_h * scale)

        gap = 8
        pad_top = 90  # room for title bar
        pad_side = 15

        grid_w = cols * cell_w + (cols + 1) * gap + 2 * pad_side
        grid_h = pad_top + rows * cell_h + (rows + 1) * gap + 2 * gap

        grid = np.full((grid_h, grid_w, 3), COLOR_BG, dtype=np.uint8)

        # Title bar
        title = "VISION_SUITE  |  Multi-Camera Monitor"
        cv2.putText(
            grid, title, (pad_side, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ACCENT, 2,
        )
        subtitle = f"{num_cams} camera(s)  •  {time.strftime('%Y-%m-%d %H:%M:%S')}"
        cv2.putText(
            grid, subtitle, (pad_side, 65),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_MUTED, 1,
        )

        for idx, cam_id in enumerate(cam_ids):
            row = idx // cols
            col = idx % cols

            x0 = pad_side + col * (cell_w + gap) + gap
            y0 = pad_top + row * (cell_h + gap) + gap
            x1 = x0 + cell_w
            y1 = y0 + cell_h

            frame = frames.get(cam_id)
            if frame is None or frame.size == 0:
                # Empty cell with status
                cv2.rectangle(grid, (x0, y0), (x1, y1), COLOR_PANEL, -1)
                msg = self.statuses.get(cam_id, CameraStatus(cam_id)).status_text()
                (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                tx = x0 + (cell_w - tw) // 2
                ty = y0 + (cell_h + th) // 2
                self._put_text_with_bg(
                    grid, msg, (tx, ty),
                    font_scale=0.7,
                    color=COLOR_WARN if "DISCONNECTED" in msg else COLOR_TEXT,
                    bg_color=COLOR_PANEL,
                    thickness=2,
                )
                # Camera ID label
                cv2.rectangle(grid, (x0, y0), (x0 + 120, y0 + 30), COLOR_PANEL, -1)
                cv2.putText(
                    grid, cam_id, (x0 + 8, y0 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2,
                )
                continue

            # Resize frame to cell preserving aspect ratio (letterbox)
            fh, fw = frame.shape[:2]
            scale_fit = min(cell_w / fw, cell_h / fh)
            nw, nh = int(fw * scale_fit), int(fh * scale_fit)
            resized = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)

            # Letterbox into cell
            cell = np.full((cell_h, cell_w, 3), COLOR_BG, dtype=np.uint8)
            ox = (cell_w - nw) // 2
            oy = (cell_h - nh) // 2
            cell[oy:oy + nh, ox:ox + nw] = resized

            # Border color based on status
            border_color = COLOR_OK if self.statuses.get(cam_id, CameraStatus(cam_id)).connected else COLOR_ERR
            cv2.rectangle(grid, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), border_color, 3)

            # Camera ID chip
            chip_w = min(140, cell_w - 10)
            cv2.rectangle(grid, (x0 + 8, y0 + 8), (x0 + 8 + chip_w, y0 + 34), COLOR_PANEL, -1)
            cv2.putText(
                grid, cam_id, (x0 + 16, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2,
            )

            # Overlay FPS/status on the cell
            status = self.statuses.get(cam_id)
            if status:
                fps_txt = f"{status.fps.tick():.1f} FPS" if status.connected else "-- FPS"
                (ftw, _), _ = cv2.getTextSize(fps_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                self._put_text_with_bg(
                    grid, fps_txt, (x1 - ftw - 12, y0 + 24),
                    font_scale=0.5,
                    color=COLOR_TEXT,
                    bg_color=COLOR_PANEL,
                    thickness=1,
                )

            # Place cell
            grid[y0:y1, x0:x1] = cell

        return grid

    # ------------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------------ #

    def display_frames(
        self,
        frames: Dict[str, np.ndarray],
        detection_counts: Optional[Dict[str, int]] = None,
    ) -> Dict[str, bool]:
        """Render frames in the configured layout.

        Args:
            frames: camera_id -> BGR frame
            detection_counts: camera_id -> number of detections (for overlay)

        Returns:
            camera_id -> whether the window is still open
        """
        detection_counts = detection_counts or {}

        if not self._initialized:
            self.initialize_windows(list(frames.keys()))

        # Mark frame status per camera
        for cam_id, frame in frames.items():
            self.mark_frame(cam_id, frame is not None and frame.size > 0)

        if self.display_cfg.mode == "multi_window":
            return self._display_multi_window(frames, detection_counts)
        if self.display_cfg.mode == "grid":
            return self._display_grid(frames, detection_counts)
        logger.error("Unknown display mode: %s", self.display_cfg.mode)
        return {cam_id: True for cam_id in frames.keys()}

    def _display_multi_window(
        self,
        frames: Dict[str, np.ndarray],
        detection_counts: Dict[str, int],
    ) -> Dict[str, bool]:
        results: Dict[str, bool] = {}
        for cam_id, frame in frames.items():
            if cam_id not in self.window_names or frame is None:
                results[cam_id] = True
                continue
            display = frame.copy()
            self._overlay_camera_info(display, cam_id, detection_counts.get(cam_id, 0))
            window_name = self.window_names[cam_id]
            cv2.imshow(window_name, display)
            results[cam_id] = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
        return results

    def _display_grid(
        self,
        frames: Dict[str, np.ndarray],
        detection_counts: Dict[str, int],
    ) -> Dict[str, bool]:
        if not frames:
            return {}

        cam_ids = list(frames.keys())
        grid_image = self._create_grid_image(frames, detection_counts)
        window_name = self.window_names.get("grid")
        if window_name and grid_image is not None:
            cv2.imshow(window_name, grid_image)
            visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 1
            return {cam_id: visible for cam_id in cam_ids}
        return {cam_id: True for cam_id in cam_ids}

    def cleanup(self) -> None:
        """Destroy all display windows."""
        logger.info("Cleaning up display windows")
        for window_name in self.window_names.values():
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow(window_name)
        self.window_names.clear()
        self._initialized = False
