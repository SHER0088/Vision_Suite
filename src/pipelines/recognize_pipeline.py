"""
Pipeline logic for 'frs recognize' command.
Recognize registered faces from camera or file.

Flow per frame:
  YOLOv8n-Face detection -> FaceNet embedding -> FAISS similarity search
  -> threshold decision -> KNOWN / UNKNOWN
"""

from __future__ import annotations

import cv2
from pathlib import Path
from typing import Optional

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_logger
from src.core.video_source import VideoSourceManager
from src.detection.face_detector import FaceDetector
from src.display.display_manager import DisplayManager
from src.recognition.embedder import FaceEmbedder
from src.recognition.matcher import FaceMatcher
from rich.console import Console

logger = get_logger("recognize_pipeline")
console = Console()

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _process_frame(
    frame,
    face_detector,
    embedder,
    matcher,
    display_manager: DisplayManager,
    source_id: str,
):
    """Detect faces, embed, match, draw. Returns annotated frame."""
    if frame is None:
        return frame

    face_dets = face_detector.detect(frame)
    for fd in face_dets:
        x1, y1, x2, y2 = fd.bbox
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        embedding = embedder.get_embedding(face_img)
        if embedding is None:
            DisplayManager.draw_detection(
                frame, (x1, y1, x2, y2), "No embedding",
                color=(0, 0, 255),
            )
            continue

        name, similarity = matcher.match_face(embedding)
        color = (0, 255, 0) if name != matcher.unknown_label else (0, 0, 255)
        label = f"{name} ({similarity:.2f})"
        DisplayManager.draw_detection(frame, (x1, y1, x2, y2), label, color=color)

    return frame


def _recognize_on_file(
    source_path, config, face_detector, embedder, matcher, display_manager
):
    """Run recognition on an image or video file."""
    source_path = Path(source_path)
    if not source_path.exists():
        console.print(f"[bold red]File not found: {source_path}[/bold red]")
        return

    is_image = source_path.suffix.lower() in _IMAGE_EXTENSIONS

    if is_image:
        img = cv2.imread(str(source_path))
        if img is None:
            console.print(f"[bold red]Failed to load image: {source_path}[/bold red]")
            return
        annotated = _process_frame(
            img, face_detector, embedder, matcher, display_manager, source_path.stem
        )
        output_dir = Path(config.paths.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{source_path.stem}_recognized{source_path.suffix}"
        cv2.imwrite(str(output_path), annotated)
        console.print(f"[bold green]Saved: {output_path}[/bold green]")
    else:
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            console.print(f"[bold red]Failed to open video: {source_path}[/bold red]")
            return
        window_name = f"{config.display.window_name_prefix} - Recognize"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                annotated = _process_frame(
                    frame, face_detector, embedder, matcher, display_manager,
                    source_path.stem,
                )
                cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def run_recognize_pipeline(source: Optional[str] = None) -> None:
    """Run face recognition on camera or an image/video file."""
    config = get_config()
    ensure_runtime_directories(config)

    face_detector = FaceDetector(config)
    embedder = FaceEmbedder(config)
    matcher = FaceMatcher(config)
    display_manager = DisplayManager(config)

    if source is not None:
        _recognize_on_file(
            source, config, face_detector, embedder, matcher, display_manager
        )
        return

    # Live camera using VideoSourceManager (consistent with other pipelines)
    manager = VideoSourceManager(config)
    active_sources = manager.get_active_sources()
    if not active_sources:
        console.print("[bold red]No enabled camera sources found in config.yaml![/bold red]")
        return

    camera_ids = [s.camera_id for s in active_sources]
    display_manager.initialize_windows(camera_ids)

    console.print("[bold green]Recognize running. Press 'q' to exit.[/bold green]")
    try:
        while True:
            frames_map = manager.read_all()
            if not frames_map:
                continue

            display_frames: dict = {}
            for video_source, frame_packet in frames_map.items():
                if frame_packet.status != "ok" or frame_packet.display_frame is None:
                    continue

                annotated = _process_frame(
                    frame_packet.display_frame,
                    face_detector, embedder, matcher, display_manager,
                    video_source.camera_id,
                )
                display_frames[video_source.camera_id] = annotated

            if not display_frames:
                continue

            window_status = display_manager.display_frames(display_frames)
            if not any(window_status.values()) or cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        manager.stop_all()
        display_manager.cleanup()
        cv2.destroyAllWindows()
