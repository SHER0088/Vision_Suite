"""
pipelines/system_pipeline.py
----------------------------
Pipeline logic for 'frs system' command.
Run full production system pipeline (Object + Face + Recognition + Logging).
Supports both live camera sources and offline image/video files.
"""

from __future__ import annotations

import time
import cv2
import signal
from pathlib import Path
from typing import Dict, Tuple

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_event_logger, get_logger, setup_system_logger
from src.core.tracker import Tracker
from src.core.video_source import VideoSourceManager
from src.detection.face_detector import FaceDetector
from src.detection.object_detector import ObjectDetector
from src.recognition.embedder import FaceEmbedder
from src.recognition.matcher import FaceMatcher
from src.display.display_manager import DisplayManager, class_color
from rich.console import Console

logger = get_logger("system_pipeline")
console = Console()


# ---------------------------------------------------------------------------
# Per-frame processing (shared by camera and file modes)
# ---------------------------------------------------------------------------

def _process_frame(
    *,
    display_img,
    inf_img,
    source_id: str,
    object_detector,
    face_detector,
    embedder,
    matcher,
    object_tracker,
    face_tracker,
    config: AppConfig,
    event_logger,
    current_time: float,
) -> Tuple[any, int]:
    """Process a single frame: detect, track, recognize, draw, log, save crops.

    Returns (annotated_display_img, detection_count).
    """
    if display_img is None or inf_img is None:
        return display_img, 0

    disp_h, disp_w = display_img.shape[:2]
    inf_h, inf_w = inf_img.shape[:2]

    scale_x = disp_w / float(inf_w)
    scale_y = disp_h / float(inf_h)

    raw_detections = []

    # 1. Object Detections
    if object_detector and inf_img is not None:
        obj_dets = object_detector.detect(inf_img)
        for d in obj_dets:
            raw_detections.append((d.bbox, d.confidence, d.class_name, None))

    tracked_objects = object_tracker.update(raw_detections)

    # 2. Face Detections and Recognition
    tracked_faces = []
    if face_detector and inf_img is not None:
        face_dets = face_detector.detect(inf_img)
        raw_faces = [(f.bbox, f.confidence, "face", None) for f in face_dets]
        tracked_faces = face_tracker.update(raw_faces)

        # Process tracked faces for recognition
        for f in tracked_faces:
            fx1, fy1, fx2, fy2 = f.bbox
            dfx1, dfy1 = int(fx1 * scale_x), int(fy1 * scale_y)
            dfx2, dfy2 = int(fx2 * scale_x), int(fy2 * scale_y)

            # Extract face image for embedding
            face_img = display_img[dfy1:dfy2, dfx1:dfx2]
            embedding = None
            if face_img.size > 0:
                embedding = embedder.get_embedding(face_img)

            # Perform recognition if we have an embedding
            person_name = "unknown"
            similarity = 0.0
            if embedding is not None:
                person_name, similarity = matcher.match_face(embedding)

            # Draw bounding box and label
            box_color = (0, 255, 0) if person_name != "unknown" else (0, 0, 255)

            if person_name == "unknown":
                label = f"Unknown #{f.track_id} ({similarity:.2f})"
            else:
                label = f"{person_name} #{f.track_id} ({similarity:.2f})"

            DisplayManager.draw_detection(
                display_img, (dfx1, dfy1, dfx2, dfy2),
                label,
                color=box_color,
                thickness=3,
            )

            # Save recognized face crops with cooldown logic
            if (config.recognition.save_crops and
                (current_time - f.last_cropped_timestamp) >= config.recognition.cooldown_seconds):

                if person_name != matcher.unknown_label or matcher.unknown_label != "unknown":
                    padding = 10
                    crop_x1 = max(0, dfx1 - padding)
                    crop_y1 = max(0, dfy1 - padding)
                    crop_x2 = min(display_img.shape[1], dfx2 + padding)
                    crop_y2 = min(display_img.shape[0], dfy2 + padding)
                    crop_img = display_img[crop_y1:crop_y2, crop_x1:crop_x2]

                    if crop_img.size > 0:
                        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                        crop_filename = f"{person_name}_{timestamp_str}_{source_id}.jpg"
                        crop_dir = Path(config.paths.output_dir) / config.recognition.crop_dir
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = crop_dir / crop_filename
                        cv2.imwrite(str(crop_path), crop_img)

                        # Log recognition event
                        event_logger.log_event(
                            event_type="FACE_RECOGNIZED",
                            camera_id=source_id,
                            class_or_name=person_name,
                            confidence=similarity,
                            bbox=(dfx1, dfy1, dfx2, dfy2),
                            crop_path=str(crop_path),
                            track_id=f.track_id,
                        )

                        # Update cooldown timestamp
                        f.last_cropped_timestamp = current_time

            # Log detection event (regardless of recognition result)
            if (current_time - f.last_logged_timestamp) >= config.events.cooldown_seconds:
                event_logger.log_event(
                    event_type="FACE_DETECTED",
                    camera_id=source_id,
                    class_or_name="face",
                    confidence=f.confidence,
                    bbox=(dfx1, dfy1, dfx2, dfy2),
                    track_id=f.track_id,
                )
                f.last_logged_timestamp = current_time

    # Process tracked objects (for logging/display)
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.bbox
        dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
        dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)

        label = f"{obj.class_name} #{obj.track_id} ({obj.confidence:.2f})"
        DisplayManager.draw_detection(
            display_img, (dx1, dy1, dx2, dy2),
            label,
            color=class_color(obj.class_name),
            thickness=3,
        )

        # Log object detection events with cooldown
        if (current_time - obj.last_logged_timestamp) >= config.events.cooldown_seconds:
            event_logger.log_event(
                event_type="OBJECT_DETECTED",
                camera_id=source_id,
                class_or_name=obj.class_name,
                confidence=obj.confidence,
                bbox=(dx1, dy1, dx2, dy2),
                track_id=obj.track_id,
            )
            obj.last_logged_timestamp = current_time

    detection_count = len(tracked_objects) + len(tracked_faces)
    return display_img, detection_count


# ---------------------------------------------------------------------------
# File-based recognition (image / video)
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _recognize_on_file(
    source: str,
    config: AppConfig,
    event_logger,
    object_detector,
    face_detector,
    embedder,
    matcher,
    object_tracker,
    face_tracker,
) -> None:
    """Dispatch recognition to an image or video file handler."""
    source_path = Path(source)
    if not source_path.exists():
        console.print(f"[bold red]File not found: {source}[/bold red]")
        logger.error("File not found: %s", source)
        return

    is_image = source_path.suffix.lower() in _IMAGE_EXTENSIONS

    if is_image:
        _recognize_on_image(
            image_path=source_path,
            config=config,
            event_logger=event_logger,
            object_detector=object_detector,
            face_detector=face_detector,
            embedder=embedder,
            matcher=matcher,
            object_tracker=object_tracker,
            face_tracker=face_tracker,
        )
    else:
        _recognize_on_video(
            video_path=source_path,
            config=config,
            event_logger=event_logger,
            object_detector=object_detector,
            face_detector=face_detector,
            embedder=embedder,
            matcher=matcher,
            object_tracker=object_tracker,
            face_tracker=face_tracker,
        )


def _recognize_on_image(
    image_path: Path,
    config: AppConfig,
    event_logger,
    object_detector,
    face_detector,
    embedder,
    matcher,
    object_tracker,
    face_tracker,
) -> None:
    """Run recognition on a single image file and save an annotated copy."""
    console.print(f"[bold cyan]Processing image: {image_path}[/bold cyan]")
    logger.info("Recognizing on image: %s", image_path)

    display_img = cv2.imread(str(image_path))
    if display_img is None:
        console.print(f"[bold red]Failed to load image: {image_path}[/bold red]")
        logger.error("Failed to load image: %s", image_path)
        return

    inf_img = display_img
    source_id = image_path.stem
    current_time = time.time()

    # Single-frame processing: lower min_hits so detections are confirmed immediately
    object_tracker.min_hits = 1
    face_tracker.min_hits = 1

    annotated_img, _ = _process_frame(
        display_img=display_img,
        inf_img=inf_img,
        source_id=source_id,
        object_detector=object_detector,
        face_detector=face_detector,
        embedder=embedder,
        matcher=matcher,
        object_tracker=object_tracker,
        face_tracker=face_tracker,
        config=config,
        event_logger=event_logger,
        current_time=current_time,
    )

    # Save annotated output
    output_dir = Path(config.paths.output_dir) / config.recognition.crop_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image_path.stem}_recognized{image_path.suffix}"
    cv2.imwrite(str(output_path), annotated_img)
    console.print(f"[bold green]Saved annotated image: {output_path}[/bold green]")
    logger.info("Saved annotated image: %s", output_path)


def _recognize_on_video(
    video_path: Path,
    config: AppConfig,
    event_logger,
    object_detector,
    face_detector,
    embedder,
    matcher,
    object_tracker,
    face_tracker,
) -> None:
    """Run recognition on a video file and save an annotated copy."""
    console.print(f"[bold cyan]Processing video: {video_path}[/bold cyan]")
    logger.info("Recognizing on video: %s", video_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        console.print(f"[bold red]Failed to open video: {video_path}[/bold red]")
        logger.error("Failed to open video: %s", video_path)
        return

    # Get video properties for output
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Setup output video writer
    output_dir = Path(config.paths.output_dir) / config.recognition.crop_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_path.stem}_recognized.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    source_id = video_path.stem
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_count += 1
            current_time = time.time()

            display_img = frame
            inf_img = frame

            annotated_img, _ = _process_frame(
                display_img=display_img,
                inf_img=inf_img,
                source_id=source_id,
                object_detector=object_detector,
                face_detector=face_detector,
                embedder=embedder,
                matcher=matcher,
                object_tracker=object_tracker,
                face_tracker=face_tracker,
                config=config,
                event_logger=event_logger,
                current_time=current_time,
            )

            writer.write(annotated_img)

            # Allow early exit with 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key pressed during video processing.")
                break

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    console.print(f"[bold green]Saved annotated video: {output_path}[/bold green]")
    logger.info("Saved annotated video: %s (%d frames processed)", output_path, frame_count)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_system_pipeline(source: str | None = None) -> None:
    """Pipeline logic for 'frs system'.

    Args:
        source: Optional path to an image or video file. When provided,
            recognition runs on that file instead of the live camera.
    """
    config = get_config()
    ensure_runtime_directories(config)
    setup_system_logger(config)
    event_logger = get_event_logger(config)

    console.print("[bold cyan][Pipeline] 'frs system' started.[/bold cyan]")

    # Check if EAR/MAR features are requested (placeholder)
    if config.features.ear_mar:
        logger.warning("EAR/MAR features requested in config, but not implemented yet.")

    # Initialize detection and recognition components
    object_detector = ObjectDetector(config) if config.features.object_detection else None
    face_detector = FaceDetector(config) if config.features.face_detection else None
    embedder = FaceEmbedder(config)
    matcher = FaceMatcher(config)

    # Initialize trackers
    object_tracker = Tracker(config)
    face_tracker = Tracker(config)

    # If a file source is provided, process it directly instead of cameras
    if source is not None:
        _recognize_on_file(
            source=source,
            config=config,
            event_logger=event_logger,
            object_detector=object_detector,
            face_detector=face_detector,
            embedder=embedder,
            matcher=matcher,
            object_tracker=object_tracker,
            face_tracker=face_tracker,
        )
        return

    # --- Live camera pipeline (existing behavior) ---
    if not config.features.face_detection and not config.features.object_detection:
        console.print("[bold red]No detection enabled in config.yaml![/bold red]")
        return

    # Initialize video source manager
    manager = VideoSourceManager(config)
    active_sources = manager.get_active_sources()

    if not active_sources:
        console.print("[bold red]No enabled camera sources found in config.yaml![/bold red]")
        return

    logger.info("Starting system pipeline on cameras: %s", [s.camera_id for s in active_sources])
    console.print(
        "[bold green]System Pipeline running. Press 'q' in any video window to exit.[/bold green]"
    )

    # Initialize display manager
    display_manager = DisplayManager(config)
    camera_ids = [s.camera_id for s in active_sources]
    display_manager.initialize_windows(camera_ids)

    try:
        while True:
            frames_map = manager.read_all()
            if not frames_map:
                time.sleep(0.01)
                continue

            all_ended = True
            display_frames: Dict[str, any] = {}
            detection_counts: Dict[str, int] = {}

            for video_source, frame_packet in frames_map.items():
                if frame_packet.status == "end_of_stream":
                    continue

                all_ended = False
                if frame_packet.status != "ok" or frame_packet.display_frame is None:
                    continue

                display_img = frame_packet.display_frame
                inf_img = frame_packet.inference_frame

                current_time = time.time()
                annotated_img, count = _process_frame(
                    display_img=display_img,
                    inf_img=inf_img,
                    source_id=video_source.camera_id,
                    object_detector=object_detector,
                    face_detector=face_detector,
                    embedder=embedder,
                    matcher=matcher,
                    object_tracker=object_tracker,
                    face_tracker=face_tracker,
                    config=config,
                    event_logger=event_logger,
                    current_time=current_time,
                )

                display_frames[video_source.camera_id] = annotated_img
                detection_counts[video_source.camera_id] = count

            if all_ended:
                logger.info("All video streams ended.")
                break

            if not display_frames:
                time.sleep(0.01)
                continue

            window_status = display_manager.display_frames(display_frames, detection_counts)

            if not any(window_status.values()) or cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key pressed or windows closed. Shutting down.")
                break

    finally:
        manager.stop_all()
        display_manager.cleanup()
        cv2.destroyAllWindows()