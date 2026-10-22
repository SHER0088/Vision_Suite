"""
pipelines/system_pipeline.py
----------------------------
Pipeline logic for 'frs system' command.
Run full production system pipeline (Object + Face + Recognition + Logging).
"""

from __future__ import annotations

import time
import cv2
import signal
from pathlib import Path
from typing import Dict

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


def run_system_pipeline() -> None:
    """Pipeline logic for 'frs system'."""
    config = get_config()
    ensure_runtime_directories(config)
    setup_system_logger(config)
    event_logger = get_event_logger(config)

    console.print("[bold cyan][Pipeline] 'frs system' started.[/bold cyan]")

    # Check if EAR/MAR features are requested (placeholder)
    if config.system.enable_ear_mar:
        logger.warning("EAR/MAR features requested in config, but not implemented yet.")

    # Initialize detection and recognition components
    object_detector = ObjectDetector(config) if config.detection.enable_object_detection else None
    face_detector = FaceDetector(config) if config.detection.enable_face_detection else None
    embedder = FaceEmbedder(config)
    matcher = FaceMatcher(config)

    # Initialize trackers
    object_tracker = Tracker(config)
    face_tracker = Tracker(config)

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

            for source, frame_packet in frames_map.items():
                if frame_packet.status == "end_of_stream":
                    continue

                all_ended = False
                if frame_packet.status != "ok" or frame_packet.display_frame is None:
                    continue

                display_img = frame_packet.display_frame
                inf_img = frame_packet.inference_frame

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
                if face_detector and inf_img is not None:
                    face_dets = face_detector.detect(inf_img)
                    raw_faces = [(f.bbox, f.confidence, "face", f.embedding) for f in face_dets]
                    tracked_faces = face_tracker.update(raw_faces)

                    # Process tracked faces for recognition
                    current_time = time.time()
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

                        display_manager.draw_detection(
                            display_img, (dfx1, dfy1, dfx2, dfy2),
                            label,
                            color=box_color,
                            thickness=3,
                        )

                        # Save recognized face crops with cooldown logic
                        if (config.recognition.save_crops and
                            (current_time - f.last_cropped_timestamp) >= config.recognition.cooldown_seconds):

                            if person_name != "unknown" or config.recognition.unknown_label != "unknown":
                                # Save crop
                                padding = 10
                                crop_x1 = max(0, dfx1 - padding)
                                crop_y1 = max(0, dfy1 - padding)
                                crop_x2 = min(display_img.shape[1], dfx2 + padding)
                                crop_y2 = min(display_img.shape[0], dfy2 + padding)
                                crop_img = display_img[crop_y1:crop_y2, crop_x1:crop_x2]

                                if crop_img.size > 0:
                                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                                    # Format: name_timestamp_cameraid.jpg as per PROGRESS.md
                                    crop_filename = f"{person_name}_{timestamp_str}_{source.camera_id}.jpg"
                                    crop_dir = Path(config.paths.output_dir) / config.recognition.recognize_crop_dir
                                    crop_dir.mkdir(parents=True, exist_ok=True)
                                    crop_path = crop_dir / crop_filename
                                    cv2.imwrite(str(crop_path), crop_img)

                                    # Log recognition event
                                    event_logger.log_event(
                                        event_type="FACE_RECOGNIZED",
                                        camera_id=source.camera_id,
                                        class_or_name=person_name,
                                        confidence=similarity,
                                        bbox=(dfx1, dfy1, dfx2, dfy2),
                                        crop_path=str(crop_path),
                                        track_id=f.track_id,
                                    )

                                    # Update cooldown timestamp
                                    f.last_cropped_timestamp = current_time

                        # Log detection event (regardless of recognition result)
                        if (current_time - f.last_logged_timestamp) >= config.logging.event_log_cooldown_seconds:
                            event_logger.log_event(
                                event_type="FACE_DETECTED",
                                camera_id=source.camera_id,
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
                    display_manager.draw_detection(
                        display_img, (dx1, dy1, dx2, dy2),
                        label,
                        color=class_color(obj.class_name),
                        thickness=3,
                    )

                    # Log object detection events with cooldown
                    if (current_time - obj.last_logged_timestamp) >= config.logging.event_log_cooldown_seconds:
                        event_logger.log_event(
                            event_type="OBJECT_DETECTED",
                            camera_id=source.camera_id,
                            class_or_name=obj.class_name,
                            confidence=obj.confidence,
                            bbox=(dx1, dy1, dx2, dy2),
                            track_id=obj.track_id,
                        )
                        obj.last_logged_timestamp = current_time

                display_frames[source.camera_id] = display_img
                detection_counts[source.camera_id] = len(tracked_objects) + len(tracked_faces)

            if all_ended:
                logger.info("All video streams ended.")
                break

            # Display frames — skip window-status check on first iterations
            # when capture threads haven't produced any frames yet.
            if not display_frames:
                time.sleep(0.01)
                continue

            window_status = display_manager.display_frames(display_frames, detection_counts)

            # Check if any window was closed or if 'q' was pressed
            if not any(window_status.values()) or cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key pressed or windows closed. Shutting down.")
                break

    finally:
        manager.stop_all()
        display_manager.cleanup()
        cv2.destroyAllWindows()