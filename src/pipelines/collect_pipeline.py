"""
pipelines/collect_pipeline.py
-----------------------------
Pipeline logic for 'frs collect' command.
Collect face samples for a person based purely on config.yaml settings.
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_event_logger, get_logger, setup_system_logger
from src.core.tracker import Tracker
from src.core.video_source import VideoSourceManager
from src.detection.face_detector import FaceDetector
from src.recognition.embedder import FaceEmbedder
from src.display.display_manager import DisplayManager, class_color
from rich.console import Console
import signal

logger = get_logger("collect_pipeline")
console = Console()


def run_collect_pipeline() -> None:
    """Pipeline logic for 'frs collect' (Fully config.yaml driven)."""
    config = get_config()
    ensure_runtime_directories(config)
    setup_system_logger(config)
    event_logger = get_event_logger(config)

    person_name = config.data_collection.person_name
    console.print(f"[bold cyan][Pipeline] 'frs collect' started for person: {person_name}[/bold cyan]")

    face_detector = FaceDetector(config) if config.detection.enable_face_detection else None
    embedder = FaceEmbedder(config)

    face_tracker = Tracker(config)

    manager = VideoSourceManager(config)
    active_sources = manager.get_active_sources()

    if not active_sources:
        console.print("[bold red]No enabled camera sources found in config.yaml![/bold red]")
        return

    logger.info("Starting collect pipeline for person: %s on cameras: %s", person_name, [s.camera_id for s in active_sources])
    console.print(
        "[bold green]Collect Pipeline running. Follow on-screen prompts for each pose.[/bold green]"
    )

    # Initialize display manager
    display_manager = DisplayManager(config)
    camera_ids = [s.camera_id for s in active_sources]
    display_manager.initialize_windows(camera_ids)

    # Setup directories for saving face samples
    person_raw_dir = Path(config.paths.data_dir) / config.paths.faces_dir / person_name / "raw"
    person_raw_dir.mkdir(parents=True, exist_ok=True)

    # Collection state
    poses = config.data_collection.poses
    samples_per_pose = config.data_collection.samples_per_pose
    quality_checks = config.data_collection.quality_checks
    capture_interval = config.data_collection.capture_interval_seconds
    overwrite_exists = config.data_collection.overwrite_if_exists

    # Handle existing data
    if person_raw_dir.exists() and any(person_raw_dir.iterdir()):
        if overwrite_exists:
            # Clear existing data
            for file in person_raw_dir.glob("*"):
                file.unlink()
            logger.info("Cleared existing data for person: %s (overwrite_if_exists=true)", person_name)
        else:
            logger.warning("Data already exists for person: %s and overwrite_if_exists=false", person_name)
            console.print(f"[yellow]Warning: Data already exists for {person_name}. Use overwrite_if_exists=true to replace.[/yellow]")
            return

    # Collection progress tracking
    current_pose_index = 0
    samples_collected_for_pose = 0
    last_capture_time = 0

    # Signal handling for graceful shutdown
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        shutdown_requested = True
        logger.info("Signal %d received, initiating shutdown...", signum)

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while current_pose_index < len(poses) and not shutdown_requested:
            current_pose = poses[current_pose_index]
            frames_map = manager.read_all()
            if not frames_map:
                time.sleep(0.01)
                continue

            # Process frames from all active sources
            display_frames: Dict[str, any] = {}
            detection_counts: Dict[str, int] = {}
            any_frame_valid = False

            for source, frame_packet in frames_map.items():
                if frame_packet.status != "ok" or frame_packet.display_frame is None:
                    continue

                any_frame_valid = True
                display_img = frame_packet.display_frame
                inf_img = frame_packet.inference_frame

                disp_h, disp_w = display_img.shape[:2]
                inf_h, inf_w = inf_img.shape[:2]

                scale_x = disp_w / float(inf_w)
                scale_y = disp_h / float(inf_h)

                tracked_faces: List = []

                # Face detection for collection
                if face_detector and inf_img is not None:
                    face_dets = face_detector.detect(inf_img)
                    raw_faces = [(f.bbox, f.confidence, "face", f.embedding) for f in face_dets]
                    tracked_faces = face_tracker.update(raw_faces)

                    # Process the largest/tracked face for collection
                    if tracked_faces:
                        # Use the first tracked face (or could implement logic to pick largest/center)
                        f = tracked_faces[0]
                        fx1, fy1, fx2, fy2 = f.bbox
                        dfx1, dfy1 = int(fx1 * scale_x), int(fy1 * scale_y)
                        dfx2, dfy2 = int(fx2 * scale_x), int(fy2 * scale_y)

                        # Draw bounding box and pose instructions
                        pose_label = f"Pose: {current_pose.upper()} ({samples_collected_for_pose}/{samples_per_pose})"
                        display_manager.draw_detection(
                            display_img, (dfx1, dfy1, dfx2, dfy2),
                            pose_label,
                            color=class_color("face"),
                            thickness=3,
                        )

                        # Check if we should capture a sample
                        current_time = time.time()
                        if (current_time - last_capture_time) >= capture_interval:
                            # Extract face region for quality check
                            face_img = display_img[dfy1:dfy2, dfx1:dfx2]
                            if face_img.size > 0:
                                # Quality checks
                                gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                                brightness = np.mean(gray_face)

                                # Calculate blur using Laplacian variance
                                blur_score = cv2.Laplacian(gray_face, cv2.CV_64F).var()

                                face_width = dfx2 - dfx1
                                face_height = dfy2 - dfy1
                                min_face_size = quality_checks.min_face_size_px

                                # Check quality gates
                                quality_ok = (
                                    quality_checks.min_brightness <= brightness <= quality_checks.max_brightness and
                                    blur_score > quality_checks.blur_threshold and
                                    min(face_width, face_height) >= min_face_size
                                )

                                # Visual feedback based on quality
                                if quality_ok:
                                    # Draw green box for good quality
                                    cv2.rectangle(display_img, (dfx1, dfy1), (dfx2, dfy2), (0, 255, 0), 3)
                                    quality_text = "QUALITY: GOOD"
                                    quality_color = (0, 255, 0)

                                    # Capture sample if we need more for this pose
                                    if samples_collected_for_pose < samples_per_pose:
                                        # Save the raw face image
                                        timestamp_str = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                                        filename = f"{current_pose}_{samples_collected_for_pose:03d}_{timestamp_str}.jpg"
                                        filepath = person_raw_dir / filename
                                        cv2.imwrite(str(filepath), face_img)

                                        samples_collected_for_pose += 1
                                        last_capture_time = current_time

                                        logger.info(
                                            "Saved sample %d/%d for pose '%s' (brightness: %.1f, blur: %.1f)",
                                            samples_collected_for_pose, samples_per_pose, current_pose,
                                            brightness, blur_score
                                        )

                                        # Check if we've completed this pose
                                        if samples_collected_for_pose >= samples_per_pose:
                                            current_pose_index += 1
                                            samples_collected_for_pose = 0
                                            logger.info("Completed pose: %s", current_pose)

                                            # Show completion message briefly
                                            completion_overlay = display_img.copy()
                                            cv2.rectangle(completion_overlay,
                                                        (0, 0),
                                                        (display_img.shape[1], display_img.shape[0]),
                                                        (0, 255, 0), -1)
                                            cv2.addWeighted(completion_overlay, 0.3, display_img, 0.7, 0, display_img)
                                            cv2.putText(
                                                display_img, f"POSE {current_pose.upper()} COMPLETE!",
                                                (display_img.shape[1]//2 - 150, display_img.shape[0]//2),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3,
                                            )
                                else:
                                    # Draw red box for poor quality
                                    cv2.rectangle(display_img, (dfx1, dfy1), (dfx2, dfy2), (0, 0, 255), 3)
                                    quality_text = f"QUALITY: POOR (B:{brightness:.0f} BL:{blur_score:.0f} SZ:{min(face_width,face_height)}"
                                    quality_color = (0, 0, 255)

                                # Display quality feedback
                                cv2.putText(
                                    display_img, quality_text,
                                    (dfx1, dfy2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, quality_color, 2,
                                )

                display_frames[source.camera_id] = display_img
                detection_counts[source.camera_id] = len(tracked_faces)

            if not any_frame_valid:
                time.sleep(0.01)
                continue

            # Display frames — skip window-status check on first iterations
            # when capture threads haven't produced any frames yet.
            if not display_frames:
                time.sleep(0.01)
                continue

            window_status = display_manager.display_frames(display_frames, detection_counts)

            # Check for quit signal
            if not any(window_status.values()) or cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key pressed or windows closed during collection.")
                break

            # Additional check for shutdown requested (in case it was set during processing)
            if shutdown_requested:
                logger.info("Shutdown requested, breaking loop.")
                break

        # If we exited the loop due to shutdown requested, we still want to log completion?
        # But we break out and go to finally.
        if shutdown_requested:
            logger.info("Collection interrupted by shutdown signal.")
        elif current_pose_index >= len(poses):
            # Collection complete
            logger.info("Face collection completed for person: %s", person_name)
            console.print(f"[bold green]Collection completed for {person_name}! Samples saved to:[/bold green]")
            console.print(f"[dim]{person_raw_dir}[/dim]")

    except Exception as exc:
        logger.error("Error during collection: %s", exc)
        console.print(f"[bold red]Error during collection: {exc}[/bold red]")
    finally:
        manager.stop_all()
        display_manager.cleanup()
        cv2.destroyAllWindows()