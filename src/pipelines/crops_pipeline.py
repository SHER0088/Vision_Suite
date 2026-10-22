"""
pipelines/crops_pipeline.py
---------------------------
Pipeline logic for 'frs crops' command.
Detect objects/faces and save cropped images per class with cooldown logic.
"""

from __future__ import annotations

import time
import cv2
from pathlib import Path
from typing import Dict

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_event_logger, get_logger, setup_system_logger
from src.core.tracker import Tracker
from src.core.video_source import VideoSourceManager
from src.detection.face_detector import FaceDetector
from src.detection.object_detector import ObjectDetector
from src.display.display_manager import DisplayManager, class_color
from rich.console import Console
import signal

logger = get_logger("crops_pipeline")
console = Console()


def run_crops_pipeline() -> None:
    """Pipeline logic for 'frs crops'."""
    config = get_config()
    ensure_runtime_directories(config)
    setup_system_logger(config)
    event_logger = get_event_logger(config)

    object_detector = ObjectDetector(config) if config.detection.enable_object_detection else None
    face_detector = FaceDetector(config) if config.detection.enable_face_detection else None

    object_tracker = Tracker(config)
    face_tracker = Tracker(config)

    manager = VideoSourceManager(config)
    active_sources = manager.get_active_sources()

    if not active_sources:
        console.print("[bold red]No enabled camera sources found in config.yaml![/bold red]")
        return

    logger.info("Starting crops pipeline on cameras: %s", [s.camera_id for s in active_sources])
    console.print(
        "[bold green]Crops Pipeline running. Press 'q' in any video window to exit.[/bold green]"
    )

    # Initialize display manager
    display_manager = DisplayManager(config)
    camera_ids = [s.camera_id for s in active_sources]
    display_manager.initialize_windows(camera_ids)

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
        while True:
            if shutdown_requested:
                logger.info("Shutdown requested, breaking loop.")
                break
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

                # 2. Face Detections
                if face_detector and inf_img is not None:
                    face_dets = face_detector.detect(inf_img)
                    raw_faces = [(f.bbox, f.confidence, "face", f.embedding) for f in face_dets]
                    tracked_faces = face_tracker.update(raw_faces)

                # Process tracked objects for cropping
                current_time = time.time()
                for obj in tracked_objects:
                    x1, y1, x2, y2 = obj.bbox
                    dx1, dy1 = int(x1 * scale_x), int(y1 * scale_y)
                    dx2, dy2 = int(x2 * scale_x), int(y2 * scale_y)

                    # Draw bounding box
                    label = f"{obj.class_name} #{obj.track_id} ({obj.confidence:.2f})"
                    display_manager.draw_detection(
                        display_img, (dx1, dy1, dx2, dy2),
                        label,
                        color=class_color(obj.class_name),
                        thickness=3,
                    )

                    # Save crops with cooldown logic
                    if (current_time - obj.last_cropped_timestamp) >= config.crops.cooldown_seconds:
                        # Extract and save object crop
                        if config.crops.save_object_crops:
                            padding = config.crops.crop_padding
                            crop_x1 = max(0, dx1 - padding)
                            crop_y1 = max(0, dy1 - padding)
                            crop_x2 = min(display_img.shape[1], dx2 + padding)
                            crop_y2 = min(display_img.shape[0], dy2 + padding)
                            crop_img = display_img[crop_y1:crop_y2, crop_x1:crop_x2]

                            if crop_img.size > 0 and min(crop_img.shape[0], crop_img.shape[1]) >= config.crops.min_crop_size:
                                # Save crop
                                timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                                crop_filename = f"{obj.class_name}_{obj.track_id}_{timestamp_str}.jpg"
                                crop_path = Path(config.paths.output_dir) / config.crops.object_crop_dir / crop_filename
                                crop_path.parent.mkdir(parents=True, exist_ok=True)
                                cv2.imwrite(str(crop_path), crop_img)

                                # Log the crop saved event
                                event_logger.log_event(
                                    event_type="OBJECT_CROP_SAVED",
                                    camera_id=source.camera_id,
                                    class_or_name=obj.class_name,
                                    confidence=obj.confidence,
                                    bbox=(dx1, dy1, dx2, dy2),
                                    crop_path=str(crop_path),
                                    track_id=obj.track_id,
                                )

                                # Update cooldown timestamp
                                obj.last_cropped_timestamp = current_time

                # Process tracked faces for cropping
                if face_detector and inf_img is not None:
                    for f in tracked_faces:
                        fx1, fy1, fx2, fy2 = f.bbox
                        dfx1, dfy1 = int(fx1 * scale_x), int(fy1 * scale_y)
                        dfx2, dfy2 = int(fx2 * scale_x), int(fy2 * scale_y)

                        # Draw bounding box
                        flabel = f"Face #{f.track_id} ({f.confidence:.2f})"
                        display_manager.draw_detection(
                            display_img, (dfx1, dfy1, dfx2, dfy2),
                            flabel,
                            color=class_color("face"),
                            thickness=3,
                        )

                        # Save face crops with cooldown logic
                        if (current_time - f.last_cropped_timestamp) >= config.crops.cooldown_seconds:
                            if config.crops.save_face_crops:
                                padding = config.crops.crop_padding
                                crop_x1 = max(0, dfx1 - padding)
                                crop_y1 = max(0, dfy1 - padding)
                                crop_x2 = min(display_img.shape[1], dfx2 + padding)
                                crop_y2 = min(display_img.shape[0], dfy2 + padding)
                                crop_img = display_img[crop_y1:crop_y2, crop_x1:crop_x2]

                                if crop_img.size > 0 and min(crop_img.shape[0], crop_img.shape[1]) >= config.crops.min_crop_size:
                                    # Save crop
                                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                                    crop_filename = f"face_{f.track_id}_{timestamp_str}.jpg"
                                    crop_path = Path(config.paths.output_dir) / config.crops.face_crop_dir / crop_filename
                                    crop_path.parent.mkdir(parents=True, exist_ok=True)
                                    cv2.imwrite(str(crop_path), crop_img)

                                    # Log the crop saved event
                                    event_logger.log_event(
                                        event_type="FACE_CROP_SAVED",
                                        camera_id=source.camera_id,
                                        class_or_name="face",
                                        confidence=f.confidence,
                                        bbox=(dfx1, dfy1, dfx2, dfy2),
                                        crop_path=str(crop_path),
                                        track_id=f.track_id,
                                    )

                                    # Update cooldown timestamp
                                    f.last_cropped_timestamp = current_time

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