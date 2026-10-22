"""
pipelines/detect_pipeline.py
---------------------------
Pipeline logic for 'frs detect' command.
Runs live object & face detection across configured camera sources (No saving).
"""

from __future__ import annotations

import time
import signal
import cv2
from typing import Dict, List

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_event_logger, get_logger, setup_system_logger
from src.core.tracker import Tracker
from src.core.video_source import VideoSourceManager
from src.detection.face_detector import FaceDetector
from src.detection.object_detector import ObjectDetector
from src.display.display_manager import DisplayManager, class_color
from rich.console import Console

logger = get_logger("detect_pipeline")
console = Console()


def run_detect_pipeline() -> None:
    """Pipeline logic for 'frs detect'."""
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

    logger.info("Starting detect pipeline on cameras: %s", [s.camera_id for s in active_sources])
    console.print(
        "[bold green]Detect Pipeline running. Press 'q' in any video window to exit.[/bold green]"
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

                    event_logger.log_event(
                        event_type="OBJECT_DETECTED",
                        camera_id=source.camera_id,
                        class_or_name=obj.class_name,
                        confidence=obj.confidence,
                        bbox=(dx1, dy1, dx2, dy2),
                        track_id=obj.track_id,
                    )

                tracked_faces: List = []

                # 2. Face Detections
                if face_detector and inf_img is not None:
                    face_dets = face_detector.detect(inf_img)
                    raw_faces = [(f.bbox, f.confidence, "face", f.embedding) for f in face_dets]
                    tracked_faces = face_tracker.update(raw_faces)

                    for f in tracked_faces:
                        fx1, fy1, fx2, fy2 = f.bbox
                        dfx1, dfy1 = int(fx1 * scale_x), int(fy1 * scale_y)
                        dfx2, dfy2 = int(fx2 * scale_x), int(fy2 * scale_y)

                        flabel = f"Face #{f.track_id} ({f.confidence:.2f})"
                        display_manager.draw_detection(
                            display_img, (dfx1, dfy1, dfx2, dfy2),
                            flabel,
                            color=class_color("face"),
                            thickness=3,
                        )

                        event_logger.log_event(
                            event_type="FACE_DETECTED",
                            camera_id=source.camera_id,
                            class_or_name="face",
                            confidence=f.confidence,
                            bbox=(dfx1, dfy1, dfx2, dfy2),
                            track_id=f.track_id,
                        )

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