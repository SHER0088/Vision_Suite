"""
test.py
----------
CLI entrypoint for Vision_Suite.
Run `python test.py -h` to see all available commands.
"""
import cv2
import typer

from src.core.config_loader import get_config, ensure_runtime_directories
from src.core.logger import setup_system_logger, get_logger
from src.core.video_source import VideoSourceManager
from src.detection.object_detector import ObjectDetector
from src.detection.face_detector import FaceDetector

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

app = typer.Typer(
    help="Vision_Suite - Object Detection, Face Detection & Face Recognition System",
    context_settings=CONTEXT_SETTINGS,
)
frs_app = typer.Typer(
    help="Core FRS pipeline commands",
    context_settings=CONTEXT_SETTINGS,
)
app.add_typer(frs_app, name="frs")

logger = get_logger("cli")


def _draw_detections(frame, object_dets, face_dets):
    for det in object_dets:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{det.class_name} {det.confidence:.2f}",
                    (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for fdet in face_dets:
        x1, y1, x2, y2 = fdet.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"face {fdet.confidence:.2f}",
                    (x1, max(y1 - 8, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return frame


def _scale_bbox(bbox, scale_x, scale_y):
    x1, y1, x2, y2 = bbox
    return (int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y))


@frs_app.command()
def detect():
    """Open camera UI and run live object + face detection (no saving)."""
    config = get_config()
    ensure_runtime_directories(config)
    setup_system_logger(config)

    object_detector = ObjectDetector(config) if config.detection.enable_object_detection else None
    face_detector = FaceDetector(config) if config.detection.enable_face_detection else None

    manager = VideoSourceManager(config)
    logger.info(f"Starting 'frs detect' on cameras: {list(manager.sources.keys())}")

    scale_x = config.resolution.display.width / config.resolution.inference.width
    scale_y = config.resolution.display.height / config.resolution.inference.height

    try:
        while True:
            results = manager.read_all()
            offline_done = []

            for cam_id, (status, display_frame, inference_frame) in results.items():
                if status in ("no_new_frame", "disconnected"):
                    continue
                if status == "end_of_stream":
                    offline_done.append(cam_id)
                    continue

                object_dets = object_detector.detect(inference_frame) if object_detector else []
                face_dets = face_detector.detect(inference_frame) if face_detector else []

                object_dets = [d._replace(bbox=_scale_bbox(d.bbox, scale_x, scale_y)) for d in object_dets]
                face_dets = [d._replace(bbox=_scale_bbox(d.bbox, scale_x, scale_y)) for d in face_dets]

                frame_to_show = _draw_detections(display_frame, object_dets, face_dets)
                cv2.imshow(f"{config.display.window_name_prefix} - {cam_id}", frame_to_show)

            if offline_done and len(offline_done) == len(manager.sources):
                logger.info("All offline sources finished. Exiting.")
                break

            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("Quit key pressed. Shutting down.")
                break
    finally:
        manager.release_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app()