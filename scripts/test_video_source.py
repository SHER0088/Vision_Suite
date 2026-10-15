"""
Manual test harness for video_source.py.
Run this BEFORE building anything else — confirm all 3 input modes work
cleanly on your machine first.

Usage:
    python scripts/test_video_source.py webcam
    python scripts/test_video_source.py sample_video
    python scripts/test_video_source.py entry_gate
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from src.core.config_loader import get_config, ensure_runtime_directories
from src.core.video_source import VideoSource

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/test_video_source.py <camera_id>")
        sys.exit(1)

    camera_id = sys.argv[1]
    config = get_config()
    ensure_runtime_directories(config)

    try:
        cam_cfg = config.get_camera(camera_id)
    except KeyError as e:
        print(e)
        sys.exit(1)

    display_res = config.resolution.display.as_tuple()
    inference_res = config.resolution.inference.as_tuple()

    print(f"Starting VideoSource test for '{camera_id}' (type={cam_cfg.type})")
    print("Press 'q' to quit.\n")

    source = VideoSource(cam_cfg, display_res, inference_res)

    try:
        while True:
            status, display_frame, inference_frame = source.read()

            if status == "end_of_stream":
                print("End of offline video reached.")
                break
            elif status == "disconnected":
                print(f"[{camera_id}] Disconnected, retrying...")
                continue
            elif status == "no_new_frame":
                continue  # live/rtsp: nothing new yet, loop again immediately

            cv2.imshow(f"VideoSource Test - {camera_id}", display_frame)
            cv2.putText(
                inference_frame, "Inference-res preview", (5, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
            cv2.imshow(f"VideoSource Test - {camera_id} (inference res)", inference_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()