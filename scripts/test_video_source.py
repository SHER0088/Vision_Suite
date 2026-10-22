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
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    cv2 = importlib.import_module("cv2")
except ImportError as e:
    raise ImportError(
        "OpenCV is required for this script. Install it with: python -m pip install opencv-python"
    ) from e
from src.core.config_loader import get_config, ensure_runtime_directories
from src.core.video_source import VideoSource, VideoSourceManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def list_available_sources(config):
    """Print available input sources from the config.input section."""
    input_cfg = config.input
    print(f"\nInput type: {input_cfg.type}")
    sources = input_cfg.active_sources()
    print(f"Active sources ({len(sources)}):")
    for s in sources:
        print(f"  - id={s['id']}, type={s['type']}, source={s['source']}")


def main():
    if len(sys.argv) > 1:
        camera_id = sys.argv[1]
    else:
        camera_id = None

    config = get_config()
    ensure_runtime_directories(config)

    list_available_sources(config)

    # Use VideoSourceManager to test all active sources
    manager = VideoSourceManager(config)
    active = manager.get_active_sources()

    if not active:
        print("No active sources found. Check config.yaml input section.")
        return

    print(f"\nTesting {len(active)} source(s). Press 'q' to quit.\n")

    try:
        while True:
            frames_map = manager.read_all()
            if not frames_map:
                continue

            any_valid = False
            for source, packet in frames_map.items():
                if packet.status == "end_of_stream":
                    print(f"[{source.camera_id}] End of stream.")
                    continue
                if packet.status == "disconnected":
                    print(f"[{source.camera_id}] Disconnected, retrying...")
                    continue
                if packet.status == "no_new_frame":
                    continue
                if packet.status != "ok" or packet.display_frame is None:
                    continue

                any_valid = True
                cv2.imshow(f"VideoSource Test - {source.camera_id}", packet.display_frame)

                if packet.inference_frame is not None:
                    inf = packet.inference_frame.copy()
                    cv2.putText(
                        inf, "Inference-res preview", (5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                    )
                    cv2.imshow(f"VideoSource Test - {source.camera_id} (inf)", inf)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        manager.stop_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
