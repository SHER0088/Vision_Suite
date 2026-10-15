"""
Manual test harness for logger.py.
Simulates detection events and verifies both system logging and
cooldown-throttled CSV event logging work correctly.

Usage:
    python scripts/test_logger.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config_loader import get_config, ensure_runtime_directories
from src.core.logger import setup_system_logger, get_logger, get_event_logger


def main():
    config = get_config()
    ensure_runtime_directories(config)

    setup_system_logger(config)
    logger = get_logger("test_logger_script")
    event_logger = get_event_logger(config)

    logger.info("Starting logger test script.")
    logger.warning("This is a sample warning message.")

    print("\n--- Testing cooldown behavior ---")
    print(f"Configured cooldown: {config.logging.event_log_cooldown_seconds}s\n")

    # Fire the SAME event 5 times rapidly - only the first should be logged
    for i in range(5):
        written = event_logger.log_event(
            event_type="detection",
            camera_id="webcam",
            class_or_name="person",
            confidence=0.91,
            bbox=(100, 100, 300, 400),
            crop_path="",
            track_id=1,
        )
        print(f"Attempt {i+1}: {'WRITTEN' if written else 'SKIPPED (cooldown)'}")
        time.sleep(0.5)

    print("\n--- Testing different track_id (should always write) ---")
    written = event_logger.log_event(
        event_type="detection",
        camera_id="webcam",
        class_or_name="car",
        confidence=0.87,
        bbox=(50, 50, 200, 200),
        crop_path="",
        track_id=2,
    )
    print(f"Different track_id: {'WRITTEN' if written else 'SKIPPED'}")

    logger.info("Logger test script completed.")
    print(f"\nCheck output at: {config.paths.logs_dir}/")
    print("- events_<date>.csv should have exactly 2 rows (track_id=1 once, track_id=2 once)")
    print("- system_<date>.log should contain the info/warning messages above")


if __name__ == "__main__":
    main()