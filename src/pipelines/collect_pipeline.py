"""
Pipeline logic for 'frs enroll' and 'frs collect' commands.

- run_enroll_pipeline: single-image enrollment from a reference photo.
- run_collect_pipeline: camera-based multi-image capture with pose guidance
  (straight / left / right / up / down), saving raw samples to
  data/faces/raw/<name>/ for later training via 'frs train'.
"""

from __future__ import annotations

import cv2
import time
from pathlib import Path
from typing import Optional

import typer

from src.core.config_loader import AppConfig, ensure_runtime_directories, get_config
from src.core.logger import get_logger
from src.core.video_source import VideoSourceManager
from src.detection.face_detector import FaceDetector
from src.display.display_manager import DisplayManager
from src.recognition.embedder import FaceEmbedder
from src.recognition.matcher import FaceMatcher
from rich.console import Console

logger = get_logger("enroll_pipeline")
console = Console()

_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def run_enroll_pipeline(
    image_path: Optional[str] = None,
    name: Optional[str] = None,
    employee_id: Optional[str] = None,
    metadata: str = "{}",
) -> None:
    """Enroll person(s) with reference photo(s).

    If image_path is omitted, images are auto-discovered from
    ``config.paths.raw_faces_dir`` (data/faces/raw/). Missing name/employee_id
    are collected via interactive prompts.
    """
    config = get_config()
    ensure_runtime_directories(config)

    # Resolve defaults from config
    if name is None:
        name = config.enrollment.default_name or ""
    if employee_id is None:
        employee_id = config.enrollment.default_employee_id or ""
    employee_id = str(employee_id or "")

    # Determine image paths
    if image_path is None:
        raw_dir = Path(config.paths.raw_faces_dir)
        if not raw_dir.exists():
            console.print(f"[bold red]Raw faces directory not found: {raw_dir}[/bold red]")
            console.print(
                "[yellow]Run 'frs collect' to capture images, or place reference "
                "photos in data/faces/raw/ and try again.[/yellow]"
            )
            return
        image_paths = sorted(
            p for p in raw_dir.iterdir()
            if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_paths:
            console.print(f"[bold red]No images found in {raw_dir}[/bold red]")
            console.print(
                "[yellow]Run 'frs collect' to capture images from camera, or place "
                "reference photos (jpg/png/etc.) in data/faces/raw/ and try again.[/yellow]"
            )
            return
        console.print(f"[cyan]Auto-discovered {len(image_paths)} image(s) in {raw_dir}[/cyan]")
    else:
        image_paths = [Path(image_path)]

    face_detector = FaceDetector(config)
    embedder = FaceEmbedder(config)
    matcher = FaceMatcher(config)

    for img_path in image_paths:
        # Interactive prompts for missing details
        person_name = name
        if not person_name:
            person_name = typer.prompt(f"Enter name for {img_path.name}")

        emp_id = employee_id
        if not emp_id:
            emp_id = typer.prompt(f"Enter employee ID for {img_path.name}", default="")
        emp_id = str(emp_id or "")

        if not person_name:
            console.print(f"[bold red]Skipping {img_path.name}: name is required[/bold red]")
            continue

        # Load and process image
        img = cv2.imread(str(img_path))
        if img is None:
            console.print(f"[bold red]Failed to load image: {img_path}[/bold red]")
            continue

        face_dets = face_detector.detect(img)
        if len(face_dets) == 0:
            console.print(f"[bold red]No face detected in {img_path.name}. Skipping.[/bold red]")
            continue
        if len(face_dets) > 1:
            console.print(f"[bold red]Multiple faces in {img_path.name}. Skipping.[/bold red]")
            continue

        fd = face_dets[0]
        x1, y1, x2, y2 = fd.bbox
        face_img = img[y1:y2, x1:x2]
        if face_img.size == 0:
            console.print(f"[bold red]Failed to crop face in {img_path.name}.[/bold red]")
            continue

        embedding = embedder.get_embedding(face_img)
        if embedding is None:
            console.print(f"[bold red]Failed to generate embedding for {img_path.name}.[/bold red]")
            continue

        identity_id = matcher.enroll(
            None,  # auto-generate identity_code
            person_name,
            str(img_path),
            embedding,
            employee_id=emp_id,
            metadata=metadata,
        )
        if identity_id is None:
            console.print(
                f"[bold red]Enrollment failed for {person_name} "
                f"(duplicate identity_code or employee_id).[/bold red]"
            )
            continue

        enrolled = matcher.face_db.get(identity_id, {})
        actual_code = enrolled.get("identity_code", "unknown")
        console.print(
            f"[bold green]Enrolled {person_name} ({actual_code}, "
            f"employee_id={emp_id}) from {img_path.name}[/bold green]"
        )


def run_collect_pipeline(
    name: Optional[str] = None,
    samples_per_pose: Optional[int] = None,
) -> None:
    """Camera-based data collection with pose guidance.

    Captures multiple face images per pose (straight / left / right / up / down)
    and saves them to data/faces/raw/<name>/.  Use 'frs train' afterwards to
    generate embeddings and enroll the person in the recognition database.

    Args:
        name: Person's name. Defaults to config.enrollment.default_name,
              falling back to an interactive prompt.
        samples_per_pose: Number of samples to capture per pose. Defaults to
                          config.data_collection.samples_per_pose.
    """
    config = get_config()
    ensure_runtime_directories(config)

    # Resolve name
    if name is None:
        name = config.enrollment.default_name or ""
    if not name:
        name = typer.prompt("Enter person's name for enrollment")

    # Resolve samples per pose
    if samples_per_pose is None:
        samples_per_pose = config.data_collection.samples_per_pose

    # Create output directory: data/faces/raw/<name>/
    person_dir = Path(config.paths.raw_faces_dir) / name
    person_dir.mkdir(parents=True, exist_ok=True)

    console.print(
        f"[bold cyan]Starting collection for '{name}' -> {person_dir}[/bold cyan]"
    )
    console.print(
        f"[dim]Poses: {', '.join(config.data_collection.poses)}  "
        f"Samples per pose: {samples_per_pose}[/dim]"
    )
    console.print("[yellow]Position your face in the circle. Images capture automatically.[/yellow]")
    console.print("[yellow]Press 'q' to cancel at any time.[/yellow]\n")

    # Initialize components
    face_detector = FaceDetector(config)
    display_manager = DisplayManager(config)
    manager = VideoSourceManager(config)
    active_sources = manager.get_active_sources()

    if not active_sources:
        console.print("[bold red]No camera sources found![/bold red]")
        return

    camera_id = active_sources[0].camera_id
    display_manager.initialize_windows([camera_id])

    poses = list(config.data_collection.poses)
    capture_interval = config.data_collection.capture_interval_seconds
    total_target = len(poses) * samples_per_pose
    total_collected = 0

    try:
        for pose in poses:
            pose_collected = 0
            last_capture_time = 0.0

            console.print(f"[cyan]>>> Pose: {pose.upper()}  Collect {samples_per_pose} samples...[/cyan]")

            while pose_collected < samples_per_pose:
                frames_map = manager.read_all()
                if not frames_map:
                    time.sleep(0.01)
                    continue

                for video_source, frame_packet in frames_map.items():
                    if frame_packet.status != "ok" or frame_packet.display_frame is None:
                        continue

                    frame = frame_packet.display_frame.copy()
                    current_time = time.time()
                    h, w = frame.shape[:2]

                    # Detect face
                    face_dets = face_detector.detect(frame)

                    face_centered = False
                    face_bbox = None
                    face_ok = False

                    if len(face_dets) == 1:
                        fd = face_dets[0]
                        x1, y1, x2, y2 = fd.bbox
                        face_bbox = (x1, y1, x2, y2)

                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        face_w = x2 - x1
                        face_h = y2 - y1

                        is_large_enough = face_w > w * 0.15 and face_h > h * 0.15
                        is_centered = (
                            abs(cx - w // 2) < w * 0.20
                            and abs(cy - h // 2) < h * 0.20
                        )

                        face_centered = is_centered
                        face_ok = is_large_enough and is_centered

                        # Auto-capture when face is well-positioned
                        if face_ok and (current_time - last_capture_time) >= capture_interval:
                            pad = 25
                            crop_x1 = max(0, x1 - pad)
                            crop_y1 = max(0, y1 - pad)
                            crop_x2 = min(w, x2 + pad)
                            crop_y2 = min(h, y2 + pad)
                            face_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                            if face_img.size > 0 and min(face_img.shape[:2]) >= 60:
                                timestamp = time.strftime("%Y%m%d_%H%M%S_%f")
                                img_path = person_dir / f"{pose}_{timestamp}.jpg"
                                cv2.imwrite(str(img_path), face_img)
                                pose_collected += 1
                                total_collected += 1
                                last_capture_time = current_time
                                console.print(
                                    f"[dim]  Captured {pose_collected}/{samples_per_pose} "
                                    f"({pose})  Total: {total_collected}/{total_target}[/dim]"
                                )
                    elif len(face_dets) > 1:
                        # Too many faces - ask user to adjust
                        pass

                    # --- Draw modern enrollment UI ---
                    box_color = (0, 255, 0) if face_ok else (0, 165, 255)
                    if face_bbox:
                        label = f"{pose.upper()}  {'OK' if face_ok else 'ADJUST'}"
                        DisplayManager.draw_detection(
                            frame, face_bbox, label, color=box_color
                        )
                        display_manager.draw_enrollment_circle(
                            frame, face_bbox, centered=face_centered
                        )
                        display_manager.draw_face_landmarks(frame, face_bbox)

                    display_manager.draw_enrollment_guide(
                        frame,
                        pose=pose,
                        samples_collected=pose_collected,
                        samples_target=samples_per_pose,
                        face_centered=face_centered,
                        face_bbox=face_bbox,
                    )

                    # Overall progress bar text
                    overall_pct = total_collected / total_target if total_target > 0 else 0
                    prog_text = f"Overall: {total_collected}/{total_target} ({overall_pct:.0%})"
                    (tw, th), _ = cv2.getTextSize(
                        prog_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                    )
                    cv2.rectangle(
                        frame,
                        (10, h - 50 - th - 6),
                        (10 + tw + 10, h - 50 + 6),
                        (40, 40, 40),
                        -1,
                    )
                    cv2.putText(
                        frame, prog_text, (15, h - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (230, 230, 230), 2,
                    )

                    # Status line
                    status_text = f"Pose: {pose.upper()}  |  Face: {'CENTERED' if face_centered else 'ADJUST'}"
                    (stw, sth), _ = cv2.getTextSize(
                        status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                    )
                    cv2.rectangle(
                        frame,
                        (10, 10),
                        (10 + stw + 10, 10 + sth + 6),
                        (40, 40, 40),
                        -1,
                    )
                    cv2.putText(
                        frame, status_text, (15, 10 + sth),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1,
                    )

                    window_status = display_manager.display_frames({camera_id: frame})
                    if not any(window_status.values()):
                        console.print("[yellow]Window closed by user.[/yellow]")
                        return

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        console.print("[yellow]Collection interrupted by user.[/yellow]")
                        return

    finally:
        manager.stop_all()
        display_manager.cleanup()
        cv2.destroyAllWindows()

    console.print(
        f"\n[bold green]Collection complete![/bold green] "
        f"{total_collected} images saved to {person_dir}"
    )
    console.print("[dim]Run 'python test.py frs train' to generate embeddings.[/dim]")
