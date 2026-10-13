# PROGRESS.md

## Project: Vision_Suite (Object Detection, Face Detection & Face Recognition System (FRS))

### Status: Architecture Finalized — Implementation Not Yet Started

---

## Locked Technical Decisions

### CLI
- Framework: `Typer`
- Entry point: `test.py`
- Commands: `frs detect`, `frs crops`, `frs collect`, `frs train`, `frs recognize`,
  `frs system`, `frs list-persons`, `frs delete-person`
- No validate-config, download-models, or export-logs commands (intentionally excluded)

### Models
- Object Detection: **YOLOv8n (Ultralytics)**, exported/run via **ONNX** Runtime
- Face Detection: **SCRFD** (InsightFace buffalo_s pack)
- Face Recognition: **ArcFace** (InsightFace buffalo_s pack)
- All models auto-download on first use, cached in models/, reused afterward
  (check-if-exists-else-download logic inside loader functions, no separate CLI command)

### Matching
- Cosine similarity, in-memory comparison
- Target scale: 25-30 registered persons (no FAISS needed at this scale)

### Tracking
- IOU-based or ByteTrack tracker assigns persistent track_id per detection
- Used to implement cooldown-based crop saving and event logging
  (prevents saving/logging every single frame for the same person/object)

### Config System
- config.yaml is single source of truth for ALL values (paths, thresholds,
  URLs, camera settings, collection details, etc.) - zero hardcoding anywhere
- Validated via Pydantic schemas at startup (clear errors on bad config)

### Logging
- Two separate log streams:
  1. output/logs/events_YYYY-MM-DD.csv - tabular, columns:
     timestamp, event_type, camera_id, class_or_name, confidence,
     bbox_x1, bbox_y1, bbox_x2, bbox_y2, crop_path, track_id
     - Cooldown-based: logs once per track_id per log_cooldown_seconds, not per-frame
  2. output/logs/system_YYYY-MM-DD.log - plain text, rotating file handler,
     for system events (camera connect/disconnect, model downloads, errors)
- output/ folder auto-created at runtime if missing

### Hardware & OS
- Target: CPU-only inference
- OS: Windows (use cv2.CAP_DSHOW backend for camera access)

### Video Input - Unified VideoSource Abstraction
- Three modes: `offline (video file path)`, `live (camera index)`, `rtsp (url)`
- All three modes feed into the **same downstream pipeline** regardless of source
- RTSP-specific anti-lag architecture (critical requirement):
  - Dedicated capture thread per camera, continuously reads frames
  - Single-slot "latest frame only" buffer (not a growing queue) -
    inference always grabs newest frame, stale frames discarded
  - Configurable transport protocol (TCP/UDP) per camera in config
  - Configurable frame_skip per camera
  - Separate inference_resolution (smaller) vs display resolution (larger)

### Resolution Defaults
- Capture/Display resolution: 1280x720 (configurable)
- Inference resolution: 640x480 (configurable, independent of display)
- Bounding boxes scaled from inference resolution back to display resolution

### Multi-Camera Architecture
- Each camera runs its own capture thread (latest-frame-only buffer)
- ONE shared model instance across all cameras (not duplicated per camera)
  - round-robin inference loop cycles through active camera frames
- Display modes (config-controlled): "grid" (tiled single window) or
  "multi_window" (separate OpenCV window per camera)
- Per-camera independent config: source type, transport, frame_skip,
  inference_resolution

### Data Collection (frs collect)
- NO CLI arguments for person name/details - everything read from
  config.yaml under data_collection section:
  - person_name, overwrite_if_exists, details (employee_id, department, etc.)
  - poses list (straight, left, right, up, down)
  - samples_per_pose
  - quality_checks (min_brightness, max_brightness, blur_threshold, min_face_size_px)
- Guided UI flow: on-screen prompts per pose, only captures frames passing
  quality gates (brightness, blur, face size), live visual feedback (green/red)
- Output: raw images saved to data/faces/<person_name>/raw/,
  embeddings saved to data/embeddings/<person_name>.npy
- If person_name already exists and overwrite_if_exists: true, old data is
  cleared and replaced (logged as a warning)

### Training (frs train)
- Aggregates all data/embeddings/*.npy files into a single database:
  models/face_recognition/face_db.pkl
- NOT training a neural network from scratch - building a searchable
  embedding index (standard, correct approach for 25-30 person scale)

### Recognition (frs recognize)
- Compares live face embeddings (ArcFace) against face_db.pkl using
  cosine similarity
- Draws bbox + name (or "unknown") on screen
- Saves labeled crops: data filename format = name_timestamp_cameraid.jpg
- Logs recognition events to events CSV (cooldown-based, per track_id)

### Future Feature (Placeholder Only)
- EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) - NOT implemented yet
- frs system command has placeholder hooks that log "not implemented yet"
- Full detection + recognition pipeline runs in frs system; EAR/MAR to be
  added in a future phase after detection+recognition is solid

### Folder Structure
```
project-root/
├── config.yaml
├── test.py
├── requirements.txt
├── README.md / ARCHITECTURE.md / CONFIG_REFERENCE.md / WORKFLOW.md / PROGRESS.md
├── src/
│   ├── core/ (video_source.py, tracker.py, logger.py, config_loader.py)
│   ├── detection/ (object_detector.py, face_detector.py)
│   ├── recognition/ (embedder.py, matcher.py)
│   ├── pipelines/ (detect_pipeline.py, crops_pipeline.py, collect_pipeline.py,
│   │               recognize_pipeline.py, system_pipeline.py)
│   └── display/ (display_manager.py)
├── models/   (gitignored, auto-downloaded)
├── data/     (gitignored, faces + embeddings)
└── output/   (gitignored, logs + crops, auto-created at runtime)
```
---

## Next Steps (Start of Next Chat)
1. Write config.yaml with full schema (all sections above)
2. Write src/core/config_loader.py (Pydantic validation)
3. Write src/core/video_source.py (unified offline/live/rtsp abstraction
   with threaded capture + anti-lag buffer)
4. Test video_source.py independently across all 3 input modes before
   building anything else on top of it