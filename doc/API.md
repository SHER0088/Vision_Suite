# Vision_Suite — API Reference

Module-by-module reference for the Vision_Suite source code. All modules are under `src/`.

---

## Core Layer

### `src/core/config_loader.py`

**`AppConfig`** — Root Pydantic model. Single source of truth. All sections validated at load time.

```python
class AppConfig(BaseConfigModel):
    project: ProjectConfig
    paths: PathsConfig
    logging: LoggingConfig
    models: ModelsConfig
    tracking: TrackingConfig
    resolution: ResolutionConfig
    display: DisplayConfig
    input: InputSourceConfig
    detection: DetectionConfig
    crops: CropsConfig
    data_collection: DataCollectionConfig
    recognition: RecognitionConfig
    system: SystemConfig
```

**`find_config_path(explicit_path=None) → Path`**
- Resolves `config.yaml` from explicit path, CWD, or by walking up parent directories.
- Raises `FileNotFoundError` if not found within 5 parent levels.

**`load_config(config_path=None) → AppConfig`**
- Loads and validates YAML into `AppConfig`. Cached as module singleton.

**`get_config() → AppConfig`**
- Returns cached config, loading default if not yet loaded.

**`ensure_runtime_directories(config)`**
- Creates `models/*`, `data/faces`, `data/embeddings`, `output/logs`, `output/crops/*` directories.

---

### `src/core/logger.py`

**`setup_system_logger(config) → logging.Logger`**
- Configures the `vision_suite` root logger with rotating file handler + console handler.
- Safe to call multiple times (no duplicate handlers).
- Log file: `output/logs/system_YYYY-MM-DD.log`, rotates at `system_log_max_bytes`.

**`get_logger(name) → logging.Logger`**
- Returns a child logger under `vision_suite.<name>`. Use in every module.

**`class EventLogger`**
- Tabular CSV event logger with cooldown-based throttling per `(camera_id, track_id, event_type)`.
- **`log_event(event_type, camera_id, class_or_name, confidence, bbox, crop_path="", track_id=-1, force=False) → bool`**
  - Writes one CSV row if not on cooldown. Returns `True` if written, `False` if skipped.
  - `force=True` bypasses cooldown.

**`get_event_logger(config) → EventLogger`**
- Returns cached singleton `EventLogger`.

---

### `src/core/model_downloader.py`

**`ensure_file(local_path, url, description="model file") → Path`**
- Downloads a file from `url` to `local_path` if missing or empty.
- Features: retry with exponential backoff (3 attempts), 300s timeout, partial download cleanup, atomic rename on success, progress bar via `tqdm`.
- Raises `RuntimeError` if all attempts fail.

---

### `src/core/tracker.py`

**`compute_iou(boxA, boxB) → float`**
- Intersection over Union for two `(x1, y1, x2, y2)` bboxes.

**`class TrackedItem`**
- Fields: `track_id`, `bbox`, `class_name`, `confidence`, `embedding`, `age`, `time_since_update`, `hits`, `last_logged_timestamp`, `last_cropped_timestamp`, `assigned_person_name`.

**`class Tracker`**
- IOU-based multi-object tracker.
- **`update(detections) → List[TrackedItem]`**
  - `detections`: list of `(bbox, confidence, class_name, embedding)` tuples.
  - Returns confirmed active tracks (`time_since_update == 0` and `hits >= min_hits`).
  - Greedy IOU matching, only matches same class (face-to-face, object-to-object).
  - Purges tracks older than `max_age` frames.

---

### `src/core/video_source.py`

**`class FramePacket`**
- `status`: `"ok"`, `"no_new_frame"`, `"end_of_stream"`, `"disconnected"`
- `display_frame`, `inference_frame`: `np.ndarray` or `None`
- `timestamp`: float

**`class FrameBuffer`**
- Thread-safe single-slot buffer. `put()` replaces, `get()` returns latest.

**`class _CaptureThread(threading.Thread)`**
- Background capture thread per camera source.
- Connects (webcam/RTSP/offline), reads frames, resizes to display + inference resolution, puts into buffer.
- RTSP: supports TCP/UDP transport, `CAP_FFMPEG` backend.
- Webcam: `CAP_DSHOW` (Windows), `CAP_V4L2` (Linux), `CAP_AVFOUNDATION` (macOS).
- Offline: `cv2.VideoCapture(filepath)`, puts `end_of_stream` packet at EOF.

**`class VideoSource`**
- Wrapper around `_CaptureThread`. `start()`, `stop()`, `read()`.

**`class VideoSourceManager`**
- Builds sources from `config.input.active_sources()`.
- **`get_active_sources() → List[VideoSource]`**
- **`read_all() → Dict[VideoSource, FramePacket]`**
- **`stop_all()`**

---

## Detection Layer

### `src/detection/object_detector.py`

**`class Detection(NamedTuple)`**
- `class_id`, `class_name`, `confidence`, `bbox` (x1, y1, x2, y2 in frame coordinates).

**`class ObjectDetector`**
- YOLOv8n → ONNX export (one-time) → ONNX Runtime inference (CPU).
- **`detect(frame_bgr) → List[Detection]`**
  - Letterbox → normalize → ONNX inference → NMS → post-process bboxes back to original resolution.

---

### `src/detection/face_detector.py`

**`class FaceDetection(NamedTuple)`**
- `confidence`, `bbox` (x1, y1, x2, y2), `landmarks` (5-point), `embedding` (None in detection-only mode).

**`class FaceDetector`**
- Thin wrapper over `face_analysis_engine` in detection-only mode (`allowed_modules=["detection"]`).
- **`detect(frame) → List[FaceDetection]`**
  - Filters by `confidence_threshold` from config.

---

### `src/detection/face_analysis_engine.py`

**`get_face_analysis_engine(config, need_recognition=True) → FaceAnalysis`**
- Singleton InsightFace `FaceAnalysis` (buffalo_s pack).
- `need_recognition=False` → loads only SCRFD detector (faster, skips ArcFace/landmark/gender-age forward passes).
- `need_recognition=True` → loads full pack (detection + recognition).
- Auto-reloads if switching from detection-only to full pack within the same process.

---

## Display Layer

### `src/display/display_manager.py`

**`class DisplayManager`**
- Professional multi-camera display with dark theme, FPS counters, status indicators, timestamps.
- Modes: `"grid"` (tiled single window) or `"multi_window"` (one window per camera).
- **`initialize_windows(camera_ids)`** — Create OpenCV windows.
- **`display_frames(frames, detection_counts) → Dict[str, bool]`** — Render frames, return per-camera window visibility.
- **`draw_detection(img, bbox, label, color, thickness)`** — Static helper: draw bbox + label background.
- **`cleanup()`** — Destroy all windows.

**`class_color(class_name) → Tuple[int, int, int]`**
- Consistent BGR color per detection class.

---

## Recognition Layer

### `src/recognition/embedder.py`

**`class FaceEmbedder`**
- FaceNet `InceptionResnetV1(pretrained="vggface2")` — frozen, never retrained.
- **`get_embedding(face_img) → Optional[np.ndarray]`**
  - BGR face crop → 160×160 RGB → `(x − 127.5) / 128.0` → model → L2-normalized 512-D float32.
  - Returns `None` on failure.
- **`get_embeddings(face_imgs) → List[Optional[np.ndarray]]`** — Batch wrapper.

---

### `src/recognition/matcher.py`

**`class FaceMatcher`**
- FAISS `IndexFlatIP` over L2-normalized embeddings (≈ cosine similarity). NumPy fallback if FAISS unavailable.
- Pickle persistence: `{person_name: [embedding, ...]}`.

**`__init__(config)`**
- Loads `face_db.pkl`, migrates old single-vector format, rebuilds FAISS index.

**`match_face(embedding) → Tuple[str, float]`**
- Returns `(person_name, similarity)`. Below threshold → `(unknown_label, score)`.

**`add_face(person_name, embedding)`** — Append embedding to person's list + add to FAISS index.
**`remove_face(person_name) → bool`** — Remove person + rebuild index.
**`save_face_db()`** — Persist pickle to disk.
**`get_database_info() → dict`** — Person count, embedding count, backend, path.

---

## Pipeline Layer

### `src/pipelines/detect_pipeline.py`

**`run_detect_pipeline() → None`**
- Live object + face detection, no saving. Display only.

### `src/pipelines/crops_pipeline.py`

**`run_crops_pipeline() → None`**
- Detect + save cropped images per class with cooldown.

### `src/pipelines/collect_pipeline.py`

**`run_collect_pipeline() → None`**
- Guided multi-pose face collection with quality gates.

### `src/pipelines/train_pipeline.py`

**`run_train_pipeline() → None`**
- Build `face_db.pkl` from collected face images using FaceNet + FAISS.

### `src/pipelines/recognize_pipeline.py`

**`run_recognize_pipeline(source: str | None = None) → None`**
- `source=None`: live camera recognition.
- `source="path/to/file"`: image or video file recognition, saves annotated output.

**Internal: `_process_frame(...)`** — Shared per-frame logic (detect, track, embed, match, draw, log, save crops). Used by both camera and file modes.

### `src/pipelines/system_pipeline.py`

**`run_system_pipeline(source: str | None = None) → None`**
- Full production pipeline (objects + faces + recognition + logging). Same `source` parameter behavior as `recognize_pipeline`.

### `src/pipelines/list_persons.py`

**`run_list_persons() → None`**
- List enrolled persons from faces dir, embeddings dir, and face_db.pkl.

### `src/pipelines/delete_person.py`

**`run_delete_person(person_name: str) → None`**
- Delete person from all data stores.

---

## CLI Entry Points

### `tests/test.py`

Typer CLI application. Commands:

| Command | Function | Arguments |
|---------|----------|-----------|
| `frs detect` | `run_detect_pipeline()` | — |
| `frs crops` | `run_crops_pipeline()` | — |
| `frs collect` | `run_collect_pipeline()` | — |
| `frs train` | `run_train_pipeline()` | — |
| `frs recognize` | `run_recognize_pipeline(source)` | `source` (optional image/video path) |
| `frs system` | `run_system_pipeline(source)` | `source` (optional image/video path) |
| `frs list-persons` | `run_list_persons()` | — |
| `frs delete-person` | `run_delete_person(name)` | `name` (required) |

### `tests/frs_driver.py`

Driver functions that delegate to pipeline modules. Each `run_*_pipeline()` imports and calls the corresponding module function, allowing `test.py` to avoid circular imports.

---

*End of API.md*