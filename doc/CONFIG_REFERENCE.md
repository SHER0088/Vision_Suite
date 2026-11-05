# Vision_Suite — Configuration Reference

`config.yaml` is the single source of truth for all system values. It is loaded and validated by `src/core/config_loader.py` using Pydantic schemas at startup. No value is hardcoded in Python.

## Top-Level Sections

| Section | Purpose |
|---------|---------|
| `project` | Project metadata (name, version, device) |
| `input` | Video input source (webcam / video file / RTSP) |
| `paths` | Directory and file paths for models, data, output, logs |
| `logging` | System log + event CSV configuration |
| `models` | Model configs: object detection, face detection, face recognition |
| `tracking` | IOU tracker parameters |
| `resolution` | Display vs inference resolution |
| `display` | Display mode (grid / multi-window) |
| `detection` | Enable/disable object and face detection |
| `crops` | Crop saving parameters (used by `frs crops`) |
| `data_collection` | Face collection parameters (used by `frs collect`) |
| `recognition` | Recognition parameters (used by `frs recognize` / `frs system`) |
| `system` | Future feature placeholders (EAR/MAR) |

---

## project

```yaml
project:
  name: "Vision_Suite"
  version: "0.1.0"
  device: "cpu"          # locked for now, future-proofed for "cuda"
  description: "Vision Suite Pipeline"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"Vision_Suite"` | Project name |
| `version` | string | `"0.1.0"` | Version string |
| `device` | string | `"cpu"` | Inference device (`"cpu"` or `"cuda"`) |
| `description` | string | `"Vision Suite Pipeline"` | Human-readable description |

---

## input

Single input type with multiple sources of that type. Only the section matching `type` is read.

```yaml
input:
  type: "video"                    # video | web_cam | live

  camera_indices: [0]              # used when type == "web_cam"
  video_paths:                     # used when type == "video"
    - "data/sample_videos/sample.mp4"
  rtsp_urls:                       # used when type == "live"
    - "rtsp://user:pass@192.168.1.10:554/stream1"

  frame_skip: 0                    # skip N frames between inferences
  reconnect_delay_seconds: 5       # delay before reconnect attempt
  transport: "tcp"                 # tcp | udp  (RTSP only)

  display_resolution: null         # null = inherit global default
  inference_resolution: null       # null = inherit global default
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | `"web_cam"` | `"web_cam"`, `"video"`, or `"live"` |
| `camera_indices` | list[int] | `[0]` | Device indices for webcam mode |
| `video_paths` | list[string] | `[]` | Offline video file paths |
| `rtsp_urls` | list[string] | `[]` | RTSP stream URLs |
| `frame_skip` | int | `0` | Process every (N+1)th frame |
| `reconnect_delay_seconds` | int | `5` | Seconds to wait before reconnect |
| `transport` | string | `"tcp"` | `"tcp"` or `"udp"` (RTSP only) |
| `display_resolution` | ResolutionItem \| null | `null` | Override display W×H per source |
| `inference_resolution` | ResolutionItem \| null | `null` | Override inference W×H per source |

---

## paths

All paths are relative to the project root. Directories are auto-created at runtime.

```yaml
paths:
  models_dir: "models"
  data_dir: "data"
  faces_dir: "data/faces"
  embeddings_dir: "data/embeddings"
  face_db_path: "models/face_recognition/face_db.pkl"
  output_dir: "output"
  logs_dir: "output/logs"
  crops_dir: "output/crops"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `models_dir` | string | `"models"` | Model weights cache root |
| `data_dir` | string | `"data"` | Collected face data root |
| `faces_dir` | string | `"data/faces"` | Raw face images per person |
| `embeddings_dir` | string | `"data/embeddings"` | Per-person embedding files |
| `face_db_path` | string | `"models/face_recognition/face_db.pkl"` | Pickle face database |
| `output_dir` | string | `"output"` | All output root |
| `logs_dir` | string | `"output/logs"` | System + event logs |
| `crops_dir` | string | `"output/crops"` | Cropped images root |

---

## logging

Two independent log streams: rotating text log + cooldown-throttled CSV event log.

```yaml
logging:
  system_log_level: "INFO"          # DEBUG, INFO, WARNING, ERROR
  system_log_file: "output/logs/system.log"
  system_log_max_bytes: 5242880     # 5 MB per log file before rotation
  system_log_backup_count: 5
  event_log_cooldown_seconds: 10    # min seconds before re-logging same track_id
  log_cooldown_seconds: 5
  events_csv_file: "output/logs/events.csv"
  event_csv_columns:
    - "timestamp"
    - "event_type"
    - "camera_id"
    - "class_or_name"
    - "confidence"
    - "bbox_x1"
    - "bbox_y1"
    - "bbox_x2"
    - "bbox_y2"
    - "crop_path"
    - "track_id"
  rotation: "1 day"
  retention: "7 days"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `system_log_level` | string | `"INFO"` | Python logging level |
| `system_log_max_bytes` | int | `5242880` | Rotate after this many bytes |
| `system_log_backup_count` | int | `5` | Number of rotated backups to keep |
| `event_log_cooldown_seconds` | float | `10.0` | Throttle duplicate CSV rows |
| `event_csv_columns` | list[string] | see above | CSV column names |

---

## models

```yaml
models:
  object_detection:
    name: "yolov8n"
    weights_url: "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    weights_filename: "yolov8n.pt"
    onnx_filename: "yolov8n.onnx"
    class_names_path: "configs/coco_classes.yaml"
    input_size: [640, 640]
    confidence_threshold: 0.45
    iou_threshold: 0.45
    classes_filter: null

  face_detection:
    name: "scrfd"
    provider: "insightface"
    pack_name: "buffalo_s"
    confidence_threshold: 0.5
    input_size: [640, 640]
    det_size: [640, 640]

  face_recognition:
    name: "facenet"
    provider: "facenet_pytorch"
    model_name: "facenet-inceptionresnetv1-vggface2"
    pack_name: "buffalo_s"
    similarity_threshold: 0.60
    embedding_size: 512
    metric: "cosine"

  onnx_providers:
    - "CPUExecutionProvider"
```

### models.object_detection

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"yolov8n"` | Model identifier |
| `weights_url` | string | GitHub release URL | Download URL for `.pt` weights |
| `weights_filename` | string | `"yolov8n.pt"` | Cached weights filename |
| `onnx_filename` | string | `"yolov8n.onnx"` | Exported ONNX filename |
| `class_names_path` | string | `"configs/coco_classes.yaml"` | COCO class names |
| `input_size` | int \| list | `640` | Inference input size |
| `confidence_threshold` | float | `0.45` | Detection confidence floor |
| `iou_threshold` | float | `0.45` | NMS IOU threshold |
| `classes_filter` | list \| null | `null` | `null` = all COCO classes |

### models.face_detection

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"scrfd"` | Detector identifier |
| `provider` | string | `"insightface"` | Backend provider |
| `pack_name` | string | `"buffalo_s"` | InsightFace model pack |
| `confidence_threshold` | float | `0.5` | Face detection score floor |
| `input_size` | int \| list | `640` | Detection input size |
| `det_size` | list[int] | `[640, 640]` | SCRFD det_size tuple |

### models.face_recognition

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | `"facenet"` | Recognizer identifier |
| `provider` | string | `"facenet_pytorch"` | Backend provider |
| `model_name` | string | `"facenet-inceptionresnetv1-vggface2"` | FaceNet model variant |
| `pack_name` | string | `"buffalo_s"` | (unused, kept for schema compatibility) |
| `similarity_threshold` | float | `0.60` | Cosine similarity floor for match |
| `embedding_size` | int | `512` | Embedding dimension |
| `metric` | string | `"cosine"` | Similarity metric (`"cosine"` or `"inner_product"`) |

> **Note**: `similarity_threshold` in `config.yaml` overrides the Pydantic default of `0.60`. The YAML value shown here (`0.60`) is the current active value. The old README mentioned `0.45` — that was from the ArcFace era and is no longer applicable.

---

## tracking

IOU-based multi-object tracker with persistent track IDs.

```yaml
tracking:
  type: "ioU"                        # options: "iou", "bytetrack"
  iou_threshold: 0.3                 # minimum IOU for track-detection match
  max_age: 30                        # frames to keep a lost track alive
  min_hits: 3                        # frames before a track is confirmed
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `type` | string | `"iou"` | Tracker algorithm |
| `iou_threshold` | float | `0.3` | Match threshold |
| `max_age` | int | `30` | Frames before track is purged |
| `min_hits` | int | `3` | Frames before track is confirmed |

> For single-image file recognition, pipelines temporarily set `min_hits=1` so detections are confirmed immediately.

---

## resolution

Decoupled display (UI clarity) vs inference (CPU performance) resolution.

```yaml
resolution:
  display:
    width: 1280
    height: 720
  inference:
    width: 640
    height: 480
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `display.width` | int | `1280` | Display window width |
| `display.height` | int | `720` | Display window height |
| `inference.width` | int | `640` | Inference frame width |
| `inference.height` | int | `480` | Inference frame height |

Bounding boxes are scaled from inference resolution back to display resolution before drawing.

---

## display

```yaml
display:
  mode: "grid"                       # options: "grid", "multi_window"
  grid_columns: 2
  window_name_prefix: "Vision_Suite"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `mode` | string | `"grid"` | `"grid"` = tiled single window, `"multi_window"` = one window per camera |
| `grid_columns` | int | `2` | Columns in grid mode |
| `window_name_prefix` | string | `"Vision_Suite"` | Window title prefix |

---

## detection

```yaml
detection:
  enable_object_detection: true
  enable_face_detection: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_object_detection` | bool | `true` | Run YOLOv8n object detection |
| `enable_face_detection` | bool | `true` | Run SCRFD face detection |

---

## crops

Used by `frs crops` and referenced by recognize/system pipelines for crop output directory.

```yaml
crops:
  save_object_crops: true
  save_face_crops: true
  crop_padding: 10
  min_crop_size: 40
  cooldown_seconds: 5
  object_crop_dir: "output/crops/objects"
  face_crop_dir: "output/crops/faces"
  unknown_crop_dir: "output/crops/unknown"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `save_object_crops` | bool | `true` | Save object detection crops |
| `save_face_crops` | bool | `true` | Save face detection crops |
| `crop_padding` | int | `10` | Pixel padding around crop box |
| `min_crop_size` | int | `40` | Minimum crop dimension in pixels |
| `cooldown_seconds` | float | `5.0` | Seconds between saves per track |
| `object_crop_dir` | string | `"output/crops/objects"` | Object crop output dir |
| `face_crop_dir` | string | `"output/crops/faces"` | Face crop output dir |
| `unknown_crop_dir` | string | `"output/crops/unknown"` | Unknown face crop dir |

---

## data_collection

Used by `frs collect`. All values come from config.yaml — no CLI arguments.

```yaml
data_collection:
  person_name: "John_Doe"
  overwrite_if_exists: true
  details:
    employee_id: "EMP1023"
    department: "Security"
  poses:
    - "straight"
    - "left"
    - "right"
    - "up"
    - "down"
  samples_per_pose: 15
  capture_interval_seconds: 0.3
  quality_checks:
    min_brightness: 60
    max_brightness: 200
    blur_threshold: 100
    min_face_size_px: 80
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `person_name` | string | `"John_Doe"` | Person being enrolled |
| `overwrite_if_exists` | bool | `true` | Replace existing data for this person |
| `details` | dict | `{}` | Free-form metadata (employee_id, department, etc.) |
| `poses` | list[string] | 5 poses | Guided pose sequence |
| `samples_per_pose` | int | `15` | Images to capture per pose |
| `capture_interval_seconds` | float | `0.3` | Seconds between captures |
| `quality_checks.min_brightness` | float | `60` | Minimum mean brightness |
| `quality_checks.max_brightness` | float | `200` | Maximum mean brightness |
| `quality_checks.blur_threshold` | float | `100` | Laplacian variance floor |
| `quality_checks.min_face_size_px` | int | `80` | Minimum face dimension |

---

## recognition

Used by `frs recognize` and `frs system`.

```yaml
recognition:
  similarity_threshold: 0.45
  unknown_label: "unknown"
  save_crops: true
  cooldown_seconds: 5
  recognize_crop_dir: "output/crops/recognized"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `similarity_threshold` | float | `0.60` | Cosine similarity floor for a match |
| `unknown_label` | string | `"unknown"` | Label when below threshold |
| `save_crops` | bool | `true` | Save recognized face crops |
| `cooldown_seconds` | float | `5.0` | Seconds between saves per track |
| `recognize_crop_dir` | string | `"output/crops/recognized"` | Recognized crop output dir |

> The YAML value `0.45` overrides the Pydantic default `0.60`. Lower = more permissive matching (more false positives), higher = stricter (more UNKNOWN results).

---

## system

Future feature placeholders.

```yaml
system:
  enable_ear_mar: false              # placeholder only, logs "not implemented yet"
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enable_ear_mar` | bool | `false` | EAR/MAR feature flag (not implemented) |

---

## Folder Structure (runtime, auto-created)

```
Vision_Suite/
├── config.yaml              # master configuration
├── configs/
│   └── coco_classes.yaml    # COCO 80-class names
├── requirements.txt         # pinned dependencies
├── doc/                     # documentation (this folder)
├── src/
│   ├── core/                # config_loader, logger, model_downloader, tracker, video_source
│   ├── detection/           # object_detector, face_detector, face_analysis_engine
│   ├── display/             # display_manager
│   ├── pipelines/           # detect, crops, collect, train, recognize, system, list-persons, delete-person
│   └── recognition/         # embedder (FaceNet), matcher (FAISS)
├── models/                  # gitignored, auto-downloaded
│   ├── object_detection/
│   ├── face_detection/
│   └── face_recognition/
├── data/                    # gitignored
│   ├── faces/<person>/raw/
│   └── embeddings/
└── output/                  # gitignored, auto-created
    ├── logs/
    └── crops/
```

---

*End of CONFIG_REFERENCE.md*