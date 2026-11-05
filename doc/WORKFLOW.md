# Vision_Suite — CLI Workflows

Step-by-step guides for every `frs` command. All commands are invoked through:

```bash
python tests/test.py frs <command> [args]
```

---

## Prerequisites

1. **Python 3.11** with virtual environment activated:
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **config.yaml** present at project root with valid Pydantic schema.

3. On first run, models auto-download to `models/` (YOLOv8n, SCRFD buffalo_s, FaceNet VGGFace2). This may take several minutes depending on connection speed.

---

## Command 1: frs detect

**Purpose:** Live object + face detection across configured camera sources. No saving, no recognition.

```bash
python tests/test.py frs detect
```

**What happens:**
1. Loads `config.yaml`, validates with Pydantic
2. Initializes YOLOv8n (object) + SCRFD (face) detectors
3. Starts capture threads for all configured camera sources
4. For each frame: detect → IOU track → draw bboxes/labels → log events (cooldown-throttled) → display
5. Press `q` or close windows to exit

**Config used:** `input`, `models.object_detection`, `models.face_detection`, `tracking`, `resolution`, `display`, `detection`, `logging`

**Output:** Display windows only. No files written.

---

## Command 2: frs crops

**Purpose:** Detect objects/faces and save cropped images per class with cooldown logic.

```bash
python tests/test.py frs crops
```

**What happens:**
1. Same detection pipeline as `detect`
2. For each tracked object/face: draw bbox, extract crop with padding
3. Save crop to `output/crops/objects/` or `output/crops/faces/`
4. Cooldown-gated: one save per track_id per `crops.cooldown_seconds`
5. Log `OBJECT_CROP_SAVED` / `FACE_CROP_SAVED` to events CSV

**Config used:** All of `detect` plus `crops.*`

**Output:**
- `output/crops/objects/<class>_<track_id>_<timestamp>.jpg`
- `output/crops/faces/face_<track_id>_<timestamp>.jpg`
- `output/logs/events_YYYY-MM-DD.csv`

---

## Command 3: frs collect

**Purpose:** Guided multi-pose face sample collection. All configuration from `config.yaml` → `data_collection` section — no CLI arguments.

```bash
python tests/test.py frs collect
```

**What happens:**
1. Reads `person_name`, `poses`, `samples_per_pose`, quality gates from config
2. Creates `data/faces/<person_name>/raw/`
3. For each pose (straight, left, right, up, down):
   - Displays on-screen pose instruction
   - Detects and tracks the largest face
   - Runs quality gates (brightness, blur, face size)
   - Green box = good quality, red box = poor quality
   - Captures one sample every `capture_interval_seconds` when quality is good
   - Auto-advances to next pose when `samples_per_pose` reached
4. Press `q` to abort early

**Config used:** `data_collection.*`, `input`, `models.face_detection`, `tracking`, `resolution`, `display`

**Output:** `data/faces/<person_name>/raw/<pose>_NNN_<timestamp>.jpg`

**Example config:**
```yaml
data_collection:
  person_name: "Alice_Smith"
  poses: ["straight", "left", "right", "up", "down"]
  samples_per_pose: 15
  capture_interval_seconds: 0.3
  quality_checks:
    min_brightness: 60
    max_brightness: 200
    blur_threshold: 100
    min_face_size_px: 80
```

---

## Command 4: frs train

**Purpose:** Build/update the face recognition database (`face_db.pkl`) from collected face images. Not neural network training — builds a searchable embedding index.

```bash
python tests/test.py frs train
```

**What happens:**
1. Scans `data/faces/<person>/raw/*.jpg` for each person directory
2. For each image: `cv2.imread` → `FaceNet.get_embedding()` → 512-D L2-normalized vector
3. Registers each embedding with `matcher.add_face(person_name, emb)` (multiple embeddings per person supported)
4. Rebuilds FAISS `IndexFlatIP` from the combined database
5. Saves pickle to `models/face_recognition/face_db.pkl`
6. Prints training summary (images processed, faces detected, embeddings extracted, persons registered)

**Config used:** `paths.*`, `models.face_recognition.*`, `recognition.similarity_threshold`

**Output:** `models/face_recognition/face_db.pkl`

**Pickle structure:**
```python
{
    "Alice":   [emb1, emb2, emb3, ...],   # multiple embeddings per person
    "Bob":     [emb1, emb2, ...],
}
```

**Prerequisite:** Run `frs collect` first to gather face images for each person.

---

## Command 5: frs recognize

**Purpose:** Recognize registered faces in live camera feed or offline image/video file.

### Live camera mode (default):
```bash
python tests/test.py frs recognize
```

### Image file mode:
```bash
python tests/test.py frs recognize path/to/photo.jpg
```
Saves annotated output to `output/crops/recognized/photo_recognized.jpg`

### Video file mode:
```bash
python tests/test.py frs recognize path/to/video.mp4
```
Saves annotated output to `output/crops/recognized/video_recognized.mp4`

**What happens:**
1. Initializes YOLOv8n (optional), SCRFD, FaceNet embedder, FAISS matcher
2. Loads `face_db.pkl` and rebuilds FAISS index
3. For each frame: detect faces → IOU track → FaceNet embedding → FAISS match → threshold check → draw bbox + label (+ "unknown" if below threshold)
4. Save recognized crops cooldown-gated to `output/crops/recognized/`
5. Log `FACE_RECOGNIZED` / `FACE_DETECTED` to events CSV
6. Display results (camera mode only; file mode saves output without display)

**Config used:** `recognition.*`, `models.face_recognition.*`, `detection.*`, `tracking.*`, `logging.*`

**Matching logic:**
```
similarity ≥ threshold  →  return person_name + score
similarity < threshold  →  return "unknown" + score  (UNKNOWN fallback)
```

**Output:**
- `output/crops/recognized/<name>_<timestamp>_<camera_id>.jpg` (camera mode)
- `output/crops/recognized/<name>_recognized.<ext>` (file mode)
- `output/logs/events_YYYY-MM-DD.csv`

---

## Command 6: frs system

**Purpose:** Full production pipeline — object detection + face detection + face recognition + logging.

### Live camera mode (default):
```bash
python tests/test.py frs system
```

### Image file mode:
```bash
python tests/test.py frs system path/to/photo.jpg
```

### Video file mode:
```bash
python tests/test.py frs system path/to/video.mp4
```

**What happens:**
1. Same as `frs recognize` but also runs YOLOv8n object detection
2. Logs `OBJECT_DETECTED` events with cooldown
3. Draws object bboxes + labels in class-specific colors
4. Full event logging for both objects and faces

**Config used:** All config sections

**Output:** Same as `frs recognize` plus object detections logged and drawn.

---

## Command 7: frs list-persons

**Purpose:** List all enrolled face profiles across all data stores.

```bash
python tests/test.py frs list-persons
```

**What happens:**
1. Scans `data/faces/` for person directories
2. Scans `data/embeddings/` for person directories
3. Loads `models/face_recognition/face_db.pkl` and lists keys
4. Consolidates all sources and displays a numbered table

**Output:** Console table of enrolled persons with data source tags.

---

## Command 8: frs delete-person

**Purpose:** Delete a specific enrolled person's profile from all data stores.

```bash
python tests/test.py frs delete-person Alice_Smith
```

**What happens:**
1. Deletes `data/faces/Alice_Smith/` directory
2. Deletes `data/embeddings/Alice_Smith/` directory
3. Removes `Alice_Smith` entry from `face_db.pkl` and re-saves
4. Deletes `output/crops/recognized/Alice_Smith_*` crop files
5. Reports all deleted items

**Arguments:**
- `name` (required): Name of the person to delete

---

## Complete End-to-End Example

```bash
# 1. Activate environment
.\venv\Scripts\activate

# 2. Edit config.yaml — set your person_name, camera index, etc.

# 3. Collect face samples for each person
python tests/test.py frs collect
# (Follow on-screen prompts for each pose; green box = good quality)

# 4. Repeat step 3 for each person (edit person_name in config.yaml between runs)

# 5. Build the recognition database
python tests/test.py frs train

# 6. Recognize faces in live camera
python tests/test.py frs recognize

# 7. Or recognize faces in a photo/video file
python tests/test.py frs recognize data/sample_videos/sample.mp4

# 8. Run full production system
python tests/test.py frs system

# 9. List enrolled persons
python tests/test.py frs list-persons

# 10. Delete a person if needed
python tests/test.py frs delete-person Old_Person
```

---

## Exit Signals

All camera-based pipelines (`detect`, `crops`, `collect`, `recognize`, `system`) handle:
- **`q` key** in any OpenCV window → graceful shutdown
- **Window close** (× button) → graceful shutdown
- **Ctrl+C (SIGINT)** → signal handler sets shutdown flag
- **SIGTERM** → same as SIGINT

On shutdown: capture threads stop, display windows close, video capture released.

---

*End of WORKFLOW.md*