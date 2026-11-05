# Vision_Suite

Minimal, production-ready face recognition system for CPU. Uses **YOLOv8n-Face** for detection, **FaceNet InceptionResnetV1 (VGGFace2)** for embeddings, **FAISS** for vector similarity search, and **SQLite** for identity metadata.

**Core principle:** Face recognition (embedding-based), not face classification. The FaceNet model is never retrained — adding or deleting a person is a database + vector-index operation only.

```
Camera frame -> YOLOv8n-Face detection -> crop/preprocessing -> FaceNet embedding
             -> FAISS similarity search -> Threshold decision -> KNOWN / UNKNOWN
```

## CLI Commands

| Command | Purpose |
|---------|---------|
| `frs enroll <code> <name> <image>` | Enroll one person with exactly one reference photo |
| `frs recognize [source]` | Recognize faces from camera or image/video file |
| `frs list-persons` | List all enrolled identities |
| `frs delete-person <code>` | Delete a person (no model retraining) |

## Configuration

`config.yaml` is the single source of truth (Pydantic-validated at startup):

```yaml
project:
  name: "Vision_Suite"
  version: "1.0.0"
  device: "cpu"

paths:
  models_dir: "models"
  data_dir: "data"
  faces_dir: "data/faces"
  db_path: "data/face_recognition.db"
  face_db_path: "models/face_recognition/face_db.pkl"

models:
  face_detection:
    name: "yolov8n-face"
    weights_url: "https://github.com/lindevs/yolov8-face/releases/download/v1.0/yolov8n-face-lindevs.onnx"
    weights_filename: "yolov8n-face.onnx"
    input_size: 640
    confidence_threshold: 0.5

  face_recognition:
    name: "facenet"
    model_name: "facenet-inceptionresnetv1-vggface2"
    embedding_size: 512
    similarity_threshold: 0.60

recognition:
  unknown_label: "unknown"
```

## Installation

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

Models auto-download on first use (YOLOv8n-Face + FaceNet VGGFace2).

## Usage Examples

```bash
# Enroll a person with one photo
python tests/test.py frs enroll UDAY-001 "Uday" data/faces/uday.jpg

# List enrolled persons
python tests/test.py frs list-persons

# Recognize from camera
python tests/test.py frs recognize

# Recognize from an image file
python tests/test.py frs recognize data/sample_videos/test.jpg

# Recognize from a video file
python tests/test.py frs recognize data/sample_videos/test.mp4

# Delete a person (no retraining)
python tests/test.py frs delete-person UDAY-001
```

Enrollment validates that the image contains exactly one detectable face:
- `No face detected. Please upload another image.`
- `Multiple faces detected. Please upload an image containing only one person.`

Recognition returns `UNKNOWN` when the best similarity score is below the configured threshold.

## Documentation

| Doc | Description |
|-----|-------------|
| `doc/ARCHITECTURE.md` | System architecture + pipeline diagrams |
| `doc/CONFIG_REFERENCE.md` | Complete `config.yaml` field reference |
| `doc/WORKFLOW.md` | CLI workflows for each command |
| `doc/API.md` | Module-by-module API reference |
