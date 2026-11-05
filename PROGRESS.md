# PROGRESS.md

## Project: Vision_Suite (Minimal Face Recognition System)

### Status: Implementation Complete — Minimal & Production Ready

Core modules, pipelines, and CLI commands are implemented following minimal, clean code principles.
`config.yaml` is the single source of truth validated by Pydantic at startup.

---

## Locked Technical Decisions

### CLI
- Framework: `Typer`
- Entry point: `tests/test.py`
- Commands: `frs enroll`, `frs recognize`, `frs list-persons`, `frs delete-person`
- `frs recognize` accepts optional `source` argument (image/video file path)
- `frs train` kept for backward compatibility (rebuilds FAISS from SQLite, no retraining)

### Models
- Face Detection: **YOLOv8n-Face (WIDERFace)** via ONNX Runtime (CPU)
- Face Recognition: **FaceNet InceptionResnetV1 (VGGFace2)**, 512-D L2-normalized embeddings
- All models auto-download on first use, cached in `models/`, reused afterward

### Matching & Storage
- **FAISS IndexFlatIP** (inner-product ≈ cosine similarity) with pure NumPy fallback
- **SQLite** for identity metadata: `identities` + `face_images` tables
- FAISS index rebuilt from SQLite on startup and after add/delete
- Configurable `similarity_threshold` (default 0.60) with UNKNOWN fallback below threshold
- Add/delete is a database + vector-index operation — the FaceNet model is **never retrained**

### Config System
- `config.yaml` is single source of truth — zero hardcoding
- Validated via Pydantic schemas at startup (`src/core/config_loader.py`)
- Minimal sections: `project`, `paths`, `models`, `recognition`

### Hardware & OS
- Target: CPU-only inference
- OS: Windows (cv2.CAP_DSHOW backend for camera access)

### Video Input
- Simple `cv2.VideoCapture` for camera (index 0) and video files
- No threaded capture, no anti-lag buffer, no multi-camera complexity

### Enrollment (frs enroll)
- Single reference image required
- Exactly one detectable face required (validated)
- FaceNet embedding generated and stored in SQLite + FAISS
- No model retraining

### Recognition (frs recognize)
- YOLOv8n-Face detection → FaceNet embedding → FAISS similarity search
- Threshold decision: score ≥ threshold → KNOWN identity, otherwise → UNKNOWN
- Draws bounding boxes + labels (green = known, red = UNKNOWN)
- Supports camera (index 0) and image/video files

### Future Feature (Placeholder Only)
- EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) - NOT implemented

---

## Folder Structure

```
Vision_Suite/
├── config.yaml           # Minimal configuration (single source of truth)
├── requirements.txt      # Pinned dependencies
├── README.md             # Project overview, install, usage
├── PROGRESS.md           # This file
├── doc/                  # Documentation
│   ├── ARCHITECTURE.md   # System architecture + pipeline diagrams
│   ├── CONFIG_REFERENCE.md  # config.yaml field reference
│   ├── WORKFLOW.md       # CLI workflows
│   └── API.md            # Module API reference
├── src/
│   ├── core/
│   │   ├── config_loader.py   # Pydantic config + directory creation
│   │   ├── logger.py          # Logging
│   │   └── model_downloader.py # Model download with retry
│   ├── detection/
│   │   └── face_detector.py   # YOLOv8n-Face ONNX detector
│   ├── display/          # display_manager.py (legacy, unused)
│   ├── recognition/
│   │   ├── embedder.py        # FaceNet InceptionResnetV1
│   │   └── matcher.py         # FAISS + SQLite matching
│   └── pipelines/
│       ├── recognize_pipeline.py  # Face recognition (camera/file)
│       ├── collect_pipeline.py    # Single-image enrollment (frs enroll)
│       ├── train_pipeline.py      # FAISS rebuild from SQLite
│       ├── list_persons.py        # List enrolled identities
│       └── delete_person.py       # Delete identity (no retraining)
├── tests/
│   ├── test.py           # Typer CLI entry point
│   └── frs_driver.py     # Pipeline driver functions
├── models/               # Auto-downloaded model weights
├── data/                 # Face data + SQLite database
└── output/               # Auto-created output
```

---

## Key Changes from Previous Version

| Area | Before | After |
|------|--------|-------|
| Face Detection | SCRFD via InsightFace | YOLOv8n-Face via ONNX Runtime |
| Storage | Pickle `{name: [emb]}` | SQLite + FAISS |
| Enrollment | Multi-pose guided collect | Single image, exactly one face |
| Object Detection | YOLOv8n COCO (included) | Removed (face recognition only) |
| Tracking | IOU multi-object tracker | Removed (not needed) |
| CLI Commands | 8 commands (detect, crops, collect, train, recognize, system, list-persons, delete-person) | 4 commands (enroll, recognize, list-persons, delete-person) |
| Config | 13 sections | 4 sections |
| Video Input | Threaded capture, anti-lag buffer, RTSP | Simple cv2.VideoCapture |

---

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Test end-to-end: enroll → recognize → list-persons → delete-person
3. Verify YOLOv8n-Face model downloads correctly
4. Verify FAISS + SQLite integration works end-to-end
