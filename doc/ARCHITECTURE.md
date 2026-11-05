# Vision_Suite — Architecture & Pipeline Diagrams

This document describes the system architecture, data flow, and pipeline workflows for Vision_Suite.

> **Note**: Face Recognition uses **FaceNet (InceptionResnetV1 / VGGFace2)** + **FAISS IndexFlatIP** (inner-product ≈ cosine similarity) with configurable threshold and UNKNOWN fallback. The old ArcFace/InsightFace recognition path is retained only for face **detection** (SCRFD via `face_analysis_engine.py`).

---

## 1. System Overview

```mermaid
graph TD
    subgraph CLI["CLI Layer (Typer)"]
        A[python tests/test.py] --> B[frs detect]
        A --> C[frs crops]
        A --> D[frs collect]
        A --> E[frs train]
        A --> F[frs recognize]
        A --> G[frs system]
        A --> H[frs list-persons]
        A --> I[frs delete-person]
    end

    subgraph CFG["Configuration Layer"]
        J[config.yaml] --> K[config_loader.py<br/>Pydantic AppConfig<br/>single source of truth]
    end

    subgraph CORE["Core Layer"]
        L[logger.py<br/>system + event logging]
        M[tracker.py<br/>IOU multi-object tracker]
        N[video_source.py<br/>threaded capture + anti-lag buffer]
        O[model_downloader.py<br/>retry + resume downloads]
    end

    subgraph DET["Detection Layer"]
        P[object_detector.py<br/>YOLOv8n ONNX Runtime]
        Q[face_detector.py<br/>SCRFD via InsightFace]
        R[face_analysis_engine.py<br/>singleton buffalo_s pack]
    end

    subgraph REC["Recognition Layer"]
        S[embedder.py<br/>FaceNet InceptionResnetV1<br/>512-D L2-normalized]
        T[matcher.py<br/>FAISS IndexFlatIP + NumPy fallback<br/>pickle {name: [emb, ...]}]
    end

    subgraph DISP["Display Layer"]
        U[display_manager.py<br/>grid / multi-window OpenCV]
    end

    subgraph PIPE["Pipeline Layer"]
        V[detect_pipeline.py]
        W[crops_pipeline.py]
        X[collect_pipeline.py]
        Y[train_pipeline.py]
        Z[recognize_pipeline.py]
        AA[system_pipeline.py]
        AB[list_persons.py]
        AC[delete_person.py]
    end

    subgraph PERSIST["Persistence (auto-created)"]
        AD[models/<packs>]
        AE[data/faces/<person>/raw/]
        AF[data/embeddings/]
        AG[models/face_recognition/face_db.pkl]
        AH[output/logs/*.csv + *.log]
        AI[output/crops/<dir>]
    end

    CFG --> CORE
    CORE --> DET
    CORE --> REC
    DET --> PIPE
    REC --> PIPE
    PIPE --> DISP
    PIPE --> PERSIST
    DISP --> PERSIST
```

---

## 2. Pipeline Data Flow (All Commands)

Every pipeline follows the same core loop. The difference between commands is which detection/recognition modules are enabled and what happens with the results.

```mermaid
flowchart LR
    Start([Start CLI command]) --> LoadConfig[Load config.yaml<br/>Pydantic validation]
    LoadConfig --> InitCore[Init core modules<br/>logger, tracker, video_source]
    InitCore --> InitDet[Init detection modules<br/>YOLOv8n + SCRFD]
    InitDet --> InitRec[Init recognition modules<br/>FaceNet embedder + FAISS matcher]
    InitRec --> InitVideo[Start VideoSourceManager<br/>capture threads per camera]
    InitVideo --> Loop{Frame loop}
    Loop --> Read[read_all()<br/>latest frame per camera]
    Read --> DetectObjs[Object detection<br/>YOLOv8n ONNX]
    DetectObjs --> TrackObjs[IOU track objects]
    TrackObjs --> DetectFaces[Face detection<br/>SCRFD]
    DetectFaces --> TrackFaces[IOU track faces]
    TrackFaces --> Embed[FaceNet embedding<br/>160x160 → 512-D]
    Embed --> Match[FAISS match<br/>threshold → name or UNKNOWN]
    Match --> Draw[Draw bboxes + labels]
    Draw --> Log[Event logger<br/>CSV + cooldown throttle]
    Log --> SaveCrop[Save face/object crops<br/>cooldown-gated]
    SaveCrop --> Display[DisplayManager<br/>grid or multi-window]
    Display --> Check{'q' pressed?}
    Check -- Yes --> Stop[Stop all threads<br/>cleanup]
    Check -- No --> Loop
```

---

## 3. Multi-Camera Capture Architecture (Anti-Lag RTSP)

Each camera source gets its own capture thread with a single-slot latest-frame buffer. This is critical for RTSP streams where stale frames cause increasing latency.

```mermaid
flowchart TD
    CFG[config.yaml input section] --> VSM[VideoSourceManager]
    VSM --> VS1[VideoSource: cam_0<br/>capture thread]
    VSM --> VS2[VideoSource: cam_1<br/>capture thread]
    VSM --> VS3[VideoSource: rtsp_0<br/>capture thread]

    VS1 --> Buf1[FrameBuffer<br/>single-slot<br/>latest only]
    VS2 --> Buf2[FrameBuffer<br/>single-slot<br/>latest only]
    VS3 --> Buf3[FrameBuffer<br/>single-slot<br/>latest only]

    Buf1 --> Read1[read() → FramePacket]
    Buf2 --> Read2[read() → FramePacket]
    Buf3 --> Read3[read() → FramePacket]

    Read1 --> FrameLoop[Pipeline frame loop<br/>read_all() merges all]
    Read2 --> FrameLoop
    Read3 --> FrameLoop

    subgraph CaptureThread["_CaptureThread (per camera)"]
        Connect[Connect / reconnect]
        ReadFrame[cap.read() frame]
        Resize[resize display + inference]
        PutBuf[buffer.put(packet)]
        Sleep[time.sleep(0.005)]
    end

    VS1 --> CaptureThread1[CaptureThread]
    VS2 --> CaptureThread2[CaptureThread]
    VS3 --> CaptureThread3[CaptureThread]

    CaptureThread1 --> Connect
    CaptureThread2 --> Connect
    CaptureThread3 --> Connect

    style Buf1 fill:#e1f5e1,stroke:#333
    style Buf2 fill:#e1f5e1,stroke:#333
    style Buf3 fill:#e1f5e1,stroke:#333
```

**Key properties:**
| Property | Value |
|----------|-------|
| Capture threads | 1 per camera source |
| Frame buffer | Single-slot (latest frame only, stale discarded) |
| Inference resolution | Configurable, independent of display (default 640×480) |
| Display resolution | Configurable per camera (default 1280×720) |
| Transport | TCP/UDP per camera (RTSP only) |
| Frame skip | Configurable per camera |

---

## 4. Face Recognition Pipeline (FaceNet + FAISS)

The recognize and system pipelines run face recognition on every tracked face:

```mermaid
flowchart TD
    Frame[Video frame / image file] --> Detect[SCRFD face detection<br/>bounding boxes + confidence]
    Detect --> Track[IOU face tracker<br/>persistent track_id]
    Track --> Crop[Extract face crop<br/>from display-resolution frame]
    Crop --> Preprocess[FaceNet preprocess<br/>160×160 RGB<br/>(x − 127.5) / 128.0]
    Preprocess --> Inference[InceptionResnetV1<br/>VGGFace2 weights<br/>512-D vector]
    Inference --> L2Norm[L2-normalize embedding]
    L2Norm --> FAISS{FAISS IndexFlatIP<br/>inner-product search}
    FAISS -- Best score --> Threshold{similarity ≥ threshold?}
    Threshold -- Yes --> MatchName[Return person_name + score]
    Threshold -- No --> Unknown[Return "unknown" + score]
    MatchName --> DrawBox[Draw green bbox + label]
    Unknown --> DrawBox2[Draw red bbox + "Unknown" label]
    DrawBox --> SaveCrop[Save recognized crop<br/>cooldown-gated]
    DrawBox2 --> SaveCrop2[Save unknown crop<br/>if configured]
    SaveCrop --> LogEvent[Log FACE_RECOGNIZED / FACE_DETECTED<br/>to events CSV]
```

**Database structure (pickle):**
```python
# models/face_recognition/face_db.pkl
{
    "Alice":   [emb1, emb2, emb3, ...],   # multiple embeddings per person
    "Bob":     [emb1, emb2, ...],
    "Charlie": [emb1, ...]
}
```

**Matching algorithm:**
1. Load pickle → rebuild FAISS `IndexFlatIP` on init (`_rebuild_index()`)
2. Query embedding → L2-normalize → FAISS search (k=1)
3. If best similarity ≥ `config.recognition.similarity_threshold` → return person name
4. Otherwise → return `config.recognition.unknown_label` (default `"unknown"`)
5. NumPy fallback if FAISS is not installed (brute-force cosine over all embeddings)

---

## 5. Data Collection Flow (frs collect)

Guided multi-pose collection with quality gates:

```mermaid
flowchart TD
    Start[Start collect for person<br/>from config.yaml data_collection] --> InitDir[Create data/faces/<person>/raw/]
    InitDir --> PoseLoop{More poses?}
    PoseLoop --> ShowPose[Display pose instruction<br/>"straight" / "left" / "right" / "up" / "down"]
    ShowPose --> Detect[SCRFD detect face]
    Detect --> Track[IOU track face]
    Track --> Crop[Extract face crop]
    Crop --> Quality{Quality gates pass?<br/>brightness / blur / size}
    Quality -- No --> RedBox[Red bbox + quality text]
    RedBox --> PoseLoop
    Quality -- Yes --> GreenBox[Green bbox + quality text]
    GreenBox --> Interval{Capture interval elapsed?}
    Interval -- No --> PoseLoop
    Interval -- Yes --> Save[Save raw face image<br/>pose_NNN_timestamp.jpg]
    Save --> Count{pose complete?<br/>samples_per_pose reached}
    Count -- No --> PoseLoop
    Count -- Yes --> NextPose[Next pose index]
    NextPose --> PoseLoop
    PoseLoop -- No --> Done[Collection complete<br/>data/faces/<person>/raw/]
```

**Quality gates (from `config.yaml` → `data_collection.quality_checks`):**
| Gate | Default | Purpose |
|------|---------|---------|
| min_brightness | 60 | Reject underexposed faces |
| max_brightness | 200 | Reject overexposed faces |
| blur_threshold | 100 | Laplacian variance ≥ threshold (reject blurry) |
| min_face_size_px | 80 | Minimum face width/height in pixels |

---

## 6. Training Flow (frs train)

```mermaid
flowchart TD
    Start[Start train] --> Scan[data/faces/<person>/raw/*.jpg]
    Scan --> ForEach{Next person?}
    ForEach --> LoadImg[cv2.imread per image]
    LoadImg --> Embed[FaceNet.get_embedding()<br/>512-D normalized]
    Embed --> Register[matcher.add_face(person, emb)<br/>append to face_db + FAISS index]
    Register --> ForEach
    ForEach -- No --> Save[matcher.save_face_db()<br/>pickle face_db.pkl]
    Save --> Report[Print training summary<br/>images, faces, embeddings, persons]
```

---

## 7. CLI Command → Pipeline Routing

| Command | Pipeline module | Detection | Recognition | Output |
|---------|----------------|-----------|-------------|--------|
| `frs detect` | `detect_pipeline.py` | YOLOv8n + SCRFD | No | Display only |
| `frs crops` | `crops_pipeline.py` | YOLOv8n + SCRFD | No | Cropped images + events CSV |
| `frs collect` | `collect_pipeline.py` | SCRFD + FaceNet | No | Raw face samples per pose |
| `frs train` | `train_pipeline.py` | No | FaceNet + FAISS | `face_db.pkl` |
| `frs recognize` | `recognize_pipeline.py` | YOLOv8n + SCRFD | FaceNet + FAISS | Annotated output + crops + events CSV |
| `frs system` | `system_pipeline.py` | YOLOv8n + SCRFD | FaceNet + FAISS | Full production: objects + faces + recognition |
| `frs list-persons` | `list_persons.py` | No | No | Console table |
| `frs delete-person` | `delete_person.py` | No | No | Deletes from all stores |

---

## 8. File Recognition Mode (`frs recognize <file>`, `frs system <file>`)

When a `source` path is provided, the pipeline bypasses `VideoSourceManager` and processes an offline image or video file:

```
Image mode:  cv2.imread → single _process_frame() → save annotated_<name>.jpg
Video mode:  cv2.VideoCapture → loop _process_frame() per frame → save annotated_<name>.mp4
```

Both modes:
- Use the same `_process_frame()` shared by the camera pipeline
- Run SCRFD detection → FaceNet embedding → FAISS match → draw → log → save crops
- Set tracker `min_hits=1` so single-frame detections are confirmed immediately
- Write output to `output/crops/recognized/`

---

## 9. Module Dependency Graph

```mermaid
graph LR
    test.py --> frs_driver.py
    frs_driver.py --> detect_pipeline.py
    frs_driver.py --> crops_pipeline.py
    frs_driver.py --> collect_pipeline.py
    frs_driver.py --> train_pipeline.py
    frs_driver.py --> recognize_pipeline.py
    frs_driver.py --> system_pipeline.py
    frs_driver.py --> list_persons.py
    frs_driver.py --> delete_person.py

    detect_pipeline.py --> config_loader.py
    detect_pipeline.py --> logger.py
    detect_pipeline.py --> tracker.py
    detect_pipeline.py --> video_source.py
    detect_pipeline.py --> object_detector.py
    detect_pipeline.py --> face_detector.py
    detect_pipeline.py --> display_manager.py

    recognize_pipeline.py --> config_loader.py
    recognize_pipeline.py --> logger.py
    recognize_pipeline.py --> tracker.py
    recognize_pipeline.py --> video_source.py
    recognize_pipeline.py --> object_detector.py
    recognize_pipeline.py --> face_detector.py
    recognize_pipeline.py --> embedder.py
    recognize_pipeline.py --> matcher.py
    recognize_pipeline.py --> display_manager.py

    object_detector.py --> model_downloader.py
    object_detector.py --> onnxruntime
    object_detector.py --> ultralytics

    face_detector.py --> face_analysis_engine.py
    face_analysis_engine.py --> insightface

    embedder.py --> facenet_pytorch
    embedder.py --> torch

    matcher.py --> faiss (optional)
    matcher.py --> numpy

    config_loader.py --> pydantic
    config_loader.py --> PyYAML
```

---

## 10. Data Flow Summary

```
Camera/File Frame
    │
    ▼
┌─────────────────┐
│  SCRFD Detect   │ ───┐
│  (face boxes)   │    │
└─────────────────┘    │
    │                   │
    ▼                   │
┌─────────────────┐    │
│   IOU Tracker   │    │
│ (track_id)      │    │
└─────────────────┘    │
    │                   │
    ▼                   │
┌─────────────────┐    │
│  FaceNet Embed  │    │
│  (512-D)        │    │
└─────────────────┘    │
    │                   │
    ▼                   │
┌─────────────────┐    │
│   FAISS Match   │◄───┘
│ (cosine sim)    │
│ threshold →     │
│ name / UNKNOWN  │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Draw + Log     │
│  Save crops     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│   Display       │
└─────────────────┘
```

---

*End of ARCHITECTURE.md*