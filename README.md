# Vision_Suite

A comprehensive CPU-only object detection, face detection, and face recognition system for Windows. Built with Python, YOLOv8n, InsightFace (SCRFD + ArcFace), and ONNX Runtime.

## Architecture

### Core Modules

| Module | Description |
|--------|-------------|
| `src/core/config_loader.py` | Pydantic config singleton (single source of truth) |
| `src/core/logger.py` | Dual logging: rotating text + cooldown-throttled CSV |
| `src/core/model_downloader.py` | Auto-download/caching model weights |
| `src/core/tracker.py` | IOU-based multi-object tracker with persistent track IDs |
| `src/core/video_source.py` | Unified video source (live/rtsp/offline) with threaded capture |
| `src/detection/object_detector.py` | YOLOv8n via ONNX Runtime |
| `src/detection/face_detector.py` | InsightFace SCRFD (detection-only mode) |
| `src/detection/face_analysis_engine.py` | Singleton InsightFace engine (buffalo_s pack) |

### Pipeline Flow

```
Video Source → Frame Buffer → Detector → Tracker → Event Logger → Display
```

Each camera runs its own capture thread with a latest-frame-only buffer. A shared model instance is used across all cameras with a round-robin inference loop.

## Installation

**Prerequisites:** Windows 10/11, Python 3.11

```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

On first run, `yolov8n` and `buffalo_s` pack are auto-downloaded to `models/`.

## Configuration

All settings in `config.yaml` (validated by Pydantic). Key sections: `project`, `paths`, `logging`, `models`, `tracking`, `resolution`, `cameras`, `detection`, `data_collection`, `recognition`.

## Usage

```bash
python tests/test.py frs detect    # Live object & face detection
python tests/test.py frs crops     # Save cropped detections (stub)
python tests/test.py frs collect   # Collect face samples (stub)
python tests/test.py frs train     # Build face DB (stub)
python tests/test.py frs recognize # Recognize faces (stub)
python tests/test.py frs system    # Full production pipeline (stub)
```

## Folder Structure

```
Vision_Suite/
├── config.yaml           # Master configuration
├── requirements.txt      # Pinned dependencies
├── src/core/             # Core utilities
├── src/detection/        # Detection modules
├── src/pipelines/        # Pipeline logic
├── models/               # Auto-download model weights (gitignored)
├── data/                 # Faces / embeddings (gitignored)
└── output/               # Logs / crops (gitignored, auto-created)
```

## Dependencies (CPU-optimized)

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | 1.26.4 | Numerical computing |
| opencv-python | 4.10.0.84 | Video/image ops |
| pydantic | 2.9.2 | Config validation |
| torch | 2.4.1 | Deep learning (CPU) |
| onnx / onnxruntime | 1.16.1 / 1.19.2 | ONNX inference |
| ultralytics | 8.3.0 | YOLOv8 utilities |
| insightface | Windows wheel | SCRFD + ArcFace |
| requests / tqdm | 2.32.3 / 4.66.5 | HTTP / progress bars |

## Known Issues / Flaws

1. **Stub pipelines**: `crops`, `collect`, `train`, `recognize`, `system`, `list-persons`, `delete-person` are not fully implemented
2. **No graceful shutdown**: Only Ctrl+C exits, may leave streams open
3. **Blocking model downloader**: UI freezes during large downloads
4. **Config `extra="allow"`**: Silently accepts unexpected YAML fields
5. **Tracker edge case**: May not assign new tracks when all IOU matches fail
6. **RTSP buffer**: Single-slot buffer may drop frames under load
7. **Windows-only pins**: May not work on Linux/macOS without adjustments

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| YOLOv8n inference | ~O(N), N≈400 grid cells | Model ~6 MB |
| SCRFD face detection | ~O(M), M=proposal boxes | InsightFace ~200+ MB |
| IOU tracking | O(T×D), T≤30, D=detections | O(T) active tracks |
| CSV logging | O(1) amortized | Grows with logged frames |
| Frame buffer R/W | O(1) | Single packet per camera |

**Typical runtime**: 1-2 GB RAM (models + buffers), ~30-80 ms/frame on modern CPU.

## Roadmap (from PROGRESS.md)

1. Implement pipeline logic for all stub commands
2. Add graceful shutdown / signal handling
3. Optimize model downloader (async/threaded)
4. Write comprehensive unit tests
5. Dockerize / cross-platform packaging
6. Add EAR/MAR features (placeholder only)

## License

MIT (see `LICENSE` file)