# PADEL TRAINER – QUICK REFERENCE CHEAT SHEET

## One-Page Overview

**Project:** Computer Vision–Assisted Padel Ball Tracking & Event Detection  
**Tech Stack:** Python 3.10, PyTorch, YOLOv8, OpenCV, Kalman Filtering  
**Deployment:** GPU (RTX 3050 Ti) + Edge (Pi 5 + Hailo-8)  
**Status:** Production-ready (v3, merged contact pipeline)  

---

## Core Components

| Component | File | Purpose | Input | Output |
|-----------|------|---------|-------|--------|
| **Ball Detection** | `detection/ball_detector.py` | YOLO-based detection | Frame | (x, y, conf) |
| **Optical Flow** | `tracking/optical_flow_tracker.py` | Motion-based tracking | Frames, seed | (x, y) |
| **Kalman Filter** | `tracking/kalman_tracker.py` | Trajectory smoothing | Detections | Predicted (x, y, vx, vy) |
| **Contact Engine** | `edge/contact_scoring.py` | Event detection | Trajectory | (frame, type, conf) |
| **Edge Pipeline** | `edge/edge_inference.py` | Lightweight inference | Video | CSV + Video |
| **Full Pipeline** | `main.py` | End-to-end processing | Video | CSV + Video + Metrics |

---

## Models Used

```
models/
├── best_ball.pt              ← Ball detection (PRIMARY) ✅
├── best_ball_finetuned.pt    ← Video 3 fine-tune (NOT DEPLOYED)
├── best_players.pt           ← Player boxes ✅
├── best_court.pt             ← Court keypoints ✅
└── onnx/                     ← Quantized for edge ✅
    ├── best_ball.onnx
    ├── best_court.onnx
    └── best_players.onnx
```

---

## Detection Pipeline (Flow)

```
YOLO Detection (Threshold: 0.01)
    ↓ [Success] → Add to trajectory
    ↓ [Fail]
Optical Flow Tracking (40×40px window)
    ↓ [Success] → Add to trajectory
    ↓ [Fail]
Kalman Prediction (≤30 frame gap)
    ↓ [Success] → Fill prediction
    ↓ [Gap too long] → Break trajectory
```

---

## Contact Detection Rules

| Rule | Condition | Reliability |
|------|-----------|-------------|
| Y-Velocity Flip | `v_y[t-1] > 0 && v_y[t] < 0` | ⭐⭐⭐⭐⭐ (85%) |
| Sharp Angle Change | `angle_change > 30–45°` | ⭐⭐⭐⭐ |
| Velocity Threshold | `\|v\| > 5 px/frame` | ⭐⭐⭐⭐ (noise filter) |
| ML Scorer (optional) | Learned model | ⭐⭐⭐ (confidence) |

---

## Key Thresholds

```python
# Ball Detection
YOLO_CONFIDENCE_THRESHOLD = 0.01         # Very permissive
INFERENCE_SIZE = 640                     # Baseline (1280 for small objects)

# Filtering
PLAYER_OVERLAP_MARGIN = 0.25             # Reject 25% of leg overlap
COURT_BOUNDARY_MARGIN = 100              # ±100px from court edges
TRAJECTORY_SMOOTHING_ALPHA = 0.3         # EMA factor

# Contact Detection
CONTACT_MIN_VELOCITY = 5                 # px/frame
CONTACT_ANGLE_THRESHOLD = 30             # degrees
CONTACT_TYPE_FRAME_TOLERANCE = 2         # Merge tolerance (frames)

# Kalman
MAX_FRAMES_TO_SKIP = 30                  # Prediction gap limit
KALMAN_PROCESS_NOISE = 0.01              # Q matrix
KALMAN_MEASUREMENT_NOISE = 5.0           # R matrix
```

---

## CSV Output Format

### Trajectory CSV
```
frame | x      | y      | velocity | vx    | vy    | detection_source | confidence
------|--------|--------|----------|-------|-------|------------------|------------
0     | 320.5  | 240.2  | 15.3     | 10.2  | 11.1  | "YOLO"          | 0.95
1     | 331.2  | 251.8  | 14.9     | 10.5  | 10.3  | "YOLO"          | 0.93
...
```

### Contact CSV
```
frame | x     | y     | velocity | type      | confidence | source | merge_group
------|-------|-------|----------|-----------|------------|--------|-------------
15    | 401.2 | 380.5 | 45.2     | "ground"  | 0.89       | "clean"| 1
127   | 398.8 | 379.9 | 46.1     | "ground"  | 0.91       | "raw"  | 1
...
```

---

## Commands

### Run Full Pipeline
```bash
cd padel_trainer
python -m src.main --video input_videos/sample.mp4
```

### Run Edge Pipeline
```bash
cd src/edge
python edge_inference.py /path/to/video.mp4 --output-name my_video
```

### Batch Process Videos
```bash
for video in input_videos/*.mp4; do
  python -m src.main --video "$video"
done
```

### View Results
```bash
ls outputs/edge/annotated_videos/       # Vis videos
head outputs/edge/hit_candidates/*.csv  # Contact events
cat outputs/metrics/*.txt               # Performance stats
```

---

## Performance Targets

| Metric | Dev (RTX 3050 Ti) | Edge (Pi 5 + Hailo-8) | CPU-Only |
|--------|-------------------|----------------------|----------|
| **FPS (720p)** | 35–50 | 40–60 | 3–5 |
| **Memory** | 2–3 GB | <500 MB | <1 GB |
| **Power** | 150W | <10W | 5W |
| **Latency** | <30ms | <30ms | 200ms |

---

## Directory Structure

```
padel_trainer/
├── src/
│   ├── main.py                          → Full pipeline
│   ├── detection/ (ball, players, court)
│   ├── tracking/ (Kalman, optical flow)
│   ├── evaluation/ (shot regions)
│   ├── edge/ (Pi-optimized)
│   └── utils/ (config, I/O, viz)
├── models/ (YOLO weights + ONNX)
├── input_videos/ (test MP4s)
├── outputs/edge/
│   ├── detections/ (raw YOLO CSV)
│   ├── clean_csv/ (filtered trajectory)
│   ├── hit_candidates/ (contact events)
│   ├── annotated_videos/ (overlay MP4s)
│   └── snapshots/ (debug frames)
├── training/ (fine-tuning scripts)
├── DOCUMENTATION.md (change log)
├── PROJECT_SUMMARY.md (full reference)
├── AI_CONTEXT_PROMPT.md (AI onboarding)
└── requirements.txt
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Few detections | ↓ `YOLO_CONFIDENCE_THRESHOLD`; check GPU status |
| High false positives | ↑ `CONTACT_MIN_VELOCITY`; enable scorer |
| Slow processing | ↑ `PROCESS_EVERY_N_FRAMES`; reduce video resolution |
| Out of memory | Reduce batch size; process videos separately |
| Poor edge FPS | Use Hailo-8; avoid annotated video output |
| Optical flow die-outs | Expected; Kalman fills gaps; adjust `MAX_FRAMES_TO_SKIP` |

---

## Test Videos

| # | Duration | Fps | Labels | Status |
|---|----------|-----|--------|--------|
| 1 | ~1 min | 25 | ~3 contacts | ✅ |
| 2 | ~1 min | 30 | ~5 contacts | ✅ |
| 3 | ~1 min | 30 | ~6 contacts | ✅ (fine-tune source) |
| 4 | ~1 min | 25 | ~4 contacts | ✅ |
| 5 | ~1 min | 30 | ~7 contacts | ✅ |
| 6 | ~1 min | 25 | 8 (manual) | ✅ (validation set) |

**Last Validation:** Video 6 merged pipeline → 7/8 matched (87.5%)

---

## Key Files to Know

- `src/utils/config.py` — All tunable parameters
- `src/edge/edge_inference.py` — Core inference logic
- `src/edge/contact_scoring.py` — Contact detection rules
- `DOCUMENTATION.md` — Full change log & milestones
- `PROJECT_SUMMARY.md` — Comprehensive reference
- `AI_CONTEXT_PROMPT.md` — AI assistant onboarding

---

## Quick Setup

```bash
cd padel_trainer
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
python -m src.main --video input_videos/sample.mp4
# Check outputs/
```

---

## Development Workflow

1. **Edit Config:** Adjust thresholds in `src/utils/config.py`
2. **Run Pipeline:** `python -m src.main --video test.mp4`
3. **Review CSV:** Compare `outputs/edge/detections/*.csv` vs. `clean_csv/*.csv`
4. **Check Video:** Play `outputs/edge/annotated_videos/*_annotated.mp4`
5. **Validate Contacts:** Cross-check `hit_candidates/*.csv` against labels
6. **Iterate:** Adjust thresholds, re-run, measure precision/recall

---

## Useful Python Snippets

### Load and plot trajectory
```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('outputs/edge/clean_csv/video_trajectory_clean.csv')
plt.plot(df['x'], df['y'], label='trajectory')
plt.gca().invert_yaxis()  # Video coords: origin top-left
plt.show()
```

### Count detections by source
```python
import pandas as pd
df = pd.read_csv('outputs/edge/clean_csv/video_trajectory_clean.csv')
print(df['detection_source'].value_counts())
```

### Filter high-velocity frames
```python
import pandas as pd
df = pd.read_csv('outputs/edge/clean_csv/video_trajectory_clean.csv')
high_speed = df[df['velocity'] > 30]
print(f"High-speed frames: {len(high_speed)}")
```

---

## References

- **Full Docs:** `DOCUMENTATION.md` (detailed milestones, findings, tuning history)
- **Full Summary:** `PROJECT_SUMMARY.md` (comprehensive with section numbers)
- **AI Onboarding:** `AI_CONTEXT_PROMPT.md` (for AI assistants)
- **Sample Paper:** `Papers/Padel_Trainer_Using_Computer_Vision youssed previos paper.pdf` (methodology reference)
- **Ultralytics YOLOv8:** https://docs.ultralytics.com/
- **OpenCV Docs:** https://docs.opencv.org/
- **FilterPy (Kalman):** https://filterpy.readthedocs.io/

---

## Status Summary

✅ **Production Ready**
- YOLO ball detection ✅
- Multi-method fallback ✅
- Contact detection engine ✅
- Merged candidate pipeline ✅
- Full GPU pipeline ✅
- Edge-optimized pipeline ✅
- ONNX export ✅

🔄 **In Progress**
- Contact scorer ML model (awaiting labeled data)
- Hailo-8 compilation (HEF format)
- Thesis background chapter

⏳ **Future**
- Web dashboard
- Multi-ball tracking
- Shot classification
- Player footwork analysis

---

**Updated:** February 2026 | **Author:** Daniel (GUC) | **Version:** v3 Stable
