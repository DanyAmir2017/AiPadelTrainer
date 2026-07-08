# Padel Trainer

Computer vision pipeline for padel ball detection, trajectory tracking, contact detection, and shot-region analysis. The project is built for local GPU development and edge deployment on Raspberry Pi 5 with a Hailo-8 accelerator.

## What it does

- Detects the ball in match videos with a custom YOLO model
- Tracks trajectories with optical flow and Kalman filtering fallbacks
- Detects contact events such as bounces and sharp direction changes
- Classifies court regions for shot placement analysis
- Exports annotated videos, trajectory CSVs, contact CSVs, and metrics

## Project layout

- `src/main.py` - Full processing pipeline
- `src/detection/` - Ball, player, and court detection
- `src/tracking/` - Optical flow and Kalman tracking
- `src/evaluation/` - Shot and region evaluation
- `src/edge/` - Lightweight edge-oriented pipeline
- `models/` - Trained weights and ONNX exports
- `training/` - Dataset and fine-tuning scripts
- `input_videos/` - Sample and test videos
- `outputs/` - Generated CSVs, metrics, and annotated videos

## Requirements

- Python 3.10+
- PyTorch
- Ultralytics YOLOv8
- OpenCV
- NumPy
- FilterPy
- ONNX Runtime

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Setup

```bash
cd padel_trainer
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run the full pipeline

```bash
python -m src.main --video input_videos/sample.mp4
```

This produces outputs under `outputs/`, including:

- annotated video
- cleaned trajectory CSV
- contact-event CSV
- metrics summary

## Run the edge pipeline

```bash
cd src/edge
python edge_inference.py /path/to/video.mp4 --output-name match_01
```

The edge pipeline is designed for lightweight inference and edge deployment workflows.

## Models

- `models/best_ball.pt` - Primary ball detector
- `models/best_ball_finetuned.pt` - Experimental fine-tuned variant
- `models/best_players.pt` - Player detector
- `models/best_court.pt` - Court keypoint detector
- `models/onnx/` - Exported ONNX models for edge use

## Outputs

- `outputs/edge/clean_csv/` - Cleaned trajectory data
- `outputs/edge/hit_candidates/` - Contact-event candidates
- `outputs/edge/annotated_videos/` - Annotated match videos
- `outputs/metrics/` - Performance summaries

## Notes

- The project supports both GPU development and edge deployment.
- The edge pipeline is intentionally smaller than the full pipeline.
- See `PROJECT_SUMMARY.md` for the full technical reference and `QUICK_REFERENCE.md` for a compact cheat sheet.
