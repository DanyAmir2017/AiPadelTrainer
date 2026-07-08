# Padel Trainer — Quick Start & Complete Usage Guide

[![license](https://img.shields.io/badge/license-MIT-blue.svg)]() [![python](https://img.shields.io/badge/python-3.10+-blue)]() [![status](https://img.shields.io/badge/status-experimental-orange)]()

One-line description
--------------------

Padel Trainer detects padel ball contacts in video, provides tools for manual labeling, trains a lightweight windowed contact scorer, and supports edge deployment (Raspberry Pi + Hailo-8).

Why this repo
--------------

- Reproducible pipeline: detection → tracking → candidate extraction → manual labeling → scorer training → scoring.
- Edge-ready: ONNX export and Hailo-8 compilation helpers included.
- Focused on researcher productivity: simple CLIs and a small Tkinter labeler for fast human-in-the-loop labeling.

Table of Contents
-----------------

- Quick start (setup & first run)
- Labeling (manual GUI)
- Train & score (scorer scripts)
- Edge & Hailo (export, compile, Pi workflow)
- Troubleshooting & tips
- Developer notes and layout

**Assumptions:** Windows PowerShell examples shown; equivalent POSIX commands (bash) are included where helpful. Replace paths with your workspace path.

**Assumptions:** Windows PowerShell examples shown; equivalent POSIX commands (bash) are included where helpful. Replace paths with your workspace path.

**Repository layout (important paths)**
- `src/main.py`: Full desktop pipeline (GPU-capable)
- `src/edge/edge_inference.py`: Edge-focused inference runner
- `src/edge/contact_labeler.py`: GUI for manual labeling of candidates
- `scripts/score_windowed_contact_scorer.py`: Score candidate CSVs using a windowed model
- `scripts/train_contact_scorer.py`: Train logistic contact scorer from labeled CSVs
- `training/`: Dataset and training helpers (YOLO training)
- `models/`: PyTorch weights and `models/onnx/` for ONNX exports
- `input_videos/`: Raw videos for processing (e.g., `Padel_video_8.mp4`)
- `outputs/edge/`: Main generated outputs (subfolders: `clean_csv/`, `hit_candidates/`, `labels/`, `annotated_videos/`, `contact_models/`)

## 0) Environment setup (Windows PowerShell)

Open PowerShell and run:

```powershell
cd "C:\path\to\padel_trainer"
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Unix / macOS (bash):

```bash
cd /path/to/padel_trainer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you use GPU training, ensure CUDA and the correct `torch` are installed (see `requirements.txt` comments).

## 1) Run the full desktop pipeline (single video)

Processes video with the full (development) pipeline producing annotated video, cleaned trajectory CSV, hit candidates and metrics.

PowerShell:

```powershell
python -m src.main --video input_videos\Padel_video_8.mp4 --output-name v8_laptop_run
```

Generic:

```bash
python -m src.main --video input_videos/Padel_video_8.mp4 --output-name v8_laptop_run
```

Outputs appear under `outputs/` and `outputs/edge/` (see file tree above).

## 2) Run the edge inference pipeline (lighter, ONNX/CPU or NPU)

Edge runner (from repo root):

PowerShell:

```powershell
python src/edge/edge_inference.py input_videos\Padel_video_8.mp4 --output-name v8_laptop_run --save-video
```

Notes:
- For ONNX-mode use exported ONNX file in `models/onnx/` or the `--onnx` flag if available.
- For Hailo-8 compilation and runtime, follow conversion steps in `convert_to_hailo.py` and the `YOLOV8N_TRAINING_GUIDE.md`.

## 3) Manually label contact candidates (GUI)

Use the `contact_labeler.py` GUI to review each candidate clip and mark `Correct` / `Wrong` and a `Contact Type` (racket, ground, glass, out_of_frame).

PowerShell (launch for Video 8 candidates):

```powershell
python src/edge/contact_labeler.py --video input_videos\Padel_video_8.mp4 --candidates-csv outputs\edge\hit_candidates\v8_laptop_run_hit_candidates.csv
```

Key GUI shortcuts:
- ← / → : prev / next candidate
- Space: play/pause clip preview
- `C` : mark Correct
- `W` : mark Wrong
- `1`..`4`: set Contact Type (Racket/Ground/Glass/Out of frame)
- `S` : Save results

Saved file (default): `outputs/edge/labels/<candidates_stem>_labels.csv` (e.g. `outputs/edge/labels/v8_laptop_run_hit_candidates_labels.csv`).

## 4) Score candidate CSVs with a windowed contact model

If you already have a trained windowed model (JSON) in `outputs/edge/contact_models/`, use:

PowerShell:

```powershell
python .\scripts\score_windowed_contact_scorer.py \
	--model .\outputs\edge\contact_models\windowed_contact_scorer_v2_v5_v8.json \
	--candidates-csv .\outputs\edge\hit_candidates\v8_laptop_run_hit_candidates.csv
```

This writes `v8_laptop_run_hit_candidates_window_scored.csv` in the same folder by default. Options:
- `--clean-csv <path>`: explicitly point to `outputs/edge/clean_csv/*_trajectory_clean.csv` if inference produced a different clean file
- `--context-before/--context-after`: override model window size
- `--keep-rejected`: keep rejected rows in output

## 5) Train the logistic contact scorer from labeled CSVs

Collect your labeled CSV files (e.g. `outputs/edge/labels/v5_after_tune_v2_hit_candidates_labels.csv outputs/edge/labels/v6_after_tune_v3_hit_candidates_labels.csv outputs/edge/labels/v8_laptop_run_hit_candidates_labels.csv`) and run the trainer:

PowerShell:

```powershell
python .\scripts\train_contact_scorer.py \
	--labels outputs\edge\labels\v5_after_tune_v2_hit_candidates_labels.csv outputs\edge\labels\v6_after_tune_v3_hit_candidates_labels.csv outputs\edge\labels\v8_laptop_run_hit_candidates_labels.csv \
	--candidates-dir outputs\edge\hit_candidates \
	--output-model outputs\edge\contact_models\windowed_contact_scorer_v_new.json
```

Training outputs:
- A JSON model with `weights`, `bias`, `feature_mean`, `feature_std`, and `accept_threshold`/`review_threshold` written to the `--output-model` path.
- Training summary printed to console and saved in the JSON under `training_summary`.

Tune training hyperparameters:
- `--epochs` default 1800, `--lr` default 0.05. Example:

```powershell
python .\scripts\train_contact_scorer.py --labels ... --epochs 3000 --lr 0.03
```

## 6) Re-score candidates with manual labels merged (recommended workflow)

Typical workflow after labeling:
1. Run the scorer (section 4) to produce scored CSV.
2. Run `contact_labeler.py` and save labels.
3. Optionally re-run scoring to include model features and merged labels for inspection.

Quick example to merge labels (re-score then inspect):

```powershell
python .\scripts\score_windowed_contact_scorer.py --model .\outputs\edge\contact_models\windowed_contact_scorer_v2_v5_v8.json --candidates-csv .\outputs\edge\hit_candidates\v8_laptop_run_hit_candidates.csv
notepad outputs\edge\hit_candidates\v8_laptop_run_hit_candidates_window_scored.csv
```

## 7) YOLO training (train `yolov8n` for edge)

This repo includes training helpers under `training/` and a dedicated guide `YOLOV8N_TRAINING_GUIDE.md`. Minimal example (requires Ultralytics CLI):

```bash
# From repo root (bash shown for YOLO CLI)
yolo train model=yolov8n.pt data=training/dataset/data.yaml epochs=100 imgsz=640 project=runs/detect

# Export ONNX after training
yolo export model=runs/detect/train/weights/best.pt format=onnx
```

On Windows PowerShell use the same commands in your activated venv.

After export, place the ONNX in `models/onnx/` (example filename: `best_ball_nano.onnx`). For Hailo conversion see `convert_to_hailo.py`.

## 8) Convert/compile for Hailo-8 (optional)

This requires Hailo toolchain (x86_64 Linux for HEF compilation). High-level steps:

1. Export ONNX (`yolo export ... format=onnx`).
2. Run `convert_to_hailo.py` to apply export-time workarounds and produce Hailo-compatible ONNX.
3. Compile ONNX→HEF using `hailomz compile` on an x86_64 Linux host or VM.
4. Transfer HEF to the Pi and run with `hailo run` or the Pi runtime wrapper.

Example conversion command (on the host that has Hailo SDK):

```bash
python convert_to_hailo.py --model training/runs/detect/train/weights/best.pt --output models/onnx/best_ball_nano_hailo.onnx
hailomz compile models/onnx/best_ball_nano_hailo.onnx --output best_ball_nano.hef
scp best_ball_nano.hef pi@raspberrypi.local:~/padel_trainer/models/onnx/
```

### Host → Pi: Concrete Hailo workflow (recommended)

Follow these step-by-step commands to export, compile and run a Hailo-compatible model on a Raspberry Pi 5 with Hailo-8.

1) Export ONNX and apply Hailo conversion helper on x86_64 host:

```bash
# Export ONNX (Ultralytics)
yolo export model=runs/detect/train/weights/best.pt format=onnx opset=11

# Convert to Hailo-compatible ONNX using helper script
python convert_to_hailo.py --model runs/detect/train/weights/best.pt --output models/onnx/best_ball_nano_hailo.onnx
```

2) Compile ONNX → HEF on x86_64 (native or Docker):

```bash
# Using Hailo Docker image (recommended when SDK is not installed locally)
docker pull hailo/hailo-sw-suite:latest
docker run --rm -v $(pwd):/workspace hailo/hailo-sw-suite:latest \
	hailo parser onnx /workspace/models/onnx/best_ball_nano_hailo.onnx
docker run --rm -v $(pwd):/workspace hailo/hailo-sw-suite:latest \
	hailo compiler /workspace/best_ball_nano_hailo.har --hw-arch hailo8 --output /workspace/best_ball_nano_hailo.hef
```

3) Transfer HEF and repo to Pi:

```bash
scp best_ball_nano_hailo.hef pi@raspberrypi.local:~/padel_trainer/models/onnx/
scp -r . pi@raspberrypi.local:~/padel_trainer/   # optional: sync repo to Pi
```

4) On Pi: set up environment and verify Hailo runtime

```bash
ssh pi@raspberrypi.local
cd ~/padel_trainer
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# Install Hailo runtime (may vary by image)
sudo apt update && sudo apt install -y hailo-all
# Confirm device is visible
hailortcli scan
```

5) Run the HEF model (quick test / benchmark):

```bash
hailo run ~/padel_trainer/models/onnx/best_ball_nano_hailo.hef --input <optional_raw_input>
hailortcli benchmark ~/padel_trainer/models/onnx/best_ball_nano_hailo.hef
```

6) Run the project's edge inference using the HEF file (if `edge_inference.py` supports the hailo backend):

```bash
python src/edge/edge_inference.py input_videos/Padel_video_8.mp4 --backend hailo --heffile ~/padel_trainer/models/onnx/best_ball_nano_hailo.hef --save-video
```

Notes:
- Hailo compilation requires x86_64 host; use the Hailo Docker image to avoid native SDK installation.
- If `hailortcli scan` does not detect the device, verify drivers and board connections.
- If `edge_inference.py` lacks `--backend hailo`, use `hailo run` or follow `src/edge/README.md` for the Pi wrapper.


## 9) Annotated video export and collision marker overlay

The pipeline can save annotated MP4s with collision markers. To produce an annotated video explicitly use the `--save-video` or `--output-name` flags when running `edge_inference.py` or `src/main.py`.

Example:

```powershell
python src/edge/edge_inference.py input_videos\Padel_video_8.mp4 --output-name v8_laptop_run --save-video
```

Annotated video path: `outputs/edge/annotated_videos/<output-name>_annotated.mp4`.

## 10) Recommended end‑to‑end workflow (short)

1. Prepare `input_videos/` and confirm `models/` contains a YOLO ONNX or `.pt` model.
2. Run `src/edge/edge_inference.py` (fast, edge-like) to generate `clean_csv` and `hit_candidates`.
3. Launch `src/edge/contact_labeler.py` to manually label candidates.
4. Train scorer: `scripts/train_contact_scorer.py --labels outputs/edge/labels/*.csv`.
5. Score candidates with the trained scorer: `scripts/score_windowed_contact_scorer.py --model outputs/edge/contact_models/<model>.json --candidates-csv ...`.
6. Inspect annotated video and metrics; iterate on labels / thresholds as needed.

## 11) Troubleshooting & tips

- If `contact_labeler.py` fails to open GUI, ensure `pillow` is installed: `pip install pillow`.
- If a candidate CSV is missing, check `outputs/edge/hit_candidates/` and re-run inference.
- Use lightweight model `yolov8n.pt` for edge runs; heavy models (`yolov8x`) will be too slow for Pi.
- To push models to Pi, use `scp` or `rsync` and run edge scripts on the Pi inside a Python venv.

## 12) Where to find more detailed docs

- Developer reference: `DOCUMENTATION.md` (in repo root) — contains detailed design notes and advanced examples.
- YOLO edge training guide: `YOLOV8N_TRAINING_GUIDE.md` — complete YOLOv8n training and export steps.

---

If you want, I can now update this `README.md` in-place, or also add a short `USAGE.md` with copy-paste command blocks tailored for Windows and Linux. Which do you prefer? 
