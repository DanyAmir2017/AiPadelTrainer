# YOLOv8n Training Guide for Edge Deployment

Complete guide to training YOLOv8n (nano) models for Raspberry Pi 5 + Hailo-8 deployment.

## Why YOLOv8n?

Your current models use **YOLOv8x** (68M parameters, 130MB) which is:
- ❌ Too large for Raspberry Pi
- ❌ Too slow on Hailo-8 accelerator (<1 FPS on Pi CPU)
- ❌ Cannot be efficiently compiled to HEF format

**YOLOv8n** (3M parameters, 6MB) is:
- ✅ Optimized for edge devices
- ✅ Fast on Hailo-8 (40-60 FPS expected)
- ✅ Maintains good accuracy (~5-10% loss vs YOLOv8x)
- ✅ Compiles efficiently to HEF format

---

## Prerequisites

### Hardware
- Development machine with NVIDIA GPU (RTX 3050 Ti or better)
- 8GB+ RAM
- 50GB+ free disk space

### Software
```bash
# Ensure you're in yolo_env
conda activate yolo_env

# Verify CUDA is available
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# Verify Ultralytics
yolo version
```

---

## Step 1: Prepare Training Data

### Check Existing Dataset

```bash
cd ~/padel_trainer/training/dataset

# Verify structure
ls -R
# Should contain:
# data.yaml
# images/train/
# images/val/
# labels/train/
# labels/val/
```

### Verify data.yaml

```yaml
# training/dataset/data.yaml
path: C:/Daniel/GUC/Bachelor/Thesis Progress/Software/padel_trainer/training/dataset
train: images/train
val: images/val
names:
  0: ball

nc: 1
```

### Check Dataset Size

```bash
# Count training images
ls training/dataset/images/train/*.jpg | wc -l

# Count validation images
ls training/dataset/images/val/*.jpg | wc -l

# Recommended: 500+ training, 100+ validation
```

---

## Step 2: Train YOLOv8n Ball Detector

### Basic Training

```bash
cd ~/padel_trainer

# Train YOLOv8n (recommended settings)
yolo train \
  model=yolov8n.pt \
  data=training/dataset/data.yaml \
  epochs=100 \
  imgsz=640 \
  batch=16 \
  device=0 \
  patience=20 \
  save=True \
  project=runs/detect_nano \
  name=ball_nano

# Training will take ~30-60 minutes on RTX 3050 Ti
```

### Advanced Training (Better Accuracy)

```bash
# Extended training with augmentation
yolo train \
  model=yolov8n.pt \
  data=training/dataset/data.yaml \
  epochs=150 \
  imgsz=640 \
  batch=16 \
  device=0 \
  patience=30 \
  save=True \
  project=runs/detect_nano \
  name=ball_nano_v2 \
  hsv_h=0.015 \
  hsv_s=0.7 \
  hsv_v=0.4 \
  degrees=10 \
  translate=0.1 \
  scale=0.5 \
  flipud=0.0 \
  fliplr=0.5 \
  mosaic=1.0
```

### Training Output

Model will be saved to:
```
runs/detect_nano/ball_nano/weights/
├── best.pt      ← Use this for inference
└── last.pt      ← Checkpoint
```

---

## Step 3: Validate Nano Model

### Check Performance

```bash
# Validate on test set
yolo val \
  model=runs/detect_nano/ball_nano/weights/best.pt \
  data=training/dataset/data.yaml \
  imgsz=640

# Check metrics:
# - mAP@0.5: Should be >0.85 (85%)
# - Precision: Should be >0.80
# - Recall: Should be >0.75
```

### Test on Video

```bash
# Quick test
yolo predict \
  model=runs/detect_nano/ball_nano/weights/best.pt \
  source=input_videos/Padel_video_5.mp4 \
  imgsz=640 \
  conf=0.01 \
  save=True

# Check output: runs/detect/predict/
```

### Compare with YOLOv8x

```bash
# Test YOLOv8x (current)
yolo predict model=models/best_ball.pt source=input_videos/Padel_video_5.mp4 conf=0.01

# Test YOLOv8n (new)
yolo predict model=runs/detect_nano/ball_nano/weights/best.pt source=input_videos/Padel_video_5.mp4 conf=0.01

# Compare:
# - Detection count
# - False positives
# - Visual accuracy
```

---

## Step 4: Export to ONNX

### Static Shape Export (Required for Hailo)

```bash
# Export with static 640x640 input
yolo export \
  model=runs/detect_nano/ball_nano/weights/best.pt \
  format=onnx \
  opset=12 \
  imgsz=640 \
  dynamic=False \
  simplify=True

# Output: best.onnx (~6-10 MB)
```

### Copy to Models Directory

```bash
# Copy to edge deployment folder
cp runs/detect_nano/ball_nano/weights/best.onnx \
   models/onnx/best_ball_nano.onnx

# Verify size
ls -lh models/onnx/best_ball_nano.onnx
# Should be ~6-10 MB (NOT 260 MB)
```

---

## Step 5: Test Edge Pipeline (Development)

### Test ONNX Inference Locally

```bash
cd src/edge

# Update config to use nano model
# (Should already point to best_ball_nano.onnx)

# Test inference
python edge_inference.py ../../input_videos/Padel_video_5.mp4 --verbose

# Check output:
# - FPS: Should be 60-80+ on RTX 3050 Ti
# - Detections: Compare with YOLOv8x baseline
# - CSV: outputs/edge/Padel_video_5_trajectory.csv
```

---

## Step 6: Deploy to Raspberry Pi

### Transfer Files

```bash
# From development machine
cd ~/padel_trainer

# Copy ONNX model
scp models/onnx/best_ball_nano.onnx \
    pi@raspberrypi5.local:~/padel_trainer/models/onnx/

# Copy edge code
scp -r src/edge \
    pi@raspberrypi5.local:~/padel_trainer/src/

# Copy test video
scp input_videos/Padel_video_5.mp4 \
    pi@raspberrypi5.local:~/padel_trainer/videos/
```

### Install on Pi

```bash
# SSH into Pi
ssh pi@raspberrypi5.local

cd ~/padel_trainer/src/edge

# Setup environment
python3 -m venv venv
source venv/bin/activate
pip install numpy opencv-python onnxruntime filterpy
```

### Test on Pi (CPU)

```bash
# Run inference
python edge_inference.py ../../videos/Padel_video_5.mp4 --verbose --benchmark

# Expected FPS: 5-10 FPS (CPU only)
```

---

## Step 7: Hailo Compilation (Optional)

### Prepare Calibration Dataset

```bash
# On development machine
cd ~/padel_trainer

# Create calibration images (100-200 random frames)
mkdir -p hailo/calibration_images

# Extract frames from video
ffmpeg -i input_videos/Padel_video_5.mp4 \
       -vf "select=not(mod(n\,30))" \
       -vsync vfr \
       hailo/calibration_images/frame_%04d.jpg

# You need 100-200 images for calibration
```

### Compile to HEF

```bash
# Requires Hailo Dataflow Compiler (separate installation)
# See: https://hailo.ai/developer-zone/

hailomz compile yolov8n \
  --ckpt models/onnx/best_ball_nano.onnx \
  --calib-path hailo/calibration_images/ \
  --hw-arch hailo8 \
  --output models/onnx/best_ball_nano.hef

# Compilation takes ~10-30 minutes
# Output: best_ball_nano.hef (~5-8 MB)
```

### Deploy HEF to Pi

```bash
# Copy HEF model
scp models/onnx/best_ball_nano.hef \
    pi@raspberrypi5.local:~/padel_trainer/models/onnx/

# On Pi, update config
# Edit src/edge/edge_config.py:
#   INFERENCE_ENGINE = 'hailo'
#   BALL_MODEL_PATH = MODELS_DIR / "best_ball_nano.hef"

# Run with Hailo
python edge_inference.py ../../videos/Padel_video_5.mp4 --verbose

# Expected FPS: 40-60 FPS (with Hailo-8)
```

---

## Expected Results

### Performance Comparison

| Model | Size | Dev Machine FPS | Pi CPU FPS | Pi Hailo FPS |
|-------|------|----------------|-----------|--------------|
| YOLOv8x (current) | 130 MB | 60-80 | <1 | N/A |
| **YOLOv8n (new)** | 6 MB | 200+ | 5-10 | **40-60** |

### Accuracy Comparison (Expected)

| Metric | YOLOv8x | YOLOv8n | Difference |
|--------|---------|---------|------------|
| mAP@0.5 | 0.92 | 0.87 | -5% |
| Precision | 0.89 | 0.84 | -5% |
| Recall | 0.86 | 0.81 | -5% |
| Detection Rate (Video 5) | 74% | 68-72% | -2-6% |

Trade-off: **Slight accuracy loss for 40-60x speed improvement on Pi!**

---

## Troubleshooting

### Training Issues

**Problem:** Loss not decreasing

```bash
# Reduce learning rate
yolo train model=yolov8n.pt data=training/dataset/data.yaml \
     epochs=100 imgsz=640 batch=16 lr0=0.001
```

**Problem:** Overfitting

```bash
# Add more augmentation
yolo train model=yolov8n.pt data=training/dataset/data.yaml \
     epochs=100 imgsz=640 batch=16 dropout=0.1
```

### Export Issues

**Problem:** ONNX file too large (>50 MB)

```bash
# Check if you're using nano model
python -c "from ultralytics import YOLO; m = YOLO('runs/detect_nano/ball_nano/weights/best.pt'); print(m.model.yaml['width_multiple'])"
# Should print: 0.25 (nano)
# If 1.25, you're using YOLOv8x by mistake!
```

### Pi Deployment Issues

**Problem:** ModuleNotFoundError

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade numpy opencv-python onnxruntime filterpy
```

---

## Advanced: Fine-tuning from YOLOv8x

If you want to start from your existing trained YOLOv8x model:

```bash
# Start from your trained YOLOv8x weights
# But use YOLOv8n architecture
yolo train \
  model=yolov8n.pt \
  data=training/dataset/data.yaml \
  epochs=50 \
  imgsz=640 \
  batch=16 \
  device=0 \
  pretrained=models/best_ball.pt \
  freeze=10

# This may give slightly better accuracy than training from scratch
```

---

## Quick Reference Commands

```bash
# 1. Train nano model
yolo train model=yolov8n.pt data=training/dataset/data.yaml epochs=100 imgsz=640 batch=16

# 2. Export to ONNX
yolo export model=runs/detect_nano/ball_nano/weights/best.pt format=onnx opset=12 imgsz=640 dynamic=False

# 3. Copy to edge folder
cp runs/detect_nano/ball_nano/weights/best.onnx models/onnx/best_ball_nano.onnx

# 4. Test locally
cd src/edge
python edge_inference.py ../../input_videos/Padel_video_5.mp4 --verbose

# 5. Deploy to Pi
scp models/onnx/best_ball_nano.onnx pi@raspberrypi5.local:~/padel_trainer/models/onnx/
```

---

## Summary

✅ Train YOLOv8n instead of YOLOv8x  
✅ Export to ONNX with static 640x640 shape  
✅ Test locally first (development machine)  
✅ Deploy to Pi and verify 5-10 FPS  
✅ Optional: Compile to HEF for 40-60 FPS  

**Your edge deployment will be production-ready! 🚀**
