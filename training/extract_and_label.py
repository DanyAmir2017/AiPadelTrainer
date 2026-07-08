"""
Frame Extraction & Pseudo-Label Generator
==========================================

Extracts frames from a video and generates YOLO-format pseudo-labels
using the current ball model at very low confidence. The labels can
then be reviewed and corrected manually in LabelImg before fine-tuning.

Usage:
    python training/extract_and_label.py

Author: Bachelor Thesis Project – GUC
"""

import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────
VIDEO_PATH = os.path.join("input_videos", "Padel_video_3.mp4")
MODEL_PATH = os.path.join("models", "best_ball.pt")

DATASET_DIR = os.path.join("training", "dataset")
IMAGES_TRAIN = os.path.join(DATASET_DIR, "images", "train")
IMAGES_VAL = os.path.join(DATASET_DIR, "images", "val")
LABELS_TRAIN = os.path.join(DATASET_DIR, "labels", "train")
LABELS_VAL = os.path.join(DATASET_DIR, "labels", "val")

EXTRACT_EVERY_N = 5        # Extract every Nth frame (avoid near-duplicate images)
VAL_SPLIT = 0.15            # 15% of frames go to validation
PSEUDO_CONF = 0.02          # Very low confidence to catch as many balls as possible
CLASS_ID = 0                # Ball class
DEVICE = 0                  # GPU


def main():
    # Ensure dirs exist
    for d in [IMAGES_TRAIN, IMAGES_VAL, LABELS_TRAIN, LABELS_VAL]:
        os.makedirs(d, exist_ok=True)

    # Load video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Video: {VIDEO_PATH}  ({width}x{height}, {total_frames} frames)")
    print(f"Extracting every {EXTRACT_EVERY_N} frames → ~{total_frames // EXTRACT_EVERY_N} images")

    # Load model
    model = YOLO(MODEL_PATH)
    print(f"Model: {MODEL_PATH}")
    print(f"Pseudo-label confidence: {PSEUDO_CONF}")
    print()

    frame_idx = 0
    extracted = 0
    labelled = 0
    val_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        if frame_idx % EXTRACT_EVERY_N != 0:
            continue

        # Decide train vs val
        is_val = (extracted % int(1 / VAL_SPLIT)) == 0 if VAL_SPLIT > 0 else False
        img_dir = IMAGES_VAL if is_val else IMAGES_TRAIN
        lbl_dir = LABELS_VAL if is_val else LABELS_TRAIN

        # Save frame as image
        img_name = f"frame_{frame_idx:05d}.jpg"
        img_path = os.path.join(img_dir, img_name)
        cv2.imwrite(img_path, frame)

        # Run inference at very low confidence
        results = model.predict(
            frame, conf=PSEUDO_CONF, imgsz=640,
            device=DEVICE, verbose=False,
        )

        # Write YOLO label file
        lbl_name = f"frame_{frame_idx:05d}.txt"
        lbl_path = os.path.join(lbl_dir, lbl_name)

        lines = []
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu())

                # Convert to YOLO format (normalised cx, cy, w, h)
                cx = ((x1 + x2) / 2.0) / width
                cy = ((y1 + y2) / 2.0) / height
                bw = (x2 - x1) / width
                bh = (y2 - y1) / height

                lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Write label (empty file = no ball in this frame, which is also valid)
        with open(lbl_path, "w") as f:
            f.write("\n".join(lines))

        extracted += 1
        if lines:
            labelled += 1
        if is_val:
            val_count += 1

        if extracted % 20 == 0:
            print(f"  Extracted {extracted} frames, {labelled} with pseudo-labels...")

    cap.release()

    train_count = extracted - val_count
    print()
    print("=" * 50)
    print(f"  Extraction Complete")
    print(f"  Total frames extracted: {extracted}")
    print(f"  Train: {train_count}  |  Val: {val_count}")
    print(f"  Frames with pseudo-labels: {labelled}")
    print(f"  Frames without labels: {extracted - labelled}")
    print("=" * 50)
    print()
    print("NEXT STEPS:")
    print("  1. Install LabelImg:  pip install labelImg")
    print("  2. Open and review:   labelImg training/dataset/images/train training/dataset/labels/train")
    print("     - Delete wrong boxes (false positives)")
    print("     - Add missing boxes where the ball is visible")
    print("     - Class name should be 'ball'")
    print("  3. Do the same for val:  labelImg training/dataset/images/val training/dataset/labels/val")
    print("  4. Run fine-tuning:   python training/finetune.py")


if __name__ == "__main__":
    main()
