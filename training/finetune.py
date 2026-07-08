"""
Fine-tune Ball Detection Model
================================

Fine-tunes the existing best_ball.pt model on the corrected
pseudo-labelled dataset from video 3.

Usage:
    python training/finetune.py

Author: Bachelor Thesis Project – GUC
"""

import os
from ultralytics import YOLO

# ── Configuration ────────────────────────────────────────────────────
BASE_MODEL = os.path.join("models", "best_ball.pt")   # Start from existing weights
DATASET_YAML = os.path.join("training", "dataset.yaml")
OUTPUT_DIR = os.path.join("training", "runs")

# Training hyperparameters
EPOCHS = 50            # Fine-tuning doesn't need many epochs
IMG_SIZE = 640
BATCH_SIZE = 8         # Adjust based on GPU VRAM (RTX 3050 Ti = 4 GB)
LEARNING_RATE = 0.001  # Lower LR for fine-tuning (don't overwrite base knowledge)
FREEZE = 10            # Freeze first 10 layers (backbone) – only fine-tune head
DEVICE = 0             # GPU


def main():
    print("=" * 50)
    print("  Fine-tuning Ball Detection Model")
    print(f"  Base model : {BASE_MODEL}")
    print(f"  Dataset    : {DATASET_YAML}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  Freeze     : {FREEZE} layers")
    print(f"  Device     : {DEVICE}")
    print("=" * 50)
    print()

    # Load existing model
    model = YOLO(BASE_MODEL)

    # Fine-tune
    results = model.train(
        data=DATASET_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        lr0=LEARNING_RATE,
        lrf=0.01,               # Final LR = lr0 * lrf
        freeze=FREEZE,
        device=DEVICE,
        project=OUTPUT_DIR,
        name="ball_finetune",
        exist_ok=True,
        patience=10,            # Early stopping
        save=True,
        save_period=10,         # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
    )

    # Copy best weights
    best_path = os.path.join(OUTPUT_DIR, "ball_finetune", "weights", "best.pt")
    if os.path.exists(best_path):
        dest = os.path.join("models", "best_ball_finetuned.pt")
        import shutil
        shutil.copy2(best_path, dest)
        print()
        print("=" * 50)
        print(f"  Fine-tuned model saved to: {dest}")
        print()
        print("  To use it, update config.py:")
        print('    BALL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_ball_finetuned.pt")')
        print("=" * 50)
    else:
        print(f"WARNING: best.pt not found at {best_path}")


if __name__ == "__main__":
    main()
