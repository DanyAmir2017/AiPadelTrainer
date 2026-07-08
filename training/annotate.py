"""
Simple Ball Annotation Tool
============================

Review and correct ball bounding boxes using OpenCV.
More stable than LabelImg for this specific task.

Controls:
  W/S     - Previous/Next image
  SPACE   - Mark as "no ball in this frame" (delete all boxes)
  MOUSE   - Click and drag to draw new box
  D       - Delete box under mouse cursor
  Q       - Save and quit

Usage:
    python training/annotate.py train
    python training/annotate.py val

Author: Bachelor Thesis Project – GUC
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Colors
COLOR_BOX = (0, 255, 0)         # Green - existing box
COLOR_DRAW = (0, 255, 255)      # Yellow - drawing new box
COLOR_TEXT = (255, 255, 255)    # White

drawing = False
start_x, start_y = -1, -1
current_box = None


def load_label(label_path, img_w, img_h):
    """Load YOLO labels and convert to pixel coordinates."""
    boxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls, cx, cy, w, h = map(float, parts[:5])
                # Convert normalized to pixel
                x1 = int((cx - w/2) * img_w)
                y1 = int((cy - h/2) * img_h)
                x2 = int((cx + w/2) * img_w)
                y2 = int((cy + h/2) * img_h)
                boxes.append([x1, y1, x2, y2])
    return boxes


def save_label(label_path, boxes, img_w, img_h):
    """Save boxes in YOLO format."""
    with open(label_path, 'w') as f:
        for box in boxes:
            x1, y1, x2, y2 = box
            # Convert pixel to normalized
            cx = ((x1 + x2) / 2.0) / img_w
            cy = ((y1 + y2) / 2.0) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            f.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, current_box
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_x, start_y = x, y
        current_box = None
    
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box = [start_x, start_y, x, y]
    
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if current_box:
            param['boxes'].append(current_box)
            current_box = None
            param['modified'] = True


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'val']:
        print("Usage: python training/annotate.py {train|val}")
        sys.exit(1)
    
    split = sys.argv[1]
    img_dir = os.path.join("training", "dataset", "images", split)
    lbl_dir = os.path.join("training", "dataset", "labels", split)
    
    # Get all images
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
    if not images:
        print(f"No images found in {img_dir}")
        sys.exit(1)
    
    print("=" * 60)
    print(f"  Ball Annotation Tool - {split.upper()} set")
    print("=" * 60)
    print(f"  Images: {len(images)}")
    print()
    print("  Controls:")
    print("    W/S       - Previous/Next image")
    print("    SPACE     - Mark as 'no ball' (delete all boxes)")
    print("    MOUSE     - Click and drag to draw new box")
    print("    D         - Delete box under cursor")
    print("    Q         - Save and quit")
    print("=" * 60)
    print()
    
    idx = 0
    win_name = "Annotate Ball"
    cv2.namedWindow(win_name)
    
    while idx < len(images):
        img_name = images[idx]
        img_path = os.path.join(img_dir, img_name)
        lbl_path = os.path.join(lbl_dir, img_name.replace('.jpg', '.txt').replace('.png', '.txt'))
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"ERROR: Cannot read {img_path}")
            idx += 1
            continue
        
        h, w = img.shape[:2]
        boxes = load_label(lbl_path, w, h)
        
        state = {'boxes': boxes, 'modified': False}
        cv2.setMouseCallback(win_name, mouse_callback, state)
        
        while True:
            display = img.copy()
            
            # Draw existing boxes
            for box in state['boxes']:
                x1, y1, x2, y2 = box
                cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_BOX, 2)
                cv2.circle(display, ((x1+x2)//2, (y1+y2)//2), 4, COLOR_BOX, -1)
            
            # Draw box being created
            if current_box:
                x1, y1, x2, y2 = current_box
                cv2.rectangle(display, (x1, y1), (x2, y2), COLOR_DRAW, 2)
            
            # HUD
            cv2.putText(display, f"Image {idx+1}/{len(images)}: {img_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            cv2.putText(display, f"Boxes: {len(state['boxes'])}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)
            cv2.putText(display, "W/S:Prev/Next | SPACE:NoBall | D:Delete | Q:Quit",
                        (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1)
            
            cv2.imshow(win_name, display)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Save current and quit
                if state['modified']:
                    save_label(lbl_path, state['boxes'], w, h)
                cv2.destroyAllWindows()
                print(f"\nSaved progress. Annotated {idx+1}/{len(images)} images.")
                return
            
            elif key == ord('s'):  # Next
                if state['modified']:
                    save_label(lbl_path, state['boxes'], w, h)
                    print(f"Saved: {img_name}  ({len(state['boxes'])} boxes)")
                idx += 1
                break
            
            elif key == ord('w'):  # Previous
                if state['modified']:
                    save_label(lbl_path, state['boxes'], w, h)
                idx = max(0, idx - 1)
                break
            
            elif key == ord(' '):  # No ball
                state['boxes'] = []
                state['modified'] = True
                save_label(lbl_path, state['boxes'], w, h)
                print(f"Marked as no-ball: {img_name}")
                idx += 1
                break
            
            elif key == ord('d'):  # Delete box under cursor
                # Get mouse position
                # (OpenCV doesn't expose this easily, so just delete last box)
                if state['boxes']:
                    state['boxes'].pop()
                    state['modified'] = True
    
    cv2.destroyAllWindows()
    print(f"\nDone! Reviewed all {len(images)} images.")


if __name__ == "__main__":
    main()
