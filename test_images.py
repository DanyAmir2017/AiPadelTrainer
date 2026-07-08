"""
Test Ball Detection on Static Images
=====================================

Load images from input_images/ folder and detect balls.
Show all detections with confidence scores.

Author: Bachelor Thesis Project – GUC
"""

import os
import cv2
import numpy as np
from pathlib import Path

# Import ball detector
from src.detection.ball_detector import BallDetector


def test_images():
    """Test ball detection on images in input_images/ folder."""
    
    input_dir = "input_images"
    output_dir = "outputs/test_images"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize detector
    print("Loading ball detector...")
    detector = BallDetector()
    print("Detector ready.\n")
    
    # Get all images
    image_files = sorted(Path(input_dir).glob("*.jpg")) + sorted(Path(input_dir).glob("*.png"))
    
    if not image_files:
        print(f"No images found in {input_dir}/")
        return
    
    print(f"Found {len(image_files)} images\n")
    print("=" * 70)
    
    for img_path in image_files:
        img_name = img_path.name
        print(f"\nProcessing: {img_name}")
        print("-" * 70)
        
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ERROR: Could not read image")
            continue
        
        h, w = img.shape[:2]
        print(f"  Resolution: {w}x{h}")
        
        # Get ALL detections (including those filtered out)
        all_detections = detector.detect_all(img)
        
        print(f"  Total detections: {len(all_detections)}")
        
        if all_detections:
            print("\n  Detections (confidence, position, size):")
            for i, det in enumerate(all_detections[:10], 1):  # Show top 10
                print(f"    {i}. conf={det.confidence:.3f}  pos=({det.x:.1f}, {det.y:.1f})  size=({det.w:.1f}x{det.h:.1f})")
        else:
            print("  No detections found!")
        
        # Draw all detections on image
        output_img = img.copy()
        for i, det in enumerate(all_detections[:10], 1):  # Draw top 10
            x1 = int(det.x - det.w / 2)
            y1 = int(det.y - det.h / 2)
            x2 = int(det.x + det.w / 2)
            y2 = int(det.y + det.h / 2)
            
            # Color based on confidence (green=high, yellow=medium, red=low)
            if det.confidence > 0.5:
                color = (0, 255, 0)  # Green
            elif det.confidence > 0.1:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Draw box
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            
            # Draw center
            cv2.circle(output_img, (int(det.x), int(det.y)), 4, color, -1)
            
            # Draw label
            label = f"#{i} {det.confidence:.2f}"
            cv2.putText(output_img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Save annotated image
        output_path = os.path.join(output_dir, f"detected_{img_name}")
        cv2.imwrite(output_path, output_img)
        print(f"\n  Saved: {output_path}")
    
    print("\n" + "=" * 70)
    print(f"Done! Check {output_dir}/ for annotated images.")


if __name__ == "__main__":
    test_images()
