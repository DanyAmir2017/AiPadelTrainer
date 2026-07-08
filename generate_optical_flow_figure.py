"""
Generate Optical Flow Visualization Figure for Thesis
======================================================

Extracts two consecutive frames from a test video, computes Lucas-Kanade
optical flow, and creates a publication-ready figure showing motion vectors.

Usage:
    python generate_optical_flow_figure.py --video input_videos/video_name.mp4 --frame 100 --output PaperLatex/figures/optical_flow_fig.png

Author: Padel Trainer
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import from project
from src.utils.config import (
    FEATURE_PARAMS,
    LK_PARAMS,
    INPUT_VIDEO_DIR,
    PROJECT_ROOT,
)


def draw_flow_vectors(image, prev_gray, curr_gray, prev_points, curr_points, mask=None, scale_factor=3.0):
    """
    Draw optical flow motion vectors on image with magnified visualization.
    
    Args:
        image: BGR image to draw on
        prev_gray: Previous grayscale frame
        curr_gray: Current grayscale frame
        prev_points: Feature points in previous frame
        curr_points: Feature points in current frame (tracked)
        mask: Binary mask for valid points
        scale_factor: Scale multiplier for motion vectors (makes small motions visible)
    
    Returns:
        Image with flow vectors drawn
    """
    img_rgb = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
    
    if prev_points is not None and curr_points is not None:
        # Draw motion vectors with magnified scale for visibility
        for i, (p1, p2) in enumerate(zip(prev_points, curr_points)):
            if mask is None or mask[i]:
                x1, y1 = int(p1[0, 0]), int(p1[0, 1])
                x2, y2 = int(p2[0, 0]), int(p2[0, 1])
                
                # Calculate motion magnitude
                dx = x2 - x1
                dy = y2 - y1
                magnitude = np.sqrt(dx**2 + dy**2)
                
                # Magnify small motions for visibility
                if magnitude > 0:
                    scale = max(1.0, scale_factor / (magnitude + 0.1))
                    x2_scaled = int(x1 + dx * scale)
                    y2_scaled = int(y1 + dy * scale)
                else:
                    x2_scaled, y2_scaled = x2, y2
                
                # Color code based on magnitude
                if magnitude > 5:
                    color = (255, 0, 0)  # Red for large motion
                    thickness = 4
                elif magnitude > 2:
                    color = (0, 165, 255)  # Orange for medium motion
                    thickness = 3
                else:
                    color = (0, 200, 0)  # Green for small motion
                    thickness = 3
                
                # Draw magnified arrow
                cv2.arrowedLine(img_rgb, (x1, y1), (x2_scaled, y2_scaled), 
                               color=color, thickness=thickness, tipLength=0.5)
                # Draw start point (larger)
                cv2.circle(img_rgb, (x1, y1), radius=6, color=(255, 0, 0), thickness=-1)
                # Draw end point
                cv2.circle(img_rgb, (x2_scaled, y2_scaled), radius=4, color=color, thickness=-1)
    
    return img_rgb


def compute_optical_flow(prev_gray, curr_gray, prev_points):
    """
    Compute Lucas-Kanade optical flow.
    
    Args:
        prev_gray: Previous grayscale frame
        curr_gray: Current grayscale frame
        prev_points: Feature points to track
    
    Returns:
        Tracked points, status mask
    """
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    
    if prev_points is None or len(prev_points) == 0:
        return None, None
    
    # Compute LK optical flow
    curr_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, prev_points, None, **lk_params
    )
    
    # Filter by status
    if status is not None:
        good_status = status.flatten().astype(bool)
    else:
        good_status = np.ones(len(prev_points), dtype=bool)
    
    return curr_points, good_status


def extract_frame_pair(video_path, frame_idx=100):
    """
    Extract two consecutive frames from video.
    
    Args:
        video_path: Path to video file
        frame_idx: Frame index to start
    
    Returns:
        frame1 (BGR), frame2 (BGR), frame_numbers
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Seek to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    ret1, frame1 = cap.read()
    ret2, frame2 = cap.read()
    
    cap.release()
    
    if not ret1 or not ret2:
        raise ValueError(f"Cannot read frames at index {frame_idx}")
    
    return frame1, frame2, (frame_idx, frame_idx + 1)


def generate_figure(video_path, frame_idx, output_path):
    """
    Generate optical flow visualization figure focused on ball motion.
    
    Args:
        video_path: Path to input video
        frame_idx: Frame index to extract
        output_path: Output image path
    """
    # Import ball detector
    from src.detection.ball_detector import BallDetector
    
    # Extract frames
    frame1, frame2, frame_nums = extract_frame_pair(video_path, frame_idx)
    
    # Detect ball in frame 1 to focus optical flow region
    detector = BallDetector()
    detection1 = detector.detect(frame1)
    
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # If ball detected, focus on that region
    roi_mask = None
    ball_center = None
    roi_x1, roi_y1 = 0, 0
    
    if detection1 is not None:
        # Get ball bounding box
        cx, cy = detection1.x, detection1.y
        w, h = detection1.w, detection1.h
        
        x1, y1 = int(cx - w/2), int(cy - h/2)
        x2, y2 = int(cx + w/2), int(cy + h/2)
        
        # Expand region around ball for better context
        margin = 80
        roi_x1 = max(0, x1 - margin)
        roi_y1 = max(0, y1 - margin)
        roi_x2 = min(frame1.shape[1], x2 + margin)
        roi_y2 = min(frame1.shape[0], y2 + margin)
        
        # Create ROI mask
        roi_mask = np.zeros(gray1.shape, dtype=np.uint8)
        roi_mask[roi_y1:roi_y2, roi_x1:roi_x2] = 255
        
        # Ball center
        ball_center = (int(cx), int(cy))
        print(f"Ball detected at: {ball_center}")
        
        # Crop frames to ROI for feature detection
        gray1_roi = gray1[roi_y1:roi_y2, roi_x1:roi_x2]
        gray2_roi = gray2[roi_y1:roi_y2, roi_x1:roi_x2]
    else:
        print("No ball detected, using full frame")
        gray1_roi = gray1
        gray2_roi = gray2
        roi_x1, roi_y1 = 0, 0
    
    # Detect features in ROI
    feature_params = dict(
        maxCorners=150,
        qualityLevel=0.01,
        minDistance=5,
        blockSize=7,
    )
    prev_points = cv2.goodFeaturesToTrack(gray1_roi, **feature_params)
    
    if prev_points is None or len(prev_points) == 0:
        print("No features detected in ROI. Trying full frame...")
        feature_params['maxCorners'] = 100
        prev_points = cv2.goodFeaturesToTrack(gray1, **feature_params)
        roi_x1, roi_y1 = 0, 0
    else:
        # Adjust points to full frame coordinates
        if roi_x1 > 0 or roi_y1 > 0:
            prev_points = prev_points.copy().astype(np.float32)
            prev_points[:, 0, 0] += roi_x1
            prev_points[:, 0, 1] += roi_y1
    
    if prev_points is None:
        raise RuntimeError("Cannot detect features in frame")
    
    # Compute optical flow
    curr_points, status = compute_optical_flow(gray1, gray2, prev_points)
    
    if curr_points is None:
        raise RuntimeError("Optical flow computation failed")
    
    # Draw flow vectors on full frame
    frame1_flow = draw_flow_vectors(frame1, gray1, gray2, prev_points, curr_points, status, scale_factor=4.0)
    frame2_with_features = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)
    
    # Highlight ball if detected
    if ball_center:
        cv2.circle(frame1_flow, ball_center, radius=15, color=(255, 215, 0), thickness=3)
        cv2.circle(frame2_with_features, ball_center, radius=15, color=(255, 215, 0), thickness=3)
    
    # Create publication-quality figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    fig.suptitle('Lucas-Kanade Optical Flow Motion Estimation (Ball Focus)', fontsize=14, fontweight='bold')
    
    # Frame 1 with flow vectors
    axes[0].imshow(frame1_flow)
    axes[0].set_title(f'Frame {frame_nums[0]}: Motion Vectors (Magnified)', fontsize=12)
    axes[0].set_xlabel('X Pixels', fontsize=11)
    axes[0].set_ylabel('Y Pixels', fontsize=11)
    axes[0].grid(True, alpha=0.2, linestyle='--')
    
    # Add legend
    start_dot = mpatches.Patch(color='red', label='Feature Points')
    red_arrow = mpatches.Patch(color='red', label='Large Motion (>5px)')
    orange_arrow = mpatches.Patch(color='orange', label='Medium Motion (2-5px)')
    green_arrow = mpatches.Patch(color='lime', label='Small Motion (<2px)')
    gold_circle = mpatches.Patch(color='gold', label='Ball Position')
    axes[0].legend(handles=[start_dot, red_arrow, orange_arrow, green_arrow, gold_circle], 
                   loc='upper right', fontsize=9)
    
    # Frame 2
    axes[1].imshow(frame2_with_features)
    axes[1].set_title(f'Frame {frame_nums[1]}: Next Frame', fontsize=12)
    axes[1].set_xlabel('X Pixels', fontsize=11)
    axes[1].set_ylabel('Y Pixels', fontsize=11)
    axes[1].grid(True, alpha=0.2, linestyle='--')
    
    plt.tight_layout()
    
    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='png')
    print(f"✓ Figure saved: {output_path}")
    
    # Also save as PDF for LaTeX inclusion
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', format='pdf')
    print(f"✓ PDF version saved: {pdf_path}")
    
    plt.close()
    
    # Print statistics
    valid_points = np.sum(status.flatten()) if status is not None else len(prev_points)
    print(f"\nOptical Flow Statistics:")
    print(f"  Total features detected: {len(prev_points)}")
    print(f"  Valid tracked points: {valid_points}")
    print(f"  Tracking success rate: {valid_points/len(prev_points)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description='Generate optical flow visualization figure for thesis'
    )
    parser.add_argument(
        '--video',
        type=str,
        default='input_videos/Padel_video_1.mp4',
        help='Path to input video (relative to project root)'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=100,
        help='Frame index to extract (default: 100)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='PaperLatex/figures/optical_flow_fig.png',
        help='Output figure path'
    )
    
    args = parser.parse_args()
    
    # Resolve video path
    video_path = Path(PROJECT_ROOT) / args.video
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    print(f"Generating optical flow figure...")
    print(f"  Video: {video_path}")
    print(f"  Frame: {args.frame}")
    print(f"  Output: {args.output}")
    
    generate_figure(video_path, args.frame, args.output)
    print("\n✓ Figure generation complete!")


if __name__ == '__main__':
    main()
