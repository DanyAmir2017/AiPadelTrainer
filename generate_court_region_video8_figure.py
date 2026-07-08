"""
Generate Court Region Grid on a Video-8 Frame
=============================================

Draws the six court classification regions directly on top of a real frame
from Padel_video_8.mp4 using the detected court area.

Output:
- PNG and PDF for thesis inclusion

Usage:
    python generate_court_region_video8_figure.py
    python generate_court_region_video8_figure.py --frame 757

Author: Padel Trainer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.detection.court_detector import CourtDetector


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "input_videos" / "Padel_video_8.mp4"
OUTPUT_DIR = PROJECT_ROOT / "PaperLatex" / "figures"

REGION_COLORS = {
    "Front Left": "#dbeafe",
    "Front Right": "#dbeafe",
    "Mid Left": "#e5e7eb",
    "Mid Right": "#e5e7eb",
    "Back Left": "#cbd5e1",
    "Back Right": "#cbd5e1",
}


def extract_frame(video_path: Path, frame_num: int):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_num} from {video_path}")
    return frame


def detect_court(frame):
    detector = CourtDetector()
    return detector.detect(frame)


def get_crop_bounds(frame, court_detection):
    h, w = frame.shape[:2]
    if court_detection.count == 0:
        # Conservative fallback around the visible court area in a typical broadcast frame.
        return int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)

    points = court_detection.as_array()
    x1 = max(0, int(points[:, 0].min() - 80))
    y1 = max(0, int(points[:, 1].min() - 80))
    x2 = min(w, int(points[:, 0].max() + 80))
    y2 = min(h, int(points[:, 1].max() + 80))
    return x1, y1, x2, y2


def draw_overlay(frame, court_detection, frame_num: int):
    x1, y1, x2, y2 = get_crop_bounds(frame, court_detection)
    crop = frame[y1:y2, x1:x2].copy()
    ch, cw = crop.shape[:2]

    # Create a light overlay so the court remains visible beneath the grid.
    overlay = crop.copy()
    region_h = ch / 3.0
    region_w = cw / 2.0

    region_specs = [
        (0, 0, "Front Left"),
        (1, 0, "Front Right"),
        (0, 1, "Mid Left"),
        (1, 1, "Mid Right"),
        (0, 2, "Back Left"),
        (1, 2, "Back Right"),
    ]

    for col, row, label in region_specs:
        rx = int(col * region_w)
        ry = int(row * region_h)
        rw = int((col + 1) * region_w) - rx
        rh = int((row + 1) * region_h) - ry
        color = REGION_COLORS[label]
        rect = Rectangle((rx, ry), rw, rh, facecolor=color, edgecolor="#111827", linewidth=1.8)
        ax_color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
        # Matplotlib handles alpha when drawing; we keep the visual soft.
        overlay[ry:ry + rh, rx:rx + rw] = cv2.addWeighted(
            overlay[ry:ry + rh, rx:rx + rw], 0.72,
            cv2.cvtColor(
                (255 * (cv2.cvtColor(crop[ry:ry + rh, rx:rx + rw], cv2.COLOR_BGR2RGB) * 0 + 1)).astype("uint8"),
                cv2.COLOR_RGB2BGR,
            ), 0.28, 0,
        )
        # Use matplotlib to place labels after conversion; the crop is drawn in the final figure.

    # Draw borders and divider lines using OpenCV for crispness.
    cv2.rectangle(overlay, (0, 0), (cw - 1, ch - 1), (17, 24, 39), 2)
    cv2.line(overlay, (cw // 2, 0), (cw // 2, ch - 1), (17, 24, 39), 2)
    cv2.line(overlay, (0, ch // 3), (cw - 1, ch // 3), (17, 24, 39), 2)
    cv2.line(overlay, (0, 2 * ch // 3), (cw - 1, 2 * ch // 3), (17, 24, 39), 2)

    # Label each cell.
    label_positions = {
        "Front Left": (int(cw * 0.25), int(ch * 0.17)),
        "Front Right": (int(cw * 0.75), int(ch * 0.17)),
        "Mid Left": (int(cw * 0.25), int(ch * 0.50)),
        "Mid Right": (int(cw * 0.75), int(ch * 0.50)),
        "Back Left": (int(cw * 0.25), int(ch * 0.83)),
        "Back Right": (int(cw * 0.75), int(ch * 0.83)),
    }

    for label, (tx, ty) in label_positions.items():
        box_w, box_h = 150, 28
        x0 = max(0, tx - box_w // 2)
        y0 = max(0, ty - box_h // 2)
        x1b = min(cw - 1, x0 + box_w)
        y1b = min(ch - 1, y0 + box_h)
        cv2.rectangle(overlay, (x0, y0), (x1b, y1b), (255, 255, 255), -1)
        cv2.rectangle(overlay, (x0, y0), (x1b, y1b), (17, 24, 39), 1)
        cv2.putText(overlay, label, (x0 + 8, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (17, 24, 39), 2)

    # Add a subtle header.
    cv2.putText(overlay, f"Video 8 | Frame {frame_num}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
    cv2.putText(overlay, f"Video 8 | Frame {frame_num}", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (17, 24, 39), 1)
    cv2.putText(overlay, "Court region classification grid", (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3)
    cv2.putText(overlay, "Court region classification grid", (14, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (17, 24, 39), 1)

    return overlay, (x1, y1, x2, y2)


def create_figure(frame_num: int):
    frame = extract_frame(VIDEO_PATH, frame_num)
    court_detection = detect_court(frame)
    overlay, bounds = draw_overlay(frame, court_detection, frame_num)

    fig, ax = plt.subplots(figsize=(13, 7), dpi=300, facecolor="white")
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title("Court Region Classification Grid on Video 8 Frame", fontsize=14, fontweight="bold")

    x1, y1, x2, y2 = bounds
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, alpha=0))
    ax.text(
        0.5, -0.03,
        "Front = top of the court view | Back = bottom of the court view",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9.5,
        color="#4b5563",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "court_region_grid_video8.png"
    pdf_path = OUTPUT_DIR / "court_region_grid_video8.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"✓ Figure saved: {png_path}")
    print(f"✓ PDF version saved: {pdf_path}")
    print(f"✓ Court keypoints detected: {court_detection.count}")


def main():
    parser = argparse.ArgumentParser(description="Generate court region grid overlay on a video-8 frame")
    parser.add_argument("--frame", type=int, default=13, help="Frame number from video 8")
    args = parser.parse_args()

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    print("Generating court region grid overlay on video 8 frame...")
    print(f"  Video: {VIDEO_PATH}")
    print(f"  Frame: {args.frame}")
    create_figure(args.frame)
    print("\n✓ Court region overlay complete!")


if __name__ == "__main__":
    main()
