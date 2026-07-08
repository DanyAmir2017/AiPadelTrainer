"""
Generate Court Region Labels on an Annotated Video-8 Frame
=========================================================

Extracts a frame from the annotated video 8 output so the image already
contains the ball bounding box, trajectory trail, and contact markers.
Then overlays the six court region labels directly on the court.

Usage:
    python generate_court_region_annotatedframe_figure.py --frame 300

Author: Padel Trainer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "outputs" / "edge" / "annotated_videos" / "v8_laptop_run_annotated.mp4"
OUTPUT_DIR = PROJECT_ROOT / "PaperLatex" / "figures"

REGION_COLORS = {
    "Front Left": (219, 234, 254),
    "Front Right": (219, 234, 254),
    "Mid Left": (229, 231, 235),
    "Mid Right": (229, 231, 235),
    "Back Left": (203, 213, 225),
    "Back Right": (203, 213, 225),
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


def estimate_court_area(frame):
    h, w = frame.shape[:2]
    # The annotated broadcast frame already shows the court clearly.
    # Use a consistent crop that focuses the visible court while keeping the overlay intact.
    return int(w * 0.05), int(h * 0.07), int(w * 0.95), int(h * 0.93)


def add_label_box(image, text, center_x, center_y, box_w, box_h):
    x0 = int(center_x - box_w / 2)
    y0 = int(center_y - box_h / 2)
    x1 = int(center_x + box_w / 2)
    y1 = int(center_y + box_h / 2)

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.84, image, 0.16, 0, image)
    cv2.rectangle(image, (x0, y0), (x1, y1), (17, 24, 39), 1)

    cv2.putText(
        image,
        text,
        (x0 + 10, y0 + int(box_h * 0.68)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (17, 24, 39),
        2,
        cv2.LINE_AA,
    )


def overlay_regions(frame):
    x1, y1, x2, y2 = estimate_court_area(frame)
    crop = frame[y1:y2, x1:x2].copy()
    h, w = crop.shape[:2]

    # Semi-transparent region fills.
    cell_w = w / 2.0
    cell_h = h / 3.0
    for col, row, label in [
        (0, 0, "Front Left"),
        (1, 0, "Front Right"),
        (0, 1, "Mid Left"),
        (1, 1, "Mid Right"),
        (0, 2, "Back Left"),
        (1, 2, "Back Right"),
    ]:
        rx = int(col * cell_w)
        ry = int(row * cell_h)
        rw = int((col + 1) * cell_w) - rx
        rh = int((row + 1) * cell_h) - ry
        color = REGION_COLORS[label]
        sub = crop[ry:ry + rh, rx:rx + rw]
        tint = sub.copy()
        tint[:] = color
        cv2.addWeighted(tint, 0.18, sub, 0.82, 0, sub)

    # Grid lines.
    cv2.rectangle(crop, (0, 0), (w - 1, h - 1), (17, 24, 39), 2)
    cv2.line(crop, (w // 2, 0), (w // 2, h - 1), (17, 24, 39), 2)
    cv2.line(crop, (0, h // 3), (w - 1, h // 3), (17, 24, 39), 2)
    cv2.line(crop, (0, 2 * h // 3), (w - 1, 2 * h // 3), (17, 24, 39), 2)

    label_positions = {
        "Front Left": (w * 0.25, h * 0.17),
        "Front Right": (w * 0.75, h * 0.17),
        "Mid Left": (w * 0.25, h * 0.50),
        "Mid Right": (w * 0.75, h * 0.50),
        "Back Left": (w * 0.25, h * 0.83),
        "Back Right": (w * 0.75, h * 0.83),
    }
    for label, (cx, cy) in label_positions.items():
        add_label_box(crop, label, cx, cy, 190, 34)

    # Small title box.
    cv2.rectangle(crop, (14, 14), (520, 78), (255, 255, 255), -1)
    cv2.rectangle(crop, (14, 14), (520, 78), (17, 24, 39), 1)
    cv2.putText(crop, "Video 8 annotated frame with court grid", (28, 41), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (17, 24, 39), 2)
    cv2.putText(crop, "Ball box, trail, and contact markers preserved", (28, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (75, 85, 99), 1)

    return crop


def create_figure(frame_num: int):
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Annotated video not found: {VIDEO_PATH}")

    frame = extract_frame(VIDEO_PATH, frame_num)
    annotated = overlay_regions(frame)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300, facecolor="white")
    ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title("Court Region Classification Grid on Annotated Video-8 Frame", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "This version uses the actual annotated system output so the ball bounding box and trajectory trail remain visible.",
              ha="center", fontsize=9.5, color="#4b5563")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "court_region_annotated_video8.png"
    pdf_path = OUTPUT_DIR / "court_region_annotated_video8.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"✓ Figure saved: {png_path}")
    print(f"✓ PDF version saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Overlay court region labels on an annotated video-8 frame")
    parser.add_argument("--frame", type=int, default=300, help="Frame number in the annotated video")
    args = parser.parse_args()

    print("Generating court-region overlay on annotated video 8 frame...")
    print(f"  Video: {VIDEO_PATH}")
    print(f"  Frame: {args.frame}")
    create_figure(args.frame)
    print("\n✓ Annotated frame court grid complete!")


if __name__ == "__main__":
    main()
