"""
Generate Court Region Labels on a Trailshot Frame
=================================================

Uses an existing trail snapshot from video 8 and overlays court region labels
(front/mid/back + left/right) directly on the visible court.

This keeps the real system result intact while adding the region grid for
thesis presentation.

Usage:
    python generate_court_region_trailshot_figure.py --input outputs/edge/trail_snapshots/v8_laptop_run/trail_0010s_frame_000300.jpg

Author: Padel Trainer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "PaperLatex" / "figures"

REGION_COLORS = {
    "Front Left": (219, 234, 254),
    "Front Right": (219, 234, 254),
    "Mid Left": (229, 231, 235),
    "Mid Right": (229, 231, 235),
    "Back Left": (203, 213, 225),
    "Back Right": (203, 213, 225),
}

LABEL_STYLE = {
    "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
    "fontScale": 0.72,
    "textColor": (15, 23, 42),
    "textThickness": 2,
    "boxColor": (255, 255, 255),
    "boxAlpha": 0.85,
    "borderColor": (15, 23, 42),
}


def estimate_court_area(frame):
    """Return a stable court crop for the provided trailshot."""
    h, w = frame.shape[:2]
    # The video-8 trailshots are already framed tightly on the court.
    # A fixed crop gives a cleaner overlay than trying to re-detect a court
    # in the presence of many marker annotations.
    x1 = int(w * 0.06)
    y1 = int(h * 0.10)
    x2 = int(w * 0.94)
    y2 = int(h * 0.92)
    return x1, y1, x2, y2


def add_label_box(image, text, center_x, center_y, box_w, box_h):
    x0 = int(center_x - box_w / 2)
    y0 = int(center_y - box_h / 2)
    x1 = int(center_x + box_w / 2)
    y1 = int(center_y + box_h / 2)

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), LABEL_STYLE["boxColor"], -1)
    cv2.addWeighted(overlay, LABEL_STYLE["boxAlpha"], image, 1 - LABEL_STYLE["boxAlpha"], 0, image)
    cv2.rectangle(image, (x0, y0), (x1, y1), LABEL_STYLE["borderColor"], 1)

    text_size, baseline = cv2.getTextSize(text, LABEL_STYLE["fontFace"], LABEL_STYLE["fontScale"], LABEL_STYLE["textThickness"])
    tx = int(center_x - text_size[0] / 2)
    ty = int(center_y + text_size[1] / 2)
    cv2.putText(
        image,
        text,
        (tx, ty),
        LABEL_STYLE["fontFace"],
        LABEL_STYLE["fontScale"],
        LABEL_STYLE["textColor"],
        LABEL_STYLE["textThickness"],
        cv2.LINE_AA,
    )


def overlay_regions(frame):
    x1, y1, x2, y2 = estimate_court_area(frame)
    crop = frame[y1:y2, x1:x2].copy()
    h, w = crop.shape[:2]

    # Draw a subtle translucent tint per zone.
    overlay = crop.copy()
    cell_w = w / 2.0
    cell_h = h / 3.0

    regions = [
        (0, 0, "Front Left"),
        (1, 0, "Front Right"),
        (0, 1, "Mid Left"),
        (1, 1, "Mid Right"),
        (0, 2, "Back Left"),
        (1, 2, "Back Right"),
    ]

    for col, row, label in regions:
        rx = int(col * cell_w)
        ry = int(row * cell_h)
        rw = int((col + 1) * cell_w) - rx
        rh = int((row + 1) * cell_h) - ry
        color = REGION_COLORS[label]

        sub = overlay[ry:ry + rh, rx:rx + rw]
        tint = sub.copy()
        tint[:] = color
        cv2.addWeighted(tint, 0.22, sub, 0.78, 0, sub)

    # Grid lines.
    cv2.rectangle(overlay, (0, 0), (w - 1, h - 1), (17, 24, 39), 2)
    cv2.line(overlay, (w // 2, 0), (w // 2, h - 1), (17, 24, 39), 2)
    cv2.line(overlay, (0, h // 3), (w - 1, h // 3), (17, 24, 39), 2)
    cv2.line(overlay, (0, 2 * h // 3), (w - 1, 2 * h // 3), (17, 24, 39), 2)

    # Labels.
    label_positions = {
        "Front Left": (w * 0.25, h * 0.17),
        "Front Right": (w * 0.75, h * 0.17),
        "Mid Left": (w * 0.25, h * 0.50),
        "Mid Right": (w * 0.75, h * 0.50),
        "Back Left": (w * 0.25, h * 0.83),
        "Back Right": (w * 0.75, h * 0.83),
    }

    for label, (cx, cy) in label_positions.items():
        add_label_box(overlay, label, cx, cy, 190, 34)

    # Footer title bar.
    cv2.rectangle(overlay, (10, 8), (460, 72), (255, 255, 255), -1)
    cv2.rectangle(overlay, (10, 8), (460, 72), (17, 24, 39), 1)
    cv2.putText(overlay, "Video 8 trailshot with court grid", (22, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (17, 24, 39), 2)
    cv2.putText(overlay, "Actual system result + region labels", (22, 59), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (75, 85, 99), 1)

    # Legend.
    legend_x, legend_y = 18, h - 110
    cv2.rectangle(overlay, (legend_x - 8, legend_y - 18), (legend_x + 240, legend_y + 70), (255, 255, 255), -1)
    cv2.rectangle(overlay, (legend_x - 8, legend_y - 18), (legend_x + 240, legend_y + 70), (17, 24, 39), 1)
    cv2.putText(overlay, "Legend", (legend_x, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (17, 24, 39), 2)
    cv2.circle(overlay, (legend_x + 18, legend_y + 28), 6, (255, 0, 255), -1)
    cv2.putText(overlay, "Trail / trajectory", (legend_x + 34, legend_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (17, 24, 39), 1)
    cv2.circle(overlay, (legend_x + 18, legend_y + 50), 6, (0, 215, 255), -1)
    cv2.putText(overlay, "Contact markers", (legend_x + 34, legend_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (17, 24, 39), 1)

    return overlay


def create_figure(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    frame = cv2.imread(str(input_path))
    if frame is None:
        raise RuntimeError(f"Could not read image: {input_path}")

    annotated = overlay_regions(frame)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300, facecolor="white")
    ax.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
    ax.axis("off")
    ax.set_title("Court Region Classification Grid on Actual Video-8 Trailshot", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.02, "The trail snapshot preserves the system output: ball trajectory, contact markers, and detection overlays.",
              ha="center", fontsize=9.5, color="#4b5563")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "court_region_trailshot_video8.png"
    pdf_path = OUTPUT_DIR / "court_region_trailshot_video8.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"✓ Figure saved: {png_path}")
    print(f"✓ PDF version saved: {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Overlay court region labels on a video-8 trail snapshot")
    parser.add_argument(
        "--input",
        type=str,
        default="outputs/edge/trail_snapshots/v8_laptop_run/trail_0010s_frame_000300.jpg",
        help="Path to a trail snapshot or annotated frame",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = PROJECT_ROOT / input_path

    print("Generating court-region overlay on a trailshot...")
    print(f"  Input: {input_path}")
    create_figure(input_path)
    print("\n✓ Trailshot court grid complete!")


if __name__ == "__main__":
    main()
