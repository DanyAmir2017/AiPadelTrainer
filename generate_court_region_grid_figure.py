"""
Generate Court Region Classification Grid for Thesis
====================================================

Creates a minimalistic academic diagram dividing the padel court into
front/mid/back and left/right regions.

Usage:
    python generate_court_region_grid_figure.py

Author: Padel Trainer
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "PaperLatex" / "figures"


def add_region(ax, x, y, w, h, label, facecolor, edgecolor="#1f2937", fontsize=12):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01",
        linewidth=1.6,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight="bold",
        color="#0f172a",
    )


def create_figure(output_path: Path):
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300, facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title and subtitle
    ax.text(6, 7.55, "Court Region Classification Grid", ha="center", va="center",
            fontsize=16, fontweight="bold", color="#111827")
    ax.text(6, 7.15, "Padel court divided into front, mid, and back regions", ha="center", va="center",
            fontsize=10.5, color="#4b5563")

    # Court outer boundary
    court_x, court_y, court_w, court_h = 2.0, 1.0, 8.0, 5.4
    outer = FancyBboxPatch(
        (court_x, court_y),
        court_w,
        court_h,
        boxstyle="round,pad=0.02",
        linewidth=2.2,
        edgecolor="#111827",
        facecolor="#f8fafc",
    )
    ax.add_patch(outer)

    # Internal grid positions
    col_w = court_w / 2.0
    row_h = court_h / 3.0

    colors = {
        "front": "#dbeafe",
        "mid": "#e5e7eb",
        "back": "#cbd5e1",
    }

    # Regions: top to bottom = front, mid, back
    regions = [
        (court_x, court_y + 2 * row_h, col_w, row_h, "Front Left", colors["front"]),
        (court_x + col_w, court_y + 2 * row_h, col_w, row_h, "Front Right", colors["front"]),
        (court_x, court_y + row_h, col_w, row_h, "Mid Left", colors["mid"]),
        (court_x + col_w, court_y + row_h, col_w, row_h, "Mid Right", colors["mid"]),
        (court_x, court_y, col_w, row_h, "Back Left", colors["back"]),
        (court_x + col_w, court_y, col_w, row_h, "Back Right", colors["back"]),
    ]

    for x, y, w, h, label, color in regions:
        add_region(ax, x, y, w, h, label, color)

    # Grid lines
    ax.plot([court_x + col_w, court_x + col_w], [court_y, court_y + court_h], color="#111827", linewidth=1.8)
    ax.plot([court_x, court_x + court_w], [court_y + row_h, court_y + row_h], color="#111827", linewidth=1.8)
    ax.plot([court_x, court_x + court_w], [court_y + 2 * row_h, court_y + 2 * row_h], color="#111827", linewidth=1.8)

    # Orientation labels
    ax.text(1.05, 3.7, "Net / Front", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#111827")
    ax.text(10.95, 3.7, "Baseline / Back", rotation=90, ha="center", va="center",
            fontsize=10, fontweight="bold", color="#111827")
    ax.text(6.0, 0.55, "Left / Right split used for regional shot and bounce analysis", ha="center",
            va="center", fontsize=9.5, color="#4b5563")

    # Small annotation describing use
    ax.text(6.0, 6.45, "Region labels used for court-based event classification", ha="center",
            va="center", fontsize=9.5, color="#374151",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#9ca3af", linewidth=0.8))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    png_path = output_path
    pdf_path = output_path.with_suffix(".pdf")
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"✓ Figure saved: {png_path}")
    print(f"✓ PDF version saved: {pdf_path}")


def main():
    output_path = OUTPUT_DIR / "court_region_grid.png"
    print("Generating court region classification grid...")
    print(f"  Output: {output_path}")
    create_figure(output_path)
    print("\n✓ Court region figure complete!")


if __name__ == "__main__":
    main()
