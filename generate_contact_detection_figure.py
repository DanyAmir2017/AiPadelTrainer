"""
Generate Contact Detection Example Figure for Thesis
====================================================

Creates a thesis-ready figure for the contact detection engine using a
confirmed ground-collision event from video 8.

Figure layout:
- Left: trajectory curve with before/after velocity arrows and highlighted contact frame
- Right: cropped contact frame from the source video

Usage:
    python generate_contact_detection_figure.py
    python generate_contact_detection_figure.py --frame 757

Author: Padel Trainer
"""

import argparse
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


PROJECT_ROOT = Path(__file__).resolve().parent
VIDEO_PATH = PROJECT_ROOT / "input_videos" / "Padel_video_8.mp4"
TRAJECTORY_CSV = PROJECT_ROOT / "outputs" / "edge" / "csv" / "v8_laptop_ground_only_tol2_trajectory.csv"
CANDIDATE_CSV = PROJECT_ROOT / "outputs" / "edge" / "hit_candidates" / "v8_laptop_ground_only_tol2_hit_candidates.csv"
OUTPUT_DIR = PROJECT_ROOT / "PaperLatex" / "figures"


def load_accepted_ground_contacts(csv_path: Path):
    contacts = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if (row.get("ContactType") or "").strip().lower() != "ground":
                continue
            if (row.get("Decision") or "").strip().lower() != "accept":
                continue
            try:
                contacts.append({
                    "frame": int(float(row["Frame"])),
                    "second": float(row["Second"]),
                    "x": float(row["X"]),
                    "y": float(row["Y"]),
                    "rule": (row.get("Rule") or "").strip(),
                    "score": float(row.get("Score") or 0.0),
                })
            except Exception:
                continue
    return contacts


def load_trajectory(csv_path: Path):
    trajectory = {}
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                frame = int(float(row["Frame"]))
                x = float(row["X"])
                y = float(row["Y"])
                source = (row.get("Source") or "").strip()
            except Exception:
                continue
            if x < 0 or y < 0:
                continue
            trajectory[frame] = {"x": x, "y": y, "source": source}
    return trajectory


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


def crop_around_point(frame, x, y, half_width=220, half_height=170):
    h, w = frame.shape[:2]
    x1 = max(0, int(x - half_width))
    y1 = max(0, int(y - half_height))
    x2 = min(w, int(x + half_width))
    y2 = min(h, int(y + half_height))
    return frame[y1:y2, x1:x2].copy(), x1, y1


def choose_contact_frame(contacts, requested_frame=None):
    if requested_frame is not None:
        for contact in contacts:
            if contact["frame"] == requested_frame:
                return contact
        raise ValueError(f"Frame {requested_frame} is not a confirmed accepted ground contact in {CANDIDATE_CSV}")

    if not contacts:
        raise RuntimeError(f"No accepted ground contacts found in {CANDIDATE_CSV}")

    # Prefer the earliest accepted ground collision: it gives the clearest local reversal in video 8.
    return sorted(contacts, key=lambda item: item["frame"])[0]


def draw_contact_figure(contact, trajectory):
    frame_num = contact["frame"]
    x = contact["x"]
    y = contact["y"]

    # Choose a local window around the collision.
    before_frames = [f for f in range(frame_num - 5, frame_num) if f in trajectory]
    after_frames = [f for f in range(frame_num + 1, frame_num + 6) if f in trajectory]
    window_frames = before_frames + [frame_num] + after_frames
    points = [(f, trajectory[f]["x"], trajectory[f]["y"], trajectory[f]["source"]) for f in window_frames]

    if len(points) < 3:
        raise RuntimeError(f"Not enough trajectory points around frame {frame_num} to draw the figure")

    frame = extract_frame(VIDEO_PATH, frame_num)
    crop, crop_x1, crop_y1 = crop_around_point(frame, x, y)

    # Prepare figure.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300, facecolor="white")
    fig.suptitle("Contact Detection Example: Ground Collision and Velocity Reversal", fontsize=14, fontweight="bold")

    # Left panel: trajectory plot.
    ax = axes[0]
    ax.set_facecolor("white")

    xs = [p[1] for p in points]
    ys = [p[2] for p in points]
    frames = [p[0] for p in points]

    # Curve + markers.
    ax.plot(xs, ys, color="#6b7280", linewidth=2.0, alpha=0.9)
    ax.scatter(xs, ys, s=22, color="#1f2937", zorder=3)

    # Color the pre/post segments.
    pre_points = [(f, px, py) for (f, px, py, _) in points if f <= frame_num]
    post_points = [(f, px, py) for (f, px, py, _) in points if f >= frame_num]

    if len(pre_points) >= 2:
        ax.plot([p[1] for p in pre_points], [p[2] for p in pre_points], color="#1d4ed8", linewidth=3.0)
    if len(post_points) >= 2:
        ax.plot([p[1] for p in post_points], [p[2] for p in post_points], color="#16a34a", linewidth=3.0)

    # Velocity arrows using the last step before and first step after contact.
    if len(pre_points) >= 2:
        p0 = pre_points[-2]
        p1 = pre_points[-1]
        arrow_before = FancyArrowPatch(
            (p0[1], p0[2]), (p1[1], p1[2]),
            arrowstyle="->,head_width=0.35,head_length=0.45",
            mutation_scale=18,
            linewidth=3,
            color="#dc2626",
        )
        ax.add_patch(arrow_before)
        ax.text((p0[1] + p1[1]) / 2, (p0[2] + p1[2]) / 2 - 18, "before bounce",
                fontsize=9, color="#dc2626", ha="center")

    if len(post_points) >= 2:
        p0 = post_points[0]
        p1 = post_points[1]
        arrow_after = FancyArrowPatch(
            (p0[1], p0[2]), (p1[1], p1[2]),
            arrowstyle="->,head_width=0.35,head_length=0.45",
            mutation_scale=18,
            linewidth=3,
            color="#059669",
        )
        ax.add_patch(arrow_after)
        ax.text((p0[1] + p1[1]) / 2 + 8, (p0[2] + p1[2]) / 2 + 18, "after bounce",
                fontsize=9, color="#059669", ha="center")

    # Contact point.
    ax.scatter([x], [y], s=120, color="#f59e0b", edgecolor="black", linewidth=1.2, zorder=5)
    ax.annotate(
        f"Contact frame {frame_num}",
        xy=(x, y), xytext=(x + 18, y - 35),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="black", linewidth=1.2),
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#111827", linewidth=0.8),
    )

    # Styling.
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    ax.set_title("Trajectory curve with velocity reversal", fontsize=12)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.2)
    ax.set_aspect("equal", adjustable="box")

    if xs and ys:
        x_pad = 40
        y_pad = 40
        ax.set_xlim(min(xs) - x_pad, max(xs) + x_pad)
        ax.set_ylim(max(ys) + y_pad, min(ys) - y_pad)

    # Right panel: contact-frame crop.
    ax2 = axes[1]
    ax2.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    ax2.axis("off")
    ax2.set_title("Source frame crop", fontsize=12)
    ax2.scatter([x - crop_x1], [y - crop_y1], s=120, facecolors="none", edgecolors="#f59e0b", linewidths=2.0)
    ax2.annotate(
        "ground collision",
        xy=(x - crop_x1, y - crop_y1),
        xytext=(x - crop_x1 + 20, y - crop_y1 - 30),
        color="#111827",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#111827", linewidth=1.0),
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#111827", linewidth=0.8),
    )

    score_text = f"rule: {contact['rule']} | score: {contact['score']:.3f}"
    fig.text(0.5, 0.03, score_text, ha="center", fontsize=9, color="#374151")

    plt.tight_layout(rect=(0, 0.05, 1, 0.95))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / "contact_detection_example_v8.png"
    pdf_path = OUTPUT_DIR / "contact_detection_example_v8.pdf"
    plt.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"✓ Figure saved: {png_path}")
    print(f"✓ PDF version saved: {pdf_path}")
    print(f"✓ Using accepted ground contact frame {frame_num} from video 8")


def main():
    parser = argparse.ArgumentParser(description="Generate contact detection example figure for video 8")
    parser.add_argument("--frame", type=int, default=None, help="Accepted ground-contact frame to use")
    args = parser.parse_args()

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")
    if not TRAJECTORY_CSV.exists():
        raise FileNotFoundError(f"Trajectory CSV not found: {TRAJECTORY_CSV}")
    if not CANDIDATE_CSV.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {CANDIDATE_CSV}")

    contacts = load_accepted_ground_contacts(CANDIDATE_CSV)
    contact = choose_contact_frame(contacts, args.frame)
    trajectory = load_trajectory(TRAJECTORY_CSV)

    print("Generating contact detection example figure...")
    print(f"  Video: {VIDEO_PATH}")
    print(f"  Contact frame: {contact['frame']}")
    print(f"  Contact type: {contact['rule']} / ground")
    print(f"  Output dir: {OUTPUT_DIR}")

    draw_contact_figure(contact, trajectory)
    print("\n✓ Contact detection figure complete!")


if __name__ == "__main__":
    main()
