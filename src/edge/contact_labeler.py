"""
Contact Candidate Labeling Tool

Simple GUI for reviewing rule-based contact candidates frame-by-frame and
marking each one as correct or wrong. Results are saved to CSV for fine-tuning.

Usage:
    python src/edge/contact_labeler.py \
        --video input_videos/Padel_video_5.mp4 \
        --candidates-csv outputs/edge/hit_candidates/v5_after_tune_hit_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

import cv2

try:
    from PIL import Image, ImageTk
except ImportError as exc:
    raise ImportError(
        "Pillow is required for the labeling GUI. Install with: pip install pillow"
    ) from exc


@dataclass
class Candidate:
    frame: int
    second: float
    x: int
    y: int
    rule: str


class ContactLabelerApp:
    def __init__(
        self,
        root: tk.Tk,
        video_path: Path,
        candidates_csv: Path,
        output_csv: Path,
        context_before: int,
        context_after: int,
        autoplay_fps: float,
    ):
        self.root = root
        self.video_path = video_path
        self.candidates_csv = candidates_csv
        self.output_csv = output_csv
        self.context_before = max(0, context_before)
        self.context_after = max(0, context_after)
        self.autoplay_fps = max(1.0, autoplay_fps)

        self.root.title("Contact Candidate Labeler")
        self.root.geometry("1200x760")

        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

        self.candidates = self._load_candidates(candidates_csv)
        if not self.candidates:
            raise ValueError(f"No candidates found in: {candidates_csv}")

        self.labels = self._load_existing_labels(output_csv)
        self.contact_types = self._load_existing_contact_types(output_csv)
        self.current_index = 0
        self.current_photo = None
        self.playing = True
        self.clip_frames = []
        self.clip_frame_numbers = []
        self.clip_cursor = 0
        self.playback_job = None

        self.correct_var = tk.BooleanVar(value=False)
        self.wrong_var = tk.BooleanVar(value=False)
        self.contact_type_var = tk.StringVar(value="")

        self._build_ui()
        self._bind_shortcuts()
        self._show_candidate(0)

    def _load_candidates(self, csv_path: Path) -> list[Candidate]:
        candidates: list[Candidate] = []

        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                candidates.append(
                    Candidate(
                        frame=int(float(row["Frame"])),
                        second=float(row["Second"]),
                        x=int(float(row["X"])),
                        y=int(float(row["Y"])),
                        rule=row["Rule"],
                    )
                )

        return candidates

    def _load_existing_labels(self, csv_path: Path) -> dict[int, str]:
        if not csv_path.exists():
            return {}

        labels: dict[int, str] = {}
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame = int(float(row["Frame"]))
                labels[frame] = row.get("Label", "")
        return labels

    def _load_existing_contact_types(self, csv_path: Path) -> dict[int, str]:
        if not csv_path.exists():
            return {}

        contact_types: dict[int, str] = {}
        with open(csv_path, newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                frame = int(float(row["Frame"]))
                contact_type = (row.get("ContactType") or "").strip()
                contact_types[frame] = contact_type
        return contact_types

    def _build_ui(self):
        container = ttk.Frame(self.root, padding=10)
        container.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(container)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(container, width=320)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        self.info_label = ttk.Label(left, text="", font=("Segoe UI", 11, "bold"))
        self.info_label.pack(anchor=tk.W, pady=(0, 8))

        self.clip_info_label = ttk.Label(left, text="", font=("Segoe UI", 10))
        self.clip_info_label.pack(anchor=tk.W, pady=(0, 8))

        self.canvas = tk.Label(left, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.status_label = ttk.Label(left, text="", font=("Segoe UI", 10))
        self.status_label.pack(anchor=tk.W, pady=(8, 0))

        # Right panel controls
        ttk.Label(right, text="Label", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 8))

        self.correct_chk = ttk.Checkbutton(
            right,
            text="Correct Contact",
            variable=self.correct_var,
            command=self._on_correct_toggle,
        )
        self.correct_chk.pack(anchor=tk.W, pady=2)

        self.wrong_chk = ttk.Checkbutton(
            right,
            text="Wrong Contact",
            variable=self.wrong_var,
            command=self._on_wrong_toggle,
        )
        self.wrong_chk.pack(anchor=tk.W, pady=2)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Label(right, text="Contact Type", font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=(0, 8))
        ttk.Radiobutton(right, text="Racket Hit", value="racket", variable=self.contact_type_var, command=self._on_contact_type_change).pack(anchor=tk.W, pady=1)
        ttk.Radiobutton(right, text="Ground", value="ground", variable=self.contact_type_var, command=self._on_contact_type_change).pack(anchor=tk.W, pady=1)
        ttk.Radiobutton(right, text="Glass", value="glass", variable=self.contact_type_var, command=self._on_contact_type_change).pack(anchor=tk.W, pady=1)
        ttk.Radiobutton(right, text="Ball Out Of Frame", value="out_of_frame", variable=self.contact_type_var, command=self._on_contact_type_change).pack(anchor=tk.W, pady=1)
        ttk.Button(right, text="Clear Contact Type", command=self._clear_contact_type).pack(fill=tk.X, pady=(6, 0))

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        nav_frame = ttk.Frame(right)
        nav_frame.pack(fill=tk.X)

        ttk.Button(nav_frame, text="Previous Candidate", command=self.prev_candidate).pack(fill=tk.X, pady=3)
        ttk.Button(nav_frame, text="Next Candidate", command=self.next_candidate).pack(fill=tk.X, pady=3)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        clip_nav_frame = ttk.Frame(right)
        clip_nav_frame.pack(fill=tk.X)

        ttk.Button(clip_nav_frame, text="⏮ Clip Frame -1", command=self.prev_clip_frame).pack(fill=tk.X, pady=3)
        self.play_pause_btn = ttk.Button(clip_nav_frame, text="⏸ Pause", command=self.toggle_playback)
        self.play_pause_btn.pack(fill=tk.X, pady=3)
        ttk.Button(clip_nav_frame, text="⏭ Clip Frame +1", command=self.next_clip_frame).pack(fill=tk.X, pady=3)

        ttk.Separator(right, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=12)

        ttk.Button(right, text="Save Results", command=self.save_results).pack(fill=tk.X, pady=3)
        ttk.Button(right, text="Save + Exit", command=self.save_and_exit).pack(fill=tk.X, pady=3)

        ttk.Label(
            right,
            text="Shortcuts:\n← Prev Candidate\n→ Next Candidate\n,/. Clip -/+1\nSpace Play/Pause\nC Correct\nW Wrong\n1/2/3/4 Type\nS Save",
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(16, 0))

    def _bind_shortcuts(self):
        self.root.bind("<Left>", lambda _e: self.prev_candidate())
        self.root.bind("<Right>", lambda _e: self.next_candidate())
        self.root.bind("<comma>", lambda _e: self.prev_clip_frame())
        self.root.bind("<period>", lambda _e: self.next_clip_frame())
        self.root.bind("<space>", lambda _e: self.toggle_playback())
        self.root.bind("<Key-c>", lambda _e: self._set_label("correct"))
        self.root.bind("<Key-C>", lambda _e: self._set_label("correct"))
        self.root.bind("<Key-w>", lambda _e: self._set_label("wrong"))
        self.root.bind("<Key-W>", lambda _e: self._set_label("wrong"))
        self.root.bind("<Key-1>", lambda _e: self._set_contact_type("racket"))
        self.root.bind("<Key-2>", lambda _e: self._set_contact_type("ground"))
        self.root.bind("<Key-3>", lambda _e: self._set_contact_type("glass"))
        self.root.bind("<Key-4>", lambda _e: self._set_contact_type("out_of_frame"))
        self.root.bind("<Key-s>", lambda _e: self.save_results())
        self.root.bind("<Key-S>", lambda _e: self.save_results())

    def _read_frame(self, frame_index: int):
        frame_index = max(1, min(frame_index, self.total_frames))
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {frame_index}")
        return frame

    def _draw_candidate_overlay(self, frame, candidate: Candidate):
        out = frame.copy()

        cv2.circle(out, (candidate.x, candidate.y), 12, (0, 0, 255), 2)
        cv2.line(out, (candidate.x - 20, candidate.y), (candidate.x + 20, candidate.y), (0, 0, 255), 2)
        cv2.line(out, (candidate.x, candidate.y - 20), (candidate.x, candidate.y + 20), (0, 0, 255), 2)

        text = f"Candidate @ frame={candidate.frame} ({candidate.second:.3f}s), rule={candidate.rule}"
        cv2.rectangle(out, (10, 8), (1180, 42), (0, 0, 0), -1)
        cv2.putText(out, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return out

    def _prepare_display_frame(self, frame):
        max_w, max_h = 860, 620
        h, w = frame.shape[:2]
        scale = min(max_w / w, max_h / h)
        if scale < 1.0:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        self.current_photo = ImageTk.PhotoImage(image=image)
        self.canvas.configure(image=self.current_photo)

    def _load_clip_for_candidate(self, candidate: Candidate):
        start_frame = max(1, candidate.frame - self.context_before)
        end_frame = min(self.total_frames, candidate.frame + self.context_after)

        self.clip_frames = []
        self.clip_frame_numbers = []

        for frame_idx in range(start_frame, end_frame + 1):
            frame = self._read_frame(frame_idx)
            frame = self._draw_candidate_overlay(frame, candidate)
            self.clip_frames.append(frame)
            self.clip_frame_numbers.append(frame_idx)

        self.clip_cursor = 0
        if candidate.frame in self.clip_frame_numbers:
            self.clip_cursor = self.clip_frame_numbers.index(candidate.frame)

    def _render_current_clip_frame(self):
        if not self.clip_frames:
            return

        frame = self.clip_frames[self.clip_cursor]
        frame_num = self.clip_frame_numbers[self.clip_cursor]
        candidate = self.candidates[self.current_index]

        display = frame.copy()
        cv2.rectangle(display, (10, 46), (520, 82), (0, 0, 0), -1)
        cv2.putText(
            display,
            f"Clip Frame {self.clip_cursor + 1}/{len(self.clip_frames)} | Global Frame {frame_num}",
            (16, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        self._prepare_display_frame(display)
        self.clip_info_label.configure(
            text=(
                f"Clip window: [{self.clip_frame_numbers[0]}..{self.clip_frame_numbers[-1]}] "
                f"around contact frame {candidate.frame}"
            )
        )

    def _playback_tick(self):
        if not self.playing or not self.clip_frames:
            return

        self.clip_cursor = (self.clip_cursor + 1) % len(self.clip_frames)
        self._render_current_clip_frame()

        delay_ms = int(1000 / self.autoplay_fps)
        self.playback_job = self.root.after(delay_ms, self._playback_tick)

    def _restart_playback(self):
        if self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None

        self._render_current_clip_frame()
        if self.playing:
            self._playback_tick()

    def toggle_playback(self):
        self.playing = not self.playing
        self.play_pause_btn.configure(text="▶ Play" if not self.playing else "⏸ Pause")

        if self.playing:
            self._restart_playback()
        elif self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None

    def prev_clip_frame(self):
        if not self.clip_frames:
            return
        self.playing = False
        self.play_pause_btn.configure(text="▶ Play")
        if self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None
        self.clip_cursor = (self.clip_cursor - 1) % len(self.clip_frames)
        self._render_current_clip_frame()

    def next_clip_frame(self):
        if not self.clip_frames:
            return
        self.playing = False
        self.play_pause_btn.configure(text="▶ Play")
        if self.playback_job is not None:
            self.root.after_cancel(self.playback_job)
            self.playback_job = None
        self.clip_cursor = (self.clip_cursor + 1) % len(self.clip_frames)
        self._render_current_clip_frame()

    def _show_candidate(self, index: int):
        self.current_index = max(0, min(index, len(self.candidates) - 1))
        candidate = self.candidates[self.current_index]

        self._load_clip_for_candidate(candidate)
        self._restart_playback()

        self.info_label.configure(
            text=(
                f"Candidate {self.current_index + 1}/{len(self.candidates)} | "
                f"frame={candidate.frame} | time={candidate.second:.3f}s | "
                f"x={candidate.x}, y={candidate.y} | rule={candidate.rule}"
            )
        )

        existing = self.labels.get(candidate.frame, "")
        self.correct_var.set(existing == "correct")
        self.wrong_var.set(existing == "wrong")
        self.contact_type_var.set(self.contact_types.get(candidate.frame, ""))

        reviewed = sum(1 for c in self.candidates if self.labels.get(c.frame) in ("correct", "wrong"))
        typed = sum(1 for c in self.candidates if (self.contact_types.get(c.frame) or "").strip() != "")
        self.status_label.configure(text=f"Reviewed: {reviewed}/{len(self.candidates)} | Contact type set: {typed}/{len(self.candidates)}")

    def _set_label(self, label: str):
        candidate = self.candidates[self.current_index]

        if label == "correct":
            self.correct_var.set(True)
            self.wrong_var.set(False)
            self.labels[candidate.frame] = "correct"
        elif label == "wrong":
            self.correct_var.set(False)
            self.wrong_var.set(True)
            self.labels[candidate.frame] = "wrong"

        self._refresh_status()

    def _set_contact_type(self, contact_type: str):
        candidate = self.candidates[self.current_index]
        self.contact_type_var.set(contact_type)
        self.contact_types[candidate.frame] = contact_type
        self._refresh_status()

    def _on_contact_type_change(self):
        candidate = self.candidates[self.current_index]
        contact_type = (self.contact_type_var.get() or "").strip()
        if contact_type:
            self.contact_types[candidate.frame] = contact_type
        else:
            self.contact_types.pop(candidate.frame, None)
        self._refresh_status()

    def _clear_contact_type(self):
        candidate = self.candidates[self.current_index]
        self.contact_type_var.set("")
        self.contact_types.pop(candidate.frame, None)
        self._refresh_status()

    def _refresh_status(self):
        reviewed = sum(1 for c in self.candidates if self.labels.get(c.frame) in ("correct", "wrong"))
        typed = sum(1 for c in self.candidates if (self.contact_types.get(c.frame) or "").strip() != "")
        self.status_label.configure(text=f"Reviewed: {reviewed}/{len(self.candidates)} | Contact type set: {typed}/{len(self.candidates)}")

    def _on_correct_toggle(self):
        candidate = self.candidates[self.current_index]
        if self.correct_var.get():
            self.wrong_var.set(False)
            self.labels[candidate.frame] = "correct"
        else:
            if self.labels.get(candidate.frame) == "correct":
                self.labels.pop(candidate.frame, None)
        self._refresh_status()

    def _on_wrong_toggle(self):
        candidate = self.candidates[self.current_index]
        if self.wrong_var.get():
            self.correct_var.set(False)
            self.labels[candidate.frame] = "wrong"
        else:
            if self.labels.get(candidate.frame) == "wrong":
                self.labels.pop(candidate.frame, None)
        self._refresh_status()

    def prev_candidate(self):
        self._show_candidate(self.current_index - 1)

    def next_candidate(self):
        self._show_candidate(self.current_index + 1)

    def save_results(self):
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(self.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "Frame",
                "Second",
                "X",
                "Y",
                "Rule",
                "Label",
                "ContactType",
                "ReviewedAt",
            ])

            now = datetime.now().isoformat(timespec="seconds")
            for candidate in self.candidates:
                writer.writerow([
                    candidate.frame,
                    f"{candidate.second:.3f}",
                    candidate.x,
                    candidate.y,
                    candidate.rule,
                    self.labels.get(candidate.frame, ""),
                    self.contact_types.get(candidate.frame, ""),
                    now,
                ])

        messagebox.showinfo("Saved", f"Results saved to:\n{self.output_csv}")

    def save_and_exit(self):
        self.save_results()
        self.cap.release()
        self.root.destroy()


def parse_args():
    parser = argparse.ArgumentParser(description="Label contact candidates with a simple GUI")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--candidates-csv", required=True, help="Hit candidates CSV path")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output labels CSV path (default: outputs/edge/labels/<candidates_name>_labels.csv)",
    )
    parser.add_argument(
        "--context-before",
        type=int,
        default=6,
        help="Number of frames to show before candidate frame in clip preview",
    )
    parser.add_argument(
        "--context-after",
        type=int,
        default=8,
        help="Number of frames to show after candidate frame in clip preview",
    )
    parser.add_argument(
        "--autoplay-fps",
        type=float,
        default=8.0,
        help="Clip autoplay speed in frames per second",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    video_path = Path(args.video)
    candidates_csv = Path(args.candidates_csv)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Candidates CSV not found: {candidates_csv}")

    if args.output_csv:
        output_csv = Path(args.output_csv)
    else:
        output_csv = candidates_csv.parent.parent / "labels" / f"{candidates_csv.stem}_labels.csv"

    root = tk.Tk()
    app = ContactLabelerApp(
        root,
        video_path,
        candidates_csv,
        output_csv,
        context_before=args.context_before,
        context_after=args.context_after,
        autoplay_fps=args.autoplay_fps,
    )

    def on_close():
        if messagebox.askyesno("Exit", "Save before exit?"):
            app.save_results()
        app.cap.release()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
