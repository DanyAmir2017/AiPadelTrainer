"""
Video Utilities Module
======================

Handles video I/O: opening input videos, creating annotated output
writers, FPS calculation, and trajectory CSV saving.

Author: Bachelor Thesis Project – GUC
"""

import csv
import logging
import os
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.utils.config import (
    ANNOTATED_VIDEO_DIR,
    TRAJECTORY_DIR,
    METRICS_DIR,
)

logger = logging.getLogger("padel_trainer.video_utils")


class VideoReader:
    """
    Wrapper around cv2.VideoCapture with metadata.

    Usage
    -----
    >>> reader = VideoReader("input.mp4")
    >>> for frame in reader:
    ...     process(frame)
    >>> reader.release()
    """

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video not found: {path}")

        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_idx = 0

        logger.info(
            "Opened video: %s  (%dx%d, %.1f FPS, %d frames)",
            os.path.basename(path), self.width, self.height,
            self.fps, self.total_frames,
        )

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        ret, frame = self.cap.read()
        if ret:
            self.frame_idx += 1
        return ret, frame

    def __iter__(self):
        while True:
            ret, frame = self.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


class VideoWriter:
    """
    Wrapper around cv2.VideoWriter for annotated output.

    Usage
    -----
    >>> writer = VideoWriter("output.mp4", fps=30.0, size=(1920, 1080))
    >>> writer.write(frame)
    >>> writer.release()
    """

    def __init__(self, filename: str, fps: float, size: Tuple[int, int],
                 output_dir: str = ANNOTATED_VIDEO_DIR):
        os.makedirs(output_dir, exist_ok=True)
        self.path = os.path.join(output_dir, filename)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(self.path, fourcc, fps, size)

        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot create video writer: {self.path}")

        logger.info("Writing output video to: %s", self.path)

    def write(self, frame: np.ndarray) -> None:
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
        logger.info("Output video saved: %s", self.path)


class FPSCounter:
    """Simple wall-clock FPS counter."""

    def __init__(self):
        self._start = time.perf_counter()
        self._frame_count = 0
        self._fps = 0.0
        self._last_update = self._start

    def tick(self) -> float:
        self._frame_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_update
        if elapsed >= 0.5:  # Update every 0.5 s
            self._fps = self._frame_count / (now - self._start)
            self._last_update = now
        return self._fps

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._frame_count

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._start


# ====================================================================== #
#  TRAJECTORY CSV
# ====================================================================== #
def save_trajectory_csv(
    records: list,
    video_name: str,
    output_dir: str = TRAJECTORY_DIR,
) -> str:
    """
    Save ball trajectory to a CSV file.

    Parameters
    ----------
    records : list[dict]
        Each dict should have keys: frame, x, y, confidence, source, speed, region.
    video_name : str
        Base name for the CSV file.
    output_dir : str
        Directory to save the CSV.

    Returns
    -------
    str
        Path to the saved CSV.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_name = os.path.splitext(video_name)[0] + "_trajectory.csv"
    csv_path = os.path.join(output_dir, csv_name)

    fieldnames = ["frame", "x", "y", "confidence", "source", "speed", "region"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k, "") for k in fieldnames})

    logger.info("Trajectory saved: %s  (%d records)", csv_path, len(records))
    return csv_path


# ====================================================================== #
#  METRICS SUMMARY
# ====================================================================== #
def save_metrics(
    stats: dict,
    video_name: str,
    output_dir: str = METRICS_DIR,
) -> str:
    """
    Save evaluation metrics to a text file.

    Parameters
    ----------
    stats : dict
        Dictionary of metric names → values.
    video_name : str
        Base name for the file.
    output_dir : str
        Directory to save the file.

    Returns
    -------
    str
        Path to the saved metrics file.
    """
    os.makedirs(output_dir, exist_ok=True)
    metrics_name = os.path.splitext(video_name)[0] + "_metrics.txt"
    metrics_path = os.path.join(output_dir, metrics_name)

    with open(metrics_path, "w") as f:
        f.write("=" * 50 + "\n")
        f.write("  Padel Trainer – Evaluation Metrics\n")
        f.write("=" * 50 + "\n\n")
        for key, value in stats.items():
            if isinstance(value, float):
                f.write(f"{key:.<35} {value:.2f}\n")
            elif isinstance(value, dict):
                f.write(f"\n{key}:\n")
                for k2, v2 in value.items():
                    f.write(f"  {k2:.<33} {v2}\n")
            else:
                f.write(f"{key:.<35} {value}\n")

    logger.info("Metrics saved: %s", metrics_path)
    return metrics_path
