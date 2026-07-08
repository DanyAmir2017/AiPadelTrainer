"""
Shot Evaluator Module
=====================

Evaluates padel shot placement by mapping the ball's detected position
to regions of the court defined by the detected court keypoints.

Features
--------
• Classifies each ball position into a court region (front/back, left/right).
• Tracks shot distribution over time.
• Computes average ball speed and consistency score.
• Provides per-rally and per-video summary statistics.

Author: Bachelor Thesis Project – GUC
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np

from src.detection.court_detector import CourtDetection

logger = logging.getLogger("padel_trainer.shot_evaluator")


# ====================================================================== #
#  COURT REGION DEFINITIONS
# ====================================================================== #
# Padel court regions (generic, refined once keypoints are mapped):
#
#   ┌─────────┬─────────┐
#   │ BACK-L  │ BACK-R  │   (far from camera)
#   ├─────────┼─────────┤
#   │ MID-L   │ MID-R   │
#   ├─────────┼─────────┤
#   │ FRONT-L │ FRONT-R │   (close to camera)
#   └─────────┴─────────┘
#
REGION_NAMES = [
    "Front-Left", "Front-Right",
    "Mid-Left", "Mid-Right",
    "Back-Left", "Back-Right",
    "Out-of-Court",
]


@dataclass
class ShotRecord:
    """Single shot / ball position record."""
    frame_id: int
    x: float
    y: float
    speed: float              # Pixels/frame
    region: str = "Unknown"
    source: str = "yolo"      # Detection source


@dataclass
class ShotStats:
    """Aggregate statistics for shot evaluation."""
    total_detections: int = 0
    avg_speed: float = 0.0
    max_speed: float = 0.0
    region_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    consistency_score: float = 0.0   # Lower std-dev of landing = more consistent


class ShotEvaluator:
    """
    Evaluates shot placement using court keypoints.

    Basic approach:
    1. Use court keypoints to define a bounding polygon / grid.
    2. For each ball position, classify which region it falls into.
    3. Accumulate statistics over all frames.

    Usage
    -----
    >>> evaluator = ShotEvaluator()
    >>> evaluator.update_court(court_detection)
    >>> evaluator.record_shot(frame_id=10, x=400, y=300, speed=5.0)
    >>> stats = evaluator.get_stats()
    """

    def __init__(self):
        self.shots: List[ShotRecord] = []
        self._court_center: Optional[Tuple[float, float]] = None
        self._court_bounds: Optional[Tuple[float, float, float, float]] = None
        self._court_keypoints: Dict[str, Tuple[float, float]] = {}
        self._court_thirds_y: Optional[Tuple[float, float]] = None

    # ------------------------------------------------------------------ #
    #  COURT GEOMETRY
    # ------------------------------------------------------------------ #
    def update_court(self, court: CourtDetection) -> None:
        """
        Update internal court geometry from the latest court detection.

        Uses detected keypoints to compute:
        - Bounding rectangle
        - Centre line (left / right split)
        - Thirds (front / mid / back split)
        """
        if court.count == 0:
            return

        # Cache keypoint positions
        self._court_keypoints = {
            name: (kp.x, kp.y)
            for name, kp in court.keypoints.items()
        }

        pts = court.as_array()
        if len(pts) < 2:
            return

        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        self._court_bounds = (x_min, y_min, x_max, y_max)
        self._court_center = ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0)

        # Split into thirds vertically
        third = (y_max - y_min) / 3.0
        self._court_thirds_y = (y_min + third, y_min + 2 * third)

    # ------------------------------------------------------------------ #
    def classify_region(self, x: float, y: float) -> str:
        """
        Classify a ball position (x, y) into a court region.

        Returns one of the REGION_NAMES strings.
        """
        if self._court_bounds is None or self._court_center is None:
            return "Unknown"

        x_min, y_min, x_max, y_max = self._court_bounds
        cx = self._court_center[0]

        # Check if inside court bounding box (with margin)
        margin = 30  # pixels
        if (x < x_min - margin or x > x_max + margin or
                y < y_min - margin or y > y_max + margin):
            return "Out-of-Court"

        # Left / Right
        side = "Left" if x < cx else "Right"

        # Front / Mid / Back  (y increases downward in image coordinates)
        if self._court_thirds_y is None:
            depth = "Mid"
        else:
            t1, t2 = self._court_thirds_y
            if y < t1:
                depth = "Back"    # Top of image = far from camera = back
            elif y < t2:
                depth = "Mid"
            else:
                depth = "Front"   # Bottom of image = close to camera

        return f"{depth}-{side}"

    # ------------------------------------------------------------------ #
    #  SHOT RECORDING
    # ------------------------------------------------------------------ #
    def record_shot(self, frame_id: int, x: float, y: float,
                    speed: float = 0.0, source: str = "yolo") -> ShotRecord:
        """Record a single ball detection as a shot."""
        region = self.classify_region(x, y)
        shot = ShotRecord(
            frame_id=frame_id, x=x, y=y,
            speed=speed, region=region, source=source,
        )
        self.shots.append(shot)
        return shot

    # ------------------------------------------------------------------ #
    #  STATISTICS
    # ------------------------------------------------------------------ #
    def get_stats(self) -> ShotStats:
        """Compute aggregate shot statistics."""
        stats = ShotStats()
        if not self.shots:
            return stats

        stats.total_detections = len(self.shots)

        speeds = [s.speed for s in self.shots if s.speed > 0]
        if speeds:
            stats.avg_speed = float(np.mean(speeds))
            stats.max_speed = float(np.max(speeds))

        for s in self.shots:
            stats.region_counts[s.region] += 1

        # Consistency = inverse of positional std-dev (normalised)
        positions = np.array([[s.x, s.y] for s in self.shots])
        if len(positions) > 1:
            std_x = float(np.std(positions[:, 0]))
            std_y = float(np.std(positions[:, 1]))
            avg_std = (std_x + std_y) / 2.0
            # Map to 0-100 score (lower std = higher consistency)
            stats.consistency_score = max(0.0, 100.0 - avg_std)

        return stats

    def get_region_distribution(self) -> Dict[str, float]:
        """Return region distribution as percentages."""
        if not self.shots:
            return {}
        total = len(self.shots)
        counts: Dict[str, int] = defaultdict(int)
        for s in self.shots:
            counts[s.region] += 1
        return {region: count / total * 100 for region, count in counts.items()}

    def reset(self) -> None:
        """Clear all recorded shots."""
        self.shots.clear()
