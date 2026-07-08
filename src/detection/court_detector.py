"""
Court Detector Module
=====================

Detects padel court keypoints (10 labelled corners / intersections)
using a custom-trained YOLO detector.  The 10 keypoints are used to
define the court geometry for shot evaluation and visualization.

Court keypoint classes:
    0: p1   1: p10  2: p2   3: p3   4: p4
    5: p5   6: p6   7: p7   8: p8   9: p9

Author: Bachelor Thesis Project – GUC
"""

import logging
from typing import Dict, Optional
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

from src.utils.config import (
    COURT_MODEL_PATH,
    COURT_CONFIDENCE,
    COURT_KEYPOINT_NAMES,
    MODEL_DEVICE,
    MODEL_IOU_THRESHOLD,
    INFERENCE_SIZE,
    USE_AMP,
)

logger = logging.getLogger("padel_trainer.court_detector")


@dataclass
class CourtKeypoint:
    """A single court keypoint detection."""
    name: str            # Keypoint label (e.g. "p1")
    class_id: int        # YOLO class index
    x: float             # Centre-x
    y: float             # Centre-y
    confidence: float


@dataclass
class CourtDetection:
    """All detected court keypoints for one frame."""
    keypoints: Dict[str, CourtKeypoint] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.keypoints)

    def get(self, name: str) -> Optional[CourtKeypoint]:
        return self.keypoints.get(name)

    def as_array(self) -> np.ndarray:
        """Return (N, 2) array of detected keypoint coordinates, ordered by name."""
        ordered = sorted(self.keypoints.values(), key=lambda kp: kp.name)
        if not ordered:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([[kp.x, kp.y] for kp in ordered], dtype=np.float32)


class CourtDetector:
    """
    Wraps a custom YOLO model for court keypoint detection.

    Each keypoint is a separate YOLO class (bounding-box centre = keypoint
    location).  The detector returns the highest-confidence detection for
    each keypoint class.

    Usage
    -----
    >>> detector = CourtDetector()
    >>> court = detector.detect(frame)
    >>> if court.count > 0:
    ...     for name, kp in court.keypoints.items():
    ...         print(f"{name}: ({kp.x:.0f}, {kp.y:.0f})")
    """

    def __init__(self, model_path: str = COURT_MODEL_PATH):
        logger.info("Loading court detection model from %s", model_path)
        self.model = YOLO(model_path)
        self.model.to(MODEL_DEVICE if isinstance(MODEL_DEVICE, str) else "cuda:0")
        logger.info("Court detector ready (%d keypoint classes)", len(COURT_KEYPOINT_NAMES))

    def detect(self, frame: np.ndarray) -> CourtDetection:
        """
        Detect court keypoints in *frame*.

        Returns the highest-confidence detection per keypoint class.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV format).

        Returns
        -------
        CourtDetection
            Contains a dict of detected keypoints keyed by name.
        """
        results = self.model.predict(
            frame,
            conf=COURT_CONFIDENCE,
            iou=MODEL_IOU_THRESHOLD,
            imgsz=INFERENCE_SIZE,
            device=MODEL_DEVICE,
            verbose=False,
            half=USE_AMP and MODEL_DEVICE != "cpu",
        )

        court = CourtDetection()

        if not results or len(results[0].boxes) == 0:
            return court

        # Group detections by class, keep highest confidence per class
        best_per_class: Dict[int, tuple] = {}  # class_id -> (conf, x, y)
        for box in results[0].boxes:
            cls_id = int(box.cls[0].cpu())
            conf = float(box.conf[0].cpu())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            if cls_id not in best_per_class or conf > best_per_class[cls_id][0]:
                best_per_class[cls_id] = (conf, cx, cy)

        for cls_id, (conf, cx, cy) in best_per_class.items():
            name = COURT_KEYPOINT_NAMES.get(cls_id, f"unknown_{cls_id}")
            kp = CourtKeypoint(
                name=name, class_id=cls_id,
                x=cx, y=cy, confidence=conf,
            )
            court.keypoints[name] = kp

        return court
