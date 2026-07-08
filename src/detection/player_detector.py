"""
Player Detector Module
======================

Detects padel players using a custom-trained YOLO model.
Returns bounding boxes for all detected persons per frame.

Author: Bachelor Thesis Project – GUC
"""

import logging
from typing import List
from dataclasses import dataclass

import numpy as np
from ultralytics import YOLO

from src.utils.config import (
    PLAYER_MODEL_PATH,
    PLAYER_CONFIDENCE,
    PLAYER_CLASS_ID,
    MODEL_DEVICE,
    MODEL_IOU_THRESHOLD,
    INFERENCE_SIZE,
    USE_AMP,
)

logger = logging.getLogger("padel_trainer.player_detector")


@dataclass
class PlayerDetection:
    """Result container for a player detection."""
    x1: float          # Top-left x
    y1: float          # Top-left y
    x2: float          # Bottom-right x
    y2: float          # Bottom-right y
    confidence: float


class PlayerDetector:
    """
    Wraps a custom YOLO model for player detection.

    Usage
    -----
    >>> detector = PlayerDetector()
    >>> players = detector.detect(frame)
    >>> for p in players:
    ...     print(f"Player at ({p.x1:.0f},{p.y1:.0f})-({p.x2:.0f},{p.y2:.0f})")
    """

    def __init__(self, model_path: str = PLAYER_MODEL_PATH):
        logger.info("Loading player detection model from %s", model_path)
        self.model = YOLO(model_path)
        self.model.to(MODEL_DEVICE if isinstance(MODEL_DEVICE, str) else "cuda:0")
        logger.info("Player detector ready (device=%s)", MODEL_DEVICE)

    def detect(self, frame: np.ndarray) -> List[PlayerDetection]:
        """
        Run inference on *frame* and return all player detections.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV format).

        Returns
        -------
        list[PlayerDetection]
            All detected players, sorted by confidence (highest first).
        """
        results = self.model.predict(
            frame,
            conf=PLAYER_CONFIDENCE,
            iou=MODEL_IOU_THRESHOLD,
            imgsz=INFERENCE_SIZE,
            device=MODEL_DEVICE,
            verbose=False,
            half=USE_AMP and MODEL_DEVICE != "cpu",
        )

        if not results or len(results[0].boxes) == 0:
            return []

        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu())
            detections.append(
                PlayerDetection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf)
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
