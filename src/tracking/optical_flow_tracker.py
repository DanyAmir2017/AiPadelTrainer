"""
Optical Flow Tracker Module
============================

Implements Lucas-Kanade sparse optical flow to track the padel ball
between frames.  Used as a secondary tracking source when the YOLO
detector misses a frame (motion blur, occlusion, etc.).

Pipeline
--------
1. Receive a seed point (previous ball position).
2. Extract good features in a local region around the seed.
3. Run pyramidal LK optical flow on the next frame.
4. Filter by forward-backward consistency check.
5. Return the estimated ball position in the new frame.

Author: Bachelor Thesis Project – GUC
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

from src.utils.config import (
    FEATURE_PARAMS,
    LK_PARAMS,
    OPTICAL_FLOW_SEARCH_RADIUS,
    OPTICAL_FLOW_MIN_QUALITY,
)

logger = logging.getLogger("padel_trainer.optical_flow")


class OpticalFlowTracker:
    """
    Sparse Lucas-Kanade optical flow tracker for ball tracking.

    Maintains the previous grayscale frame and feature points.
    Call ``update()`` each frame with the new frame and optionally
    a fresh seed position (from YOLO).

    Usage
    -----
    >>> of = OpticalFlowTracker()
    >>> of.initialize(prev_gray, seed_point=(320, 240))
    >>> pos = of.update(curr_gray)
    """

    def __init__(self):
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.active = False

        # OpenCV termination criteria for LK
        self.lk_params = dict(
            winSize=LK_PARAMS["winSize"],
            maxLevel=LK_PARAMS["maxLevel"],
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                LK_PARAMS["criteria"][1],
                LK_PARAMS["criteria"][2],
            ),
        )

    # ------------------------------------------------------------------ #
    #  PUBLIC API
    # ------------------------------------------------------------------ #
    def initialize(self, gray_frame: np.ndarray,
                   seed_point: Tuple[float, float]) -> None:
        """
        Reset the tracker with a known ball position.

        Extracts Shi-Tomasi features in a local ROI around *seed_point*.
        """
        self.prev_gray = gray_frame.copy()
        sx, sy = int(seed_point[0]), int(seed_point[1])

        # Define ROI around the seed point
        h, w = gray_frame.shape[:2]
        r = OPTICAL_FLOW_SEARCH_RADIUS
        x1 = max(0, sx - r)
        y1 = max(0, sy - r)
        x2 = min(w, sx + r)
        y2 = min(h, sy + r)

        roi = gray_frame[y1:y2, x1:x2]

        if roi.size == 0:
            self.prev_points = np.array([[[sx, sy]]], dtype=np.float32)
            self.active = True
            return

        # Detect features within the ROI
        local_pts = cv2.goodFeaturesToTrack(roi, **FEATURE_PARAMS)
        if local_pts is not None and len(local_pts) > 0:
            # Shift back to full-frame coordinates
            local_pts[:, 0, 0] += x1
            local_pts[:, 0, 1] += y1
            # Also include the seed point itself
            seed_arr = np.array([[[sx, sy]]], dtype=np.float32)
            self.prev_points = np.vstack([seed_arr, local_pts])
        else:
            self.prev_points = np.array([[[sx, sy]]], dtype=np.float32)

        self.active = True

    # ------------------------------------------------------------------ #
    def update(self, gray_frame: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Track features from ``prev_gray`` to *gray_frame*.

        Returns
        -------
        (x, y) | None
            Estimated ball position, or None if tracking failed.
        """
        if not self.active or self.prev_gray is None or self.prev_points is None:
            return None

        if len(self.prev_points) == 0:
            self.active = False
            return None

        # Forward flow: prev → curr
        next_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_frame, self.prev_points, None, **self.lk_params
        )

        if next_pts is None or status is None:
            self.active = False
            return None

        # Backward flow: curr → prev (consistency check)
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            gray_frame, self.prev_gray, next_pts, None, **self.lk_params
        )

        # Forward-backward error
        if back_pts is not None:
            fb_error = np.linalg.norm(
                self.prev_points.reshape(-1, 2) - back_pts.reshape(-1, 2),
                axis=1,
            )
            fb_mask = fb_error < 2.0  # Pixel threshold for consistency
        else:
            fb_mask = np.ones(len(status), dtype=bool)

        # Combine masks
        good_mask = (status.ravel() == 1) & fb_mask

        if not np.any(good_mask):
            self.active = False
            return None

        good_next = next_pts[good_mask].reshape(-1, 2)

        # Weighted mean – points closer to the seed (first point) get higher weight
        if len(good_next) == 0:
            self.active = False
            return None

        # Use median for robustness against outliers
        cx = float(np.median(good_next[:, 0]))
        cy = float(np.median(good_next[:, 1]))

        # Update state for next call
        self.prev_gray = gray_frame.copy()
        self.prev_points = good_next.reshape(-1, 1, 2).astype(np.float32)

        return (cx, cy)

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Deactivate tracker and clear state."""
        self.prev_gray = None
        self.prev_points = None
        self.active = False
