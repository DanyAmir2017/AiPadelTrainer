"""
Ball Detector Module
====================

Detects the padel ball using a custom-trained YOLO model.
Returns the best detection (highest confidence) per frame.

Author: Bachelor Thesis Project – GUC
"""

import logging
from typing import Optional
from dataclasses import dataclass

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.config import (
    BALL_MODEL_PATH,
    BALL_CONFIDENCE,
    BALL_CLASS_ID,
    MODEL_DEVICE,
    MODEL_IOU_THRESHOLD,
    INFERENCE_SIZE,
    USE_AMP,
)

logger = logging.getLogger("padel_trainer.ball_detector")


@dataclass
class BallDetection:
    """Result container for a single ball detection."""
    x: float            # Centre-x in pixels
    y: float            # Centre-y in pixels
    w: float            # Box width
    h: float            # Box height
    confidence: float   # Detection confidence
    source: str = "yolo"  # Detection source label


class BallDetector:
    """
    Wraps a custom YOLO model for padel ball detection.

    Includes a **static-point filter** that learns which positions in the
    frame consistently produce detections (logos, scoreboards, etc.) and
    suppresses them so only the real moving ball is returned.

    Usage
    -----
    >>> detector = BallDetector()
    >>> detection = detector.detect(frame)
    >>> if detection:
    ...     print(f"Ball at ({detection.x:.0f}, {detection.y:.0f})")
    """

    # Static-point filter parameters (DISABLED - player-overlap filter handles false positives)
    STATIC_RADIUS = 15          # Pixels – detections within this radius are "same spot"
    STATIC_HISTORY = 30         # Frames of history to keep
    STATIC_THRESHOLD = 0.7      # If a spot appears in ≥70 % of recent frames → static
    ENABLE_STATIC_FILTER = False  # Disabled - player filter handles yellow shoes
    
    # Movement filter parameters (immediate filtering from frame 1)
    MIN_MOVEMENT = 10.0          # Minimum pixels movement from any recent position
    MOVEMENT_HISTORY = 10        # Check against last 10 positions
    ENABLE_MOVEMENT_FILTER = False  # Disabled to see all detected objects
    
    # Player-overlap filter parameters
    PLAYER_OVERLAP_FILTER = True   # Filter out detections in player regions
    PLAYER_LEG_RATIO = 0.25          # Focus on bottom 25% of player box (feet only)
    
    # Court boundary filter parameters
    COURT_BOUNDARY_FILTER = True     # Filter out detections outside court region
    COURT_MARGIN = 100                # Pixels margin outside court bounds
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) parameters
    ENABLE_CLAHE = False              # Apply CLAHE to normalize lighting (DISABLED - hurt performance)
    CLAHE_CLIP_LIMIT = 2.0            # Threshold for contrast limiting (1.0-4.0)
    CLAHE_TILE_SIZE = 8               # Size of grid for histogram equalization
    
    # Region-adaptive confidence parameters (DISABLED - didn't help)
    ENABLE_REGION_ADAPTIVE_CONFIDENCE = False  # Use lower thresholds for bottom-half
    NORMAL_CONFIDENCE = 0.01          # Standard threshold for upper-half (Back)
    LOWERED_CONFIDENCE = 0.005        # Relaxed threshold for bottom-half (Front/Mid)

    def __init__(self, model_path: str = BALL_MODEL_PATH):
        logger.info("Loading ball detection model from %s", model_path)
        self.model = YOLO(model_path)
        self.model.to(MODEL_DEVICE if isinstance(MODEL_DEVICE, str) else "cuda:0")
        logger.info("Ball detector ready (device=%s)", MODEL_DEVICE)

        # Static-point filter state
        # Each entry: (cx, cy, hit_count, total_frames_tracked)
        self._static_spots: list[list] = []   # [[cx, cy, hits, tracked]]
        self._frame_count = 0
        
        # Movement filter state - track recent detection positions
        self._recent_positions: list[tuple[float, float]] = []
        
        # Initialize CLAHE for lighting normalization
        if self.ENABLE_CLAHE:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.CLAHE_CLIP_LIMIT, 
                tileGridSize=(self.CLAHE_TILE_SIZE, self.CLAHE_TILE_SIZE)
            )
            logger.info("CLAHE preprocessing enabled (clip=%.1f, tile=%d)", 
                       self.CLAHE_CLIP_LIMIT, self.CLAHE_TILE_SIZE)

    # ------------------------------------------------------------------ #
    #  STATIC FILTER HELPERS
    # ------------------------------------------------------------------ #
    def _is_static(self, cx: float, cy: float) -> bool:
        """Return True if (cx, cy) matches a known static spot."""
        for spot in self._static_spots:
            dist = np.sqrt((cx - spot[0]) ** 2 + (cy - spot[1]) ** 2)
            if dist < self.STATIC_RADIUS:
                ratio = spot[2] / max(spot[3], 1)
                if ratio >= self.STATIC_THRESHOLD:
                    return True
        return False

    def _update_static_filter(self, detections: list[BallDetection]) -> None:
        """Update static-spot statistics with this frame's detections."""
        self._frame_count += 1

        # Increment tracked count for all spots
        for spot in self._static_spots:
            spot[3] += 1

        # Match each detection to existing spots or create new ones
        for det in detections:
            matched = False
            for spot in self._static_spots:
                dist = np.sqrt((det.x - spot[0]) ** 2 + (det.y - spot[1]) ** 2)
                if dist < self.STATIC_RADIUS:
                    spot[2] += 1  # hit_count++
                    # Running average of position
                    spot[0] = 0.9 * spot[0] + 0.1 * det.x
                    spot[1] = 0.9 * spot[1] + 0.1 * det.y
                    matched = True
                    break
            if not matched:
                self._static_spots.append([det.x, det.y, 1, 1])

        # Prune old spots that haven't been seen recently
        self._static_spots = [
            s for s in self._static_spots
            if s[3] <= self.STATIC_HISTORY or s[2] / s[3] >= 0.1
        ]

    # ------------------------------------------------------------------ #
    #  PLAYER OVERLAP FILTER
    # ------------------------------------------------------------------ #
    def _overlaps_with_player(self, ball_x: float, ball_y: float, players) -> bool:
        """
        Check if ball detection overlaps with any player's lower region (legs/feet).
        
        Parameters
        ----------
        ball_x, ball_y : float
            Ball center coordinates
        players : list[PlayerDetection]
            List of detected players
            
        Returns
        -------
        bool
            True if ball overlaps with player lower body (likely shoes/legs)
        """
        if not self.PLAYER_OVERLAP_FILTER or not players:
            return False
        
        for player in players:
            # Focus on lower portion of player bounding box (legs/feet)
            player_height = player.y2 - player.y1
            leg_region_top = player.y1 + player_height * (1 - self.PLAYER_LEG_RATIO)
            
            # Check if ball center is within player's leg region
            if (player.x1 <= ball_x <= player.x2 and 
                leg_region_top <= ball_y <= player.y2):
                logger.debug(f"Filtering ball at ({ball_x:.1f}, {ball_y:.1f}) - overlaps with player leg region")
                return True
        
        return False

    # ------------------------------------------------------------------ #
    #  COURT BOUNDARY FILTER
    # ------------------------------------------------------------------ #
    def _is_out_of_court(self, ball_x: float, ball_y: float, court) -> bool:
        """
        Check if ball detection is outside the court boundary.
        
        Parameters
        ----------
        ball_x, ball_y : float
            Ball center coordinates
        court : CourtDetection
            Detected court keypoints
            
        Returns
        -------
        bool
            True if ball is outside court bounds (should be filtered)
        """
        if not self.COURT_BOUNDARY_FILTER or court is None or court.count < 2:
            return False
        
        # Get court bounding box from keypoints
        pts = court.as_array()
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        
        # Check if ball is outside court bounds (with generous margin)
        if (ball_x < x_min - self.COURT_MARGIN or ball_x > x_max + self.COURT_MARGIN or
            ball_y < y_min - self.COURT_MARGIN or ball_y > y_max + self.COURT_MARGIN):
            logger.debug(f"Filtering ball at ({ball_x:.1f}, {ball_y:.1f}) - outside court bounds")
            return True
        
        return False
    
    # ------------------------------------------------------------------ #
    #  REGION-ADAPTIVE CONFIDENCE FILTER
    # ------------------------------------------------------------------ #
    def _is_in_bottom_half(self, ball_y: float, court) -> bool:
        """
        Check if ball is in bottom-half of court (Front/Mid regions).
        
        Parameters
        ----------
        ball_y : float
            Ball y-coordinate
        court : CourtDetection
            Detected court keypoints
            
        Returns
        -------
        bool
            True if in Front or Mid region (bottom-half where detection struggles)
        """
        if not self.ENABLE_REGION_ADAPTIVE_CONFIDENCE or court is None or court.count < 2:
            return False
        
        # Calculate court thirds (same logic as shot_evaluator)
        pts = court.as_array()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        third = (y_max - y_min) / 3.0
        
        # Bottom-half = Mid + Front (y >= first third boundary)
        # In image coords, y increases downward: Back (top) < Mid < Front (bottom)
        bottom_half_threshold = y_min + third
        
        return ball_y >= bottom_half_threshold
    
    def _filter_by_confidence_and_region(self, detections: list[BallDetection], court) -> list[BallDetection]:
        """
        Filter detections using region-adaptive confidence thresholds.
        
        - Upper-half (Back): require confidence >= 0.01
        - Bottom-half (Front/Mid): accept confidence >= 0.005
        
        Parameters
        ----------
        detections : list[BallDetection]
            All raw detections from YOLO
        court : CourtDetection
            Court keypoints for region determination
            
        Returns
        -------
        list[BallDetection]
            Filtered detections meeting region-adaptive criteria
        """
        if not self.ENABLE_REGION_ADAPTIVE_CONFIDENCE or court is None:
            # No filtering - all detections already passed YOLO threshold
            return detections
        
        filtered = []
        for det in detections:
            in_bottom_half = self._is_in_bottom_half(det.y, court)
            
            # Accept if: high confidence OR (low confidence AND bottom-half)
            if det.confidence >= self.NORMAL_CONFIDENCE:
                filtered.append(det)
            elif det.confidence >= self.LOWERED_CONFIDENCE and in_bottom_half:
                logger.debug(f"Accepting low-conf detection in bottom-half: {det.confidence:.3f} at ({det.x:.0f}, {det.y:.0f})")
                filtered.append(det)
            else:
                logger.debug(f"Rejecting low-conf detection in upper-half: {det.confidence:.3f} at ({det.x:.0f}, {det.y:.0f})")
        
        return filtered
    
    # ------------------------------------------------------------------ #
    #  CLAHE PREPROCESSING
    # ------------------------------------------------------------------ #
    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to normalize lighting variations.
        
        Converts to LAB color space, applies CLAHE to luminance channel (L),
        then converts back to BGR. This helps detect the ball consistently
        in both dark shadows and bright areas.
        
        Parameters
        ----------
        frame : np.ndarray
            Input BGR image
            
        Returns
        -------
        np.ndarray
            Lighting-normalized BGR image
        """
        # Convert BGR to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # Split into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to luminance channel
        l_channel = self.clahe.apply(l_channel)
        
        # Merge channels back
        lab = cv2.merge([l_channel, a_channel, b_channel])
        
        # Convert back to BGR
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # ------------------------------------------------------------------ #
    #  RAW INFERENCE
    # ------------------------------------------------------------------ #
    def _run_inference(self, frame: np.ndarray) -> list[BallDetection]:
        """
        Run YOLO and return all raw detections (no filtering).
        
        Applies CLAHE preprocessing if enabled before inference.
        Uses lowered confidence if region-adaptive mode is enabled.
        """
        # Apply CLAHE preprocessing to normalize lighting
        if self.ENABLE_CLAHE:
            frame = self._apply_clahe(frame)
        
        # Use lower confidence threshold if region-adaptive mode is on
        # We'll filter by region later
        conf_threshold = self.LOWERED_CONFIDENCE if self.ENABLE_REGION_ADAPTIVE_CONFIDENCE else BALL_CONFIDENCE
        
        results = self.model.predict(
            frame,
            conf=conf_threshold,
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
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            detections.append(
                BallDetection(
                    x=cx, y=cy,
                    w=x2 - x1, h=y2 - y1,
                    confidence=conf,
                    source="yolo",
                )
            )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------ #
    #  PUBLIC API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray, players=None, court=None) -> Optional[BallDetection]:
        """
        Run inference and return the best *moving* ball detection.

        Static false positives (logos, scoreboards), player overlaps, and 
        out-of-court detections are filtered out.

        Parameters
        ----------
        frame : np.ndarray
            BGR image (OpenCV format).
        players : list[PlayerDetection], optional
            List of detected players to filter out ball detections in player regions.
        court : CourtDetection, optional
            Detected court keypoints to filter out detections outside court bounds.

        Returns
        -------
        BallDetection | None
            Best non-static, non-player-overlapping, in-court detection, or None if nothing found.
        """
        all_dets = self._run_inference(frame)

        # Update the static filter with ALL detections (for statistics)
        self._update_static_filter(all_dets)
        
        # Apply region-adaptive confidence filtering first (uses court info)
        if court is not None:
            all_dets = self._filter_by_confidence_and_region(all_dets, court)

        # Filter out static detections (OPTIONAL - disabled by default)
        if self.ENABLE_STATIC_FILTER and self._frame_count > self.STATIC_HISTORY:
            all_dets = [d for d in all_dets if not self._is_static(d.x, d.y)]
        
        # Filter out detections that overlap with player regions (yellow shoes!)
        if players is not None:
            all_dets = [d for d in all_dets if not self._overlaps_with_player(d.x, d.y, players)]
        
        # Filter out detections outside court bounds (logos, scoreboards in corners)
        if court is not None:
            all_dets = [d for d in all_dets if not self._is_out_of_court(d.x, d.y, court)]
        
        if not all_dets:
            return None
        
        best_det = all_dets[0]
        
        # Movement filter (immediate, works from frame 1) - OPTIONAL
        if self.ENABLE_MOVEMENT_FILTER:
            # Reject if detection is too close to ANY recent position
            for prev_x, prev_y in self._recent_positions:
                dist = np.sqrt((best_det.x - prev_x) ** 2 + (best_det.y - prev_y) ** 2)
                if dist < self.MIN_MOVEMENT:
                    logger.debug(f"Rejecting near-stationary detection at ({best_det.x:.1f}, {best_det.y:.1f}) - within {dist:.1f}px of recent position")
                    return None
            
            # Accepted - add to recent positions history
            self._recent_positions.append((best_det.x, best_det.y))
            if len(self._recent_positions) > self.MOVEMENT_HISTORY:
                self._recent_positions.pop(0)  # Keep only last N positions
        
        return best_det

    # ------------------------------------------------------------------ #
    def detect_all(self, frame: np.ndarray) -> list[BallDetection]:
        """Return *all* non-static ball detections (sorted by confidence desc)."""
        all_dets = self._run_inference(frame)
        self._update_static_filter(all_dets)

        if not self.ENABLE_STATIC_FILTER or self._frame_count <= self.STATIC_HISTORY:
            return all_dets

        return [d for d in all_dets if not self._is_static(d.x, d.y)]
