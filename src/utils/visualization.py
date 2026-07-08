"""
Visualization Module
====================

Drawing and annotation utilities for the padel trainer output.
All drawing goes through this module so visual style is consistent.

Author: Bachelor Thesis Project – GUC
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.config import (
    COLOR_BALL_BOX,
    COLOR_BALL_CENTER,
    COLOR_TRAJECTORY,
    COLOR_OPTICAL_FLOW,
    COLOR_PLAYER,
    COLOR_COURT_KP,
    COLOR_COURT_LABEL,
    COLOR_TEXT,
    COURT_KEYPOINT_RADIUS,
    THICKNESS_BOX,
    THICKNESS_TRAJECTORY,
    THICKNESS_CENTER,
    SHOW_BALL,
    SHOW_PLAYERS,
    SHOW_COURT,
    SHOW_TRAJECTORY,
    SHOW_SPEED,
    SHOW_FPS,
    SHOW_DETECTION_SOURCE,
)
from src.detection.ball_detector import BallDetection
from src.detection.player_detector import PlayerDetection
from src.detection.court_detector import CourtDetection

logger = logging.getLogger("padel_trainer.visualization")


class Visualizer:
    """
    Draws all annotations onto a BGR frame.

    Usage
    -----
    >>> vis = Visualizer()
    >>> vis.draw_ball(frame, detection)
    >>> vis.draw_trajectory(frame, positions)
    """

    # ------------------------------------------------------------------ #
    #  BALL
    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_ball(
        frame: np.ndarray,
        detection: Optional[BallDetection],
        source: str = "yolo",
    ) -> None:
        """Draw ball bounding box + centre dot."""
        if not SHOW_BALL or detection is None:
            return

        cx, cy = int(detection.x), int(detection.y)
        hw, hh = int(detection.w / 2), int(detection.h / 2)

        # Bounding box
        color = COLOR_OPTICAL_FLOW if source == "optical_flow" else COLOR_BALL_BOX
        cv2.rectangle(
            frame,
            (cx - hw, cy - hh), (cx + hw, cy + hh),
            color, THICKNESS_BOX,
        )

        # Centre dot
        cv2.circle(frame, (cx, cy), 4, COLOR_BALL_CENTER, THICKNESS_CENTER)

        # Confidence label
        label = f"{detection.confidence:.0%}"
        if SHOW_DETECTION_SOURCE:
            label = f"{source} {label}"
        cv2.putText(
            frame, label,
            (cx - hw, cy - hh - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
        )

    # ------------------------------------------------------------------ #
    #  TRAJECTORY
    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_trajectory(
        frame: np.ndarray,
        positions: List[Tuple[float, float]],
    ) -> None:
        """Draw the ball trajectory trail with fading opacity."""
        if not SHOW_TRAJECTORY or len(positions) < 2:
            return

        n = len(positions)
        for i in range(1, n):
            # Fade: older points are more transparent
            alpha = i / n
            color = tuple(int(c * alpha) for c in COLOR_TRAJECTORY)
            pt1 = (int(positions[i - 1][0]), int(positions[i - 1][1]))
            pt2 = (int(positions[i][0]), int(positions[i][1]))
            cv2.line(frame, pt1, pt2, color, THICKNESS_TRAJECTORY)

    # ------------------------------------------------------------------ #
    #  PLAYERS
    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_players(
        frame: np.ndarray,
        players: List[PlayerDetection],
    ) -> None:
        """Draw bounding boxes for detected players."""
        if not SHOW_PLAYERS:
            return

        for i, p in enumerate(players):
            cv2.rectangle(
                frame,
                (int(p.x1), int(p.y1)),
                (int(p.x2), int(p.y2)),
                COLOR_PLAYER, THICKNESS_BOX,
            )
            label = f"Player {i + 1} ({p.confidence:.0%})"
            cv2.putText(
                frame, label,
                (int(p.x1), int(p.y1) - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PLAYER, 1,
            )

    # ------------------------------------------------------------------ #
    #  COURT KEYPOINTS
    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_court(
        frame: np.ndarray,
        court: Optional[CourtDetection],
    ) -> None:
        """Draw detected court keypoints as labelled circles."""
        if not SHOW_COURT or court is None or court.count == 0:
            return

        for name, kp in court.keypoints.items():
            cx, cy = int(kp.x), int(kp.y)
            cv2.circle(frame, (cx, cy), COURT_KEYPOINT_RADIUS, COLOR_COURT_KP, -1)
            cv2.putText(
                frame, name,
                (cx + 8, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_COURT_LABEL, 1,
            )

    # ------------------------------------------------------------------ #
    #  HUD  (head-up display)
    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_hud(
        frame: np.ndarray,
        fps: float = 0.0,
        frame_idx: int = 0,
        total_frames: int = 0,
        speed: float = 0.0,
        source: str = "",
        region: str = "",
    ) -> None:
        """Draw overlay text (FPS, speed, region, etc.) at bottom-left corner."""
        h, w = frame.shape[:2]
        line_h = 24

        lines = []

        if SHOW_FPS:
            lines.append(f"FPS: {fps:.1f}")

        if total_frames > 0:
            pct = frame_idx / total_frames * 100
            lines.append(f"Frame: {frame_idx}/{total_frames} ({pct:.0f}%)")

        if SHOW_SPEED and speed > 0:
            lines.append(f"Ball speed: {speed:.1f} px/f")

        if SHOW_DETECTION_SOURCE and source:
            lines.append(f"Source: {source}")

        if region:
            lines.append(f"Region: {region}")

        # Start from bottom and work upwards
        y = h - 20  # Start 20px from bottom
        for line in reversed(lines):  # Reverse so first line is at bottom
            cv2.putText(
                frame, line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2,
            )
            y -= line_h  # Move up for next line

    # ------------------------------------------------------------------ #
    #  STATS OVERLAY  (for final frame / summary)
    # ------------------------------------------------------------------ #
    @staticmethod
    def draw_stats_overlay(
        frame: np.ndarray,
        stats: Dict[str, any],
        position: str = "bottom-right",
    ) -> None:
        """Draw a semi-transparent stats panel."""
        lines = [f"{k}: {v}" for k, v in stats.items()]
        if not lines:
            return

        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        padding = 10
        line_h = 20

        # Compute panel size
        max_text_w = max(
            cv2.getTextSize(l, font, font_scale, thickness)[0][0] for l in lines
        )
        panel_w = max_text_w + 2 * padding
        panel_h = len(lines) * line_h + 2 * padding

        if position == "bottom-right":
            x0 = w - panel_w - 10
            y0 = h - panel_h - 10
        else:
            x0, y0 = 10, 10

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw text
        ty = y0 + padding + 14
        for line in lines:
            cv2.putText(frame, line, (x0 + padding, ty), font, font_scale, COLOR_TEXT, thickness)
            ty += line_h
