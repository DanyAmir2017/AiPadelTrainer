"""
Padel Trainer – Main Pipeline
==============================

End-to-end pipeline for a Computer Vision–Assisted Padel Trainer.

Pipeline per frame:
  1. Read frame from video.
  2. Detect ball  (YOLO custom model).
  3. Detect players (YOLO custom model).
  4. Detect court keypoints (YOLO custom model).
  5. If ball not detected by YOLO → try Optical Flow.
  6. Pass detection to Kalman filter → smooth / predict.
  7. Evaluate shot placement (court region).
  8. Annotate frame (boxes, trajectory, HUD).
  9. Write annotated frame to output video.
  10. Record trajectory for CSV export.

After all frames:
  • Save trajectory CSV.
  • Save metrics summary.
  • Print statistics to console.

Usage
-----
    python -m src.main                       # uses config INPUT_VIDEO
    python -m src.main --video path/to.mp4   # specific video

Author: Bachelor Thesis Project – GUC
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np

# ── Config ──────────────────────────────────────────────────────────
from src.utils.config import (
    INPUT_VIDEO,
    MODEL_DEVICE,
    OPTICAL_FLOW_ENABLED,
    KALMAN_ENABLED,
    TRAJECTORY_LENGTH,
    MAX_BALL_DISTANCE,
    MAX_FRAMES_TO_SKIP,
    PROCESS_EVERY_N_FRAMES,
    DISPLAY_WINDOW,
    DISPLAY_SCALE,
    SAVE_OUTPUT_VIDEO,
    SAVE_TRAJECTORY_CSV,
    SAVE_METRICS,
    PRINT_DETECTIONS,
    PRINT_EVERY_N_FRAMES,
    MAX_FRAMES_TO_PROCESS,
    logger,
)

# ── Detection ───────────────────────────────────────────────────────
from src.detection.ball_detector import BallDetector, BallDetection
from src.detection.player_detector import PlayerDetector
from src.detection.court_detector import CourtDetector

# ── Tracking ────────────────────────────────────────────────────────
from src.tracking.optical_flow_tracker import OpticalFlowTracker
from src.tracking.kalman_tracker import KalmanBallTracker

# ── Evaluation ──────────────────────────────────────────────────────
from src.evaluation.shot_evaluator import ShotEvaluator

# ── Utils ───────────────────────────────────────────────────────────
from src.utils.video_utils import (
    VideoReader, VideoWriter, FPSCounter,
    save_trajectory_csv, save_metrics,
)
from src.utils.visualization import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description="Padel Trainer – Ball Tracking Pipeline")
    parser.add_argument(
        "--video", type=str, default=INPUT_VIDEO,
        help="Path to input video file.",
    )
    return parser.parse_args()


def run(video_path: str) -> None:
    """Run the full padel tracking pipeline on a single video."""

    # ================================================================
    #  1. INITIALISE COMPONENTS
    # ================================================================
    logger.info("=" * 60)
    logger.info("  Padel Trainer – Starting Pipeline")
    logger.info("  Device  : %s", MODEL_DEVICE)
    logger.info("  Video   : %s", video_path)
    logger.info("=" * 60)

    # Video I/O
    reader = VideoReader(video_path)
    video_name = os.path.basename(video_path)
    writer = None
    if SAVE_OUTPUT_VIDEO:
        out_name = os.path.splitext(video_name)[0] + "_annotated.mp4"
        writer = VideoWriter(out_name, reader.fps, (reader.width, reader.height))

    # Detectors
    ball_detector = BallDetector()
    player_detector = PlayerDetector()
    court_detector = CourtDetector()

    # Trackers
    of_tracker = OpticalFlowTracker() if OPTICAL_FLOW_ENABLED else None
    kalman = KalmanBallTracker() if KALMAN_ENABLED else None

    # Evaluator
    evaluator = ShotEvaluator()

    # Visualizer
    vis = Visualizer()

    # State
    fps_counter = FPSCounter()
    trajectory: list[tuple[float, float]] = []   # Smoothed ball positions
    csv_records: list[dict] = []                  # For CSV export
    prev_gray = None
    frames_since_detection = 0
    current_source = ""
    has_valid_seed = False  # True once YOLO returns a post-filter detection

    # ================================================================
    #  2. FRAME LOOP
    # ================================================================
    for frame in reader:
        frame_idx = reader.frame_idx

        # Optional: max frames cap
        if MAX_FRAMES_TO_PROCESS and frame_idx > MAX_FRAMES_TO_PROCESS:
            break

        # Skip frames for performance
        if frame_idx % PROCESS_EVERY_N_FRAMES != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 2a. Detection ───────────────────────────────────────────
        court = court_detector.detect(frame)  # Detect court first for boundary filtering
        players = player_detector.detect(frame)  # Detect players for overlap filtering
        ball_det = ball_detector.detect(frame, players, court)  # Pass players and court to filter false positives

        # Update court geometry for evaluator
        evaluator.update_court(court)

        # ── 2b. Multi-Method Detection (Fallback Chain) ─────────────
        ball_x, ball_y = None, None
        ball_conf = 0.0
        source = ""

        # METHOD 1: Try YOLO detection (primary)
        if ball_det is not None:
            ball_x, ball_y = ball_det.x, ball_det.y
            ball_conf = ball_det.confidence
            source = "yolo"
            frames_since_detection = 0
            has_valid_seed = True

            # Seed optical flow tracker
            if of_tracker is not None:
                of_tracker.initialize(gray, (ball_x, ball_y))

            # Update Kalman with YOLO measurement
            if kalman is not None:
                kalman.update(ball_x, ball_y)

        else:
            # METHOD 2: Try Optical Flow as backup detector
            of_pos = None
            if of_tracker is not None and of_tracker.active:
                of_pos = of_tracker.update(gray)

            if of_pos is not None:
                ball_x, ball_y = of_pos
                ball_conf = 0.0
                source = "optical_flow"
                frames_since_detection += 1

                # Update Kalman with OF measurement
                if kalman is not None:
                    kalman.update(ball_x, ball_y)

                # Create synthetic detection for visualization
                ball_det = BallDetection(
                    x=ball_x, y=ball_y, w=16, h=16,
                    confidence=0.0, source="optical_flow",
                )

            else:
                # METHOD 3: Try Kalman prediction as final backup
                if kalman is not None and kalman.active:
                    pred = kalman.predict_no_measurement()
                    if pred is not None and frames_since_detection < MAX_FRAMES_TO_SKIP:
                        ball_x, ball_y = pred
                        source = "kalman"
                        frames_since_detection += 1
                        
                        ball_det = BallDetection(
                            x=ball_x, y=ball_y, w=12, h=12,
                            confidence=0.0, source="kalman",
                        )
                    else:
                        frames_since_detection += 1
                else:
                    frames_since_detection += 1

        # ── 2c. Trajectory ──────────────────────────────────────────
        speed = 0.0
        if ball_x is not None and ball_y is not None:
            # Distance check (reject jumps)
            if trajectory:
                lx, ly = trajectory[-1]
                dist = np.sqrt((ball_x - lx) ** 2 + (ball_y - ly) ** 2)
                if dist > MAX_BALL_DISTANCE:
                    # Suspicious jump – ignore this position
                    ball_x, ball_y = None, None
                    ball_det = None
                    source = ""

        if ball_x is not None and ball_y is not None:
            trajectory.append((ball_x, ball_y))
            if len(trajectory) > TRAJECTORY_LENGTH:
                trajectory = trajectory[-TRAJECTORY_LENGTH:]

            # Speed estimate
            if kalman is not None and kalman.active:
                speed = kalman.get_speed()
            elif len(trajectory) >= 2:
                dx = trajectory[-1][0] - trajectory[-2][0]
                dy = trajectory[-1][1] - trajectory[-2][1]
                speed = np.sqrt(dx ** 2 + dy ** 2)

            current_source = source

        # ── 2d. Shot Evaluation ─────────────────────────────────────
        region = ""
        if ball_x is not None and ball_y is not None:
            shot = evaluator.record_shot(
                frame_id=frame_idx, x=ball_x, y=ball_y,
                speed=speed, source=source,
            )
            region = shot.region

        # ── 2e. CSV record ──────────────────────────────────────────
        if SAVE_TRAJECTORY_CSV and ball_x is not None:
            csv_records.append({
                "frame": frame_idx,
                "x": round(ball_x, 1),
                "y": round(ball_y, 1),
                "confidence": round(ball_conf, 3),
                "source": source,
                "speed": round(speed, 2),
                "region": region,
            })

        # ── 2f. Visualisation ───────────────────────────────────────
        vis.draw_ball(frame, ball_det, source=source)
        vis.draw_trajectory(frame, trajectory)
        vis.draw_players(frame, players)
        vis.draw_court(frame, court)

        current_fps = fps_counter.tick()
        vis.draw_hud(
            frame,
            fps=current_fps,
            frame_idx=frame_idx,
            total_frames=reader.total_frames,
            speed=speed,
            source=current_source,
            region=region,
        )

        # ── 2g. Output ─────────────────────────────────────────────
        if writer is not None:
            writer.write(frame)

        if DISPLAY_WINDOW:
            if DISPLAY_SCALE != 1.0:
                disp = cv2.resize(frame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            else:
                disp = frame
            cv2.imshow("Padel Trainer", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                logger.info("User pressed 'q' – stopping early.")
                break

        # ── 2h. Logging ─────────────────────────────────────────────
        if PRINT_DETECTIONS and frame_idx % PRINT_EVERY_N_FRAMES == 0:
            ball_str = (
                f"({ball_x:.0f},{ball_y:.0f}) [{source}]"
                if ball_x is not None else "—"
            )
            logger.info(
                "Frame %d/%d | Ball: %s | Players: %d | Court: %d kp | FPS: %.1f",
                frame_idx, reader.total_frames, ball_str,
                len(players), court.count, current_fps,
            )

        prev_gray = gray

    # ================================================================
    #  3. CLEANUP
    # ================================================================
    reader.release()
    if writer is not None:
        writer.release()
    if DISPLAY_WINDOW:
        cv2.destroyAllWindows()

    # ================================================================
    #  4. SAVE OUTPUTS
    # ================================================================
    if SAVE_TRAJECTORY_CSV and csv_records:
        save_trajectory_csv(csv_records, video_name)

    if SAVE_METRICS:
        stats = evaluator.get_stats()
        dist = evaluator.get_region_distribution()
        metrics_dict = {
            "Total frames processed": fps_counter.total_frames,
            "Total ball detections": stats.total_detections,
            "Average speed (px/frame)": stats.avg_speed,
            "Max speed (px/frame)": stats.max_speed,
            "Consistency score": stats.consistency_score,
            "Processing FPS": fps_counter.fps,
            "Processing time (s)": fps_counter.elapsed,
            "Device": str(MODEL_DEVICE),
            "Region distribution (%)": {k: f"{v:.1f}" for k, v in dist.items()},
        }
        save_metrics(metrics_dict, video_name)

    # ── Final summary ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  Pipeline Complete")
    logger.info("  Frames processed : %d", fps_counter.total_frames)
    logger.info("  Average FPS      : %.1f", fps_counter.fps)
    logger.info("  Time elapsed     : %.1f s", fps_counter.elapsed)
    if csv_records:
        logger.info("  Ball detections  : %d", len(csv_records))
    logger.info("=" * 60)


# ====================================================================
#  ENTRY POINT
# ====================================================================
if __name__ == "__main__":
    args = parse_args()
    run(args.video)
