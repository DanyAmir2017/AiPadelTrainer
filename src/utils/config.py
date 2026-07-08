"""
Configuration Module for Padel Trainer System
==============================================

Centralized configuration for all system parameters.
All hyperparameters, paths, toggles, and thresholds are defined here
so they can be tuned from a single location.

Author: Bachelor Thesis Project – GUC
Date: February 2026
"""

import os
import torch

# ============================================================
# BASE PATHS (relative to padel_trainer/ root)
# ============================================================
# Determine project root (padel_trainer/) dynamically
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # src/utils/
_SRC_DIR = os.path.dirname(_THIS_DIR)                           # src/
PROJECT_ROOT = os.path.dirname(_SRC_DIR)                        # padel_trainer/

# ============================================================
# MODEL PATHS
# ============================================================
# Custom-trained YOLO models (placed in padel_trainer/models/)
BALL_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_ball.pt")  # Using original model
PLAYER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_players.pt")
COURT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best_court.pt")

# ============================================================
# VIDEO INPUT / OUTPUT
# ============================================================
INPUT_VIDEO_DIR = os.path.join(PROJECT_ROOT, "input_videos")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
ANNOTATED_VIDEO_DIR = os.path.join(OUTPUT_DIR, "annotated_videos")
TRAJECTORY_DIR = os.path.join(OUTPUT_DIR, "trajectories")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
DEBUG_FRAMES_DIR = os.path.join(OUTPUT_DIR, "frames_debug")

# Default input video (change this or pass via CLI)
INPUT_VIDEO = os.path.join(INPUT_VIDEO_DIR, "sample_padel.mp4")

# Ensure output directories exist
for _d in [ANNOTATED_VIDEO_DIR, TRAJECTORY_DIR, METRICS_DIR, DEBUG_FRAMES_DIR]:
    os.makedirs(_d, exist_ok=True)

# ============================================================
# DEVICE CONFIGURATION
# ============================================================
# Auto-detect GPU; fall back to CPU if CUDA not available
MODEL_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# ============================================================
# MODEL CONFIDENCE THRESHOLDS
# ============================================================
BALL_CONFIDENCE = 0.01       # Extremely low threshold - relying on filters
PLAYER_CONFIDENCE = 0.40     # Players are large, higher threshold is fine
COURT_CONFIDENCE = 0.30      # Court keypoints
MODEL_IOU_THRESHOLD = 0.45   # NMS IoU threshold

# ============================================================
# CLASS MAPPINGS (from custom-trained models)
# ============================================================
# Ball model: {0: 'ball'}
BALL_CLASS_ID = 0

# Player model: {0: 'person'}
PLAYER_CLASS_ID = 0

# Court model: 10 keypoints representing court corners / intersections
# {0: 'p1', 1: 'p10', 2: 'p2', 3: 'p3', 4: 'p4',
#  5: 'p5', 6: 'p6',  7: 'p7', 8: 'p8', 9: 'p9'}
COURT_CLASS_IDS = list(range(10))
COURT_KEYPOINT_NAMES = {
    0: "p1", 1: "p10", 2: "p2", 3: "p3", 4: "p4",
    5: "p5", 6: "p6",  7: "p7", 8: "p8", 9: "p9",
}

# ============================================================
# INFERENCE SETTINGS
# ============================================================
INFERENCE_SIZE = 640          # YOLO input image size (640 for speed, 1280 for small-ball detection)
USE_AMP = True                # Mixed-precision (faster on GPU)

# ============================================================
# OPTICAL FLOW PARAMETERS  (Lucas-Kanade sparse flow)
# ============================================================
OPTICAL_FLOW_ENABLED = True  # Backup detector when YOLO fails

# Shi-Tomasi corner detector – finds features near the ball
FEATURE_PARAMS = dict(
    maxCorners=50,
    qualityLevel=0.1,
    minDistance=5,
    blockSize=7,
)

# Lucas-Kanade optical flow parameters
LK_PARAMS = dict(
    winSize=(21, 21),           # Search window size
    maxLevel=3,                 # Pyramid levels
    criteria=(3, 30, 0.01),     # TERM_CRITERIA_EPS | TERM_CRITERIA_COUNT
)

OPTICAL_FLOW_SEARCH_RADIUS = 80    # Pixels around predicted position
OPTICAL_FLOW_MIN_QUALITY = 0.5     # Minimum quality to accept OF estimate

# ============================================================
# KALMAN FILTER PARAMETERS
# ============================================================
KALMAN_ENABLED = True  # Final backup detector via prediction

# Process noise – trust in constant-velocity motion model
# Higher → more responsive / jittery; Lower → smoother / slower
PROCESS_NOISE = 1e-3

# Measurement noise – trust in detector measurement
# Higher → smoother; Lower → follows raw detection closely
MEASUREMENT_NOISE = 1e-1

# ============================================================
# BALL TRACKING PARAMETERS
# ============================================================
MAX_BALL_DISTANCE = 150       # Max px distance between consecutive detections
MIN_DETECTION_FRAMES = 3      # Minimum consecutive detections for valid track
MAX_FRAMES_TO_SKIP = 10       # Frames to keep predicting without detection

# ============================================================
# TRAJECTORY & SPEED
# ============================================================
TRAJECTORY_LENGTH = 30        # Past positions to keep for trail drawing
SPEED_CALCULATION_FRAMES = 5  # Moving-average window for speed

# ============================================================
# VISUALIZATION  (BGR colours for OpenCV)
# ============================================================
COLOR_BALL_BOX = (0, 255, 0)          # Green – ball bounding box
COLOR_BALL_CENTER = (0, 0, 255)       # Red   – ball center dot
COLOR_TRAJECTORY = (255, 0, 255)      # Magenta – trajectory trail
COLOR_OPTICAL_FLOW = (255, 255, 0)    # Cyan  – optical-flow detections
COLOR_PLAYER = (255, 200, 0)          # Light blue – player boxes
COLOR_COURT_KP = (0, 165, 255)        # Orange – court keypoints
COLOR_COURT_LABEL = (0, 255, 255)     # Yellow – keypoint labels
COLOR_TEXT = (255, 255, 255)          # White  – general text

COURT_KEYPOINT_RADIUS = 6            # Drawn circle radius

# Visualization toggles
SHOW_BALL = True
SHOW_PLAYERS = True
SHOW_COURT = True
SHOW_TRAJECTORY = True
SHOW_SPEED = True
SHOW_FPS = True
SHOW_DETECTION_SOURCE = True          # YOLO / OptFlow / Kalman label

# Drawing
THICKNESS_BOX = 2
THICKNESS_TRAJECTORY = 2
THICKNESS_CENTER = -1                 # Filled circle

# ============================================================
# PERFORMANCE / DEBUG
# ============================================================
PROCESS_EVERY_N_FRAMES = 1           # 1 = every frame
DISPLAY_WINDOW = True                 # Show live CV window
DISPLAY_SCALE = 0.5                   # Resize display window (0.5 = half size)
SAVE_OUTPUT_VIDEO = True
SAVE_TRAJECTORY_CSV = True
SAVE_METRICS = True
SAVE_DEBUG_FRAMES = False             # Save individual annotated frames
PRINT_DETECTIONS = False
PRINT_EVERY_N_FRAMES = 30
MAX_FRAMES_TO_PROCESS = None          # None = whole video

# ============================================================
# LOGGING
# ============================================================
import logging

LOG_LEVEL = logging.INFO
LOG_FORMAT = "[%(levelname)s] %(name)s – %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger("padel_trainer")
