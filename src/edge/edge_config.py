"""
Edge Deployment Configuration
Optimized for Raspberry Pi 5 + Hailo-8 AI Accelerator

Target Hardware: reComputer AI R2000
- Raspberry Pi 5 (ARM64)
- Hailo-8 AI Accelerator (26 TOPS)
- 8GB RAM
"""

import os
from pathlib import Path

# ============================================================================
# PATH CONFIGURATION (Cross-platform compatible)
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models" / "onnx"

# Output paths
OUTPUT_BASE_DIR = BASE_DIR / "outputs"
OUTPUT_DIR = OUTPUT_BASE_DIR / "edge"

# Neat edge output subfolders
EDGE_CSV_DIR = OUTPUT_DIR / "csv"
EDGE_DETECTIONS_DIR = OUTPUT_DIR / "detections"
EDGE_CLEAN_CSV_DIR = OUTPUT_DIR / "clean_csv"
EDGE_HIT_CANDIDATES_DIR = OUTPUT_DIR / "hit_candidates"
EDGE_ANNOTATED_VIDEOS_DIR = OUTPUT_DIR / "annotated_videos"
EDGE_TRAIL_SNAPSHOTS_DIR = OUTPUT_DIR / "trail_snapshots"
EDGE_CONTACT_MODEL_DIR = OUTPUT_DIR / "contact_models"
EDGE_CONTACT_MODEL_PATH = EDGE_CONTACT_MODEL_DIR / "contact_scorer_v3.json"

# Model Paths
# NOTE: Use YOLOv8n (nano) model - NOT YOLOv8x
# YOLOv8n: ~3M params, 6MB file size, suitable for Pi/Hailo
# YOLOv8x: ~68M params, 130MB file size, TOO HEAVY for edge
BALL_MODEL_PATH = MODELS_DIR / "best_ball_nano.onnx"

# ============================================================================
# INFERENCE CONFIGURATION
# ============================================================================
# Static input shape (required for Hailo compilation)
INFERENCE_SIZE = 640  # Fixed 640x640, do not change
BALL_CONFIDENCE = 0.01  # Low threshold, relies on Kalman filtering

# ============================================================================
# KALMAN FILTER SETTINGS
# ============================================================================
KALMAN_ENABLED = True
PROCESS_NOISE = 10.0  # Motion uncertainty
MEASUREMENT_NOISE = 5.0  # Detection uncertainty
MAX_PREDICTION_FRAMES = 10  # Max frames to predict without detection

# ============================================================================
# OPTICAL FLOW SETTINGS (Fallback tracker)
# ============================================================================
OPTICAL_FLOW_ENABLED = True
OPTICAL_FLOW_SEARCH_RADIUS = 80
OPTICAL_FLOW_FB_MAX_ERROR = 2.0

FEATURE_PARAMS = dict(
	maxCorners=30,
	qualityLevel=0.1,
	minDistance=5,
	blockSize=7,
)

LK_PARAMS = dict(
	winSize=(21, 21),
	maxLevel=3,
	criteria=(3, 30, 0.01),
)

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================
CSV_SAVE_ENABLED = True
SAVE_FULL_TRAJECTORY = True  # Save all frames (even without detection)

# ============================================================================
# INFERENCE ENGINE SELECTION
# ============================================================================
# Options: 'onnx', 'hailo' (future)
INFERENCE_ENGINE = 'onnx'  # Will be 'hailo' after .hef compilation

# ONNX Runtime Settings (temporary, before Hailo)
ONNX_USE_CPU = True  # Force CPU on Pi (no CUDA available)
ONNX_THREADS = 4  # Optimize for Pi 5's 4 cores

# Hailo Runtime Settings (for future use)
HAILO_DEVICE_ID = 0
HAILO_BATCH_SIZE = 1

# ============================================================================
# OPTIMIZATION FLAGS
# ============================================================================
ENABLE_PREPROCESSING_CACHE = False  # Disabled for video (each frame unique)
ENABLE_MEMORY_OPTIMIZATION = True  # Minimize array copies
FRAME_SKIP = 0  # Process every frame (0 = no skip)

# ============================================================================
# DEBUG SETTINGS
# ============================================================================
VERBOSE = False  # Set True for detailed logging
BENCHMARK_MODE = False  # Set True to measure FPS

# ============================================================================
# CONTACT SCORING SETTINGS
# ============================================================================
CONTACT_SCORING_ENABLED = True
CONTACT_ACCEPT_THRESHOLD = 0.62
CONTACT_REVIEW_THRESHOLD = 0.45
CONTACT_SCORING_KEEP_REJECTED = False

# ============================================================================
# CONTACT TYPE FILTERING
# ============================================================================
# Keep only ground collisions in the final collision/contact outputs.
CONTACT_TYPE_FILTER = {"ground"}
CONTACT_TYPE_FRAME_TOLERANCE = 2  # Match labels within +/- 2 frames when filtering
