"""
Edge Ball Detector - Simplified for Raspberry Pi 5 + Hailo-8

Lightweight ball detection pipeline optimized for edge deployment.
Uses inference abstraction layer for ONNX → Hailo migration.
"""

import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from typing import Optional, Tuple
from pathlib import Path

from inference_engine import create_inference_engine
import edge_config as config


class SimpleKalmanTracker:
    """
    Lightweight Kalman filter for ball trajectory tracking
    Uses constant velocity motion model
    """
    
    def __init__(self):
        # Initialize Kalman filter: 4D state (x, y, vx, vy), 2D measurement (x, y)
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we only observe position)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process and measurement noise
        self.kf.Q *= config.PROCESS_NOISE
        self.kf.R *= config.MEASUREMENT_NOISE
        
        # Initial covariance
        self.kf.P *= 1000.0
        
        # Tracking state
        self.initialized = False
        self.frames_without_detection = 0
    
    def update(self, detection: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Update tracker with detection or predict if no detection
        
        Args:
            detection: (x, y) ball position or None
            
        Returns:
            (x, y) estimated ball position or None if tracking lost
        """
        
        if detection is not None:
            x, y = detection
            
            if not self.initialized:
                # Initialize state with first detection
                self.kf.x = np.array([[x], [y], [0], [0]], dtype=np.float32)
                self.initialized = True
            else:
                # Predict and update
                self.kf.predict()
                self.kf.update(np.array([[x], [y]], dtype=np.float32))
            
            self.frames_without_detection = 0
            return (int(self.kf.x[0][0]), int(self.kf.x[1][0]))
        
        else:
            # No detection, predict only if tracker is active
            if self.initialized and self.frames_without_detection < config.MAX_PREDICTION_FRAMES:
                self.kf.predict()
                self.frames_without_detection += 1
                return (int(self.kf.x[0][0]), int(self.kf.x[1][0]))
            
            return None
    
    def reset(self):
        """Reset tracker state"""
        self.initialized = False
        self.frames_without_detection = 0


class SimpleOpticalFlowTracker:
    """Lightweight Lucas-Kanade optical flow tracker for fallback detections."""

    def __init__(self):
        self.prev_gray = None
        self.prev_points = None
        self.active = False

        self.lk_params = dict(
            winSize=config.LK_PARAMS["winSize"],
            maxLevel=config.LK_PARAMS["maxLevel"],
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                config.LK_PARAMS["criteria"][1],
                config.LK_PARAMS["criteria"][2],
            ),
        )

    def initialize(self, gray_frame: np.ndarray, seed_point: Tuple[int, int]) -> None:
        """Initialize tracker from a known ball location."""
        self.prev_gray = gray_frame.copy()
        sx, sy = int(seed_point[0]), int(seed_point[1])

        h, w = gray_frame.shape[:2]
        r = config.OPTICAL_FLOW_SEARCH_RADIUS
        x1 = max(0, sx - r)
        y1 = max(0, sy - r)
        x2 = min(w, sx + r)
        y2 = min(h, sy + r)

        roi = gray_frame[y1:y2, x1:x2]
        seed_arr = np.array([[[sx, sy]]], dtype=np.float32)

        if roi.size == 0:
            self.prev_points = seed_arr
            self.active = True
            return

        local_pts = cv2.goodFeaturesToTrack(roi, **config.FEATURE_PARAMS)
        if local_pts is not None and len(local_pts) > 0:
            local_pts[:, 0, 0] += x1
            local_pts[:, 0, 1] += y1
            self.prev_points = np.vstack([seed_arr, local_pts]).astype(np.float32)
        else:
            self.prev_points = seed_arr

        self.active = True

    def update(self, gray_frame: np.ndarray) -> Optional[Tuple[int, int]]:
        """Track from previous gray frame to current gray frame."""
        if not self.active or self.prev_gray is None or self.prev_points is None:
            return None

        if len(self.prev_points) == 0:
            self.active = False
            return None

        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray_frame, self.prev_points, None, **self.lk_params
        )

        if next_pts is None or status is None:
            self.active = False
            return None

        back_pts, _, _ = cv2.calcOpticalFlowPyrLK(
            gray_frame, self.prev_gray, next_pts, None, **self.lk_params
        )

        if back_pts is not None:
            fb_error = np.linalg.norm(
                self.prev_points.reshape(-1, 2) - back_pts.reshape(-1, 2), axis=1
            )
            fb_mask = fb_error < config.OPTICAL_FLOW_FB_MAX_ERROR
        else:
            fb_mask = np.ones(len(status), dtype=bool)

        good_mask = (status.ravel() == 1) & fb_mask
        if not np.any(good_mask):
            self.active = False
            return None

        good_next = next_pts[good_mask].reshape(-1, 2)
        if len(good_next) == 0:
            self.active = False
            return None

        cx = int(np.median(good_next[:, 0]))
        cy = int(np.median(good_next[:, 1]))

        self.prev_gray = gray_frame.copy()
        self.prev_points = good_next.reshape(-1, 1, 2).astype(np.float32)
        return (cx, cy)

    def reset(self):
        self.prev_gray = None
        self.prev_points = None
        self.active = False


class EdgeBallDetector:
    """
    Simplified ball detector for edge deployment
    - Ball detection only (no players, no court)
    - Kalman filtering for smooth tracking
    - Inference engine abstraction (ONNX → Hailo ready)
    """
    
    def __init__(self):
        """Initialize detector with inference engine and Kalman tracker"""
        
        if config.VERBOSE:
            print("\n" + "="*60)
            print("EDGE BALL DETECTOR - Initializing")
            print("="*60)
            print(f"Target: Raspberry Pi 5 + Hailo-8")
            print(f"Inference Engine: {config.INFERENCE_ENGINE.upper()}")
            print(f"Model: {config.BALL_MODEL_PATH.name}")
            print(f"Input Size: {config.INFERENCE_SIZE}x{config.INFERENCE_SIZE}")
            print("="*60 + "\n")
        
        # Create inference engine
        self.engine = create_inference_engine(
            config.INFERENCE_ENGINE,
            num_threads=config.ONNX_THREADS if config.INFERENCE_ENGINE == 'onnx' else None,
            device_id=config.HAILO_DEVICE_ID if config.INFERENCE_ENGINE == 'hailo' else None
        )
        
        # Load model
        self.engine.load_model(config.BALL_MODEL_PATH)
        
        # Warmup inference engine
        if config.BENCHMARK_MODE:
            self.engine.warmup()

        # Initialize optical flow tracker (fallback)
        self.optical_flow = SimpleOpticalFlowTracker() if config.OPTICAL_FLOW_ENABLED else None
        
        # Initialize Kalman tracker
        self.tracker = SimpleKalmanTracker() if config.KALMAN_ENABLED else None
        
        # Cache original frame dimensions
        self.frame_width = None
        self.frame_height = None
        
        if config.VERBOSE:
            print("✓ Detector initialized successfully\n")
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for inference
        - Resize to 640x640 (static shape)
        - Convert BGR → RGB
        - Normalize to [0, 1]
        - Transpose to NCHW format
        """
        
        # Cache frame dimensions for postprocessing
        if self.frame_height is None:
            self.frame_height, self.frame_width = frame.shape[:2]
        
        # Resize to inference size
        resized = cv2.resize(frame, (config.INFERENCE_SIZE, config.INFERENCE_SIZE))
        
        # BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # HWC to CHW
        chw = normalized.transpose(2, 0, 1)
        
        # Add batch dimension: CHW → NCHW
        nchw = np.expand_dims(chw, axis=0)
        
        return nchw
    
    def postprocess_detections(self, output: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Postprocess model output to extract best ball detection
        
        YOLOv8 output format: [batch, num_boxes, 5]
        Each box: [x_center, y_center, width, height, confidence]
        
        Returns:
            (x, y) in original frame coordinates or None
        """
        
        # YOLOv8 output is typically [1, 5, num_predictions]
        # We need to transpose to [1, num_predictions, 5]
        if output.ndim == 3 and output.shape[1] < output.shape[2]:
            output = output.transpose(0, 2, 1)
        
        # Remove batch dimension
        detections = output[0]  # Shape: [num_predictions, 5]
        
        # Extract confidence scores (last column)
        confidences = detections[:, 4]
        
        # Filter by confidence threshold
        mask = confidences >= config.BALL_CONFIDENCE
        filtered_detections = detections[mask]
        
        if len(filtered_detections) == 0:
            return None
        
        # Get detection with highest confidence
        best_idx = filtered_detections[:, 4].argmax()
        best_detection = filtered_detections[best_idx]
        
        # Extract center coordinates from model output.
        # Depending on export/backend, coordinates can be either:
        # 1) normalized [0,1], or
        # 2) model-space pixels [0, INFERENCE_SIZE].
        x_center = float(best_detection[0])
        y_center = float(best_detection[1])

        if x_center <= 1.5 and y_center <= 1.5:
            # Normalized coordinates
            x = int(x_center * self.frame_width)
            y = int(y_center * self.frame_height)
        else:
            # Model-space coordinates (typically 640x640)
            x = int(x_center * self.frame_width / config.INFERENCE_SIZE)
            y = int(y_center * self.frame_height / config.INFERENCE_SIZE)
        
        # Clamp to frame boundaries
        x = max(0, min(x, self.frame_width - 1))
        y = max(0, min(y, self.frame_height - 1))
        
        return (x, y)
    
    def detect(self, frame: np.ndarray) -> Tuple[Optional[Tuple[int, int]], str]:
        """
        Detect ball in frame
        
        Args:
            frame: BGR image from video
            
        Returns:
            ((x, y), source) tuple where source is 'yolo', 'kalman', or 'none'
        """
        
        # Preprocess
        preprocessed = self.preprocess_frame(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run inference
        output = self.engine.predict(preprocessed)
        
        # Postprocess
        detection = self.postprocess_detections(output)
        
        # Primary method: YOLO
        if detection is not None:
            if self.optical_flow:
                self.optical_flow.initialize(gray, detection)

            if self.tracker:
                tracked_pos = self.tracker.update(detection)
                if tracked_pos:
                    return (tracked_pos, 'yolo')
            return (detection, 'yolo')

        # Fallback 1: Optical Flow
        of_detection = None
        if self.optical_flow and self.optical_flow.active:
            of_detection = self.optical_flow.update(gray)

        if of_detection is not None:
            if self.tracker:
                tracked_pos = self.tracker.update(of_detection)
                if tracked_pos:
                    return (tracked_pos, 'optical_flow')
            return (of_detection, 'optical_flow')

        # Fallback 2: Kalman prediction
        if self.tracker:
            tracked_pos = self.tracker.update(None)
            
            if tracked_pos:
                return (tracked_pos, 'kalman')
            else:
                return (None, 'none')

        return (None, 'none')
    
    def reset(self):
        """Reset detector state (useful between videos)"""
        if self.tracker:
            self.tracker.reset()
        if self.optical_flow:
            self.optical_flow.reset()
        self.frame_width = None
        self.frame_height = None
