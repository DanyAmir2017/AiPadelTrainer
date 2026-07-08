"""
Kalman Filter Tracker Module
==============================

Implements a 2-D constant-velocity Kalman filter using FilterPy.
The state vector is [x, y, vx, vy] and the measurement is [x, y].

The Kalman filter is used to:
  • Smooth noisy detections from YOLO / optical flow.
  • Predict the ball's position when no detection is available.
  • Estimate ball velocity for speed calculations.

Falls back to OpenCV's cv2.KalmanFilter if FilterPy is not installed.

Author: Bachelor Thesis Project – GUC
"""

import logging
from typing import Optional, Tuple

import numpy as np

from src.utils.config import PROCESS_NOISE, MEASUREMENT_NOISE

logger = logging.getLogger("padel_trainer.kalman_tracker")

# Try FilterPy first, fall back to OpenCV
try:
    from filterpy.kalman import KalmanFilter as FilterPyKF
    _USE_FILTERPY = True
    logger.info("Using FilterPy Kalman filter")
except ImportError:
    import cv2
    _USE_FILTERPY = False
    logger.warning("FilterPy not found – falling back to OpenCV KalmanFilter")


class KalmanBallTracker:
    """
    Kalman filter for tracking a padel ball in 2-D.

    State:   [x, y, vx, vy]
    Measure: [x, y]

    Usage
    -----
    >>> kf = KalmanBallTracker()
    >>> kf.initialize(300.0, 200.0)
    >>> predicted = kf.predict()
    >>> corrected = kf.update(310.0, 205.0)
    """

    def __init__(self, process_noise: float = PROCESS_NOISE,
                 measurement_noise: float = MEASUREMENT_NOISE):
        self.active = False
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._kf = None
        self._frames_without_measurement = 0

    # ------------------------------------------------------------------ #
    #  INITIALIZATION
    # ------------------------------------------------------------------ #
    def initialize(self, x: float, y: float) -> None:
        """Create and seed the Kalman filter at position (x, y)."""
        if _USE_FILTERPY:
            self._init_filterpy(x, y)
        else:
            self._init_opencv(x, y)
        self.active = True
        self._frames_without_measurement = 0

    def _init_filterpy(self, x: float, y: float) -> None:
        kf = FilterPyKF(dim_x=4, dim_z=2)

        # State transition: constant-velocity model
        dt = 1.0
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        # Measurement matrix: observe x, y
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise
        q = self._process_noise
        kf.Q = np.array([
            [q,  0,  0,  0],
            [0,  q,  0,  0],
            [0,  0,  q,  0],
            [0,  0,  0,  q],
        ], dtype=np.float64)

        # Measurement noise
        r = self._measurement_noise
        kf.R = np.array([
            [r, 0],
            [0, r],
        ], dtype=np.float64)

        # Initial state
        kf.x = np.array([x, y, 0.0, 0.0], dtype=np.float64)

        # Initial covariance
        kf.P *= 100.0

        self._kf = kf

    def _init_opencv(self, x: float, y: float) -> None:
        import cv2 as cv
        kf = cv.KalmanFilter(4, 2)

        kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self._process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self._measurement_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32) * 100.0

        kf.statePost = np.array([x, y, 0, 0], dtype=np.float32).reshape(4, 1)

        self._kf = kf

    # ------------------------------------------------------------------ #
    #  PREDICT
    # ------------------------------------------------------------------ #
    def predict(self) -> Optional[Tuple[float, float]]:
        """
        Predict the next state (without measurement update).

        Returns
        -------
        (x, y) | None
            Predicted position.
        """
        if not self.active or self._kf is None:
            return None

        if _USE_FILTERPY:
            self._kf.predict()
            state = self._kf.x
            return (float(state[0]), float(state[1]))
        else:
            prediction = self._kf.predict()
            return (float(prediction[0]), float(prediction[1]))

    # ------------------------------------------------------------------ #
    #  UPDATE
    # ------------------------------------------------------------------ #
    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Update (correct) the Kalman filter with a new measurement.

        Returns
        -------
        (x, y)
            Corrected (filtered) position.
        """
        if not self.active or self._kf is None:
            self.initialize(x, y)
            return (x, y)

        self._frames_without_measurement = 0

        if _USE_FILTERPY:
            self._kf.update(np.array([x, y], dtype=np.float64))
            state = self._kf.x
            return (float(state[0]), float(state[1]))
        else:
            measurement = np.array([[x], [y]], dtype=np.float32)
            corrected = self._kf.correct(measurement)
            return (float(corrected[0]), float(corrected[1]))

    # ------------------------------------------------------------------ #
    #  PREDICT WITHOUT MEASUREMENT
    # ------------------------------------------------------------------ #
    def predict_no_measurement(self) -> Optional[Tuple[float, float]]:
        """
        Predict and track missing measurements counter.

        Use when YOLO and optical flow both miss the ball.
        """
        pos = self.predict()
        if pos is not None:
            self._frames_without_measurement += 1
        return pos

    # ------------------------------------------------------------------ #
    #  VELOCITY
    # ------------------------------------------------------------------ #
    def get_velocity(self) -> Optional[Tuple[float, float]]:
        """Return estimated (vx, vy) in pixels/frame."""
        if not self.active or self._kf is None:
            return None

        if _USE_FILTERPY:
            return (float(self._kf.x[2]), float(self._kf.x[3]))
        else:
            state = self._kf.statePost
            return (float(state[2]), float(state[3]))

    def get_speed(self) -> float:
        """Return |v| in pixels/frame."""
        vel = self.get_velocity()
        if vel is None:
            return 0.0
        return float(np.sqrt(vel[0] ** 2 + vel[1] ** 2))

    # ------------------------------------------------------------------ #
    @property
    def frames_without_measurement(self) -> int:
        return self._frames_without_measurement

    def reset(self) -> None:
        """Deactivate and clear state."""
        self.active = False
        self._kf = None
        self._frames_without_measurement = 0
