"""
Contact candidate scoring utilities.

Provides:
- Heuristic scoring out of the box (no extra dependencies)
- Optional learned logistic model loading from JSON
- Decision bands: accept / review / reject
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Dict, Any


@dataclass
class ContactDecision:
    score: float
    decision: str


class ContactScorer:
    def __init__(
        self,
        model_path: str | Path | None = None,
        accept_threshold: float = 0.62,
        review_threshold: float = 0.45,
        min_total_turn_speed_px: float = 14.0,
        min_vertical_delta_px: float = 8.0,
    ):
        self.accept_threshold = float(accept_threshold)
        self.review_threshold = float(review_threshold)
        self.min_total_turn_speed_px = float(min_total_turn_speed_px)
        self.min_vertical_delta_px = float(min_vertical_delta_px)

        self.model_loaded = False
        self.model_bias = 0.0
        self.model_weights: dict[str, float] = {}
        self.model_feature_order: list[str] = []
        self.model_feature_mean: dict[str, float] = {}
        self.model_feature_std: dict[str, float] = {}

        if model_path:
            self._try_load_model(Path(model_path))

    def _try_load_model(self, model_path: Path):
        if not model_path.exists():
            return

        data = json.loads(model_path.read_text(encoding="utf-8"))
        weights = data.get("weights")
        if not isinstance(weights, dict) or not weights:
            return

        self.model_weights = {str(k): float(v) for k, v in weights.items()}
        self.model_bias = float(data.get("bias", 0.0))
        self.model_feature_order = list(data.get("feature_order", self.model_weights.keys()))

        raw_mean = data.get("feature_mean", {})
        raw_std = data.get("feature_std", {})
        self.model_feature_mean = {str(k): float(v) for k, v in raw_mean.items()}
        self.model_feature_std = {str(k): max(1e-6, float(v)) for k, v in raw_std.items()}

        self.accept_threshold = float(data.get("accept_threshold", self.accept_threshold))
        self.review_threshold = float(data.get("review_threshold", self.review_threshold))
        self.model_loaded = True

    def score(self, features: Dict[str, Any]) -> ContactDecision:
        if self.model_loaded:
            score = self._score_model(features)
        else:
            score = self._score_heuristic(features)

        if score >= self.accept_threshold:
            decision = "accept"
        elif score >= self.review_threshold:
            decision = "review"
        else:
            decision = "reject"

        return ContactDecision(score=score, decision=decision)

    def _score_model(self, features: Dict[str, Any]) -> float:
        z = self.model_bias
        for name in self.model_feature_order:
            raw_value = float(features.get(name, 0.0))
            mean = self.model_feature_mean.get(name, 0.0)
            std = self.model_feature_std.get(name, 1.0)
            norm_value = (raw_value - mean) / max(1e-6, std)
            z += self.model_weights.get(name, 0.0) * norm_value
        return 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, z))))

    def _score_heuristic(self, features: Dict[str, Any]) -> float:
        total_turn_speed = float(features.get("total_turn_speed", 0.0))
        vertical_delta = float(features.get("vertical_delta", 0.0))
        cos_turn = float(features.get("cos_turn", 1.0))
        y_sign_flip = 1.0 if bool(features.get("y_sign_flip", False)) else 0.0
        source_quality = float(features.get("source_quality", 0.7))

        speed_norm = min(1.0, total_turn_speed / max(1e-6, self.min_total_turn_speed_px * 1.6))
        vertical_norm = min(1.0, vertical_delta / max(1e-6, self.min_vertical_delta_px * 1.5))

        # cos_turn: lower means sharper turn. Convert to [0,1] with practical clipping.
        turn_norm = min(1.0, max(0.0, (0.25 - cos_turn) / 1.25))

        score = (
            0.30 * speed_norm
            + 0.25 * vertical_norm
            + 0.20 * turn_norm
            + 0.15 * y_sign_flip
            + 0.10 * min(1.0, max(0.0, source_quality))
        )
        return min(1.0, max(0.0, score))


def source_quality_score(source: str) -> float:
    source = (source or "").strip().lower()
    if source == "yolo":
        return 1.0
    if source == "optical_flow":
        return 0.75
    if source == "kalman":
        return 0.35
    return 0.5
