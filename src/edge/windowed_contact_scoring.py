"""Windowed contact scoring utilities.

This module supports a CSV-based contact model that uses a short trajectory
window around each collision candidate. It is intentionally independent from
the image detector so it can be trained and run from cleaned trajectory data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple
import csv
import json
import math


@dataclass
class ContactDecision:
    score: float
    decision: str


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _normalize_rule(rule: str) -> str:
    rule = (rule or "").strip().lower()
    return rule.replace(" ", "_").replace("-", "_")


def normalize_contact_type(contact_type: str) -> str:
    contact_type = (contact_type or "").strip().lower()
    return contact_type.replace(" ", "_").replace("-", "_")


def _offset_token(offset: int) -> str:
    if offset == 0:
        return "c0"
    return f"m{abs(offset)}" if offset < 0 else f"p{offset}"


def source_quality_score(source: str) -> float:
    source = (source or "").strip().lower()
    if source == "yolo":
        return 1.0
    if source == "optical_flow":
        return 0.75
    if source == "kalman":
        return 0.35
    return 0.5


def load_clean_trajectory(path: Path) -> Dict[int, Tuple[float, float]]:
    """Load a cleaned trajectory CSV into a frame->(x, y) mapping."""
    points: Dict[int, Tuple[float, float]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["Frame"]))
            points[frame] = (_to_float(row["X"]), _to_float(row["Y"]))
    return points


def load_candidate_rows(path: Path) -> List[dict]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def infer_clean_trajectory_path(candidate_path: Path, clean_dir: Path | None = None) -> Path:
    stem = candidate_path.stem
    if stem.endswith("_hit_candidates"):
        stem = stem[: -len("_hit_candidates")]
    clean_name = f"{stem}_trajectory_clean.csv"
    if clean_dir is not None:
        return clean_dir / clean_name
    return candidate_path.parent.parent / "clean_csv" / clean_name


def infer_candidate_path(label_path: Path, candidates_dir: Path) -> Path:
    name = label_path.name.replace("_labels.csv", ".csv")
    return candidates_dir / name


def build_feature_order(context_before: int, context_after: int) -> List[str]:
    feature_order: List[str] = []

    for offset in range(-context_before, context_after + 1):
        tag = _offset_token(offset)
        feature_order.extend([
            f"present_{tag}",
            f"x_rel_{tag}",
            f"y_rel_{tag}",
            f"vx_{tag}",
            f"vy_{tag}",
            f"speed_{tag}",
        ])

    feature_order.extend([
        "score",
        "source_quality",
        "rule_y_velocity_sign_flip",
        "rule_sharp_direction_change",
        "candidate_speed_before",
        "candidate_speed_after",
        "candidate_total_turn_speed",
        "candidate_vertical_delta",
        "candidate_turn_cos",
        "candidate_abs_v1y",
        "candidate_abs_v2y",
        "pre_present_count",
        "post_present_count",
        "pre_speed_mean",
        "post_speed_mean",
        "pre_speed_max",
        "post_speed_max",
        "pre_path_length",
        "post_path_length",
        "pre_y_trend",
        "post_y_trend",
        "pre_x_trend",
        "post_x_trend",
        "pre_speed_delta",
        "post_speed_delta",
        "window_span",
    ])

    return feature_order


def _window_point(frame: int, clean_by_frame: Dict[int, Tuple[float, float]]) -> Tuple[float, float] | None:
    return clean_by_frame.get(frame)


def _side_stats(
    frames: Iterable[int],
    center_frame: int,
    candidate_x: float,
    candidate_y: float,
    clean_by_frame: Dict[int, Tuple[float, float]],
) -> Dict[str, float]:
    frames = list(frames)
    present_points: List[Tuple[int, float, float]] = []
    speeds: List[float] = []
    prev_point: Tuple[float, float] | None = None
    first_xy: Tuple[float, float] | None = None
    last_xy: Tuple[float, float] | None = None

    for frame in frames:
        point = _window_point(frame, clean_by_frame)
        if point is None:
            continue

        x, y = point
        present_points.append((frame, x, y))
        if first_xy is None:
            first_xy = (x, y)
        last_xy = (x, y)

        if prev_point is not None:
            vx = x - prev_point[0]
            vy = y - prev_point[1]
            speeds.append(math.hypot(vx, vy))
        prev_point = (x, y)

    if len(speeds) > 0:
        speed_mean = sum(speeds) / len(speeds)
        speed_max = max(speeds)
        speed_delta = speeds[-1] - speeds[0] if len(speeds) >= 2 else 0.0
        path_length = sum(speeds)
    else:
        speed_mean = speed_max = speed_delta = path_length = 0.0

    if first_xy is not None and last_xy is not None and len(present_points) >= 2:
        x_trend = last_xy[0] - first_xy[0]
        y_trend = last_xy[1] - first_xy[1]
    else:
        x_trend = y_trend = 0.0

    return {
        "present_count": float(len(present_points)),
        "speed_mean": float(speed_mean),
        "speed_max": float(speed_max),
        "speed_delta": float(speed_delta),
        "path_length": float(path_length),
        "x_trend": float(x_trend),
        "y_trend": float(y_trend),
    }


def build_window_features(
    candidate_row: dict,
    clean_by_frame: Dict[int, Tuple[float, float]],
    context_before: int,
    context_after: int,
) -> Dict[str, float]:
    candidate_frame = int(float(candidate_row["Frame"]))
    candidate_x = _to_float(candidate_row.get("X"))
    candidate_y = _to_float(candidate_row.get("Y"))
    rule = _normalize_rule(candidate_row.get("Rule"))

    features: Dict[str, float] = {}

    # Optional candidate-level fields from scored CSVs; missing values become zero.
    features["score"] = _to_float(candidate_row.get("Score", 0.0))
    features["source_quality"] = _to_float(candidate_row.get("SourceQuality", 0.0))

    for offset in range(-context_before, context_after + 1):
        frame = candidate_frame + offset
        tag = _offset_token(offset)
        point = _window_point(frame, clean_by_frame)

        if point is None:
            features[f"present_{tag}"] = 0.0
            features[f"x_rel_{tag}"] = 0.0
            features[f"y_rel_{tag}"] = 0.0
            features[f"vx_{tag}"] = 0.0
            features[f"vy_{tag}"] = 0.0
            features[f"speed_{tag}"] = 0.0
            continue

        x, y = point
        prev_point = _window_point(frame - 1, clean_by_frame)
        if prev_point is not None:
            vx = x - prev_point[0]
            vy = y - prev_point[1]
            speed = math.hypot(vx, vy)
        else:
            vx = vy = speed = 0.0

        features[f"present_{tag}"] = 1.0
        features[f"x_rel_{tag}"] = x - candidate_x
        features[f"y_rel_{tag}"] = y - candidate_y
        features[f"vx_{tag}"] = float(vx)
        features[f"vy_{tag}"] = float(vy)
        features[f"speed_{tag}"] = float(speed)

    prev_point = _window_point(candidate_frame - 1, clean_by_frame)
    next_point = _window_point(candidate_frame + 1, clean_by_frame)
    if prev_point is not None and next_point is not None:
        v1x = candidate_x - prev_point[0]
        v1y = candidate_y - prev_point[1]
        v2x = next_point[0] - candidate_x
        v2y = next_point[1] - candidate_y
        s1 = math.hypot(v1x, v1y)
        s2 = math.hypot(v2x, v2y)
        dot = v1x * v2x + v1y * v2y
        denom = max(1e-6, s1 * s2)
        turn_cos = max(-1.0, min(1.0, dot / denom))
        vertical_delta = abs(v2y - v1y)
        features["candidate_speed_before"] = float(s1)
        features["candidate_speed_after"] = float(s2)
        features["candidate_total_turn_speed"] = float(s1 + s2)
        features["candidate_vertical_delta"] = float(vertical_delta)
        features["candidate_turn_cos"] = float(turn_cos)
        features["candidate_abs_v1y"] = float(abs(v1y))
        features["candidate_abs_v2y"] = float(abs(v2y))
    else:
        features["candidate_speed_before"] = 0.0
        features["candidate_speed_after"] = 0.0
        features["candidate_total_turn_speed"] = 0.0
        features["candidate_vertical_delta"] = 0.0
        features["candidate_turn_cos"] = 0.0
        features["candidate_abs_v1y"] = 0.0
        features["candidate_abs_v2y"] = 0.0

    pre_frames = range(candidate_frame - context_before, candidate_frame)
    post_frames = range(candidate_frame + 1, candidate_frame + context_after + 1)
    pre_stats = _side_stats(pre_frames, candidate_frame, candidate_x, candidate_y, clean_by_frame)
    post_stats = _side_stats(post_frames, candidate_frame, candidate_x, candidate_y, clean_by_frame)

    features["pre_present_count"] = pre_stats["present_count"]
    features["post_present_count"] = post_stats["present_count"]
    features["pre_speed_mean"] = pre_stats["speed_mean"]
    features["post_speed_mean"] = post_stats["speed_mean"]
    features["pre_speed_max"] = pre_stats["speed_max"]
    features["post_speed_max"] = post_stats["speed_max"]
    features["pre_path_length"] = pre_stats["path_length"]
    features["post_path_length"] = post_stats["path_length"]
    features["pre_y_trend"] = pre_stats["y_trend"]
    features["post_y_trend"] = post_stats["y_trend"]
    features["pre_x_trend"] = pre_stats["x_trend"]
    features["post_x_trend"] = post_stats["x_trend"]
    features["pre_speed_delta"] = pre_stats["speed_delta"]
    features["post_speed_delta"] = post_stats["speed_delta"]
    features["window_span"] = float(context_before + context_after + 1)

    features["rule_y_velocity_sign_flip"] = 1.0 if rule == "y_velocity_sign_flip" else 0.0
    features["rule_sharp_direction_change"] = 1.0 if rule == "sharp_direction_change" else 0.0

    return features


def fit_logistic_regression(
    x: "Any",
    y: "Any",
    epochs: int = 2000,
    lr: float = 0.03,
    l2: float = 1e-3,
):
    import numpy as np

    mean = x.mean(axis=0)
    std = x.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)

    x_norm = (x - mean) / std

    n_samples, n_features = x_norm.shape
    w = np.zeros(n_features, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        z = x_norm @ w + b
        z = np.clip(z, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))

        error = p - y
        grad_w = (x_norm.T @ error) / n_samples + l2 * w
        grad_b = float(error.mean())

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, mean, std


def probabilities(x, w, b: float, mean, std):
    import numpy as np

    x_norm = (x - mean) / std
    z = np.clip(x_norm @ w + b, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def metrics_at_threshold(y_true, prob, threshold: float) -> dict:
    import numpy as np

    pred = (prob >= threshold).astype(np.float64)
    tp = float(((pred == 1) & (y_true == 1)).sum())
    tn = float(((pred == 0) & (y_true == 0)).sum())
    fp = float(((pred == 1) & (y_true == 0)).sum())
    fn = float(((pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(1.0, (tp + tn + fp + fn))

    return {
        "threshold": threshold,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def best_threshold(y_true, prob) -> float:
    thresholds = sorted(set(float(v) for v in prob))
    if not thresholds:
        return 0.5

    best_t = 0.5
    best_f1 = -1.0
    for t in thresholds:
        m = metrics_at_threshold(y_true, prob, t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = t
    return float(best_t)


class WindowedContactScorer:
    """Logistic scorer for windowed contact features."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        accept_threshold: float = 0.62,
        review_threshold: float = 0.45,
    ):
        self.accept_threshold = float(accept_threshold)
        self.review_threshold = float(review_threshold)

        self.model_loaded = False
        self.model_bias = 0.0
        self.model_weights: dict[str, float] = {}
        self.model_feature_order: list[str] = []
        self.model_feature_mean: dict[str, float] = {}
        self.model_feature_std: dict[str, float] = {}
        self.window_before = 0
        self.window_after = 0
        self.contact_type_filter: set[str] = set()

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

        raw_contact_filter = data.get("contact_type_filter", [])
        if isinstance(raw_contact_filter, str):
            raw_contact_filter = [raw_contact_filter]
        self.contact_type_filter = {normalize_contact_type(item) for item in raw_contact_filter if str(item).strip()}

        self.accept_threshold = float(data.get("accept_threshold", self.accept_threshold))
        self.review_threshold = float(data.get("review_threshold", self.review_threshold))
        self.window_before = int(data.get("context_before", 0))
        self.window_after = int(data.get("context_after", 0))
        self.model_loaded = True

    def contact_type_allowed(self, contact_type: str) -> bool:
        if not self.contact_type_filter:
            return True
        return normalize_contact_type(contact_type) in self.contact_type_filter

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
        z = max(-35.0, min(35.0, z))
        return 1.0 / (1.0 + math.exp(-z))

    def _score_heuristic(self, features: Dict[str, Any]) -> float:
        total_turn_speed = float(features.get("candidate_total_turn_speed", 0.0))
        vertical_delta = float(features.get("candidate_vertical_delta", 0.0))
        cos_turn = float(features.get("candidate_turn_cos", 1.0))
        y_sign_flip = 1.0 if bool(features.get("rule_y_velocity_sign_flip", False)) else 0.0
        source_quality = float(features.get("source_quality", 0.7))

        speed_norm = min(1.0, total_turn_speed / 70.0)
        vertical_norm = min(1.0, vertical_delta / 40.0)
        turn_norm = min(1.0, max(0.0, (0.25 - cos_turn) / 1.25))

        score = (
            0.28 * speed_norm
            + 0.27 * vertical_norm
            + 0.18 * turn_norm
            + 0.17 * y_sign_flip
            + 0.10 * min(1.0, max(0.0, source_quality))
        )
        return min(1.0, max(0.0, score))
