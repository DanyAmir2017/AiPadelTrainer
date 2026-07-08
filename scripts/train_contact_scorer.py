"""
Train a lightweight logistic contact scorer from labeled contact CSV files.

Usage example:
  python scripts/train_contact_scorer.py \
    --labels outputs/edge/labels/v5_after_tune_v2_hit_candidates_labels.csv outputs/edge/labels/v6_after_tune_v3_hit_candidates_labels.csv \
    --candidates-dir outputs/edge/hit_candidates \
    --output-model outputs/edge/contact_models/contact_scorer_v1.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import math
from typing import Dict, List, Tuple

import numpy as np


FEATURE_ORDER = [
    "score",
    "source_quality",
    "speed_before",
    "speed_after",
    "total_turn_speed",
    "vertical_delta",
    "turn_cos",
    "abs_v1y",
    "abs_v2y",
    "rule_y_velocity_sign_flip",
    "rule_sharp_direction_change",
]


def _to_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def load_candidates(path: Path) -> Dict[int, dict]:
    by_frame: Dict[int, dict] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            frame = int(float(row["Frame"]))
            by_frame[frame] = row
    return by_frame


def infer_candidate_path(label_path: Path, candidates_dir: Path) -> Path:
    # v6_after_tune_v3_hit_candidates_labels.csv -> v6_after_tune_v3_hit_candidates.csv
    name = label_path.name.replace("_labels.csv", ".csv")
    return candidates_dir / name


def candidate_to_features(row: dict) -> dict:
    rule = (row.get("Rule") or "").strip().lower()
    feat = {
        "score": _to_float(row.get("Score", 0.0)),
        "source_quality": _to_float(row.get("SourceQuality", 0.0)),
        "speed_before": _to_float(row.get("SpeedBefore", 0.0)),
        "speed_after": _to_float(row.get("SpeedAfter", 0.0)),
        "total_turn_speed": _to_float(row.get("TotalTurnSpeed", 0.0)),
        "vertical_delta": _to_float(row.get("VerticalDelta", 0.0)),
        "turn_cos": _to_float(row.get("TurnCos", 0.0)),
        "abs_v1y": _to_float(row.get("AbsV1Y", 0.0)),
        "abs_v2y": _to_float(row.get("AbsV2Y", 0.0)),
        "rule_y_velocity_sign_flip": 1.0 if rule == "y_velocity_sign_flip" else 0.0,
        "rule_sharp_direction_change": 1.0 if rule == "sharp_direction_change" else 0.0,
    }
    return feat


def load_training_rows(label_paths: List[Path], candidates_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    x_rows: List[List[float]] = []
    y_rows: List[float] = []

    for label_path in label_paths:
        candidate_path = infer_candidate_path(label_path, candidates_dir)
        if not candidate_path.exists():
            print(f"[WARN] Missing candidates file for labels: {label_path} -> {candidate_path}")
            continue

        candidates = load_candidates(candidate_path)

        with label_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("Label") or "").strip().lower()
                if label not in {"correct", "wrong"}:
                    continue

                frame = int(float(row["Frame"]))
                c_row = candidates.get(frame)
                if c_row is None:
                    continue

                feat = candidate_to_features(c_row)
                x_rows.append([feat[name] for name in FEATURE_ORDER])
                y_rows.append(1.0 if label == "correct" else 0.0)

    if not x_rows:
        raise RuntimeError("No matched labeled rows found for training.")

    return np.array(x_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64)


def fit_logistic_regression(
    x: np.ndarray,
    y: np.ndarray,
    epochs: int = 1800,
    lr: float = 0.05,
    l2: float = 1e-3,
):
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
        grad_b = error.mean()

        w -= lr * grad_w
        b -= lr * grad_b

    return w, b, mean, std


def probabilities(x: np.ndarray, w: np.ndarray, b: float, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    x_norm = (x - mean) / std
    z = np.clip(x_norm @ w + b, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def metrics_at_threshold(y_true: np.ndarray, prob: np.ndarray, threshold: float) -> dict:
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


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> float:
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


def main():
    parser = argparse.ArgumentParser(description="Train contact scorer model from labeled contact candidates")
    parser.add_argument("--labels", nargs="+", required=True, help="Paths to *_hit_candidates_labels.csv files")
    parser.add_argument("--candidates-dir", type=str, default="outputs/edge/hit_candidates", help="Directory containing candidate CSV files")
    parser.add_argument("--output-model", type=str, default="outputs/edge/contact_models/contact_scorer_v1.json", help="Output model JSON path")
    parser.add_argument("--epochs", type=int, default=1800)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    label_paths = [Path(p) for p in args.labels]
    candidates_dir = Path(args.candidates_dir)
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    x, y = load_training_rows(label_paths, candidates_dir)
    w, b, mean, std = fit_logistic_regression(x, y, epochs=args.epochs, lr=args.lr)
    prob = probabilities(x, w, b, mean, std)

    accept_t = best_threshold(y, prob)
    review_t = max(0.20, accept_t - 0.17)

    m = metrics_at_threshold(y, prob, accept_t)

    model = {
        "model_type": "logistic_regression",
        "version": 1,
        "feature_order": FEATURE_ORDER,
        "weights": {name: float(w[i]) for i, name in enumerate(FEATURE_ORDER)},
        "bias": float(b),
        "feature_mean": {name: float(mean[i]) for i, name in enumerate(FEATURE_ORDER)},
        "feature_std": {name: float(std[i]) for i, name in enumerate(FEATURE_ORDER)},
        "accept_threshold": float(accept_t),
        "review_threshold": float(review_t),
        "training_summary": {
            "samples": int(len(y)),
            "positive": int((y == 1).sum()),
            "negative": int((y == 0).sum()),
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "tp": m["tp"],
            "fp": m["fp"],
            "tn": m["tn"],
            "fn": m["fn"],
        },
    }

    output_model.write_text(json.dumps(model, indent=2), encoding="utf-8")

    print("Saved model:", output_model)
    print("Samples:", len(y))
    print("Accept threshold:", round(accept_t, 4), "Review threshold:", round(review_t, 4))
    print("Train metrics @accept:")
    print(
        "  acc={:.3f} prec={:.3f} rec={:.3f} f1={:.3f} tp={} fp={} tn={} fn={}".format(
            m["accuracy"],
            m["precision"],
            m["recall"],
            m["f1"],
            int(m["tp"]),
            int(m["fp"]),
            int(m["tn"]),
            int(m["fn"]),
        )
    )


if __name__ == "__main__":
    main()
