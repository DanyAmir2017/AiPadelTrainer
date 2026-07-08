"""Train a windowed contact scorer from labeled contact CSVs.

This trainer uses a short before/after trajectory window around each candidate,
so the model can learn motion patterns around the contact instead of only the
single candidate row.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge.windowed_contact_scoring import (
    build_feature_order,
    build_window_features,
    best_threshold,
    fit_logistic_regression,
    infer_candidate_path,
    infer_clean_trajectory_path,
    load_candidate_rows,
    load_clean_trajectory,
    metrics_at_threshold,
    probabilities,
)


def _load_training_rows(
    label_paths: List[Path],
    candidates_dir: Path,
    clean_dir: Path,
    context_before: int,
    context_after: int,
    contact_type_filter: set[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    x_rows: List[List[float]] = []
    y_rows: List[float] = []
    feature_order = build_feature_order(context_before, context_after)

    for label_path in label_paths:
        candidate_path = infer_candidate_path(label_path, candidates_dir)
        clean_path = infer_clean_trajectory_path(candidate_path, clean_dir)

        if not candidate_path.exists():
            print(f"[WARN] Missing candidate CSV for labels: {label_path} -> {candidate_path}")
            continue
        if not clean_path.exists():
            print(f"[WARN] Missing clean trajectory for labels: {label_path} -> {clean_path}")
            continue

        candidates = {int(float(row["Frame"])): row for row in load_candidate_rows(candidate_path)}
        clean_by_frame = load_clean_trajectory(clean_path)

        with label_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = (row.get("Label") or "").strip().lower()
                if label not in {"correct", "wrong"}:
                    continue

                contact_type = (row.get("ContactType") or row.get("contact_type") or "").strip().lower()
                if contact_type_filter and label == "correct" and contact_type not in contact_type_filter:
                    # For ground-only training, non-ground positives become negatives.
                    label = "wrong"

                frame = int(float(row["Frame"]))
                c_row = candidates.get(frame)
                if c_row is None:
                    continue

                feat = build_window_features(c_row, clean_by_frame, context_before, context_after)
                x_rows.append([feat[name] for name in feature_order])
                y_rows.append(1.0 if label == "correct" else 0.0)

    if not x_rows:
        raise RuntimeError("No matched labeled rows found for training.")

    return np.array(x_rows, dtype=np.float64), np.array(y_rows, dtype=np.float64), feature_order


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a windowed contact scorer model from labeled contact candidates")
    parser.add_argument("--labels", nargs="+", required=True, help="Paths to *_hit_candidates_labels.csv files")
    parser.add_argument("--candidates-dir", type=str, default="outputs/edge/hit_candidates", help="Directory containing candidate CSV files")
    parser.add_argument("--clean-dir", type=str, default="outputs/edge/clean_csv", help="Directory containing cleaned trajectory CSV files")
    parser.add_argument("--output-model", type=str, default="outputs/edge/contact_models/windowed_contact_scorer_v1.json", help="Output model JSON path")
    parser.add_argument("--context-before", type=int, default=5, help="Rows before candidate to include")
    parser.add_argument("--context-after", type=int, default=5, help="Rows after candidate to include")
    parser.add_argument("--contact-type-filter", nargs="+", default=["ground"], help="Contact types to treat as positive during training")
    parser.add_argument("--epochs", type=int, default=2200)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()

    label_paths = [Path(p) for p in args.labels]
    candidates_dir = Path(args.candidates_dir)
    clean_dir = Path(args.clean_dir)
    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    contact_type_filter = {str(item).strip().lower() for item in args.contact_type_filter if str(item).strip()}

    x, y, feature_order = _load_training_rows(
        label_paths,
        candidates_dir,
        clean_dir,
        args.context_before,
        args.context_after,
        contact_type_filter,
    )

    w, b, mean, std = fit_logistic_regression(x, y, epochs=args.epochs, lr=args.lr, l2=args.l2)
    prob = probabilities(x, w, b, mean, std)

    accept_t = best_threshold(y, prob)
    review_t = max(0.20, accept_t - 0.17)
    m = metrics_at_threshold(y, prob, accept_t)

    model = {
        "model_type": "windowed_logistic_regression",
        "version": 1,
        "context_before": int(args.context_before),
        "context_after": int(args.context_after),
        "feature_order": feature_order,
        "weights": {name: float(w[i]) for i, name in enumerate(feature_order)},
        "bias": float(b),
        "feature_mean": {name: float(mean[i]) for i, name in enumerate(feature_order)},
        "feature_std": {name: float(std[i]) for i, name in enumerate(feature_order)},
        "accept_threshold": float(accept_t),
        "review_threshold": float(review_t),
        "contact_type_filter": sorted(contact_type_filter),
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
    print("Context window:", f"before={args.context_before}", f"after={args.context_after}")
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
