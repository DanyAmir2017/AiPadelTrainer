from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Tuple


def load_labels(path: Path) -> Dict[int, int]:
    out: Dict[int, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            label = (row.get("Label") or "").strip().lower()
            if label not in {"correct", "wrong"}:
                continue
            frame = int(float(row["Frame"]))
            out[frame] = 1 if label == "correct" else 0
    return out


def load_scored_candidates(path: Path) -> Dict[int, Tuple[float, str]]:
    out: Dict[int, Tuple[float, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            frame = int(float(row["Frame"]))
            score = float(row.get("Score", 0.0))
            decision = (row.get("Decision") or "").strip().lower()
            out[frame] = (score, decision)
    return out


def load_baseline_frames(path: Path) -> set[int]:
    frames = set()
    with path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            frames.add(int(float(row["Frame"])))
    return frames


def _match_with_tolerance(predicted_frames: set[int], true_frames: set[int], tol: int) -> Tuple[int, int, int]:
    matched_true = set()
    tp = 0
    fp = 0

    for p in sorted(predicted_frames):
        candidates = [t for t in true_frames if abs(t - p) <= tol and t not in matched_true]
        if candidates:
            t = min(candidates, key=lambda x: abs(x - p))
            matched_true.add(t)
            tp += 1
        else:
            fp += 1

    fn = len(true_frames - matched_true)
    return tp, fp, fn


def metrics_from_predictions(y_true: Dict[int, int], predicted_positive: set[int], tolerance_frames: int = 0):
    positives = {f for f, y in y_true.items() if y == 1}
    negatives = {f for f, y in y_true.items() if y == 0}

    if tolerance_frames > 0:
        tp, fp, fn = _match_with_tolerance(predicted_positive, positives, tolerance_frames)
        # Approximate TN by negatives not directly selected (exact negative hits remain counted as FP)
        tn = max(0, len(negatives) - fp)
    else:
        all_frames = set(y_true.keys())
        tp = len([f for f in all_frames if y_true[f] == 1 and f in predicted_positive])
        fp = len([f for f in all_frames if y_true[f] == 0 and f in predicted_positive])
        tn = len([f for f in all_frames if y_true[f] == 0 and f not in predicted_positive])
        fn = len([f for f in all_frames if y_true[f] == 1 and f not in predicted_positive])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": acc,
    }


def pick_threshold(
    y_true: Dict[int, int],
    scored: Dict[int, Tuple[float, str]],
    tolerance_frames: int = 0,
    min_recall: float = 0.0,
):
    values = sorted(set(score for _f, (score, _d) in scored.items()))
    if not values:
        return 0.62

    best_t = values[0]
    best = None
    for t in values:
        pred = {f for f, (s, _d) in scored.items() if s >= t}
        m = metrics_from_predictions(y_true, pred, tolerance_frames=tolerance_frames)
        if m["recall"] < min_recall:
            continue
        # prioritize F1, then precision
        key = (m["f1"], m["precision"], m["recall"])
        if best is None or key > best:
            best = key
            best_t = t

    if best is None:
        for t in values:
            pred = {f for f, (s, _d) in scored.items() if s >= t}
            m = metrics_from_predictions(y_true, pred, tolerance_frames=tolerance_frames)
            key = (m["f1"], m["precision"], m["recall"])
            if best is None or key > best:
                best = key
                best_t = t
    return float(best_t)


def pick_accept_threshold_from_review(
    y_true: Dict[int, int],
    scored: Dict[int, Tuple[float, str]],
    review_threshold: float,
):
    # Accept threshold does not affect filtering, only accept vs review split for UI/analysis.
    # Choose a stricter band above review while keeping enough accepted positives.
    candidate_scores = sorted(
        [s for f, (s, _d) in scored.items() if f in y_true and s >= review_threshold]
    )
    if not candidate_scores:
        return max(0.0, min(0.98, review_threshold + 0.15))

    q75_idx = int(0.75 * (len(candidate_scores) - 1))
    q75 = candidate_scores[q75_idx]
    accept_candidate = max(review_threshold + 0.12, q75)
    return float(max(0.0, min(0.98, accept_candidate)))


def update_model_thresholds(model_path: Path, accept_t: float, review_t: float):
    data = json.loads(model_path.read_text(encoding="utf-8"))
    data["accept_threshold"] = float(accept_t)
    data["review_threshold"] = float(review_t)
    model_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description="Tune contact thresholds and create v3 vs scored-v4 report")
    p.add_argument("--labels", required=True)
    p.add_argument("--baseline-v3", required=True)
    p.add_argument("--scored-all", required=True)
    p.add_argument("--scored-filtered", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--tolerance-frames", type=int, default=2)
    p.add_argument("--min-recall", type=float, default=0.90)
    args = p.parse_args()

    labels = load_labels(Path(args.labels))
    baseline_v3 = load_baseline_frames(Path(args.baseline_v3))
    scored_all = load_scored_candidates(Path(args.scored_all))

    tuned_review = pick_threshold(
        labels,
        scored_all,
        tolerance_frames=args.tolerance_frames,
        min_recall=max(0.0, min(1.0, args.min_recall)),
    )
    tuned_accept = pick_accept_threshold_from_review(labels, scored_all, tuned_review)
    update_model_thresholds(Path(args.model), tuned_accept, tuned_review)

    # metrics: baseline v3 and scored filtered
    baseline_metrics = metrics_from_predictions(
        labels,
        set(baseline_v3),
        tolerance_frames=args.tolerance_frames,
    )

    scored_filtered_frames = load_baseline_frames(Path(args.scored_filtered))
    scored_metrics = metrics_from_predictions(
        labels,
        set(scored_filtered_frames),
        tolerance_frames=args.tolerance_frames,
    )

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = []
    report.append("# Contact Scoring Comparison Report (V3 vs Scored-V4)\n")
    report.append("## Threshold Tuning")
    report.append(f"- Tuned review threshold (filter cutoff): `{tuned_review:.4f}`")
    report.append(f"- Tuned accept threshold (accept/review split): `{tuned_accept:.4f}`")
    report.append(f"- Minimum recall target used: `{max(0.0, min(1.0, args.min_recall)):.3f}`")
    report.append(f"- Model updated: `{args.model}`\n")

    report.append("## Evaluation Set")
    report.append(f"- Labels file: `{args.labels}`")
    report.append(f"- Labeled samples: `{len(labels)}`\n")
    report.append(f"- Matching tolerance: `±{args.tolerance_frames}` frames")
    report.append("")

    report.append("## Side-by-Side Metrics")
    report.append("| Metric | V3 (rule-only) | Scored-V4 | Delta (V4-V3) |")
    report.append("|---|---:|---:|---:|")
    for k in ["tp", "fp", "tn", "fn"]:
        v3 = baseline_metrics[k]
        v4 = scored_metrics[k]
        report.append(f"| {k.upper()} | {v3} | {v4} | {v4 - v3:+} |")
    for k in ["precision", "recall", "f1", "accuracy"]:
        v3 = baseline_metrics[k] * 100.0
        v4 = scored_metrics[k] * 100.0
        report.append(f"| {k.capitalize()} (%) | {v3:.3f} | {v4:.3f} | {v4 - v3:+.3f} |")

    report.append("\n## Candidate Counts")
    report.append(f"- V3 candidates: `{len(baseline_v3)}`")
    report.append(f"- Scored-V4 candidates: `{len(scored_filtered_frames)}`")

    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Report written: {report_path}")


if __name__ == "__main__":
    main()
