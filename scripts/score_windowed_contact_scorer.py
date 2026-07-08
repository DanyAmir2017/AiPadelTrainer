"""Score candidate CSVs with a windowed trajectory-based contact model."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.edge.windowed_contact_scoring import (
    WindowedContactScorer,
    build_window_features,
    infer_clean_trajectory_path,
    load_candidate_rows,
    load_clean_trajectory,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score contact candidates using a windowed CSV model")
    parser.add_argument("--model", required=True, help="Path to the trained windowed scorer JSON")
    parser.add_argument("--candidates-csv", required=True, help="Candidate CSV path")
    parser.add_argument("--clean-csv", default=None, help="Clean trajectory CSV path (optional; inferred if omitted)")
    parser.add_argument("--output-csv", default=None, help="Output scored CSV path")
    parser.add_argument("--context-before", type=int, default=None, help="Override model context-before")
    parser.add_argument("--context-after", type=int, default=None, help="Override model context-after")
    parser.add_argument("--contact-types", nargs="+", default=None, help="Only keep these contact types in scored output (default: ground if not set in model)")
    parser.add_argument("--keep-rejected", action="store_true", help="Keep rejected rows in the output")
    args = parser.parse_args()

    model_path = Path(args.model)
    candidates_csv = Path(args.candidates_csv)
    if not candidates_csv.exists():
        raise FileNotFoundError(f"Candidate CSV not found: {candidates_csv}")

    scorer = WindowedContactScorer(model_path=model_path)
    context_before = args.context_before if args.context_before is not None else scorer.window_before or 5
    context_after = args.context_after if args.context_after is not None else scorer.window_after or 5
    contact_types = {str(item).strip().lower() for item in (args.contact_types or scorer.contact_type_filter or {"ground"})}

    clean_csv = Path(args.clean_csv) if args.clean_csv else infer_clean_trajectory_path(candidates_csv)
    if not clean_csv.exists():
        raise FileNotFoundError(f"Clean trajectory CSV not found: {clean_csv}")

    output_csv = Path(args.output_csv) if args.output_csv else candidates_csv.with_name(f"{candidates_csv.stem}_window_scored.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    candidates = load_candidate_rows(candidates_csv)
    clean_by_frame = load_clean_trajectory(clean_csv)

    scored_rows = []
    for row in candidates:
        candidate_contact_type = (row.get("ContactType") or row.get("contact_type") or "").strip().lower()
        if contact_types and candidate_contact_type and candidate_contact_type not in contact_types:
            continue

        feat = build_window_features(row, clean_by_frame, context_before, context_after)
        decision = scorer.score(feat)
        if decision.decision == "reject" and not args.keep_rejected:
            continue

        out_row = dict(row)
        out_row["Score"] = f"{decision.score:.3f}"
        out_row["Decision"] = decision.decision
        out_row["ContextBefore"] = str(context_before)
        out_row["ContextAfter"] = str(context_after)
        out_row["ContactTypeFilter"] = ",".join(sorted(contact_types))
        for key in scorer.model_feature_order:
            out_row[key] = f"{float(feat.get(key, 0.0)):.6f}"
        scored_rows.append(out_row)

    fieldnames = list(candidates[0].keys()) if candidates else ["Frame", "Second", "X", "Y", "Rule"]
    for extra in ["Score", "Decision", "ContextBefore", "ContextAfter", "ContactTypeFilter"]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    for key in scorer.model_feature_order:
        if key not in fieldnames:
            fieldnames.append(key)

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored_rows)

    print(f"Saved scored candidates: {output_csv}")
    print(f"Rows written: {len(scored_rows)}")
    print(f"Context window: before={context_before}, after={context_after}")


if __name__ == "__main__":
    main()
