# Doctor 7-Slide Data Pack (Ready Content)

Use this directly to build slides or to verify AI-generated slides.

## Slide 1 — Problem, Objective, Scope
- Problem: padel ball is tiny, fast, frequently blurred, and perspective-dependent.
- Objective: detect and track ball, classify shot zones, and export usable analytics.
- Scope delivered: full pipeline + fallback tracking + edge deployment path.
- Pipeline summary: YOLO → Optical Flow → Kalman → trajectory/metrics/video outputs.

Speaker note cues:
- Emphasize real-world sports constraints (motion, scale, occlusion).
- State this is an engineering + validation thesis, not only a model-training task.

## Slide 2 — Architecture and Stack
- Modules: detection, tracking, filters, evaluation, visualization/output.
- Stack: Python 3.10, Ultralytics YOLOv8, OpenCV, FilterPy, ONNX Runtime.
- Models: `best_ball.pt`, `best_players.pt`, `best_court.pt` and ONNX exports.
- Core robustness design: fallback chain keeps output alive when YOLO misses.

Speaker note cues:
- Explain why modular design helped iterative fixes and deployment adaptation.

## Slide 3 — Baseline Performance and Bottlenecks
Key baseline summary (640px):
- Video 1: 40.6% (YOLO 46.9%, Optical Flow 53.1%).
- Video 2: 74.0% (best case).
- Video 3: 15.1% (worst case).
- Video 5: 39.6%.
- Video 6: 24.4%.
- Average ≈ 38% (excluding best-case Video 2 in interpretation).

Bottlenecks:
- Motion blur and tiny object size.
- Spatial bias, especially weak bottom-left coverage on some videos.

Speaker note cues:
- Show this is the starting point that motivated system-level changes.

## Slide 4 — Experiments and Decisions (including failures)
- Fine-tuning (Video 3 dataset): no meaningful gain → not deployed.
- CLAHE (Video 5): detections 237 → 227 and speed drop ~10.2 FPS → 7.9 FPS → rejected.
- Region-adaptive confidence (Video 6): no bottom-left recovery → rejected.
- 1280 inference: +bottom-left coverage (Video 6 gained 37), but speed and total detections dropped → kept as optional trade-off mode.

Speaker note cues:
- Highlight scientific value of negative results and evidence-based pruning.

## Slide 5 — Edge Deployment Progress (Pi 5 + Hailo)
- Edge pipeline refactored for deployment constraints.
- First successful Pi edge run documented at ~5.3 FPS and 97.6% detections (Video 1 run profile).
- Hailo critical finding: ONNX→HEF compilation tools not available on ARM64 Pi (x86_64 required).
- Hailo runtime verified with precompiled model at ~154.96 FPS.

Speaker note cues:
- Explain this as practical deployment realism, not a blocker to thesis contribution.

## Slide 6 — Contact Validation Milestone (latest)
- Added: clean trajectory + rule-based contact candidates + manual GUI labeler.
- Added contact-type labeling: racket/ground/glass/out_of_frame.
- Video 5 tuning: v1 72.0% (18/25) → v2 90.0% (18/20).
- Video 6 tuning: v2 56.604% (30/53) → v3 73.684% (28/38), with fewer false positives.
- Error signal: wrong detections concentrated in `unspecified` contact type.
- Latest rerun: Video 5 `v5_after_tune_v3` on Pi with 5-second snapshots verified at 5s/10s/15s + final remainder frame 598.

Speaker note cues:
- Emphasize measurable quality improvement and better review workflow.

## Slide 7 — Current Status, Contribution, Next Steps
Completed:
- End-to-end system, experiments, deployment validation, detailed documentation.

Core contributions:
- Robust fallback architecture.
- Quantified accuracy-speed-coverage trade-offs.
- Deployment-aware engineering under hardware/toolchain constraints.
- Validated contact-event workflow with measurable gains.

Immediate next steps:
- Final benchmark chapter and comparisons.
- Optional custom Hailo compile via x86_64 environment.
- Improve handling/filtering of ambiguous (`unspecified`) contacts.
- Finalize thesis writing + demo narrative.

Suggested closing sentence:
- The project evolved from proof-of-concept detection into a validated, deployment-aware sports CV pipeline with clear and evidence-backed next milestones.