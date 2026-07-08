# Doctor 7-Slide Master File (Prompt + Data)

Use this **single file** to generate the final deck. Each slide includes:
1) what the model should generate, and
2) the exact project data to use.

---

## Global Prompt Header (copy this before slide sections)

Create a **7-slide PowerPoint presentation** for my thesis doctor/supervisor.

**Project Title:** Padel Trainer - Computer Vision Ball Detection System  
**Institution:** German University in Cairo (GUC)  
**Author:** Daniel  
**Period Covered:** From project start until current status (April 2026)

### Global requirements
- Tone: professional, technical, concise.
- Audience: thesis doctor (expects evidence and decisions).
- Keep slides clean: short bullets, clear heading, one visual/table suggestion per slide.
- Include speaker notes for each slide (4–6 lines).
- Do not invent experiments or numbers.

---

## Slide 1 — Problem, Objective, and Scope

### Generate
- Explain problem difficulty, objective, scope, and pipeline summary.

### Data (use exactly)
- Problem: padel ball is tiny, fast, frequently blurred, and perspective-dependent.
- Objective: detect and track ball, classify shot zones, and export usable analytics.
- Scope delivered: full pipeline + fallback tracking + edge deployment path.
- Pipeline summary: YOLO → Optical Flow → Kalman → trajectory/metrics/video outputs.

### Speaker note cues
- Emphasize real-world sports constraints (motion, scale, occlusion).
- State this is an engineering + validation thesis, not only model training.

---

## Slide 2 — Architecture and Technical Stack

### Generate
- Show modular architecture and core technologies.

### Data (use exactly)
- Modules: detection, tracking, filters, evaluation, visualization/output.
- Stack: Python 3.10, Ultralytics YOLOv8, OpenCV, FilterPy, ONNX Runtime.
- Models: `best_ball.pt`, `best_players.pt`, `best_court.pt` + ONNX exports.
- Core robustness design: fallback chain keeps output alive when YOLO misses.

### Speaker note cues
- Explain why modular design enabled iterative fixes and deployment adaptation.

---

## Slide 3 — Baseline Performance and Bottlenecks

### Generate
- Present baseline results and identify key bottlenecks.

### Data (use exactly)
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

### Speaker note cues
- Show this baseline motivated system-level changes.

---

## Slide 4 — Experiments and Decisions (Including Failures)

### Generate
- Summarize major experiments in “Hypothesis → Result → Decision” style.

### Data (use exactly)
- Fine-tuning (Video 3 dataset): no meaningful gain → not deployed.
- CLAHE (Video 5): detections 237 → 227 and speed ~10.2 FPS → 7.9 FPS → rejected.
- Region-adaptive confidence (Video 6): no bottom-left recovery → rejected.
- 1280 inference: +bottom-left coverage (Video 6 gained 37), but speed and total detections dropped → optional trade-off mode.

### Speaker note cues
- Highlight scientific value of negative results and evidence-based pruning.

---

## Slide 5 — Edge Deployment Progress (Pi 5 + Hailo)

### Generate
- Explain edge adaptation progress, constraints, and verified acceleration evidence.

### Data (use exactly)
- Edge pipeline refactored for deployment constraints.
- First successful Pi edge run documented at ~5.3 FPS and 97.6% detections (Video 1 run profile).
- Hailo critical finding: ONNX→HEF compilation tools not available on ARM64 Pi (x86_64 required).
- Hailo runtime verified with precompiled model at ~154.96 FPS.

### Speaker note cues
- Explain this is deployment realism, not a blocker to thesis contribution.

---

## Slide 6 — Contact Validation Milestone (Most Recent)

### Generate
- Present contact workflow additions and measured tuning improvements.

### Data (use exactly)
- Added: clean trajectory + rule-based contact candidates + manual GUI labeler.
- Added contact-type labeling: racket/ground/glass/out_of_frame.
- Video 5 tuning: v1 72.0% (18/25) → v2 90.0% (18/20).
- Video 6 tuning: v2 56.604% (30/53) → v3 73.684% (28/38), with fewer false positives.
- Error signal: wrong detections concentrated in `unspecified` contact type.
- Latest rerun: Video 5 `v5_after_tune_v3` on Pi with 5-second snapshots verified at 5s/10s/15s + final remainder frame 598.

### Speaker note cues
- Emphasize measurable quality gains and stronger review workflow.

---

## Slide 7 — Current Status, Contribution, and Next Steps

### Generate
- Conclude with completion status, contributions, and a supervisor-facing ask.

### Data (use exactly)
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

---

## Final Output Format (for AI slide generation)
- For each of the 7 slides, return:
  1) Slide title
  2) 4–6 concise bullets
  3) One visual suggestion
  4) Speaker notes (4–6 lines)