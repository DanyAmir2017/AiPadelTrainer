"""
Edge Inference Pipeline - Raspberry Pi 5 + Hailo-8 Optimized

Lightweight ball tracking pipeline for edge deployment.
Single-threaded, minimal memory footprint, CSV output only.
"""

import cv2
import csv
import time
import math
from pathlib import Path
from edge_detector import EdgeBallDetector
from contact_scoring import ContactScorer, source_quality_score
import edge_config as config


class EdgeInferencePipeline:
    """
    Main inference pipeline for edge deployment
    
    Features:
    - Ball detection only
    - Kalman tracking
    - CSV trajectory export
    - Optional annotated video export
    - Performance statistics
    - Optional live preview
    """
    
    def __init__(self, video_path: str, output_name: str = None, contact_labels_csv: str | None = None):
        """
        Initialize inference pipeline
        
        Args:
            video_path: Path to input video
            output_name: Optional output name (default: video filename)
        """
        
        self.video_path = Path(video_path)
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Output name
        if output_name is None:
            output_name = self.video_path.stem
        self.output_name = output_name
        
        # Create output directory tree
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        config.EDGE_CSV_DIR.mkdir(parents=True, exist_ok=True)
        config.EDGE_DETECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        config.EDGE_CLEAN_CSV_DIR.mkdir(parents=True, exist_ok=True)
        config.EDGE_HIT_CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        config.EDGE_ANNOTATED_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        config.EDGE_TRAIL_SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
        config.EDGE_CONTACT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        
        # Output CSV path
        self.csv_path = config.EDGE_CSV_DIR / f"{output_name}_trajectory.csv"
        self.detections_csv_path = config.EDGE_DETECTIONS_DIR / f"{output_name}_detections.csv"
        self.clean_csv_path = config.EDGE_CLEAN_CSV_DIR / f"{output_name}_trajectory_clean.csv"
        self.hit_candidates_csv_path = config.EDGE_HIT_CANDIDATES_DIR / f"{output_name}_hit_candidates.csv"
        self.annotated_video_path = config.EDGE_ANNOTATED_VIDEOS_DIR / f"{output_name}_annotated.mp4"
        self.snapshots_dir = config.EDGE_TRAIL_SNAPSHOTS_DIR / output_name
        
        # Statistics
        self.total_frames = 0
        self.detections = 0
        self.yolo_detections = 0
        self.optical_flow_detections = 0
        self.kalman_predictions = 0
        self.start_time = None
        self.saved_snapshots = 0

        # Trail settings for annotated video/preview
        self.trail_points = []  # cleaned trail list of (x, y) or None (segment break)
        self.trail_max_len = 120
        self.frames_without_ball = 0
        self.trail_reset_after = 20
        self.trail_gap_after_misses = 2
        self.trail_last_valid_point = None
        self.trail_max_jump_px = 150

        # Clean trajectory state (for snapshots + rule-based contact checks)
        self.clean_smooth_alpha = 0.45
        self.clean_break_jump_px = 100
        self.clean_spike_jump_px = 180
        self.smooth_last_point = None
        self.clean_points_history = []  # (frame, x, y) for hit/event checks
        self.clean_points_full = []     # (frame, x, y) valid points only
        self.detection_points_history = []  # (frame, x, y, source) for hit/event checks
        self.detection_points_full = []     # raw detections only
        self.clean_track_broken = False

        # Basic hit/contact candidate rule config
        self.hit_candidate_records = []
        self.hit_candidates = []
        self.hit_candidates_rejected = 0
        self.hit_candidates_review = 0
        self.hit_candidates_accepted = 0
        self.hit_min_speed_px = 6.0
        self.hit_min_vertical_speed_px = 6.0
        self.hit_turn_cos_threshold = -0.15
        self.hit_min_vertical_delta_px = 8.0
        self.hit_min_total_turn_speed_px = 14.0
        self.hit_cooldown_frames = 8
        self.last_hit_frame_by_source = {'clean': -100000, 'raw': -100000}

        # Keep only configured collision/contact types (default: ground only).
        raw_filter = getattr(config, "CONTACT_TYPE_FILTER", None)
        self.contact_type_filter = {str(item).strip().lower() for item in raw_filter} if raw_filter else set()
        self.contact_type_frame_tolerance = int(getattr(config, "CONTACT_TYPE_FRAME_TOLERANCE", 0) or 0)
        self.skipped_non_ground_candidates = 0

        # Optional manual contact types (from labeling GUI)
        self.contact_type_by_frame = {}
        inferred_labels_csv = self.hit_candidates_csv_path.parent.parent / "labels" / f"{self.hit_candidates_csv_path.stem}_labels.csv"
        labels_csv_path = Path(contact_labels_csv) if contact_labels_csv else inferred_labels_csv
        if labels_csv_path.exists():
            self._load_contact_types_csv(labels_csv_path)

        # Collision marker overlay (drawn on annotated video)
        self.collision_markers = []  # list of dicts: frame, x, y, kind, label, color
        self.collision_marker_styles = {
            # Manual contact types
            'racket': ((0, 255, 0), 'RACKET'),
            'ground': ((0, 165, 255), 'GROUND'),
            'glass': ((255, 0, 255), 'GLASS'),
            'out_of_frame': ((0, 0, 255), 'OUT'),
            # Rule / fallback types
            'y_velocity_sign_flip': ((0, 215, 255), 'Y-FLIP'),
            'sharp_direction_change': ((255, 128, 0), 'TURN'),
            # Scoring decisions
            'accept': ((0, 200, 0), 'ACCEPT'),
            'review': ((0, 220, 255), 'REVIEW'),
            'reject': ((150, 150, 150), 'REJECT'),
            'unknown': ((255, 255, 255), 'COLLISION'),
        }

        self.contact_scorer = ContactScorer(
            model_path=config.EDGE_CONTACT_MODEL_PATH,
            accept_threshold=config.CONTACT_ACCEPT_THRESHOLD,
            review_threshold=config.CONTACT_REVIEW_THRESHOLD,
            min_total_turn_speed_px=self.hit_min_total_turn_speed_px,
            min_vertical_delta_px=self.hit_min_vertical_delta_px,
        )
        
        # Initialize detector
        print("\nInitializing Edge Ball Detector...")
        self.detector = EdgeBallDetector()
        print("✓ Ready for inference\n")
    
    def save_trajectory_csv(self, trajectory: list):
        """
        Save trajectory to CSV file
        
        Args:
            trajectory: List of (frame_num, x, y, source) tuples
        """
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'X', 'Y', 'Source'])
            
            for frame_num, x, y, source in trajectory:
                writer.writerow([frame_num, x, y, source])
        
        if config.VERBOSE:
            print(f"\n✓ Trajectory saved: {self.csv_path}")

    def save_clean_trajectory_csv(self):
        """Save cleaned trail points for downstream trajectory analysis."""
        with open(self.clean_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'X', 'Y'])
            for frame_num, x, y in self.clean_points_full:
                writer.writerow([frame_num, x, y])

        if config.VERBOSE:
            print(f"✓ Clean trajectory saved: {self.clean_csv_path}")

    def save_detection_csv(self):
        """Save raw detections for contact detection and debugging."""
        with open(self.detections_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'X', 'Y', 'Source'])
            for frame_num, x, y, source in self.detection_points_full:
                writer.writerow([frame_num, x, y, source])

        if config.VERBOSE:
            print(f"✓ Detection CSV saved: {self.detections_csv_path}")

    def save_hit_candidates_csv(self):
        """Save rule-based contact candidate events."""
        with open(self.hit_candidates_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Frame', 'Second', 'X', 'Y', 'Rule',
                'ContactType',
                'Score', 'Decision', 'SourceQuality',
                'SpeedBefore', 'SpeedAfter', 'TotalTurnSpeed',
                'VerticalDelta', 'TurnCos', 'AbsV1Y', 'AbsV2Y',
            ])
            for event in self.hit_candidates:
                writer.writerow([
                    event['frame'],
                    f"{event['second']:.3f}",
                    event['x'],
                    event['y'],
                    event['rule'],
                    event.get('contact_type', ''),
                    f"{event['score']:.3f}",
                    event['decision'],
                    f"{event['source_quality']:.3f}",
                    f"{event['speed_before']:.3f}",
                    f"{event['speed_after']:.3f}",
                    f"{event['total_turn_speed']:.3f}",
                    f"{event['vertical_delta']:.3f}",
                    f"{event['turn_cos']:.6f}",
                    f"{event['abs_v1y']:.3f}",
                    f"{event['abs_v2y']:.3f}",
                ])

        if config.VERBOSE:
            print(f"✓ Hit candidates saved: {self.hit_candidates_csv_path}")

    def _load_contact_types_csv(self, csv_path: Path):
        """Load manual contact types by frame number from a labels CSV."""
        try:
            with open(csv_path, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    frame_text = (row.get('Frame') or row.get('frame') or '').strip()
                    contact_type = (row.get('ContactType') or row.get('contact_type') or '').strip().lower()
                    if not frame_text or not contact_type:
                        continue
                    try:
                        frame = int(float(frame_text))
                    except ValueError:
                        continue
                    self.contact_type_by_frame[frame] = contact_type

            if config.VERBOSE and self.contact_type_by_frame:
                print(f"✓ Loaded contact types for {len(self.contact_type_by_frame)} frames from: {csv_path}")
        except Exception as exc:
            if config.VERBOSE:
                print(f"⚠ Could not load contact types from {csv_path}: {exc}")

    def _draw_trail(self, image, trail_points, color=(255, 0, 255), thickness=2):
        """Draw trail segments, skipping None breaks."""
        if len(trail_points) < 2:
            return

        for i in range(1, len(trail_points)):
            p1 = trail_points[i - 1]
            p2 = trail_points[i]
            if p1 is None or p2 is None:
                continue
            cv2.line(image, p1, p2, color, thickness)

    def _normalize_collision_kind(self, kind: str) -> str:
        kind = (kind or '').strip().lower()
        if not kind:
            return 'unknown'
        kind = kind.replace(' ', '_').replace('-', '_')
        if kind in {'ball_out_of_frame', 'outoframe'}:
            return 'out_of_frame'
        return kind

    def _contact_type_allowed(self, contact_type: str) -> bool:
        if not self.contact_type_filter:
            return True
        contact_type = self._normalize_collision_kind(contact_type)
        return contact_type in self.contact_type_filter

    def _lookup_contact_type(self, frame_num: int) -> str:
        """Return a nearby labeled contact type within the configured frame tolerance."""
        exact = (self.contact_type_by_frame.get(int(frame_num)) or '').strip().lower()
        if exact:
            return exact

        if self.contact_type_frame_tolerance <= 0 or not self.contact_type_by_frame:
            return ''

        best_type = ''
        best_offset = None
        for label_frame, label_type in self.contact_type_by_frame.items():
            offset = abs(int(frame_num) - int(label_frame))
            if offset > self.contact_type_frame_tolerance:
                continue
            if best_offset is None or offset < best_offset:
                best_offset = offset
                best_type = (label_type or '').strip().lower()

        return best_type

    def _get_collision_style(self, kind: str):
        key = self._normalize_collision_kind(kind)
        color, label = self.collision_marker_styles.get(key, self.collision_marker_styles['unknown'])
        return key, color, label

    def _register_collision_marker(self, event: dict):
        """Store a collision marker for annotated-video overlays."""
        kind = self.contact_type_by_frame.get(int(event['frame'])) or event.get('contact_type') or event.get('rule') or event.get('decision') or 'unknown'
        key, color, label = self._get_collision_style(kind)
        self.collision_markers.append({
            'frame': int(event['frame']),
            'x': int(event['x']),
            'y': int(event['y']),
            'kind': key,
            'label': label,
            'color': color,
        })

    def _draw_collision_markers(self, image, frame_num: int):
        """Draw all collision markers up to the current frame."""
        if not self.collision_markers:
            return

        for marker in self.collision_markers:
            if marker['frame'] > frame_num:
                continue

            x = int(marker['x'])
            y = int(marker['y'])
            color = marker['color']

            # Small filled dot + thin ring to make markers visible without obscuring the ball.
            cv2.circle(image, (x, y), 5, color, -1)
            cv2.circle(image, (x, y), 9, (0, 0, 0), 2)
            cv2.putText(image, marker['label'], (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

    def _draw_collision_legend(self, image):
        """Draw a compact legend for collision marker colors."""
        if not self.collision_markers:
            return

        kinds = []
        seen = set()
        for marker in self.collision_markers:
            kind = marker['kind']
            if kind not in seen:
                seen.add(kind)
                kinds.append(kind)

        if not kinds:
            return

        x0, y0 = 10, 135
        box_w, box_h = 18, 12
        padding = 6
        line_h = 20
        panel_w = 235
        panel_h = min(24 + line_h * len(kinds), 210)

        overlay = image.copy()
        cv2.rectangle(overlay, (x0 - 6, y0 - 18), (x0 - 6 + panel_w, y0 - 18 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, image)

        cv2.putText(image, "Collision markers", (x0, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for i, kind in enumerate(kinds):
            color, label = self.collision_marker_styles.get(kind, self.collision_marker_styles['unknown'])
            yy = y0 + 10 + i * line_h
            cv2.rectangle(image, (x0, yy), (x0 + box_w, yy + box_h), color, -1)
            cv2.rectangle(image, (x0, yy), (x0 + box_w, yy + box_h), (255, 255, 255), 1)
            cv2.putText(image, label, (x0 + box_w + padding, yy + 11),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    def _save_trail_snapshot(self, frame, frame_num, video_fps):
        """Save a snapshot image showing the current cleaned trajectory trail."""
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        snapshot = frame.copy()
        self._draw_trail(snapshot, self.trail_points, color=(255, 0, 255), thickness=2)
        self._draw_collision_markers(snapshot, frame_num)
        self._draw_collision_legend(snapshot)

        second = frame_num / video_fps if video_fps > 0 else 0.0
        cv2.putText(snapshot, f"t={second:.1f}s", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(snapshot, f"frame={frame_num}", (10, 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        out_path = self.snapshots_dir / f"trail_{int(second):04d}s_frame_{frame_num:06d}.jpg"
        cv2.imwrite(str(out_path), snapshot)
        self.saved_snapshots += 1

    def _register_hit_candidate(self, frame_num, x, y, video_fps, rule, features, keep_rejected=False, candidate_source='clean'):
        """Record a potential contact event with cooldown to avoid duplicates."""
        last_hit_frame = self.last_hit_frame_by_source.get(candidate_source, -100000)
        if frame_num - last_hit_frame < self.hit_cooldown_frames:
            return

        contact_type = self._lookup_contact_type(frame_num)
        if self.contact_type_filter and not self._contact_type_allowed(contact_type):
            self.skipped_non_ground_candidates += 1
            return

        if config.CONTACT_SCORING_ENABLED:
            decision = self.contact_scorer.score(features)
            score = decision.score
            decision_label = decision.decision
            if decision_label == 'reject':
                if not keep_rejected:
                    return
        else:
            score = 1.0
            decision_label = 'accept'

        second = frame_num / video_fps if video_fps > 0 else 0.0
        self.hit_candidate_records.append({
            'frame': frame_num,
            'second': second,
            'x': x,
            'y': y,
            'rule': rule,
            'contact_type': contact_type,
            'score': score,
            'decision': decision_label,
            'source_quality': float(features.get('source_quality', 0.5)),
            'speed_before': float(features.get('speed_before', 0.0)),
            'speed_after': float(features.get('speed_after', 0.0)),
            'total_turn_speed': float(features.get('total_turn_speed', 0.0)),
            'vertical_delta': float(features.get('vertical_delta', 0.0)),
            'turn_cos': float(features.get('cos_turn', 0.0)),
            'abs_v1y': float(features.get('abs_v1y', 0.0)),
            'abs_v2y': float(features.get('abs_v2y', 0.0)),
            'candidate_source': candidate_source,
        })
        self._register_collision_marker(self.hit_candidate_records[-1])
        self.last_hit_frame_by_source[candidate_source] = frame_num

    def _check_contact_candidate_from_history(self, points_history, video_fps, keep_rejected=False, candidate_source='clean'):
        """Simple rule-based detection of possible bounce/contact events from a point history."""
        if len(points_history) < 3:
            return

        f0, x0, y0, _s0 = points_history[-3]
        f1, x1, y1, _s1 = points_history[-2]
        f2, x2, y2, src2 = points_history[-1]

        if (f1 - f0) > 3 or (f2 - f1) > 3:
            return

        v1x, v1y = x1 - x0, y1 - y0
        v2x, v2y = x2 - x1, y2 - y1

        s1 = math.hypot(v1x, v1y)
        s2 = math.hypot(v2x, v2y)
        if s1 < self.hit_min_speed_px or s2 < self.hit_min_speed_px:
            return

        if (s1 + s2) < self.hit_min_total_turn_speed_px:
            return

        dot = v1x * v2x + v1y * v2y
        denom = s1 * s2
        if denom <= 1e-6:
            return

        cos_turn = max(-1.0, min(1.0, dot / denom))
        vertical_delta = abs(v2y - v1y)
        y_sign_flip = (v1y * v2y < 0) and (abs(v1y) >= self.hit_min_vertical_speed_px) and (abs(v2y) >= self.hit_min_vertical_speed_px)
        sharp_turn = (cos_turn < self.hit_turn_cos_threshold) and (vertical_delta >= self.hit_min_vertical_delta_px)

        features = {
            'speed_before': s1,
            'speed_after': s2,
            'total_turn_speed': (s1 + s2),
            'vertical_delta': vertical_delta,
            'cos_turn': cos_turn,
            'abs_v1y': abs(v1y),
            'abs_v2y': abs(v2y),
            'y_sign_flip': bool(y_sign_flip),
            'source_quality': source_quality_score(src2),
            'rule_y_velocity_sign_flip': 1.0 if y_sign_flip else 0.0,
            'rule_sharp_direction_change': 1.0 if sharp_turn else 0.0,
        }

        if y_sign_flip:
            self._register_hit_candidate(f2, x2, y2, video_fps, 'y_velocity_sign_flip', features, keep_rejected=keep_rejected, candidate_source=candidate_source)
        elif sharp_turn:
            self._register_hit_candidate(f2, x2, y2, video_fps, 'sharp_direction_change', features, keep_rejected=keep_rejected, candidate_source=candidate_source)

    def _check_clean_contact_candidate(self, video_fps, keep_rejected=False):
        self._check_contact_candidate_from_history(self.clean_points_history, video_fps, keep_rejected=keep_rejected, candidate_source='clean')

    def _check_raw_contact_candidate(self, video_fps, keep_rejected=False):
        self._check_contact_candidate_from_history(self.detection_points_history, video_fps, keep_rejected=keep_rejected, candidate_source='raw')

    def _update_detection_history(self, frame_num, x, y, source, video_fps, keep_rejected=False):
        """Update raw detection history and run event rule checks."""
        self.detection_points_full.append((frame_num, x, y, source))
        self.detection_points_history.append((frame_num, x, y, source))
        if len(self.detection_points_history) > 20:
            self.detection_points_history = self.detection_points_history[-20:]

    def _update_clean_trail(self, frame_num, x, y, source, video_fps, keep_rejected=False):
        """Update cleaned single-line trail for visualization and primary contact checks."""
        if source not in ('yolo', 'optical_flow'):
            return False

        curr_point = (x, y)

        if self.trail_last_valid_point is not None:
            dx = curr_point[0] - self.trail_last_valid_point[0]
            dy = curr_point[1] - self.trail_last_valid_point[1]
            jump = math.hypot(dx, dy)

            # Ignore extreme spike outliers entirely
            if jump > self.clean_spike_jump_px:
                self.clean_track_broken = True
                return False

            # Break segment on large jump (avoid long spike lines)
            if jump > self.clean_break_jump_px:
                self.trail_points.append(None)
                self.smooth_last_point = None
                self.clean_track_broken = True

        # EMA smoothing for cleaner continuous line
        if self.smooth_last_point is None:
            smooth_x, smooth_y = x, y
        else:
            sx, sy = self.smooth_last_point
            smooth_x = int(self.clean_smooth_alpha * x + (1.0 - self.clean_smooth_alpha) * sx)
            smooth_y = int(self.clean_smooth_alpha * y + (1.0 - self.clean_smooth_alpha) * sy)

        smooth_point = (smooth_x, smooth_y)
        self.trail_points.append(smooth_point)
        self.trail_last_valid_point = smooth_point
        self.smooth_last_point = smooth_point
        self.clean_track_broken = False

        self.clean_points_full.append((frame_num, smooth_x, smooth_y))
        self.clean_points_history.append((frame_num, smooth_x, smooth_y, source))
        if len(self.clean_points_history) > 20:
            self.clean_points_history = self.clean_points_history[-20:]
        if len(self.trail_points) > self.trail_max_len:
            self.trail_points = self.trail_points[-self.trail_max_len:]

        return True

    def _merge_hit_candidate_records(self):
        """Merge clean/raw candidate records into one final list."""
        if not self.hit_candidate_records:
            self.hit_candidates = []
            self.hit_candidates_accepted = 0
            self.hit_candidates_review = 0
            self.hit_candidates_rejected = 0
            return

        merge_tolerance_frames = max(2, self.contact_type_frame_tolerance)
        ordered_records = sorted(
            self.hit_candidate_records,
            key=lambda event: (
                int(event['frame']),
                0 if event.get('candidate_source') == 'clean' else 1,
                -float(event.get('score', 0.0)),
            ),
        )

        merged = []
        cluster = []

        def choose_cluster_record(cluster_records: list[dict]) -> dict:
            clean_records = [event for event in cluster_records if event.get('candidate_source') == 'clean']
            preferred = clean_records if clean_records else cluster_records
            chosen = max(
                preferred,
                key=lambda event: (
                    float(event.get('score', 0.0)),
                    -int(event['frame']),
                ),
            )

            # Only keep one representative per event. Prefer the clean candidate when present.
            for key in ('candidate_source',):
                chosen.pop(key, None)
            return chosen

        for event in ordered_records:
            if not cluster:
                cluster = [event]
                continue

            if int(event['frame']) - int(cluster[-1]['frame']) <= merge_tolerance_frames:
                cluster.append(event)
            else:
                merged.append(choose_cluster_record(cluster))
                cluster = [event]

        if cluster:
            merged.append(choose_cluster_record(cluster))

        self.hit_candidates = sorted(merged, key=lambda event: int(event['frame']))
        self.hit_candidates_accepted = sum(1 for event in self.hit_candidates if event['decision'] == 'accept')
        self.hit_candidates_review = sum(1 for event in self.hit_candidates if event['decision'] == 'review')
        self.hit_candidates_rejected = sum(1 for event in self.hit_candidates if event['decision'] == 'reject')
    
    def print_progress(self, frame_num: int, total_frames: int):
        """Print progress bar"""
        
        if frame_num % 100 == 0 or frame_num == total_frames:
            progress = (frame_num / total_frames) * 100
            bar_length = 40
            filled = int(bar_length * frame_num / total_frames)
            bar = '█' * filled + '░' * (bar_length - filled)
            
            elapsed = time.time() - self.start_time
            fps = frame_num / elapsed if elapsed > 0 else 0
            
            print(f"\rProgress: [{bar}] {progress:5.1f}% | Frame {frame_num}/{total_frames} | {fps:5.1f} FPS", end='')
    
    def print_statistics(self):
        """Print final processing statistics"""
        
        elapsed = time.time() - self.start_time
        fps = self.total_frames / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*60)
        print("EDGE INFERENCE STATISTICS")
        print("="*60)
        print(f"Video: {self.video_path.name}")
        print(f"Total Frames: {self.total_frames}")
        print(f"Processing Time: {elapsed:.2f}s")
        print(f"Average FPS: {fps:.2f}")
        print("-"*60)
        print(f"Total Detections: {self.detections} ({self.detections/self.total_frames*100:.1f}%)")
        print(f"  YOLO Detections: {self.yolo_detections} ({self.yolo_detections/self.total_frames*100:.1f}%)")
        print(f"  Optical Flow: {self.optical_flow_detections} ({self.optical_flow_detections/self.total_frames*100:.1f}%)")
        print(f"  Kalman Predictions: {self.kalman_predictions} ({self.kalman_predictions/self.total_frames*100:.1f}%)")
        print(f"  No Detection: {self.total_frames - self.detections} ({(self.total_frames - self.detections)/self.total_frames*100:.1f}%)")
        print(f"Trail Snapshots: {self.saved_snapshots}")
        print(f"Hit Candidates: {len(self.hit_candidates)}")
        if self.contact_type_filter:
            print(f"  Skipped Non-Ground Candidates: {self.skipped_non_ground_candidates}")
        if config.CONTACT_SCORING_ENABLED:
            print(f"  Accepted: {self.hit_candidates_accepted}")
            print(f"  Review: {self.hit_candidates_review}")
            print(f"  Rejected: {self.hit_candidates_rejected}")
        print("-"*60)
        print(f"Output CSV: {self.csv_path}")
        print(f"Detection CSV: {self.detections_csv_path}")
        print(f"Clean CSV: {self.clean_csv_path}")
        print(f"Hit CSV: {self.hit_candidates_csv_path}")
        print(f"Trail Images: {self.snapshots_dir}")
        if self.annotated_video_path.exists():
            print(f"Output Video: {self.annotated_video_path}")
        print("="*60 + "\n")
    
    def run(self, show_preview: bool = False, save_video: bool = False, snapshot_interval_sec: float = 10.0, keep_rejected_candidates: bool = False):
        """
        Run inference pipeline
        
        Args:
            show_preview: If True, display live preview window (requires display)
            save_video: If True, save annotated video to outputs/edge
            snapshot_interval_sec: Save one trail image every N seconds
            keep_rejected_candidates: If True, keep scored reject events in hit CSV (for threshold tuning)
        
        Returns:
            trajectory: List of (frame_num, x, y, source) tuples
        """
        
        # Open video
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")
        
        # Get video properties
        video_fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Dynamic anti-spike threshold based on frame size
        self.trail_max_jump_px = int(max(width, height) * 0.12)
        self.clean_break_jump_px = int(max(width, height) * 0.10)
        self.clean_spike_jump_px = int(max(width, height) * 0.20)

        if video_fps <= 0:
            video_fps = 30

        self.hit_cooldown_frames = max(8, int(video_fps * 0.33))
        snapshot_interval_frames = max(1, int(round(video_fps * snapshot_interval_sec)))

        # Optional video writer for annotated output
        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(self.annotated_video_path), fourcc, video_fps, (width, height))
        
        if config.VERBOSE:
            print(f"Video Properties:")
            print(f"  Resolution: {width}x{height}")
            print(f"  FPS: {video_fps}")
            print(f"  Total Frames: {total_frames}")
            print(f"  Snapshot interval: {snapshot_interval_sec:.1f}s ({snapshot_interval_frames} frames)")
            if save_video:
                print(f"  Save Video: {self.annotated_video_path}")
            print(f"\nProcessing...")
        
        # Trajectory storage
        trajectory = []
        
        # Start processing
        self.start_time = time.time()
        frame_num = 0
        last_snapshot_frame = 0
        last_frame_for_snapshot = None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_num += 1
            self.total_frames += 1
            last_frame_for_snapshot = frame.copy()
            
            # Skip frames if configured
            if config.FRAME_SKIP > 0 and frame_num % (config.FRAME_SKIP + 1) != 0:
                continue
            
            # Detect ball
            ball_pos, source = self.detector.detect(frame)
            should_annotate = show_preview or (writer is not None)
            
            # Record detection
            if ball_pos:
                x, y = ball_pos
                trajectory.append((frame_num, x, y, source))
                self.detections += 1
                self.frames_without_ball = 0

                # Raw detections are retained for fallback checks and debugging.
                self._update_detection_history(
                    frame_num,
                    x,
                    y,
                    source,
                    video_fps,
                    keep_rejected=keep_rejected_candidates,
                )

                # Update cleaned trail first; this is the primary contact source.
                clean_updated = self._update_clean_trail(
                    frame_num,
                    x,
                    y,
                    source,
                    video_fps,
                    keep_rejected=keep_rejected_candidates,
                )

                if clean_updated:
                    self._check_clean_contact_candidate(video_fps, keep_rejected=keep_rejected_candidates)
                self._check_raw_contact_candidate(video_fps, keep_rejected=keep_rejected_candidates)
                
                if source == 'yolo':
                    self.yolo_detections += 1
                elif source == 'optical_flow':
                    self.optical_flow_detections += 1
                elif source == 'kalman':
                    self.kalman_predictions += 1
                
                # Draw on frame for preview/output video
                if should_annotate:
                    if source == 'yolo':
                        color = (0, 255, 0)
                    elif source == 'optical_flow':
                        color = (255, 255, 0)
                    else:
                        color = (0, 255, 255)
                    cv2.circle(frame, (x, y), 10, color, 2)
                    cv2.putText(frame, source.upper(), (x + 15, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            elif config.SAVE_FULL_TRAJECTORY:
                # Save frame even without detection
                trajectory.append((frame_num, -1, -1, 'none'))
                self.frames_without_ball += 1

                # Break trail quickly after short detection gaps
                if self.frames_without_ball >= self.trail_gap_after_misses:
                    self.trail_points.append(None)
                    self.smooth_last_point = None
                    self.clean_track_broken = True

                # Hard reset after longer gap
                if self.frames_without_ball > self.trail_reset_after:
                    self.trail_points = []
                    self.trail_last_valid_point = None
                    self.smooth_last_point = None
                    self.clean_track_broken = True

            # Draw trajectory trail
            if should_annotate and len(self.trail_points) >= 2:
                self._draw_trail(frame, self.trail_points, color=(255, 0, 255), thickness=2)

            # Draw collision markers (one colored point per collision candidate)
            if should_annotate and self.collision_markers:
                self._draw_collision_markers(frame, frame_num)
                self._draw_collision_legend(frame)

            # Save trail snapshot every N seconds based on FPS/frame count
            if frame_num % snapshot_interval_frames == 0:
                self._save_trail_snapshot(frame, frame_num, video_fps)
                last_snapshot_frame = frame_num
            
            # Draw HUD for preview/output video
            if should_annotate:
                elapsed = time.time() - self.start_time
                processing_fps = frame_num / elapsed if elapsed > 0 else 0

                # Add info overlay
                cv2.putText(frame, f"Frame: {frame_num}/{total_frames}", (10, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Detections: {self.detections}", (10, 65),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"FPS: {processing_fps:.1f}", (10, 100),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Write annotated frame
            if writer is not None:
                writer.write(frame)

            # Show preview
            if show_preview:
                
                cv2.imshow('Edge Inference - Press Q to quit', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n\nStopped by user")
                    break
            
            # Progress indicator
            if not show_preview:
                self.print_progress(frame_num, total_frames)
        
        # Cleanup
        cap.release()
        if writer is not None:
            writer.release()
        if show_preview:
            cv2.destroyAllWindows()
        
        if not show_preview:
            print()  # New line after progress bar

        # Save final trailing snapshot if last segment didn't hit an exact interval boundary
        if frame_num > 0 and last_snapshot_frame != frame_num and last_frame_for_snapshot is not None:
            self._save_trail_snapshot(last_frame_for_snapshot, frame_num, video_fps)

        # Merge clean and raw candidate records into a single final list before saving.
        self._merge_hit_candidate_records()
        
        # Save trajectory
        if config.CSV_SAVE_ENABLED and trajectory:
            self.save_trajectory_csv(trajectory)

        # Save raw detections used for contact checks.
        self.save_detection_csv()

        # Save cleaned trajectory and hit candidates
        self.save_clean_trajectory_csv()
        self.save_hit_candidates_csv()
        
        # Print statistics
        if config.VERBOSE or config.BENCHMARK_MODE:
            self.print_statistics()
        
        return trajectory


def main():
    """Main entry point with argument parsing"""
    
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Edge Ball Tracking Inference - Raspberry Pi 5 + Hailo-8',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python edge_inference.py video.mp4
  python edge_inference.py video.mp4 --preview
  python edge_inference.py video.mp4 --output-name match_01 --verbose
    python edge_inference.py video.mp4 --snapshot-interval-sec 10
        """
    )
    
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--output-name', type=str, default=None,
                       help='Output name (default: video filename)')
    parser.add_argument('--preview', action='store_true',
                       help='Show live preview window')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    parser.add_argument('--benchmark', action='store_true',
                       help='Enable benchmark mode (measure FPS)')
    parser.add_argument('--save-video', action='store_true',
                       help='Save annotated output video')
    parser.add_argument('--contact-labels-csv', type=str, default=None,
                       help='Optional labels CSV with Frame and ContactType columns for colored collision markers')
    parser.add_argument('--snapshot-interval-sec', type=float, default=10.0,
                       help='Save one trail snapshot image every N seconds (default: 10)')
    parser.add_argument('--keep-rejected-candidates', action='store_true',
                       help='Keep scored reject candidates in hit CSV (useful for threshold tuning)')
    
    args = parser.parse_args()
    
    # Update config
    if args.verbose:
        config.VERBOSE = True
    if args.benchmark:
        config.BENCHMARK_MODE = True
    
    # Run pipeline
    try:
        pipeline = EdgeInferencePipeline(args.video, args.output_name, contact_labels_csv=args.contact_labels_csv)
        trajectory = pipeline.run(
            show_preview=args.preview,
            save_video=args.save_video,
            snapshot_interval_sec=args.snapshot_interval_sec,
            keep_rejected_candidates=args.keep_rejected_candidates,
        )
        
        if not config.VERBOSE:
            print(f"\n✓ Processing complete!")
            print(f"  Detections: {pipeline.detections}/{pipeline.total_frames} ({pipeline.detections/pipeline.total_frames*100:.1f}%)")
            print(f"  Output: {pipeline.csv_path}\n")
            if args.save_video:
                print(f"  Output Video: {pipeline.annotated_video_path}\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        raise


if __name__ == "__main__":
    main()
