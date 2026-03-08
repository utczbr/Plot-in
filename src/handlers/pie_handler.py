"""
Pie chart handler implementing polar coordinate processing.

This handler processes pie charts by detecting slices using keypoint detection
and calculating angles and values for each slice.

§4.5: RANSAC circle fit from Pie_pose.onnx boundary keypoints (Kåsa's method).
§4.5.3: Keypoint-based angular spans with max-gap method (Pac-Man fix).
§4.6: Sum-to-one normalization with data label override.
"""
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import re
import logging
from handlers.base_handler import PolarChartHandler, ExtractionResult, ChartCoordinateSystem
from services.orientation_service import Orientation


class PieHandler(PolarChartHandler):
    """Pie chart handler with polar coordinate processing."""

    COORDINATE_SYSTEM = ChartCoordinateSystem.POLAR

    def __init__(self, classifier=None, legend_matcher=None, **kwargs):
        super().__init__(legend_matcher=legend_matcher, **kwargs)
        self.classifier = classifier

    def get_chart_type(self) -> str:
        return "pie"

    def process(
        self,
        image: np.ndarray,
        detections: Dict[str, Any],
        axis_labels: List[Dict],
        chart_elements: List[Dict],
        orientation: Orientation,
        **kwargs
    ) -> ExtractionResult:
        """Process pie chart and extract slice values using geometric analysis."""
        try:
            pie_slices = detections.get('pie_slice', []) or detections.get('slice', []) or chart_elements

            if not pie_slices:
                self.logger.warning("No pie slices detected")
                return ExtractionResult(
                    chart_type=self.get_chart_type(),
                    coordinate_system=self.get_coordinate_system(),
                    elements=[],
                    orientation=orientation
                )

            h, w = image.shape[:2]
            diagnostics = {'slice_count': len(pie_slices)}

            # 1. Classify Labels
            classified_labels = {'legend_labels': [], 'data_labels': []}
            if self.classifier:
                clf_result = self.classifier.classify(axis_labels, pie_slices, w, h)
                classified_labels = clf_result.metadata
                self.logger.info(
                    f"Pie classification: {len(classified_labels['legend_labels'])} legends, "
                    f"{len(classified_labels['data_labels'])} data labels"
                )

            # 2. Center Detection — §4.5: Try RANSAC circle fit from keypoints first
            boundary_kps = self._extract_boundary_keypoints(pie_slices)
            radius = None

            if len(boundary_kps) >= 6:
                # Enough boundary keypoints for RANSAC circle fit
                fit_result = self._fit_circle_ransac(boundary_kps)
                if fit_result is not None:
                    center_point, radius, inlier_count = fit_result
                    diagnostics['center_method'] = 'ransac_circle'
                    diagnostics['ransac_inliers'] = inlier_count
                    diagnostics['pie_radius'] = float(radius)
                    self.logger.info(
                        f"RANSAC circle fit: center=({center_point[0]:.1f}, {center_point[1]:.1f}), "
                        f"r={radius:.1f}, inliers={inlier_count}/{len(boundary_kps)}"
                    )
                else:
                    center_point = self._find_pie_center_robust(pie_slices, w, h)
                    diagnostics['center_method'] = 'mad_centroid_fallback'
            else:
                center_point = self._find_pie_center_robust(pie_slices, w, h)
                diagnostics['center_method'] = 'mad_centroid'

            self.logger.info(f"Detected Pie Center: {center_point}")

            # 3. Geometric Analysis — §4.5.3: Keypoint-based angles with max-gap method
            slice_data = []
            use_keypoint_spans = True  # Track whether all slices have keypoint spans

            for slice_det in pie_slices:
                bbox = slice_det['xyxy']
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                # Centroid angle (always computed for sorting/fallback)
                dx = cx - center_point[0]
                dy = cy - center_point[1]
                angle_rad = np.arctan2(dy, dx)
                angle_deg = float(np.degrees(angle_rad))
                if angle_deg < 0:
                    angle_deg += 360.0

                # §4.5.3: Try keypoint-based angular span (max-gap method)
                kp_span = self._compute_slice_span_maxgap(slice_det, center_point)

                entry = {
                    'bbox': bbox,
                    'center': (cx, cy),
                    'mid_angle': angle_deg,
                    'kp_span': kp_span,  # radians, or None if no keypoints
                    'det': slice_det
                }
                slice_data.append(entry)

                if kp_span is None:
                    use_keypoint_spans = False

            # Sort slices by angle
            slice_data.sort(key=lambda x: x['mid_angle'])
            n = len(slice_data)

            # 4. Calculate angular spans
            if use_keypoint_spans:
                # §4.5.3: Use keypoint-derived spans
                spans_rad = [s['kp_span'] for s in slice_data]
                diagnostics['span_method'] = 'keypoint_maxgap'
            else:
                # Fallback: centroid-based neighbor-distance spans
                spans_rad = []
                for i in range(n):
                    curr_angle = slice_data[i]['mid_angle']
                    prev_angle = slice_data[(i - 1 + n) % n]['mid_angle']
                    next_angle = slice_data[(i + 1) % n]['mid_angle']

                    dist_prev = curr_angle - prev_angle
                    if dist_prev < 0:
                        dist_prev += 360.0
                    dist_next = next_angle - curr_angle
                    if dist_next <= 0:
                        dist_next += 360.0

                    estimated_span_deg = (dist_prev + dist_next) / 2.0
                    spans_rad.append(np.radians(estimated_span_deg))
                diagnostics['span_method'] = 'centroid_neighbor'

            # §4.6.1: Geometric normalization — g_i = Δθ_i / T, ensures Σg_i = 1
            total_span = sum(spans_rad)
            if total_span > 0:
                geometric_values = [s / total_span for s in spans_rad]
            else:
                geometric_values = [1.0 / n for _ in range(n)]

            # §4.6.2: Data label integration — normalize with labels
            data_labels = classified_labels.get('data_labels', [])
            final_values = self._normalize_sum_to_one(
                slice_data, geometric_values, data_labels, diagnostics
            )

            # 5. Build elements
            elements = []
            for i in range(n):
                curr = slice_data[i]

                # Match Legend
                label = "Unknown"
                if self.legend_matcher:
                    label = self.legend_matcher.match_slice_to_legend(
                        curr['det'],
                        classified_labels.get('legend_labels', [])
                    )

                elements.append({
                    'type': 'pie_slice',
                    'bbox': curr['bbox'],
                    'value': float(final_values[i]),
                    'label': label,
                    'angle': curr['mid_angle'],
                    'span_rad': float(spans_rad[i]) if i < len(spans_rad) else 0.0,
                    'confidence': curr['det'].get('conf', 1.0),
                    'value_source': curr.get('value_source', 'geometric')
                })

            return ExtractionResult(
                chart_type=self.get_chart_type(),
                coordinate_system=self.get_coordinate_system(),
                elements=elements,
                diagnostics=diagnostics,
                orientation=orientation
            )

        except Exception as e:
            self.logger.error(f"Error in PieHandler.process: {e}")
            return ExtractionResult.from_error(self.get_chart_type(), e)

    # ── §4.5: Keypoint Extraction and RANSAC Circle Fit ──────────────────

    def _extract_boundary_keypoints(self, slices: List[Dict]) -> np.ndarray:
        """
        Collect boundary keypoints from all slices.

        Pie_pose.onnx provides 5 keypoints per slice. Keypoint 0 is assumed
        center-like; keypoints 1-4 are boundary points on the arc/radii.
        Returns array of shape (N, 2) with all boundary keypoints.
        """
        points = []
        for s in slices:
            kps = s.get('keypoints')
            if kps is None:
                continue
            kps = np.asarray(kps)
            if kps.ndim == 1:
                # Flat array — reshape assuming (x, y, conf) triplets
                if kps.size >= 15:  # 5 keypoints × 3
                    kps = kps.reshape(-1, 3)
                elif kps.size >= 10:  # 5 keypoints × 2
                    kps = kps.reshape(-1, 2)
                else:
                    continue
            # Take boundary keypoints (indices 1-4)
            for j in range(1, min(5, len(kps))):
                x, y = float(kps[j, 0]), float(kps[j, 1])
                # Skip invalid keypoints (zero or negative)
                if x > 0 and y > 0:
                    points.append([x, y])
        return np.array(points) if points else np.empty((0, 2))

    def _fit_circle_ransac(
        self, points: np.ndarray, max_iter: int = 100, inlier_thresh: float = 2.0
    ) -> Optional[Tuple[Tuple[float, float], float, int]]:
        """
        §4.5.2: RANSAC circle fit using Kåsa's algebraic method.

        Returns (center, radius, n_inliers) or None if fit fails.
        Kåsa's method solves x²+y² + Dx + Ey + F = 0 via least-squares on 3 points.
        """
        n = len(points)
        if n < 3:
            return None

        best_inlier_count = 0
        best_center = None
        best_radius = None
        best_inlier_mask = None

        rng = np.random.RandomState(42)  # Reproducible

        for _ in range(max_iter):
            # Sample 3 random distinct points
            idx = rng.choice(n, size=3, replace=False)
            p = points[idx]  # (3, 2)

            # Kåsa's method: solve [x, y, 1] · [D, E, F]^T = -(x² + y²)
            A = np.column_stack([p[:, 0], p[:, 1], np.ones(3)])
            b = -(p[:, 0] ** 2 + p[:, 1] ** 2)

            try:
                DEF = np.linalg.solve(A, b)
            except np.linalg.LinAlgError:
                continue  # Collinear points

            D, E, F = DEF
            cx = -D / 2.0
            cy = -E / 2.0
            r_sq = cx ** 2 + cy ** 2 - F
            if r_sq <= 0:
                continue
            r = np.sqrt(r_sq)

            # Compute residuals for all points
            dists = np.sqrt((points[:, 0] - cx) ** 2 + (points[:, 1] - cy) ** 2)
            residuals = np.abs(dists - r)
            inlier_mask = residuals < inlier_thresh
            n_inliers = int(np.sum(inlier_mask))

            if n_inliers > best_inlier_count:
                best_inlier_count = n_inliers
                best_center = (cx, cy)
                best_radius = r
                best_inlier_mask = inlier_mask

        if best_center is None or best_inlier_count < 3:
            return None

        # Refine on inliers via least-squares (Kåsa on all inliers)
        inlier_pts = points[best_inlier_mask]
        if len(inlier_pts) >= 3:
            A = np.column_stack([inlier_pts[:, 0], inlier_pts[:, 1], np.ones(len(inlier_pts))])
            b = -(inlier_pts[:, 0] ** 2 + inlier_pts[:, 1] ** 2)
            DEF, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            D, E, F = DEF
            cx = -D / 2.0
            cy = -E / 2.0
            r_sq = cx ** 2 + cy ** 2 - F
            if r_sq > 0:
                best_center = (cx, cy)
                best_radius = np.sqrt(r_sq)

        return (best_center, float(best_radius), best_inlier_count)

    # ── §4.5.3: Max-Gap Method for Slice Angular Span ────────────────────

    def _compute_slice_span_maxgap(
        self, slice_det: Dict, center: Tuple[float, float]
    ) -> Optional[float]:
        """
        Compute angular span of a slice using the max-gap method on boundary keypoints.

        Staff Refinement — Pac-Man fix: The max-gap method correctly handles slices
        of any size. Sort boundary angles circularly, find the largest angular gap
        between adjacent angles (including wrap-around), then span = 2π - max_gap.
        The largest gap always corresponds to the arc *outside* the slice.

        Returns span in radians, or None if keypoints unavailable.
        """
        kps = slice_det.get('keypoints')
        if kps is None:
            return None

        kps = np.asarray(kps)
        if kps.ndim == 1:
            if kps.size >= 15:
                kps = kps.reshape(-1, 3)
            elif kps.size >= 10:
                kps = kps.reshape(-1, 2)
            else:
                return None

        # Collect boundary keypoint angles (indices 1-4)
        angles = []
        for j in range(1, min(5, len(kps))):
            x, y = float(kps[j, 0]), float(kps[j, 1])
            if x <= 0 and y <= 0:
                continue
            theta = np.arctan2(y - center[1], x - center[0])
            if theta < 0:
                theta += 2.0 * np.pi
            angles.append(theta)

        if len(angles) < 2:
            return None

        # Sort angles
        angles.sort()

        # Compute gaps between consecutive angles (including wrap-around)
        gaps = []
        for i in range(len(angles) - 1):
            gaps.append(angles[i + 1] - angles[i])
        # Wrap-around gap from last to first + 2π
        wrap_gap = (angles[0] + 2.0 * np.pi) - angles[-1]
        gaps.append(wrap_gap)

        # Max-gap method: span = 2π - max_gap
        max_gap = max(gaps)
        span = 2.0 * np.pi - max_gap

        # Clamp to valid range
        span = max(0.0, min(2.0 * np.pi, span))

        return float(span)

    # ── §4.6: Sum-to-One Normalization with Data Label Override ───────────

    def _normalize_sum_to_one(
        self,
        slice_data: List[Dict],
        geometric_values: List[float],
        data_labels: List[Dict],
        diagnostics: Dict
    ) -> List[float]:
        """
        §4.6: Normalize slice values to sum to 1.0, integrating data labels.

        1. Match data labels to slices spatially.
        2. Parse percentage/numeric labels.
        3. Partition into labeled (L) and unlabeled (U) sets.
        4. Case A: Labels consistent (sum ≤ 1) → use labels for L, share remainder proportionally for U.
        5. Case B: Labels overshoot (sum > 1) → pure geometry fallback or normalize labels.
        """
        n = len(slice_data)
        if n == 0:
            return []

        # Match data labels to slices
        label_map = self._match_data_labels_to_slices(slice_data, data_labels)

        # Parse and validate labels
        parsed = {}  # slice_index → parsed fraction
        discarded_count = 0

        for i, raw_label in label_map.items():
            parsed_val = self._parse_data_label(raw_label.get('text', ''))
            if parsed_val is not None:
                # §4.6.2 Staff Refinement: Sanity pre-filter
                if 0.0 < parsed_val <= 1.0:
                    parsed[i] = parsed_val
                else:
                    discarded_count += 1
                    self.logger.warning(
                        f"Discarded data label for slice {i}: value={parsed_val:.3f} out of (0, 1]"
                    )

        if discarded_count > 0:
            diagnostics['pie_labels_discarded'] = discarded_count

        # If no valid labels, return pure geometric values (already sum to 1)
        if not parsed:
            diagnostics['pie_normalization'] = 'geometric'
            return geometric_values

        # Partition into labeled (L) and unlabeled (U)
        L_sum = sum(parsed.values())
        U_indices = [i for i in range(n) if i not in parsed]
        G_U = sum(geometric_values[i] for i in U_indices)

        # Case A: Labels self-consistent
        if 0.0 < L_sum <= 1.0:
            U_share = 1.0 - L_sum
            final = [0.0] * n
            for i in range(n):
                if i in parsed:
                    final[i] = parsed[i]
                    slice_data[i]['value_source'] = 'data_label'
                else:
                    if G_U > 0:
                        final[i] = U_share * (geometric_values[i] / G_U)
                    else:
                        # All unlabeled slices have zero geometry — distribute equally
                        n_unlabeled = len(U_indices)
                        final[i] = U_share / n_unlabeled if n_unlabeled > 0 else 0.0
                    slice_data[i]['value_source'] = 'geometric_residual'
            diagnostics['pie_normalization'] = 'label_consistent'
            diagnostics['pie_labeled_sum'] = float(L_sum)
            return final

        # Case B: Labels overshoot or inconsistent
        diagnostics['pie_label_inconsistency'] = True
        self.logger.warning(f"Pie label sum={L_sum:.3f} > 1.0 — inconsistent labels")

        if not U_indices:
            # All slices labeled — normalize labels to sum to 1
            final = [0.0] * n
            for i in range(n):
                final[i] = parsed.get(i, 0.0) / L_sum
                slice_data[i]['value_source'] = 'data_label_normalized'
            diagnostics['pie_normalization'] = 'label_normalized'
            return final

        # Fallback: pure geometry
        diagnostics['pie_normalization'] = 'geometric_fallback'
        return geometric_values

    def _match_data_labels_to_slices(
        self, slice_data: List[Dict], data_labels: List[Dict]
    ) -> Dict[int, Dict]:
        """
        Match data labels to slices using nearest-neighbor spatial matching.
        Returns dict of {slice_index: label_dict}.
        """
        if not data_labels:
            return {}

        matches = {}
        used_labels = set()

        for i, s in enumerate(slice_data):
            sx = (s['bbox'][0] + s['bbox'][2]) / 2
            sy = (s['bbox'][1] + s['bbox'][3]) / 2

            best_dist = float('inf')
            best_j = None

            for j, label in enumerate(data_labels):
                if j in used_labels:
                    continue
                l_bbox = label.get('xyxy', label.get('bbox', [0, 0, 0, 0]))
                lx = (l_bbox[0] + l_bbox[2]) / 2
                ly = (l_bbox[1] + l_bbox[3]) / 2
                dist = (sx - lx) ** 2 + (sy - ly) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j is not None:
                used_labels.add(best_j)
                matches[i] = data_labels[best_j]

        return matches

    @staticmethod
    def _parse_data_label(text: str) -> Optional[float]:
        """
        Parse a data label string into a fraction (0, 1].

        Handles: "25%", "25.0%", "0.25", "25", unicode minus signs.
        Returns None if unparseable.
        """
        if not text or not text.strip():
            return None

        text = text.strip()
        # Normalize unicode minus
        text = text.replace('\u2212', '-').replace('\u2013', '-').replace('\u2014', '-')

        # Try percentage pattern: "25%", "25.0%", "25 %"
        pct_match = re.match(r'^([+-]?\d+\.?\d*)\s*%$', text)
        if pct_match:
            try:
                return float(pct_match.group(1)) / 100.0
            except ValueError:
                return None

        # Try plain number — if ≤ 1.0, treat as fraction; if > 1.0, treat as percentage
        try:
            val = float(text)
            if val <= 1.0:
                return val
            elif val <= 100.0:
                return val / 100.0
            else:
                return None  # Implausibly large
        except ValueError:
            return None

    # ── Legacy Fallback Methods ──────────────────────────────────────────

    def _find_pie_center_robust(self, slices: List[Dict], w: int, h: int) -> Tuple[float, float]:
        """Find center robustly handling exploded slices using MAD outlier removal."""
        points = []
        for s in slices:
            bbox = s['xyxy']
            points.append([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])

        points = np.array(points)
        if len(points) < 3:
            if len(points) == 0:
                return (w / 2, h / 2)
            return tuple(np.mean(points, axis=0))

        mask = np.ones(len(points), dtype=bool)
        for _ in range(3):
            current_center = np.mean(points[mask], axis=0)
            dists = np.linalg.norm(points - current_center, axis=1)

            non_outlier_dists = dists[mask]
            if len(non_outlier_dists) == 0:
                break

            median_dist = np.median(non_outlier_dists)
            mad = np.median(np.abs(non_outlier_dists - median_dist))

            if mad < 1e-6:
                break

            new_mask = dists < (median_dist + 3 * mad)
            if np.sum(new_mask) < 2:
                break

            if np.array_equal(mask, new_mask):
                break
            mask = new_mask

        return tuple(np.mean(points[mask], axis=0))

    def _match_slice_to_legend(self, slice_det: Dict[str, Any], axis_labels: List[Dict]) -> str:
         if self.legend_matcher:
             return self.legend_matcher.match_slice_to_legend(slice_det, axis_labels)
         return "Slice"

    def extract_values(self, img, detections, calibration, baselines, orientation) -> List[Dict]:
         return []
