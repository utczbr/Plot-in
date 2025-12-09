"""
Robust bar-to-label association layer with multiple fallback strategies.

Handles:
- Bars narrower than tick labels (thin bars)
- Bars wider than tick labels (wide bars)
- Stacked bars (multiple bars on same tick)
- Grouped bars (multiple bars per category)
- Mixed-width bars in same chart
- Irregular spacing
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging
from dataclasses import dataclass
from enum import Enum

class ChartLayout(Enum):
    """Detected chart layout type."""
    SIMPLE = "simple"  # One bar per tick label
    GROUPED = "grouped"  # Multiple bars per category (side-by-side)
    STACKED = "stacked"  # Multiple bars stacked on same tick
    MIXED = "mixed"  # Irregular layout

@dataclass
class AssociationStrategy:
    """Strategy used for association."""
    name: str
    threshold: float
    confidence: float
    reason: str

class RobustBarAssociator:
    """
    Multi-strategy bar-to-label associator with automatic layout detection.
    
    Association Strategies (in order of priority):
    1. Direct Overlap: Label center inside bar bounds → 100% confidence
    2. Proximity: Label within bar_width × factor of bar center
    3. Spacing-based: Label within spacing × factor (for regular layouts)
    4. Zone-based: Assign to nearest bar in spatial zone (fallback)
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
        # Strategy parameters (tunable via config)
        self.overlap_threshold = 0.1  # 10% overlap considered "aligned"
        self.proximity_factor = 1.5   # Within 1.5× bar width
        self.spacing_factor = 0.4     # Within 40% of spacing
        self.zone_factor = 2.0        # Fallback: within 2× spacing
    
    def detect_layout(self, bars: List[Dict], orientation: str) -> ChartLayout:
        """
        Detect chart layout type from bar positions.
        
        Returns:
            ChartLayout enum indicating layout complexity
        """
        if len(bars) < 2:
            return ChartLayout.SIMPLE
        
        # Extract bar positions along relevant axis
        if orientation == 'vertical':
            positions = sorted([(b['xyxy'][0] + b['xyxy'][2])/2.0 for b in bars])
            widths = [b['xyxy'][2] - b['xyxy'][0] for b in bars]
        else:
            positions = sorted([(b['xyxy'][1] + b['xyxy'][3])/2.0 for b in bars])
            widths = [b['xyxy'][3] - b['xyxy'][1] for b in bars]
        
        spacings = np.diff(positions)
        median_spacing = np.median(spacings)
        median_width = np.median(widths)
        
        # Check for grouped bars (bimodal spacing distribution)
        if len(spacings) >= 3:
            min_spacing = np.min(spacings)
            max_spacing = np.max(spacings)
            
            # Grouped: large gap between groups, small gap within groups
            if max_spacing > 2.5 * min_spacing:
                self.logger.info(
                    f"Detected GROUPED layout: "
                    f"max_spacing={max_spacing:.1f}px, min_spacing={min_spacing:.1f}px"
                )
                return ChartLayout.GROUPED
        
        # Check for stacked bars (overlapping positions)
        position_tolerance = median_width * 0.3
        for i in range(len(positions) - 1):
            if abs(positions[i+1] - positions[i]) < position_tolerance:
                self.logger.info(
                    f"Detected STACKED layout: "
                    f"bars overlap at position {positions[i]:.1f}px"
                )
                return ChartLayout.STACKED
        
        # Check for mixed/irregular layout
        if len(spacings) > 1:
            spacing_cv = np.std(spacings) / (median_spacing + 1e-6)
            width_cv = np.std(widths) / (median_width + 1e-6)
            
            if spacing_cv > 0.5 or width_cv > 0.5:
                self.logger.info(
                    f"Detected MIXED layout: "
                    f"spacing_cv={spacing_cv:.2f}, width_cv={width_cv:.2f}"
                )
                return ChartLayout.MIXED
        
        # Default: simple layout
        self.logger.info("Detected SIMPLE layout")
        return ChartLayout.SIMPLE
    
    def _compute_overlap_1d(self, bar_min: float, bar_max: float, 
                           label_min: float, label_max: float) -> float:
        """
        Compute 1D overlap ratio between bar and label.
        
        Returns:
            Overlap ratio in [0, 1], where 1 = label fully inside bar
        """
        overlap_start = max(bar_min, label_min)
        overlap_end = min(bar_max, label_max)
        overlap = max(0, overlap_end - overlap_start)
        
        label_span = label_max - label_min
        if label_span < 1e-6:
            return 0.0
        
        return overlap / label_span
    
    def _strategy_direct_overlap(
        self, 
        bar: Dict, 
        label: Dict, 
        orientation: str
    ) -> Optional[AssociationStrategy]:
        """
        Strategy 1: Direct overlap detection.
        
        Check if label overlaps with bar bounds (handles bars wider than labels).
        """
        if orientation == 'vertical':
            # Check X-axis overlap
            bar_left, bar_right = bar['xyxy'][0], bar['xyxy'][2]
            label_left, label_right = label['xyxy'][0], label['xyxy'][2]
            overlap = self._compute_overlap_1d(bar_left, bar_right, label_left, label_right)
        else:
            # Check Y-axis overlap
            bar_top, bar_bottom = bar['xyxy'][1], bar['xyxy'][3]
            label_top, label_bottom = label['xyxy'][1], label['xyxy'][3]
            overlap = self._compute_overlap_1d(bar_top, bar_bottom, label_top, label_bottom)
        
        if overlap >= self.overlap_threshold:
            return AssociationStrategy(
                name="direct_overlap",
                threshold=overlap,
                confidence=min(1.0, overlap),
                reason=f"{overlap*100:.0f}% label overlap with bar"
            )
        return None
    
    def _strategy_proximity(
        self, 
        bar: Dict, 
        label: Dict, 
        orientation: str
    ) -> Optional[AssociationStrategy]:
        """
        Strategy 2: Proximity-based association.
        
        Check if label center is within bar_width × proximity_factor of bar center.
        (Handles bars narrower than labels)
        """
        bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
        bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
        label_cx = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
        label_cy = (label['xyxy'][1] + label['xyxy'][3]) / 2.0
        
        if orientation == 'vertical':
            bar_width = bar['xyxy'][2] - bar['xyxy'][0]
            distance = abs(label_cx - bar_cx)
            threshold = bar_width * self.proximity_factor
        else:
            bar_height = bar['xyxy'][3] - bar['xyxy'][1]
            distance = abs(label_cy - bar_cy)
            threshold = bar_height * self.proximity_factor
        
        if distance <= threshold:
            # Confidence inversely proportional to distance
            confidence = 1.0 - (distance / threshold) * 0.5  # Range [0.5, 1.0]
            return AssociationStrategy(
                name="proximity",
                threshold=threshold,
                confidence=confidence,
                reason=f"Within {distance:.1f}px of bar center (threshold={threshold:.1f}px)"
            )
        return None
    
    def _strategy_spacing_based(
        self, 
        bar: Dict, 
        label: Dict, 
        orientation: str,
        median_spacing: float
    ) -> Optional[AssociationStrategy]:
        """
        Strategy 3: Spacing-based association.
        
        Check if label is within spacing × spacing_factor of bar center.
        (Handles regular layouts with consistent spacing)
        """
        bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
        bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
        label_cx = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
        label_cy = (label['xyxy'][1] + label['xyxy'][3]) / 2.0
        
        if orientation == 'vertical':
            distance = abs(label_cx - bar_cx)
        else:
            distance = abs(label_cy - bar_cy)
        
        threshold = median_spacing * self.spacing_factor
        
        if distance <= threshold:
            confidence = 1.0 - (distance / threshold) * 0.3  # Range [0.7, 1.0]
            return AssociationStrategy(
                name="spacing_based",
                threshold=threshold,
                confidence=confidence,
                reason=f"Within {distance:.1f}px (spacing={median_spacing:.1f}px)"
            )
        return None
    
    def _strategy_zone_fallback(
        self, 
        bar: Dict, 
        label: Dict, 
        orientation: str,
        median_spacing: float
    ) -> Optional[AssociationStrategy]:
        """
        Strategy 4: Zone-based fallback.
        
        Assign label to nearest bar within 2× spacing (last resort).
        """
        bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
        bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
        label_cx = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
        label_cy = (label['xyxy'][1] + label['xyxy'][3]) / 2.0
        
        if orientation == 'vertical':
            distance = abs(label_cx - bar_cx)
        else:
            distance = abs(label_cy - bar_cy)
        
        threshold = median_spacing * self.zone_factor
        
        if distance <= threshold:
            confidence = max(0.3, 1.0 - (distance / threshold))  # Range [0.3, 1.0]
            return AssociationStrategy(
                name="zone_fallback",
                threshold=threshold,
                confidence=confidence,
                reason=f"Fallback: nearest bar (distance={distance:.1f}px)"
            )
        return None
    
    def associate_elements(
        self,
        bars: List[Dict],
        error_bars: List[Dict],
        tick_labels: List[Dict],
        orientation: str = 'vertical',
        **kwargs
    ) -> List[Dict]:
        """
        Robust multi-strategy bar-to-label association.
        
        Args:
            bars: List of bar detections
            error_bars: List of error_bar detections
            tick_labels: List of tick_label detections
            orientation: 'vertical' or 'horizontal'
        
        Returns:
            List of enriched bars with associations and diagnostics
        """
        if not bars:
            return []
        
        # Step 1: Detect layout type
        layout = self.detect_layout(bars, orientation)
        
        # Step 2: Compute median spacing (for spacing-based strategies)
        if orientation == 'vertical':
            positions = sorted([(b['xyxy'][0] + b['xyxy'][2])/2.0 for b in bars])
        else:
            positions = sorted([(b['xyxy'][1] + b['xyxy'][3])/2.0 for b in bars])
        
        spacings = np.diff(positions) if len(positions) > 1 else [100.0]
        median_spacing = np.median(spacings)
        
        self.logger.info(
            f"Association parameters: layout={layout.value}, "
            f"median_spacing={median_spacing:.1f}px, "
            f"n_bars={len(bars)}, n_labels={len(tick_labels)}"
        )
        
        # Step 3: For each bar, find its best matching label
        enriched_bars = []
        
        for bar_idx, bar in enumerate(bars):
            enriched_bar = {
                **bar,
                'associated_error_bar': None,
                'associated_tick_labels': [],
                'association_strategy': None,
                'association_diagnostics': {
                    'layout': layout.value,
                    'strategies_tried': [],
                    'confidence': 0.0
                }
            }
            
            # Find best matching tick_label for this specific bar
            if tick_labels:
                best_label = None
                best_strategy = None
                best_distance = float('inf')
                best_confidence = 0.0
                
                for label in tick_labels:
                    # Try strategies in priority order
                    strategies = [
                        self._strategy_direct_overlap(bar, label, orientation),
                        self._strategy_proximity(bar, label, orientation),
                        self._strategy_spacing_based(bar, label, orientation, median_spacing),
                        self._strategy_zone_fallback(bar, label, orientation, median_spacing)
                    ]
                    
                    # Find first successful strategy for this specific bar-label pair
                    for strategy in strategies:
                        if strategy is not None:
                            # Compute actual distance for this specific bar-label pair
                            bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
                            bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
                            label_cx = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
                            label_cy = (label['xyxy'][1] + label['xyxy'][3]) / 2.0
                            
                            if orientation == 'vertical':
                                distance = abs(label_cx - bar_cx)
                            else:
                                distance = abs(label_cy - bar_cy)
                            
                            # Update best match if this strategy is better (higher confidence or same confidence + shorter distance)
                            if (strategy.confidence > best_confidence or
                                (strategy.confidence == best_confidence and distance < best_distance)):
                                best_label = label
                                best_strategy = strategy
                                best_distance = distance
                                best_confidence = strategy.confidence
                            
                            break  # Use first successful strategy for this label
                
                # Assign best match if found within reasonable distance
                if best_label is not None and best_distance < median_spacing * 2.0:  # Reasonable distance threshold
                    enriched_bar['associated_tick_labels'] = [best_label]
                    enriched_bar['association_strategy'] = best_strategy.name
                    enriched_bar['association_diagnostics']['confidence'] = best_strategy.confidence
                    enriched_bar['association_diagnostics']['reason'] = best_strategy.reason
                    enriched_bar['association_diagnostics']['distance'] = best_distance
                    enriched_bar['association_diagnostics']['strategies_tried'] = [best_strategy.name]
                else:
                    self.logger.warning(
                        f"Bar {bar_idx}: No tick_label association found "
                        f"(tried multiple strategies, best_distance={best_distance:.1f}px)"
                    )
                    enriched_bar['association_diagnostics']['strategies_tried'] = []
            else:
                enriched_bar['association_diagnostics']['strategies_tried'] = []
            
            # Associate error_bars (similar multi-strategy approach)
            if error_bars:
                best_error_bar = None
                best_error_distance = float('inf')
                
                for eb in error_bars:
                    # Try direct overlap first
                    strategy = self._strategy_direct_overlap(bar, eb, orientation)
                    if strategy is None:
                        strategy = self._strategy_proximity(bar, eb, orientation)
                    
                    if strategy is not None:
                        bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
                        bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
                        eb_cx = (eb['xyxy'][0] + eb['xyxy'][2]) / 2.0
                        eb_cy = (eb['xyxy'][1] + eb['xyxy'][3]) / 2.0
                        
                        if orientation == 'vertical':
                            distance = abs(eb_cx - bar_cx)
                        else:
                            distance = abs(eb_cy - bar_cy)
                        
                        if distance < best_error_distance:
                            best_error_distance = distance
                            best_error_bar = eb
                
                if best_error_bar is not None:
                    enriched_bar['associated_error_bar'] = best_error_bar
            
            enriched_bars.append(enriched_bar)
        
        # Step 4: Resolve conflicts (multiple bars claiming same label)
        # This is needed because multiple bars might have been assigned the same best label
        enriched_bars = self._resolve_conflicts(enriched_bars, tick_labels, orientation)
        
        return enriched_bars
    
    def _resolve_conflicts(
        self,
        enriched_bars: List[Dict],
        tick_labels: List[Dict],
        orientation: str
    ) -> List[Dict]:
        """
        Resolve conflicts where multiple bars claim the same tick_label.

        Resolution strategy:
        - Assign to bar with highest confidence
        - If tied confidence, assign to bar with shortest distance
        - Mark conflicted bars with warning
        """
        # Use layout-aware conflict resolution for better handling of grouped layouts
        layout = self.detect_layout([bar for bar in enriched_bars], orientation)
        return self._resolve_conflicts_grouped_aware(enriched_bars, tick_labels, orientation, layout)

    def _resolve_conflicts_grouped_aware(
        self,
        enriched_bars: List[Dict],
        tick_labels: List[Dict],
        orientation: str,
        layout: 'ChartLayout'  # NEW: Pass layout type
    ) -> List[Dict]:
        """
        Layout-aware conflict resolution.

        - SIMPLE: 1 bar per tick label (resolve conflicts)
        - GROUPED: Multiple bars can share same tick label (validate clustering)
        - STACKED: Multiple bars can share same tick label (validate position overlap)
        - MIXED: Use heuristics based on spatial clustering
        """
        label_usage = {}  # label_id → [(bar_idx, confidence, distance), ...]

        # Build usage map
        for bar_idx, bar in enumerate(enriched_bars):
            for label in bar.get('associated_tick_labels', []):
                label_id = id(label)
                if label_id not in label_usage:
                    label_usage[label_id] = []

                confidence = bar['association_diagnostics'].get('confidence', 0.0)
                distance = bar['association_diagnostics'].get('distance', float('inf'))
                label_usage[label_id].append((bar_idx, confidence, distance))

        # Resolve based on layout
        if layout == ChartLayout.GROUPED or layout == ChartLayout.STACKED:
            # GROUPED/STACKED: Multiple bars can share same label
            for label_id, claims in label_usage.items():
                if len(claims) > 1:
                    # Get bar positions
                    bar_positions = []
                    for bar_idx, conf, dist in claims:
                        bar = enriched_bars[bar_idx]
                        if orientation == 'vertical':
                            pos = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
                        else:
                            pos = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
                        bar_positions.append((bar_idx, pos))

                    # Sort by position
                    bar_positions.sort(key=lambda x: x[1])
                    positions_only = [pos for _, pos in bar_positions]
                    max_span = max(positions_only) - min(positions_only)

                    # Verify bars form a spatial cluster
                    # Use adaptive threshold based on bar count
                    cluster_threshold = min(150, 50 * np.sqrt(len(claims)))

                    if max_span < cluster_threshold:
                        # Valid cluster - allow all bars to keep the label
                        self.logger.info(
                            f"Tick_label: {len(claims)} bars in {layout.value} layout "
                            f"(span={max_span:.1f}px < {cluster_threshold:.1f}px threshold) - "
                            f"all share same label"
                        )
                        # Enrich with group metadata
                        for bar_idx, conf, dist in claims:
                            enriched_bars[bar_idx]['association_diagnostics']['group_size'] = len(claims)
                            enriched_bars[bar_idx]['association_diagnostics']['group_span'] = max_span
                            enriched_bars[bar_idx]['association_diagnostics']['is_grouped'] = True
                    else:
                        # Bars too spread - likely false positive, resolve normally
                        self.logger.warning(
                            f"Detected {layout.value} but bars span {max_span:.1f}px > "
                            f"{cluster_threshold:.1f}px threshold - resolving conflict"
                        )
                        self._resolve_single_conflict(enriched_bars, claims, label_id, label_id)

        elif layout == ChartLayout.SIMPLE:
            # SIMPLE: 1 bar per tick label (traditional conflict resolution)
            for label_id, claims in label_usage.items():
                if len(claims) > 1:
                    self._resolve_single_conflict(enriched_bars, claims, label_id, label_id)

        else:  # MIXED
            # MIXED: Use spatial clustering heuristics
            for label_id, claims in label_usage.items():
                if len(claims) > 1:
                    # Check if bars form tight cluster
                    bar_positions = []
                    for bar_idx, conf, dist in claims:
                        bar = enriched_bars[bar_idx]
                        if orientation == 'vertical':
                            pos = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
                        else:
                            pos = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
                        bar_positions.append(pos)

                    max_span = max(bar_positions) - min(bar_positions)

                    if max_span < 100:
                        # Likely grouped - allow multiple
                        self.logger.info(
                            f"MIXED layout: {len(claims)} bars cluster tightly "
                            f"(span={max_span:.1f}px) - allowing shared label"
                        )
                        for bar_idx, conf, dist in claims:
                            enriched_bars[bar_idx]['association_diagnostics']['group_size'] = len(claims)
                            enriched_bars[bar_idx]['association_diagnostics']['is_grouped'] = True
                    else:
                        # Likely true conflict
                        self._resolve_single_conflict(enriched_bars, claims, label_id, label_id)

        return enriched_bars

    def _identify_stacks(
        self,
        bars: List[Dict],
        orientation: str
    ) -> List[Dict]:
        """
        Identify stacks of bars at the same position.

        In stacked bar charts, multiple bars share the same x-position (vertical)
        or y-position (horizontal). This method groups them into stacks.

        Args:
            bars: List of bar detections
            orientation: 'vertical' or 'horizontal'

        Returns:
            List of stack dictionaries, each containing:
            - 'position': float, the shared position of the stack
            - 'bars': List[Dict], all bars in this stack
            - 'bar_indices': List[int], original indices of bars in the stack
        """
        if not bars:
            return []

        # Compute median width/height for tolerance
        if orientation == 'vertical':
            widths = [abs(b['xyxy'][2] - b['xyxy'][0]) for b in bars]
            median_dimension = np.median(widths)
        else:
            heights = [abs(b['xyxy'][3] - b['xyxy'][1]) for b in bars]
            median_dimension = np.median(heights)

        position_tolerance = median_dimension * 0.3

        # Group bars by position
        stacks = []
        processed = set()

        for i, bar in enumerate(bars):
            if i in processed:
                continue

            # Get bar position
            if orientation == 'vertical':
                bar_pos = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
            else:
                bar_pos = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0

            # Find all bars at the same position
            stack_bars = [bar]
            stack_indices = [i]
            processed.add(i)

            for j, other_bar in enumerate(bars):
                if j in processed or j == i:
                    continue

                if orientation == 'vertical':
                    other_pos = (other_bar['xyxy'][0] + other_bar['xyxy'][2]) / 2.0
                else:
                    other_pos = (other_bar['xyxy'][1] + other_bar['xyxy'][3]) / 2.0

                # Check if positions overlap within tolerance
                if abs(bar_pos - other_pos) < position_tolerance:
                    stack_bars.append(other_bar)
                    stack_indices.append(j)
                    processed.add(j)

            # Sort bars within stack by position in the perpendicular axis
            # (bottom to top for vertical, left to right for horizontal)
            if orientation == 'vertical':
                # Sort by y-coordinate (bottom to top = larger y to smaller y)
                stack_bars_sorted = sorted(
                    zip(stack_bars, stack_indices),
                    key=lambda x: x[0]['xyxy'][3],  # Bottom edge
                    reverse=True  # Bottom bars have larger y
                )
            else:
                # Sort by x-coordinate (left to right)
                stack_bars_sorted = sorted(
                    zip(stack_bars, stack_indices),
                    key=lambda x: x[0]['xyxy'][0]  # Left edge
                )

            stack_bars, stack_indices = zip(*stack_bars_sorted)

            stacks.append({
                'position': bar_pos,
                'bars': list(stack_bars),
                'bar_indices': list(stack_indices),
                'stack_size': len(stack_bars)
            })

        self.logger.info(
            f"Identified {len(stacks)} stacks from {len(bars)} bars "
            f"(orientation={orientation})"
        )

        return stacks

    def _resolve_single_conflict(self, enriched_bars: List[Dict], claims: List, label_id: int):
        """
        Helper to resolve a single conflict by picking winner based on confidence and distance.
        """
        # Sort by confidence (desc), then distance (asc)
        claims.sort(key=lambda x: (-x[1], x[2]))
        winner_idx = claims[0][0]

        self.logger.warning(
            f"Tick_label conflict: {len(claims)} bars compete, "
            f"assigned to bar {winner_idx} "
            f"(confidence={claims[0][1]:.2f}, distance={claims[0][2]:.1f}px)"
        )

        # Remove label from losers
        for bar_idx, conf, dist in claims[1:]:
            enriched_bars[bar_idx]['associated_tick_labels'] = [
                lbl for lbl in enriched_bars[bar_idx]['associated_tick_labels']
                if id(lbl) != label_id
            ]
            enriched_bars[bar_idx]['association_diagnostics']['conflict'] = True
            enriched_bars[bar_idx]['association_diagnostics']['conflict_reason'] = (
                f"Lost to bar {winner_idx} (conf={conf:.2f} vs {claims[0][1]:.2f})"
            )

    def associate_elements_with_stacks(
        self,
        bars: List[Dict],
        error_bars: List[Dict],
        tick_labels: List[Dict],
        orientation: str = 'vertical',
        **kwargs
    ) -> List[Dict]:
        """
        Enhanced association that treats stacks as single units.

        For STACKED layouts:
        - Groups bars into stacks
        - Associates tick label with the stack (not individual bars)
        - Propagates the tick label to ALL bars in the stack

        For other layouts (SIMPLE, GROUPED, MIXED):
        - Uses standard association logic

        Returns:
            List of enriched bars with tick_label associations
        """
        if not bars:
            return []

        # Detect layout
        layout = self.detect_layout(bars, orientation)

        # Compute median spacing
        if orientation == 'vertical':
            positions = sorted([(b['xyxy'][0] + b['xyxy'][2])/2.0 for b in bars])
        else:
            positions = sorted([(b['xyxy'][1] + b['xyxy'][3])/2.0 for b in bars])

        spacings = np.diff(positions) if len(positions) > 1 else [100.0]
        median_spacing = np.median(spacings)

        self.logger.info(
            f"Association parameters: layout={layout.value}, "
            f"median_spacing={median_spacing:.1f}px, "
            f"n_bars={len(bars)}, n_labels={len(tick_labels)}"
        )

        # Branch based on layout
        if layout == ChartLayout.STACKED:
            # Use stack-based association
            enriched_bars = self._associate_stacks_with_labels(
                bars, tick_labels, orientation, median_spacing
            )
        else:
            # Use standard bar-by-bar association
            enriched_bars = self._associate_bars_with_labels(
                bars, tick_labels, orientation, median_spacing
            )

            # Apply layout-aware conflict resolution
            enriched_bars = self._resolve_conflicts_grouped_aware(
                enriched_bars, tick_labels, orientation, layout
            )

        return enriched_bars

    def _associate_stacks_with_labels(
        self,
        bars: List[Dict],
        tick_labels: List[Dict],
        orientation: str,
        median_spacing: float
    ) -> List[Dict]:
        """
        Associate tick labels with stacks, then propagate to all bars in each stack.

        Strategy:
        1. Identify stacks (groups of bars at same position)
        2. For each stack, find best matching tick label
        3. Propagate that tick label to ALL bars in the stack

        Returns:
            List of enriched bars with propagated tick_label associations
        """
        # Step 1: Identify stacks
        stacks = self._identify_stacks(bars, orientation)

        # Step 2: Associate each stack with a tick label
        for stack_idx, stack in enumerate(stacks):
            stack_pos = stack['position']

            # Find best matching tick label for this stack
            best_label = None
            best_distance = float('inf')
            best_strategy = None
            best_confidence = 0.0

            for label in tick_labels:
                # Get label position
                if orientation == 'vertical':
                    label_pos = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
                else:
                    label_pos = (label['xyxy'][1] + label['xyxy'][3]) / 2.0

                distance = abs(stack_pos - label_pos)

                # Try association strategies
                # Use representative bar (first bar in stack) for strategy evaluation
                representative_bar = stack['bars'][0]

                strategies = [
                    self._strategy_direct_overlap(representative_bar, label, orientation),
                    self._strategy_proximity(representative_bar, label, orientation),
                    self._strategy_spacing_based(representative_bar, label, orientation, median_spacing),
                    self._strategy_zone_fallback(representative_bar, label, orientation, median_spacing)
                ]

                # Find first successful strategy
                for strategy in strategies:
                    if strategy is not None:
                        if (strategy.confidence > best_confidence or
                            (strategy.confidence == best_confidence and distance < best_distance)):
                            best_label = label
                            best_distance = distance
                            best_strategy = strategy
                            best_confidence = strategy.confidence
                        break

            # Store association for this stack
            stack['assigned_label'] = best_label
            stack['label_distance'] = best_distance
            stack['strategy'] = best_strategy
            stack['confidence'] = best_confidence

            if best_label:
                self.logger.info(
                    f"Stack {stack_idx} (pos={stack_pos:.1f}, {stack['stack_size']} bars): "
                    f"assigned to tick_label '{best_label.get('text', 'N/A')}' "
                    f"(strategy={best_strategy.name if best_strategy else 'none'}, "
                    f"confidence={best_confidence:.2f}, distance={best_distance:.1f}px)"
                )

        # Step 3: Propagate tick label to all bars in each stack
        enriched_bars = [None] * len(bars)  # Maintain original ordering

        for stack_idx, stack in enumerate(stacks):
            assigned_label = stack['assigned_label']

            if assigned_label is None:
                self.logger.warning(
                    f"Stack {stack_idx} at pos={stack['position']:.1f}: "
                    f"No tick label assigned"
                )

            # Propagate to all bars in this stack
            for pos_in_stack, bar_idx in enumerate(stack['bar_indices']):
                bar = stack['bars'][pos_in_stack]

                enriched_bar = dict(bar)

                if assigned_label:
                    enriched_bar['associated_tick_labels'] = [assigned_label]
                    enriched_bar['association_diagnostics'] = {
                        'strategy': stack['strategy'].name if stack['strategy'] else 'none',
                        'confidence': stack['confidence'],
                        'distance': stack['label_distance'],
                        'layout': 'stacked',
                        'stack_id': stack_idx,
                        'stack_size': stack['stack_size'],
                        'position_in_stack': pos_in_stack,
                        'is_stacked': True,
                        'propagated': True  # Indicates label was propagated from stack
                    }

                    self.logger.debug(
                        f"  Bar {bar_idx} (pos_in_stack={pos_in_stack}/{stack['stack_size']-1}): "
                        f"propagated tick_label '{assigned_label.get('text', 'N/A')}'"
                    )
                else:
                    enriched_bar['associated_tick_labels'] = []
                    enriched_bar['association_diagnostics'] = {
                        'strategy': 'none',
                        'confidence': 0.0,
                        'distance': float('inf'),
                        'layout': 'stacked',
                        'stack_id': stack_idx,
                        'stack_size': stack['stack_size'],
                        'position_in_stack': pos_in_stack,
                        'is_stacked': True,
                        'error': 'no_label_found_for_stack'
                    }

                enriched_bars[bar_idx] = enriched_bar

        return enriched_bars

    def _associate_bars_with_labels(
        self,
        bars: List[Dict],
        tick_labels: List[Dict],
        orientation: str,
        median_spacing: float
    ) -> List[Dict]:
        """
        Standard bar-by-bar association (for SIMPLE, GROUPED, MIXED layouts).
        This is the existing logic.
        """
        enriched_bars = []

        for bar_idx, bar in enumerate(bars):
            best_label = None
            best_strategy = None
            best_distance = float('inf')
            best_confidence = 0.0

            for label in tick_labels:
                # Try strategies in priority order
                strategies = [
                    self._strategy_direct_overlap(bar, label, orientation),
                    self._strategy_proximity(bar, label, orientation),
                    self._strategy_spacing_based(bar, label, orientation, median_spacing),
                    self._strategy_zone_fallback(bar, label, orientation, median_spacing)
                ]

                # Find first successful strategy
                for strategy in strategies:
                    if strategy is not None:
                        # Compute distance
                        bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
                        bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
                        label_cx = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
                        label_cy = (label['xyxy'][1] + label['xyxy'][3]) / 2.0

                        if orientation == 'vertical':
                            distance = abs(label_cx - bar_cx)
                        else:
                            distance = abs(label_cy - bar_cy)

                        # Update best match
                        if (strategy.confidence > best_confidence or
                            (strategy.confidence == best_confidence and distance < best_distance)):
                            best_label = label
                            best_strategy = strategy
                            best_distance = distance
                            best_confidence = strategy.confidence

                        break

            # Create enriched bar
            enriched_bar = dict(bar)

            if best_label:
                enriched_bar['associated_tick_labels'] = [best_label]
                enriched_bar['association_diagnostics'] = {
                    'strategy': best_strategy.name,
                    'confidence': best_confidence,
                    'distance': best_distance,
                    'threshold': best_strategy.threshold,
                    'reason': best_strategy.reason
                }
            else:
                enriched_bar['associated_tick_labels'] = []
                enriched_bar['association_diagnostics'] = {
                    'strategy': 'none',
                    'confidence': 0.0,
                    'distance': float('inf'),
                    'error': 'no_matching_label'
                }

            enriched_bars.append(enriched_bar)

        return enriched_bars

    def associate_elements(
        self,
        bars: List[Dict],
        error_bars: List[Dict],
        tick_labels: List[Dict],
        orientation: str = 'vertical',
        **kwargs
    ) -> List[Dict]:
        """
        Main entry point for bar-to-label association.

        Routes to appropriate association method based on layout:
        - STACKED: Stack-based association with label propagation
        - SIMPLE/GROUPED/MIXED: Bar-by-bar association with conflict resolution
        """
        # Use the enhanced method that handles both cases
        enriched_bars = self.associate_elements_with_stacks(
            bars=bars,
            error_bars=error_bars,
            tick_labels=tick_labels,
            orientation=orientation,
            **kwargs
        )

        return enriched_bars

    def associate_data_labels(
        self,
        enriched_bars: List[Dict],
        data_labels: List[Dict],
        orientation: str
    ) -> List[Dict]:
        """
        Apply multi-strategy association to data labels with conflict resolution.
        Uses the same robust strategies as tick_labels.

        Args:
            enriched_bars: Bars already enriched with tick_label associations
            data_labels: List of data_label detections
            orientation: 'vertical' or 'horizontal'

        Returns:
            Bars further enriched with data_label associations
        """
        if not data_labels:
            return enriched_bars

        # Compute median spacing for spacing-based strategy
        if orientation == 'vertical':
            positions = sorted([(b['xyxy'][0] + b['xyxy'][2])/2.0 for b in enriched_bars])
        else:
            positions = sorted([(b['xyxy'][1] + b['xyxy'][3])/2.0 for b in enriched_bars])

        spacings = np.diff(positions) if len(positions) > 1 else [100.0]
        median_spacing = np.median(spacings)

        # For each bar, find best matching data_label
        for bar_idx, bar in enumerate(enriched_bars):
            best_label = None
            best_strategy = None
            best_distance = float('inf')
            best_confidence = 0.0

            for label in data_labels:
                # Try strategies in priority order
                strategies = [
                    self._strategy_direct_overlap(bar, label, orientation),
                    self._strategy_proximity(bar, label, orientation),
                    self._strategy_spacing_based(bar, label, orientation, median_spacing),
                    self._strategy_zone_fallback(bar, label, orientation, median_spacing)
                ]

                # Find first successful strategy
                for strategy in strategies:
                    if strategy is not None:
                        # Compute distance
                        bar_cx = (bar['xyxy'][0] + bar['xyxy'][2]) / 2.0
                        bar_cy = (bar['xyxy'][1] + bar['xyxy'][3]) / 2.0
                        label_cx = (label['xyxy'][0] + label['xyxy'][2]) / 2.0
                        label_cy = (label['xyxy'][1] + label['xyxy'][3]) / 2.0

                        if orientation == 'vertical':
                            distance = abs(label_cx - bar_cx)
                        else:
                            distance = abs(label_cy - bar_cy)

                        # Update best match
                        if (strategy.confidence > best_confidence or
                            (strategy.confidence == best_confidence and distance < best_distance)):
                            best_label = label
                            best_strategy = strategy
                            best_distance = distance
                            best_confidence = strategy.confidence

                        break  # Use first successful strategy

            # Assign best match
            if best_label is not None:
                enriched_bars[bar_idx]['associated_data_label'] = {
                    'label': best_label,
                    'strategy': best_strategy.name,
                    'confidence': best_strategy.confidence,
                    'distance': best_distance,
                    'reason': best_strategy.reason
                }

        # Resolve conflicts (multiple bars claiming same data_label)
        enriched_bars = self._resolve_data_label_conflicts(enriched_bars, orientation)

        return enriched_bars

    def _resolve_data_label_conflicts(
        self,
        enriched_bars: List[Dict],
        orientation: str
    ) -> List[Dict]:
        """
        Resolve conflicts where multiple bars claim the same data_label.

        Resolution strategy:
        - Assign to bar with highest confidence
        - If tied, assign to bar with shortest distance
        - Mark conflicted bars
        """
        label_usage = {}  # label_id → [(bar_idx, confidence, distance), ...]

        # Build usage map
        for bar_idx, bar in enumerate(enriched_bars):
            if 'associated_data_label' in bar:
                label = bar['associated_data_label']['label']
                label_id = id(label)

                if label_id not in label_usage:
                    label_usage[label_id] = []

                confidence = bar['associated_data_label']['confidence']
                distance = bar['associated_data_label']['distance']
                label_usage[label_id].append((bar_idx, confidence, distance))

        # Resolve conflicts
        for label_id, claims in label_usage.items():
            if len(claims) > 1:
                # Sort by confidence (desc), then distance (asc)
                claims.sort(key=lambda x: (-x[1], x[2]))
                winner_idx = claims[0][0]

                self.logger.warning(
                    f"Data_label conflict: {len(claims)} bars compete, "
                    f"assigned to bar {winner_idx} "
                    f"(confidence={claims[0][1]:.2f}, distance={claims[0][2]:.1f}px)"
                )

                # Remove label from losers and mark conflict
                for bar_idx, conf, dist in claims[1:]:
                    del enriched_bars[bar_idx]['associated_data_label']
                    if 'association_conflicts' not in enriched_bars[bar_idx]:
                        enriched_bars[bar_idx]['association_conflicts'] = []
                    enriched_bars[bar_idx]['association_conflicts'].append({
                        'type': 'data_label',
                        'lost_confidence': conf,
                        'lost_distance': dist,
                        'winner_bar': winner_idx
                    })

        return enriched_bars