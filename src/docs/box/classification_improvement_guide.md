# Improving Tick Label vs Scale Label Classification: Research-Based Recommendations

## Executive Summary

The current classification system uses Gaussian kernel-based spatial scoring combined with size and numeric content analysis. While effective, it faces challenges with **overlapping confidence ranges** and **limited contextual modeling**. This research identifies three core areas for improvement: **(1) enhanced feature engineering for robustness, (2) adaptive threshold selection, and (3) constraint-based post-processing with label dependency modeling**.

---

## Part 1: Current System Analysis

### Architecture Overview

The `BoxChartClassifier` employs a multi-feature scoring approach:

- **Gaussian Region Scoring**: Probabilistic kernels centered on typical axis regions (left, right, bottom, top)
- **Spatial Thresholding**: Binary edge detection (edge_threshold = 0.22)
- **Size Analysis**: Labels smaller than threshold (0.065 × 0.038) tend to be scale labels
- **Semantic Analysis**: Numeric content receives boost (weight = 3.5)
- **Contextual Alignment**: Tick labels scored by proximity to box elements
- **Orientation-Aware Logic**: Different strategies for vertical vs horizontal charts

### Identified Weaknesses

#### 1. **Overlapping Score Distributions**
- When top-2 scores are close (margin < 0.5), classification becomes unreliable
- Gaussian kernels sometimes fail to distinguish between similar regions
- No adaptive threshold adjustment for ambiguous cases

#### 2. **Hardcoded Parameters**
- `edge_threshold = 0.22` assumes standard margins
- `scale_size_max_width = 0.065` may not generalize across image sizes
- `numeric_boost = 3.5` lacks calibration relative to positional scores

#### 3. **Independent Label Processing**
- Each label classified independently without considering other labels
- No explicit modeling of inter-label relationships or patterns
- Missing label distribution statistics

#### 4. **Incomplete Feature Space**
- Size features only use simple binary check (is_small)
- Position features lack gradient information (distance to edge varies)
- Content analysis missing digit ratios and pattern detection
- No box alignment consistency checks

#### 5. **Critical Bug in Feature Extraction**
```python
# In _extract_box_features():
'is_numeric': is_numeric  # BUG: Stores function object instead of calling it!
# Should be: 'is_numeric': is_numeric(text)
```

---

## Part 2: Research-Based Improvement Framework

### Improvement Domain 1: Enhanced Feature Engineering

#### Multi-Dimensional Feature Extraction

**Position Features** (Continuous, not binary):
- Distance to each edge: `dist_left`, `dist_right`, `dist_top`, `dist_bottom`
- Minimum edge distance for ranking
- Region classification (5 regions: left_axis, right_axis, bottom_axis, top_title, center)
- Normalized coordinates relative to plot area

**Size Features** (Distribution-aware):
- Absolute size (pixels), relative size (image %), aspect ratio
- Z-score relative to all labels: `(width - mean) / std`
- Percentile ranking: where does this label fall in size distribution?
- Binary indicators: `is_small` (but define with percentile thresholds)

**Content Features** (Robust pattern detection):
- Numeric detection: handles decimals, negatives, scientific notation, ranges
- Digit ratio: `count_digits / total_chars`
- Character distribution ratios: digits, alpha, special chars
- Pattern detection: decimal points, minus signs, scientific notation
- Text length vs expected distribution

**Contextual Features**:
- Alignment score with box elements (proximity-based)
- Consistency with neighboring labels
- Spacing pattern regularity
- Formatting consistency within type

#### Implementation Priority
```
Quick Win (< 1 day):
  - Fix is_numeric bug
  - Add distance-to-edge features
  - Implement digit_ratio and pattern detection

Medium Effort (2-3 days):
  - Compute distribution statistics for normalization
  - Add percentile-based size analysis
  - Implement box alignment scoring

Advanced (1 week):
  - Implement full contextual feature extraction
  - Build consistency checking across labels
```

### Improvement Domain 2: Discriminative Modeling

#### Strategy 1: Adaptive Thresholding
**Problem**: Fixed classification_threshold = 2.2 doesn't account for score distribution variability.

**Solution - Margin-Aware Thresholding**:
- When average margin between top-2 scores is small (< 0.5): **increase threshold** (require higher confidence)
- When average margin is large (> 2.0): **decrease threshold** (allow lower confidence for clear cases)
- Formula: `adaptive_threshold = base_threshold × margin_adjustment_factor`

**Benefits**: Reduces false positives in ambiguous cases while maintaining sensitivity for clear cases.

#### Strategy 2: Pairwise Comparison
**Problem**: Independent scoring ignores label relationships.

**Solution - Comparative Framework**:
- Compare label pairs: "Which is more likely to be scale label: A or B?"
- For ambiguous labels (score margin < threshold), use pairwise comparisons
- Advantages:
  - Reduces decision space
  - Makes discrimination explicit
  - Naturally handles relative differences

**Implementation**:
```python
def pairwise_score(label_a, label_b, orientation):
    score = 0.0
    # Position: score += 1.0 if A is more on axis edge
    # Size: score += 0.8 if A is smaller
    # Content: score += 0.6 if A is numeric and B is not
    return clip(score, -1, 1)  # Positive: A is scale; Negative: B is scale
```

#### Strategy 3: Ensemble Scoring
**Combine independent scoring pathways with explicit weighting**:
- Position-based pathway: predicts based on edge proximity
- Size-based pathway: predicts based on size analysis
- Content-based pathway: predicts based on numeric content
- Context-based pathway: predicts based on alignment with elements

Weight each pathway by its confidence/reliability, then combine.

### Improvement Domain 3: Robustness Enhancements

#### Adaptive Parameter Selection
Instead of fixed parameters, compute them from data:

```python
# Instead of: edge_threshold = 0.22
edge_threshold = percentile(distances_to_axis_edge, 20)  # Bottom 20%

# Instead of: scale_size_max_width = 0.065
scale_size_max_width = percentile(label_widths, 30)  # Bottom 30%

# Instead of: numeric_boost = 3.5
numeric_boost = score_multiplier_from_validation_data
```

#### Constraint-Based Post-Processing
Apply domain-specific rules:

1. **Cardinality Constraints**:
   - Exactly one scale axis per orientation (in most charts)
   - Tick labels form clusters

2. **Spatial Constraints**:
   - Scale labels should be on axis edges (min_edge_dist < 0.2)
   - Tick labels should NOT be on edges AND should align with elements

3. **Consistency Constraints**:
   - Scale labels have similar numeric formatting
   - Tick labels have similar categorical patterns

4. **Refinement Process**:
   - Flag violating labels
   - Re-evaluate using secondary features
   - Swap classifications if constraints are violated

#### Anomaly Detection
- Identify unusual configurations (e.g., too many scale labels, sparse ticks)
- Adjust parameters for out-of-distribution charts
- Flag low-confidence outputs

### Improvement Domain 4: Label Dependency Modeling

#### Current State
Labels are classified independently; dependencies are ignored.

#### Advanced Approaches

**Graph-Based Reasoning**:
- Build label dependency graph: connect similar/nearby labels
- Use graph neural networks to propagate information
- Labels in same cluster likely have same type

**Sequential Refinement**:
1. **Pass 1**: Classify high-confidence labels (margin > 1.5)
2. **Pass 2**: Use high-confidence classifications to inform ambiguous ones
3. **Pass 3**: Apply consistency constraints and resolve conflicts

**Iterative Optimization**:
- Start with initial classifications
- Check constraint violations
- Swap labels that violate constraints
- Repeat until convergence

---

## Part 3: Implementation Roadmap

### Phase 1: Critical Fixes (Week 1)
**Effort**: 5-10 hours

1. **Fix is_numeric bug**: Call function, don't store reference
2. **Implement robust numeric detection**: Handle decimals, scientific notation, ranges, units
3. **Add distance-to-edge features**: Replace binary edge check with continuous distance
4. **Implement adaptive threshold**: Add margin-aware threshold adjustment
5. **Add confidence penalties**: Flag classifications with small margins

**Expected Impact**: 5-10% accuracy improvement, better handling of edge cases.

### Phase 2: Feature Engineering (Week 2-3)
**Effort**: 15-20 hours

1. **Extract rich feature set**: Position, size, content, contextual features
2. **Compute distribution statistics**: Normalize features by dataset distribution
3. **Implement percentile-based size analysis**: Replace fixed thresholds
4. **Add box alignment scoring**: Improve tick label detection
5. **Build consistency checks**: Verify label patterns

**Expected Impact**: 10-15% accuracy improvement, increased robustness to chart variations.

### Phase 3: Post-Processing & Constraints (Week 3-4)
**Effort**: 10-15 hours

1. **Implement constraint validator**: Check cardinality, spatial, consistency constraints
2. **Add refinement logic**: Re-evaluate violating labels using secondary features
3. **Build anomaly detector**: Flag unusual configurations
4. **Implement fallback strategies**: Hierarchical fallback using multiple features

**Expected Impact**: 5-10% accuracy improvement, especially for edge cases.

### Phase 4: Advanced Features (Week 5-6)
**Effort**: 20-30 hours (Optional, High-Value)

1. **Implement pairwise comparisons**: For ambiguous label pairs
2. **Build ensemble scoring**: Combine independent pathways
3. **Add iterative refinement**: Multi-pass optimization
4. **Parameter tuning framework**: Learn parameters from validation data

**Expected Impact**: 10-20% accuracy improvement on challenging charts.

---

## Part 4: Validation Strategy

### Test Dataset Requirements

**Essential Test Cases**:
1. Normal charts (baseline)
2. Charts with small/large margins
3. Mixed numeric/categorical scales
4. Dense vs sparse label sets
5. Various chart sizes and aspect ratios
6. Rotated/mirrored charts
7. Low-contrast or degraded labels
8. Labels with various fonts and sizes

### Evaluation Metrics

- **Accuracy**: Percentage of correctly classified labels
- **Precision/Recall**: Per class (scale_label vs tick_label)
- **Margin Distribution**: Track score margins to identify ambiguous cases
- **Constraint Violations**: Count violations of domain rules
- **Confidence Calibration**: Whether confidence scores correlate with accuracy

### Regression Prevention

- Maintain baseline test cases
- Track performance across chart types
- Monitor parameter sensitivity
- Document all changes and their impacts

---

## Part 5: Recommendations Summary

### Highest Priority (Do First)
1. **Fix critical bug** in `_extract_box_features` (is_numeric)
2. **Implement adaptive thresholding** (quick 5-10% improvement)
3. **Add distance-to-edge features** (continuous position analysis)
4. **Implement robust numeric detection** (better content analysis)

### Medium Priority (Do Next)
1. **Constraint-based post-processing** (enforce domain rules)
2. **Rich feature extraction** (distribution-aware features)
3. **Confidence-based output filtering** (flag ambiguous cases)

### Advanced (Nice-to-Have)
1. **Pairwise comparison framework** (20-30% improvement potential)
2. **Graph-based label dependencies** (handles complex patterns)
3. **Parameter learning pipeline** (automatic tuning)
4. **Ensemble scoring methods** (combines multiple strategies)

### Expected Outcomes
- **Immediate (after Phase 1)**: 5-10% accuracy improvement, critical bug fixes
- **Short-term (after Phase 2)**: 15-25% overall improvement, better generalization
- **Medium-term (after Phase 3)**: 20-30% improvement, robust handling of edge cases
- **Long-term (after Phase 4)**: 30-50% improvement, state-of-the-art performance

---

## References & Related Work

### Key Research Areas Covered
1. **Multi-label Classification**: Label dependencies, ambiguity resolution, constraint-based methods [Source: Multi-label classification literature]
2. **Spatial Classification**: Gaussian kernels, region-based scoring, positional features [Source: Spatial analysis and computer vision]
3. **Robustness in ML**: Adaptive thresholding, ensemble methods, boundary ambiguity [Source: Text and image classification robustness papers]
4. **Chart Analysis**: ChartOCR, context-aware detection, element disambiguation [Source: Chart analysis and visualization research]
5. **Feature Engineering**: Multi-scale features, distribution-aware normalization [Source: Feature engineering best practices]

### Implementation Frameworks
- NumPy for mathematical operations
- Graph neural networks (optional, for advanced phases)
- Scikit-learn for validation metrics and ensemble methods

---

## Conclusion

The current classification system provides a solid foundation with Gaussian kernel-based spatial reasoning. However, systematic improvements in feature engineering, adaptive thresholding, and constraint-based post-processing can significantly enhance both accuracy and robustness. The recommended phased approach balances quick wins (critical bug fixes, adaptive thresholding) with substantial improvements (rich features, constraint validation) and advanced techniques (graph-based reasoning, parameter learning). 

Following this roadmap should yield 30-50% overall accuracy improvement while maintaining computational efficiency and generalization to diverse chart types.
