# Detailed Implementation Guide: Classification Improvements

## Section 1: Priority 1 Implementations (Can Be Done This Week)

### 1.1 Fix the is_numeric Bug

**Current Code (BUGGY)**:
```python
# In _extract_box_features():
'is_numeric': is_numeric,  # Stores function object!
```

**Corrected Code**:
```python
'is_numeric': is_numeric(text),  # Call the function
```

**Verify the fix works**:
```python
# Test cases
assert is_numeric("123") == True
assert is_numeric("12.34") == True
assert is_numeric("Category A") == False
assert is_numeric("-5") == True
assert is_numeric("1.23e-4") == True
```

---

### 1.2 Implement Robust Numeric Detection

**Problem**: Current `is_numeric()` may not handle all scale label formats (e.g., "0-10", "50%", "100ms").

**Solution**:
```python
def is_numeric_robust(text: str) -> bool:
    """
    Robust numeric detection for chart labels.
    
    Handles:
    - Integers: 123, -456
    - Decimals: 12.34, -0.5
    - Scientific: 1.23e-4, 5E+10
    - Ranges: 0-10, 10-20
    - With units: 50%, 100ms, 5°C
    - Special chars: ±100, ≤50
    """
    if not text or not text.strip():
        return False
    
    text = text.strip()
    
    # Remove common units and symbols
    text_clean = text
    for unit in ['%', 'ms', 'MHz', 'GHz', 'GB', 'MB', 'KB', '°C', '°F', 
                 's', 'ms', 'μs', '$', '€', '£', '¥', '±', '≤', '≥']:
        text_clean = text_clean.replace(unit, '')
    
    text_clean = text_clean.strip()
    if not text_clean:
        return False
    
    # Check characters are allowed
    allowed = set('0123456789.-+eE() ')
    if not all(c in allowed for c in text_clean):
        return False
    
    # Try parsing as float
    try:
        float(text_clean)
        return True
    except ValueError:
        pass
    
    # Try range format (e.g., "0-10")
    if '-' in text_clean and text_clean.count('-') == 1:
        parts = text_clean.split('-')
        if len(parts) == 2:
            try:
                float(parts[0])
                float(parts[1])
                return True
            except ValueError:
                pass
    
    # Try parenthesized range (e.g., "(0-10)")
    if text_clean.startswith('(') and text_clean.endswith(')'):
        inner = text_clean[1:-1]
        if '-' in inner and inner.count('-') == 1:
            parts = inner.split('-')
            if len(parts) == 2:
                try:
                    float(parts[0])
                    float(parts[1])
                    return True
                except ValueError:
                    pass
    
    return False
```

**Update in classifier**:
```python
# In _extract_box_features:
from utils.validation_utils import is_numeric_robust  # Import new function

is_num = is_numeric_robust(text)  # Use robust version
```

---

### 1.3 Replace Binary Edge Detection with Continuous Distance Features

**Current Code**:
```python
if nx < self.params['edge_threshold']:
    scores['scale_label'] += self.params['scale_edge_weight']
```

**Improved Code**:
```python
# Extract distance features
dist_left = nx
dist_right = 1.0 - nx
dist_top = ny
dist_bottom = 1.0 - ny
min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)

# Use continuous distance instead of binary threshold
# Closer to edge = higher scale label score
edge_score = max(0, (0.2 - min_edge_dist) / 0.2)  # Normalize to [0, 1]
scores['scale_label'] += edge_score * self.params['scale_edge_weight']
```

**Benefits**:
- Smooth scoring function (no cliff at threshold)
- Better gradients for optimization
- Handles unusual margins better

---

### 1.4 Implement Adaptive Thresholding

**Current Code**:
```python
threshold = self.params['classification_threshold']  # Fixed: 2.2
if scores[best_class] > threshold:
    classified[best_class].append(feat['label'])
```

**Improved Code**:
```python
# Compute margin between top-2 scores
sorted_scores = sorted(scores.values(), reverse=True)
margin = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else float('inf')

# Adaptive threshold
base_threshold = self.params['classification_threshold']
if margin < 0.5:  # Ambiguous
    threshold = base_threshold * 1.3  # Increase threshold
elif margin > 2.0:  # Clear
    threshold = base_threshold * 0.9  # Decrease threshold
else:
    threshold = base_threshold

if scores[best_class] > threshold:
    classified[best_class].append(feat['label'])
else:
    # Use enhanced fallback
    self._apply_fallback_classification(feat, classified)
```

---

### 1.5 Enhanced Fallback Logic

**Current Code**:
```python
# Box default: numeric = scale, non-numeric = tick
if feat['is_numeric']:
    classified['scale_label'].append(feat['label'])
else:
    classified['tick_label'].append(feat['label'])
```

**Improved Code**:
```python
def _apply_fallback_classification(self, feat, classified):
    """Apply multi-stage fallback when primary classification is ambiguous."""
    
    # Stage 1: Position-based fallback
    min_edge_dist = feat['min_edge_dist']  # From enhanced features
    
    if min_edge_dist < 0.15:  # Very close to edge
        classified['scale_label'].append(feat['label'])
        return
    elif min_edge_dist > 0.3:  # Far from edges
        classified['tick_label'].append(feat['label'])
        return
    
    # Stage 2: Size-based fallback
    is_small = feat['is_small']
    if is_small:
        classified['scale_label'].append(feat['label'])
        return
    
    # Stage 3: Content-based fallback (original logic)
    if feat['is_numeric']:
        classified['scale_label'].append(feat['label'])
    else:
        classified['tick_label'].append(feat['label'])
```

---

## Section 2: Priority 2 Implementations (2-3 Days Work)

### 2.1 Extract Enhanced Feature Set

```python
def _extract_enhanced_features(self, labels, boxes, img_width, img_height):
    """
    Extract comprehensive features for classification.
    
    Returns Dict with:
    - Position features (continuous and regional)
    - Size features (absolute, relative, percentile-based)
    - Content features (robust numeric detection)
    - Context features (alignment with elements)
    """
    
    features = []
    
    # First pass: compute statistics for normalization
    all_widths = []
    all_heights = []
    all_texts = []
    
    for label in labels:
        x1, y1, x2, y2 = label['xyxy']
        w, h = x2 - x1, y2 - y1
        all_widths.append(w)
        all_heights.append(h)
        if 'text' in label:
            all_texts.append(label['text'])
    
    # Compute statistics
    size_stats = {
        'width_mean': np.mean(all_widths) if all_widths else 0,
        'width_std': np.std(all_widths) if all_widths else 1,
        'height_mean': np.mean(all_heights) if all_heights else 0,
        'height_std': np.std(all_heights) if all_heights else 1,
        'width_p25': np.percentile(all_widths, 25) if all_widths else 0,
        'width_p75': np.percentile(all_widths, 75) if all_widths else 0,
    }
    
    text_stats = {
        'mean_len': np.mean([len(t) for t in all_texts]) if all_texts else 0,
        'max_len': max([len(t) for t in all_texts]) if all_texts else 0,
    }
    
    # Second pass: extract features for each label
    for label in labels:
        x1, y1, x2, y2 = label['xyxy']
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
        text = label.get('text', '')
        
        # Position features
        nx, ny = cx / img_width, cy / img_height
        dist_left = nx
        dist_right = 1.0 - nx
        dist_top = ny
        dist_bottom = 1.0 - ny
        min_edge_dist = min(dist_left, dist_right, dist_top, dist_bottom)
        
        # Region classification
        region = 'center'
        if nx < 0.2 and 0.1 < ny < 0.9:
            region = 'left_axis'
        elif nx > 0.8 and 0.1 < ny < 0.9:
            region = 'right_axis'
        elif ny > 0.8 and 0.15 < nx < 0.85:
            region = 'bottom_axis'
        elif ny < 0.15 and 0.15 < nx < 0.85:
            region = 'top_title'
        
        # Size features
        rel_w = width / img_width
        rel_h = height / img_height
        aspect = width / (height + 1e-6)
        
        # Size z-scores
        width_zscore = (width - size_stats['width_mean']) / (size_stats['width_std'] + 1e-6)
        height_zscore = (height - size_stats['height_mean']) / (size_stats['height_std'] + 1e-6)
        
        # Size percentiles (what % of labels are smaller?)
        width_percentile = 100 * np.sum(np.array(all_widths) < width) / len(all_widths) if all_widths else 0
        height_percentile = 100 * np.sum(np.array(all_heights) < height) / len(all_heights) if all_heights else 0
        
        is_small = (rel_w < 0.08) and (rel_h < 0.05)
        
        # Content features
        is_num = is_numeric_robust(text)
        digit_count = sum(1 for c in text if c.isdigit())
        digit_ratio = digit_count / max(len(text), 1)
        alpha_count = sum(1 for c in text if c.isalpha())
        alpha_ratio = alpha_count / max(len(text), 1)
        
        has_decimal = '.' in text
        has_minus = '-' in text or '−' in text
        has_scientific = 'e' in text.lower()
        
        # Text length z-score
        text_len_zscore = 0
        if text_stats['mean_len'] > 0:
            text_len_zscore = (len(text) - text_stats['mean_len']) / (text_stats['mean_len'] + 1e-6)
        
        # Box alignment
        box_alignment_score = 0.0
        if boxes:
            box_x_centers = [(b['xyxy'][0] + b['xyxy'][2])/2 for b in boxes]
            box_y_centers = [(b['xyxy'][1] + b['xyxy'][3])/2 for b in boxes]
            
            if box_x_centers:
                min_x_dist = min(abs(cx - bx) for bx in box_x_centers)
                if min_x_dist < img_width * 0.15:
                    box_alignment_score += 0.5
            
            if box_y_centers:
                min_y_dist = min(abs(cy - by) for by in box_y_centers)
                if min_y_dist < img_height * 0.15:
                    box_alignment_score += 0.5
        
        # Compile features
        feat = {
            'label': label,
            'text': text,
            # Position
            'nx': nx, 'ny': ny,
            'cx': cx, 'cy': cy,
            'dist_left': dist_left,
            'dist_right': dist_right,
            'dist_top': dist_top,
            'dist_bottom': dist_bottom,
            'min_edge_dist': min_edge_dist,
            'region': region,
            # Size
            'width': width,
            'height': height,
            'rel_w': rel_w,
            'rel_h': rel_h,
            'aspect': aspect,
            'width_zscore': width_zscore,
            'height_zscore': height_zscore,
            'width_percentile': width_percentile,
            'height_percentile': height_percentile,
            'is_small': is_small,
            # Content
            'is_numeric': is_num,
            'digit_ratio': digit_ratio,
            'alpha_ratio': alpha_ratio,
            'has_decimal': has_decimal,
            'has_minus': has_minus,
            'has_scientific': has_scientific,
            'text_len_zscore': text_len_zscore,
            # Context
            'box_alignment_score': box_alignment_score,
        }
        
        features.append(feat)
    
    return features, size_stats, text_stats
```

---

### 2.2 Implement Constraint-Based Post-Processing

```python
def _apply_constraints(self, classified, features):
    """
    Apply domain-specific constraints to refine classifications.
    
    Constraints:
    1. Scale labels should be on axis edges (min_edge_dist < 0.2)
    2. Tick labels should NOT be on edges (min_edge_dist > 0.1)
    3. Tick labels should align with box elements
    4. Consistent formatting within each type
    """
    
    scale_labels = classified.get('scale_label', [])
    tick_labels = classified.get('tick_label', [])
    
    # Map labels to features
    label_to_feat = {f['label']['xyxy']: f for f in features}
    
    refined_scale = []
    refined_tick = []
    ambiguous = []
    
    # Check scale labels
    for label in scale_labels:
        xyxy = tuple(label['xyxy'])
        feat = label_to_feat.get(xyxy)
        if feat:
            if feat['min_edge_dist'] < 0.2:  # Actually on edge
                refined_scale.append(label)
            else:  # Violates constraint
                ambiguous.append((feat, label, 'scale'))
        else:
            refined_scale.append(label)  # Keep if feature not found
    
    # Check tick labels
    for label in tick_labels:
        xyxy = tuple(label['xyxy'])
        feat = label_to_feat.get(xyxy)
        if feat:
            if feat['min_edge_dist'] > 0.1 and feat['box_alignment_score'] > 0.2:
                refined_tick.append(label)
            else:  # Violates constraint
                ambiguous.append((feat, label, 'tick'))
        else:
            refined_tick.append(label)
    
    # Resolve ambiguous cases
    for feat, label, original_type in ambiguous:
        if original_type == 'scale':
            if feat['min_edge_dist'] >= 0.2:
                # Should be tick instead
                refined_tick.append(label)
            else:
                refined_scale.append(label)
        else:  # was tick
            if feat['min_edge_dist'] < 0.1:
                # Should be scale instead
                refined_scale.append(label)
            else:
                refined_tick.append(label)
    
    return {
        'scale_label': refined_scale,
        'tick_label': refined_tick,
        'axis_title': classified.get('axis_title', [])
    }
```

---

## Section 3: Testing and Validation

### 3.1 Unit Tests for New Features

```python
def test_robust_numeric_detection():
    """Test numeric detection with various formats."""
    test_cases = [
        ("123", True),
        ("12.34", True),
        ("-5", True),
        ("1.23e-4", True),
        ("0-10", True),  # Range
        ("50%", True),   # With unit
        ("100ms", True), # With unit
        ("±5", True),    # Special char
        ("Category", False),
        ("", False),
    ]
    
    for text, expected in test_cases:
        result = is_numeric_robust(text)
        assert result == expected, f"Failed for '{text}': got {result}, expected {expected}"
    
    print("✓ All numeric detection tests passed")

def test_distance_features():
    """Test continuous distance feature extraction."""
    # Mock label at (100, 200) in 1000x1000 image
    nx, ny = 100/1000, 200/1000  # 0.1, 0.2
    
    dist_left = nx
    dist_right = 1.0 - nx
    min_edge_dist = min(dist_left, dist_right, ny, 1.0 - ny)
    
    assert abs(dist_left - 0.1) < 0.001
    assert abs(dist_right - 0.9) < 0.001
    assert min_edge_dist == 0.1  # Close to left edge
    
    print("✓ All distance feature tests passed")

def test_adaptive_threshold():
    """Test adaptive threshold computation."""
    # Ambiguous scores (close to each other)
    scores_ambiguous = [
        {'scale': 2.0, 'tick': 1.8, 'title': 0.5},
        {'scale': 2.1, 'tick': 1.9, 'title': 0.4},
    ]
    
    threshold = compute_adaptive_threshold(scores_ambiguous, base_threshold=2.2)
    assert threshold > 2.2, "Ambiguous cases should have higher threshold"
    
    # Clear scores (well separated)
    scores_clear = [
        {'scale': 3.5, 'tick': 1.2, 'title': 0.3},
        {'scale': 3.8, 'tick': 1.1, 'title': 0.2},
    ]
    
    threshold = compute_adaptive_threshold(scores_clear, base_threshold=2.2)
    assert threshold < 2.2, "Clear cases should have lower threshold"
    
    print("✓ All adaptive threshold tests passed")
```

---

## Section 4: Integration Checklist

- [ ] Fix is_numeric bug
- [ ] Implement is_numeric_robust()
- [ ] Replace binary edge detection with continuous distance
- [ ] Add adaptive thresholding
- [ ] Implement enhanced fallback logic
- [ ] Extract enhanced features
- [ ] Implement constraint validation
- [ ] Add unit tests
- [ ] Test on validation dataset
- [ ] Measure accuracy improvement
- [ ] Document all changes
- [ ] Deploy to production

---

## Section 5: Performance Monitoring

### Metrics to Track

```python
def compute_metrics(predictions, ground_truth):
    """Compute comprehensive metrics."""
    
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    # Convert to binary labels
    pred_labels = [1 if p == 'scale_label' else 0 for p in predictions]
    true_labels = [1 if t == 'scale_label' else 0 for t in ground_truth]
    
    # Precision, Recall, F1
    precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels)
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, pred_labels)
    
    # Margin analysis
    margins = []  # Track score margins
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'avg_margin': np.mean(margins),
        'ambiguous_count': sum(1 for m in margins if m < 0.5),
    }
```

---

## Conclusion

This implementation guide provides step-by-step instructions for improving the classification system. Start with Priority 1 (critical fixes and quick wins), move to Priority 2 (robust features and constraints), and then optionally implement advanced techniques. Each section includes working code that can be directly integrated into the existing system.
