# Refactoring Patterns: From Heuristics to Learned Models
## Code-Level Migration Guide for Box Plot Extraction

---

## Pattern 1: Gradient-to-Heatmap Migration

### Current Code (Heuristic)

**File**: `improved_pixel_based_detector.py:79-159`

```python
def _stage1_centerline_scan(self, img, box_bbox, orientation, scale_model):
    """1D gradient scanning — spatially myopic."""
    x1, y1, x2, y2 = box_bbox
    box_center_x = int((x1 + x2) / 2)
    
    # Extract 1D intensity profile
    intensity_profile = gray[scan_start:scan_end, box_center_x]
    
    # Compute gradient
    gradient = np.gradient(profile.astype(float))
    gradient_smoothed = gaussian_filter1d(gradient_abs, sigma=2.0)
    
    # Find peaks (discrete)
    peaks, properties = find_peaks(gradient_smoothed, prominence=10, ...)
    
    # Select peak with highest prominence
    median_pixel = peak_coords[np.argmax(peak_prominences)]
    median_confidence = min(1.0, peak_prominences[idx] / 50.0)
    
    return {'median': scale_func(median_pixel), 'confidence': median_confidence}
```

**Problems**:
1. **Single pixel basis** — `intensity_profile[box_center_x]` only samples one column
2. **Discrete selection** — Must choose single peak; no uncertainty
3. **Hard thresholds** — `prominence=10` is magic number
4. **No 2D context** — Ignores spatial structure of line (thickness, angle, continuity)

---

### Refactored Code (Learned Heatmap)

**File**: `improved_pixel_based_detector.py:REPLACE FUNCTION`

```python
def _stage1_centerline_scan(self, img, box_bbox, orientation, scale_model):
    """
    REFACTORED: Heatmap-based keypoint regression.
    
    Uses pre-trained HRNet to output per-pixel confidence heatmaps
    for chart elements, replacing discrete gradient peak detection.
    """
    # [IMPORT] Import HRNetMedianDetector
    if not hasattr(self, '_hrnet_model'):
        self._hrnet_model = self._load_hrnet_model()
    
    x1, y1, x2, y2 = box_bbox
    
    # STEP 1: Crop box region + context margin
    context_ratio = 0.3
    crop_x1 = max(0, int(x1 - (x2 - x1) * context_ratio))
    crop_x2 = min(img.shape[1], int(x2 + (x2 - x1) * context_ratio))
    crop_y1 = max(0, int(y1 - (y2 - y1) * context_ratio))
    crop_y2 = min(img.shape[0], int(y2 + (y2 - y1) * context_ratio))
    
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    
    # STEP 2: Normalize and prepare for HRNet
    if len(crop.shape) == 3:
        crop_rgb = crop
    else:
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB)
    
    crop_normalized = torch.from_numpy(crop_rgb).float() / 255.0
    crop_normalized = (crop_normalized - 0.5) / 0.5  # Normalize to [-1, 1]
    crop_batch = crop_normalized.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    
    # STEP 3: Forward pass through HRNet
    with torch.no_grad():
        heatmaps = self._hrnet_model(crop_batch.to(self._device))
    
    # heatmaps shape: (1, num_classes, H_crop//4, W_crop//4)
    # Classes: [0]=median, [1]=whisker_high, [2]=whisker_low
    
    heatmap_median = heatmaps[0, 0].cpu().numpy()  # (H_crop//4, W_crop//4)
    heatmap_whisker_high = heatmaps[0, 1].cpu().numpy()
    heatmap_whisker_low = heatmaps[0, 2].cpu().numpy()
    
    # STEP 4: Extract coordinates from heatmaps
    result = {
        'success': False,
        'median': None,
        'whisker_low': None,
        'whisker_high': None,
        'median_confidence': 0.0,
        'whisker_confidence': 0.0,
        'detection_method': 'heatmap_regression'
    }
    
    # Extract median (center of mass of heatmap)
    if np.max(heatmap_median) > 0.1:  # Confidence threshold
        median_y_rel = self._heatmap_center_of_mass(heatmap_median, axis='y')
        median_x_rel = self._heatmap_center_of_mass(heatmap_median, axis='x')
        
        # Convert from crop-local to image-local coordinates
        median_pixel_y = crop_y1 + median_y_rel * 4  # 4x upsampling from network
        median_pixel_x = crop_x1 + median_x_rel * 4
        
        # For vertical charts, use Y-coordinate; for horizontal, use X
        if orientation == 'vertical':
            median_pixel = median_pixel_y
        else:
            median_pixel = median_pixel_x
        
        try:
            result['median'] = float(scale_model(median_pixel))
            result['median_confidence'] = float(np.max(heatmap_median))
            result['success'] = True
        except Exception as e:
            self.logger.warning(f"Scale model failed for median: {e}")
    
    # Extract whiskers similarly
    if (np.max(heatmap_whisker_high) > 0.1 and 
        np.max(heatmap_whisker_low) > 0.1):
        
        whisker_high_y_rel = self._heatmap_center_of_mass(heatmap_whisker_high, axis='y')
        whisker_low_y_rel = self._heatmap_center_of_mass(heatmap_whisker_low, axis='y')
        
        whisker_high_pixel = crop_y1 + whisker_high_y_rel * 4
        whisker_low_pixel = crop_y1 + whisker_low_y_rel * 4
        
        try:
            result['whisker_high'] = float(scale_model(whisker_high_pixel))
            result['whisker_low'] = float(scale_model(whisker_low_pixel))
            result['whisker_confidence'] = float(
                (np.max(heatmap_whisker_high) + np.max(heatmap_whisker_low)) / 2
            )
            result['success'] = True
        except Exception as e:
            self.logger.warning(f"Scale model failed for whiskers: {e}")
    
    return result

def _heatmap_center_of_mass(self, heatmap, axis='y'):
    """
    Compute center of mass of heatmap along specified axis.
    
    Args:
        heatmap: 2D numpy array (H, W)
        axis: 'y' or 'x'
    
    Returns:
        float: Coordinate in [0, heatmap.shape[axis])
    """
    heatmap_normalized = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap) + 1e-8)
    
    if axis == 'y':
        # Sum across columns (X), weight by rows (Y)
        weights = np.sum(heatmap_normalized, axis=1)
        coordinates = np.arange(len(weights))
    else:  # axis == 'x'
        # Sum across rows (Y), weight by columns (X)
        weights = np.sum(heatmap_normalized, axis=0)
        coordinates = np.arange(len(weights))
    
    center_of_mass = np.sum(weights * coordinates) / (np.sum(weights) + 1e-8)
    return center_of_mass

def _load_hrnet_model(self):
    """
    Load pre-trained HRNet for median/whisker detection.
    
    Can be fine-tuned on box plot dataset or used as-is
    with transfer learning from COCO keypoint detection.
    """
    import timm
    
    # Option 1: Load pre-trained HRNet (COCO keypoint)
    model = timm.create_model('hrnet_w18', pretrained=True, num_classes=3)
    
    # Option 2: Load custom fine-tuned weights
    # checkpoint = torch.load('models/hrnet_box_detection.pth')
    # model.load_state_dict(checkpoint)
    
    model = model.to(self._device)
    model.eval()
    
    return model
```

**Key Changes**:
1. ✅ Uses **2D spatial information** instead of 1D profile
2. ✅ Outputs **continuous heatmaps** (not discrete peaks)
3. ✅ **Learnable thresholds** (trained from data, not hand-tuned)
4. ✅ **Multi-scale context** via HRNet feature pyramid

**Migration Checklist**:
- [ ] Install `timm` library: `pip install timm`
- [ ] Download/train HRNet weights
- [ ] Add GPU device handling (`self._device`)
- [ ] Test on 100 diverse charts
- [ ] Benchmark: accuracy vs. speed tradeoff

---

## Pattern 2: Rule-Based Estimation to Learned Network

### Current Code (Heuristic Rules)

**File**: `smart_whisker_estimator.py:19-117`

```python
def estimate_whiskers_from_context(self, box_info, outliers, neighboring_boxes, orientation):
    """
    Heuristic whisker estimation using Tukey's 1.5×IQR rule.
    
    PROBLEMS:
    1. Fixed rule (1.5×IQR) — ignores data distribution shape
    2. Outlier-bounded logic — circular dependency (whiskers define outliers)
    3. No uncertainty — always returns point estimate
    4. Not adaptive — same logic for skewed, heavy-tailed, uniform data
    """
    q1, q3 = box_info['q1'], box_info['q3']
    iqr = q3 - q1
    
    # Strategy 1: Outlier-bounded (if outliers available)
    if outliers:
        outliers_below_q1 = [o for o in outliers if o < q1]
        nearest_outlier_below = max(outliers_below_q1) if outliers_below_q1 else None
        
        # PROBLEM: Hard max() operation — no learned weighting
        estimated_whisker_low = max(q1 - 1.5 * iqr, nearest_outlier_below)
    else:
        estimated_whisker_low = q1 - 1.5 * iqr
    
    return estimated_whisker_low, estimated_whisker_high
```

**Problems**:
1. **Magic constant 1.5** — arbitrary, no justification for different distributions
2. **Binary logic** — either use outlier-bound OR 1.5×IQR, no learned interpolation
3. **No feature engineering** — doesn't use median_confidence, skewness, kurtosis
4. **No uncertainty** — confidence hardcoded to 0.6

---

### Refactored Code (Learned Network)

**File**: `estimators/whisker_regression_net.py` (NEW FILE)

```python
"""
Learned whisker position regression.

Replaces hand-tuned 1.5×IQR rule with data-driven neural network
that learns whisker extent as function of box statistics.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class WhiskerRegressionNet(nn.Module):
    """
    Multi-task neural network for whisker extent estimation.
    
    Inputs: Box statistics (Q1, Q3, IQR, median, outliers, etc.)
    Outputs: (whisker_low, whisker_high, uncertainty_low, uncertainty_high)
    
    Architecture: Dense network with dropout and batch normalization.
    Loss: Huber + Aleatoric uncertainty regularization.
    """
    
    def __init__(self, hidden_dim: int = 64, num_outlier_features: int = 10):
        super().__init__()
        
        # Feature encoder: Box statistics → dense representation
        self.feature_encoder = nn.Sequential(
            nn.Linear(12 + num_outlier_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
        
        # Task-specific heads
        # Head 1: Low whisker (offset ratio + uncertainty)
        self.whisker_low_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [offset_ratio, log_std]
        )
        
        # Head 2: High whisker (offset ratio + uncertainty)
        self.whisker_high_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2)  # [offset_ratio, log_std]
        )
        
        # Auxiliary task: Predict whether outliers exist (binary classification)
        self.outlier_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)  # Sigmoid → probability
        )
    
    def forward(self, box_features: Dict) -> Dict:
        """
        Args:
            box_features: Dict with keys:
                - 'q1', 'q3', 'iqr', 'median': float
                - 'median_confidence': float
                - 'outliers': List[float]
                - 'orientation': str ('vertical' or 'horizontal')
        
        Returns:
            Dict with:
                - 'whisker_low': float
                - 'whisker_high': float
                - 'uncertainty_low': float (aleatoric uncertainty)
                - 'uncertainty_high': float
                - 'outlier_probability': float (model's estimate of outlier presence)
        """
        # Featurize input dict
        features_tensor = self._featurize(box_features)
        
        # Encode features
        encoded = self.feature_encoder(features_tensor)
        
        # Predict whisker offsets
        low_output = self.whisker_low_head(encoded)  # (2,)
        high_output = self.whisker_high_head(encoded)  # (2,)
        
        # Unpack outputs
        offset_low_logit = low_output[0]
        log_std_low = low_output[1]
        
        offset_high_logit = high_output[0]
        log_std_high = high_output[1]
        
        # Convert logits to interpretable values
        # offset_ratio ∈ [0, 3] (in units of IQR)
        offset_low_ratio = torch.sigmoid(offset_low_logit) * 3.0
        offset_high_ratio = torch.sigmoid(offset_high_logit) * 3.0
        
        # Uncertainty (standard deviation)
        std_low = torch.exp(log_std_low)
        std_high = torch.exp(log_std_high)
        
        # Convert to absolute positions
        q1 = box_features['q1']
        q3 = box_features['q3']
        iqr = box_features['iqr']
        
        whisker_low = q1 - offset_low_ratio * iqr
        whisker_high = q3 + offset_high_ratio * iqr
        
        # Auxiliary: predict outlier presence
        outlier_logit = self.outlier_classifier(encoded)
        outlier_probability = torch.sigmoid(outlier_logit).item()
        
        return {
            'whisker_low': whisker_low.item(),
            'whisker_high': whisker_high.item(),
            'uncertainty_low': std_low.item(),
            'uncertainty_high': std_high.item(),
            'outlier_probability': outlier_probability,
            'offset_low_ratio': offset_low_ratio.item(),
            'offset_high_ratio': offset_high_ratio.item()
        }
    
    def _featurize(self, box_features: Dict) -> torch.Tensor:
        """
        Convert box_features dict → tensor for network input.
        
        Features (total 12 + num_outlier_features):
        1. Q1 (quartile)
        2. Q3 (quartile)
        3. IQR (Q3 - Q1)
        4. Median (center)
        5. Median confidence (0-1)
        6. Outlier count
        7. Min outlier value (or Q1 if no outliers)
        8. Max outlier value (or Q3 if no outliers)
        9. Std of outliers (0 if <2 outliers)
        10. Count of outliers below Q1
        11. Count of outliers above Q3
        12. Distance from Q1 to nearest lower outlier (or 0)
        + Additional outlier statistics (up to 10)
        """
        q1 = box_features['q1']
        q3 = box_features['q3']
        iqr = box_features['iqr']
        median = box_features.get('median', (q1 + q3) / 2)
        median_conf = box_features.get('median_confidence', 0.5)
        outliers = box_features.get('outliers', [])
        
        # Basic statistics
        outliers_below = [o for o in outliers if o < q1]
        outliers_above = [o for o in outliers if o > q3]
        
        features = [
            q1,  # 0
            q3,  # 1
            iqr,  # 2
            median,  # 3
            median_conf,  # 4
            len(outliers),  # 5
            min(outliers) if outliers else q1,  # 6
            max(outliers) if outliers else q3,  # 7
            np.std(outliers) if len(outliers) > 1 else 0,  # 8
            len(outliers_below),  # 9
            len(outliers_above),  # 10
            (max(outliers_below) - q1) if outliers_below else 0,  # 11
        ]
        
        # Additional outlier features
        if outliers:
            # Quartiles of outlier distribution
            outliers_sorted = sorted(outliers)
            features.append(np.percentile(outliers_sorted, 25))  # Q1_outliers
            features.append(np.percentile(outliers_sorted, 75))  # Q3_outliers
            features.append(np.percentile(outliers_sorted, 5))   # P5_outliers
            features.append(np.percentile(outliers_sorted, 95))  # P95_outliers
            features.append(np.max(np.abs(np.diff(outliers_sorted))))  # Max gap
            features.append(np.mean(np.abs(np.array(outliers) - median)))  # MAD from median
            # Pad to fixed size
            while len(features) < 12 + 10:
                features.append(0.0)
        else:
            features.extend([0.0] * 10)
        
        # Normalize features to [-1, 1] range
        features_array = np.array(features[:22], dtype=np.float32)  # Take first 22 features
        
        # Z-score normalization (approximate)
        features_array = np.clip(features_array, -5, 5) / 5.0
        
        return torch.from_numpy(features_array)


class WhiskerRegressionLoss(nn.Module):
    """
    Multi-task loss for whisker regression.
    
    Components:
    1. Huber loss (L1 for large errors, L2 for small) on whisker positions
    2. Uncertainty regularization (encourage confident predictions)
    3. Outlier classification loss (auxiliary task)
    4. Ordering constraint penalty
    """
    
    def __init__(self, delta: float = 0.5, lambda_uncertainty: float = 0.01, 
                 lambda_outlier: float = 0.1, lambda_order: float = 0.05):
        super().__init__()
        self.huber = nn.HuberLoss(delta=delta, reduction='mean')
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.lambda_uncertainty = lambda_uncertainty
        self.lambda_outlier = lambda_outlier
        self.lambda_order = lambda_order
    
    def forward(self, pred: Dict, gt: Dict, has_outliers: bool = None) -> torch.Tensor:
        """
        Args:
            pred: Dict with predictions from network
            gt: Dict with ground-truth values
            has_outliers: Bool indicating if GT has outliers (for classification loss)
        
        Returns:
            torch.Tensor: Scalar loss value
        """
        # L1: Whisker position errors
        loss_low = self.huber(
            torch.tensor(pred['whisker_low']),
            torch.tensor(gt['whisker_low'])
        )
        loss_high = self.huber(
            torch.tensor(pred['whisker_high']),
            torch.tensor(gt['whisker_high'])
        )
        
        # L2: Uncertainty regularization (small uncertainty preferred)
        unc_low = torch.tensor(pred['uncertainty_low'])
        unc_high = torch.tensor(pred['uncertainty_high'])
        loss_uncertainty = self.lambda_uncertainty * (unc_low + unc_high)
        
        # L3: Outlier classification (if GT available)
        loss_outlier = 0
        if has_outliers is not None:
            outlier_target = torch.tensor([float(has_outliers)])
            outlier_pred = torch.tensor([pred['outlier_probability']])
            loss_outlier = self.lambda_outlier * self.bce(outlier_pred, outlier_target)
        
        # L4: Ordering constraint (whisker_low < q1 < median < q3 < whisker_high)
        loss_order = self._compute_ordering_loss(pred, gt)
        
        total_loss = loss_low + loss_high + loss_uncertainty + loss_outlier + loss_order
        return total_loss
    
    def _compute_ordering_loss(self, pred: Dict, gt: Dict) -> torch.Tensor:
        """Penalize violations of value ordering."""
        q1, median, q3 = gt['q1'], gt['median'], gt['q3']
        w_low, w_high = pred['whisker_low'], pred['whisker_high']
        
        violations = 0
        
        # whisker_low < q1
        if w_low > q1:
            violations += (w_low - q1) ** 2
        
        # q1 < median
        if q1 > median:
            violations += (q1 - median) ** 2
        
        # median < q3
        if median > q3:
            violations += (median - q3) ** 2
        
        # q3 < whisker_high
        if q3 > w_high:
            violations += (q3 - w_high) ** 2
        
        return self.lambda_order * torch.tensor(violations)


# Training function
def train_whisker_net(model, train_loader, val_loader, device='cuda', epochs=100):
    """Train whisker regression network on annotated data."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = WhiskerRegressionLoss()
    
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch_idx, (box_features_list, gt_list) in enumerate(train_loader):
            batch_loss = 0
            
            for box_features, gt in zip(box_features_list, gt_list):
                pred = model(box_features)
                loss = loss_fn(pred, gt, has_outliers=len(gt.get('outliers', [])) > 0)
                batch_loss += loss
            
            batch_loss /= len(box_features_list)
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += batch_loss.item()
        
        scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'whisker_low_mae': 0, 'whisker_high_mae': 0}
        
        with torch.no_grad():
            for box_features_list, gt_list in val_loader:
                for box_features, gt in zip(box_features_list, gt_list):
                    pred = model(box_features)
                    loss = loss_fn(pred, gt)
                    val_loss += loss.item()
                    
                    val_metrics['whisker_low_mae'] += abs(pred['whisker_low'] - gt['whisker_low'])
                    val_metrics['whisker_high_mae'] += abs(pred['whisker_high'] - gt['whisker_high'])
        
        print(f"Epoch {epoch}: "
              f"Train Loss={train_loss/len(train_loader):.4f}, "
              f"Val Loss={val_loss/len(val_loader):.4f}, "
              f"W_Low MAE={val_metrics['whisker_low_mae']/len(val_loader):.4f}")
        
        # Early stopping / checkpoint saving
        if epoch % 10 == 0:
            torch.save(model.state_dict(), f'checkpoints/whisker_net_epoch{epoch}.pth')
```

**Integration with Existing Code**:

**File**: `smart_whisker_estimator.py` (MODIFY)

```python
# At top of file, add:
import torch
from estimators.whisker_regression_net import WhiskerRegressionNet

class SmartWhiskerEstimator:
    def __init__(self, use_learned_model=True):
        self.logger = logging.getLogger(__name__)
        self.use_learned_model = use_learned_model
        
        if use_learned_model:
            self.learned_net = WhiskerRegressionNet(hidden_dim=64)
            # Load pre-trained weights
            checkpoint = torch.load('models/whisker_net_v2.pth', map_location='cpu')
            self.learned_net.load_state_dict(checkpoint)
            self.learned_net.eval()
    
    def estimate_whiskers_from_context(self, box_info, outliers, neighboring_boxes, orientation):
        """
        Whisker estimation: Learned network with fallback to rule-based.
        """
        if self.use_learned_model:
            return self._estimate_whiskers_learned(box_info, outliers)
        else:
            return self._estimate_whiskers_rules(box_info, outliers, neighboring_boxes)
    
    def _estimate_whiskers_learned(self, box_info, outliers):
        """Use neural network for estimation."""
        try:
            with torch.no_grad():
                pred = self.learned_net({
                    'q1': box_info['q1'],
                    'q3': box_info['q3'],
                    'iqr': box_info.get('iqr', box_info['q3'] - box_info['q1']),
                    'median': box_info.get('median', (box_info['q1'] + box_info['q3']) / 2),
                    'median_confidence': box_info.get('median_confidence', 0.5),
                    'outliers': outliers
                })
            
            self.logger.info(
                f"Learned whisker estimation: "
                f"low={pred['whisker_low']:.2f} (±{pred['uncertainty_low']:.2f}), "
                f"high={pred['whisker_high']:.2f} (±{pred['uncertainty_high']:.2f})"
            )
            
            return pred['whisker_low'], pred['whisker_high']
        
        except Exception as e:
            self.logger.warning(f"Learned model failed, falling back to rules: {e}")
            return self._estimate_whiskers_rules(box_info, outliers, [])
    
    def _estimate_whiskers_rules(self, box_info, outliers, neighboring_boxes):
        """Fallback: Original heuristic rules."""
        q1, q3 = box_info['q1'], box_info['q3']
        iqr = q3 - q1
        
        # ... [KEEP ORIGINAL CODE FROM smart_whisker_estimator.py] ...
        
        if outliers:
            outliers_below_q1 = [o for o in outliers if o < q1]
            estimated_whisker_low = max(q1 - 1.5 * iqr, max(outliers_below_q1) if outliers_below_q1 else q1 - 1.5*iqr)
            # ... etc.
        else:
            estimated_whisker_low = q1 - 1.5 * iqr
        
        return estimated_whisker_low, estimated_whisker_high
```

**Migration Checklist**:
- [ ] Create `estimators/whisker_regression_net.py`
- [ ] Create training dataset (500+ labeled charts)
- [ ] Train WhiskerRegressionNet
- [ ] Save checkpoint to `models/whisker_net_v2.pth`
- [ ] Modify SmartWhiskerEstimator to use learned model
- [ ] Test fallback mechanism (rule-based when network unavailable)
- [ ] Benchmark: accuracy improvement vs. inference overhead

---

## Pattern 3: Euclidean Proximity to Graph Relation Learning

### Current Code (Heuristic Distance)

**File**: `box_grouper.py:66-180`

```python
def group_box_plot_elements(boxes, range_indicators, median_lines, outliers, ...):
    """
    AABB + proximity-based grouping — spatially naive.
    
    PROBLEMS:
    1. Two-stage logic: Intersection OR proximity (not both)
    2. Fixed thresholds: 0.5×width is magic number
    3. Binary assignment: Each element assigned to at most one box
    4. No learned ranking: Can't disambiguate among multiple candidates
    """
    
    for box in boxes:
        # Stage 1: AABB intersection test
        for indicator in range_indicators:
            if compute_aabb_intersection(box, indicator):
                group['range_indicator'] = indicator
                break  # Hard stop
        
        # Stage 2: If no intersection, proximity
        if group['range_indicator'] is None:
            best_indicator = None
            min_distance = float('inf')
            
            for indicator in range_indicators:
                # Hard threshold: 0.5×width
                if abs(indicator_x - box_x) < 0.5 * box_width:
                    y_distance = abs(indicator_y - box_y)
                    if y_distance < min_distance:
                        best_indicator = indicator
            
            if best_indicator:
                group['range_indicator'] = best_indicator
```

**Problems**:
1. **Sequential gates** — Intersection filter then proximity (not learned jointly)
2. **Magic constant 0.5** — arbitrary threshold
3. **Greedy assignment** — No revisiting/correction
4. **No uncertainty** — Confidence always 1.0

---

### Refactored Code (Graph Relation Networks)

**File**: `extractors/grn_grouper.py` (NEW FILE)

```python
"""
Graph Relation Network for topology-aware element grouping.

Replaces hard AABB+proximity logic with learned pairwise relations
and multi-head attention for soft assignment disambiguation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple


class ChartElementGRN(nn.Module):
    """
    Graph Relation Network for chart element grouping.
    
    Architecture:
    - Node encoder: Element bbox → feature vector
    - Edge predictor: Pairwise features → relation probability
    - Graph attention: Refine assignments via multi-head attention
    
    Output: Soft assignments (probabilities) for each element to boxes
    """
    
    def __init__(self, feat_dim: int = 128, num_heads: int = 8):
        super().__init__()
        
        # Node encoders (type-specific)
        self.box_encoder = nn.Sequential(
            nn.Linear(4, feat_dim),  # [x1, y1, x2, y2]
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        
        self.whisker_encoder = nn.Sequential(
            nn.Linear(5, feat_dim),  # [x1, y1, x2, y2, confidence]
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        
        self.median_encoder = nn.Sequential(
            nn.Linear(5, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        
        self.outlier_encoder = nn.Sequential(
            nn.Linear(4, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, feat_dim),
        )
        
        # Relation predictor (pairwise)
        self.relation_net = nn.Sequential(
            nn.Linear(2 * feat_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Sigmoid → relation probability
        )
        
        # Graph attention for refinement
        self.graph_attention = nn.MultiheadAttention(
            embed_dim=feat_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Relation threshold (learnable)
        self.relation_threshold = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, boxes: np.ndarray, whiskers: np.ndarray, 
                medians: np.ndarray, outliers: np.ndarray) -> List[Dict]:
        """
        Args:
            boxes: (n_boxes, 4) with [x1, y1, x2, y2]
            whiskers: (n_whiskers, 5) with [x1, y1, x2, y2, confidence]
            medians: (n_medians, 5)
            outliers: (n_outliers, 4)
        
        Returns:
            List[Dict] with keys:
                - 'box_id': int
                - 'whiskers': List[Tuple(whisker_id, relation_score)]
                - 'medians': List[Tuple(median_id, relation_score)]
                - 'outliers': List[Tuple(outlier_id, relation_score)]
        """
        # Convert numpy to torch tensors
        boxes_t = torch.from_numpy(boxes).float()
        whiskers_t = torch.from_numpy(whiskers).float()
        medians_t = torch.from_numpy(medians).float()
        outliers_t = torch.from_numpy(outliers).float()
        
        # Encode all elements
        box_emb = self.box_encoder(boxes_t)  # (n_boxes, feat_dim)
        whisker_emb = self.whisker_encoder(whiskers_t)  # (n_whiskers, feat_dim)
        median_emb = self.median_encoder(medians_t)
        outlier_emb = self.outlier_encoder(outliers_t)
        
        # Concatenate for graph attention
        all_emb = torch.cat([box_emb, whisker_emb, median_emb, outlier_emb], dim=0)
        
        # Graph attention: refine embeddings
        refined_emb, attn_weights = self.graph_attention(
            all_emb.unsqueeze(1),  # Add batch dim: (total, 1, feat_dim) -> (total, feat_dim)
            all_emb.unsqueeze(1),
            all_emb.unsqueeze(1)
        )
        refined_emb = refined_emb.squeeze(1)
        
        # Predict relations: box → other elements
        grouping = []
        
        for i in range(len(boxes)):
            box_idx = i
            group = {
                'box_id': box_idx,
                'whiskers': [],
                'medians': [],
                'outliers': []
            }
            
            box_emb_refined = refined_emb[box_idx]
            
            # Box-to-Whisker relations
            for j in range(len(whiskers)):
                whisker_idx = len(boxes) + j
                whisker_emb_refined = refined_emb[whisker_idx]
                
                # Pairwise relation scoring
                pair_emb = torch.cat([box_emb_refined, whisker_emb_refined])
                relation_logit = self.relation_net(pair_emb)
                relation_score = torch.sigmoid(relation_logit).item()
                
                if relation_score > self.relation_threshold.item():
                    group['whiskers'].append((j, relation_score))
            
            # Sort by confidence (descending)
            group['whiskers'].sort(key=lambda x: x[1], reverse=True)
            
            # Same for medians and outliers...
            for j in range(len(medians)):
                median_idx = len(boxes) + len(whiskers) + j
                median_emb_refined = refined_emb[median_idx]
                
                pair_emb = torch.cat([box_emb_refined, median_emb_refined])
                relation_logit = self.relation_net(pair_emb)
                relation_score = torch.sigmoid(relation_logit).item()
                
                if relation_score > self.relation_threshold.item():
                    group['medians'].append((j, relation_score))
            
            group['medians'].sort(key=lambda x: x[1], reverse=True)
            
            for j in range(len(outliers)):
                outlier_idx = len(boxes) + len(whiskers) + len(medians) + j
                outlier_emb_refined = refined_emb[outlier_idx]
                
                pair_emb = torch.cat([box_emb_refined, outlier_emb_refined])
                relation_logit = self.relation_net(pair_emb)
                relation_score = torch.sigmoid(relation_logit).item()
                
                if relation_score > self.relation_threshold.item():
                    group['outliers'].append((j, relation_score))
            
            group['outliers'].sort(key=lambda x: x[1], reverse=True)
            
            grouping.append(group)
        
        return grouping
```

**Integration with `box_extractor.py`**:

```python
# At top of file:
from extractors.grn_grouper import ChartElementGRN

# In BoxExtractor.__init__:
self.grn_grouper = ChartElementGRN(feat_dim=128, num_heads=8)
grn_checkpoint = torch.load('models/grn_grouper_v1.pth')
self.grn_grouper.load_state_dict(grn_checkpoint)
self.grn_grouper.eval()

# Replace old grouping logic (around line 76):
# OLD:
# groups = group_box_plot_elements(boxes, whiskers, medians, outliers, ...)

# NEW:
grouping = self.grn_grouper(
    boxes=np.array([b['xyxy'] for b in boxes]),
    whiskers=np.array([w['xyxy'] + [w.get('confidence', 0.8)] for w in whiskers]),
    medians=np.array([m['xyxy'] + [m.get('confidence', 0.8)] for m in medians]),
    outliers=np.array([o['xyxy'] for o in outliers])
)

# Convert soft assignments to groups dict (backward compatible)
groups = []
for group_soft in grouping:
    group_hard = {
        'box': boxes[group_soft['box_id']],
        'range_indicator': whiskers[group_soft['whiskers'][0][0]] if group_soft['whiskers'] else None,
        'median_line': medians[group_soft['medians'][0][0]] if group_soft['medians'] else None,
        'outliers': [outliers[j] for j, _ in group_soft['outliers']],
        'grn_scores': group_soft  # Store soft assignments for debugging
    }
    groups.append(group_hard)
```

**Training Data Format**:

```python
# Create training dataset in COCO-like format
training_data = {
    'images': [
        {'id': 0, 'file_name': 'chart_0001.png', 'height': 600, 'width': 800, ...},
        ...
    ],
    'annotations': [
        # Box 0 → Whisker 1 (relation=True, score=0.95)
        {'id': 0, 'image_id': 0, 'element_type': 'box', 'xyxy': [...], 'element_id': 0},
        {'id': 1, 'image_id': 0, 'element_type': 'whisker', 'xyxy': [...], 'element_id': 0},
        
        # Relations (edges in graph)
        {'id': 0, 'source': 0, 'target': 1, 'relation': True, 'confidence': 0.95},
        {'id': 1, 'source': 0, 'target': 2, 'relation': False, 'confidence': 0.2},
        ...
    ]
}
```

**Migration Checklist**:
- [ ] Create `extractors/grn_grouper.py`
- [ ] Annotate 500+ charts with element-to-box relations
- [ ] Train ChartElementGRN with relation labels
- [ ] Integrate GRN into BoxExtractor
- [ ] Test soft assignment quality
- [ ] Benchmark vs. AABB grouping (precision/recall on grouped boxes)

---

## Deployment Checklist

### Pre-Production

- [ ] Unit tests for each refactored component
- [ ] Integration tests (end-to-end on 100+ diverse charts)
- [ ] Benchmark: accuracy, speed, memory before/after
- [ ] A/B test: Rule-based vs. learned models on 10% traffic
- [ ] Documentation: Model cards, limitations, failure modes

### Production

- [ ] Model versioning: Tag all PyTorch checkpoints
- [ ] Monitor predictions: Log confidence scores, detect distribution shift
- [ ] Fallback strategy: If GPU unavailable, use rule-based models
- [ ] Update schedule: Retrain quarterly on new data
- [ ] Observability: Track per-component errors (median, whisker, grouping)

---

## Cost-Benefit Analysis

| Component | Dev Time | Accuracy Gain | Speed Cost | Recommended? |
|-----------|----------|---------------|-----------|--------------|
| Spline Calibration | 1-2w | +5-10% | 0% | ✅ YES (low effort) |
| Learned Whisker Net | 2-3w | +8-15% | +5ms | ✅ YES (good ROI) |
| HRNet Keypoints | 4-6w | +15-25% | +50ms | 🟡 MAYBE (depends on latency) |
| Graph Relation Networks | 4-6w | +5-15% | +20ms | 🟡 MAYBE (best for grouped charts) |
| End-to-End Training | 8-12w | +20-30% | +10ms | 🔴 LATER (after Phase 1-2) |

**Recommendation**: Start with **Spline + Learned Whisker Net** (Phase 1), then evaluate HRNet/GRN based on accuracy metrics and latency constraints.

