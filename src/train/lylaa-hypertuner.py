import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from typing import Dict, List, Tuple, Any, Optional
import json
from pathlib import Path
import logging
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class LYLAAHypertuningResult:
    chart_type: str
    optimal_parameters: Dict
    best_loss: float
    best_accuracy: float
    best_f1_score: float
    final_loss: float
    final_accuracy: float
    epochs_trained: int
    total_samples: int
    optimization_method: str
    class_distribution: Dict

class TypeSpecificLYLAAHypertuner(nn.Module):
    """
    Enhanced LYLAA hypertuner with type-specific parameters and advanced optimization
    """
    
    def __init__(self, 
                 chart_type: str = 'bar',
                 device: str = 'cpu',
                 learning_rate: float = 0.01,
                 use_focal_loss: bool = True,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0):
        super().__init__()
        
        self.chart_type = chart_type
        self.device = device
        self.use_focal_loss = use_focal_loss
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        
        # Type-specific parameter initialization
        self.params = self._initialize_type_specific_params(chart_type)
        
        # Parameter constraints
        self.param_constraints = self._get_param_constraints()
        
        # Move to device
        self.to(device)
        
        # Initialize optimizer with weight decay (L2 regularization)
        self.optimizer = optim.Adam(
            self.parameters(),
            lr=learning_rate,
            weight_decay=1e-4  # L2 regularization
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True,
            min_lr=1e-6
        )
        
        # Training history
        self.history = {
            'losses': [],
            'accuracies': [],
            'f1_scores': [],
            'learning_rates': [],
            'parameters': [],
            'parameter_gradients': []
        }
        
        logger.info(f"TypeSpecificLYLAAHypertuner initialized for {chart_type} on {device}")
        logger.info(f"Using focal loss: {use_focal_loss}, Parameters: {len(self.params)}")
    
    def _initialize_type_specific_params(self, chart_type: str) -> nn.ParameterDict:
        """
        Initialize parameters with type-specific defaults based on empirical analysis
        """
        params = nn.ParameterDict()
        
        # Common parameters for all types
        common_params = {
            'sigma_x': 0.09,
            'sigma_y': 0.09,
            'size_constraint_primary': 3.0,
            'size_constraint_secondary': 2.5,
            'aspect_ratio_weight': 2.5,
            'distance_weight': 2.0,
            'classification_threshold': 1.5,
            'size_threshold_width': 0.08,
            'size_threshold_height': 0.04,
            'aspect_ratio_min': 0.5,
            'aspect_ratio_max': 3.5
        }
        
        # Type-specific parameters
        if chart_type == 'bar':
            type_params = {
                'left_y_axis_weight': 6.0,
                'right_y_axis_weight': 5.0,
                'bottom_x_axis_weight': 5.5,
                'top_title_weight': 4.0,
                'center_data_weight': 2.0,
                'position_weight_primary': 5.5,
                'position_weight_secondary': 5.0,
                'context_weight_primary': 4.5,
                'context_weight_secondary': 5.5,
                'ocr_numeric_boost': 2.5,
                'ocr_numeric_penalty': 1.0,
                'bar_alignment_weight': 5.0,
                'bar_spacing_multiplier': 1.5
            }
        elif chart_type == 'line':
            type_params = {
                'left_y_axis_weight': 6.5,
                'right_y_axis_weight': 5.5,
                'bottom_x_axis_weight': 6.0,
                'top_title_weight': 3.5,
                'center_data_weight': 1.5,
                'position_weight_primary': 6.0,
                'position_weight_secondary': 4.5,
                'context_weight_primary': 5.0,
                'context_weight_secondary': 4.5,
                'ocr_numeric_boost': 3.0,
                'ocr_numeric_penalty': 0.8,
                'line_proximity_weight': 4.5,
                'trend_fit_weight': 4.0
            }
        elif chart_type == 'scatter':
            type_params = {
                'left_y_axis_weight': 7.0,
                'right_y_axis_weight': 6.0,
                'bottom_x_axis_weight': 6.5,
                'top_title_weight': 4.0,
                'center_data_weight': 1.0,
                'position_weight_primary': 6.5,
                'position_weight_secondary': 4.0,
                'context_weight_primary': 5.0,
                'context_weight_secondary': 4.0,
                'ocr_numeric_boost': 3.5,
                'ocr_numeric_penalty': 0.5,
                'point_cloud_proximity_weight': 5.0,
                'dual_axis_penalty': 0.7
            }
        elif chart_type == 'box':
            type_params = {
                'left_y_axis_weight': 6.0,
                'right_y_axis_weight': 5.0,
                'bottom_x_axis_weight': 5.5,
                'top_title_weight': 4.0,
                'center_data_weight': 2.0,
                'position_weight_primary': 5.5,
                'position_weight_secondary': 5.5,
                'context_weight_primary': 4.5,
                'context_weight_secondary': 5.0,
                'ocr_numeric_boost': 2.5,
                'ocr_numeric_penalty': 1.0,
                'whisker_dist_weight': 4.0,
                'box_spacing_weight': 4.0
            }
        elif chart_type == 'histogram':
            type_params = {
                'left_y_axis_weight': 6.5,
                'right_y_axis_weight': 5.0,
                'bottom_x_axis_weight': 6.0,
                'top_title_weight': 4.0,
                'center_data_weight': 1.5,
                'position_weight_primary': 5.5,
                'position_weight_secondary': 5.0,
                'context_weight_primary': 4.5,
                'context_weight_secondary': 5.0,
                'ocr_numeric_boost': 2.8,
                'ocr_numeric_penalty': 0.8,
                'bin_alignment_weight': 4.5,
                'continuous_scale_weight': 3.5
            }
        else:
            type_params = {}
        
        # Merge common and type-specific
        all_params = {**common_params, **type_params}
        
        # Convert to nn.Parameters
        for name, value in all_params.items():
            params[name] = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        
        return params
    
    def _get_param_constraints(self) -> Dict[str, Tuple[float, float]]:
        """Define parameter constraints for all parameters"""
        return {
            'sigma_x': (0.01, 0.5),
            'sigma_y': (0.01, 0.5),
            'left_y_axis_weight': (0.1, 10.0),
            'right_y_axis_weight': (0.1, 10.0),
            'bottom_x_axis_weight': (0.1, 10.0),
            'top_title_weight': (0.1, 10.0),
            'center_data_weight': (0.1, 10.0),
            'size_constraint_primary': (0.1, 10.0),
            'size_constraint_secondary': (0.1, 10.0),
            'aspect_ratio_weight': (0.1, 10.0),
            'position_weight_primary': (0.1, 10.0),
            'position_weight_secondary': (0.1, 10.0),
            'distance_weight': (0.1, 10.0),
            'context_weight_primary': (0.1, 10.0),
            'context_weight_secondary': (0.1, 10.0),
            'ocr_numeric_boost': (0.1, 10.0),
            'ocr_numeric_penalty': (0.1, 10.0),
            'classification_threshold': (0.1, 5.0),
            'size_threshold_width': (0.01, 0.5),
            'size_threshold_height': (0.01, 0.5),
            'aspect_ratio_min': (0.1, 2.0),
            'aspect_ratio_max': (1.5, 10.0),
            # Type-specific constraints
            'bar_alignment_weight': (0.1, 10.0),
            'bar_spacing_multiplier': (0.5, 3.0),
            'line_proximity_weight': (0.1, 10.0),
            'trend_fit_weight': (0.1, 10.0),
            'point_cloud_proximity_weight': (0.1, 10.0),
            'dual_axis_penalty': (0.1, 2.0),
            'whisker_dist_weight': (0.1, 10.0),
            'box_spacing_weight': (0.1, 10.0),
            'bin_alignment_weight': (0.1, 10.0),
            'continuous_scale_weight': (0.1, 10.0)
        }
    
    def constrain_parameters(self):
        """Apply constraints to parameters"""
        with torch.no_grad():
            for name, param in self.params.items():
                if name in self.param_constraints:
                    min_val, max_val = self.param_constraints[name]
                    param.data.clamp_(min_val, max_val)
    
    def compute_focal_loss(self, pred_tensor: torch.Tensor, gt_tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss to handle class imbalance
        
        Focal Loss = -α(1-pt)^γ * log(pt)
        where pt is the probability of the true class
        """
        # Standard cross-entropy
        ce_loss = nn.functional.cross_entropy(pred_tensor, gt_tensor, reduction='none')
        
        # Get probabilities
        pt = torch.exp(-ce_loss)
        
        # Compute focal loss
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        
        return focal_loss.mean()
    
    def compute_loss(self, predictions: List[torch.Tensor], ground_truth: List[int]) -> torch.Tensor:
        """
        Compute loss with optional focal loss for class imbalance
        """
        pred_tensor = torch.stack(predictions)
        gt_tensor = torch.tensor(ground_truth, device=self.device, dtype=torch.long)
        
        if self.use_focal_loss:
            loss = self.compute_focal_loss(pred_tensor, gt_tensor)
        else:
            loss = nn.functional.cross_entropy(pred_tensor, gt_tensor)
        
        # Add L2 regularization on parameters (already in optimizer, but can add explicit)
        l2_reg = sum(p.pow(2.0).sum() for p in self.parameters())
        loss = loss + 1e-5 * l2_reg
        
        return loss
    
    def compute_metrics(self, predictions: List[torch.Tensor], ground_truth: List[int]) -> Dict[str, float]:
        """
        Compute comprehensive metrics including F1 score
        """
        with torch.no_grad():
            pred_classes = torch.stack([torch.argmax(pred) for pred in predictions]).cpu().numpy()
            gt_array = np.array(ground_truth)
            
            # Overall accuracy
            accuracy = np.mean(pred_classes == gt_array)
            
            # Per-class metrics
            from sklearn.metrics import precision_recall_fscore_support
            precision, recall, f1, support = precision_recall_fscore_support(
                gt_array, pred_classes, average='weighted', zero_division=0
            )
            
            return {
                'accuracy': float(accuracy),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
    
    def differentiable_gaussian_score(self, nx: torch.Tensor, ny: torch.Tensor,
                                     center_x: float, center_y: float) -> torch.Tensor:
        """Differentiable Gaussian kernel score"""
        dx = (nx - center_x) / self.params['sigma_x']
        dy = (ny - center_y) / self.params['sigma_y']
        return torch.exp(-(dx**2 + dy**2) / 2)
    
    def differentiable_region_scores(self, normalized_pos: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute region scores differentiably"""
        nx, ny = normalized_pos[0], normalized_pos[1]
        scores = {}
        
        # Left Y-axis
        left_mask = (nx < 0.20) & (ny > 0.1) & (ny < 0.9)
        scores['left_y_axis'] = torch.where(
            left_mask,
            self.differentiable_gaussian_score(nx, ny, 0.08, 0.5) * self.params['left_y_axis_weight'],
            torch.tensor(0.0, device=self.device)
        )
        
        # Right Y-axis
        right_mask = (nx > 0.80) & (ny > 0.1) & (ny < 0.9)
        scores['right_y_axis'] = torch.where(
            right_mask,
            self.differentiable_gaussian_score(nx, ny, 0.92, 0.5) * self.params['right_y_axis_weight'],
            torch.tensor(0.0, device=self.device)
        )
        
        # Bottom X-axis
        bottom_mask = (nx > 0.15) & (nx < 0.85) & (ny > 0.80)
        scores['bottom_x_axis'] = torch.where(
            bottom_mask,
            self.differentiable_gaussian_score(nx, ny, 0.5, 0.92) * self.params['bottom_x_axis_weight'],
            torch.tensor(0.0, device=self.device)
        )
        
        # Top title
        top_mask = (nx > 0.15) & (nx < 0.85) & (ny < 0.15)
        scores['top_title'] = torch.where(
            top_mask,
            self.differentiable_gaussian_score(nx, ny, 0.5, 0.08) * self.params['top_title_weight'],
            torch.tensor(0.0, device=self.device)
        )
        
        # Center data
        center_mask = (nx > 0.2) & (nx < 0.8) & (ny > 0.2) & (ny < 0.8)
        center_dist = torch.sqrt((nx - 0.5)**2 + (ny - 0.5)**2)
        scores['center_data'] = torch.where(
            center_mask,
            torch.exp(-center_dist**2 / 0.08) * self.params['center_data_weight'],
            torch.tensor(0.0, device=self.device)
        )
        
        return scores
    
    def differentiable_multi_feature_scores(self, features: Dict[str, torch.Tensor],
                                           region_scores: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-feature scores differentiably"""
        rel_width = features['rel_width']
        rel_height = features['rel_height']
        aspect_ratio = features['aspect_ratio']
        nx, ny = features['nx'], features['ny']
        
        scores = {
            'scale_label': torch.tensor(0.0, device=self.device),
            'tick_label': torch.tensor(0.0, device=self.device),
            'axis_title': torch.tensor(0.0, device=self.device)
        }
        
        # Size constraints for scale labels
        size_mask = (rel_width < self.params['size_threshold_width']) & (rel_height < self.params['size_threshold_height'])
        scores['scale_label'] += torch.where(size_mask, self.params['size_constraint_primary'], torch.tensor(0.0, device=self.device))
        
        # Aspect ratio constraints
        aspect_mask = (aspect_ratio > self.params['aspect_ratio_min']) & (aspect_ratio < self.params['aspect_ratio_max'])
        scores['scale_label'] += torch.where(aspect_mask, self.params['aspect_ratio_weight'], torch.tensor(0.0, device=self.device))
        
        # Position-based scoring
        left_right_max = torch.max(region_scores['left_y_axis'], region_scores['right_y_axis'])
        scores['scale_label'] += left_right_max * self.params['position_weight_primary']
        
        # Distance from center
        center_dist = torch.sqrt((nx - 0.5)**2 + (ny - 0.5)**2)
        distance_bonus = torch.where(center_dist > 0.3, (center_dist - 0.3) * self.params['distance_weight'], torch.tensor(0.0, device=self.device))
        scores['scale_label'] += distance_bonus
        
        # Title detection
        title_mask = (aspect_ratio > 4.0) | (aspect_ratio < 0.25)
        scores['axis_title'] += torch.where(title_mask, self.params['context_weight_primary'], torch.tensor(0.0, device=self.device))
        
        # Large size indicates title
        large_size_mask = (rel_width > 0.15) | (rel_height > 0.08)
        scores['axis_title'] += torch.where(large_size_mask, self.params['context_weight_secondary'], torch.tensor(0.0, device=self.device))
        
        # Tick label scoring
        scores['tick_label'] += region_scores['bottom_x_axis'] * self.params['position_weight_secondary']
        
        return scores
    
    def forward(self, label_features: Dict[str, Any]) -> torch.Tensor:
        """Forward pass through the model"""
        # Convert features to tensors
        nx = torch.tensor(label_features['normalized_pos'][0], device=self.device, dtype=torch.float32)
        ny = torch.tensor(label_features['normalized_pos'][1], device=self.device, dtype=torch.float32)
        rel_width = torch.tensor(label_features['relative_size'][0], device=self.device, dtype=torch.float32)
        rel_height = torch.tensor(label_features['relative_size'][1], device=self.device, dtype=torch.float32)
        aspect_ratio = torch.tensor(label_features['aspect_ratio'], device=self.device, dtype=torch.float32)
        
        # Compute region scores
        region_scores = self.differentiable_region_scores(torch.stack([nx, ny]))
        
        # Prepare features
        features = {
            'nx': nx, 'ny': ny,
            'rel_width': rel_width, 'rel_height': rel_height,
            'aspect_ratio': aspect_ratio
        }
        
        # Compute classification scores
        class_scores = self.differentiable_multi_feature_scores(features, region_scores)
        
        # Convert to logits
        logits = torch.stack([
            class_scores['scale_label'],
            class_scores['tick_label'],
            class_scores['axis_title']
        ])
        
        # Threshold-based enhancement
        max_score = torch.max(logits)
        threshold_weight = torch.sigmoid(10 * (max_score - self.params['classification_threshold']))
        enhanced_logits = logits * threshold_weight
        
        # Default boost if below threshold
        default_boost = torch.tensor([2.0, 0.0, 0.0], device=self.device)
        final_logits = enhanced_logits + (1 - threshold_weight) * default_boost
        
        return final_logits
    
    def get_current_params_dict(self) -> Dict[str, float]:
        """Get current parameter values"""
        return {name: param.item() for name, param in self.params.items()}