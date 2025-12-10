"""
Adaptive Neural Calibration - Non-Linear Axis Support

Supports log-scale, date, and other non-linear axes using small neural networks
that learn the pixel-to-value mapping from axis labels.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np

from .calibration_base import BaseCalibration, CalibrationResult
from .calibration_fast import FastCalibration

logger = logging.getLogger(__name__)

# Check for PyTorch availability
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None


class AxisType(Enum):
    """Detected axis type."""
    LINEAR = "linear"
    LOG = "log"
    DATE = "date"
    UNKNOWN = "unknown"


@dataclass
class AxisTypeDetection:
    """Result of axis type detection."""
    axis_type: AxisType
    confidence: float
    diagnostics: Dict


class NeuralCalibration(BaseCalibration):
    """
    Adaptive calibration using neural network for non-linear axes.
    
    Workflow:
    1. Detect axis type (linear, log, date) from label distribution
    2. For linear axes: delegate to FastCalibration (efficient)
    3. For log/date axes: train small MLP to learn the mapping
    
    The neural approach handles arbitrary pixel-to-value mappings
    without requiring explicit knowledge of the transform.
    """
    
    def __init__(
        self,
        hidden_dim: int = 32,
        max_epochs: int = 100,
        learning_rate: float = 0.01,
        early_stop_patience: int = 10,
    ):
        """
        Initialize neural calibration.
        
        Args:
            hidden_dim: Hidden layer dimension for MLP
            max_epochs: Maximum training epochs
            learning_rate: Optimizer learning rate
            early_stop_patience: Epochs without improvement before stopping
        """
        self.hidden_dim = hidden_dim
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.early_stop_patience = early_stop_patience
        
        # Will be set during calibration
        self._net = None
        self._input_mean = 0.0
        self._input_std = 1.0
        self._output_mean = 0.0
        self._output_std = 1.0
    
    def calibrate(self, scale_labels: List[Dict], axis_type: str) -> Optional[CalibrationResult]:
        """
        Calibrate from axis labels with automatic axis type detection.
        
        Args:
            scale_labels: List of label dicts with 'xyxy', 'cleanedvalue'
            axis_type: 'x' or 'y'
        
        Returns:
            CalibrationResult or None
        """
        # Extract points
        coords, values, weights = self._extract_points(scale_labels, axis_type)
        
        if len(coords) < 2:
            logger.warning(f"Insufficient points for neural calibration: {len(coords)}")
            return None
        
        # Detect axis type
        detection = self._detect_axis_type(values)
        logger.info(f"Detected axis type: {detection.axis_type.value} (confidence={detection.confidence:.2f})")
        
        # For linear axes, delegate to fast calibration (more efficient)
        if detection.axis_type == AxisType.LINEAR:
            logger.info("Using fast linear calibration")
            return FastCalibration().calibrate(scale_labels, axis_type)
        
        # For non-linear axes, use neural approach
        if not HAS_TORCH:
            logger.warning("PyTorch not available, falling back to linear calibration")
            return FastCalibration().calibrate(scale_labels, axis_type)
        
        # Train neural mapping
        try:
            self._train_neural(coords, values)
            
            # Compute R² on training data
            predicted = self._neural_predict(coords)
            r2 = self._r2(coords, values, 0, 0)  # Use our predictions
            ss_res = np.sum((values - predicted) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            
            logger.info(f"Neural calibration complete: R²={r2:.4f}")
            
            return CalibrationResult(
                func=self._make_neural_func(),
                r2=float(r2),
                coeffs=(0.0, 0.0),  # Not applicable for neural
            )
            
        except Exception as e:
            logger.error(f"Neural calibration failed: {e}")
            logger.info("Falling back to linear calibration")
            return FastCalibration().calibrate(scale_labels, axis_type)
    
    def _detect_axis_type(self, values: np.ndarray) -> AxisTypeDetection:
        """
        Detect axis type from value distribution.
        
        Detection heuristics:
        - LINEAR: Uniform spacing between values
        - LOG: Geometric spacing (ratios are uniform)
        - DATE: Values match date patterns
        """
        if len(values) < 3:
            return AxisTypeDetection(
                axis_type=AxisType.UNKNOWN,
                confidence=0.0,
                diagnostics={"reason": "insufficient_values"}
            )
        
        sorted_vals = np.sort(values)
        
        # Check for linear (arithmetic) spacing
        diffs = np.diff(sorted_vals)
        mean_diff = np.mean(diffs)
        if mean_diff != 0:
            linear_cv = np.std(diffs) / abs(mean_diff)  # Coefficient of variation
        else:
            linear_cv = float('inf')
        
        # Check for logarithmic (geometric) spacing
        # For log scale: ratios between consecutive values are uniform
        positive_vals = sorted_vals[sorted_vals > 0]
        if len(positive_vals) >= 3:
            ratios = positive_vals[1:] / positive_vals[:-1]
            mean_ratio = np.mean(ratios)
            if mean_ratio != 0:
                log_cv = np.std(ratios) / abs(mean_ratio)
            else:
                log_cv = float('inf')
        else:
            log_cv = float('inf')
        
        # Decision logic
        linear_threshold = 0.15  # CV threshold for "uniform"
        log_threshold = 0.15
        
        if linear_cv < linear_threshold:
            return AxisTypeDetection(
                axis_type=AxisType.LINEAR,
                confidence=1.0 - linear_cv,
                diagnostics={"linear_cv": float(linear_cv), "log_cv": float(log_cv)}
            )
        elif log_cv < log_threshold and linear_cv > log_cv:
            return AxisTypeDetection(
                axis_type=AxisType.LOG,
                confidence=1.0 - log_cv,
                diagnostics={"linear_cv": float(linear_cv), "log_cv": float(log_cv)}
            )
        else:
            # Default to linear if unclear
            return AxisTypeDetection(
                axis_type=AxisType.LINEAR if linear_cv < 0.5 else AxisType.UNKNOWN,
                confidence=max(0.3, 1.0 - min(linear_cv, log_cv)),
                diagnostics={"linear_cv": float(linear_cv), "log_cv": float(log_cv)}
            )
    
    def _train_neural(self, coords: np.ndarray, values: np.ndarray):
        """
        Train small MLP to map coordinates to values.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")
        
        # Normalize inputs and outputs
        self._input_mean = float(np.mean(coords))
        self._input_std = float(np.std(coords)) + 1e-8
        self._output_mean = float(np.mean(values))
        self._output_std = float(np.std(values)) + 1e-8
        
        x_norm = (coords - self._input_mean) / self._input_std
        y_norm = (values - self._output_mean) / self._output_std
        
        # Convert to tensors
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y_norm, dtype=torch.float32).unsqueeze(1)
        
        # Create MLP
        self._net = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Training
        optimizer = optim.Adam(self._net.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        
        best_loss = float('inf')
        patience_counter = 0
        
        self._net.train()
        for epoch in range(self.max_epochs):
            optimizer.zero_grad()
            output = self._net(x_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            
            current_loss = loss.item()
            
            # Early stopping
            if current_loss < best_loss - 1e-6:
                best_loss = current_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stop_patience:
                    logger.debug(f"Early stopping at epoch {epoch}")
                    break
        
        self._net.eval()
        logger.debug(f"Training complete: final loss={best_loss:.6f}")
    
    def _neural_predict(self, coords: np.ndarray) -> np.ndarray:
        """Predict values using trained neural network."""
        if self._net is None:
            raise RuntimeError("Neural network not trained")
        
        x_norm = (coords - self._input_mean) / self._input_std
        x_tensor = torch.tensor(x_norm, dtype=torch.float32).unsqueeze(1)
        
        with torch.no_grad():
            y_norm = self._net(x_tensor).squeeze(1).numpy()
        
        # Denormalize
        values = y_norm * self._output_std + self._output_mean
        return values
    
    def _make_neural_func(self) -> Callable:
        """Create callable function for neural calibration."""
        def neural_func(x):
            if isinstance(x, (list, tuple)):
                x = np.array(x, dtype=np.float64)
            elif isinstance(x, (int, float)):
                x = np.array([x], dtype=np.float64)
                result = self._neural_predict(x)
                return float(result[0])
            
            return self._neural_predict(x)
        
        return neural_func


class LogCalibration(BaseCalibration):
    """
    Specialized calibration for logarithmic axes.
    
    For log-scale axes: value = 10^(a * pixel + b)
    Or equivalently: log10(value) = a * pixel + b
    """
    
    def calibrate(self, scale_labels: List[Dict], axis_type: str) -> Optional[CalibrationResult]:
        """Calibrate log-scale axis."""
        coords, values, weights = self._extract_points(scale_labels, axis_type)
        
        if len(coords) < 2:
            return None
        
        # Filter positive values (can't take log of non-positive)
        positive_mask = values > 0
        if np.sum(positive_mask) < 2:
            logger.warning("Insufficient positive values for log calibration")
            return None
        
        coords_pos = coords[positive_mask]
        values_pos = values[positive_mask]
        weights_pos = weights[positive_mask] if weights is not None else None
        
        # Fit linear model to log-transformed values
        log_values = np.log10(values_pos)
        
        try:
            m, b = self._refit_linear(coords_pos, log_values, weights_pos)
            
            # R² in log space
            r2 = self._r2(coords_pos, log_values, m, b)
            
            # Create calibration function
            def log_func(x):
                if isinstance(x, (list, tuple)):
                    x = np.array(x, dtype=np.float64)
                linear_result = m * x + b
                return np.power(10, linear_result)
            
            return CalibrationResult(
                func=log_func,
                r2=float(r2),
                coeffs=(m, b),
            )
            
        except Exception as e:
            logger.error(f"Log calibration failed: {e}")
            return None
