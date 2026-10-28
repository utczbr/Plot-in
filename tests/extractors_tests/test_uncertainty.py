"""
Tests for uncertainty quantification in bar chart extraction.
"""

from extractors.bar_extractor import _compute_value_uncertainty


class TestUncertaintyQuantification:
    """Test suite for uncertainty computation in BarExtractor."""

    def test_uncertainty_returns_tuple(self):
        """Uncertainty function should return a 3-tuple."""
        scale_model = lambda x: 0.5 * x
        result = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.95,
            detection_confidence=0.9,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_uncertainty_with_high_r_squared(self):
        """High R² should result in lower uncertainty."""
        scale_model = lambda x: 0.5 * x
        uncertainty_high, _, _ = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.99,
            detection_confidence=0.9,
        )
        uncertainty_low, _, _ = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.5,
            detection_confidence=0.9,
        )
        assert uncertainty_high < uncertainty_low

    def test_uncertainty_with_high_confidence(self):
        """High detection confidence should result in lower uncertainty."""
        scale_model = lambda x: 0.5 * x
        uncertainty_high, _, _ = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.9,
            detection_confidence=0.95,
        )
        uncertainty_low, _, _ = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.9,
            detection_confidence=0.5,
        )
        assert uncertainty_high < uncertainty_low

    def test_confidence_interval_contains_value(self):
        """95% confidence interval should contain the estimated value."""
        scale_model = lambda x: 0.5 * x
        _, lower, upper = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.9,
            detection_confidence=0.9,
        )
        assert lower is not None and upper is not None
        assert lower < 50.0 < upper

    def test_none_when_no_scale_model(self):
        """Should return None values when scale model is None."""
        result = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=None,
            r_squared=0.9,
            detection_confidence=0.9,
        )
        assert result == (None, None, None)

    def test_none_when_no_value(self):
        """Should return None values when estimated_value is None."""
        scale_model = lambda x: 0.5 * x
        result = _compute_value_uncertainty(
            estimated_value=None,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.9,
            detection_confidence=0.9,
        )
        assert result == (None, None, None)

    def test_uncertainty_scales_with_model_slope(self):
        """Steeper scale model should result in higher uncertainty."""
        scale_model_low = lambda x: 0.1 * x
        scale_model_high = lambda x: 2.0 * x
        uncertainty_low, _, _ = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model_low,
            r_squared=0.9,
            detection_confidence=0.9,
        )
        uncertainty_high, _, _ = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model_high,
            r_squared=0.9,
            detection_confidence=0.9,
        )
        assert uncertainty_low < uncertainty_high
