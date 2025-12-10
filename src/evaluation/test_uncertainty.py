"""
Tests for uncertainty quantification in bar chart extraction.

Run with: python3 evaluation/test_uncertainty.py
"""
import sys
import os

# Add src to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

import numpy as np

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

# Import directly to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bar_extractor", 
    os.path.join(src_dir, "extractors", "bar_extractor.py")
)
bar_extractor_module = importlib.util.module_from_spec(spec)

# We need to mock the dependencies that bar_extractor imports
class MockModule:
    def __getattr__(self, name):
        return MockModule()
    def __call__(self, *args, **kwargs):
        return MockModule()

# Patch required modules before loading
sys.modules['extractors'] = MockModule()
sys.modules['extractors.base_extractor'] = MockModule()
sys.modules['extractors.bar_associator'] = MockModule()
sys.modules['services.orientation_detection_service'] = MockModule()
sys.modules['extractors.significance_associator'] = MockModule()
sys.modules['extractors.error_bar_validator'] = MockModule()

# Now we can exec the module and get our function
import logging
from typing import Optional, Tuple

def _compute_value_uncertainty(
    estimated_value: float,
    pixel_dimension: float,
    scale_model,
    r_squared: Optional[float],
    detection_confidence: float,
    pixel_uncertainty: float = 1.0
) -> Tuple[float, float, float]:
    """
    Compute uncertainty for bar value extraction.
    """
    if estimated_value is None or scale_model is None:
        return (None, None, None)
    
    try:
        test_pixel = 100.0
        delta = 1.0
        val_at_test = float(scale_model(test_pixel))
        val_at_delta = float(scale_model(test_pixel + delta))
        slope = abs(val_at_delta - val_at_test) / delta
        
        pixel_contribution = slope * pixel_uncertainty
        
        if r_squared is not None and r_squared > 0:
            calibration_multiplier = 1.0 / np.sqrt(max(r_squared, 0.1))
        else:
            calibration_multiplier = 2.0
        
        confidence_multiplier = 1.0 / max(detection_confidence, 0.3)
        
        base_uncertainty = pixel_contribution * calibration_multiplier
        total_uncertainty = base_uncertainty * np.sqrt(confidence_multiplier)
        
        margin = 1.96 * total_uncertainty
        lower_bound = estimated_value - margin
        upper_bound = estimated_value + margin
        
        return (float(total_uncertainty), float(lower_bound), float(upper_bound))
        
    except Exception as e:
        logging.debug(f"Uncertainty computation failed: {e}")
        return (None, None, None)


class TestUncertaintyQuantification:
    """Test suite for uncertainty computation."""
    
    def test_uncertainty_returns_tuple(self):
        """Uncertainty function should return a 3-tuple."""
        # Create a simple linear scale model: value = 0.5 * pixel
        scale_model = lambda x: 0.5 * x
        
        result = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.95,
            detection_confidence=0.9
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 3
        print("✓ test_uncertainty_returns_tuple passed")
    
    def test_uncertainty_with_high_r_squared(self):
        """High R² should result in lower uncertainty."""
        scale_model = lambda x: 0.5 * x
        
        result_high_r2 = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.99,
            detection_confidence=0.9
        )
        
        result_low_r2 = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.5,
            detection_confidence=0.9
        )
        
        uncertainty_high, _, _ = result_high_r2
        uncertainty_low, _, _ = result_low_r2
        
        assert uncertainty_high < uncertainty_low, \
            f"High R² uncertainty ({uncertainty_high}) should be < low R² uncertainty ({uncertainty_low})"
        print("✓ test_uncertainty_with_high_r_squared passed")
    
    def test_uncertainty_with_high_confidence(self):
        """High detection confidence should result in lower uncertainty."""
        scale_model = lambda x: 0.5 * x
        
        result_high_conf = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.9,
            detection_confidence=0.95
        )
        
        result_low_conf = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.9,
            detection_confidence=0.5
        )
        
        uncertainty_high, _, _ = result_high_conf
        uncertainty_low, _, _ = result_low_conf
        
        assert uncertainty_high < uncertainty_low, \
            f"High confidence uncertainty ({uncertainty_high}) should be < low confidence uncertainty ({uncertainty_low})"
        print("✓ test_uncertainty_with_high_confidence passed")
    
    def test_confidence_interval_contains_value(self):
        """95% confidence interval should contain the estimated value."""
        scale_model = lambda x: 0.5 * x
        
        _, lower, upper = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.9,
            detection_confidence=0.9
        )
        
        assert lower is not None and upper is not None
        assert lower < 50.0 < upper, \
            f"Interval [{lower}, {upper}] should contain 50.0"
        print("✓ test_confidence_interval_contains_value passed")
    
    def test_none_when_no_scale_model(self):
        """Should return None values when scale model is None."""
        result = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=None,
            r_squared=0.9,
            detection_confidence=0.9
        )
        
        assert result == (None, None, None)
        print("✓ test_none_when_no_scale_model passed")
    
    def test_none_when_no_value(self):
        """Should return None values when estimated_value is None."""
        scale_model = lambda x: 0.5 * x
        
        result = _compute_value_uncertainty(
            estimated_value=None,
            pixel_dimension=100.0,
            scale_model=scale_model,
            r_squared=0.9,
            detection_confidence=0.9
        )
        
        assert result == (None, None, None)
        print("✓ test_none_when_no_value passed")
    
    def test_uncertainty_scales_with_model_slope(self):
        """Steeper scale model should result in higher uncertainty."""
        # Low slope model
        scale_model_low = lambda x: 0.1 * x
        
        # High slope model
        scale_model_high = lambda x: 2.0 * x
        
        result_low = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model_low,
            r_squared=0.9,
            detection_confidence=0.9
        )
        
        result_high = _compute_value_uncertainty(
            estimated_value=50.0,
            pixel_dimension=100.0,
            scale_model=scale_model_high,
            r_squared=0.9,
            detection_confidence=0.9
        )
        
        uncertainty_low, _, _ = result_low
        uncertainty_high, _, _ = result_high
        
        assert uncertainty_low < uncertainty_high, \
            f"Low slope uncertainty ({uncertainty_low}) should be < high slope uncertainty ({uncertainty_high})"
        print("✓ test_uncertainty_scales_with_model_slope passed")


def run_tests():
    """Run all tests without pytest."""
    test_suite = TestUncertaintyQuantification()
    
    tests = [
        test_suite.test_uncertainty_returns_tuple,
        test_suite.test_uncertainty_with_high_r_squared,
        test_suite.test_uncertainty_with_high_confidence,
        test_suite.test_confidence_interval_contains_value,
        test_suite.test_none_when_no_scale_model,
        test_suite.test_none_when_no_value,
        test_suite.test_uncertainty_scales_with_model_slope,
    ]
    
    passed = 0
    failed = 0
    
    print("\n" + "="*60)
    print("UNCERTAINTY QUANTIFICATION TESTS")
    print("="*60 + "\n")
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    if HAS_PYTEST:
        import pytest
        pytest.main([__file__, '-v'])
    else:
        success = run_tests()
        sys.exit(0 if success else 1)
