"""
Tests for neural and log calibration.

Run with: python3 evaluation/test_neural_calibration.py
"""
import sys
import os
import numpy as np

# Add src to path for imports
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, src_dir)

try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from calibration.calibration_neural import NeuralCalibration, LogCalibration, AxisType


class TestNeuralCalibration:
    """Test suite for neural calibration."""
    
    def test_factory_registration(self):
        """Factory should create NeuralCalibration."""
        from calibration.calibration_factory import CalibrationFactory
        
        calibrator = CalibrationFactory.create('neural')
        assert isinstance(calibrator, NeuralCalibration)
        print("✓ test_factory_registration passed")
    
    def test_factory_with_params(self):
        """Factory should accept valid parameters."""
        from calibration.calibration_factory import CalibrationFactory
        
        calibrator = CalibrationFactory.create(
            'neural',
            hidden_dim=64,
            max_epochs=50
        )
        assert calibrator.hidden_dim == 64
        assert calibrator.max_epochs == 50
        print("✓ test_factory_with_params passed")
    
    def test_axis_type_detection_linear(self):
        """Should detect linear axis from uniform spacing."""
        calibrator = NeuralCalibration()
        
        # Linear values: 0, 10, 20, 30, 40
        values = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        
        detection = calibrator._detect_axis_type(values)
        
        assert detection.axis_type == AxisType.LINEAR
        assert detection.confidence > 0.8
        print(f"✓ test_axis_type_detection_linear passed (confidence={detection.confidence:.2f})")
    
    def test_axis_type_detection_log(self):
        """Should detect log axis from geometric spacing."""
        calibrator = NeuralCalibration()
        
        # Log values: 1, 10, 100, 1000, 10000 (powers of 10)
        values = np.array([1.0, 10.0, 100.0, 1000.0, 10000.0])
        
        detection = calibrator._detect_axis_type(values)
        
        assert detection.axis_type == AxisType.LOG
        assert detection.confidence > 0.5
        print(f"✓ test_axis_type_detection_log passed (confidence={detection.confidence:.2f})")
    
    def test_linear_delegation(self):
        """Should delegate linear axes to fast calibration."""
        calibrator = NeuralCalibration()
        
        # Create mock linear labels
        # X positions: 15, 35, 55, 75 (centers of xyxy boxes)
        # Values: 0, 10, 20, 30
        # This creates slope = 10/20 = 0.5, so pixel 35 -> 10
        labels = [
            {'xyxy': [10, 100, 20, 110], 'text': '0', 'cleanedvalue': 0.0},
            {'xyxy': [30, 100, 40, 110], 'text': '10', 'cleanedvalue': 10.0},
            {'xyxy': [50, 100, 60, 110], 'text': '20', 'cleanedvalue': 20.0},
            {'xyxy': [70, 100, 80, 110], 'text': '30', 'cleanedvalue': 30.0},
        ]
        
        result = calibrator.calibrate(labels, 'x')
        
        assert result is not None
        assert result.r2 > 0.99  # Should be near-perfect
        
        # Test prediction: pixel 35 (center of second box) should be ~10
        predicted = result.func(35.0)
        assert abs(predicted - 10.0) < 1.0, f"Expected ~10, got {predicted}"
        print("✓ test_linear_delegation passed")
    
    def test_neural_training(self):
        """Should train neural network for non-linear data."""
        if not HAS_TORCH:
            print("⊘ test_neural_training skipped (torch not available)")
            return
        
        calibrator = NeuralCalibration(max_epochs=50, hidden_dim=16)
        
        # Create non-linear data (quadratic)
        coords = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        values = coords ** 2 / 100  # 1, 4, 9, 16, 25
        
        calibrator._train_neural(coords, values)
        
        # Test prediction
        predicted = calibrator._neural_predict(coords)
        
        # Should be reasonably close
        mse = np.mean((predicted - values) ** 2)
        assert mse < 1.0, f"MSE too high: {mse}"
        print(f"✓ test_neural_training passed (MSE={mse:.4f})")


class TestLogCalibration:
    """Test suite for logarithmic calibration."""
    
    def test_factory_registration(self):
        """Factory should create LogCalibration."""
        from calibration.calibration_factory import CalibrationFactory
        
        calibrator = CalibrationFactory.create('log')
        assert isinstance(calibrator, LogCalibration)
        print("✓ test_log_factory_registration passed")
    
    def test_log_calibration_powers_of_10(self):
        """Should calibrate log axis with powers of 10."""
        calibrator = LogCalibration()
        
        # Labels at pixels 100, 200, 300 with values 1, 10, 100
        labels = [
            {'xyxy': [90, 100, 110, 110], 'text': '1', 'cleanedvalue': 1.0},
            {'xyxy': [190, 100, 210, 110], 'text': '10', 'cleanedvalue': 10.0},
            {'xyxy': [290, 100, 310, 110], 'text': '100', 'cleanedvalue': 100.0},
        ]
        
        result = calibrator.calibrate(labels, 'x')
        
        assert result is not None
        assert result.r2 > 0.99  # Should be perfect in log space
        
        # Test prediction: at pixel 200, should be ~10
        predicted = result.func(200.0)
        assert abs(predicted - 10.0) < 2.0, f"Expected ~10, got {predicted}"
        
        # At pixel 250, should be between 10 and 100
        mid_predicted = result.func(250.0)
        assert 10 < mid_predicted < 100, f"Expected 10-100, got {mid_predicted}"
        print(f"✓ test_log_calibration_powers_of_10 passed (predicted at 250px: {mid_predicted:.1f})")
    
    def test_log_rejects_non_positive(self):
        """Should handle non-positive values gracefully."""
        calibrator = LogCalibration()
        
        # Labels with zero and negative values
        labels = [
            {'xyxy': [100, 100, 110, 110], 'text': '-10', 'cleanedvalue': -10.0},
            {'xyxy': [200, 100, 210, 110], 'text': '0', 'cleanedvalue': 0.0},
        ]
        
        result = calibrator.calibrate(labels, 'x')
        
        # Should return None or fallback since can't log non-positive
        assert result is None
        print("✓ test_log_rejects_non_positive passed")


def run_tests():
    """Run all tests without pytest."""
    test_suites = [
        TestNeuralCalibration(),
        TestLogCalibration(),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    print("\n" + "="*60)
    print("NEURAL & LOG CALIBRATION TESTS")
    print("="*60 + "\n")
    
    for suite in test_suites:
        for name in dir(suite):
            if name.startswith('test_'):
                test = getattr(suite, name)
                try:
                    test()
                    passed += 1
                except AssertionError as e:
                    print(f"✗ {name} FAILED: {e}")
                    failed += 1
                except Exception as e:
                    if "skipped" in str(e).lower():
                        skipped += 1
                    else:
                        print(f"✗ {name} ERROR: {e}")
                        failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*60 + "\n")
    
    return failed == 0


if __name__ == '__main__':
    if HAS_PYTEST:
        import pytest
        pytest.main([__file__, '-v'])
    else:
        success = run_tests()
        sys.exit(0 if success else 1)
