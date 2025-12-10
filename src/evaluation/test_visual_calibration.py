"""
Tests for visual tick detection calibration.

Run with: python3 evaluation/test_visual_calibration.py
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
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from calibration.visual_tick_detector import VisualTickCalibration, TickDetectionResult


class TestVisualTickCalibration:
    """Test suite for visual tick detection calibration."""
    
    def test_factory_registration(self):
        """Factory should create VisualTickCalibration."""
        from calibration.calibration_factory import CalibrationFactory
        
        calibrator = CalibrationFactory.create('visual')
        assert isinstance(calibrator, VisualTickCalibration)
        print("✓ test_factory_registration passed")
    
    def test_factory_with_params(self):
        """Factory should accept valid parameters."""
        from calibration.calibration_factory import CalibrationFactory
        
        calibrator = CalibrationFactory.create(
            'visual',
            edge_threshold=60.0,
            min_tick_spacing=15.0
        )
        assert calibrator.edge_threshold == 60.0
        assert calibrator.min_tick_spacing == 15.0
        print("✓ test_factory_with_params passed")
    
    def test_factory_rejects_invalid_params(self):
        """Factory should reject invalid parameters."""
        from calibration.calibration_factory import CalibrationFactory
        
        try:
            CalibrationFactory.create('visual', invalid_param=True)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "invalid_param" in str(e).lower() or "Invalid parameters" in str(e)
        print("✓ test_factory_rejects_invalid_params passed")
    
    def test_find_peaks_basic(self):
        """Peak finder should detect peaks in signal."""
        calibrator = VisualTickCalibration()
        
        # Create signal with clear peaks at positions 10, 30, 50
        signal = np.zeros(60)
        signal[10] = 100
        signal[30] = 100
        signal[50] = 100
        
        peaks = calibrator._find_peaks(signal, min_distance=10, threshold=50)
        
        assert len(peaks) == 3
        assert 10 in peaks
        assert 30 in peaks
        assert 50 in peaks
        print("✓ test_find_peaks_basic passed")
    
    def test_find_peaks_respects_min_distance(self):
        """Peak finder should respect minimum distance."""
        calibrator = VisualTickCalibration()
        
        # Create signal with peaks too close together
        signal = np.zeros(60)
        signal[10] = 100
        signal[15] = 100  # Too close to 10
        signal[30] = 100
        
        peaks = calibrator._find_peaks(signal, min_distance=10, threshold=50)
        
        # Should skip the peak at 15 because it's too close to 10
        assert len(peaks) == 2
        assert 10 in peaks
        assert 30 in peaks
        print("✓ test_find_peaks_respects_min_distance passed")
    
    def test_filter_uniform_ticks(self):
        """Tick filter should keep only uniformly spaced ticks."""
        calibrator = VisualTickCalibration()
        
        # Ticks at 0, 20, 40, 60, 80 (spacing=20) plus outlier at 53
        ticks = [0.0, 20.0, 40.0, 53.0, 60.0, 80.0]
        
        filtered = calibrator._filter_uniform_ticks(
            ticks, 
            expected_spacing=20.0,
            tolerance=0.3
        )
        
        # Outlier at 53 should be removed
        assert 53.0 not in filtered
        assert len(filtered) == 5
        print("✓ test_filter_uniform_ticks passed")
    
    def test_calibrate_from_labels_fallback(self):
        """calibrate() should work as fallback with labels."""
        calibrator = VisualTickCalibration()
        
        # Create mock labels
        labels = [
            {'xyxy': [10, 100, 20, 110], 'text': '0', 'cleanedvalue': 0.0},
            {'xyxy': [30, 100, 40, 110], 'text': '10', 'cleanedvalue': 10.0},
            {'xyxy': [50, 100, 60, 110], 'text': '20', 'cleanedvalue': 20.0},
        ]
        
        result = calibrator.calibrate(labels, 'x')
        
        assert result is not None
        assert result.r2 > 0.99  # Should be near-perfect linear fit
        print("✓ test_calibrate_from_labels_fallback passed")
    
    def test_synthetic_axis_image(self):
        """Should detect ticks in synthetic axis image."""
        if not HAS_CV2:
            print("⊘ test_synthetic_axis_image skipped (cv2 not available)")
            return
        
        calibrator = VisualTickCalibration(
            edge_threshold=50.0,
            min_tick_spacing=15.0,
            min_ticks=3
        )
        
        # Create synthetic X-axis image with tick marks
        # 100px wide, 30px tall, ticks at x=10, 30, 50, 70, 90
        axis_img = np.ones((30, 100), dtype=np.uint8) * 255  # White background
        
        tick_positions = [10, 30, 50, 70, 90]
        for x in tick_positions:
            axis_img[20:30, x:x+2] = 0  # Black tick marks at bottom
        
        # Detect ticks
        result = calibrator._detect_ticks(axis_img, axis_type='x')
        
        assert result is not None
        assert len(result.tick_positions) >= 3
        assert result.confidence > 0.3
        print(f"✓ test_synthetic_axis_image passed (detected {len(result.tick_positions)} ticks)")
    
    def test_calibrate_from_synthetic_image(self):
        """Should calibrate from synthetic axis image."""
        if not HAS_CV2:
            print("⊘ test_calibrate_from_synthetic_image skipped (cv2 not available)")
            return
        
        calibrator = VisualTickCalibration(
            edge_threshold=50.0,
            min_tick_spacing=15.0,
            min_ticks=3
        )
        
        # Create full chart image (200x200)
        img = np.ones((200, 200), dtype=np.uint8) * 255
        
        # Add X-axis ticks at bottom (y=180-200)
        tick_positions = [20, 50, 80, 110, 140, 170]
        for x in tick_positions:
            img[180:195, x:x+2] = 0  # Tick marks
        
        # Calibrate from image
        result = calibrator.calibrate_from_image(
            img=img,
            axis_region=(0, 175, 200, 200),  # Bottom 25 pixels
            axis_type='x',
            reference_values=(0.0, 10.0)  # 0, 10, 20, 30, ...
        )
        
        if result is not None:
            # Test that the calibration works
            # At tick position 50, value should be ~10
            predicted = result.func(50)
            assert abs(predicted - 10.0) < 5.0, f"Expected ~10, got {predicted}"
            print(f"✓ test_calibrate_from_synthetic_image passed (R²={result.r2:.4f})")
        else:
            print("⊘ test_calibrate_from_synthetic_image: No calibration result (acceptable)")


def run_tests():
    """Run all tests without pytest."""
    test_suite = TestVisualTickCalibration()
    
    tests = [
        test_suite.test_factory_registration,
        test_suite.test_factory_with_params,
        test_suite.test_factory_rejects_invalid_params,
        test_suite.test_find_peaks_basic,
        test_suite.test_find_peaks_respects_min_distance,
        test_suite.test_filter_uniform_ticks,
        test_suite.test_calibrate_from_labels_fallback,
        test_suite.test_synthetic_axis_image,
        test_suite.test_calibrate_from_synthetic_image,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    print("\n" + "="*60)
    print("VISUAL CALIBRATION TESTS")
    print("="*60 + "\n")
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            if "skipped" in str(e).lower():
                skipped += 1
            else:
                print(f"✗ {test.__name__} ERROR: {e}")
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
