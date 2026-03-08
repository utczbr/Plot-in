
import unittest
import sys
from unittest.mock import MagicMock

sys.path.append('src')

# Mock PyQt6 before importing modules that rely on it
if 'PyQt6' not in sys.modules:
    mock_qt = MagicMock()
    sys.modules['PyQt6'] = mock_qt
    sys.modules['PyQt6.QtCore'] = mock_qt
    sys.modules['PyQt6.QtGui'] = mock_qt
    sys.modules['PyQt6.QtWidgets'] = mock_qt

from services.service_container import ServiceContainer, create_service_container
from services.orientation_service import Orientation
from services.orientation_detection_service import OrientationDetectionService

class TestServices(unittest.TestCase):
    
    def test_container_registration(self):
        """Test service container registration and retrieval"""
        container = ServiceContainer()
        container.register('test_service', lambda c: 'test_value', singleton=True)
        
        self.assertEqual(container.get('test_service'), 'test_value')
        
    def test_orientation_service_integration(self):
        """Test OrientationDetectionService via container"""
        container = create_service_container()
        service = container.get('orientation_detector')
        
        self.assertIsInstance(service, OrientationDetectionService)
        
        # Test basic detection interface
        result = service.detect(
            elements=[],
            img_width=100,
            img_height=100
        )
        self.assertIn(result.orientation, [Orientation.VERTICAL, Orientation.HORIZONTAL])

if __name__ == '__main__':
    unittest.main()
