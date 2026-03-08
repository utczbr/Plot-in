
import unittest
import sys
from unittest.mock import MagicMock

sys.path.append('src')

from extractors.bar_extractor import BarExtractor
from extractors.line_extractor import LineExtractor
from extractors.scatter_extractor import ScatterExtractor
from extractors.box_extractor import BoxExtractor

class TestExtractors(unittest.TestCase):
    
    def test_instantiation(self):
        """Verify extractors can be instantiated"""
        extractors = [
            BarExtractor,
            LineExtractor,
            ScatterExtractor,
            BoxExtractor
        ]
        
        for cls in extractors:
            instance = cls()
            self.assertIsNotNone(instance)
            
    def test_bar_extractor_interface(self):
        """Test BarExtractor interface"""
        extractor = BarExtractor()
        # Mock dependencies in extraction context if needed, 
        # but for now just check if method exists and accepts arguments
        self.assertTrue(hasattr(extractor, 'extract'))

if __name__ == '__main__':
    unittest.main()
