
import unittest
import sys
import os

def run_tests():
    # Ensure src is in python path
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    
    loader = unittest.TestLoader()
    start_dir = 'tests'
    suite = loader.discover(start_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        sys.exit(1)

if __name__ == '__main__':
    run_tests()
