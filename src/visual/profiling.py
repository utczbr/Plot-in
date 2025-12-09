# profiling.py - PERFORMANCE MONITORING
import time
import functools
from contextlib import contextmanager

class PerformanceMonitor:
    """Track execution times and bottlenecks"""
    
    def __init__(self):
        self.timings = {}
        
    def record(self, operation, duration):
        if operation not in self.timings:
            self.timings[operation] = []
        self.timings[operation].append(duration)
        
    def report(self):
        """Generate performance report"""
        print("\n=== PERFORMANCE REPORT ===")
        for op, durations in sorted(self.timings.items()):
            avg = sum(durations) / len(durations)
            total = sum(durations)
            count = len(durations)
            print(f"{op:30s}: {avg:6.2f}s avg, {total:6.2f}s total, {count:3d} calls")
            
    @contextmanager
    def measure(self, operation):
        """Context manager for timing"""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.record(operation, duration)
            
def timed(monitor, operation_name=None):
    """Decorator for timing functions"""
    def decorator(func):
        op_name = operation_name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with monitor.measure(op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator