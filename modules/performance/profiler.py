"""
Performance Profiler Module

Provides decorators and utilities for performance monitoring:
- Memory usage tracking
- Execution time measurement
- Resource utilization monitoring
"""

import time
import psutil
import functools
import logging
from typing import Callable, Any, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure"""
    function_name: str
    execution_time: float
    memory_before: float
    memory_after: float
    memory_peak: float
    memory_delta: float
    cpu_percent: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'function': self.function_name,
            'time_seconds': round(self.execution_time, 4),
            'memory_before_mb': round(self.memory_before, 2),
            'memory_after_mb': round(self.memory_after, 2),
            'memory_peak_mb': round(self.memory_peak, 2),
            'memory_delta_mb': round(self.memory_delta, 2),
            'cpu_percent': round(self.cpu_percent, 2)
        }

class PerformanceProfiler:
    """Centralized performance profiler"""
    
    def __init__(self):
        self.metrics_history = []
        self.enabled = True
        
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics"""
        if self.enabled:
            self.metrics_history.append(metrics)
            logger.info(f"Performance: {metrics.function_name} - {metrics.execution_time:.4f}s, {metrics.memory_delta:+.2f}MB")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics_history:
            return {}
            
        total_time = sum(m.execution_time for m in self.metrics_history)
        avg_memory_delta = sum(m.memory_delta for m in self.metrics_history) / len(self.metrics_history)
        
        return {
            'total_functions': len(self.metrics_history),
            'total_time_seconds': round(total_time, 4),
            'avg_memory_delta_mb': round(avg_memory_delta, 2),
            'recent_metrics': [m.to_dict() for m in self.metrics_history[-5:]]
        }
    
    def save_report(self, filepath: Optional[Path] = None):
        """Save performance report to file"""
        if filepath is None:
            filepath = Path("/app/results/latest/performance_report.json")
            
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'summary': self.get_summary(),
                'all_metrics': [m.to_dict() for m in self.metrics_history]
            }, f, indent=2)

# Global profiler instance
profiler = PerformanceProfiler()

def memory_monitor(func: Callable) -> Callable:
    """Decorator to monitor memory usage"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not profiler.enabled:
            return func(*args, **kwargs)
            
        # Get initial state
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB
        cpu_before = psutil.cpu_percent()
        
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
        finally:
            # Get final state
            end_time = time.time()
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            cpu_after = psutil.cpu_percent()
            
            # Calculate metrics
            metrics = PerformanceMetrics(
                function_name=func.__name__,
                execution_time=end_time - start_time,
                memory_before=memory_before,
                memory_after=memory_after,
                memory_peak=max(memory_before, memory_after),
                memory_delta=memory_after - memory_before,
                cpu_percent=(cpu_before + cpu_after) / 2
            )
            
            profiler.record_metrics(metrics)
        
        return result
    return wrapper

def time_monitor(func: Callable) -> Callable:
    """Decorator to monitor execution time only"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not profiler.enabled:
            return func(*args, **kwargs)
            
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"⏱️  {func.__name__}: {execution_time:.4f}s")
        
        return result
    return wrapper

def get_system_info() -> Dict[str, Any]:
    """Get current system information"""
    process = psutil.Process()
    
    return {
        'memory_mb': round(process.memory_info().rss / 1024 / 1024, 2),
        'memory_percent': round(process.memory_percent(), 2),
        'cpu_percent': round(psutil.cpu_percent(), 2),
        'num_threads': process.num_threads(),
        'open_files': len(process.open_files()),
        'connections': len(process.connections())
    } 