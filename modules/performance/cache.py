"""
Cache Management Module

Provides intelligent caching for expensive computations:
- Function result caching
- DataFrame caching with hash-based keys
- Automatic cache invalidation
- Memory-aware cache limits
"""

import hashlib
import pickle
import functools
import logging
from typing import Callable, Any, Dict, Optional, Union
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """Intelligent cache manager with memory limits"""
    
    def __init__(self, max_memory_mb: int = 500, cache_dir: Optional[Path] = None):
        self.max_memory_mb = max_memory_mb
        self.cache_dir = cache_dir or Path("/app/data/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_cache = {}
        self.cache_metadata = {}
        self.current_memory_mb = 0
        
    def _calculate_size_mb(self, obj: Any) -> float:
        """Calculate object size in MB"""
        try:
            if isinstance(obj, pd.DataFrame):
                return obj.memory_usage(deep=True).sum() / 1024 / 1024
            elif isinstance(obj, np.ndarray):
                return obj.nbytes / 1024 / 1024
            else:
                return len(pickle.dumps(obj)) / 1024 / 1024
        except:
            return 0.1  # Default fallback
    
    def _generate_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function name and arguments"""
        # Create a stable hash from arguments
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        
        key_string = str(key_data)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _evict_oldest(self):
        """Evict oldest cache entries to free memory"""
        if not self.cache_metadata:
            return
            
        # Sort by last access time
        sorted_keys = sorted(
            self.cache_metadata.keys(),
            key=lambda k: self.cache_metadata[k]['last_access']
        )
        
        # Remove oldest entries until under memory limit
        for key in sorted_keys:
            if self.current_memory_mb <= self.max_memory_mb * 0.8:
                break
                
            if key in self.memory_cache:
                size_mb = self.cache_metadata[key]['size_mb']
                del self.memory_cache[key]
                del self.cache_metadata[key]
                self.current_memory_mb -= size_mb
                logger.info(f"Cache evicted: {key[:8]}... ({size_mb:.2f}MB freed)")
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key in self.memory_cache:
            self.cache_metadata[key]['last_access'] = datetime.now()
            self.cache_metadata[key]['hits'] += 1
            return self.memory_cache[key]
        
        # Try file cache
        file_path = self.cache_dir / f"{key}.pkl"
        if file_path.exists():
            try:
                with open(file_path, 'rb') as f:
                    value = pickle.load(f)
                    
                # Load back to memory cache if space available
                size_mb = self._calculate_size_mb(value)
                if self.current_memory_mb + size_mb <= self.max_memory_mb:
                    self.memory_cache[key] = value
                    self.cache_metadata[key] = {
                        'size_mb': size_mb,
                        'last_access': datetime.now(),
                        'hits': 1,
                        'created': datetime.now()
                    }
                    self.current_memory_mb += size_mb
                    
                return value
            except Exception as e:
                logger.warning(f"Failed to load cache file {key}: {e}")
                file_path.unlink(missing_ok=True)
        
        return None
    
    def set(self, key: str, value: Any, persist: bool = True):
        """Set cached value"""
        size_mb = self._calculate_size_mb(value)
        
        # Memory cache
        if size_mb <= self.max_memory_mb:
            # Ensure we have space
            if self.current_memory_mb + size_mb > self.max_memory_mb:
                self._evict_oldest()
            
            self.memory_cache[key] = value
            self.cache_metadata[key] = {
                'size_mb': size_mb,
                'last_access': datetime.now(),
                'hits': 0,
                'created': datetime.now()
            }
            self.current_memory_mb += size_mb
        
        # File cache for persistence
        if persist:
            try:
                file_path = self.cache_dir / f"{key}.pkl"
                with open(file_path, 'wb') as f:
                    pickle.dump(value, f)
            except Exception as e:
                logger.warning(f"Failed to persist cache {key}: {e}")
    
    def clear(self, pattern: Optional[str] = None):
        """Clear cache entries"""
        if pattern is None:
            # Clear all
            self.memory_cache.clear()
            self.cache_metadata.clear()
            self.current_memory_mb = 0
            
            # Clear file cache
            for file_path in self.cache_dir.glob("*.pkl"):
                file_path.unlink()
        else:
            # Clear matching pattern
            keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
            for key in keys_to_remove:
                if key in self.memory_cache:
                    self.current_memory_mb -= self.cache_metadata[key]['size_mb']
                    del self.memory_cache[key]
                    del self.cache_metadata[key]
                
                # Remove file
                file_path = self.cache_dir / f"{key}.pkl"
                file_path.unlink(missing_ok=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_hits = sum(meta['hits'] for meta in self.cache_metadata.values())
        
        return {
            'memory_usage_mb': round(self.current_memory_mb, 2),
            'memory_limit_mb': self.max_memory_mb,
            'memory_usage_percent': round(self.current_memory_mb / self.max_memory_mb * 100, 1),
            'entries_count': len(self.memory_cache),
            'total_hits': total_hits,
            'cache_files': len(list(self.cache_dir.glob("*.pkl")))
        }

# Global cache manager
cache_manager = CacheManager()

def cached_function(expire_hours: int = 24, persist: bool = True, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            func_name = f"{key_prefix}{func.__name__}" if key_prefix else func.__name__
            cache_key = cache_manager._generate_key(func_name, args, kwargs)
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache hit: {func.__name__}")
                return cached_result
            
            # Execute function
            logger.debug(f"Cache miss: {func.__name__}")
            result = func(*args, **kwargs)
            
            # Cache result
            cache_manager.set(cache_key, result, persist=persist)
            
            return result
        return wrapper
    return decorator

def cache_dataframe_operation(func: Callable) -> Callable:
    """Specialized caching for DataFrame operations"""
    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, *args, **kwargs):
        # Create hash from DataFrame content and arguments
        df_hash = hashlib.md5(pd.util.hash_pandas_object(df).values).hexdigest()[:8]
        args_hash = hashlib.md5(str((args, kwargs)).encode()).hexdigest()[:8]
        cache_key = f"df_{func.__name__}_{df_hash}_{args_hash}"
        
        # Try cache
        cached_result = cache_manager.get(cache_key)
        if cached_result is not None:
            logger.debug(f"DataFrame cache hit: {func.__name__}")
            return cached_result
        
        # Execute and cache
        result = func(df, *args, **kwargs)
        cache_manager.set(cache_key, result, persist=True)
        
        return result
    return wrapper 