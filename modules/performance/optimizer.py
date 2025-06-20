"""
Data Optimizer Module

Provides optimization utilities for DataFrames and memory management:
- DataFrame memory optimization
- Automatic data type optimization
- Memory usage reduction
- Garbage collection utilities
"""

import gc
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
import psutil

logger = logging.getLogger(__name__)

class DataFrameOptimizer:
    """DataFrame memory optimization utilities"""
    
    @staticmethod
    def optimize_dtypes(df: pd.DataFrame, aggressive: bool = False) -> pd.DataFrame:
        """Optimize DataFrame data types for memory efficiency"""
        df_optimized = df.copy()
        original_memory = df_optimized.memory_usage(deep=True).sum()
        
        for col in df_optimized.columns:
            col_type = df_optimized[col].dtype
            
            # Optimize numeric columns
            if pd.api.types.is_numeric_dtype(col_type):
                df_optimized[col] = DataFrameOptimizer._optimize_numeric_column(
                    df_optimized[col], aggressive
                )
            
            # Optimize string columns
            elif pd.api.types.is_object_dtype(col_type):
                df_optimized[col] = DataFrameOptimizer._optimize_string_column(
                    df_optimized[col], aggressive
                )
        
        optimized_memory = df_optimized.memory_usage(deep=True).sum()
        reduction_percent = (1 - optimized_memory / original_memory) * 100
        
        logger.info(f"DataFrame optimized: {original_memory/1024/1024:.2f}MB → {optimized_memory/1024/1024:.2f}MB ({reduction_percent:.1f}% reduction)")
        
        return df_optimized
    
    @staticmethod
    def _optimize_numeric_column(series: pd.Series, aggressive: bool = False) -> pd.Series:
        """Optimize numeric column data type"""
        if pd.api.types.is_integer_dtype(series):
            # Integer optimization
            min_val, max_val = series.min(), series.max()
            
            if min_val >= 0:
                # Unsigned integers
                if max_val < 255:
                    return series.astype(np.uint8)
                elif max_val < 65535:
                    return series.astype(np.uint16)
                elif max_val < 4294967295:
                    return series.astype(np.uint32)
            else:
                # Signed integers
                if min_val > -128 and max_val < 127:
                    return series.astype(np.int8)
                elif min_val > -32768 and max_val < 32767:
                    return series.astype(np.int16)
                elif min_val > -2147483648 and max_val < 2147483647:
                    return series.astype(np.int32)
        
        elif pd.api.types.is_float_dtype(series):
            # Float optimization
            if aggressive:
                # Check if can be converted to int
                if series.notna().all() and (series % 1 == 0).all():
                    return DataFrameOptimizer._optimize_numeric_column(
                        series.astype(int), aggressive
                    )
            
            # Use float32 if precision allows
            if series.max() < 3.4e38 and series.min() > -3.4e38:
                return series.astype(np.float32)
        
        return series
    
    @staticmethod
    def _optimize_string_column(series: pd.Series, aggressive: bool = False) -> pd.Series:
        """Optimize string column data type"""
        if series.dtype == 'object':
            # Check if all values are strings
            non_null_series = series.dropna()
            if non_null_series.empty:
                return series
            
            # Convert to category if few unique values
            unique_count = non_null_series.nunique()
            total_count = len(non_null_series)
            
            if unique_count / total_count < 0.5:  # Less than 50% unique
                return series.astype('category')
            
            # Use string dtype if available (pandas >= 1.0)
            if hasattr(pd, 'StringDtype'):
                try:
                    return series.astype('string')
                except:
                    pass
        
        return series
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        memory_usage = df.memory_usage(deep=True)
        total_mb = memory_usage.sum() / 1024 / 1024
        
        column_info = []
        for col in df.columns:
            col_memory = memory_usage[col] / 1024 / 1024
            column_info.append({
                'column': col,
                'dtype': str(df[col].dtype),
                'memory_mb': round(col_memory, 3),
                'memory_percent': round(col_memory / total_mb * 100, 1),
                'null_count': df[col].isnull().sum(),
                'unique_count': df[col].nunique()
            })
        
        # Sort by memory usage
        column_info.sort(key=lambda x: x['memory_mb'], reverse=True)
        
        return {
            'total_memory_mb': round(total_mb, 2),
            'shape': df.shape,
            'columns': column_info
        }

def memory_optimizer(func):
    """Decorator to optimize memory usage around function calls"""
    def wrapper(*args, **kwargs):
        # Force garbage collection before
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        try:
            result = func(*args, **kwargs)
            
            # Optimize result if it's a DataFrame
            if isinstance(result, pd.DataFrame):
                result = DataFrameOptimizer.optimize_dtypes(result)
            
            return result
        finally:
            # Force garbage collection after
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.debug(f"Memory usage: {func.__name__} {initial_memory:.1f}MB → {final_memory:.1f}MB ({final_memory-initial_memory:+.1f}MB)")
    
    return wrapper

def optimize_dataframe_for_processing(df: pd.DataFrame) -> pd.DataFrame:
    """Optimize DataFrame specifically for processing operations"""
    df_opt = df.copy()
    
    # Identify and optimize common patterns
    for col in df_opt.columns:
        series = df_opt[col]
        
        # Boolean-like columns
        if series.dtype == 'object':
            unique_vals = set(series.dropna().unique())
            if unique_vals.issubset({True, False, 'True', 'False', 1, 0, '1', '0'}):
                df_opt[col] = series.astype(bool)
                continue
        
        # ID columns (high cardinality integers)
        if 'id' in col.lower() and pd.api.types.is_integer_dtype(series):
            if series.nunique() / len(series) > 0.9:
                df_opt[col] = series.astype('category')
                continue
        
        # Price/fee columns (convert to float32 if possible)
        if any(keyword in col.lower() for keyword in ['fee', 'price', 'cost']):
            if pd.api.types.is_numeric_dtype(series):
                df_opt[col] = series.astype(np.float32)
                continue
    
    return DataFrameOptimizer.optimize_dtypes(df_opt, aggressive=True)

def get_memory_recommendations(df: pd.DataFrame) -> List[str]:
    """Get memory optimization recommendations"""
    recommendations = []
    memory_info = DataFrameOptimizer.get_memory_usage(df)
    
    # Check for high-memory columns
    high_memory_cols = [
        col for col in memory_info['columns'] 
        if col['memory_mb'] > 50  # > 50MB
    ]
    
    for col_info in high_memory_cols:
        col_name = col_info['column']
        dtype = col_info['dtype']
        
        if dtype == 'object':
            unique_ratio = col_info['unique_count'] / df.shape[0]
            if unique_ratio < 0.1:
                recommendations.append(f"Convert '{col_name}' to category (only {unique_ratio:.1%} unique values)")
        
        elif 'int64' in dtype:
            recommendations.append(f"Consider smaller integer type for '{col_name}'")
        
        elif 'float64' in dtype:
            recommendations.append(f"Consider float32 for '{col_name}' if precision allows")
    
    # Check overall memory usage
    total_mb = memory_info['total_memory_mb']
    if total_mb > 1000:  # > 1GB
        recommendations.append(f"Large DataFrame ({total_mb:.0f}MB) - consider chunked processing")
    
    return recommendations

class MemoryMonitor:
    """Memory monitoring context manager"""
    
    def __init__(self, operation_name: str = "operation"):
        self.operation_name = operation_name
        self.initial_memory = 0
        self.peak_memory = 0
    
    def __enter__(self):
        gc.collect()
        self.initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = self.initial_memory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        
        logger.info(
            f"Memory Monitor [{self.operation_name}]: "
            f"{self.initial_memory:.1f}MB → {final_memory:.1f}MB "
            f"(Peak: {self.peak_memory:.1f}MB, Delta: {final_memory-self.initial_memory:+.1f}MB)"
        )
    
    def update_peak(self):
        """Update peak memory usage"""
        current_memory = psutil.Process().memory_info().rss / 1024 / 1024
        self.peak_memory = max(self.peak_memory, current_memory) 