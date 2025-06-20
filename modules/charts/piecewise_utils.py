"""
Piecewise Utils Module

This module contains utility functions for piecewise linear regression and change point detection.
"""

import numpy as np
import logging
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

def detect_change_points(feature_values, costs, min_segment_size=3):
    """
    Detect change points in cost structure using slope analysis.
    
    Args:
        feature_values: Array of feature values (sorted)
        costs: Array of corresponding costs
        min_segment_size: Minimum points per segment
        
    Returns:
        List of change point indices
    """
    if len(feature_values) < min_segment_size * 2:
        return []
    
    # Calculate local slopes using moving windows
    slopes = []
    for i in range(len(feature_values) - 1):
        if feature_values[i+1] != feature_values[i]:  # Avoid division by zero
            slope = (costs[i+1] - costs[i]) / (feature_values[i+1] - feature_values[i])
            slopes.append(slope)
        else:
            slopes.append(0)
    
    if len(slopes) < min_segment_size:
        return []
    
    # Find significant slope changes
    change_points = []
    window_size = max(2, min_segment_size // 2)
    
    for i in range(window_size, len(slopes) - window_size):
        # Calculate average slope before and after this point
        before_slope = np.mean(slopes[i-window_size:i])
        after_slope = np.mean(slopes[i:i+window_size])
        
        # Check if there's a significant change (>20% difference)
        if abs(before_slope) > 0 and abs(after_slope - before_slope) / abs(before_slope) > 0.2:
            change_points.append(i)
    
    # Merge nearby change points
    if len(change_points) > 1:
        filtered_points = [change_points[0]]
        for cp in change_points[1:]:
            if cp - filtered_points[-1] >= min_segment_size:
                filtered_points.append(cp)
        change_points = filtered_points
    
    return change_points

def fit_piecewise_linear(feature_values, costs, change_points):
    """
    Fit piecewise linear model with detected change points.
    
    Args:
        feature_values: Array of feature values
        costs: Array of costs
        change_points: List of change point indices
        
    Returns:
        Dictionary with segment information
    """
    segments = []
    
    # Create segments based on change points
    start_idx = 0
    segment_boundaries = change_points + [len(feature_values)]
    
    for end_idx in segment_boundaries:
        if end_idx <= start_idx:
            continue
            
        # Extract segment data
        seg_features = feature_values[start_idx:end_idx]
        seg_costs = costs[start_idx:end_idx]
        
        if len(seg_features) < 2:
            start_idx = end_idx
            continue
        
        # Fit linear regression for this segment
        try:
            # Calculate slope and intercept
            x_mean = np.mean(seg_features)
            y_mean = np.mean(seg_costs)
            
            numerator = np.sum((seg_features - x_mean) * (seg_costs - y_mean))
            denominator = np.sum((seg_features - x_mean) ** 2)
            
            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
            else:
                slope = 0
                intercept = y_mean
            
            segments.append({
                'start_feature': float(seg_features[0]),
                'end_feature': float(seg_features[-1]),
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'slope': float(slope),
                'intercept': float(intercept),
                'points': len(seg_features),
                'marginal_cost': float(slope)  # Slope is the marginal cost
            })
            
        except Exception as e:
            logger.warning(f"Error fitting segment {start_idx}-{end_idx}: {e}")
        
        start_idx = end_idx
    
    return segments

def fit_piecewise_linear_segments(feature_values, costs, change_points):
    """
    Fit piecewise linear segments with proper marginal rates for each segment.
    Each segment has its own marginal rate (cost per unit) calculated from the data points in that segment.
    
    Args:
        feature_values: Array of feature values (sorted)
        costs: Array of corresponding costs
        change_points: List of change point indices
        
    Returns:
        List of segment dictionaries with marginal rates for each segment
    """
    if len(feature_values) == 0:
        return []
    
    segments = []
    segment_starts = [0] + change_points + [len(feature_values)]
    
    for i in range(len(segment_starts) - 1):
        start_idx = segment_starts[i]
        end_idx = segment_starts[i + 1]
        
        if start_idx == end_idx:
            continue
            
        # Get segment data
        seg_features = feature_values[start_idx:end_idx]
        seg_costs = costs[start_idx:end_idx]
        
        if len(seg_features) < 2:
            continue
        
        # Calculate marginal rate for this segment using ROBUST method
        X = seg_features.reshape(-1, 1)
        y = seg_costs
        
        # Calculate simple slope as primary method (more robust)
        feature_diff = seg_features[-1] - seg_features[0]
        cost_diff = seg_costs[-1] - seg_costs[0]
        
        if feature_diff == 0 or len(seg_features) < 2:
            segment_marginal_rate = 0
            segment_intercept = seg_costs[0] if len(seg_costs) > 0 else 0
        else:
            # Use simple slope calculation (more robust than LinearRegression for small segments)
            segment_marginal_rate = cost_diff / feature_diff
            segment_intercept = seg_costs[0] - segment_marginal_rate * seg_features[0]
            
            # Validate with LinearRegression only if we have enough points and reasonable slope
            if len(seg_features) >= 3 and abs(segment_marginal_rate) < 100000:  # Sanity check
                try:
                    reg = LinearRegression().fit(X, y)
                    lr_rate = reg.coef_[0]
                    lr_intercept = reg.intercept_
                    
                    # Use LinearRegression result only if it's reasonable
                    if abs(lr_rate) < 100000 and abs(lr_rate - segment_marginal_rate) / max(abs(segment_marginal_rate), 1) < 2.0:
                        segment_marginal_rate = lr_rate
                        segment_intercept = lr_intercept
                except:
                    # Keep simple slope calculation
                    pass
        
        # Ensure positive marginal rate (negative rates don't make economic sense)
        if segment_marginal_rate < 0:
            logger.warning(f"Negative marginal rate in segment {i}: {segment_marginal_rate:.2f}, setting to 0")
            segment_marginal_rate = 0
            
        # Cap extremely high rates (likely calculation errors)
        if segment_marginal_rate > 50000:  # ₩50,000/unit is unrealistic
            logger.warning(f"Extremely high marginal rate in segment {i}: ₩{segment_marginal_rate:.0f}, capping to ₩10,000")
            segment_marginal_rate = 10000
        
        # Calculate segment endpoints
        start_feature = seg_features[0]
        end_feature = seg_features[-1]
        start_cost = seg_costs[0]
        end_cost = seg_costs[-1]
        
        segment = {
            'start_feature': float(start_feature),
            'end_feature': float(end_feature),
            'marginal_rate': float(segment_marginal_rate),
            'intercept': float(segment_intercept),
            'start_cost': float(start_cost),
            'end_cost': float(end_cost),
            'data_points': len(seg_features),
            'segment_index': i
        }
        
        segments.append(segment)
    
    return segments 