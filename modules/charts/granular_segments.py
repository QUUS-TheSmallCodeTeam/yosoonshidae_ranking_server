"""
Granular Segments Module

Contains granular segment creation and calculation functions.
Extracted from marginal_cost.py for better modularity.

Functions:
- create_granular_segments_with_intercepts: Create granular segments with unlimited intercepts
- calculate_granular_piecewise_cost_with_intercepts: Calculate costs using granular segments
"""

import logging
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

def create_granular_segments_with_intercepts(df, feature, unlimited_flag=None):
    """
    Create granular segments with one segment per feature value change.
    Also calculate unlimited intercept coefficient if applicable.
    
    Args:
        df: DataFrame with plan data
        feature: Continuous feature name
        unlimited_flag: Corresponding unlimited flag name (optional)
        
    Returns:
        Dict with granular segments and unlimited intercept info
    """
    logger.info(f"Creating GRANULAR segments for {feature} (one per value change)")
    
    # Process non-unlimited plans for continuous segments
    if unlimited_flag and unlimited_flag in df.columns:
        df_limited = df[(df[unlimited_flag] == 0) & df['original_fee'].notna()].copy()
    else:
        df_limited = df[df['original_fee'].notna()].copy()
    
    if df_limited.empty:
        logger.warning(f"No limited plans found for {feature}")
        return {'segments': [], 'unlimited_intercept': None}
    
    # Get all unique feature values and their minimum costs
    unique_vals = sorted(df_limited[feature].unique())
    frontier_points = []
    
    for val in unique_vals:
        matching_plans = df_limited[df_limited[feature] == val]
        if not matching_plans.empty:
            min_cost = matching_plans['original_fee'].min()
            frontier_points.append((val, min_cost))
    
    logger.info(f"Found {len(frontier_points)} unique feature values for {feature}")
    
    # Apply basic monotonicity (remove decreasing costs only)
    filtered_points = []
    for i, (feat_val, cost) in enumerate(frontier_points):
        if i == 0:
            filtered_points.append((feat_val, cost))
            continue
            
        prev_feat, prev_cost = filtered_points[-1]
        
        # Only remove if cost decreases with more features (basic economics violation)
        if feat_val > prev_feat and cost >= prev_cost:
            filtered_points.append((feat_val, cost))
        elif feat_val > prev_feat and cost < prev_cost:
            # Skip this point - it violates basic economics
            logger.debug(f"Skipping decreasing cost point for {feature}: {feat_val} at ₩{cost} (prev: {prev_feat} at ₩{prev_cost})")
            continue
        else:
            filtered_points.append((feat_val, cost))
    
    logger.info(f"After basic monotonicity: {len(filtered_points)} points for {feature}")
    
    # Create granular segments (one per consecutive value pair)
    segments = []
    for i in range(len(filtered_points) - 1):
        start_val, start_cost = filtered_points[i]
        end_val, end_cost = filtered_points[i + 1]
        
        # Calculate marginal rate for this specific segment
        if end_val > start_val:
            marginal_rate = (end_cost - start_cost) / (end_val - start_val)
        else:
            marginal_rate = 0
        
        segments.append({
            'start_feature': float(start_val),
            'end_feature': float(end_val),
            'start_cost': float(start_cost),
            'end_cost': float(end_cost),
            'marginal_rate': float(marginal_rate),
            'segment_id': i,
            'feature_range': f"{start_val}-{end_val}",
            'cost_range': f"₩{start_cost:,.0f}-₩{end_cost:,.0f}"
        })
    
    logger.info(f"Created {len(segments)} granular segments for {feature}")
    
    # Calculate unlimited intercept coefficient
    unlimited_intercept = None
    if unlimited_flag and unlimited_flag in df.columns:
        unlimited_plans = df[(df[unlimited_flag] == 1) & df['original_fee'].notna()]
        if not unlimited_plans.empty:
            # Calculate unlimited premium as intercept
            avg_unlimited_cost = unlimited_plans['original_fee'].mean()
            
            # Compare to similar limited plans (at high feature levels)
            if filtered_points:
                max_limited_feature = max(point[0] for point in filtered_points)
                high_feature_plans = df_limited[df_limited[feature] >= max_limited_feature * 0.8]
                
                if not high_feature_plans.empty:
                    avg_high_limited_cost = high_feature_plans['original_fee'].mean()
                    unlimited_premium = avg_unlimited_cost - avg_high_limited_cost
                    
                    unlimited_intercept = {
                        'coefficient': float(unlimited_premium),
                        'unlimited_avg_cost': float(avg_unlimited_cost),
                        'limited_avg_cost': float(avg_high_limited_cost),
                        'premium': float(unlimited_premium),
                        'unlimited_count': len(unlimited_plans),
                        'flag_name': unlimited_flag
                    }
                    
                    logger.info(f"Calculated unlimited intercept for {feature}: ₩{unlimited_premium:,.0f} premium")
    
    return {
        'segments': segments,
        'unlimited_intercept': unlimited_intercept,
        'total_segments': len(segments),
        'feature_name': feature
    }

def calculate_granular_piecewise_cost_with_intercepts(feature_values, unlimited_flags, 
                                                    granular_segments, unlimited_intercepts):
    """
    Calculate total cost using granular piecewise segments plus unlimited intercepts.
    
    Args:
        feature_values: Dict of continuous feature values
        unlimited_flags: Dict of unlimited flag values (0 or 1)
        granular_segments: Dict of granular segments for each feature
        unlimited_intercepts: Dict of unlimited intercept coefficients
        
    Returns:
        Dict with detailed cost breakdown
    """
    total_cost = 0
    feature_costs = {}
    intercept_costs = {}
    
    # Calculate continuous feature costs using granular segments
    for feature, value in feature_values.items():
        if feature not in granular_segments or value <= 0:
            continue
            
        segments = granular_segments[feature]['segments']
        feature_cost = 0
        segment_details = []
        
        for segment in segments:
            start_feat = segment['start_feature']
            end_feat = segment['end_feature']
            rate = segment['marginal_rate']
            
            # Calculate usage within this segment
            if value > start_feat:
                segment_end = min(end_feat, value)
                segment_usage = segment_end - start_feat
                
                if segment_usage > 0:
                    segment_cost = segment_usage * rate
                    feature_cost += segment_cost
                    
                    segment_details.append({
                        'segment_id': segment['segment_id'],
                        'range': f"{start_feat}-{min(end_feat, value)}",
                        'usage': segment_usage,
                        'rate': rate,
                        'cost': segment_cost
                    })
        
        feature_costs[feature] = {
            'total_cost': feature_cost,
            'segments_used': segment_details,
            'feature_value': value
        }
        total_cost += feature_cost
    
    # Add unlimited intercept costs
    for flag, is_unlimited in unlimited_flags.items():
        if is_unlimited == 1 and flag in unlimited_intercepts:
            intercept_cost = unlimited_intercepts[flag]['coefficient']
            intercept_costs[flag] = {
                'intercept_cost': intercept_cost,
                'is_active': True,
                'flag_name': flag
            }
            total_cost += intercept_cost
        elif flag in unlimited_intercepts:
            intercept_costs[flag] = {
                'intercept_cost': 0,
                'is_active': False,
                'flag_name': flag
            }
    
    return {
        'total_cost': total_cost,
        'feature_costs': feature_costs,
        'intercept_costs': intercept_costs,
        'calculation_method': 'granular_piecewise_with_intercepts'
    } 