#!/usr/bin/env python3
"""
Test script for TRUE cumulative marginal cost calculation with ALL FEATURES COMBINED.

This implements multi-dimensional cumulative pricing where:
- All features (data, voice, messages, tethering) are considered together
- Each feature has its own incremental segments
- Total cost = sum of all feature segment costs
- Uses entire dataset with monotonicity and 1KRW rule

Example for a plan with 5GB data, 100min voice, 50 messages:
Total cost = data_cost(5GB) + voice_cost(100min) + message_cost(50) + base_cost
"""

import sys
import os
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, '/app')

from modules.cost_spec import UNLIMITED_FLAGS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_processed_data():
    """Load processed data."""
    data_path = Path('/app/data/processed')
    csv_files = list(data_path.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError("No CSV data files found in /app/data/processed/")
    
    latest_file = max(csv_files, key=lambda f: f.stat().st_mtime)
    df = pd.read_csv(latest_file)
    logger.info(f"Loaded {len(df)} plans from {latest_file.name}")
    
    return df

def apply_monotonicity_and_1krw_rule(data_points):
    """
    Apply monotonicity and 1 KRW/unit rule to data points.
    
    Args:
        data_points: List of (feature_value, cost) tuples
        
    Returns:
        Filtered list of (feature_value, cost) tuples
    """
    if len(data_points) < 2:
        return data_points
    
    # Sort by feature value
    sorted_points = sorted(data_points, key=lambda x: x[0])
    
    # Apply monotonicity and 1 KRW rule
    filtered_points = [sorted_points[0]]  # Always include first point
    
    for feature_val, cost in sorted_points[1:]:
        last_feature, last_cost = filtered_points[-1]
        
        # Skip if feature value doesn't increase
        if feature_val <= last_feature:
            continue
            
        # Skip if cost doesn't increase
        if cost <= last_cost:
            continue
            
        # Check 1 KRW/unit rule
        cost_per_unit = (cost - last_cost) / (feature_val - last_feature)
        if cost_per_unit >= 1.0:
            filtered_points.append((feature_val, cost))
        # If rule violated, skip this point
    
    return filtered_points

def calculate_feature_segments(df, feature):
    """
    Calculate incremental segments for a single feature using ALL data.
    
    Args:
        df: DataFrame with plan data
        feature: Feature column name
        
    Returns:
        List of segment dictionaries with incremental rates
    """
    logger.info(f"Calculating segments for {feature}")
    
    # Handle unlimited plans
    unlimited_flag = UNLIMITED_FLAGS.get(feature)
    if unlimited_flag and unlimited_flag in df.columns:
        df_limited = df[(df[unlimited_flag] == 0) & df['original_fee'].notna()].copy()
    else:
        df_limited = df[df['original_fee'].notna()].copy()
    
    if df_limited.empty:
        logger.warning(f"No data available for {feature}")
        return []
    
    # Get ALL unique feature values
    unique_values = sorted(df_limited[feature].unique())
    
    # For each unique value, find the cheapest plan
    all_points = []
    for val in unique_values:
        matching_plans = df_limited[df_limited[feature] == val]
        if not matching_plans.empty:
            min_cost = matching_plans['original_fee'].min()
            all_points.append((val, min_cost))
    
    # Apply monotonicity and 1 KRW rule
    filtered_points = apply_monotonicity_and_1krw_rule(all_points)
    
    logger.info(f"{feature}: {len(unique_values)} unique values → {len(filtered_points)} filtered points")
    
    if len(filtered_points) < 2:
        return []
    
    # Create incremental segments
    segments = []
    
    for i in range(len(filtered_points) - 1):
        start_feature, start_cost = filtered_points[i]
        end_feature, end_cost = filtered_points[i + 1]
        
        feature_range = end_feature - start_feature
        cost_range = end_cost - start_cost
        
        # Incremental rate for this specific segment
        incremental_rate = cost_range / feature_range if feature_range > 0 else 0
        
        segment = {
            'segment_index': i,
            'start_feature': start_feature,
            'end_feature': end_feature,
            'feature_range': feature_range,
            'cost_range': cost_range,
            'incremental_rate': incremental_rate,
            'start_cost': start_cost,
            'end_cost': end_cost
        }
        
        segments.append(segment)
    
    logger.info(f"{feature}: Created {len(segments)} segments")
    return segments

def calculate_multi_feature_segments(df, features):
    """
    Calculate incremental segments for ALL features combined.
    
    Args:
        df: DataFrame with plan data
        features: List of feature column names
        
    Returns:
        Dictionary with segments for each feature
    """
    logger.info(f"Calculating multi-feature segments for: {features}")
    
    all_segments = {}
    
    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in data")
            continue
            
        segments = calculate_feature_segments(df, feature)
        if segments:
            all_segments[feature] = segments
        else:
            logger.warning(f"No segments created for {feature}")
    
    logger.info(f"Successfully created segments for {len(all_segments)} features")
    return all_segments

def calculate_feature_cost(feature_value, segments):
    """
    Calculate cumulative cost for a single feature value using segment-based pricing.
    
    Args:
        feature_value: Target feature value
        segments: List of segment dictionaries for this feature
        
    Returns:
        Dictionary with cost breakdown
    """
    if not segments or feature_value <= 0:
        return {'total_cost': 0, 'breakdown': []}
    
    total_cost = 0
    breakdown = []
    
    # Start from the beginning
    current_position = segments[0]['start_feature']
    
    for segment in segments:
        start_feat = segment['start_feature']
        end_feat = segment['end_feature']
        rate = segment['incremental_rate']
        
        # Skip segments that are before our current position
        if end_feat <= current_position:
            continue
            
        # Calculate how much of this segment we need
        segment_start = max(start_feat, current_position)
        segment_end = min(end_feat, feature_value)
        
        if segment_end > segment_start:
            segment_usage = segment_end - segment_start
            segment_cost = segment_usage * rate
            total_cost += segment_cost
            
            breakdown.append({
                'segment_index': segment['segment_index'],
                'range': f"{segment_start:.1f}-{segment_end:.1f}",
                'usage': segment_usage,
                'rate': rate,
                'cost': segment_cost
            })
            
            current_position = segment_end
        
        # Stop if we've reached our target
        if current_position >= feature_value:
            break
    
    return {
        'total_cost': total_cost,
        'breakdown': breakdown,
        'feature_value': feature_value
    }

def calculate_multi_feature_cost(feature_values, all_segments):
    """
    Calculate total cost for a plan with multiple features.
    
    Args:
        feature_values: Dictionary of {feature_name: value}
        all_segments: Dictionary of {feature_name: segments}
        
    Returns:
        Dictionary with complete cost breakdown
    """
    total_cost = 0
    feature_costs = {}
    
    for feature, value in feature_values.items():
        if feature in all_segments and value > 0:
            feature_result = calculate_feature_cost(value, all_segments[feature])
            feature_costs[feature] = feature_result
            total_cost += feature_result['total_cost']
        else:
            feature_costs[feature] = {'total_cost': 0, 'breakdown': [], 'feature_value': value}
    
    return {
        'total_cost': total_cost,
        'feature_costs': feature_costs,
        'feature_values': feature_values
    }

def test_multi_feature_examples(all_segments):
    """
    Test multi-feature cumulative pricing with realistic examples.
    
    Args:
        all_segments: Dictionary of segments for each feature
    """
    logger.info("\n=== MULTI-FEATURE CUMULATIVE PRICING EXAMPLES ===")
    
    # Define test cases with realistic feature combinations
    test_cases = [
        {
            'name': 'Basic Plan',
            'basic_data_clean': 1.0,
            'voice_clean': 50,
            'message_clean': 30,
            'tethering_gb': 0
        },
        {
            'name': 'Standard Plan', 
            'basic_data_clean': 5.0,
            'voice_clean': 100,
            'message_clean': 100,
            'tethering_gb': 1.0
        },
        {
            'name': 'Premium Plan',
            'basic_data_clean': 20.0,
            'voice_clean': 300,
            'message_clean': 300,
            'tethering_gb': 5.0
        },
        {
            'name': 'Data-Heavy Plan',
            'basic_data_clean': 50.0,
            'voice_clean': 30,
            'message_clean': 50,
            'tethering_gb': 10.0
        }
    ]
    
    for test_case in test_cases:
        plan_name = test_case.pop('name')
        result = calculate_multi_feature_cost(test_case, all_segments)
        
        logger.info(f"\n📱 {plan_name}:")
        logger.info(f"   Features: {test_case}")
        logger.info(f"   💰 Total Cost: ₩{result['total_cost']:,.0f}")
        
        logger.info("   📊 Feature Breakdown:")
        for feature, cost_info in result['feature_costs'].items():
            if cost_info['total_cost'] > 0:
                logger.info(f"     {feature}: ₩{cost_info['total_cost']:,.0f}")
                for item in cost_info['breakdown'][:3]:  # Show first 3 segments
                    logger.info(f"       {item['range']}: {item['usage']:.1f} × ₩{item['rate']:,.0f} = ₩{item['cost']:,.0f}")
                if len(cost_info['breakdown']) > 3:
                    logger.info(f"       ... and {len(cost_info['breakdown'])-3} more segments")

def analyze_average_rates(all_segments):
    """
    Analyze average rates across all features.
    
    Args:
        all_segments: Dictionary of segments for each feature
        
    Returns:
        Dictionary with analysis results
    """
    logger.info(f"\n=== MULTI-FEATURE AVERAGE RATE ANALYSIS ===")
    
    analysis = {}
    
    for feature, segments in all_segments.items():
        if not segments:
            continue
            
        rates = [seg['incremental_rate'] for seg in segments]
        feature_ranges = [seg['feature_range'] for seg in segments]
        
        total_range = sum(feature_ranges)
        weighted_avg = sum(rate * range_size for rate, range_size in zip(rates, feature_ranges)) / total_range
        
        feature_analysis = {
            'num_segments': len(segments),
            'min_rate': min(rates),
            'max_rate': max(rates),
            'weighted_average_rate': weighted_avg,
            'median_rate': np.median(rates),
            'std_rate': np.std(rates)
        }
        
        analysis[feature] = feature_analysis
        
        logger.info(f"\n{feature}:")
        logger.info(f"  Segments: {feature_analysis['num_segments']}")
        logger.info(f"  Rate range: ₩{feature_analysis['min_rate']:,.0f} - ₩{feature_analysis['max_rate']:,.0f}")
        logger.info(f"  Weighted avg: ₩{feature_analysis['weighted_average_rate']:,.0f}/unit")
        logger.info(f"  Std dev: ₩{feature_analysis['std_rate']:,.0f}")
    
    return analysis

def main():
    """Run the multi-feature cumulative pricing test."""
    logger.info("Starting MULTI-FEATURE cumulative marginal cost test...")
    
    try:
        # Load data
        df = load_processed_data()
        
        # Define features to analyze
        features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
        
        # Check which features are available
        available_features = [f for f in features if f in df.columns]
        logger.info(f"Available features: {available_features}")
        
        if len(available_features) < 2:
            logger.error("Insufficient features available")
            return
        
        # Calculate segments for all features
        all_segments = calculate_multi_feature_segments(df, available_features)
        
        if not all_segments:
            logger.error("No segments created for any feature")
            return
        
        # Test multi-feature pricing examples
        test_multi_feature_examples(all_segments)
        
        # Analyze average rates
        analysis = analyze_average_rates(all_segments)
        
        logger.info("\n✅ MULTI-FEATURE cumulative marginal cost test completed successfully!")
        
        # Summary
        logger.info(f"\n📋 SUMMARY:")
        logger.info(f"• Total plans in dataset: {len(df)}")
        logger.info(f"• Features analyzed: {len(all_segments)}")
        for feature, segments in all_segments.items():
            logger.info(f"• {feature}: {len(segments)} segments")
        
        # Overall average
        all_weighted_avgs = [analysis[f]['weighted_average_rate'] for f in analysis.keys()]
        overall_avg = np.mean(all_weighted_avgs)
        logger.info(f"• Overall average rate: ₩{overall_avg:,.0f}/unit")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 