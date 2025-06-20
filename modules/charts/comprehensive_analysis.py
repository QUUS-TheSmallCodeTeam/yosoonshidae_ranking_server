"""
Comprehensive Analysis Module

Contains comprehensive granular marginal cost frontier analysis functions.
Extracted from marginal_cost.py for better modularity.

Functions:
- prepare_granular_marginal_cost_frontier_data: Comprehensive granular analysis using entire dataset
"""

import logging
import numpy as np
import pandas as pd
from .piecewise_utils import detect_change_points, fit_piecewise_linear_segments

# Configure logging
logger = logging.getLogger(__name__)

def prepare_granular_marginal_cost_frontier_data(df, multi_frontier_breakdown, core_continuous_features):
    """
    Prepare COMPREHENSIVE GRANULAR marginal cost frontier charts using ENTIRE DATASET.
    Uses full dataset regression instead of frontier points for more comprehensive analysis.
    
    Args:
        df: DataFrame with plan data
        multi_frontier_breakdown: Multi-frontier regression results (optional, for fallback coefficients)
        core_continuous_features: List of features to visualize
        
    Returns:
        Dictionary with comprehensive granular marginal cost frontier data including intercepts
    """
    logger.info("Preparing COMPREHENSIVE GRANULAR marginal cost frontier charts using ENTIRE DATASET")
    logger.info(f"Processing ALL features: {core_continuous_features}")
    
    from modules.cost_spec import UNLIMITED_FLAGS, FullDatasetMultiFeatureRegression
    
    # Feature display configuration
    feature_display_names = {
        'basic_data_clean': 'Data (GB)',
        'voice_clean': 'Voice (min)', 
        'message_clean': 'Messages',
        'tethering_gb': 'Tethering (GB)',
        'is_5g': '5G Support'
    }
    
    feature_units = {
        'basic_data_clean': 'KRW/GB',
        'voice_clean': 'KRW/min',
        'message_clean': 'KRW/msg', 
        'tethering_gb': 'KRW/GB',
        'is_5g': 'KRW/feature'
    }
    
    # Use ENTIRE DATASET for regression analysis
    logger.info(f"Using ENTIRE DATASET: {len(df)} plans for comprehensive analysis")
    
    # Prepare clean dataset for regression
    clean_df = df.copy()
    
    # Remove plans with missing essential data
    essential_columns = ['original_fee'] + [f for f in core_continuous_features if f in df.columns]
    clean_df = clean_df.dropna(subset=essential_columns)
    
    logger.info(f"Clean dataset: {len(clean_df)} plans after removing missing data")
    
    if len(clean_df) < 50:
        logger.error(f"Insufficient data for full dataset analysis: {len(clean_df)} plans")
        return {}
    
    # Run full dataset multi-feature regression
    try:
        full_dataset_analyzer = FullDatasetMultiFeatureRegression()
        coefficients = full_dataset_analyzer.solve_full_dataset_coefficients(clean_df)
        full_dataset_result = full_dataset_analyzer.get_coefficient_breakdown()
        
        if not full_dataset_result or 'feature_costs' not in full_dataset_result:
            logger.error("Full dataset regression failed")
            return {}
            
        logger.info(f"✅ Full dataset regression successful with {len(clean_df)} plans")
        
        # Extract coefficients from full dataset analysis
        feature_costs = full_dataset_result['feature_costs']
        base_cost = full_dataset_result.get('base_cost', 0)
        
        logger.info(f"Full dataset base cost: ₩{base_cost:,.0f}")
        for feature, cost_info in feature_costs.items():
            coeff = cost_info.get('coefficient', 0)
            logger.info(f"Full dataset {feature}: ₩{coeff:.2f} per unit")
            
    except Exception as e:
        logger.error(f"Full dataset regression failed: {str(e)}")
        # Fallback to provided multi_frontier_breakdown if available
        if multi_frontier_breakdown and multi_frontier_breakdown.get('feature_costs'):
            logger.info("Falling back to provided multi-frontier breakdown")
            feature_costs = multi_frontier_breakdown['feature_costs']
            base_cost = multi_frontier_breakdown.get('base_cost', 0)
        else:
            logger.error("No fallback data available")
            return {}
    
    granular_frontier_data = {}
    features_processed = 0
    
    # Process each feature using full dataset coefficients
    for feature in core_continuous_features:
        if feature not in clean_df.columns or feature not in feature_costs:
            logger.warning(f"Feature {feature} not available for analysis")
            continue
            
        # Get coefficient from full dataset analysis
        cost_info = feature_costs[feature]
        pure_coefficient = cost_info.get('coefficient', 0)
        
        logger.info(f"Processing {feature} with full dataset coefficient: ₩{pure_coefficient:.2f}")
        
        # Get feature range from entire dataset
        feature_values = clean_df[feature].dropna()
        if feature_values.empty:
            continue
            
        min_val = float(feature_values.min())
        max_val = float(feature_values.max())
        unique_values = sorted(feature_values.unique())
        
        logger.info(f"Feature {feature} range: {min_val} to {max_val} ({len(unique_values)} unique values)")
        
        # Handle unlimited plans separately
        unlimited_flag = UNLIMITED_FLAGS.get(feature)
        unlimited_info = None
        
        if unlimited_flag and unlimited_flag in clean_df.columns:
            unlimited_plans = clean_df[(clean_df[unlimited_flag] == 1) & clean_df['original_fee'].notna()]
            if not unlimited_plans.empty:
                min_unlimited_cost = unlimited_plans['original_fee'].min()
                unlimited_plan_name = unlimited_plans.loc[unlimited_plans['original_fee'].idxmin()].get('plan_name', 'Unknown')
                unlimited_info = {
                    'has_unlimited': True,
                    'min_cost': float(min_unlimited_cost),
                    'plan_name': unlimited_plan_name,
                    'count': len(unlimited_plans)
                }
                logger.info(f"Found {len(unlimited_plans)} unlimited {feature} plans, cheapest: ₩{min_unlimited_cost}")
        
        # Create piecewise analysis using actual market data points
        # Get actual plans at each feature level for piecewise analysis
        feature_level_data = []
        for val in unique_values:
            matching_plans = clean_df[clean_df[feature] == val]
            if not matching_plans.empty:
                costs = matching_plans['original_fee'].values
                min_cost = costs.min()
                avg_cost = costs.mean()
                plan_count = len(costs)
                
                feature_level_data.append({
                    'feature_value': float(val),
                    'min_cost': float(min_cost),
                    'avg_cost': float(avg_cost),
                    'plan_count': int(plan_count),
                    'actual_costs': costs.tolist()
                })
        
        logger.info(f"Collected data for {len(feature_level_data)} feature levels for {feature}")
        
        # Calculate piecewise segments based on actual market data
        if len(feature_level_data) >= 3:
            # Extract features and costs for piecewise analysis
            features_array = np.array([point['feature_value'] for point in feature_level_data])
            costs_array = np.array([point['min_cost'] for point in feature_level_data])
            
            # Detect change points and fit piecewise linear segments
            change_points = detect_change_points(features_array, costs_array)
            segments = fit_piecewise_linear_segments(features_array, costs_array, change_points)
            
            logger.info(f"Created {len(segments)} piecewise segments for {feature}")
            
            # Create frontier points for visualization using PROPER PIECEWISE calculation
            frontier_points = []
            for i, point_data in enumerate(feature_level_data):
                feature_val = point_data['feature_value']
                actual_cost = point_data['min_cost']
                
                # Find which segment this point belongs to and calculate CUMULATIVE cost
                segment_marginal_cost = pure_coefficient  # fallback
                segment_info = "linear"
                cumulative_cost = base_cost + (feature_val * pure_coefficient)  # fallback
                
                # Calculate PROPER cumulative cost using piecewise segments
                if segments:
                    total_feature_cost = 0
                    current_feature_position = 0
                    
                    # Find which segment this feature value falls into
                    target_segment = None
                    for seg_idx, segment in enumerate(segments):
                        if segment['start_feature'] <= feature_val <= segment['end_feature']:
                            target_segment = segment
                            segment_marginal_cost = segment['marginal_rate']
                            segment_info = f"Segment {seg_idx+1}: {segment['start_feature']:.1f}-{segment['end_feature']:.1f}"
                            break
                    
                    # Calculate PROPER cumulative cost by accumulating through segments
                    current_position = 0
                    
                    for seg_idx, segment in enumerate(segments):
                        segment_start = segment['start_feature']
                        segment_end = segment['end_feature']
                        segment_rate = segment['marginal_rate']
                        
                        # Calculate the actual usage range within this segment
                        usage_start = max(segment_start, current_position)
                        usage_end = min(segment_end, feature_val)
                        
                        if usage_end > usage_start:
                            # We use part or all of this segment
                            usage_in_segment = usage_end - usage_start
                            segment_contribution = usage_in_segment * segment_rate
                            total_feature_cost += segment_contribution
                            current_position = usage_end
                            
                            logger.debug(f"Segment {seg_idx+1}: {usage_start:.1f}-{usage_end:.1f} GB × ₩{segment_rate:.0f} = ₩{segment_contribution:.0f}")
                        
                        if feature_val <= segment_end:
                            break
                    
                    cumulative_cost = base_cost + total_feature_cost
                
                frontier_points.append({
                    'feature_value': feature_val,
                    'pure_cost': float(cumulative_cost),
                    'marginal_cost': float(segment_marginal_cost),
                    'cumulative_cost': float(cumulative_cost),
                    'segment': segment_info,
                    'actual_market_cost': float(actual_cost),
                    'plan_count': point_data['plan_count']
                })
        else:
            # Linear fallback for insufficient data
            logger.warning(f"Insufficient data for piecewise analysis of {feature}, using linear model")
            frontier_points = []
            for point_data in feature_level_data:
                feature_val = point_data['feature_value']
                cumulative_cost = base_cost + (feature_val * pure_coefficient)
                
                frontier_points.append({
                    'feature_value': feature_val,
                    'pure_cost': float(cumulative_cost),
                    'marginal_cost': float(pure_coefficient),
                    'cumulative_cost': float(cumulative_cost),
                    'segment': 'linear',
                    'actual_market_cost': float(point_data['min_cost']),
                    'plan_count': point_data['plan_count']
                })
            
            segments = []
        
        # Store comprehensive granular data
        granular_frontier_data[feature] = {
            'display_name': feature_display_names.get(feature, feature),
            'unit': feature_units.get(feature, 'KRW/unit'),
            'frontier_points': frontier_points,
            'unlimited_info': unlimited_info,
            'base_coefficient': float(pure_coefficient),
            'segments_count': len(segments),
            'feature_name': feature,
            'data_source': 'full_dataset_regression',
            'total_plans_analyzed': len(clean_df),
            'feature_levels_analyzed': len(feature_level_data)
        }
        
        features_processed += 1
        logger.info(f"✅ Processed {feature}: {len(frontier_points)} points, {len(segments)} segments")
    
    logger.info(f"✅ Completed comprehensive granular analysis for {features_processed} features")
    
    return {
        'granular_frontier_data': granular_frontier_data,
        'analysis_summary': {
            'method': 'comprehensive_full_dataset_analysis',
            'total_features_processed': features_processed,
            'total_plans_analyzed': len(clean_df),
            'base_cost': base_cost,
            'data_source': 'full_dataset_regression'
        }
    } 