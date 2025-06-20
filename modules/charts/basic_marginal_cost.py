"""
Basic Marginal Cost Module

Contains basic marginal cost frontier data preparation functions.
Extracted from marginal_cost.py for better modularity.

Functions:
- prepare_marginal_cost_frontier_data: Basic piecewise linear marginal cost frontier charts
"""

import logging
import numpy as np
import pandas as pd
from .piecewise_utils import detect_change_points, fit_piecewise_linear_segments

# Configure logging
logger = logging.getLogger(__name__)

def prepare_marginal_cost_frontier_data(df, multi_frontier_breakdown, core_continuous_features):
    """
    Prepare feature frontier charts using PIECEWISE LINEAR model for realistic marginal costs.
    This shows economies of scale with different marginal costs across feature ranges.
    
    Args:
        df: DataFrame with plan data
        multi_frontier_breakdown: Pure marginal costs from MultiFeatureFrontierRegression
        core_continuous_features: List of features to visualize
        
    Returns:
        Dictionary with piecewise marginal cost frontier data for visualization
    """
    if not multi_frontier_breakdown or not multi_frontier_breakdown.get('feature_costs'):
        logger.warning("No multi-frontier breakdown data available for marginal cost frontiers")
        return {}

    logger.info("Preparing PIECEWISE LINEAR marginal cost frontier charts with economies of scale")
    
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
    
    marginal_cost_frontier_data = {}
    feature_costs = multi_frontier_breakdown.get('feature_costs', {})
    base_cost = multi_frontier_breakdown.get('base_cost', 0)
    
    for feature in core_continuous_features:
        if feature not in df.columns or feature not in feature_costs:
            logger.warning(f"Feature {feature} not available for marginal cost frontier")
            continue
            
        # Get pure marginal cost coefficient (as baseline)
        cost_info = feature_costs[feature]
        pure_coefficient = cost_info.get('coefficient', 0)
        
        if pure_coefficient <= 0:
            logger.warning(f"Invalid coefficient for {feature}: {pure_coefficient}")
            continue
            
        logger.info(f"Creating PIECEWISE LINEAR frontier for {feature} with base coefficient: {pure_coefficient}")
        
        # Get RAW MARKET DATA for actual plan points (same as traditional frontier)
        feature_values = df[feature].dropna()
        if feature_values.empty:
            continue
        
        from modules.cost_spec import UNLIMITED_FLAGS
        
        # Apply same unlimited handling as original system
        unlimited_flag = UNLIMITED_FLAGS.get(feature)
        
        # Process non-unlimited plans to get RAW cheapest points (no monotonic filtering yet!)
        if unlimited_flag and unlimited_flag in df.columns:
            df_non_unlimited = df[(df[unlimited_flag] == 0) & df['original_fee'].notna()].copy()
        else:
            df_non_unlimited = df[df['original_fee'].notna()].copy()
        
        if df_non_unlimited.empty:
            logger.warning(f"No non-unlimited plans found for {feature}")
            continue
        
        # Get ALL cheapest points at each feature level (same as traditional frontier!)
        unique_vals = sorted(df_non_unlimited[feature].unique())
        frontier_data_points = []
        
        for val in unique_vals:
            matching_plans = df_non_unlimited[df_non_unlimited[feature] == val]
            if not matching_plans.empty:
                min_cost = matching_plans['original_fee'].min()
                frontier_data_points.append((val, min_cost))
        
        logger.info(f"✅ Collected RAW market data for {feature}: {len(frontier_data_points)} cheapest points at each level")
        
        # Handle unlimited plans SEPARATELY as FLAG (not continuous data)
        unlimited_info = None
        if unlimited_flag and unlimited_flag in df.columns:
            unlimited_plans = df[(df[unlimited_flag] == 1) & df['original_fee'].notna()]
            if not unlimited_plans.empty:
                min_unlimited_cost = unlimited_plans['original_fee'].min()
                unlimited_plan_name = unlimited_plans.loc[unlimited_plans['original_fee'].idxmin()].get('plan_name', 'Unknown')
                unlimited_info = {
                    'has_unlimited': True,
                    'min_cost': float(min_unlimited_cost),
                    'plan_name': unlimited_plan_name,
                    'count': len(unlimited_plans)
                }
                logger.info(f"✅ Found unlimited {feature} plans: cheapest at ₩{min_unlimited_cost} (FLAG, not continuous data)")
        
        logger.info(f"✅ Applied monotonic filtering for {feature}: {len(frontier_data_points)} points (was {len(feature_values.unique())} raw points)")
        
        if len(frontier_data_points) < 3:
            # Fallback to linear model if insufficient data
            logger.warning(f"Insufficient frontier data for {feature}, using linear model")
            min_val = feature_values.min()
            max_val = feature_values.max()
            feature_points = np.linspace(min_val, max_val, 20)
            
            frontier_points = []
            actual_plan_points = []
            segments = []
            marginal_costs = [pure_coefficient]
            
            for feature_val in feature_points:
                pure_cost = base_cost + (feature_val * pure_coefficient)
                frontier_points.append({
                    'feature_value': float(feature_val),
                    'pure_cost': float(pure_cost),
                    'marginal_cost': float(pure_coefficient),
                    'cumulative_cost': float(feature_val * pure_coefficient),
                    'segment': 'linear'
                })
        else:
            # PIECEWISE LINEAR ANALYSIS - Calculate segments on RAW market data first!
            
            # Step 1: Sort raw market data and apply basic monotonicity (but keep more points than full filtering)
            sorted_frontier_data = sorted(frontier_data_points, key=lambda x: x[0])
            
            # Apply LIGHT filtering: remove only extreme outliers, keep most points
            # This removes obvious pricing errors while preserving market reality
            filtered_data = []
            for i, (feat_val, cost) in enumerate(sorted_frontier_data):
                if i == 0:
                    filtered_data.append((feat_val, cost))
                    continue
                
                prev_feat, prev_cost = filtered_data[-1]
                
                # Only remove if it violates basic economics (cost decreases significantly with more features)
                if feat_val > prev_feat and cost < prev_cost * 0.8:  # Allow 20% cost decrease max
                    logger.debug(f"Skipping outlier for {feature}: {feat_val} units at ₩{cost} (previous: {prev_feat} units at ₩{prev_cost})")
                    continue
                    
                # Only remove if marginal cost is extremely high (>10x the base coefficient)
                if feat_val > prev_feat:
                    implied_marginal = (cost - prev_cost) / (feat_val - prev_feat)
                    if implied_marginal > pure_coefficient * 10:  # More than 10x base coefficient
                        logger.debug(f"Skipping extreme marginal cost for {feature}: ₩{implied_marginal:.0f}/unit (>10x base: ₩{pure_coefficient:.0f})")
                        continue
                        
                filtered_data.append((feat_val, cost))
            
            raw_features = np.array([point[0] for point in filtered_data])
            raw_costs = np.array([point[1] for point in filtered_data])
            
            logger.info(f"✅ Light filtering for {feature}: kept {len(filtered_data)}/{len(sorted_frontier_data)} points (removed extreme outliers only)")
            
            logger.info(f"✅ Calculating piecewise segments on {len(filtered_data)} lightly-filtered market points for {feature}")
            
            # Step 2: Detect change points in sorted raw market data
            change_points = detect_change_points(raw_features, raw_costs)
            logger.info(f"Detected {len(change_points)} change points in sorted raw market data for {feature}")
            
            # Step 3: Fit piecewise linear segments on sorted raw market data
            segments = fit_piecewise_linear_segments(raw_features, raw_costs, change_points)
            logger.info(f"Fitted {len(segments)} segments on sorted raw market data for {feature}")
            
            # Step 4: Create filtered trendline ONLY for visualization (after segments are calculated!)
            from modules.cost_spec import create_robust_monotonic_frontier
            
            # Create temporary dataframe for monotonic filtering
            temp_df = pd.DataFrame(frontier_data_points, columns=[feature, 'original_fee'])
            monotonic_frontier = create_robust_monotonic_frontier(temp_df, feature, 'original_fee')
            
            if monotonic_frontier.empty:
                logger.warning(f"No valid monotonic trendline for {feature}")
                continue
                
            trendline_features = np.array(list(monotonic_frontier.index))
            trendline_costs = np.array(list(monotonic_frontier.values))
            
            # Step 5: Create visualization points using CUMULATIVE COST calculation
            frontier_points = []
            marginal_costs = []
            
            for i, (feat_val, cost) in enumerate(filtered_data):
                # Find which segment this point belongs to
                segment_marginal_cost = pure_coefficient  # fallback
                segment_info = "linear"
                
                if segments:
                    for seg_idx, segment in enumerate(segments):
                        if segment['start_feature'] <= feat_val <= segment['end_feature']:
                            segment_marginal_cost = segment['marginal_rate']
                            segment_info = f"Segment {seg_idx+1}: {segment['start_feature']:.1f}-{segment['end_feature']:.1f}"
                            break
                
                # Calculate CUMULATIVE cost using piecewise segments
                cumulative_cost = base_cost
                current_position = 0
                
                if segments:
                    for segment in segments:
                        segment_start = segment['start_feature']
                        segment_end = segment['end_feature']
                        segment_rate = segment['marginal_rate']
                        
                        # Calculate usage within this segment
                        usage_start = max(segment_start, current_position)
                        usage_end = min(segment_end, feat_val)
                        
                        if usage_end > usage_start:
                            usage_in_segment = usage_end - usage_start
                            cumulative_cost += usage_in_segment * segment_rate
                            current_position = usage_end
                        
                        if feat_val <= segment_end:
                            break
                else:
                    # Linear fallback
                    cumulative_cost += feat_val * pure_coefficient
                
                frontier_points.append({
                    'feature_value': float(feat_val),
                    'pure_cost': float(cumulative_cost),
                    'marginal_cost': float(segment_marginal_cost),
                    'cumulative_cost': float(cumulative_cost),
                    'segment': segment_info,
                    'actual_market_cost': float(cost)
                })
                
                marginal_costs.append(segment_marginal_cost)
            
            # Add actual plan points for comparison
            actual_plan_points = []
            for feat_val, cost in frontier_data_points[:20]:  # Limit to 20 points for visualization
                actual_plan_points.append({
                    'feature_value': float(feat_val),
                    'actual_cost': float(cost),
                    'point_type': 'market_plan'
                })
        
        # Store feature data
        marginal_cost_frontier_data[feature] = {
            'display_name': feature_display_names.get(feature, feature),
            'unit': feature_units.get(feature, 'KRW/unit'),
            'frontier_points': frontier_points,
            'actual_plan_points': actual_plan_points,
            'unlimited_info': unlimited_info,
            'base_coefficient': float(pure_coefficient),
            'segments_count': len(segments) if 'segments' in locals() else 1,
            'feature_name': feature
        }
        
        if len(segments) > 1:
            logger.info(f"✅ PIECEWISE model for {feature}: {len(segments)} segments, marginal cost range: ₩{min(marginal_costs):.0f}-₩{max(marginal_costs):.0f}")
        else:
            logger.info(f"📊 Linear model for {feature}: constant marginal cost ₩{pure_coefficient:.0f}")
    
        logger.info(f"Prepared piecewise frontier for {feature}: {len(frontier_points)} points, {len(actual_plan_points)} actual plans")

    logger.info(f"✅ Completed PIECEWISE LINEAR frontier preparation for {len(marginal_cost_frontier_data)} features")
    return marginal_cost_frontier_data 