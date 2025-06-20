"""
Marginal Cost Chart Module

This module handles marginal cost frontier chart data preparation and granular cost analysis.
Extracted from report_charts_legacy.py for better modularity.

Functions:
- prepare_marginal_cost_frontier_data: Piecewise linear marginal cost frontier charts
- create_granular_segments_with_intercepts: Granular segment creation with unlimited intercepts
- calculate_granular_piecewise_cost_with_intercepts: Granular cost calculation
- prepare_granular_marginal_cost_frontier_data: Comprehensive granular marginal cost analysis
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
            
            logger.info(f"✅ Created filtered trendline for {feature}: {len(trendline_features)} points for visualization (segments based on {len(frontier_data_points)} raw points)")
            logger.info(f"Fitted {len(segments)} segments for {feature} trendline (monotonicity applied to line only)")
            
            # Transform ACTUAL market plans to marginal cost space (no theoretical points!)
            frontier_points = []
            
            # For each actual market data point, calculate its marginal cost based on trendline segments
            for feature_val, actual_cost in frontier_data_points:
                # Find which segment this actual point belongs to
                segment_marginal_cost = pure_coefficient  # fallback to original coefficient
                segment_info = {'index': 0, 'range': 'linear'}
                
                for seg_idx, segment in enumerate(segments):
                    if segment['start_feature'] <= feature_val <= segment['end_feature']:
                        segment_marginal_cost = segment['marginal_rate']
                        segment_info = {
                            'index': seg_idx,
                            'start': float(segment['start_feature']),
                            'end': float(segment['end_feature']),
                            'slope': float(segment['marginal_rate']),
                            'range': f"{segment['start_feature']:.1f}-{segment['end_feature']:.1f}"
                        }
                        break
                
                # Calculate cumulative cost up to this point (integral of piecewise function)
                cumulative_cost = 0
                for prev_seg in segments:
                    if prev_seg['end_feature'] < feature_val:
                        # Full segment contribution
                        seg_length = prev_seg['end_feature'] - prev_seg['start_feature']
                        avg_cost = prev_seg['marginal_rate']
                        cumulative_cost += seg_length * avg_cost
                    elif prev_seg['start_feature'] <= feature_val <= prev_seg['end_feature']:
                        # Partial segment contribution
                        seg_length = feature_val - prev_seg['start_feature']
                        avg_cost = prev_seg['marginal_rate']
                        cumulative_cost += seg_length * avg_cost
                        break
                
                # Calculate total marginal cost for this feature
                total_marginal_cost = feature_val * segment_marginal_cost
                
                # Debug: Check if marginal cost exceeds actual cost (mathematical inconsistency)
                if total_marginal_cost > actual_cost:
                    logger.warning(f"🚨 INCONSISTENCY: {feature} plan with {feature_val} units has total marginal cost ₩{total_marginal_cost:,.0f} > actual cost ₩{actual_cost:,.0f} (marginal rate: ₩{segment_marginal_cost:.0f}/unit)")
                
                # Calculate what the model predicts for this plan (features only, no base cost)
                # Note: This is a single-feature analysis, so full model prediction would need all features
                predicted_cost_single_feature = total_marginal_cost  # No base cost - you pay for what you get
                
                # Transform actual market point to marginal cost representation
                frontier_points.append({
                    'feature_value': float(feature_val),
                    'actual_cost': float(actual_cost),  # Keep original actual cost for reference
                    'marginal_cost': float(segment_marginal_cost),  # Marginal cost from trendline segment
                    'total_marginal_cost': float(total_marginal_cost),  # This feature's total contribution
                    'predicted_single_feature_cost': float(predicted_cost_single_feature),  # Base + this feature only
                    'cumulative_cost': float(cumulative_cost),
                    'segment': f'segment_{segment_info["index"]}',
                    'segment_info': segment_info,
                    'is_actual_plan': True  # Flag to show these are real market plans, not theoretical
                })
            
            logger.info(f"✅ Transformed {len(frontier_points)} ACTUAL market plans to marginal cost space for {feature}")
        
        # Find actual plans for comparison (ALL cheapest points, not sampled!)
        actual_plan_points = []
        
        # Use ALL frontier data points as actual market plans (same as traditional frontier)
        for feature_val, actual_cost in frontier_data_points:
            # Find the actual plan for this point
            matching_plans = df_non_unlimited[
                (df_non_unlimited[feature] == feature_val) & 
                (df_non_unlimited['original_fee'] == actual_cost)
            ]
            
            if not matching_plans.empty:
                plan_row = matching_plans.iloc[0]
                
                # Find which trendline segment this point belongs to (for info only)
                segment_info = "market_point"
                if 'segments' in locals() and len(segments) > 1:
                    for seg in segments:
                        if seg['start_feature'] <= feature_val <= seg['end_feature']:
                            segment_info = f"Trendline Segment {seg['start_feature']:.1f}-{seg['end_feature']:.1f}"
                            break
                
                actual_plan_points.append({
                    'feature_value': float(feature_val),
                    'actual_cost': float(actual_cost),
                    'plan_name': plan_row.get('plan_name', 'Unknown'),
                    'segment_info': segment_info
                })
        
        logger.info(f"✅ Created {len(actual_plan_points)} actual market plan points for {feature} (same count as traditional frontier)")
        
        # Calculate comprehensive cost analysis
        if frontier_points:
            marginal_costs = [p['marginal_cost'] for p in frontier_points]
            cumulative_costs = [p['cumulative_cost'] for p in frontier_points]
            
            marginal_cost_frontier_data[feature] = {
                'feature_name': feature,
                'display_name': feature_display_names.get(feature, feature),
                'unit': feature_units.get(feature, ''),
                'pure_coefficient': float(pure_coefficient),
                'base_cost': float(base_cost),
                'frontier_points': frontier_points,
                'actual_plan_points': actual_plan_points,
                'unlimited_info': unlimited_info,  # 🔥 SEPARATE FLAG for unlimited
                'piecewise_info': {
                    'is_piecewise': len(segments) > 1 if 'segments' in locals() else False,
                    'num_segments': len(segments) if 'segments' in locals() else 1,
                    'segments': segments if 'segments' in locals() else [],
                    'change_points_detected': len(change_points) if 'change_points' in locals() else 0,
                    'continuous_data_only': True  # 🔥 CONFIRM: no unlimited mixed in trendline
                },
                'feature_range': {
                    'min': float(feature_values.min()),
                    'max': float(feature_values.max()),
                    'unique_values': len(feature_values.unique()),
                    'filtered_frontier_points': len(frontier_data_points)
                },
                'cost_analysis': {
                    'min_marginal_cost': float(min(marginal_costs)),
                    'max_marginal_cost': float(max(marginal_costs)),
                    'avg_marginal_cost': float(np.mean(marginal_costs)),
                    'marginal_cost_range': float(max(marginal_costs) - min(marginal_costs)),
                    'economies_of_scale': float(max(marginal_costs) - min(marginal_costs)) > 0,
                    'total_cost_range': float(max(cumulative_costs) - min(cumulative_costs)) if cumulative_costs else 0
                }
            }
            
            if len(segments) > 1:
                logger.info(f"✅ PIECEWISE model for {feature}: {len(segments)} segments, marginal cost range: ₩{min(marginal_costs):.0f}-₩{max(marginal_costs):.0f}")
            else:
                logger.info(f"📊 Linear model for {feature}: constant marginal cost ₩{pure_coefficient:.0f}")
        
        logger.info(f"Prepared piecewise frontier for {feature}: {len(frontier_points)} points, {len(actual_plan_points)} actual plans")
    
    logger.info(f"✅ Completed PIECEWISE LINEAR frontier preparation for {len(marginal_cost_frontier_data)} features")
    return marginal_cost_frontier_data

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
        'continuous_costs': feature_costs,
        'intercept_costs': intercept_costs,
        'breakdown': {
            'continuous_total': sum(fc['total_cost'] for fc in feature_costs.values()),
            'intercept_total': sum(ic['intercept_cost'] for ic in intercept_costs.values() if ic['is_active'])
        }
    }

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
                        
                        # Stop if we've reached our target feature value
                        if current_position >= feature_val:
                            break
                    
                    # Final cumulative cost
                    cumulative_cost = base_cost + total_feature_cost
                    
                    # Set fallback values if no target segment found
                    if target_segment is None:
                        segment_marginal_cost = pure_coefficient
                        segment_info = "Outside segment range"
                
                predicted_single_feature_cost = total_feature_cost if 'total_feature_cost' in locals() else feature_val * pure_coefficient
                
                frontier_points.append({
                    'feature_value': feature_val,
                    'actual_cost': actual_cost,
                    'marginal_cost': segment_marginal_cost,
                    'total_marginal_cost': predicted_single_feature_cost,
                    'predicted_single_feature_cost': predicted_single_feature_cost,
                    'cumulative_cost': cumulative_cost,
                    'segment': segment_info,
                    'segment_info': segment_info,
                    'is_actual_plan': True,
                    'plan_count': point_data['plan_count']
                })
            
            # Create actual plan points for market comparison
            actual_plan_points = []
            for _, plan in clean_df.iterrows():
                if pd.notna(plan[feature]) and pd.notna(plan['original_fee']):
                    feature_val = plan[feature]
                    actual_cost = plan['original_fee']
                    plan_name = plan.get('plan_name', 'Unknown')
                    
                    # Find segment
                    segment_info = "market_point"
                    for seg_idx, segment in enumerate(segments):
                        if segment['start_feature'] <= feature_val <= segment['end_feature']:
                            segment_info = f"Segment {seg_idx+1}: {segment['start_feature']:.1f}-{segment['end_feature']:.1f}"
                            break
                    
                    actual_plan_points.append({
                        'feature_value': float(feature_val),
                        'actual_cost': float(actual_cost),
                        'plan_name': plan_name,
                        'segment_info': segment_info
                    })
            
            logger.info(f"Created {len(actual_plan_points)} actual market plan points for {feature}")
            
        else:
            # Fallback to linear model
            logger.warning(f"Using linear model for {feature} due to insufficient data points")
            segments = []
            frontier_points = []
            actual_plan_points = []
            
            # Create linear points
            for point_data in feature_level_data:
                feature_val = point_data['feature_value']
                predicted_cost = base_cost + (feature_val * pure_coefficient)
                
                frontier_points.append({
                    'feature_value': feature_val,
                    'actual_cost': point_data['min_cost'],
                    'marginal_cost': pure_coefficient,
                    'total_marginal_cost': feature_val * pure_coefficient,
                    'predicted_single_feature_cost': feature_val * pure_coefficient,
                    'cumulative_cost': predicted_cost,
                    'segment': 'linear',
                    'segment_info': 'Linear model',
                    'is_actual_plan': True,
                    'plan_count': point_data['plan_count']
                })
        
        # Store comprehensive data for this feature
        granular_frontier_data[feature] = {
            'feature_name': feature,
            'display_name': feature_display_names.get(feature, feature),
            'unit': feature_units.get(feature, 'KRW/unit'),
            'pure_coefficient': pure_coefficient,
            'base_cost': base_cost,
            'frontier_points': frontier_points,
            'actual_plan_points': actual_plan_points,
            'unlimited_info': unlimited_info,
            'piecewise_info': {
                'is_piecewise': len(segments) > 1,
                'num_segments': len(segments),
                'segments': segments,
                'change_points_detected': len(change_points) if 'change_points' in locals() else 0,
                'continuous_data_only': True
            },
            'feature_range': {
                'min': min_val,
                'max': max_val,
                'unique_values': len(unique_values),
                'filtered_frontier_points': len(frontier_points)
            },
            'cost_analysis': {
                'min_marginal_cost': min([p['marginal_cost'] for p in frontier_points]) if frontier_points else 0,
                'max_marginal_cost': max([p['marginal_cost'] for p in frontier_points]) if frontier_points else 0,
                'avg_marginal_cost': np.mean([p['marginal_cost'] for p in frontier_points]) if frontier_points else 0,
                'marginal_cost_range': max([p['marginal_cost'] for p in frontier_points]) - min([p['marginal_cost'] for p in frontier_points]) if frontier_points else 0,
                'economies_of_scale': len(segments) > 1,
                'total_cost_range': max([p['cumulative_cost'] for p in frontier_points]) - min([p['cumulative_cost'] for p in frontier_points]) if frontier_points else 0
            }
        }
        
        features_processed += 1
        logger.info(f"✅ Processed {feature}: {len(frontier_points)} frontier points, {len(actual_plan_points)} market plans")
    
    if features_processed == 0:
        logger.error("No features were successfully processed!")
        return {}
    
    logger.info(f"✅ Successfully processed {features_processed}/{len(core_continuous_features)} features using ENTIRE DATASET")
    
    # Add method metadata
    method_info = {
        'name': 'Full Dataset Marginal Cost Analysis',
        'description': 'Marginal costs extracted from entire dataset regression',
        'total_plans_analyzed': len(clean_df),
        'features_analyzed': features_processed
    }
    
    # Add cost breakdown for comprehensive analysis
    cost_breakdown = {
        'base_cost': base_cost,
        'feature_costs': [
            {
                'feature': feature,
                'display_name': feature_display_names.get(feature, feature),
                'coefficient': feature_costs[feature].get('coefficient', 0),
                'unit': feature_units.get(feature, 'KRW/unit'),
                'cost_per_unit': feature_costs[feature].get('coefficient', 0)
            }
            for feature in core_continuous_features
            if feature in feature_costs
        ]
    }
    
    # Add coefficient comparison with PIECEWISE SEGMENTS (not fixed rates)
    coefficient_comparison = {
        'features': [],
        'piecewise_segments': [],
        'display_names': [],
        'units': []
    }
    
    for feature in core_continuous_features:
        if feature in feature_costs and feature in granular_frontier_data:
            feature_data = granular_frontier_data[feature]
            display_name = feature_display_names.get(feature, feature)
            unit = feature_units.get(feature, 'KRW/unit')
            
            # Get piecewise segments for this feature
            piecewise_info = feature_data.get('piecewise_info', {})
            segments = piecewise_info.get('segments', [])
            
            if segments and len(segments) > 1:
                # Multiple segments - show piecewise structure
                segment_descriptions = []
                for i, segment in enumerate(segments):
                    start_val = segment['start_feature']
                    end_val = segment['end_feature']
                    marginal_rate = segment['marginal_rate']
                    segment_descriptions.append(f"Segment {i+1} ({start_val:.1f}-{end_val:.1f}): ₩{marginal_rate:.2f}")
                
                coefficient_comparison['features'].append(display_name)
                coefficient_comparison['piecewise_segments'].append(segment_descriptions)
                coefficient_comparison['display_names'].append(display_name)
                coefficient_comparison['units'].append(unit)
            else:
                # Single segment or linear - show single rate
                base_coefficient = feature_costs[feature].get('coefficient', 0)
                coefficient_comparison['features'].append(display_name)
                coefficient_comparison['piecewise_segments'].append([f"Linear: ₩{base_coefficient:.2f}"])
                coefficient_comparison['display_names'].append(display_name)
                coefficient_comparison['units'].append(unit)
    
    return {
        **granular_frontier_data,
        'method_info': method_info,
        'cost_breakdown': cost_breakdown,
        'coefficient_comparison': coefficient_comparison
    }
