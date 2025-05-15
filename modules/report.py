"""
Report Generation Module

This module handles generating and saving HTML reports for the Moyo Ranking Model.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import numpy as np

# Define unlimited flag mappings (copied from cost_spec.py)
UNLIMITED_FLAGS = {
    'basic_data_clean': 'basic_data_unlimited',
    'daily_data_clean': 'daily_data_unlimited',
    'voice_clean': 'voice_unlimited',
    'message_clean': 'message_unlimited',
    'speed_when_exhausted': 'has_unlimited_speed'
}

# Feature units for formatting (assuming these are defined or can be added)
FEATURE_UNITS = {
    'basic_data_clean': 'GB/month',
    'daily_data_clean': 'GB/day',
    'voice_clean': 'min',
    'message_clean': 'SMS',
    'additional_call': 'KRW/call', # Or appropriate unit
    'speed_when_exhausted': 'Mbps',
    'tethering_gb': 'GB'
}

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Configure logging
logger = logging.getLogger(__name__)

# Python equivalent of featureDisplayNames for use in HTML generation
FEATURE_DISPLAY_NAMES_PY = {
    'basic_data_clean': 'Basic Data',
    'daily_data_clean': 'Daily Data',
    'voice_clean': 'Voice',
    'message_clean': 'SMS',
    'additional_call': 'Additional Call',
    'speed_when_exhausted': 'Throttled Speed',
    'tethering_gb': 'Tethering'
}

def get_richness_score(plan_row, features_to_score, all_core_features, unlimited_flags_map):
    """Calculate a richness score for a plan based on a subset of its features."""
    score = 0
    # Ensure plan_row is a Series or dict, not a DataFrame
    if isinstance(plan_row, pd.DataFrame):
        if len(plan_row) == 1:
            plan_row = plan_row.iloc[0]
        else:
            # Or handle error: raise ValueError("plan_row must be a single row (Series) or dict")
            return 0 # Or some default error score

    for feature_col in features_to_score:
        if feature_col not in plan_row: # Check if feature exists in the plan data
            continue
            
        value = plan_row[feature_col]
        is_unlimited = False
        unlimited_flag_col = unlimited_flags_map.get(feature_col)
        if unlimited_flag_col and unlimited_flag_col in plan_row and plan_row[unlimited_flag_col]:
            is_unlimited = True
        
        if is_unlimited:
            score += 100
        elif pd.notna(value) and isinstance(value, (int, float)) and value > 0:
            score += 10
        elif pd.notna(value) and isinstance(value, (int, float)) and value <= 0:
            score += 1
        # else: NA or non-numeric values get 0 score for this feature
    return score

def estimate_value_on_visual_frontier(value_to_estimate, frontier_points_tuples):
    """
    Estimates the cost (y-value, typically original_fee) for a given feature value (x-value)
    based on a list of (value, cost) tuples representing the visual frontier.
    Uses linear interpolation, extrapolation, or exact match.
    `frontier_points_tuples` must be sorted by feature value (x).
    """
    if not frontier_points_tuples:
        return None  # Cannot estimate if frontier is empty

    if pd.isna(value_to_estimate):
        return None # Cannot estimate for NaN input

    # Convert to numeric, if not already, for comparison
    try:
        x_target = float(value_to_estimate)
    except (ValueError, TypeError):
        return None # Cannot estimate for non-numeric input if conversion fails

    # Ensure frontier_points_tuples are sorted by x-value (feature value)
    # Assuming they are already sorted as per function docstring, but a check/sort could be added if necessary.
    # sorted_points = sorted(frontier_points_tuples, key=lambda p: p[0])
    # For simplicity, assuming frontier_points_tuples is pre-sorted.
    sorted_points = frontier_points_tuples


    # Exact match
    for x, y in sorted_points:
        if x == x_target:
            return y

    # Extrapolation: target is less than the smallest frontier value
    if x_target < sorted_points[0][0]:
        # Option 1: Return cost of the first point (step function)
        # return sorted_points[0][1]
        # Option 2: Linear extrapolation from first two points (if available)
        if len(sorted_points) >= 2:
            x1, y1 = sorted_points[0]
            x2, y2 = sorted_points[1]
            if x2 - x1 != 0: # Avoid division by zero
                 return y1 + (x_target - x1) * (y2 - y1) / (x2 - x1)
            return y1 # Fallback to first point's cost
        return sorted_points[0][1] # Fallback if only one point

    # Extrapolation: target is greater than the largest frontier value
    if x_target > sorted_points[-1][0]:
        # Option 1: Return cost of the last point (step function)
        # return sorted_points[-1][1]
        # Option 2: Linear extrapolation from last two points (if available)
        if len(sorted_points) >= 2:
            x1, y1 = sorted_points[-2]
            x2, y2 = sorted_points[-1]
            if x2 - x1 != 0: # Avoid division by zero
                return y1 + (x_target - x1) * (y2 - y1) / (x2 - x1)
            return y2 # Fallback to last point's cost
        return sorted_points[-1][1] # Fallback if only one point

    # Interpolation
    for i in range(len(sorted_points) - 1):
        x1, y1 = sorted_points[i]
        x2, y2 = sorted_points[i+1]
        if x1 <= x_target < x2: # x_target is between x1 and x2
            if x2 - x1 == 0: # Should not happen in a well-formed frontier
                return y1 
            return y1 + (x_target - x1) * (y2 - y1) / (x2 - x1)
    
    # Should not be reached if logic is correct and sorted_points is not empty
    # but as a fallback, maybe return cost of nearest point or None
    logger.warning(f"Estimate_value_on_visual_frontier: Value {x_target} fell through interpolation logic. Points: {sorted_points}")
    return None


def format_other_core_specs_string(plan_row, feature_analyzed, core_continuous_features, feature_units_map, unlimited_flags_map):
    """Formats the 'Other Core Specs' string for the residual analysis table."""
    parts = []
    # Ensure plan_row is a Series or dict
    if isinstance(plan_row, pd.DataFrame):
        if len(plan_row) == 1:
            plan_row = plan_row.iloc[0]
        else:
            return "Error: Invalid plan_row"

    for f_col in core_continuous_features:
        if f_col == feature_analyzed:
            continue
        
        if f_col not in plan_row: # Check if feature exists in the plan data
            parts.append(f"N/A {feature_units_map.get(f_col, '')}".strip())
            continue

        value = plan_row[f_col]
        unit = feature_units_map.get(f_col, "")
        
        is_unlimited = False
        unlimited_flag_col = unlimited_flags_map.get(f_col)
        if unlimited_flag_col and unlimited_flag_col in plan_row and plan_row[unlimited_flag_col]:
            is_unlimited = True

        if is_unlimited:
            # For unlimited, check if the base feature column (e.g., basic_data_clean) has a numeric value.
            # If it does (e.g., some plans might list a high number for "unlimited" in the numeric col), use it.
            # Otherwise, just say "Unlimited".
            if pd.notna(value) and isinstance(value, (int, float)) and value > 0: # and value might be a very high number for "unlimited"
                 parts.append(f"Unlimited ({value} {unit})".strip())
            else:
                 parts.append(f"Unlimited {unit}".strip())
        elif pd.notna(value) and isinstance(value, (int,float)):
            parts.append(f"{value:.1f} {unit}".strip() if isinstance(value, float) else f"{value} {unit}".strip())
        else: # Handles None, NaN, or non-numeric strings not caught by unlimited
            parts.append(f"N/A {unit}".strip())
            
    specs_str = " + ".join(filter(None, parts)) # filter(None,...) in case any part becomes empty
    
    total_original_fee_val = plan_row.get('original_fee', 0)
    if pd.notna(total_original_fee_val):
        total_original_fee_str = f"{total_original_fee_val:,.0f} KRW"
    else:
        total_original_fee_str = "N/A KRW"
        
    return f"{specs_str} = {total_original_fee_str}"


def generate_html_report(df, timestamp, is_dea=False, is_cs=True, title="Mobile Plan Rankings"):
    """Generate an HTML report of the rankings.
    
    Args:
        df: DataFrame with ranking data
        timestamp: Timestamp for the report
        is_dea: Deprecated parameter, kept for backward compatibility
        is_cs: Whether this is a Cost-Spec report (default: True)
        title: Title for the report (default: "Mobile Plan Rankings")
        
    Returns:
        HTML content as string
    """
    # Get ranking method and log transform from the dataframe attributes if available
    ranking_method = df.attrs.get('ranking_method', 'relative')
    use_log_transform = df.attrs.get('use_log_transform', False)
    
    # Get the features used for calculation
    used_features = df.attrs.get('used_features', [])
    
    # Get current timestamp
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Set report title based on method
    report_title = "Cost-Spec Mobile Plan Rankings"
    
    # Sort the DataFrame based on rank column
    rank_col = 'rank_number'
    df_sorted = df.sort_values(rank_col) if rank_col in df.columns else df
    
    # Calculate feature contributions to baseline costs
    feature_contribution_data = {}
    
    # Get contribution columns (from cost_spec.py)
    contribution_cols = [col for col in df.columns if col.startswith("contribution_")]
    
    # Calculate average, min, max contribution for each feature
    for col in contribution_cols:
        feature_name = col.replace("contribution_", "")
        avg_contrib = df[col].mean() if col in df.columns else 0
        max_contrib = df[col].max() if col in df.columns else 0
        min_contrib = df[col].min() if col in df.columns else 0
        
        # Percentage of baseline cost
        avg_baseline_cost = df['B'].mean() if 'B' in df.columns else 1
        contribution_percentage = (avg_contrib / avg_baseline_cost * 100) if avg_baseline_cost > 0 else 0
        
        feature_contribution_data[feature_name] = {
            'avg_contribution': avg_contrib,
            'max_contribution': max_contrib,
            'min_contribution': min_contrib,
            'percentage': contribution_percentage
        }
    
    # Build calculation summary data
    total_plans = len(df)
    
    # Top plan info
    if len(df_sorted) > 0:
        top_plan_name = str(df_sorted.iloc[0]['plan_name']) if 'plan_name' in df_sorted.iloc[0] else 'N/A'
        top_value = df_sorted.iloc[0]['CS'] if 'CS' in df_sorted.iloc[0] else None
        if isinstance(top_value, float):
            top_plan_value = f"{top_value:.4f}"
        else:
            top_plan_value = "N/A"
    else:
        top_plan_name = 'N/A'
        top_plan_value = 'N/A'
    
    # Prepare feature frontier data
    # This will hold feature values and their corresponding baseline costs
    feature_frontier_data = {}
    
    # Core continuous features to visualize (those that likely have frontiers)
    core_continuous_features = [
        'basic_data_clean', 'daily_data_clean', 'voice_clean', 
        'message_clean', 'speed_when_exhausted', 'additional_call',
        'tethering_gb'
    ]
    
    # Prepare data points for feature-specific charts
    # Initialize all_chart_data here to store comprehensive chart data including full plan series for frontiers
    all_chart_data = {}
    visual_frontiers_for_residual_table = {} # Stores (value, cost) tuples for estimate_value_on_visual_frontier

    for feature in core_continuous_features:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in dataframe, skipping visualization")
            continue
            
        # For visualization, we will use 'original_fee' as the cost metric for the frontier.
        # The 'contribution_col' (derived from 'fee') is used for backend calculations but not directly for this plot's Y-axis.
        cost_metric_for_visualization = 'original_fee'
        if cost_metric_for_visualization not in df.columns:
            logger.warning(f"'{cost_metric_for_visualization}' not found in dataframe, skipping visualization for {feature}")
            continue
            
        logger.info(f"Preparing frontier chart data for feature: {feature} using '{cost_metric_for_visualization}'")
        
        # Check if this feature has an unlimited flag
        unlimited_flag = UNLIMITED_FLAGS.get(feature)
        has_unlimited_data = False
        unlimited_min_visual_cost = None # Renamed to avoid confusion with 'fee' based unlimited costs
        unlimited_min_plan = None
        
        # If unlimited flag exists, extract unlimited value data
        if unlimited_flag and unlimited_flag in df.columns:
            unlimited_plans_df = df[df[unlimited_flag] == 1] 
            if not unlimited_plans_df.empty and cost_metric_for_visualization in unlimited_plans_df.columns:
                has_unlimited_data = True
                min_visual_cost_idx = unlimited_plans_df[cost_metric_for_visualization].idxmin()
                unlimited_min_visual_cost = unlimited_plans_df.loc[min_visual_cost_idx, cost_metric_for_visualization]
                unlimited_min_plan = unlimited_plans_df.loc[min_visual_cost_idx, 'plan_name'] if 'plan_name' in unlimited_plans_df.columns else "Unknown"
                logger.info(f"Found unlimited {feature} with minimum '{cost_metric_for_visualization}' {unlimited_min_visual_cost} from plan '{unlimited_min_plan}'")
                df_for_frontier = df[(df[unlimited_flag] == 0) & df[cost_metric_for_visualization].notna()].copy()
            else:
                df_for_frontier = df[df[cost_metric_for_visualization].notna()].copy()
        else:
            df_for_frontier = df[df[cost_metric_for_visualization].notna()].copy()
        
        # Step 1: Get all unique feature values and their minimum costs (using cost_metric_for_visualization)
        # candidate_points_details will now store the full Pandas Series for each candidate plan.
        candidate_points_details_series = [] 
        if not df_for_frontier.empty:
            min_cost_indices = df_for_frontier.loc[df_for_frontier.groupby(feature)[cost_metric_for_visualization].idxmin()].index
            min_cost_candidates_df = df_for_frontier.loc[min_cost_indices]
            
            # Sort these candidates by feature value, then by cost_metric_for_visualization
            min_cost_candidates_df = min_cost_candidates_df.sort_values(by=[feature, cost_metric_for_visualization])

            for _, plan_series_row in min_cost_candidates_df.iterrows():
                candidate_points_details_series.append(plan_series_row) # Append the full Series
            
            # Only log the count of candidate points, not every addition
            logger.info(f"Found {len(candidate_points_details_series)} minimum-cost candidate points for feature {feature}")
            
        # Step 2: Build the true monotonic frontier (list of Pandas Series for plans on the frontier)
        actual_frontier_plans_series_list = [] 

        for candidate_plan_series in candidate_points_details_series:
            # If this is the first point, always add it
            if not actual_frontier_plans_series_list:
                actual_frontier_plans_series_list.append(candidate_plan_series)
                continue
                
            last_frontier_plan_series = actual_frontier_plans_series_list[-1]
            current_value = candidate_plan_series[feature]
            current_cost = candidate_plan_series[cost_metric_for_visualization]
            last_value = last_frontier_plan_series[feature]
            last_cost = last_frontier_plan_series[cost_metric_for_visualization]
            
            # Pop points that this one dominates (more value for less cost)
            while actual_frontier_plans_series_list and \
                  current_value > actual_frontier_plans_series_list[-1][feature] and \
                  current_cost < actual_frontier_plans_series_list[-1][cost_metric_for_visualization]:
                actual_frontier_plans_series_list.pop()
                if actual_frontier_plans_series_list:
                    last_frontier_plan_series = actual_frontier_plans_series_list[-1]
                    last_value = last_frontier_plan_series[feature]
                    last_cost = last_frontier_plan_series[cost_metric_for_visualization]
            
            # Add point if:
            # 1. It offers more value than last point
            # 2. It costs more than the last point
            # 3. The cost increase is at least 1.0 KRW
            if (current_value > last_value and 
                current_cost > last_cost and
                (current_cost - last_cost) >= 1.0):
                actual_frontier_plans_series_list.append(candidate_plan_series)
                
        # Ensure frontier points are sorted by feature value 
        actual_frontier_plans_series_list.sort(key=lambda p: p[feature])
        
        # Add (0,0) as the starting point if not present
        min_feature_value = actual_frontier_plans_series_list[0][feature] if actual_frontier_plans_series_list else 0
        if min_feature_value > 0 and not df_for_frontier.empty:
            # Create a synthetic starting point at (0,0)
            zero_point_series = actual_frontier_plans_series_list[0].copy() if actual_frontier_plans_series_list else pd.Series({})
            zero_point_series[feature] = 0
            zero_point_series[cost_metric_for_visualization] = 0
            zero_point_series['plan_name'] = "Free Baseline"
            # Insert at the beginning
            actual_frontier_plans_series_list.insert(0, zero_point_series)
            logger.info(f"Added (0,0) starting point to feature {feature} frontier")
        
        # Find the maximum feature value in the dataset, and add the lowest cost point for that value to the frontier if not present
        if not df_for_frontier.empty:
            max_feature_value = df_for_frontier[feature].max()
            max_feature_rows = df_for_frontier[df_for_frontier[feature] == max_feature_value]
            
            if not max_feature_rows.empty:
                # Find minimum cost for the maximum feature value
                min_cost_for_max_value = max_feature_rows[cost_metric_for_visualization].min()
                max_value_min_cost_row = max_feature_rows.loc[max_feature_rows[cost_metric_for_visualization] == min_cost_for_max_value].iloc[0]
                
                # Check if this max value point is already in our frontier 
                existing_max_values = [p[feature] for p in actual_frontier_plans_series_list if p[feature] == max_feature_value]
                
                # If not in frontier, or if our frontier is empty, add it
                if not existing_max_values and actual_frontier_plans_series_list:
                    # Only add if it maintains monotonicity and 1.0 KRW minimum increase
                    last_frontier_point = actual_frontier_plans_series_list[-1]
                    if (max_feature_value > last_frontier_point[feature] and 
                        min_cost_for_max_value > last_frontier_point[cost_metric_for_visualization] and
                        (min_cost_for_max_value - last_frontier_point[cost_metric_for_visualization]) >= 1.0):
                        actual_frontier_plans_series_list.append(max_value_min_cost_row)
                        logger.info(f"Added maximum value point ({max_feature_value}) with minimum cost to feature {feature} frontier")
        
        # Log summary of frontier building, not every point
        logger.info(f"Built monotonic frontier with {len(actual_frontier_plans_series_list)} points for feature {feature}")
        
        # Populate visual_frontiers_for_residual_table with (value, original_fee) tuples from these Series
        current_feature_visual_frontier_tuples = [(p[feature], p[cost_metric_for_visualization]) for p in actual_frontier_plans_series_list]
        visual_frontiers_for_residual_table[feature] = current_feature_visual_frontier_tuples
        
        # Store the list of frontier plan Series in all_chart_data for later use in residual analysis
        if feature not in all_chart_data: all_chart_data[feature] = {}
        all_chart_data[feature]['actual_frontier_plans_series'] = actual_frontier_plans_series_list


        # Step 3: Classify all points from df_for_frontier based on the visual frontier
        # Extract necessary data for JS charts from actual_frontier_plans_series_list and candidate_points_details_series
        frontier_feature_values = [p[feature] for p in actual_frontier_plans_series_list]
        frontier_visual_costs = [p[cost_metric_for_visualization] for p in actual_frontier_plans_series_list]
        frontier_plan_names = [p['plan_name'] if 'plan_name' in p else "Unknown" for p in actual_frontier_plans_series_list]
        
        excluded_feature_values = []
        excluded_visual_costs = [] 
        excluded_plan_names = []
        
        other_feature_values = [] # Not plotted, but calculated for completeness/logging
        other_visual_costs = []
        other_plan_names = []

        # Create a set of (value, cost, plan_name) for quick lookup of true frontier points
        true_frontier_signature_set = set(
            (p[feature], p[cost_metric_for_visualization], p['plan_name'] if 'plan_name' in p else "Unknown") 
            for p in actual_frontier_plans_series_list
        )

        # Identify excluded points from the initial candidates (min cost for each value)
        for candidate_plan_series in candidate_points_details_series:
            sig = (candidate_plan_series[feature], candidate_plan_series[cost_metric_for_visualization], candidate_plan_series['plan_name'] if 'plan_name' in candidate_plan_series else "Unknown")
            if sig not in true_frontier_signature_set:
                excluded_feature_values.append(float(candidate_plan_series[feature]))
                excluded_visual_costs.append(float(candidate_plan_series[cost_metric_for_visualization]))
                excluded_plan_names.append(candidate_plan_series['plan_name'] if 'plan_name' in candidate_plan_series else "Unknown")
        
        # Identify 'other' points (not on frontier, not excluded - i.e., not a min cost for their value)
        # This requires iterating through df_for_frontier again.
        if not df_for_frontier.empty:
            # Create a set of (value, cost) for all candidates to quickly identify non-candidate points
            all_candidate_min_value_cost_pairs = set(
                (p_series[feature], p_series[cost_metric_for_visualization]) for p_series in candidate_points_details_series
            )
            for _, row_series in df_for_frontier.iterrows():
                f_val = row_series[feature]
                c_cost = row_series[cost_metric_for_visualization]
                # Check if this point (value, cost) was among the min-cost candidates for its value
                if (f_val, c_cost) not in all_candidate_min_value_cost_pairs:
                    other_feature_values.append(float(f_val))
                    other_visual_costs.append(float(c_cost))
                    other_plan_names.append(row_series['plan_name'] if 'plan_name' in row_series else "Unknown")
        
        all_values_for_js = []
        all_visual_costs_for_js = [] # Renamed
        all_plan_names_for_js = []
        all_is_frontier_for_js = []
        all_is_excluded_for_js = []

        for i in range(len(frontier_feature_values)):
            all_values_for_js.append(frontier_feature_values[i])
            all_visual_costs_for_js.append(frontier_visual_costs[i])
            all_plan_names_for_js.append(frontier_plan_names[i])
            all_is_frontier_for_js.append(True)
            all_is_excluded_for_js.append(False)
        
        for i in range(len(excluded_feature_values)):
            all_values_for_js.append(excluded_feature_values[i])
            all_visual_costs_for_js.append(excluded_visual_costs[i])
            all_plan_names_for_js.append(excluded_plan_names[i])
            all_is_frontier_for_js.append(False)
            all_is_excluded_for_js.append(True)

        for i in range(len(other_feature_values)):
            all_values_for_js.append(other_feature_values[i])
            all_visual_costs_for_js.append(other_visual_costs[i]) # Using original_fee based list
            all_plan_names_for_js.append(other_plan_names[i])
            all_is_frontier_for_js.append(False)
            all_is_excluded_for_js.append(False)
            
        frontier_points_count = len(frontier_feature_values)
        excluded_points_count = len(excluded_feature_values)
        other_points_count = len(other_feature_values)
        unlimited_count = 1 if has_unlimited_data else 0
        
        logger.info(f"Feature {feature}: Found {frontier_points_count} frontier, {excluded_points_count} excluded, {unlimited_count} unlimited points")
        
        # For the JS chart, we will only pass frontier and excluded points.
        # The 'other_values', 'other_visual_costs', etc., are calculated for logging/debugging but not sent for plotting.
        js_chart_values = []
        js_chart_costs = []
        js_chart_plan_names = []
        js_chart_is_frontier = []
        js_chart_is_excluded = []

        for i in range(len(frontier_feature_values)):
            js_chart_values.append(frontier_feature_values[i])
            js_chart_costs.append(frontier_visual_costs[i])
            js_chart_plan_names.append(frontier_plan_names[i])
            js_chart_is_frontier.append(True)
            js_chart_is_excluded.append(False)
        
        for i in range(len(excluded_feature_values)):
            js_chart_values.append(excluded_feature_values[i])
            js_chart_costs.append(excluded_visual_costs[i])
            js_chart_plan_names.append(excluded_plan_names[i])
            js_chart_is_frontier.append(False)
            js_chart_is_excluded.append(True)

        # Only add to feature_frontier_data if we have frontier or excluded points, or an unlimited point.
        if frontier_points_count > 0 or excluded_points_count > 0 or has_unlimited_data:
            # Ensure frontier points are sorted by feature value to guarantee proper line connection
            sorted_frontier_values = []
            sorted_frontier_costs = []
            sorted_frontier_names = []

            # Create sorted lists based on feature values
            sorted_indices = sorted(range(len(frontier_feature_values)), key=lambda i: frontier_feature_values[i])
            for idx in sorted_indices:
                sorted_frontier_values.append(frontier_feature_values[idx])
                sorted_frontier_costs.append(frontier_visual_costs[idx])
                sorted_frontier_names.append(frontier_plan_names[idx])
                
            # Check if we need to add a (0,0) point for proper charting
            if len(sorted_frontier_values) > 0 and sorted_frontier_values[0] > 0:
                sorted_frontier_values.insert(0, 0)
                sorted_frontier_costs.insert(0, 0)
                sorted_frontier_names.insert(0, "Free Baseline")
                logger.info(f"Added (0,0) point to sorted frontier values for chart rendering")
                
            feature_frontier_data[feature] = {
                # Pass only the combined frontier/excluded lists to JS under generic 'all_...' keys
                'all_values': js_chart_values, 
                'all_contributions': js_chart_costs, # JS expects all_contributions for costs
                'all_is_frontier': js_chart_is_frontier,
                'all_is_excluded': js_chart_is_excluded,
                # all_is_unlimited will be determined in JS based on the separate unlimited point data
                'all_plan_names': js_chart_plan_names,
                
                # Use the sorted frontier values for proper line connection
                'frontier_values': sorted_frontier_values,
                'frontier_contributions': sorted_frontier_costs, 
                'frontier_plan_names': sorted_frontier_names,
                'excluded_values': excluded_feature_values,
                'excluded_contributions': excluded_visual_costs, 
                'excluded_plan_names': excluded_plan_names,
                # 'other_values', 'other_contributions', 'other_plan_names' are no longer sent for charting
                
                'has_unlimited': has_unlimited_data,
                'unlimited_value': unlimited_min_visual_cost if has_unlimited_data else None, 
                'unlimited_plan': unlimited_min_plan if has_unlimited_data else None
            }
        else:
            logger.info(f"Skipping chart data for feature {feature}: no frontier, excluded, or unlimited points")
    
    # Log the state of all_chart_data after the loop
    logger.info("Finished populating all_chart_data for charts.")
    
    # --- Start: New section for Residual Analysis Data Preparation ---
    residual_analysis_table_data = []
    logger.info("Starting Residual Original Fee Analysis.")

    for feature_analyzed in core_continuous_features:
        # Simplified logging - just log once if we're skipping a feature
        if feature_analyzed not in all_chart_data or \
           'actual_frontier_plans_series' not in all_chart_data[feature_analyzed] or \
           not all_chart_data[feature_analyzed]['actual_frontier_plans_series']:
            logger.info(f"Skipping residual analysis for '{feature_analyzed}': missing frontier data.")
            continue
        if feature_analyzed not in FEATURE_DISPLAY_NAMES_PY:
            logger.info(f"Skipping residual analysis for '{feature_analyzed}': missing display name.")
            continue

        # 1. Get the list of plans on the visual frontier for the current feature_analyzed
        current_feature_frontier_plans = all_chart_data[feature_analyzed]['actual_frontier_plans_series']

        # 2. Find plans in this list with the minimum value for feature_analyzed
        min_val_for_feature_on_frontier = min(p[feature_analyzed] for p in current_feature_frontier_plans)
        
        candidate_target_plans = [
            p for p in current_feature_frontier_plans if p[feature_analyzed] == min_val_for_feature_on_frontier
        ]

        if not candidate_target_plans:
            logger.info(f"Skipping residual analysis for '{feature_analyzed}': no min-value plans found.")
            continue

        # 3. Tie-breaking:
        #    a) Lowest 'original_fee' (which is 'cost_metric_for_visualization' for these frontier plans)
        candidate_target_plans.sort(key=lambda p: p[cost_metric_for_visualization])
        
        min_original_fee_for_candidates = candidate_target_plans[0][cost_metric_for_visualization]
        
        # Filter by this min_original_fee for the next tie-breaking step
        plans_tied_on_fee = [
            p for p in candidate_target_plans if p[cost_metric_for_visualization] == min_original_fee_for_candidates
        ]

        target_plan_series = None
        if len(plans_tied_on_fee) == 1:
            target_plan_series = plans_tied_on_fee[0]
        else:
            # b) Highest "richness score" for *other* core features
            other_core_features_for_richness = [f for f in core_continuous_features if f != feature_analyzed]
            best_target_plan_series_richness = None
            highest_richness_score = -1

            for plan_series_item in plans_tied_on_fee:
                current_richness = get_richness_score(plan_series_item, other_core_features_for_richness, core_continuous_features, UNLIMITED_FLAGS)
                if current_richness > highest_richness_score:
                    highest_richness_score = current_richness
                    best_target_plan_series_richness = plan_series_item
            
            target_plan_series = best_target_plan_series_richness if best_target_plan_series_richness is not None else plans_tied_on_fee[0] # Fallback

        if target_plan_series is None:
             logger.warning(f"Could not select a target plan from visual frontier for {feature_analyzed}. Skipping.")
             continue
        
        # 4. Prepare data for the table row
        target_plan_name_display = target_plan_series['plan_name'] if 'plan_name' in target_plan_series else "Unknown"
        plan_specs_string = format_plan_specs_display_string(target_plan_series, core_continuous_features, FEATURE_DISPLAY_NAMES_PY, FEATURE_UNITS, UNLIMITED_FLAGS)

        # Cost of the analyzed feature is its cost on its own frontier (which is the point we selected)
        cost_of_analyzed_feature_on_frontier = target_plan_series[cost_metric_for_visualization]
        
        # Estimate combined cost of other core features
        combined_est_cost_others = 0
        all_other_costs_valid_for_sum = True
        other_core_features_list = [f for f in core_continuous_features if f != feature_analyzed]

        for f_other in other_core_features_list:
            if f_other not in visual_frontiers_for_residual_table or not visual_frontiers_for_residual_table[f_other]:
                logger.warning(f"Visual frontier (value,cost) tuples for other feature '{f_other}' not found or empty. Cannot estimate its cost for plan '{target_plan_name_display}'.")
                all_other_costs_valid_for_sum = False # Mark that we can't get a complete sum
                # For now, let's be strict. If any part needed for full calculation is missing, we might skip the row or mark it.
                continue # Skip this feature's cost if its frontier is missing for estimation
            
            val_f_other_in_target_plan = target_plan_series.get(f_other)
            frontier_tuples_f_other = visual_frontiers_for_residual_table[f_other]
            
            cost_component_f_other = None
            if pd.notna(val_f_other_in_target_plan):
                 cost_component_f_other = estimate_value_on_visual_frontier(val_f_other_in_target_plan, frontier_tuples_f_other)
            
            if cost_component_f_other is not None:
                combined_est_cost_others += cost_component_f_other
            else:
                logger.info(f"Could not estimate cost component for '{f_other}' (value: {val_f_other_in_target_plan}) in plan '{target_plan_name_display}'. This component will be excluded from 'Combined Est. Cost of Other Core Features'.")
                all_other_costs_valid_for_sum = False # Mark as incomplete sum
        
        plan_total_original_fee = target_plan_series['original_fee'] # Should be same as cost_metric_for_visualization if that is 'original_fee'

        # Format the breakdown string
        # If any part of "other costs" failed, we might indicate it or show N/A for combined
        combined_others_display = f"{combined_est_cost_others:,.0f}" if combined_est_cost_others is not None else "N/A (estimation incomplete)"
        if not all_other_costs_valid_for_sum and combined_est_cost_others == 0 : # If all failed or were zero.
             combined_others_display = "N/A (estimation failed for all other features)"
        elif not all_other_costs_valid_for_sum and combined_est_cost_others > 0:
             combined_others_display = f"{combined_est_cost_others:,.0f} (estimation incomplete for some other features)"


        fee_breakdown_str = (
            f"{cost_of_analyzed_feature_on_frontier:,.0f} (Analyzed Feature)"
            f" + {combined_others_display} (Other Features)"
            f" = {plan_total_original_fee:,.0f} KRW (Plan Total)"
        )
        
        residual_analysis_table_data.append({
            'analyzed_feature_display': FEATURE_DISPLAY_NAMES_PY[feature_analyzed],
            'target_plan_name': target_plan_name_display,
            'plan_specs_string': plan_specs_string,
            'fee_breakdown_string': fee_breakdown_str
        })
        logger.info(f"Added row for '{feature_analyzed}' to residual table. Plan: {target_plan_name_display}. Breakdown: {fee_breakdown_str}")

    logger.info(f"Completed residual analysis with {len(residual_analysis_table_data)} feature entries.")
    # --- End: New section for Residual Analysis Data Preparation ---
    
    # Serialize feature frontier data to JSON
    try:
        feature_frontier_json = json.dumps(feature_frontier_data, cls=NumpyEncoder)
        logger.info(f"Serialized frontier data for {len(feature_frontier_data)} features")
    except Exception as e:
        logger.error(f"Error serializing feature frontier data: {e}")
        feature_frontier_json = "{}"
    
    # Check if we have any frontier data to display
    if not feature_frontier_data:
        logger.warning("No feature frontier data available for charts")
        
        # Skip logging for each feature frontier data addition
        other_feature_values = [] # Not plotted, but calculated for completeness/logging
        other_visual_costs = []
        other_plan_names = []
    
    # Generate the residual table HTML from residual_analysis_table_data
    residual_table_html = ""
    for entry in residual_analysis_table_data:
        residual_table_html += f"""
        <tr>
            <td>{entry['analyzed_feature_display']}</td>
            <td>{entry['target_plan_name']}</td>
            <td>{entry['plan_specs_string']}</td>
            <td>{entry['fee_breakdown_string']}</td>
        </tr>
        """
    
    # If no residual data, show a message
    if not residual_table_html:
        residual_table_html = """
        <tr>
            <td colspan="4" style="text-align: center;">No residual analysis data available</td>
        </tr>
        """
    
    # Generate the top plans HTML (typically top 20 plans)
    top_plans_html = ""
    display_limit = 50  # Display top 50 plans
    top_plans_df = df_sorted.head(display_limit)
    
    if not top_plans_df.empty:
        for _, row in top_plans_df.iterrows():
            rank = row.get('rank_number', 'N/A')
            rank_display = row.get('rank_display', str(rank))
            plan_name = row.get('plan_name', 'N/A')
            mvno = row.get('mvno', 'N/A')
            fee = row.get('original_fee', 0)
            cs_ratio = row.get('CS', 0)
            baseline_cost = row.get('B', 0)
            
            # Format numbers with commas for better readability
            fee_formatted = f"{fee:,.0f}" if isinstance(fee, (int, float)) else fee
            baseline_cost_formatted = f"{baseline_cost:,.0f}" if isinstance(baseline_cost, (int, float)) else baseline_cost
            cs_ratio_formatted = f"{cs_ratio:.4f}" if isinstance(cs_ratio, float) else cs_ratio
            
            top_plans_html += f"""
            <tr>
                <td>{rank_display}</td>
                <td>{plan_name}</td>
                <td>{mvno}</td>
                <td>{fee_formatted}</td>
                <td>{cs_ratio_formatted}</td>
                <td>{baseline_cost_formatted}</td>
            </tr>
            """
    else:
        top_plans_html = """
        <tr>
            <td colspan="6" style="text-align: center;">No ranking data available</td>
        </tr>
        """
    
    # Generate the complete rankings HTML
    all_plans_html = ""
    
    if not df_sorted.empty:
        for _, row in df_sorted.iterrows():
            rank = row.get('rank_number', 'N/A')
            rank_display = row.get('rank_display', str(rank))
            plan_name = row.get('plan_name', 'N/A')
            mvno = row.get('mvno', 'N/A')
            fee = row.get('original_fee', 0)
            cs_ratio = row.get('CS', 0)
            baseline_cost = row.get('B', 0)
            
            # Format numbers with commas for better readability
            fee_formatted = f"{fee:,.0f}" if isinstance(fee, (int, float)) else fee
            baseline_cost_formatted = f"{baseline_cost:,.0f}" if isinstance(baseline_cost, (int, float)) else baseline_cost
            cs_ratio_formatted = f"{cs_ratio:.4f}" if isinstance(cs_ratio, float) else cs_ratio
            
            all_plans_html += f"""
            <tr>
                <td>{rank_display}</td>
                <td>{plan_name}</td>
                <td>{mvno}</td>
                <td>{fee_formatted}</td>
                <td>{cs_ratio_formatted}</td>
                <td>{baseline_cost_formatted}</td>
            </tr>
            """
    else:
        all_plans_html = """
        <tr>
            <td colspan="6" style="text-align: center;">No ranking data available</td>
        </tr>
        """
    
    # Create HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title} - {timestamp_str}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; z-index: 10; }}
            tr:hover {{ background-color: #f5f5f5; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .good-value {{ color: green; font-weight: bold; }}
            .bad-value {{ color: red; }}
            .container {{ max-width: 100%; overflow-x: auto; }}
            .note {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; margin-bottom: 20px; }}
            
            /* Feature category colors */
            .core-feature {{ background-color: #e6f7ff; }}
            .cs-metrics {{ background-color: #f9f0ff; }}
            .input-feature {{ background-color: #f9f0ff; }}
            .output-feature {{ background-color: #f6ffed; }}
            
            /* Bar chart styles */
            .bar-container {{ 
                width: 100%; 
                background-color: #f1f1f1; 
                margin-top: 5px;
                border-radius: 4px;
                overflow: hidden;
            }}
            .bar {{ 
                height: 20px; 
                background-color: #4CAF50; 
                text-align: right; 
                color: white; 
                padding-right: 5px;
                border-radius: 4px;
            }}
            
            /* Feature charts grid */
            .chart-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                width: 100%;
            }}
            
            .chart-container {{
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 15px;
                position: relative;
                width: 100%;
                height: 400px;
            }}
            
            .chart-title {{
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
                text-align: center;
            }}
            
            /* Feature table */
            .feature-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            
            .feature-table th,
            .feature-table td {{
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            
            .feature-table th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            
            .feature-table caption {{
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
                caption-side: top;
            }}
            
            /* Residual Analysis table */
            .residual-table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }}
            
            .residual-table th,
            .residual-table td {{
                padding: 8px 12px;
                border: 1px solid #ddd;
                text-align: left;
            }}
            
            .residual-table th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            
            .residual-table caption {{
                font-weight: bold;
                font-size: 16px;
                margin-bottom: 10px;
                caption-side: top;
            }}
            
            /* Better table styles */
            table {{
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                border-radius: 5px;
                overflow: hidden;
            }}
            
            /* Responsive styles */
            @media (max-width: 768px) {{
                .container {{
                    padding: 0;
                }}
            }}
        </style>
        
        <!-- Include Chart.js from CDN -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js@3.7.1/dist/chart.min.js"></script>
        
        <!-- Table sorting -->
        <script>
            function sortTable(table, column, asc = true) {{
                const dirModifier = asc ? 1 : -1;
                const tBody = table.tBodies[0];
                const rows = Array.from(tBody.querySelectorAll("tr"));
                
                // Sort rows
                const sortedRows = rows.sort((a, b) => {{
                    const aColText = a.cells[column].textContent.trim();
                    const bColText = b.cells[column].textContent.trim();
                    
                    // Check if column contains numbers with commas
                    if (!isNaN(parseInt(aColText.replace(/,/g, "")))) {{
                        return dirModifier * (parseInt(aColText.replace(/,/g, "")) - parseInt(bColText.replace(/,/g, "")));
                    }} else {{
                        return dirModifier * aColText.localeCompare(bColText);
                    }}
                }});
                
                // Remove existing rows
                while (tBody.firstChild) {{
                    tBody.removeChild(tBody.firstChild);
                }}
                
                // Add sorted rows
                tBody.append(...sortedRows);
                
                // Update header classes
                table.querySelectorAll("th").forEach(th => th.classList.remove("th-sort-asc", "th-sort-desc"));
                table.querySelector(`th:nth-child(${{column + 1}})`).classList.toggle("th-sort-asc", asc);
                table.querySelector(`th:nth-child(${{column + 1}})`).classList.toggle("th-sort-desc", !asc);
            }}
            
            document.addEventListener('DOMContentLoaded', function() {{
                const tables = document.querySelectorAll('table');
                tables.forEach(table => {{
                    const headerRow = table.querySelector('tr');
                    if (headerRow) {{
                        const thList = headerRow.querySelectorAll('th');
                        thList.forEach((th, idx) => {{
                            th.addEventListener('click', () => {{
                                const isAsc = th.classList.contains('th-sort-asc');
                                sortTable(table, idx, !isAsc);
                            }});
                        }});
                    }}
                }});
            }});
        </script>
        
        <!-- Chart.js initialization for feature frontier charts -->
        <script>
        // Feature frontier data
            const FRONTIER_DATA_JSON = {feature_frontier_json};
        
        // Create feature frontier charts
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Initializing feature frontier charts...');
                
                // Check if we have data
                if (!FRONTIER_DATA_JSON || Object.keys(FRONTIER_DATA_JSON).length === 0) {{
                    console.warn('No frontier data available, charts will not be displayed');
                    document.getElementById('feature-charts-container').innerHTML = '<p>No frontier data available for visualization.</p>';
                    return;
                }}
            
            // Feature display names
            const featureDisplayNames = {{
                'basic_data_clean': 'Basic Data (GB)',
                    'daily_data_clean': 'Daily Data (GB/day)',
                'voice_clean': 'Voice Minutes',
                'message_clean': 'SMS Messages',
                    'additional_call': 'Additional Call Price (KRW)',
                'speed_when_exhausted': 'Throttled Speed (Mbps)',
                'tethering_gb': 'Tethering Data (GB)'
            }};
            
                // Create a chart for each feature
                const chartsContainer = document.getElementById('feature-charts-container');
                chartsContainer.innerHTML = ''; // Clear previous content
            
            // Create a chart for each feature
                for (const [feature, data] of Object.entries(FRONTIER_DATA_JSON)) {{
                    console.log(`Processing feature: ${{feature}}`);
                    
                    // Validate data
                    if (!data.all_values || !data.all_contributions || data.all_values.length === 0) {{
                        console.warn(`Invalid data for feature ${{feature}}, skipping`);
                        continue;
                    }}
                    
                // Create chart container
                const chartContainer = document.createElement('div');
                chartContainer.className = 'chart-container';
                chartContainer.style.width = '100vw';  // Full viewport width
                chartContainer.style.maxWidth = '100%';  // Prevent horizontal overflow
                chartContainer.style.margin = '0';      // No margin
                chartContainer.style.padding = '10px';  // Small padding inside container
                chartContainer.style.boxSizing = 'border-box'; // Include padding in width
                chartContainer.style.height = '500px';  // Taller charts
                
                // Create chart title
                const chartTitle = document.createElement('div');
                chartTitle.className = 'chart-title';
                chartTitle.textContent = featureDisplayNames[feature] || feature;
                chartContainer.appendChild(chartTitle);
                
                // Create canvas for chart
                const canvas = document.createElement('canvas');
                canvas.id = `chart-${{feature}}`;
                chartContainer.appendChild(canvas);
                
                    // Add chart container to grid
                    chartsContainer.appendChild(chartContainer);
                    
                    // Prepare data points
                    let frontierPoints = [];
                    let excludedPoints = [];
                    let unlimitedPoints = [];
                    
                    // Sort frontier points by x-value (feature value) to ensure proper line connections
                    if (data.frontier_values && data.frontier_values.length > 0) {{
                        // Create data points for frontier points
                        for (let i = 0; i < data.frontier_values.length; i++) {{
                            const x = data.frontier_values[i];
                            const y = data.frontier_contributions[i];
                            const planName = data.frontier_plan_names[i] || 'Unknown';
                            
                            // Ensure x and y are numbers
                            if (typeof x === 'number' && typeof y === 'number' && !isNaN(x) && !isNaN(y)) {{
                                frontierPoints.push({{
                                    x: x,
                                    y: y,
                                    plan_name: planName,
                                    is_frontier: true,
                                    is_excluded: false,
                                }});
                            }}
                        }}
                        
                        // Sort frontier points by x-value to ensure proper line connections
                        frontierPoints.sort((a, b) => a.x - b.x);
                    }}
                    
                    // Create data points for excluded points
                    for (let i = 0; i < data.excluded_values.length; i++) {{
                        const x = data.excluded_values[i];
                        const y = data.excluded_contributions[i];
                        const planName = data.excluded_plan_names[i] || 'Unknown';
                        
                        if (typeof x === 'number' && typeof y === 'number' && !isNaN(x) && !isNaN(y)) {{
                            excludedPoints.push({{
                                x: x,
                                y: y,
                                plan_name: planName,
                                is_frontier: false,
                                is_excluded: true,
                            }});
                        }}
                    }}
                    
                    // Create a special point for unlimited if available
                    if (data.has_unlimited && data.unlimited_value !== null) {{
                        const unlimitedValue = data.unlimited_value;
                        const unlimitedPlan = data.unlimited_plan || 'Unknown';
                        
                        // Create a special mark for unlimited (using the right edge of the chart)
                        unlimitedPoints.push({{
                            // Position at the maximum of actual data points plus 20% or at a default position
                            x: data.frontier_values.length > 0 ? Math.max(...data.frontier_values) * 1.2 : 10,
                            y: unlimitedValue,
                            plan_name: unlimitedPlan,
                            is_frontier: true,
                            is_excluded: false,
                        }});
                    }}
                    
                    // Create datasets for Chart.js
                    const datasets = [];
                    
                    // Frontier points dataset (red line)
                    if (frontierPoints.length > 0) {{
                        datasets.push({{
                            label: 'Frontier',
                        data: frontierPoints,
                        borderColor: 'rgba(255, 99, 132, 1)',
                            backgroundColor: 'rgba(255, 99, 132, 0.5)',
                            pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                            pointBorderColor: '#fff',
                        pointRadius: 6,
                            pointHoverRadius: 8,
                        showLine: true,
                            fill: false,
                            borderWidth: 2.5
                        }});
                    }}
                    
                    // Excluded points dataset (yellow points)
                    if (excludedPoints.length > 0) {{
                        datasets.push({{
                            label: 'Excluded',
                            data: excludedPoints,
                            borderColor: 'rgba(255, 205, 86, 1)',
                            backgroundColor: 'rgba(255, 205, 86, 0.5)',
                            pointBackgroundColor: 'rgba(255, 205, 86, 1)',
                            pointBorderColor: '#fff',
                            pointRadius: 5,
                            pointHoverRadius: 7,
                            showLine: false
                        }});
                    }}
                    
                    // Unlimited point dataset (purple marker with dashed vertical line)
                    if (unlimitedPoints.length > 0) {{
                        datasets.push({{
                            label: 'Unlimited',
                            data: unlimitedPoints,
                            borderColor: 'rgba(128, 0, 128, 1)',
                            backgroundColor: 'rgba(128, 0, 128, 0.5)',
                            pointBackgroundColor: 'rgba(128, 0, 128, 1)',
                            pointBorderColor: '#fff',
                            pointStyle: 'rectRot',
                            pointRadius: 10,
                            pointHoverRadius: 12,
                            showLine: false
                        }});
                        
                        // Add text label for unlimited value if present
                        const unlimitedLabel = document.createElement('div');
                        unlimitedLabel.style.position = 'absolute';
                        unlimitedLabel.style.right = '10px';
                        unlimitedLabel.style.top = '30px'; // Adjusted top position for taller chart
                        unlimitedLabel.style.backgroundColor = 'rgba(128, 0, 128, 0.1)';
                        unlimitedLabel.style.borderLeft = '3px solid rgba(128, 0, 128, 0.8)';
                        unlimitedLabel.style.padding = '4px 8px';
                        unlimitedLabel.style.borderRadius = '0 4px 4px 0';
                        unlimitedLabel.innerHTML = `<b>Unlimited</b>: ${{unlimitedPoints[0].plan_name}} (${{unlimitedPoints[0].y.toLocaleString()}} KRW)`;
                        chartContainer.appendChild(unlimitedLabel);
                    }}
                    
                    // Create chart
                    const ctx = canvas.getContext('2d');
                    new Chart(ctx, {{
                        type: 'scatter',
                        data: {{
                            datasets: datasets
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            const point = context.raw;
                                            return `${{point.plan_name}}: ${{point.x}} → ${{point.y.toLocaleString()}} KRW`;
                                        }}
                                    }}
                                }},
                                legend: {{
                                    position: 'top',
                                    labels: {{
                                        usePointStyle: true,
                                        padding: 10,
                                        boxWidth: 10,
                                        boxHeight: 10
                                    }}
                                }}
                            }},
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: featureDisplayNames[feature] || feature,
                                        font: {{
                                            size: 11
                                        }},
                                        padding: {{
                                            top: 0,
                                            bottom: 0
                                        }}
                                    }},
                                    ticks: {{
                                        font: {{
                                            size: 10
                                        }}
                                    }},
                                    min: 0 // Always start x-axis at 0
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'Price (KRW)',
                                        font: {{
                                            size: 11
                                        }},
                                        padding: {{
                                            top: 0,
                                            bottom: 0
                                        }}
                                    }},
                                    ticks: {{
                                        callback: function(value) {{
                                            return value.toLocaleString();
                                        }},
                                        font: {{
                                            size: 10
                                        }}
                                    }},
                                    min: 0 // Always start y-axis at 0
                                }}
                            }}
                        }}
                    }});
            }}
        }});
        </script>
    </head>
    <body>
        <div class="content-wrapper" style="padding: 20px;">
            <h1>{report_title} - {timestamp_str}</h1>
            
            <div class="note">
                <p>This report shows mobile plan rankings based on the Cost-Spec Ratio methodology. Plans with higher CS ratios offer better value relative to their specifications.</p>
                <p>All costs shown are in Korean Won (KRW).</p>
            </div>
        </div>
        
        <!-- Feature Frontier Charts -->
        <div class="charts-wrapper" style="width: 100%; margin: 0; padding: 0;">
            <div style="padding: 0 20px;">
                <h2>Feature Frontier Charts</h2>
                <div class="note">
                    <p>These charts show the cost frontiers for each feature, representing the optimal (minimum cost) options available at each feature value.</p>
                    <p>Red points and lines represent the frontier, while yellow points are non-frontier minimum cost options that were excluded due to lack of monotonicity.</p>
                </div>
            </div>
            <div class="chart-grid" id="feature-charts-container" style="width: 100vw; max-width: 100%; margin: 0; padding: 0;">
                <!-- Charts will be inserted here by JavaScript -->
            </div>
        </div>
        
        <div class="content-wrapper" style="padding: 20px;">
            <!-- Residual analysis table -->
            <h2>Residual Fee Analysis</h2>
            <div class="container">
                <div class="note">
                    <p>This table shows how much of a plan's fee is attributable to its analyzed feature versus other features, for plans on the feature cost frontier.</p>
                </div>
                <table class="residual-table">
                    <tr>
                        <th>Analyzed Feature</th>
                        <th>Sample Plan</th>
                        <th>Plan Core Specs</th>
                        <th>Fee Breakdown</th>
                    </tr>
                    
                    <!-- Generate table rows for residual analysis -->
                    {residual_table_html}
                </table>
            </div>
            
            <!-- All plans table -->
            <h2>Complete Plan Rankings</h2>
            <div class="container">
                <table>
                    <tr>
                        <th>Rank</th>
                        <th>Plan Name</th>
                        <th>MVNO</th>
                        <th>Fee</th>
                        <th>CS Ratio</th>
                        <th>Baseline Cost</th>
                    </tr>
                    <!-- Generate rows for all ranked plans -->
                    {all_plans_html}
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html

def save_report(html_content, timestamp, directory=None, prefix="ranking", description=None):
    """Save an HTML report to a file.
    
    Args:
        html_content: HTML content as string
        timestamp: Timestamp to use in filename
        directory: Directory to save to (optional)
        prefix: Prefix for the filename (default: "ranking")
        description: Optional description to include in filename
    
    Returns:
        Path object of the saved file
    """
    # Generate filename with timestamp
    filename_parts = [prefix]
    if description:
        filename_parts.append(description)
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{'-'.join(filename_parts)}_{timestamp_str}.html"
    
    # Determine directory
    if directory is None:
        directory = Path("./reports")
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create file path
    file_path = Path(directory) / filename
    
    # Write content to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Report saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return None

def format_plan_specs_display_string(plan_row_series, core_continuous_features, feature_display_names_map, feature_units_map, unlimited_flags_map):
    """Formats a plan's core specs into a semicolon-separated string."""
    parts = []
    if not isinstance(plan_row_series, pd.Series):
        logger.warning("format_plan_specs_display_string: plan_row_series is not a Series.")
        return "Invalid plan data"

    for f_col in core_continuous_features:
        display_name = feature_display_names_map.get(f_col, f_col)
        value_str = "N/A"
        
        if f_col in plan_row_series:
            value = plan_row_series[f_col]
            unit = feature_units_map.get(f_col, "")
            
            is_unlimited = False
            unlimited_flag_col = unlimited_flags_map.get(f_col)
            if unlimited_flag_col and unlimited_flag_col in plan_row_series and plan_row_series[unlimited_flag_col]:
                is_unlimited = True

            if is_unlimited:
                value_str = f"Unlimited {unit}".strip()
            elif pd.notna(value) and isinstance(value, (int,float)):
                value_str = f"{value:.1f} {unit}".strip() if isinstance(value, float) else f"{value} {unit}".strip()
            # else value_str remains "N/A" or could be refined for other types
        
        parts.append(f"{display_name}: {value_str}")
            
    return "; ".join(parts)
