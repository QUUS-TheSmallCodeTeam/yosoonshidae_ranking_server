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
            
            # Pop points that this one dominates (more value for less or equal cost)
            # Don't log each individual point being popped
            points_popped = 0
            while actual_frontier_plans_series_list and \
                  current_value > last_frontier_plan_series[feature] and \
                  current_cost <= last_frontier_plan_series[cost_metric_for_visualization]:
                points_popped += 1
                actual_frontier_plans_series_list.pop()
                if actual_frontier_plans_series_list:
                    last_frontier_plan_series = actual_frontier_plans_series_list[-1]
                    last_value = last_frontier_plan_series[feature]
                    last_cost = last_frontier_plan_series[cost_metric_for_visualization]
            
            # Only log if points were actually popped
            if points_popped > 0:
                logger.debug(f"Removed {points_popped} dominated points for feature {feature}")
            
            # Add point if:
            # 1. It offers more value at the same cost
            # 2. It offers more value at a higher cost (maintaining monotonicity)
            # 3. It's the last point in the sequence (to ensure connection)
            if (current_value > last_value and current_cost >= last_cost) or \
                (current_value > last_value and abs(current_cost - last_cost) < 1.0):  # Handle floating point comparison
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
                    # Only add if it maintains monotonicity
                    last_frontier_point = actual_frontier_plans_series_list[-1]
                    if max_feature_value > last_frontier_point[feature] and min_cost_for_max_value >= last_frontier_point[cost_metric_for_visualization]:
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
            feature_frontier_data[feature] = {
                # Pass only the combined frontier/excluded lists to JS under generic 'all_...' keys
                'all_values': js_chart_values, 
                'all_contributions': js_chart_costs, # JS expects all_contributions for costs
                'all_is_frontier': js_chart_is_frontier,
                'all_is_excluded': js_chart_is_excluded,
                # all_is_unlimited will be determined in JS based on the separate unlimited point data
                'all_plan_names': js_chart_plan_names,
                
                # Keep specific lists for clarity and direct use in JS if needed, though JS primarily uses 'all_...'
                'frontier_values': frontier_feature_values,
                'frontier_contributions': frontier_visual_costs, 
                'frontier_plan_names': frontier_plan_names,
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
    
    # Create HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title} - {timestamp_str}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
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
                grid-template-columns: repeat(2, 1fr); /* Changed to 2 columns */
                gap: 20px; /* Increased gap for better separation */
                margin-top: 15px;
            }}
            
            .chart-container {{
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 15px; /* Increased padding */
                background-color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                height: 380px; /* Increased height */
                width: 100%; /* Fill grid cell width */
                max-width: 650px; /* Allow larger charts, capped */
                margin: 0 auto; /* Center if grid cell is wider than max-width */
                position: relative; /* Ensure absolute positioning of unlimited label is contained */
            }}
            
            .chart-title {{
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
                text-align: center;
            }}
            
            .hidden {{ display: none; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0/dist/chartjs-plugin-annotation.min.js"></script>
    </head>
    <body>
        <h1>{report_title}</h1>
        <p>Generated: {timestamp_str}</p>
        
 
    """
    
    # Add method explanation section - CS method only now
    html += """
        <h2>Cost-Spec Ratio Explanation</h2>
        <div class="note">
            <p><strong>Cost-Spec Ratio (CS)</strong> is a method that evaluates the value of mobile plans by comparing their fees to a theoretical baseline cost.</p>
            <p>In this analysis:</p>
            <ul>
                <li><strong>Baseline Feature Cost (E):</strong> For each feature value, the minimum fee among plans with that value</li>
                <li><strong>Plan Baseline Cost (B):</strong> Sum of baseline costs for all features in a plan</li>
                <li><strong>Cost-Spec Ratio (CS):</strong> B / fee - the ratio of theoretical cost to actual fee</li>
            </ul>
            <p>Plans are ranked based on their CS Ratio (higher is better).</p>
        </div>
        """
    
    # Add feature contributions section (new section)
    html += """
        <h2>Feature Contributions to Baseline Cost</h2>
        <div class="container">
        <div class="note">
            <p>This section shows how each feature contributes to the baseline cost calculation. The baseline cost for each feature is determined by finding the minimum fee among plans with that feature value. Features with higher contributions have a greater impact on the overall ranking.</p>
        </div>
        <table>
            <tr>
                <th>Feature</th>
                <th>Avg Contribution (KRW)</th>
                <th>Min Contribution (KRW)</th>
                <th>Max Contribution (KRW)</th>
                <th>% of Baseline Cost</th>
                <th>Contribution Distribution</th>
            </tr>
    """
    
    # Sort features by average contribution (descending)
    sorted_features = sorted(
        feature_contribution_data.items(),
        key=lambda x: x[1]['avg_contribution'],
        reverse=True
    )
    
    # Add rows for each feature contribution
    for feature_name, data in sorted_features:
        avg_contrib = data['avg_contribution']
        min_contrib = data['min_contribution']
        max_contrib = data['max_contribution']
        percentage = data['percentage']
        
        # Create a simple bar chart for the percentage
        bar_width = min(percentage, 100)  # Cap at 100%
        
        html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>{int(avg_contrib):,} KRW</td>
            <td>{int(min_contrib):,} KRW</td>
            <td>{int(max_contrib):,} KRW</td>
            <td>{percentage:.1f}%</td>
            <td>
                <div class="bar-container">
                    <div class="bar" style="width: {bar_width}%">{percentage:.1f}%</div>
                </div>
            </td>
        </tr>
        """
    
    html += """
        </table>
        </div>
        """
    
    # Add Features List
    if used_features:
        html += """
            <h2>Features Used</h2>
        <div class="container">
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Category</th>
                </tr>
            """
        
        for feature in used_features:
            category = "Output" if feature != "fee" else "Input"
            html += f"""
            <tr>
                <td>{feature}</td>
                <td>{category}</td>
            </tr>
            """
        
        html += """
        </table>
        </div>
        """
    
    # Add rankings table
    html += """
        <h2>Plan Rankings</h2>
        <div class="container">
        <table>
            <tr>
                <th>Rank</th>
                <th>Plan Name</th>
                <th>MVNO</th>
                <th>Fee (KRW)</th>
                <th>Original Fee (KRW)</th>
                <th>Baseline Cost (B)</th>
                <th>CS Ratio</th>
                <th>Data (GB)</th>
                <th>Voice (min)</th>
                <th>Message (SMS)</th>
                <th>Additional Call (min)</th>
                <th>Throttled Speed (Mbps)</th>
                <th>5G</th>
            </tr>
    """
    
    # Generate table rows
    for _, row in df_sorted.iterrows():
        # Format rank
        rank = int(row['rank_number']) if 'rank_number' in row and not pd.isna(row['rank_number']) else ""
        rank_str = f"{rank}" if rank else ""
        
        # Get plan data
        plan_name = row['plan_name'] if 'plan_name' in row else ""
        mvno = row['mvno'] if 'mvno' in row else ""
        fee = int(row['fee']) if 'fee' in row and not pd.isna(row['fee']) else 0
        original_fee = int(row['original_fee']) if 'original_fee' in row and not pd.isna(row['original_fee']) else 0
        
        # CS-specific metrics
        baseline_cost = int(row['B']) if 'B' in row and not pd.isna(row['B']) else 0
        cs_ratio = row['CS'] if 'CS' in row else ""
        
        # Format CS ratio with proper handling of types
        formatted_cs_ratio = f"{cs_ratio:.4f}" if isinstance(cs_ratio, float) else str(cs_ratio)
        
        # Get feature data
        data_gb = row['basic_data_clean'] if 'basic_data_clean' in row else "N/A"
        voice = row['voice_clean'] if 'voice_clean' in row else "N/A"
        message = row['message_clean'] if 'message_clean' in row else "N/A"
        additional_call = row['additional_call'] if 'additional_call' in row else "N/A"
        is_5g = "Yes" if row.get('is_5g') == 1 else "No"
        
        # Throttled speed data - using raw speed since that's used in calculations
        raw_speed = row['speed_when_exhausted'] if 'speed_when_exhausted' in row else 0
        
        # Format throttled speed
        if 'has_unlimited_speed' in row and row['has_unlimited_speed'] == 1:
            throttled_speed = "Unlimited"
        elif raw_speed > 0:
            throttled_speed = f"{raw_speed} Mbps"
        else:
            throttled_speed = "N/A"
        
        # Handle unlimited values
        if 'basic_data_unlimited' in row and row['basic_data_unlimited'] == 1:
            data_gb = "Unlimited"
        if 'voice_unlimited' in row and row['voice_unlimited'] == 1:
            voice = "Unlimited"
        if 'message_unlimited' in row and row['message_unlimited'] == 1:
            message = "Unlimited"
        
        # Generate the row HTML
        html += f"""
            <tr>
                <td>{rank_str}</td>
                <td>{plan_name}</td>
                <td>{mvno}</td>
                <td>{fee:,}</td>
                <td>{original_fee:,}</td>
                <td>{baseline_cost:,}</td>
                <td class="good-value">{formatted_cs_ratio}</td>
                <td>{data_gb}</td>
                <td>{voice}</td>
                <td>{message}</td>
                <td>{additional_call}</td>
                <td>{throttled_speed}</td>
                <td>{is_5g}</td>
            </tr>
        """
    
    # Close the rankings table
    html += """
        </table>
        </div>
    """
    
    # Add calculation summary table with updated metrics
    html += f"""
        <h2>Calculation Summary</h2>
        <div class="container">
            <div class="note">
                <p><strong>Feature Importance:</strong> This summary shows the overall contribution of features to the baseline cost, which determines the CS ratio and ranking.</p>
            </div>
            
            <div style="display: flex; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px; margin-right: 20px;">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Plans Analyzed</td>
                            <td>{total_plans}</td>
                        </tr>
                        <tr>
                            <td>Top Contributing Feature</td>
                            <td>{sorted_features[0][0] if sorted_features else 'N/A'}</td>
                        </tr>
                        <tr>
                            <td>Avg Baseline Cost</td>
                            <td>{int(df['B'].mean()) if 'B' in df.columns else 0:,} KRW</td>
                        </tr>
                        <tr>
                            <td>Top Plan</td>
                            <td>{top_plan_name}</td>
                        </tr>
                        <tr>
                            <td>Top Plan CS Ratio</td>
                            <td>{top_plan_value}</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
    """
    
    # Add Feature Frontier Charts section
    html += """
        <h2>Feature Frontier Analysis</h2>
        <div class="container">
            <div class="note">
                <p><strong>Feature Frontier Analysis:</strong> These charts show how each feature contributes to the baseline cost. Points on the frontier (red) represent the minimum cost for each feature value and are used in the baseline calculation. The baseline cost increases as feature value increases, reflecting that higher feature values typically cost more.</p>
            </div>
    """
    
    # Add div for the feature charts grid
    html += """
            <div class="chart-grid" id="feature-charts-container">
                <!-- Feature charts will be generated here -->
            </div>
    """
    
    # Close container
    html += """
        </div>
    """
    
    # Add JavaScript for feature frontier charts
    script_template = """
        <script>
        // Feature frontier data
        const featureFrontierData = FRONTIER_DATA_JSON;
        
        // Create feature frontier charts
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing feature frontier charts...');
            
            // Check if we have data
            if (!featureFrontierData || Object.keys(featureFrontierData).length === 0) {
                console.warn('No frontier data available, charts will not be displayed');
                document.getElementById('feature-charts-container').innerHTML = '<p>No frontier data available for visualization.</p>';
                return;
            }
            
            // Feature display names
            const featureDisplayNames = {
                'basic_data_clean': 'Basic Data (GB)',
                'daily_data_clean': 'Daily Data (GB)',
                'voice_clean': 'Voice Minutes',
                'message_clean': 'SMS Messages',
                'additional_call': 'Additional Call Minutes',
                'speed_when_exhausted': 'Throttled Speed (Mbps)',
                'tethering_gb': 'Tethering Data (GB)'
            };
            
            // Get the container
            const container = document.getElementById('feature-charts-container');
            if (!container) {
                console.error('Could not find feature charts container');
                return;
            }
            
            // Create a chart for each feature
            for (const [feature, data] of Object.entries(featureFrontierData)) {
                console.log(`Processing feature: ${feature}`);
                
                // Validate data
                if (!data.all_values || !data.all_contributions || data.all_values.length === 0) {
                    console.warn(`Invalid data for feature ${feature}, skipping`);
                    continue;
                }
                
                // Create chart container
                const chartContainer = document.createElement('div');
                chartContainer.className = 'chart-container';
                
                // Create chart title
                const chartTitle = document.createElement('div');
                chartTitle.className = 'chart-title';
                const hasUnlimited = data.has_unlimited ? ' U:1' : '';
                // Updated title to exclude 'Other' points count
                chartTitle.textContent = (featureDisplayNames[feature] || feature) + 
                    ` (F:${data.frontier_values.length} E:${data.excluded_values.length}${hasUnlimited})`;
                chartContainer.appendChild(chartTitle);
                
                // Create canvas for chart
                const canvas = document.createElement('canvas');
                canvas.id = "chart-" + feature;
                chartContainer.appendChild(canvas);
                
                // Add to main container
                container.appendChild(chartContainer);
                
                // Prepare chart data points for frontier points
                try {
                const frontierPoints = [];
                    const excludedPoints = [];
                    // const otherPoints = []; // No longer creating 'otherPoints' dataset
                    const unlimitedPoints = [];
                    
                    // Create data points for frontier points
                    for (let i = 0; i < data.frontier_values.length; i++) {
                        const x = data.frontier_values[i];
                        const y = data.frontier_contributions[i];
                        const planName = data.frontier_plan_names[i] || 'Unknown';
                        
                        // Ensure x and y are numbers
                        if (typeof x === 'number' && typeof y === 'number' && !isNaN(x) && !isNaN(y)) {
                            frontierPoints.push({
                                x: x,
                                y: y,
                                plan_name: planName,
                                is_frontier: true,
                                is_excluded: false,
                                is_unlimited: false
                            });
                        }
                    }
                    
                    // Sort frontier points by x-value (feature value) to ensure proper line connections
                    frontierPoints.sort((a, b) => a.x - b.x);
                    
                    // Create data points for excluded points
                    for (let i = 0; i < data.excluded_values.length; i++) {
                        const x = data.excluded_values[i];
                        const y = data.excluded_contributions[i];
                        const planName = data.excluded_plan_names[i] || 'Unknown';
                        
                        if (typeof x === 'number' && typeof y === 'number' && !isNaN(x) && !isNaN(y)) {
                            excludedPoints.push({
                                x: x,
                                y: y,
                                plan_name: planName,
                                is_frontier: false,
                                is_excluded: true,
                                is_unlimited: false
                            });
                        }
                    }
                    
                    // 'Other points' data processing is removed here
                    
                    if (data.has_unlimited && data.unlimited_value !== null) {
                        const unlimitedValue = data.unlimited_value;
                        const unlimitedPlan = data.unlimited_plan || 'Unknown';
                        
                        // Create a special mark for unlimited (using the right edge of the chart)
                        unlimitedPoints.push({
                            // Position at the maximum of actual data points plus 20% or at a default position
                            x: data.frontier_values.length > 0 ? Math.max(...data.frontier_values) * 1.2 : 10,
                            y: unlimitedValue,
                            plan_name: unlimitedPlan,
                            is_frontier: true,
                            is_excluded: false,
                            is_unlimited: true,
                            originalValue: "Unlimited"
                        });
                    }
                    
                    if (frontierPoints.length === 0 && excludedPoints.length === 0 && unlimitedPoints.length === 0) {
                        console.warn(`No valid data points for ${feature}, skipping chart`);
                        continue;
                    }
                    
                    // Ensure x-axis starts at 0 if first point is not at x=0
                    const startAtZero = frontierPoints.length > 0 && frontierPoints[0].x > 0;
                    if (startAtZero) {
                        // If there's no point at x=0, add one that connects to the first point
                        frontierPoints.unshift({
                            x: 0,
                            y: 0, // Start at 0 cost
                            plan_name: "Free Baseline",
                            is_frontier: true,
                            is_excluded: false,
                            is_unlimited: false
                        });
                    }
                    
                    // Create Chart.js datasets
                    const frontierDataset = {
                        label: 'Frontier Points',
                        data: frontierPoints,
                        backgroundColor: 'rgba(255, 0, 0, 1)',
                        borderColor: 'rgba(255, 0, 0, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        showLine: true,
                        tension: 0.1,
                        borderWidth: 2,
                        fill: false
                    };
                    
                    // Add a dataset for the frontier line area
                    const frontierAreaDataset = {
                        label: 'Frontier Line',
                        data: frontierPoints,
                        backgroundColor: 'rgba(255, 0, 0, 0.1)',
                        borderColor: 'rgba(255, 0, 0, 0)',
                        pointRadius: 0,
                        showLine: true,
                        tension: 0.1,
                        fill: 'origin',
                        hidden: false,
                        // Hide from legend
                        hidden: false,
                        showLine: true,
                        display: true,
                        spanGaps: true
                    };
                    
                    const excludedDataset = {
                        label: 'Excluded Points',
                        data: excludedPoints,
                        backgroundColor: 'rgba(255, 165, 0, 0.8)',  // Orange for excluded
                        borderColor: 'rgba(255, 165, 0, 0.8)',
                        pointRadius: 4,
                        pointHoverRadius: 7,
                        showLine: false
                    };
                    
                    const unlimitedDataset = {
                        label: 'Unlimited',
                        data: unlimitedPoints,
                        backgroundColor: 'rgba(128, 0, 128, 1)',  // Purple for unlimited
                        borderColor: 'rgba(128, 0, 128, 1)',
                        pointRadius: 6,
                        pointHoverRadius: 9,
                        pointStyle: 'star',
                        showLine: false
                    };
                    
                    // Determine which datasets to include
                    const datasets = [];
                    // if (otherPoints.length > 0) datasets.push(otherDataset); // Removed otherDataset
                    if (excludedPoints.length > 0) datasets.push(excludedDataset);
                    if (unlimitedPoints.length > 0) datasets.push(unlimitedDataset);
                    if (frontierPoints.length > 0) {
                        if (frontierPoints.length > 1) datasets.push(frontierAreaDataset);
                        datasets.push(frontierDataset);
                    }
                    
                    // Create Chart.js chart options with possible unlimited annotation
                    const chartOptions = {
                            responsive: true,
                        maintainAspectRatio: false, // Set to false to fill container dimensions
                        // aspectRatio: 1.8, // No longer primary driver if maintainAspectRatio is false
                        plugins: {
                            tooltip: {
                                titleFont: {
                                    size: 12
                                },
                                bodyFont: {
                                    size: 11
                                },
                                callbacks: {
                                    label: function(context) {
                                            const point = context.raw;
                                        // Don't show tooltip for area dataset
                                        if (context.dataset.label === 'Frontier Line') return null;
                                        
                                        // Base info
                                        const tooltipLines = [
                                            "Plan: " + point.plan_name,
                                            "Value: " + (point.is_unlimited ? "Unlimited" : point.x),
                                            "Cost: " + point.y.toLocaleString() + " KRW"
                                        ];
                                        
                                        // Add point type explanation
                                        if (point.is_frontier) {
                                            tooltipLines.push("Type: Frontier point - used in baseline cost");
                                        } else if (point.is_excluded) {
                                            tooltipLines.push("Type: Excluded - minimum cost for value but not monotonic");
                                        } else if (point.is_unlimited) {
                                            tooltipLines.push("Type: Unlimited value");
                                        } else {
                                            tooltipLines.push("Type: Other - not minimum cost for value");
                                        }
                                        
                                        return tooltipLines;
                                    }
                                }
                            },
                            legend: {
                                    position: 'top',
                                labels: {
                                    font: {
                                        size: 11
                                    },
                                    boxWidth: 12,
                                    padding: 8,
                                    filter: function(legendItem, chartData) {
                                        // Don't show frontier area in legend
                                        return legendItem.text !== 'Frontier Line';
                                    }
                                }
                            },
                                                            title: {
                                    display: false
                                }
                        },
                        scales: {
                            x: {
                                title: {
                                        display: true,
                                    text: featureDisplayNames[feature] || feature,
                                    font: {
                                        size: 11
                                    },
                                    padding: {
                                        top: 0,
                                        bottom: 0
                                    }
                                },
                                ticks: {
                                    font: {
                                        size: 10
                                    },
                                    min: 0 // Always start x-axis at 0
                                }
                            },
                            y: {
                                title: {
                                        display: true,
                                    text: 'Original Fee (KRW)',
                                    font: {
                                        size: 11
                                    }
                                },
                                ticks: {
                                    callback: function(value) {
                                            return value.toLocaleString();
                                    },
                                    font: {
                                        size: 10
                                    },
                                    min: 0 // Always start y-axis at 0
                                }
                            }
                        },
                        layout: {
                            padding: {
                                left: 0,
                                right: 0,
                                top: 0,
                                bottom: 0
                            }
                        }
                    };
                    
                    new Chart(canvas, {
                        type: 'scatter',
                        data: {
                            datasets: datasets
                        },
                        options: chartOptions
                    });
                    
                    // Add text label for unlimited value if present
                    if (data.has_unlimited && data.unlimited_value !== null) {
                        const unlimitedLabel = document.createElement('div');
                        unlimitedLabel.style.position = 'absolute';
                        unlimitedLabel.style.right = '10px';
                        unlimitedLabel.style.top = '30px'; // Adjusted top position for taller chart
                        // unlimitedLabel.style.transform = 'translateY(-50%)'; // Keep if vertical centering is desired relative to top
                        unlimitedLabel.style.backgroundColor = 'rgba(128, 0, 128, 0.1)';
                        unlimitedLabel.style.borderLeft = '3px solid rgba(128, 0, 128, 0.8)';
                        unlimitedLabel.style.padding = '4px 8px';
                        unlimitedLabel.style.fontSize = '11px';
                        unlimitedLabel.style.color = '#333';
                        unlimitedLabel.style.borderRadius = '0 4px 4px 0';
                        unlimitedLabel.innerHTML = `<strong>Unlimited:</strong> ${data.unlimited_value.toLocaleString()} KRW<br>Plan: ${data.unlimited_plan}`;
                        chartContainer.style.position = 'relative';
                        chartContainer.appendChild(unlimitedLabel);
                    }
                } catch (err) {
                    console.error("Error creating chart for " + feature + ":", err);
                }
            }
        });
        </script>
    """
    
    # Replace the placeholder with actual JSON data
    script_html = script_template.replace('FRONTIER_DATA_JSON', feature_frontier_json)
    html += script_html
    
    # --- Start: New HTML section for Residual Analysis Table ---
    logger.info(f"Preparing to add Residual Analysis Table. Number of rows in residual_analysis_table_data: {len(residual_analysis_table_data)}")
    if residual_analysis_table_data:
        logger.info(f"First row of residual_analysis_table_data (sample): {residual_analysis_table_data[0] if residual_analysis_table_data else 'Empty'}")
        html += """
        <h2>Residual Original Fee Analysis (Based on Minimum Value Points from Each Feature's Visual Frontier)</h2>
        <div class="container">
            <div class="note">
                <p>This table examines a specific plan selected from each feature's <strong>visual frontier</strong> (which is plotted using original fees). 
                The selected plan is the one on that frontier which has the <em>minimum value for the analyzed feature</em> (e.g., for 'Basic Data' analysis, it's the frontier plan with the least Basic Data). 
                Tie-breaking favors lower original fee, then richer other specs.
                The 'Original Fee Breakdown' shows: 
                [Cost of Analyzed Feature at its Frontier Point] + [Estimated Combined Original Fee of Other Core Features in that Plan] = [Plan's Total Original Fee].
                This illustrates how the plan's total original fee is composed when anchored at that feature's starting frontier point.</p>
            </div>
            <table>
                <thead>
                <tr>
                    <th>Analyzed Feature</th>
                    <th>Target Plan (from its visual frontier, at min value for analyzed feature)</th>
                    <th>Plan Specifications</th>
                    <th>Original Fee Breakdown (KRW)</th>
                </tr>
                </thead>
                <tbody>
        """
        for row_data in residual_analysis_table_data:
            html += f"""
                <tr>
                    <td>{row_data['analyzed_feature_display']}</td>
                    <td>{row_data['target_plan_name']}</td>
                    <td>{row_data['plan_specs_string']}</td>
                    <td>{row_data['fee_breakdown_string']}</td>
                </tr>
            """
        html += """
            </tbody>
            </table>
        </div>
        """
    # --- End: New HTML section for Residual Analysis Table ---
    
    # Close HTML
    html += """
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
