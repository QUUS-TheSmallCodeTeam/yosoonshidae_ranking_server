"""
Report Utilities Module

This module provides utility functions and classes for report generation.
"""

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import os
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

# Define unlimited flag mappings (copied from cost_spec.py)
UNLIMITED_FLAGS = {
    'basic_data_clean': 'basic_data_unlimited',
    'daily_data_clean': 'daily_data_unlimited',
    'voice_clean': 'voice_unlimited',
    'message_clean': 'message_unlimited',
    'speed_when_exhausted': 'has_unlimited_speed'
}

# Feature units for formatting
FEATURE_UNITS = {
    'basic_data_clean': 'GB/month',
    'daily_data_clean': 'GB/day',
    'voice_clean': 'min',
    'message_clean': 'SMS',
    'additional_call': 'KRW/call', # Or appropriate unit
    'speed_when_exhausted': 'Mbps',
    'tethering_gb': 'GB'
}

# Python equivalent of featureDisplayNames for use in HTML generation
FEATURE_DISPLAY_NAMES = {
    'basic_data_clean': 'Basic Data',
    'daily_data_clean': 'Daily Data',
    'voice_clean': 'Voice',
    'message_clean': 'SMS',
    'additional_call': 'Additional Call',
    'speed_when_exhausted': 'Throttled Speed',
    'tethering_gb': 'Tethering'
}

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that can handle NumPy types."""
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