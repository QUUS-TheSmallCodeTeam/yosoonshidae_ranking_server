"""
Cost-Spec Ratio implementation for MVNO plan ranking.

This module calculates plan rankings based on a cost-spec ratio approach
which compares each plan's actual fee with a theoretical baseline cost derived
from a monotonic frontier of feature costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Feature set definitions
FEATURE_SETS = {
    'basic': [
        'basic_data_clean', 'basic_data_unlimited',
        'daily_data_clean', 'daily_data_unlimited',
        'voice_clean', 'voice_unlimited',
        'message_clean', 'message_unlimited',
        'additional_call', 'is_5g',
        'tethering_gb', 'has_throttled_data',
        'has_unlimited_speed', 'speed_when_exhausted'
    ]
}

# Unlimited flag mappings
UNLIMITED_FLAGS = {
    'basic_data_clean': 'basic_data_unlimited',
    'daily_data_clean': 'daily_data_unlimited',
    'voice_clean': 'voice_unlimited',
    'message_clean': 'message_unlimited',
    'speed_when_exhausted': 'has_unlimited_speed'
}

# Use all features for frontier calculation, not just core ones
CORE_FEATURES = FEATURE_SETS['basic']

def create_robust_monotonic_frontier(df_feature_specific: pd.DataFrame, 
                                   feature_col: str, 
                                   cost_col: str) -> pd.Series:
    """
    Create a robust, strictly increasing monotonic frontier for a given feature.
    This ensures the frontier "crawls the bottom" of optimal points.
    
    Args:
        df_feature_specific: DataFrame filtered for non-unlimited values of the specific feature.
        feature_col: The name of the feature column (e.g., 'basic_data_clean').
        cost_col: The name of the column representing the cost for that feature (e.g., 'fee' or 'contribution_feature').
                  For calculate_feature_frontiers, this will typically be the overall plan 'fee'.
        
    Returns:
        A pandas Series where the index is the feature value and the values are the frontier costs.
    """
    if df_feature_specific.empty:
        return pd.Series(dtype=float)

    # Step 1: Identify candidate points (minimum cost for each unique feature value)
    # Sort by feature_col, then by cost_col to handle ties consistently (though idxmin handles it)
    candidate_points_df = df_feature_specific.loc[df_feature_specific.groupby(feature_col)[cost_col].idxmin()]
    candidate_points_df = candidate_points_df.sort_values(by=[feature_col, cost_col])
    
    candidate_details = []
    for _, row in candidate_points_df.iterrows():
        candidate_details.append({
            'value': row[feature_col],
            'cost': row[cost_col]
            # We don't need plan_name here for cost_spec.py's frontier generation
        })

    # Step 2: Build the true monotonic frontier with minimum 1 KRW cost increase rule
    actual_frontier_stack = []
    for candidate in candidate_details:
        # Part 1: Bottom-crawling based on cost.
        # If current candidate is strictly cheaper than stack top, stack top is suboptimal.
        while actual_frontier_stack and candidate['cost'] < actual_frontier_stack[-1]['cost']:
            actual_frontier_stack.pop()
        
        # Part 2: Add candidate if it forms a valid next step from the (new) stack top.
        if not actual_frontier_stack:
            # If stack is empty, add current candidate.
            actual_frontier_stack.append(candidate)
        else:
            last_frontier_point = actual_frontier_stack[-1]
            # Conditions for adding to the frontier:
            # 1. Candidate's feature value must be strictly greater.
            # 2. Candidate's cost must be strictly greater.
            # 3. Cost increase must be at least 1.0 KRW.
            if (candidate['value'] > last_frontier_point['value'] and
                candidate['cost'] > last_frontier_point['cost'] and
                (candidate['cost'] - last_frontier_point['cost']) >= 1.0):
                actual_frontier_stack.append(candidate)
            # Else: Candidate is not added (due to plateau, insufficient increase, or same feature value).
            
    if not actual_frontier_stack:
        return pd.Series(dtype=float)

    # Convert stack to pandas Series
    frontier_s = pd.Series({p['value']: p['cost'] for p in actual_frontier_stack})
    frontier_s = frontier_s.sort_index() # Ensure it's sorted by feature value
    return frontier_s

def calculate_feature_frontiers(df: pd.DataFrame, features: List[str], 
                              unlimited_flags: Dict[str, str], 
                              fee_column: str = 'fee') -> Dict[str, pd.Series]:
    """
    Compute cost frontiers for each feature using the robust monotonicity logic.
    The `fee_column` here is the overall plan fee, used to establish the initial feature cost frontiers.
    """
    frontiers = {}
    
    # Define the cost column to be used for creating feature frontiers.
    # This should be 'original_fee' as per the new requirement for B calculation.
    cost_col_for_frontier_creation = 'original_fee'

    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in dataframe for frontier calculation, skipping")
            continue

        if cost_col_for_frontier_creation not in df.columns:
            logger.error(f"Cost column '{cost_col_for_frontier_creation}' not found in DataFrame. Cannot calculate frontiers.")
            continue

        if feature in unlimited_flags.values():
            continue

        unlimited_flag = unlimited_flags.get(feature)
        
        if unlimited_flag and unlimited_flag in df.columns:
            # Process non-unlimited plans for this feature
            df_non_unlimited = df[(df[unlimited_flag] == 0) & df[cost_col_for_frontier_creation].notna()].copy()
            if not df_non_unlimited.empty:
                robust_frontier = create_robust_monotonic_frontier(df_non_unlimited, feature, cost_col_for_frontier_creation)
                if not robust_frontier.empty:
                    frontiers[feature] = robust_frontier
                    logger.info(f"Created ROBUST monotonic frontier for {feature} with {len(robust_frontier)} points using '{cost_col_for_frontier_creation}'")
                else:
                    logger.warning(f"Robust frontier for {feature} (using '{cost_col_for_frontier_creation}') is empty.")
            else: # This else corresponds to `if not df_non_unlimited.empty:`
                logger.info(f"No non-unlimited plans with valid '{cost_col_for_frontier_creation}' for {feature} to build its main frontier.")

            # Process unlimited plans for this feature
            unlimited_plans_df = df[(df[unlimited_flag] == 1) & df[cost_col_for_frontier_creation].notna()]
            if not unlimited_plans_df.empty:
                min_cost_unlimited = unlimited_plans_df[cost_col_for_frontier_creation].min()
                frontiers[unlimited_flag] = pd.Series([min_cost_unlimited])
                logger.info(f"Added unlimited case for {feature} (as {unlimited_flag}) with '{cost_col_for_frontier_creation}' {min_cost_unlimited}")
            else: # This else corresponds to `if not unlimited_plans_df.empty:` for unlimited_plans_df
                logger.info(f"No unlimited plans with valid '{cost_col_for_frontier_creation}' found for {feature} (flag: {unlimited_flag})")

        else:  # This else corresponds to `if unlimited_flag and unlimited_flag in df.columns:`
            # Feature does not have an unlimited flag
            df_feature_specific = df[df[cost_col_for_frontier_creation].notna()].copy()
            if not df_feature_specific.empty:
                robust_frontier = create_robust_monotonic_frontier(df_feature_specific, feature, cost_col_for_frontier_creation)
                if not robust_frontier.empty:
                    frontiers[feature] = robust_frontier
                    logger.info(f"Created ROBUST monotonic frontier for {feature} (no unlimited option, using '{cost_col_for_frontier_creation}') with {len(robust_frontier)} points")
                else:  # This else corresponds to `if not robust_frontier.empty:` for robust_frontier
                    logger.warning(f"Robust frontier for {feature} (no unlimited option, using '{cost_col_for_frontier_creation}') is empty.")
            else:  # This else corresponds to `if not df_feature_specific.empty:`
                logger.warning(f"No plans with valid '{cost_col_for_frontier_creation}' for {feature} (no unlimited option) to build frontier.")
    
    return frontiers

def estimate_frontier_value(feature_value: float, frontier: pd.Series) -> float:
    """
    Estimate the frontier value for a given feature value.
    
    Args:
        feature_value: The feature value to estimate
        frontier: The frontier series indexed by feature values
        
    Returns:
        The estimated frontier value
    """
    if frontier.empty:
        logger.warning(f"Attempting to estimate value from an empty frontier for feature value {feature_value}. Returning 0.0.")
        return 0.0

    if feature_value in frontier.index:
        # Exact match
        return frontier[feature_value]
    
    # np.searchsorted finds the insertion point to maintain order.
    # The feature values in frontier.index are sorted.
    idx = np.searchsorted(frontier.index, feature_value)
    
    if idx == 0:
        # Feature value is lower than any in frontier, use the cost of the smallest feature value.
        # This is extrapolation using the first point.
        return frontier.iloc[0]
    elif idx == len(frontier.index): # Corrected to len(frontier.index) as idx can be equal to length
        # Feature value is higher than any in frontier, use the cost of the largest feature value.
        # This is extrapolation using the last point.
        return frontier.iloc[-1]
    else:
        # Feature value is between two frontier points. Perform linear interpolation.
        x1 = frontier.index[idx-1]
        y1 = frontier.iloc[idx-1]
        x2 = frontier.index[idx]
        y2 = frontier.iloc[idx]

        # Prevent division by zero if feature values are identical (should not happen with strictly monotonic frontier)
        if x2 == x1:
            logger.warning(f"Frontier points for interpolation have same feature value ({x1}). Returning cost of first point ({y1}).")
            return y1 
        
        # Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        interpolated_cost = y1 + (feature_value - x1) * (y2 - y1) / (x2 - x1)
        return interpolated_cost

def calculate_plan_baseline_cost(row: pd.Series, frontiers: Dict[str, pd.Series],
                               unlimited_flags: Dict[str, str]) -> float:
    """
    Calculate the theoretical baseline cost for a single plan using feature frontiers.
    
    Args:
        row: Series containing a single plan's data
        frontiers: Dictionary mapping features to their cost frontiers
        unlimited_flags: Mapping of feature columns to their unlimited flag columns
        
    Returns:
        Total baseline cost for the plan
    """
    total_cost = 0.0
    
    # Calculate cost for each feature (excluding unlimited flags which are handled with their features)
    for feature in [f for f in CORE_FEATURES if f not in unlimited_flags.values()]:
        # Skip if feature not available
        if feature not in row:
            continue
            
        # Check if this feature has an unlimited flag
        unlimited_flag = unlimited_flags.get(feature)
        
        if unlimited_flag and unlimited_flag in row and row[unlimited_flag] == 1:
            # This feature is unlimited for this plan
            # Use the minimum fee among unlimited plans for this feature
            if unlimited_flag in frontiers:
                total_cost += frontiers[unlimited_flag].iloc[0]
        else:
            # This feature is not unlimited or doesn't have an unlimited option
            feature_value = row[feature]
            
            # Get the frontier-based cost for this feature value
            if feature in frontiers:
                # Check if the index is numeric using pandas-version-agnostic approach
                if pd.api.types.is_numeric_dtype(frontiers[feature].index):
                    # Numeric index - use estimation
                    frontier_value = estimate_frontier_value(feature_value, frontiers[feature])
                    total_cost += frontier_value
                else:
                    # Categorical index - direct lookup
                    if feature_value in frontiers[feature].index:
                        total_cost += frontiers[feature][feature_value]
    
    return total_cost

def calculate_cs_ratio(df: pd.DataFrame, feature_set: str = 'basic', 
                      fee_column: str = 'fee') -> pd.DataFrame:
    """
    Calculate Cost-Spec ratio for each plan using frontier-based costs.
    
    Args:
        df: DataFrame with plan data
        feature_set: Name of the feature set to use
        fee_column: Column containing the fee to use
        
    Returns:
        DataFrame with added CS ratio calculations
    """
    # Make a copy to avoid modifying the original
    df_result = df.copy()
    
    # Get the features for this feature set
    if feature_set in FEATURE_SETS:
        features = FEATURE_SETS[feature_set]
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
    
    # Calculate feature frontiers
    frontiers = calculate_feature_frontiers(df, features, UNLIMITED_FLAGS, fee_column)
    
    # Add individual feature contribution columns
    for feature in [f for f in features if f not in UNLIMITED_FLAGS.values()]:
        # Skip features not in frontiers
        if feature not in frontiers and not UNLIMITED_FLAGS.get(feature) in frontiers:
            logger.warning(f"Feature {feature} not found in frontiers, skipping contribution calculation")
            continue
            
        # Calculate contribution for each row
        contribution_column = f"contribution_{feature}"
        
        # Define a function to calculate contribution for this feature
        def calculate_feature_contribution(row):
            # Skip if feature not available
            if feature not in row:
                return 0.0
                
            # Check if this feature has an unlimited flag
            unlimited_flag = UNLIMITED_FLAGS.get(feature)
            
            if unlimited_flag and unlimited_flag in row and row[unlimited_flag] == 1:
                # This feature is unlimited for this plan
                # Use the minimum fee among unlimited plans for this feature
                if unlimited_flag in frontiers:
                    return frontiers[unlimited_flag].iloc[0]
                return 0.0
            else:
                # This feature is not unlimited or doesn't have an unlimited option
                feature_value = row[feature]
                
                # Get the frontier-based cost for this feature value
                if feature in frontiers:
                    # Check if the index is numeric
                    if pd.api.types.is_numeric_dtype(frontiers[feature].index):
                        # Numeric index - use estimation
                        return estimate_frontier_value(feature_value, frontiers[feature])
                    else:
                        # Categorical index - direct lookup
                        if feature_value in frontiers[feature].index:
                            return frontiers[feature][feature_value]
                return 0.0
        
        # Apply the function to calculate contributions
        df_result[contribution_column] = df_result.apply(calculate_feature_contribution, axis=1)
        logger.info(f"Added contribution column for {feature}: min={df_result[contribution_column].min()}, max={df_result[contribution_column].max()}")
    
    # Calculate baseline cost for each plan (sum of all contributions)
    df_result['B'] = df_result.apply(
        lambda row: calculate_plan_baseline_cost(row, frontiers, UNLIMITED_FLAGS), 
        axis=1
    )
    
    # Calculate CS ratio (B/fee)
    df_result['CS'] = df_result['B'] / df_result[fee_column]
    
    return df_result

def rank_plans_by_cs(df: pd.DataFrame, feature_set: str = 'basic',
                    fee_column: str = 'fee', 
                    top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Rank plans by Cost-Spec ratio using frontier-based costs.
    
    Args:
        df: DataFrame with plan data
        feature_set: Name of the feature set to use
        fee_column: Column containing the fee to use
        top_n: If provided, return only the top N plans
        
    Returns:
        DataFrame with added CS ratio and rank
    """
    # Calculate CS ratio
    df_result = calculate_cs_ratio(df, feature_set, fee_column)
    
    # Sort by CS ratio (descending)
    df_result = df_result.sort_values('CS', ascending=False)
    
    # Add rank number
    df_result['rank_number'] = range(1, len(df_result) + 1)
    
    # Format rank display (with ties)
    df_result['rank_display'] = df_result['rank_number'].astype(str)
    
    # Return top N if specified
    if top_n is not None and top_n < len(df_result):
        return df_result.head(top_n)
    
    return df_result 