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

# Core continuous features to use for frontier calculation
CORE_FEATURES = [
    'basic_data_clean',  # Data GB
    'voice_clean',       # Voice minutes
    'message_clean'      # SMS count
]

def create_monotonic_frontier(x: pd.Series, y: pd.Series) -> pd.Series:
    """
    Create a monotonic frontier where y values never decrease as x increases.
    
    Args:
        x: Series of x values (feature values, sorted)
        y: Series of y values (fees)
        
    Returns:
        Series of frontier y values indexed by x
    """
    if len(x) == 0 or len(y) == 0:
        return pd.Series()
    
    # Ensure x and y are sorted by x
    sorted_idx = np.argsort(x)
    x_sorted = x.iloc[sorted_idx]
    y_sorted = y.iloc[sorted_idx]
    
    # Create a dictionary to store frontier points
    frontier_points = {}
    
    # Start with the lowest x value
    current_min_y = y_sorted.iloc[0]
    frontier_points[x_sorted.iloc[0]] = current_min_y
    
    # Iterate through remaining points
    for i in range(1, len(x_sorted)):
        x_val = x_sorted.iloc[i]
        y_val = y_sorted.iloc[i]
        
        # If we find a cheaper price for a higher feature value, update our frontier
        if y_val < current_min_y:
            # This violates monotonicity - keep the previous minimum
            frontier_points[x_val] = current_min_y
        else:
            # Price is higher than current minimum, respects monotonicity
            frontier_points[x_val] = y_val
            current_min_y = y_val
    
    # Convert to pandas Series
    return pd.Series(frontier_points)

def calculate_feature_frontiers(df: pd.DataFrame, features: List[str], 
                              unlimited_flags: Dict[str, str], 
                              fee_column: str = 'fee') -> Dict[str, pd.Series]:
    """
    Compute cost frontiers for each feature with monotonicity constraints.
    
    Args:
        df: DataFrame with plan data
        features: List of feature columns to consider
        unlimited_flags: Mapping of feature columns to their unlimited flag columns
        fee_column: Column containing the fee to use
        
    Returns:
        Dictionary mapping features to their cost frontiers
    """
    frontiers = {}
    
    # Process each core continuous feature and create a monotonic frontier
    for feature in CORE_FEATURES:
        # Skip if feature not in dataframe
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in dataframe, skipping")
            continue

        # Check if this feature has an unlimited flag
        unlimited_flag = unlimited_flags.get(feature)
        
        if unlimited_flag and unlimited_flag in df.columns:
            # For features with unlimited options, handle non-unlimited values first
            not_unlimited = df[df[unlimited_flag] == 0].copy()
            
            if not not_unlimited.empty:
                # Sort by feature value and find minimum fee at each value
                feature_min_fees = not_unlimited.groupby(feature)[fee_column].min()
                
                # Create monotonic frontier
                feature_values = feature_min_fees.index
                feature_fees = feature_min_fees.values
                
                # Create Series for easier manipulation
                x = pd.Series(feature_values)
                y = pd.Series(feature_fees)
                
                # Generate monotonic frontier
                frontier = create_monotonic_frontier(x, y)
                frontiers[feature] = frontier
                
                logger.info(f"Created monotonic frontier for {feature} with {len(frontier)} points")
            
            # Handle unlimited feature values separately
            unlimited_plans = df[df[unlimited_flag] == 1]
            if not unlimited_plans.empty:
                # For unlimited features, use the minimum fee among unlimited plans
                min_fee_unlimited = unlimited_plans[fee_column].min()
                # Add as a special case with a dedicated key
                frontiers[unlimited_flag] = pd.Series([min_fee_unlimited])
                
                logger.info(f"Added unlimited case for {feature} with fee {min_fee_unlimited}")
        else:
            # For binary/categorical features, use the minimum fee for each value
            if feature in df.columns:
                feature_min_fees = df.groupby(feature)[fee_column].min()
                frontiers[feature] = feature_min_fees
                
                logger.info(f"Created feature costs for {feature} with {len(feature_min_fees)} values")
    
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
    if feature_value in frontier.index:
        # Exact match
        return frontier[feature_value]
    
    # Find the closest value in the frontier
    idx = np.searchsorted(frontier.index, feature_value)
    
    if idx == 0:
        # Feature value is lower than any in frontier, use lowest
        return frontier.iloc[0]
    elif idx == len(frontier):
        # Feature value is higher than any in frontier, use highest
        return frontier.iloc[-1]
    else:
        # Feature value is between two frontier points
        # For monotonicity, we use the lower bound frontier value
        return frontier.iloc[idx-1]

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
    
    # Calculate cost for each core feature
    for feature in CORE_FEATURES:
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
                frontier_value = estimate_frontier_value(feature_value, frontiers[feature])
                total_cost += frontier_value
    
    # Add binary/categorical features
    binary_features = [f for f in FEATURE_SETS['basic'] if f not in CORE_FEATURES and f not in unlimited_flags.values()]
    
    for feature in binary_features:
        if feature in row and feature in frontiers:
            feature_value = row[feature]
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
    
    # Calculate baseline cost for each plan
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