"""
Cost-Spec Ratio implementation for MVNO plan ranking.

This module calculates plan rankings based on a cost-spec ratio approach
which compares each plan's actual fee with a theoretical baseline cost derived
from minimum costs of individual features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

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

def calculate_baseline_costs(df: pd.DataFrame, features: List[str], 
                           unlimited_flags: Dict[str, str], 
                           fee_column: str = 'fee') -> Dict[str, pd.Series]:
    """
    Compute baseline minimum costs for each feature value.
    
    For continuous features with 'unlimited' options, we exclude those plans
    when calculating the baseline. For other features, we use all plans.
    
    Args:
        df: DataFrame with plan data
        features: List of feature columns to consider
        unlimited_flags: Mapping of feature columns to their unlimited flag columns
        fee_column: Column containing the fee to use
        
    Returns:
        Dictionary mapping feature values to minimum fees
    """
    baseline_costs = {}
    
    for feature in features:
        # Skip unlimited flag columns, they'll be handled with their corresponding features
        if feature in unlimited_flags.values():
            continue
        
        # Check if this feature has an unlimited flag
        unlimited_flag = unlimited_flags.get(feature)
        
        if unlimited_flag and unlimited_flag in df.columns:
            # For features with unlimited options, we need to handle them separately
            # Get plans where this feature is not unlimited
            not_unlimited = df[df[unlimited_flag] == 0].copy()
            
            if not not_unlimited.empty:
                # Group by feature value and find minimum fee
                grouped = not_unlimited.groupby(feature)[fee_column].min()
                baseline_costs[feature] = grouped
            
            # Handle unlimited feature values
            unlimited_plans = df[df[unlimited_flag] == 1]
            if not unlimited_plans.empty:
                # For unlimited features, use the minimum fee among unlimited plans
                min_fee_unlimited = unlimited_plans[fee_column].min()
                # Add as a special case with a dedicated key
                baseline_costs[unlimited_flag] = pd.Series([min_fee_unlimited])
        else:
            # For standard features without unlimited options
            # Group by feature value and find minimum fee
            grouped = df.groupby(feature)[fee_column].min()
            baseline_costs[feature] = grouped
    
    return baseline_costs

def calculate_plan_baseline_cost(row: pd.Series, features: List[str], 
                               baseline_costs: Dict[str, pd.Series],
                               unlimited_flags: Dict[str, str]) -> float:
    """
    Calculate the theoretical baseline cost for a single plan.
    
    Args:
        row: Series containing a single plan's data
        features: List of feature columns to consider
        baseline_costs: Dictionary mapping feature values to minimum fees
        unlimited_flags: Mapping of feature columns to their unlimited flag columns
        
    Returns:
        Total baseline cost for the plan
    """
    total_cost = 0.0
    
    for feature in features:
        # Skip unlimited flag columns, they'll be handled with their corresponding features
        if feature in unlimited_flags.values():
            continue
        
        # Check if this feature has an unlimited flag
        unlimited_flag = unlimited_flags.get(feature)
        
        if unlimited_flag and unlimited_flag in row and row[unlimited_flag] == 1:
            # This feature is unlimited for this plan
            # Use the minimum fee among unlimited plans for this feature
            if unlimited_flag in baseline_costs:
                total_cost += baseline_costs[unlimited_flag].iloc[0]
        else:
            # This feature is not unlimited or doesn't have an unlimited option
            feature_value = row[feature]
            
            # Get the baseline cost for this feature value
            if feature in baseline_costs and feature_value in baseline_costs[feature].index:
                total_cost += baseline_costs[feature].loc[feature_value]
    
    return total_cost

def calculate_cs_ratio(df: pd.DataFrame, feature_set: str = 'basic', 
                      fee_column: str = 'fee') -> pd.DataFrame:
    """
    Calculate Cost-Spec ratio for each plan in the DataFrame.
    
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
    
    # Calculate baseline costs for each feature value
    baseline_costs = calculate_baseline_costs(df, features, UNLIMITED_FLAGS, fee_column)
    
    # Calculate baseline cost for each plan
    df_result['B'] = df_result.apply(
        lambda row: calculate_plan_baseline_cost(row, features, baseline_costs, UNLIMITED_FLAGS), 
        axis=1
    )
    
    # Calculate CS ratio (B/fee)
    df_result['CS'] = df_result['B'] / df_result[fee_column]
    
    return df_result

def rank_plans_by_cs(df: pd.DataFrame, feature_set: str = 'basic',
                    fee_column: str = 'fee', 
                    top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Rank plans by Cost-Spec ratio.
    
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
    # This uses a more advanced ranking method with ties
    # For simplicity, we'll use the basic rank_number for now
    df_result['rank_display'] = df_result['rank_number'].astype(str)
    
    # Return top N if specified
    if top_n is not None and top_n < len(df_result):
        return df_result.head(top_n)
    
    return df_result 