"""
Ranking Logic Module

Contains ranking calculation and statistical analysis functionality.
Extracted from ranking.py for better modularity.

Functions:
- calculate_rankings_with_ties: Calculate rankings with Korean tie notation
- format_number_with_commas: Number formatting utility
- shorten_plan_name: Plan name truncation utility
"""

import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def format_number_with_commas(value):
    """Format a numeric value with commas."""
    if pd.isna(value) or value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if value == int(value):  # Check if it's a whole number
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)

def shorten_plan_name(name, max_length=40):
    """Shorten plan name if it's too long."""
    if not name:
        return ""
    if len(name) <= max_length:
        return name
    return name[:max_length-3] + "..."

def calculate_rankings_with_ties(df, value_column='value_ratio', ascending=False):
    """
    Calculate rankings with ties using Korean notation.
    
    Args:
        df: DataFrame with plan data
        value_column: Column to rank by
        ascending: Sort order (False for descending)
        
    Returns:
        DataFrame with ranking column added
    """
    # Create a copy to avoid modifying the original
    df_ranked = df.copy()
    
    # Sort by the ranking column
    df_ranked = df_ranked.sort_values(by=value_column, ascending=ascending)
    
    # Calculate rankings with ties
    df_ranked['ranking'] = df_ranked[value_column].rank(method='min', ascending=ascending)
    
    # Create display rankings with Korean tie notation
    display_ranks = []
    current_rank = 1
    
    # Group by rounded values to handle ties
    value_groups = df_ranked.groupby(df_ranked[value_column].round(10)).indices
    
    for value, indices in sorted(value_groups.items(), reverse=not ascending):
        if len(indices) > 1:  # Tied plans
            for idx in indices:
                display_ranks.append(f"공동 {current_rank}위")
        else:  # Single plan
            display_ranks.append(f"{current_rank}위")
        
        current_rank += len(indices)  # Skip positions for ties
    
    # Map display ranks back to original order
    rank_mapping = {}
    for i, idx in enumerate(df_ranked.index):
        rank_mapping[idx] = display_ranks[i]
    
    df_ranked['display_ranking'] = df_ranked.index.map(rank_mapping)
    
    return df_ranked

def get_ranking_statistics(df, value_column='value_ratio'):
    """
    Get ranking statistics for summary display.
    
    Args:
        df: DataFrame with ranking data
        value_column: Column containing value ratios
        
    Returns:
        Dictionary with ranking statistics
    """
    if df.empty:
        return {
            'total_plans': 0,
            'avg_ratio': 0,
            'good_value_count': 0,
            'poor_value_count': 0,
            'best_value_plan': None,
            'worst_value_plan': None
        }
    
    good_value_count = (df[value_column] >= 1).sum()
    poor_value_count = (df[value_column] < 1).sum()
    total_plans = len(df)
    avg_ratio = df[value_column].mean()
    best_value_plan = df.iloc[0] if not df.empty else None
    worst_value_plan = df.iloc[-1] if not df.empty else None
    
    return {
        'total_plans': total_plans,
        'avg_ratio': avg_ratio,
        'good_value_count': good_value_count,
        'poor_value_count': poor_value_count,
        'best_value_plan': best_value_plan,
        'worst_value_plan': worst_value_plan
    } 