"""
Cost-Spec ratio calculation functions.

This module contains functions for calculating CS ratios using various methods
including frontier-based, fixed rates, and enhanced approaches.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import from parent modules
from ..config import FEATURE_SETS, UNLIMITED_FLAGS
from ..frontier.core import calculate_feature_frontiers, estimate_frontier_value, calculate_plan_baseline_cost
from ..regression.full_dataset import FullDatasetMultiFeatureRegression
from ..regression.multi_feature import MultiFeatureFrontierRegression

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
    
    # Create cost structure for visualization compatibility
    cost_structure = {
        'base_cost': 0.0,
        'feature_costs': {},
        'total_plans_used': len(df),
        'outliers_removed': 0,
        'features_analyzed': len([f for f in features if f in frontiers or UNLIMITED_FLAGS.get(f) in frontiers]),
        'method': 'frontier_based',
        'frontiers_used': list(frontiers.keys())
    }
    
    # Extract frontier-based cost estimates for each feature
    for feature in features:
        if feature in UNLIMITED_FLAGS.values():
            continue  # Skip unlimited flags
            
        unlimited_flag = UNLIMITED_FLAGS.get(feature)
        feature_data = {}
        
        if feature in frontiers:
            # For continuous features, use the marginal cost from frontier
            frontier_data = frontiers[feature]
            if len(frontier_data) >= 2:
                # Calculate marginal cost from frontier
                first_point = frontier_data.iloc[0]
                second_point = frontier_data.iloc[1]
                if frontier_data.index[1] != frontier_data.index[0]:
                    marginal_cost = (second_point - first_point) / (frontier_data.index[1] - frontier_data.index[0])
                else:
                    marginal_cost = first_point
            else:
                marginal_cost = frontier_data.iloc[0] if len(frontier_data) > 0 else 0.0
            
            feature_data['coefficient'] = float(marginal_cost)
            feature_data['cost_per_unit'] = float(marginal_cost)
            feature_data['frontier_type'] = 'continuous'
            
        if unlimited_flag and unlimited_flag in frontiers:
            # Add unlimited flag cost
            unlimited_cost = frontiers[unlimited_flag].iloc[0] if len(frontiers[unlimited_flag]) > 0 else 0.0
            feature_data['unlimited_cost'] = float(unlimited_cost)
            
        if feature_data:  # Only add if we have data
            cost_structure['feature_costs'][feature] = feature_data
    
    # Store cost structure in DataFrame attrs for compatibility
    df_result.attrs['cost_structure'] = cost_structure
    df_result.attrs['frontier_breakdown'] = cost_structure  # Alternative name
    
    logger.info(f"Created frontier-based cost structure with {len(cost_structure['feature_costs'])} features")
    
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

def calculate_cs_ratio_enhanced(df: pd.DataFrame, method: str = 'frontier',
                              feature_set: str = 'basic', fee_column: str = 'fee',
                              **method_kwargs) -> pd.DataFrame:
    """
    Enhanced Cost-Spec ratio calculation supporting multiple methods.
    
    Args:
        df: DataFrame with plan data
        method: Calculation method ('frontier', 'multi_frontier', or 'fixed_rates')
        feature_set: Name of the feature set to use
        fee_column: Column containing the fee to use
        **method_kwargs: Additional arguments passed to specific methods
        
    Returns:
        DataFrame with added CS ratio calculations
    """
    if method == 'frontier':
        # Use existing frontier-based method
        return calculate_cs_ratio(df, feature_set, fee_column)
    
    elif method == 'fixed_rates':
        # Use fixed marginal rates from pure coefficients for entire dataset
        # NOW USES EFFICIENCY FRONTIER BY DEFAULT
        logger.info("Starting fixed rates method using Efficiency Frontier Regression")
        df_result = df.copy()
        
        # Get ALL features for this feature set (including unlimited flags)
        if feature_set in FEATURE_SETS:
            # Start with the base feature set
            all_features = FEATURE_SETS[feature_set].copy()
            # Add corresponding unlimited flags
            for continuous_feature, unlimited_flag in UNLIMITED_FLAGS.items():
                if continuous_feature in all_features and unlimited_flag not in all_features:
                    all_features.append(unlimited_flag)
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")

        # Only use features that actually exist in the dataframe
        analysis_features = [f for f in all_features if f in df.columns]
        analysis_features = method_kwargs.get('features', analysis_features)
        
        logger.info(f"Fixed rates analysis features: {analysis_features}")
        
        try:
            # Use FullDatasetMultiFeatureRegression with Efficiency Frontier enabled
            regressor = FullDatasetMultiFeatureRegression(
                features=analysis_features, 
                use_efficiency_frontier=True  # ENABLE EFFICIENCY FRONTIER
            )
            
            # Solve for pure marginal costs using efficiency frontier (efficient plans only)
            coefficients = regressor.solve_full_dataset_coefficients(df)
            logger.info(f"Successfully solved efficiency frontier coefficients: {coefficients}")
            
            # Calculate baselines using pure coefficients for ALL plans
            baselines = np.full(len(df), coefficients[0])  # Start with intercept
            
            for i, feature in enumerate(analysis_features):
                if feature in df.columns and i + 1 < len(coefficients):
                    # Use the actual feature values (continuous features already zeroed out for unlimited plans, 
                    # unlimited flags have their own coefficients)
                    baselines += coefficients[i+1] * df[feature].values
                else:
                    if feature in df.columns:
                        logger.warning(f"Feature {feature} index out of range in coefficients")
                    else:
                        logger.warning(f"Feature {feature} not found in data, treating as 0")
            
            # Add results to dataframe
            df_result['B_fixed_rates'] = baselines
            df_result['CS_fixed_rates'] = baselines / df_result[fee_column]
            
            # Set primary columns to fixed rates values
            df_result['B'] = df_result['B_fixed_rates']
            df_result['CS'] = df_result['CS_fixed_rates']
            
            # Add coefficient breakdown for visualization
            coefficient_breakdown = regressor.get_coefficient_breakdown()
            df_result.attrs['fixed_rates_breakdown'] = coefficient_breakdown
            df_result.attrs['cost_structure'] = coefficient_breakdown  # For compatibility
            
            logger.info(f"Created efficiency frontier breakdown: {coefficient_breakdown}")
            logger.info(f"Processed {len(df_result)} plans using efficiency frontier regression")
            
            # Log efficiency statistics
            if hasattr(regressor, 'efficiency_regressor') and regressor.efficiency_regressor:
                efficiency_ratio = regressor.efficiency_regressor.efficiency_ratio
                efficient_count = len(regressor.efficiency_regressor.efficient_plans) if regressor.efficiency_regressor.efficient_plans is not None else 0
                logger.info(f"Efficiency statistics: {efficient_count} efficient plans ({efficiency_ratio:.1%})")
            
            return df_result
            
        except Exception as e:
            logger.error(f"Fixed rates method (efficiency frontier) failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            # Fallback to frontier method
            logger.info("Falling back to frontier method")
            return calculate_cs_ratio(df, feature_set, fee_column)

    elif method == 'linear_decomposition':
        # Linear decomposition method deprecated - redirect to fixed_rates
        logger.warning("linear_decomposition method is deprecated, using fixed_rates instead")
        return calculate_cs_ratio_enhanced(df, 'fixed_rates', feature_set, fee_column, **method_kwargs)
    
    elif method == 'multi_frontier':
        # Use new multi-feature frontier regression method
        logger.info("Starting multi-feature frontier regression method")
        df_result = df.copy()
        
        # Get features for this feature set
        if feature_set in FEATURE_SETS:
            features = [f for f in FEATURE_SETS[feature_set] if f not in UNLIMITED_FLAGS.values()]
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
        
        # Get all features for this feature set (excluding unlimited flags)
        if feature_set in FEATURE_SETS:
            all_features = [f for f in FEATURE_SETS[feature_set] if f not in UNLIMITED_FLAGS.values()]
        else:
            # Fallback to safe features if feature_set not found
            all_features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb', 'is_5g']
        
        # Only use features that actually exist in the dataframe
        analysis_features = [f for f in all_features if f in df.columns]
        analysis_features = method_kwargs.get('features', analysis_features)
        
        logger.info(f"Multi-frontier analysis features: {analysis_features}")
        
        try:
            # Initialize multi-feature frontier regression
            regressor = MultiFeatureFrontierRegression(features=analysis_features)
            
            # Solve for pure marginal costs
            coefficients = regressor.solve_multi_feature_coefficients(df)
            logger.info(f"Successfully solved multi-frontier coefficients: {coefficients}")
            
            # Calculate baselines using pure coefficients
            baselines = np.full(len(df), coefficients[0])  # Start with base cost
            
            for i, feature in enumerate(analysis_features):
                if feature in df.columns:
                    # Handle unlimited features
                    unlimited_flag = UNLIMITED_FLAGS.get(feature)
                    if unlimited_flag and unlimited_flag in df.columns:
                        # For unlimited plans, use large value
                        feature_values = df[feature].copy()
                        unlimited_mask = df[unlimited_flag] == 1
                        max_value = df[feature].max() * 2
                        feature_values.loc[unlimited_mask] = max_value
                        baselines += coefficients[i+1] * feature_values.values
                    else:
                        baselines += coefficients[i+1] * df[feature].values
                else:
                    logger.warning(f"Feature {feature} not found in data, treating as 0")
            
            # Add results to dataframe
            df_result['B_multi_frontier'] = baselines
            df_result['CS_multi_frontier'] = baselines / df_result[fee_column]
            
            # Also calculate traditional method for comparison
            df_traditional = calculate_cs_ratio(df, feature_set, fee_column)
            df_result['B_frontier'] = df_traditional['B']
            df_result['CS_frontier'] = df_traditional['CS']
            
            # Set primary columns to multi-frontier values
            df_result['B'] = df_result['B_multi_frontier']
            df_result['CS'] = df_result['CS_multi_frontier']
            
            # Add coefficient breakdown for visualization
            coefficient_breakdown = regressor.get_coefficient_breakdown()
            df_result.attrs['multi_frontier_breakdown'] = coefficient_breakdown
            df_result.attrs['cost_structure'] = coefficient_breakdown  # For compatibility
            
            logger.info(f"Created multi-frontier breakdown: {coefficient_breakdown}")
            
            return df_result
            
        except Exception as e:
            logger.error(f"Multi-feature frontier regression failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            # Fallback to frontier method
            logger.info("Falling back to frontier method")
            return calculate_cs_ratio(df, feature_set, fee_column)
    
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'frontier', 'multi_frontier', 'fixed_rates'")

def rank_plans_by_cs_enhanced(df: pd.DataFrame, method: str = 'frontier',
                            feature_set: str = 'basic', fee_column: str = 'fee',
                            top_n: Optional[int] = None, **method_kwargs) -> pd.DataFrame:
    """
    Enhanced plan ranking supporting multiple CS calculation methods.
    
    Args:
        df: DataFrame with plan data
        method: Calculation method ('frontier', 'multi_frontier', or 'fixed_rates')
        feature_set: Name of the feature set to use
        fee_column: Column containing the fee to use
        top_n: If provided, return only the top N plans
        **method_kwargs: Additional arguments passed to specific methods
        
    Returns:
        DataFrame with added CS ratio and rank
    """
    # Calculate CS ratio using specified method
    df_result = calculate_cs_ratio_enhanced(df, method, feature_set, fee_column, **method_kwargs)
    
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
