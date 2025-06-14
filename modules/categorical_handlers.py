"""
Categorical Feature Handlers for Marginal Cost Frontier Analysis

This module provides advanced methods for handling categorical features (like unlimited flags)
in the context of marginal cost analysis and piecewise linear models.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class CategoricalFeatureHandler:
    """
    Advanced handler for categorical features in marginal cost analysis.
    
    Provides multiple encoding strategies for categorical features like unlimited flags:
    1. Dummy variable encoding (one-hot)
    2. Effect coding (sum-to-zero)
    3. Cost-based encoding (using actual cost premiums)
    4. Mixed categorical-continuous modeling
    """
    
    def __init__(self, unlimited_flags: Dict[str, str]):
        """
        Initialize the categorical feature handler.
        
        Args:
            unlimited_flags: Mapping of continuous features to their unlimited flag columns
        """
        self.unlimited_flags = unlimited_flags
        self.encoders = {}
        self.cost_premiums = {}
        
    def encode_dummy_variables(self, df: pd.DataFrame, 
                             categorical_features: List[str]) -> pd.DataFrame:
        """
        Encode categorical features using dummy variables (one-hot encoding).
        
        This is the most common approach for including categorical features in regression:
        - Creates binary indicator variables for each category
        - Drops one category as reference (to avoid multicollinearity)
        
        Args:
            df: Input dataframe
            categorical_features: List of categorical column names
            
        Returns:
            DataFrame with dummy variables added
        """
        df_encoded = df.copy()
        
        for feature in categorical_features:
            if feature not in df.columns:
                logger.warning(f"Categorical feature {feature} not found in dataframe")
                continue
                
            # Create dummy variables
            dummies = pd.get_dummies(df[feature], prefix=feature, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            
            logger.info(f"Created {len(dummies.columns)} dummy variables for {feature}")
            
        return df_encoded
    
    def encode_unlimited_as_premium(self, df: pd.DataFrame, 
                                  continuous_feature: str,
                                  unlimited_flag: str) -> pd.DataFrame:
        """
        Encode unlimited features as cost premiums over continuous equivalents.
        
        This approach treats unlimited as a premium tier:
        - Calculate cost premium for unlimited plans
        - Create interaction terms between unlimited status and base feature
        
        Args:
            df: Input dataframe
            continuous_feature: Name of continuous feature (e.g., 'basic_data_clean')
            unlimited_flag: Name of unlimited flag (e.g., 'basic_data_unlimited')
            
        Returns:
            DataFrame with premium encoding
        """
        if continuous_feature not in df.columns or unlimited_flag not in df.columns:
            logger.warning(f"Missing columns for premium encoding: {continuous_feature}, {unlimited_flag}")
            return df
            
        df_encoded = df.copy()
        
        # Calculate unlimited premium
        limited_plans = df[df[unlimited_flag] == 0]
        unlimited_plans = df[df[unlimited_flag] == 1]
        
        if not limited_plans.empty and not unlimited_plans.empty:
            # Compare costs at similar feature levels
            median_feature_value = limited_plans[continuous_feature].median()
            
            # Find similar limited plans
            similar_limited = limited_plans[
                (limited_plans[continuous_feature] >= median_feature_value * 0.8) &
                (limited_plans[continuous_feature] <= median_feature_value * 1.2)
            ]
            
            if not similar_limited.empty:
                avg_limited_cost = similar_limited['original_fee'].mean()
                avg_unlimited_cost = unlimited_plans['original_fee'].mean()
                
                premium = avg_unlimited_cost - avg_limited_cost
                self.cost_premiums[continuous_feature] = premium
                
                # Create premium feature
                df_encoded[f"{continuous_feature}_unlimited_premium"] = df_encoded[unlimited_flag] * premium
                
                logger.info(f"Calculated unlimited premium for {continuous_feature}: ₩{premium:.0f}")
        
        return df_encoded
    
    def encode_mixed_categorical_continuous(self, df: pd.DataFrame,
                                          continuous_feature: str,
                                          unlimited_flag: str,
                                          high_value_threshold: float = None) -> pd.DataFrame:
        """
        Create mixed encoding where unlimited is treated as very high continuous value.
        
        This approach:
        - Sets unlimited plans to a very high continuous value
        - Allows them to participate in piecewise linear analysis
        - Maintains continuity in the marginal cost function
        
        Args:
            df: Input dataframe
            continuous_feature: Name of continuous feature
            unlimited_flag: Name of unlimited flag
            high_value_threshold: Value to assign to unlimited (default: 2x max observed)
            
        Returns:
            DataFrame with mixed encoding
        """
        if continuous_feature not in df.columns or unlimited_flag not in df.columns:
            logger.warning(f"Missing columns for mixed encoding: {continuous_feature}, {unlimited_flag}")
            return df
            
        df_encoded = df.copy()
        
        # Calculate high value threshold
        max_observed = df[df[unlimited_flag] == 0][continuous_feature].max()
        if high_value_threshold is None:
            high_value_threshold = max_observed * 2.0
            
        # Create mixed feature
        mixed_feature_name = f"{continuous_feature}_mixed"
        df_encoded[mixed_feature_name] = df_encoded[continuous_feature].copy()
        
        # Set unlimited plans to high value
        unlimited_mask = df_encoded[unlimited_flag] == 1
        df_encoded.loc[unlimited_mask, mixed_feature_name] = high_value_threshold
        
        # Add flag to track which values are synthetic
        df_encoded[f"{continuous_feature}_is_synthetic"] = unlimited_mask.astype(int)
        
        logger.info(f"Created mixed encoding for {continuous_feature}: unlimited = {high_value_threshold}")
        
        return df_encoded
    
    def create_categorical_segments(self, df: pd.DataFrame,
                                   continuous_feature: str,
                                   unlimited_flag: str) -> Dict[str, pd.DataFrame]:
        """
        Create separate segments for categorical analysis.
        
        This approach:
        - Analyzes limited and unlimited plans separately
        - Calculates separate marginal cost functions for each segment
        - Allows for different cost structures between segments
        
        Args:
            df: Input dataframe
            continuous_feature: Name of continuous feature
            unlimited_flag: Name of unlimited flag
            
        Returns:
            Dictionary with separate dataframes for each segment
        """
        if continuous_feature not in df.columns or unlimited_flag not in df.columns:
            logger.warning(f"Missing columns for segmentation: {continuous_feature}, {unlimited_flag}")
            return {}
            
        segments = {}
        
        # Limited plans segment
        limited_mask = df[unlimited_flag] == 0
        if limited_mask.sum() > 0:
            segments['limited'] = df[limited_mask].copy()
            logger.info(f"Limited segment for {continuous_feature}: {len(segments['limited'])} plans")
            
        # Unlimited plans segment
        unlimited_mask = df[unlimited_flag] == 1
        if unlimited_mask.sum() > 0:
            segments['unlimited'] = df[unlimited_mask].copy()
            logger.info(f"Unlimited segment for {continuous_feature}: {len(segments['unlimited'])} plans")
            
        return segments
    
    def calculate_categorical_marginal_costs(self, df: pd.DataFrame,
                                           continuous_feature: str,
                                           unlimited_flag: str,
                                           base_coefficient: float) -> Dict[str, float]:
        """
        Calculate marginal costs for categorical feature levels.
        
        Args:
            df: Input dataframe
            continuous_feature: Name of continuous feature
            unlimited_flag: Name of unlimited flag
            base_coefficient: Base marginal cost for continuous feature
            
        Returns:
            Dictionary with marginal costs for each category level
        """
        marginal_costs = {
            'limited': base_coefficient,
            'unlimited': None
        }
        
        # Calculate unlimited marginal cost
        segments = self.create_categorical_segments(df, continuous_feature, unlimited_flag)
        
        if 'limited' in segments and 'unlimited' in segments:
            limited_df = segments['limited']
            unlimited_df = segments['unlimited']
            
            # Compare average costs
            if not limited_df.empty and not unlimited_df.empty:
                avg_limited_cost = limited_df['original_fee'].mean()
                avg_unlimited_cost = unlimited_df['original_fee'].mean()
                
                # Estimate unlimited value equivalent
                max_limited_feature = limited_df[continuous_feature].max()
                unlimited_equivalent = max_limited_feature * 2  # Assume 2x max as unlimited equivalent
                
                # Calculate unlimited marginal cost
                cost_difference = avg_unlimited_cost - avg_limited_cost
                feature_difference = unlimited_equivalent - limited_df[continuous_feature].mean()
                
                if feature_difference > 0:
                    unlimited_marginal_cost = cost_difference / feature_difference
                    marginal_costs['unlimited'] = unlimited_marginal_cost
                    
                    logger.info(f"Calculated categorical marginal costs for {continuous_feature}:")
                    logger.info(f"  Limited: ₩{marginal_costs['limited']:.2f}/unit")
                    logger.info(f"  Unlimited: ₩{unlimited_marginal_cost:.2f}/unit (equivalent)")
        
        return marginal_costs

def apply_categorical_encoding(df: pd.DataFrame, 
                             encoding_strategy: str = 'separate_handling',
                             unlimited_flags: Dict[str, str] = None) -> pd.DataFrame:
    """
    Apply categorical encoding strategy to dataframe.
    
    Args:
        df: Input dataframe
        encoding_strategy: Strategy to use ('separate_handling', 'dummy_variables', 'premium_encoding')
        unlimited_flags: Mapping of continuous features to unlimited flags
        
    Returns:
        DataFrame with categorical encoding applied
    """
    if unlimited_flags is None:
        from modules.cost_spec import UNLIMITED_FLAGS
        unlimited_flags = UNLIMITED_FLAGS
        
    handler = CategoricalFeatureHandler(unlimited_flags)
    
    if encoding_strategy == 'separate_handling':
        # Current approach - no changes needed
        return df
        
    elif encoding_strategy == 'dummy_variables':
        # Create dummy variables for all unlimited flags
        categorical_features = list(unlimited_flags.values())
        return handler.encode_dummy_variables(df, categorical_features)
        
    elif encoding_strategy == 'premium_encoding':
        # Encode unlimited as cost premiums
        df_encoded = df.copy()
        for continuous_feature, unlimited_flag in unlimited_flags.items():
            df_encoded = handler.encode_unlimited_as_premium(
                df_encoded, continuous_feature, unlimited_flag
            )
        return df_encoded
        
    else:
        raise ValueError(f"Unknown encoding strategy: {encoding_strategy}") 