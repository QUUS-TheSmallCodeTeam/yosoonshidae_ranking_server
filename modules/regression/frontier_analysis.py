"""
Frontier Analysis Module

Contains frontier collection and analysis functionality for multi-feature regression.
Extracted from multi_feature.py for better modularity.

Classes:
- FrontierAnalyzer: Handles frontier plan collection and feature analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

# Import constants from parent modules
from ..config import CORE_FEATURES, UNLIMITED_FLAGS
from ..frontier.core import create_robust_monotonic_frontier

class FrontierAnalyzer:
    """
    Handles frontier plan collection and feature analysis for multi-feature regression.
    """
    
    def __init__(self, features=None):
        """
        Initialize the frontier analyzer.
        
        Args:
            features: List of features to analyze. If None, uses CORE_FEATURES.
        """
        self.features = features or CORE_FEATURES
        self.frontier_plans = None
        self.min_increments = {}
        self.feature_frontiers = {}
        
    def collect_all_frontier_plans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collect all plans that appear in any feature frontier.
        
        Args:
            df: DataFrame with plan data
            
        Returns:
            DataFrame containing only frontier plans from all features
        """
        frontier_plan_indices = set()
        
        # Calculate frontiers for each feature and collect plan indices
        for feature in self.features:
            if feature not in df.columns:
                logger.warning(f"Feature {feature} not found in dataframe, skipping frontier collection")
                continue
                
            if feature in UNLIMITED_FLAGS.values():
                continue
                
            unlimited_flag = UNLIMITED_FLAGS.get(feature)
            
            # Process non-unlimited plans
            if unlimited_flag and unlimited_flag in df.columns:
                df_non_unlimited = df[(df[unlimited_flag] == 0) & df['original_fee'].notna()].copy()
            else:
                df_non_unlimited = df[df['original_fee'].notna()].copy()
                
            if not df_non_unlimited.empty:
                frontier = create_robust_monotonic_frontier(df_non_unlimited, feature, 'original_fee')
                self.feature_frontiers[feature] = frontier
                
                # Find actual plans corresponding to frontier points
                for feature_val, min_cost in frontier.items():
                    matching_plans = df_non_unlimited[
                        (df_non_unlimited[feature] == feature_val) & 
                        (df_non_unlimited['original_fee'] == min_cost)
                    ]
                    frontier_plan_indices.update(matching_plans.index)
                    
                logger.info(f"Feature {feature}: Added {len(frontier)} frontier points to collection")
            
            # Process unlimited plans
            if unlimited_flag and unlimited_flag in df.columns:
                unlimited_plans = df[(df[unlimited_flag] == 1) & df['original_fee'].notna()]
                if not unlimited_plans.empty:
                    min_cost_idx = unlimited_plans['original_fee'].idxmin()
                    frontier_plan_indices.add(min_cost_idx)
                    logger.info(f"Feature {feature}: Added unlimited plan to collection")
        
        self.frontier_plans = df.loc[list(frontier_plan_indices)].copy()
        logger.info(f"Collected {len(self.frontier_plans)} unique frontier plans from all features")
        
        return self.frontier_plans
    
    def calculate_min_increments(self, df: pd.DataFrame):
        """
        Calculate minimum increments for feature normalization.
        Same logic as current system for consistency.
        
        Args:
            df: DataFrame with plan data
        """
        for feature in self.features:
            if feature not in df.columns:
                continue
                
            if feature in UNLIMITED_FLAGS.values():
                continue
                
            # Get unique feature values and calculate differences
            unique_values = sorted(df[feature].dropna().unique())
            if len(unique_values) > 1:
                differences = [
                    unique_values[i] - unique_values[i-1] 
                    for i in range(1, len(unique_values)) 
                    if unique_values[i] - unique_values[i-1] > 0
                ]
                self.min_increments[feature] = min(differences) if differences else 1
            else:
                self.min_increments[feature] = 1
                
            logger.info(f"Feature {feature}: minimum increment = {self.min_increments[feature]}")
    
    def prepare_feature_matrix(self, frontier_plans: pd.DataFrame, df: pd.DataFrame) -> tuple:
        """
        Prepare feature matrix and target vector for regression analysis.
        
        Args:
            frontier_plans: DataFrame with frontier plans
            df: Original DataFrame for unlimited value calculation
            
        Returns:
            Tuple of (X matrix, y vector, analysis_features)
        """
        # Build feature matrix (exclude unlimited flags)
        analysis_features = [f for f in self.features if f not in UNLIMITED_FLAGS.values()]
        
        # Handle unlimited features by converting to large values
        X_data = []
        y_data = []
        
        for _, plan in frontier_plans.iterrows():
            feature_vector = []
            
            for feature in analysis_features:
                if feature not in plan:
                    feature_vector.append(0)
                    continue
                    
                # Check if this feature is unlimited for this plan
                unlimited_flag = UNLIMITED_FLAGS.get(feature)
                if unlimited_flag and unlimited_flag in plan and plan[unlimited_flag] == 1:
                    # Use a large value to represent unlimited
                    max_value = df[feature].max() * 2
                    feature_vector.append(max_value)
                else:
                    feature_vector.append(plan[feature])
            
            X_data.append(feature_vector)
            y_data.append(plan['original_fee'])
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        return X, y, analysis_features
    
    def get_frontier_summary(self) -> dict:
        """
        Get summary of frontier analysis results.
        
        Returns:
            Dictionary with frontier analysis summary
        """
        return {
            'total_frontier_plans': len(self.frontier_plans) if self.frontier_plans is not None else 0,
            'features_analyzed': len(self.features),
            'feature_frontiers': {
                feature: len(frontier) 
                for feature, frontier in self.feature_frontiers.items()
            },
            'min_increments': self.min_increments.copy()
        } 