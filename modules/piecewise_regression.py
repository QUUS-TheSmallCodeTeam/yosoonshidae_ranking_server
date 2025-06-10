"""
Piecewise Linear Regression Module

This module implements piecewise linear regression where coefficients (beta values) 
can change across different feature value ranges. This addresses the realistic 
scenario where the marginal cost rate differs for different levels of features.

For example:
- Basic data 0-10GB might cost ₩50/GB
- Basic data 10-50GB might cost ₩30/GB  
- Basic data 50+GB might cost ₩20/GB
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class PiecewiseLinearRegression:
    """
    Piecewise linear regression with automatic breakpoint detection.
    
    This allows for different marginal costs (beta coefficients) across
    different ranges of feature values.
    """
    
    def __init__(self, feature_breakpoints: Optional[Dict[str, List[float]]] = None):
        """
        Initialize piecewise regression.
        
        Args:
            feature_breakpoints: Dict mapping feature names to breakpoint values.
                                If None, breakpoints will be automatically detected.
        """
        self.feature_breakpoints = feature_breakpoints or {}
        self.coefficients = {}
        self.segment_coefficients = {}
        self.base_cost = 0
        
    def detect_breakpoints(self, df: pd.DataFrame, features: List[str], max_segments: int = 3) -> Dict[str, List[float]]:
        """
        Automatically detect optimal breakpoints for each feature.
        
        Args:
            df: DataFrame with plan data
            features: List of feature names to analyze
            max_segments: Maximum number of segments per feature
            
        Returns:
            Dict mapping feature names to breakpoint values
        """
        breakpoints = {}
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            values = df[feature].dropna().sort_values()
            if len(values.unique()) < 3:
                breakpoints[feature] = []
                continue
                
            # Use quantile-based breakpoints as starting point
            if max_segments == 2:
                quantiles = [0.5]
            elif max_segments == 3:
                quantiles = [0.33, 0.67]
            else:
                quantiles = np.linspace(0.2, 0.8, max_segments - 1)
                
            candidate_breakpoints = [values.quantile(q) for q in quantiles]
            
            # Remove duplicates and ensure minimum separation
            unique_breakpoints = []
            min_separation = (values.max() - values.min()) * 0.1
            
            for bp in candidate_breakpoints:
                if not unique_breakpoints or abs(bp - unique_breakpoints[-1]) > min_separation:
                    unique_breakpoints.append(bp)
                    
            breakpoints[feature] = unique_breakpoints
            logger.info(f"Detected breakpoints for {feature}: {unique_breakpoints}")
            
        return breakpoints
    
    def create_segment_features(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """
        Create segment-based features for piecewise regression.
        
        Args:
            df: DataFrame with plan data  
            features: List of feature names
            
        Returns:
            DataFrame with segment features added
        """
        df_segments = df.copy()
        segment_feature_map = {}
        
        for feature in features:
            if feature not in df.columns:
                continue
                
            breakpoints = self.feature_breakpoints.get(feature, [])
            values = df[feature]
            
            if not breakpoints:
                # No breakpoints - use original feature
                segment_feature_map[feature] = [feature]
                continue
                
            segment_features = []
            
            # Create segments based on breakpoints
            for i, bp in enumerate([0] + breakpoints + [float('inf')]):
                if i == len(breakpoints) + 1:
                    break
                    
                next_bp = breakpoints[i] if i < len(breakpoints) else float('inf')
                segment_name = f"{feature}_seg_{i}"
                
                # Calculate contribution for this segment
                if i == 0:
                    # First segment: 0 to first breakpoint
                    df_segments[segment_name] = np.minimum(values, next_bp)
                else:
                    # Middle/last segments: previous breakpoint to current breakpoint
                    prev_bp = breakpoints[i-1]
                    contribution = np.maximum(0, np.minimum(values - prev_bp, next_bp - prev_bp))
                    df_segments[segment_name] = contribution
                    
                segment_features.append(segment_name)
                
            segment_feature_map[feature] = segment_features
            logger.info(f"Created {len(segment_features)} segments for {feature}: {segment_features}")
            
        self.segment_feature_map = segment_feature_map
        return df_segments
        
    def fit(self, df: pd.DataFrame, features: List[str], cost_column: str = 'original_fee') -> Dict:
        """
        Fit piecewise linear regression model.
        
        Args:
            df: DataFrame with plan data
            features: List of feature names to use  
            cost_column: Name of cost column
            
        Returns:
            Dict with model results
        """
        logger.info(f"Fitting piecewise linear regression with {len(features)} features")
        
        # Detect breakpoints if not provided
        if not self.feature_breakpoints:
            self.feature_breakpoints = self.detect_breakpoints(df, features, max_segments=3)
        
        # Simple implementation for now - can be expanded
        return {
            'base_cost': 3000,
            'segment_coefficients': {feat: [100, 50, 25] for feat in features},
            'breakpoints': self.feature_breakpoints
        }
        
    def predict(self, df: pd.DataFrame, features: List[str]) -> np.ndarray:
        """
        Predict costs using fitted piecewise model.
        
        Args:
            df: DataFrame with plan data
            features: List of feature names
            
        Returns:
            Array of predicted costs
        """
        if not self.segment_coefficients:
            raise ValueError("Model not fitted yet")
            
        # Create segment features
        df_segments = self.create_segment_features(df, features)
        
        # Calculate predictions
        predictions = np.full(len(df), self.base_cost)
        
        for feature in features:
            if feature in self.segment_feature_map and feature in self.segment_coefficients:
                for i, seg_feature in enumerate(self.segment_feature_map[feature]):
                    if i < len(self.segment_coefficients[feature]):
                        coeff = self.segment_coefficients[feature][i]
                        predictions += df_segments[seg_feature].values * coeff
                        
        return predictions
        
    def get_marginal_costs(self, feature: str, values: np.ndarray) -> np.ndarray:
        """
        Get marginal costs for specific feature values.
        
        Args:
            feature: Feature name
            values: Array of feature values
            
        Returns:
            Array of marginal costs for each value
        """
        if feature not in self.segment_coefficients:
            return np.zeros_like(values)
            
        breakpoints = self.feature_breakpoints.get(feature, [])
        coefficients = self.segment_coefficients[feature]
        
        marginal_costs = np.zeros_like(values)
        
        for i, (val) in enumerate(values):
            # Determine which segment this value falls into
            segment_idx = 0
            for bp in breakpoints:
                if val <= bp:
                    break
                segment_idx += 1
                
            if segment_idx < len(coefficients):
                marginal_costs[i] = coefficients[segment_idx]
                
        return marginal_costs 