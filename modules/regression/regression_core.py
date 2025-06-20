"""
Regression Core Module

Contains the core regression analysis functionality for full dataset analysis.
Extracted from full_dataset.py for better modularity.

Classes:
- FullDatasetRegressionCore: Core regression analysis and outlier removal
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy.optimize import minimize

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
        'tethering_gb', 'speed_when_exhausted',
        'data_throttled_after_quota', 'data_unlimited_speed', 'has_unlimited_speed'
    ]
}

class FullDatasetRegressionCore:
    """
    Core regression analysis functionality for full dataset analysis.
    Handles outlier removal, constrained regression, and coefficient calculation.
    """
    
    def __init__(self, features=None, outlier_threshold=3.0, alpha=1.0):
        if features is None:
            features = FEATURE_SETS['basic']
        self.features = features
        self.outlier_threshold = outlier_threshold
        self.alpha = alpha
        self.coefficients = None
        self.unconstrained_coefficients = None
        self.coefficient_bounds = None
        self.all_plans = None
        self.outliers_removed = 0
        
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove obvious pricing outliers that would skew regression.
        
        Args:
            df: DataFrame with plan data
            
        Returns:
            DataFrame with outliers removed
        """
        df_clean = df.copy()
        initial_count = len(df_clean)
        
        # Remove plans with extremely high total cost (more than 3 std devs from mean)
        total_cost_mean = df_clean['original_fee'].mean()
        total_cost_std = df_clean['original_fee'].std()
        
        if total_cost_std > 0:
            z_scores = np.abs((df_clean['original_fee'] - total_cost_mean) / total_cost_std)
            df_clean = df_clean[z_scores <= self.outlier_threshold]
        
        self.outliers_removed = initial_count - len(df_clean)
        logger.info(f"Removed {self.outliers_removed} outliers ({self.outliers_removed/initial_count*100:.1f}%) from {initial_count} plans")
        
        return df_clean
        
    def solve_full_dataset_coefficients(self, df: pd.DataFrame) -> np.ndarray:
        """
        Solve for coefficients using ALL plans in the dataset with constrained optimization.
        
        Args:
            df: DataFrame with plan data
            
        Returns:
            Array of coefficients [β₀, β₁, β₂, ...]
        """
        # Step 1: Remove outliers
        df_clean = self.remove_outliers(df)
        self.all_plans = df_clean
        
        # Step 2: Build feature matrix using ALL features in the feature set
        analysis_features = [f for f in self.features if f in df_clean.columns]
        
        if len(df_clean) < len(analysis_features) + 1:
            raise ValueError(f"Insufficient plans ({len(df_clean)}) after outlier removal for {len(analysis_features)} features")
        
        logger.info(f"Using {len(analysis_features)} features: {analysis_features}")
        
        # Build feature matrix X and target vector y
        X_data = []
        y_data = []
        
        for _, plan in df_clean.iterrows():
            feature_vector = []
            
            for feature in analysis_features:
                if feature not in plan:
                    feature_vector.append(0)
                else:
                    # Use the actual preprocessed values (unlimited features are already zeroed out)
                    feature_vector.append(plan[feature])
            
            X_data.append(feature_vector)
            y_data.append(plan['original_fee'])
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Step 3: Use constrained regression
        coefficients = self._solve_constrained_regression(X, y, analysis_features)
        
        self.coefficients = coefficients
        
        # Validation logging
        predicted = X @ self.coefficients[1:]  # Skip base cost (which is 0)
        actual = y
        mae = np.mean(np.abs(predicted - actual))
        max_error = np.max(np.abs(predicted - actual))
        rmse = np.sqrt(np.mean((predicted - actual) ** 2))
        
        logger.info(f"Full dataset multi-feature regression solved successfully:")
        logger.info(f"  No base cost (β₀ = 0) - regression through origin")
        for i, feature in enumerate(analysis_features):
            logger.info(f"  {feature} cost: ₩{self.coefficients[i+1]:,.2f}")
        logger.info(f"  Used {len(df_clean)} plans (removed {self.outliers_removed} outliers)")
        logger.info(f"  Method: Constrained (Ridge disabled)")
        logger.info(f"  Mean absolute error: ₩{mae:,.0f}")
        logger.info(f"  Root mean square error: ₩{rmse:,.0f}")
        logger.info(f"  Max absolute error: ₩{max_error:,.0f}")
        
        return self.coefficients
    
    def _solve_constrained_regression(self, X: np.ndarray, y: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Solve using constrained optimization.
        No intercept - regression forced through origin.
        """
        # NO intercept column - force regression through origin
        X_matrix = X
        
        def objective(beta):
            return np.sum((X_matrix @ beta - y) ** 2)
        
        # First, solve unconstrained OLS for comparison
        try:
            from sklearn.linear_model import LinearRegression
            ols_model = LinearRegression(fit_intercept=False)
            ols_model.fit(X_matrix, y)
            self.unconstrained_coefficients = ols_model.coef_
            logger.info("Unconstrained OLS coefficients calculated for comparison")
        except Exception as e:
            logger.warning(f"Could not calculate unconstrained coefficients: {e}")
            self.unconstrained_coefficients = None
        
        # Simplified bounds for faster convergence
        usage_based_features = [
            'basic_data_clean', 'daily_data_clean', 'voice_clean', 'message_clean', 
            'tethering_gb', 'speed_when_exhausted'
        ]
        
        bounds = []  # No intercept bound
        for feature in features:
            if feature in usage_based_features:
                # Usage-based features: minimum ₩0.1 per unit (reduced for speed)
                bounds.append((0.1, None))
            elif feature == 'is_5g':
                # 5G premium feature: minimum ₩100
                bounds.append((100.0, None))
            elif feature == 'additional_call':
                # Additional call: minimum ₩0.1
                bounds.append((0.1, None))
            elif 'unlimited' in feature or 'throttled' in feature or 'has_unlimited' in feature:
                # Unlimited/throttled features: minimum ₩100 (reduced from 1000 for speed)
                bounds.append((100.0, 20000.0))
            else:
                # All other features: non-negative
                bounds.append((0.0, None))
        
        # Store bounds for later reference
        self.coefficient_bounds = bounds
        
        # Solve with bounds
        try:
            # Use initial guess of small positive values for faster convergence
            initial_guess = np.ones(len(features)) * 10.0
            
            result = minimize(
                objective, 
                initial_guess, 
                bounds=bounds,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-6}  # Relaxed tolerance for speed
            )
            
            if result.success:
                # Return coefficients WITHOUT base cost (no intercept)
                # Add 0 as base cost for compatibility with existing code structure
                return np.concatenate([[0.0], result.x])
            else:
                raise ValueError(f"Optimization failed: {result.message}")
                
        except Exception as e:
            raise ValueError(f"Constrained regression failed: {str(e)}")
    
    def get_coefficient_breakdown(self) -> dict:
        """
        Get coefficient breakdown for visualization.
        
        Returns:
            Dictionary with coefficient information including both raw and constrained values
        """
        if self.coefficients is None:
            raise ValueError("Must solve coefficients first")
            
        analysis_features = [f for f in self.features if f in self.all_plans.columns]
        
        breakdown = {
            'base_cost': 0.0,  # No base cost - regression through origin
            'feature_costs': {},
            'total_plans_used': len(self.all_plans) if self.all_plans is not None else 0,
            'outliers_removed': self.outliers_removed,
            'features_analyzed': len(analysis_features),
            'method': 'full_dataset'
        }
        
        # Include both unconstrained and constrained coefficients
        for i, feature in enumerate(analysis_features):
            feature_data = {
                'coefficient': self.coefficients[i+1],
                'cost_per_unit': self.coefficients[i+1]
            }
            
            # Add unconstrained coefficient if available
            if hasattr(self, 'unconstrained_coefficients') and self.unconstrained_coefficients is not None:
                feature_data['unconstrained_coefficient'] = self.unconstrained_coefficients[i]
            
            # Add bounds information if available
            if hasattr(self, 'coefficient_bounds') and self.coefficient_bounds is not None:
                if i < len(self.coefficient_bounds):
                    lower_bound, upper_bound = self.coefficient_bounds[i]
                    feature_data['bounds'] = {
                        'lower': lower_bound,
                        'upper': upper_bound
                    }
            
            breakdown['feature_costs'][feature] = feature_data
        
        return breakdown 