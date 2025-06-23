"""
Multi-Feature Regression Module

Contains multi-feature regression analysis and coefficient calculation functionality.
Extracted from multi_feature.py for better modularity.

Classes:
- MultiFeatureRegressor: Handles regression analysis and coefficient calculation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy.optimize import minimize

# Configure logging
logger = logging.getLogger(__name__)

class MultiFeatureRegressor:
    """
    Handles multi-feature regression analysis and coefficient calculation.
    """
    
    def __init__(self):
        self.coefficients = None
        self.unconstrained_coefficients = None
        self.coefficient_bounds = None
        self.multicollinearity_fixes = {}
        self.multicollinearity_detected = False
        self.correlation_matrix = None
        
    def solve_multi_feature_coefficients(self, X: np.ndarray, y: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Solve for pure marginal costs using multi-feature regression.
        
        Args:
            X: Feature matrix
            y: Target vector (prices)
            features: List of feature names
            
        Returns:
            Array of coefficients [β₀, β₁, β₂, ...]
        """
        if len(y) < len(features) + 1:
            raise ValueError(f"Insufficient data points ({len(y)}) for {len(features)} features")
        
        # Detect multicollinearity
        self._detect_multicollinearity(X, features)
        
        # Solve constrained regression
        if self.multicollinearity_detected:
            logger.info("Multicollinearity detected but using constrained regression (Ridge disabled per user request)")
        else:
            logger.info("Using constrained least squares (no multicollinearity)")
        
        coefficients = self._solve_constrained_regression(X, y, features)
        
        # Fix multicollinearity if detected
        if self.multicollinearity_detected and self.correlation_matrix is not None:
            coefficients = self._fix_multicollinearity_coefficients(coefficients, features, X, y)
        
        self.coefficients = coefficients
        return self.coefficients

    def _detect_multicollinearity(self, X: np.ndarray, features: List[str], threshold: float = 0.8):
        """
        Detect multicollinearity using correlation matrix.
        
        Args:
            X: Feature matrix
            features: List of feature names
            threshold: Correlation threshold for multicollinearity detection
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)
        self.correlation_matrix = pd.DataFrame(corr_matrix, index=features, columns=features)
        
        # Check for high correlations
        self.multicollinearity_detected = False
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                if abs(corr_matrix[i, j]) > threshold:
                    self.multicollinearity_detected = True
                    logger.warning(f"High correlation detected: {features[i]} ↔ {features[j]} = {corr_matrix[i, j]:.3f}")
                    break
            if self.multicollinearity_detected:
                break

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

    def _fix_multicollinearity_coefficients(self, coefficients: np.ndarray, features: List[str], 
                                           X: np.ndarray = None, y: np.ndarray = None) -> np.ndarray:
        """
        Fix multicollinearity by redistributing coefficients for highly correlated features.
        
        Args:
            coefficients: Original coefficients [β₀, β₁, β₂, ...]
            features: List of feature names
            X: Feature matrix (optional, for future Commonality Analysis)
            y: Target variable (optional, for future Commonality Analysis)
            
        Returns:
            Adjusted coefficients with redistributed values
        """
        if self.correlation_matrix is None:
            return coefficients
        
        fixed_coefficients = coefficients.copy()
        
        # Store multicollinearity fixes for detailed reporting
        self.multicollinearity_fixes = {}
        
        # Find high correlation pairs (threshold 0.8)
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                feature1 = features[i]
                feature2 = features[j]
                
                if feature1 in self.correlation_matrix.index and feature2 in self.correlation_matrix.columns:
                    corr_val = abs(self.correlation_matrix.loc[feature1, feature2])
                    
                    if corr_val > 0.8:  # High correlation
                        # Get current coefficients (skip base cost at index 0)
                        coeff1 = fixed_coefficients[i+1]
                        coeff2 = fixed_coefficients[j+1]
                        
                        # Calculate total value and redistribute equally
                        total_value = coeff1 + coeff2
                        redistributed_value = total_value / 2
                        
                        logger.info(f"Redistributing coefficients for {feature1} ↔ {feature2} (correlation: {corr_val:.3f})")
                        logger.info(f"  Before: {feature1}=₩{coeff1:,.2f}, {feature2}=₩{coeff2:,.2f}")
                        logger.info(f"  After: {feature1}=₩{redistributed_value:,.2f}, {feature2}=₩{redistributed_value:,.2f}")
                        
                        # Store detailed calculation steps for HTML display
                        self.multicollinearity_fixes[feature1] = {
                            'paired_with': feature2,
                            'correlation': corr_val,
                            'original_value': coeff1,
                            'partner_original_value': coeff2,
                            'total_value': total_value,
                            'redistributed_value': redistributed_value,
                            'calculation_formula': f"({coeff1:.2f} + {coeff2:.2f}) / 2 = {redistributed_value:.2f}"
                        }
                        
                        self.multicollinearity_fixes[feature2] = {
                            'paired_with': feature1,
                            'correlation': corr_val,
                            'original_value': coeff2,
                            'partner_original_value': coeff1,
                            'total_value': total_value,
                            'redistributed_value': redistributed_value,
                            'calculation_formula': f"({coeff1:.2f} + {coeff2:.2f}) / 2 = {redistributed_value:.2f}"
                        }
                        
                        # Apply redistribution
                        fixed_coefficients[i+1] = redistributed_value
                        fixed_coefficients[j+1] = redistributed_value
        
        return fixed_coefficients
    
    def get_coefficient_breakdown(self, frontier_plans: pd.DataFrame, features: List[str]) -> dict:
        """
        Get coefficient breakdown for visualization.
        
        Returns:
            Dictionary with coefficient information including both raw and constrained values
        """
        if self.coefficients is None:
            raise ValueError("Must solve coefficients first")
            
        analysis_features = [f for f in features if f in frontier_plans.columns]
        
        breakdown = {
            'base_cost': 0.0,  # No base cost - regression through origin
            'feature_costs': {},
            'total_plans_used': len(frontier_plans),
            'outliers_removed': 0,
            'features_analyzed': len(analysis_features),
            'method': 'multi_frontier'
        }
        
        # Include both unconstrained and constrained coefficients
        for i, feature in enumerate(analysis_features):
            feature_data = {
                'coefficient': self.coefficients[i+1],
                'cost_per_unit': self.coefficients[i+1]
            }
            
            # Add unconstrained coefficient if available
            if self.unconstrained_coefficients is not None:
                feature_data['unconstrained_coefficient'] = self.unconstrained_coefficients[i]
            
            # Add bounds information if available
            if self.coefficient_bounds is not None and i < len(self.coefficient_bounds):
                lower_bound, upper_bound = self.coefficient_bounds[i]
                feature_data['bounds'] = {
                    'lower': lower_bound,
                    'upper': upper_bound
                }
            
            breakdown['feature_costs'][feature] = feature_data
        
        return breakdown 