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
    
    def __init__(self, features=None, outlier_threshold=3.0, alpha=0.0):
        self.coefficients = None
        self.features = features
        self.unconstrained_coefficients = None
        self.alpha = alpha  # Ridge regularization parameter for multicollinearity control
        self.feature_bounds = {
            # Usage-based features (minimum cost per unit)
            'basic_data_clean': (0.1, None),
            'voice_clean': (0.1, None), 
            'message_clean': (0.1, None),
            'additional_call': (0.1, None),
            'tethering_gb': (0.1, None),
            'speed_when_exhausted': (0.1, None),
            'daily_data_clean': (0.1, None),
            
            # Premium service flags (minimum value for enabling feature)
            'is_5g': (100.0, None),
            'voice_unlimited': (100.0, 20000.0),
            'message_unlimited': (100.0, 20000.0),
            'data_throttled_after_quota': (100.0, 20000.0),
            'data_unlimited_speed': (100.0, 20000.0)
        }
        self.outlier_threshold = outlier_threshold
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
        # Remove outliers first
        self.all_plans = self.remove_outliers(df)
        
        # Filter to only features that exist in the data
        analysis_features = [f for f in self.features if f in self.all_plans.columns]
        
        if not analysis_features:
            raise ValueError(f"None of the specified features {self.features} found in data columns: {list(self.all_plans.columns)}")
        
        logger.info(f"Using {len(analysis_features)} features: {analysis_features}")
        
        # Prepare feature matrix X and target vector y
        X_data = []
        y_data = []
        
        for _, plan in self.all_plans.iterrows():
            # Skip plans with NaN values in target or features
            target_value = plan.get('monthly_fee', plan.get('fee', 0))
            if pd.isna(target_value) or target_value <= 0:
                continue
                
            feature_vector = []
            has_nan = False
            
            for feature in analysis_features:
                if feature not in plan:
                    feature_vector.append(0)
                else:
                    value = plan[feature]
                    if pd.isna(value):
                        # Replace NaN with 0 for features
                        feature_vector.append(0)
                    else:
                        feature_vector.append(value)
            
            # Only include if no NaN values
            if not has_nan:
                X_data.append(feature_vector)
                y_data.append(target_value)
        
        if len(X_data) == 0:
            raise ValueError("No valid data points found after removing NaN values")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        logger.info(f"Prepared data matrix: X shape {X.shape}, y shape {y.shape}")
        logger.info(f"Removed {len(self.all_plans) - len(y_data)} plans with NaN values")
        
        # Solve using constrained regression
        coefficients = self._solve_constrained_regression(X, y, analysis_features)
        
        # Store results
        self.coefficients = coefficients
        
        # Log results
        logger.info("Full dataset Ridge Regression solved successfully:")
        logger.info(f"  No base cost (β₀ = 0) - regression through origin")
        logger.info(f"  Ridge regularization parameter α = {self.alpha}")
        for i, feature in enumerate(analysis_features):
            logger.info(f"  {feature} cost: ₩{coefficients[i+1]:.2f}")
        logger.info(f"  Used {len(y_data)} plans (removed {self.outliers_removed} outliers)")
        logger.info(f"  Method: Constrained Ridge Regression")
        
        # Performance metrics
        if hasattr(self, '_last_X') and hasattr(self, '_last_y'):
            try:
                predicted = self._last_X @ coefficients[1:]  # Skip intercept
                residuals = self._last_y - predicted
                mae = np.mean(np.abs(residuals))
                rmse = np.sqrt(np.mean(residuals**2))
                max_error = np.max(np.abs(residuals))
                
                logger.info(f"  Mean absolute error: ₩{mae:,.0f}")
                logger.info(f"  Root mean square error: ₩{rmse:,.0f}")
                logger.info(f"  Max absolute error: ₩{max_error:,.0f}")
            except Exception as e:
                logger.warning(f"Could not calculate performance metrics: {e}")
        
        return coefficients
    
    def _solve_constrained_regression(self, X: np.ndarray, y: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Solve using constrained optimization with Ridge regularization.
        No intercept - regression forced through origin.
        
        Ridge objective: f(β) = ||Xβ - y||² + α||β||²
        Where α is the regularization parameter to handle multicollinearity.
        
        For Ridge regression:
        - Gradient: ∇f(β) = 2X'(Xβ - y) + 2αβ
        - Hessian: H = 2X'X + 2αI (constant, well-conditioned)
        """
        # NO intercept column - force regression through origin
        X_matrix = X
        
        # Ridge regularization parameter (can be tuned)
        alpha = self.alpha  # Use instance variable for regularization strength
        
        # Pre-compute constant matrices for efficiency
        XTX = X_matrix.T @ X_matrix  # X'X
        XTy = X_matrix.T @ y         # X'y
        I = np.eye(X_matrix.shape[1])  # Identity matrix for regularization
        
        def objective(beta):
            """Ridge objective function: f(β) = ||Xβ - y||² + α||β||²"""
            residual = X_matrix @ beta - y
            data_term = np.sum(residual ** 2)
            if alpha > 0:
                regularization_term = alpha * np.sum(beta ** 2)
                return data_term + regularization_term
            else:
                return data_term
        
        def gradient(beta):
            """Ridge gradient: ∇f(β) = 2X'(Xβ - y) + 2αβ"""
            data_gradient = 2 * (XTX @ beta - XTy)
            if alpha > 0:
                regularization_gradient = 2 * alpha * beta
                return data_gradient + regularization_gradient
            else:
                return data_gradient
        
        def hessian(beta):
            """OLS/Ridge Hessian: H = 2X'X + 2αI"""
            if alpha > 0:
                return 2 * XTX + 2 * alpha * I
            else:
                return 2 * XTX
        
        # First, solve unconstrained OLS for comparison (Ridge disabled for debugging)
        try:
            if alpha > 0:
                from sklearn.linear_model import Ridge as SklearnRidge
                ridge_model = SklearnRidge(alpha=alpha, fit_intercept=False)
                ridge_model.fit(X_matrix, y)
                self.unconstrained_coefficients = ridge_model.coef_
                logger.info(f"Unconstrained Ridge coefficients calculated (α={alpha})")
            else:
                from sklearn.linear_model import LinearRegression
                ols_model = LinearRegression(fit_intercept=False)
                ols_model.fit(X_matrix, y)
                self.unconstrained_coefficients = ols_model.coef_
                logger.info(f"Unconstrained OLS coefficients calculated (Ridge disabled)")
        except Exception as e:
            logger.warning(f"Could not calculate unconstrained coefficients: {e}")
            self.unconstrained_coefficients = None
        
        # Economic bounds for different feature types
        usage_based_features = [
            'basic_data_clean', 'daily_data_clean', 'voice_clean', 'message_clean', 
            'tethering_gb', 'speed_when_exhausted'
        ]
        
        bounds = []  # No intercept bound
        for feature in features:
            if feature in usage_based_features:
                # Usage-based features: minimum ₩0.1 per unit
                bounds.append((0.1, None))
            elif feature == 'is_5g':
                # 5G premium feature: minimum ₩100
                bounds.append((100.0, None))
            elif feature == 'additional_call':
                # Additional call: minimum ₩0.1
                bounds.append((0.1, None))
            elif 'unlimited' in feature or 'throttled' in feature or 'has_unlimited' in feature:
                # Unlimited/throttled features: minimum ₩100, max ₩20,000
                bounds.append((100.0, 20000.0))
            else:
                # All other features: non-negative
                bounds.append((0.0, None))
        
        # Store bounds for later reference
        self.coefficient_bounds = bounds
        
        # Check multicollinearity before optimization
        correlation_matrix = np.corrcoef(X_matrix.T)
        max_correlation = np.max(np.abs(correlation_matrix - np.eye(len(features))))
        logger.info(f"Maximum feature correlation: {max_correlation:.3f}")
        
        if max_correlation > 0.8:
            logger.warning(f"High multicollinearity detected (max correlation: {max_correlation:.3f})")
            logger.info(f"Ridge regularization (α={alpha}) will help stabilize coefficients")
        
        # Store X and y for performance metrics calculation
        self._last_X = X_matrix
        self._last_y = y
        
        # Solve with Ridge regularization and exact second-order information
        try:
            # Use initial guess of small positive values
            initial_guess = np.ones(len(features)) * 10.0
            
            # For Ridge regression, we have analytical solution when α > 0
            if alpha > 0:
                logger.info(f"Using analytical Ridge solution (α={alpha})")
                try:
                    # Analytical Ridge solution: β = (X'X + αI)^(-1) X'y
                    ridge_matrix = XTX + alpha * I
                    
                    # Ensure matrix is well-conditioned
                    cond_num = np.linalg.cond(ridge_matrix)
                    logger.info(f"Ridge matrix condition number: {cond_num:.2e}")
                    
                    if cond_num < 1e12:  # Well-conditioned
                        beta_ridge = np.linalg.solve(ridge_matrix, XTy)
                        
                        # Apply bounds constraints by projection
                        for i, (lower, upper) in enumerate(bounds):
                            if lower is not None:
                                beta_ridge[i] = max(beta_ridge[i], lower)
                            if upper is not None:
                                beta_ridge[i] = min(beta_ridge[i], upper)
                        
                        # Add intercept of 0 (no base cost)
                        coefficients = np.concatenate([[0.0], beta_ridge])
                        
                        logger.info(f"Analytical Ridge solution successful")
                        return coefficients
                    else:
                        logger.warning(f"Ridge matrix poorly conditioned (cond={cond_num:.2e}), using optimization")
                        
                except np.linalg.LinAlgError as e:
                    logger.warning(f"Analytical Ridge solution failed: {e}, using optimization")
            
            # Fallback to iterative optimization with more relaxed settings
            try:
                logger.info(f"Using L-BFGS-B optimization with relaxed settings")
                result = minimize(
                    objective,
                    initial_guess,
                    method='L-BFGS-B',
                    bounds=bounds,
                    jac=gradient,
                    options={
                        'maxiter': 1000,  # Reduced from default
                        'ftol': 1e-6,     # Less strict tolerance
                        'gtol': 1e-5,     # Less strict gradient tolerance
                        'maxfun': 5000    # Limit function evaluations
                    }
                )
                
                if result.success:
                    logger.info(f"L-BFGS-B optimization successful (iterations: {result.nit})")
                    # Add intercept of 0 (no base cost)
                    coefficients = np.concatenate([[0.0], result.x])
                    return coefficients
                else:
                    logger.warning(f"L-BFGS-B failed: {result.message}")
                    
            except Exception as opt_error:
                logger.warning(f"L-BFGS-B optimization error: {opt_error}")
            
            # Final fallback: Use sklearn Ridge with projected bounds
            logger.info("Using simple Ridge regression with bounds projection")
            if alpha > 0:
                from sklearn.linear_model import Ridge as SklearnRidge
                ridge_model = SklearnRidge(alpha=alpha, fit_intercept=False, max_iter=1000)
                ridge_model.fit(X_matrix, y)
                beta_simple = ridge_model.coef_
            else:
                from sklearn.linear_model import LinearRegression
                ols_model = LinearRegression(fit_intercept=False)
                ols_model.fit(X_matrix, y)
                beta_simple = ols_model.coef_
            
            # Apply bounds by simple projection
            for i, (lower, upper) in enumerate(bounds):
                if lower is not None:
                    beta_simple[i] = max(beta_simple[i], lower)
                if upper is not None:
                    beta_simple[i] = min(beta_simple[i], upper)
            
            # Add intercept of 0 (no base cost)
            coefficients = np.concatenate([[0.0], beta_simple])
            logger.info("Simple Ridge regression with bounds projection successful")
            return coefficients
            
        except Exception as e:
            # Absolute fallback: equal weight method
            logger.error(f"All Ridge optimization methods failed: {e}")
            logger.info("Using equal weight fallback method")
            
            # Simple equal weight method
            equal_weight = np.mean(y) / len(features)
            beta_fallback = np.full(len(features), max(equal_weight, 1.0))
            
            # Apply bounds
            for i, (lower, upper) in enumerate(bounds):
                if lower is not None:
                    beta_fallback[i] = max(beta_fallback[i], lower)
                if upper is not None:
                    beta_fallback[i] = min(beta_fallback[i], upper)
            
            coefficients = np.concatenate([[0.0], beta_fallback])
            logger.info("Equal weight fallback method applied")
            return coefficients
    
    def get_coefficient_breakdown(self) -> dict:
        """
        Get coefficient breakdown for visualization.
        
        Returns:
            Dictionary with coefficient information including both raw and constrained values
        """
        if self.coefficients is None:
            logger.error("Cannot get coefficient breakdown: coefficients not solved yet")
            raise ValueError("Must solve coefficients first")
            
        analysis_features = [f for f in self.features if f in self.all_plans.columns]
        logger.info(f"Creating coefficient breakdown for {len(analysis_features)} features")
        
        breakdown = {
            'base_cost': 0.0,  # No base cost - regression through origin
            'feature_costs': {},
            'total_plans_used': len(self.all_plans) if self.all_plans is not None else 0,
            'outliers_removed': self.outliers_removed,
            'features_analyzed': len(analysis_features),
            'method': 'full_dataset_ridge',
            'alpha': self.alpha,
            'optimization_method': getattr(self, 'optimization_method', 'unknown'),
            'optimization_iterations': getattr(self, 'optimization_iterations', 0)
        }
        
        # Include both unconstrained and constrained coefficients
        for i, feature in enumerate(analysis_features):
            feature_data = {
                'coefficient': float(self.coefficients[i+1]),
                'cost_per_unit': float(self.coefficients[i+1])
            }
            
            # Add unconstrained coefficient if available
            if hasattr(self, 'unconstrained_coefficients') and self.unconstrained_coefficients is not None:
                feature_data['unconstrained_coefficient'] = float(self.unconstrained_coefficients[i])
                feature_data['adjustment'] = float(self.coefficients[i+1]) - float(self.unconstrained_coefficients[i])
            
            # Add bounds information if available
            if hasattr(self, 'coefficient_bounds') and self.coefficient_bounds is not None:
                if i < len(self.coefficient_bounds):
                    lower_bound, upper_bound = self.coefficient_bounds[i]
                    feature_data['bounds'] = {
                        'lower': float(lower_bound) if lower_bound is not None else None,
                        'upper': float(upper_bound) if upper_bound is not None else None
                    }
            
            breakdown['feature_costs'][feature] = feature_data
        
        logger.info(f"Successfully created coefficient breakdown with {len(breakdown['feature_costs'])} features")
        return breakdown 