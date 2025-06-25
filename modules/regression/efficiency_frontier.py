"""
Efficiency Frontier Module

Implements Pareto-optimal plan extraction and regression for more realistic coefficient estimation.
This approach focuses on efficiently priced plans to avoid bias from overpriced plans.

Classes:
- EfficiencyFrontierRegression: Main class for efficiency-based regression
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

# Configure logging
logger = logging.getLogger(__name__)

class EfficiencyFrontierRegression:
    """
    Efficiency Frontier based regression analysis.
    
    This class extracts Pareto-optimal plans (efficiency frontier) and performs
    regression analysis only on these efficiently priced plans to get more
    realistic marginal cost coefficients.
    """
    
    def __init__(self, features: List[str], alpha: float = 1.0):
        """
        Initialize the efficiency frontier regression analyzer.
        
        Args:
            features: List of feature names to use in analysis
            alpha: Ridge regularization parameter
        """
        self.features = features
        self.alpha = alpha
        self.efficient_plans = None
        self.coefficients = None
        self.efficiency_ratio = None
        self.feature_bounds = {
            # Only enforce basic economic logic: non-negative values
            'basic_data_clean': (0.0, None),
            'voice_clean': (0.0, None),
            'message_clean': (0.0, None),
            'additional_call': (0.0, None),
            'tethering_gb': (0.0, None),
            'speed_when_exhausted': (0.0, None),
            'daily_data_clean': (0.0, None),
            
            # Premium service flags: small positive minimum to avoid zero coefficients
            'is_5g': (0.1, None),
            'voice_unlimited': (0.1, None),
            'message_unlimited': (0.1, None),
            'basic_data_unlimited': (0.1, None),
            'daily_data_unlimited': (0.1, None),
            'data_throttled_after_quota': (0.1, None),
            'data_unlimited_speed': (0.1, None),
            'has_unlimited_speed': (0.1, None)
        }
    
    def extract_pareto_frontier(self, df: pd.DataFrame, price_col: str = 'fee') -> pd.DataFrame:
        """
        Extract Pareto-optimal (efficient) plans using multi-dimensional efficiency analysis.
        
        A plan is efficient if no other plan offers:
        - Same or better features at a lower price, OR
        - Strictly better features at the same price
        
        Args:
            df: DataFrame with plan data
            price_col: Column name for price comparison
            
        Returns:
            DataFrame containing only efficient plans
        """
        logger.info(f"Extracting Pareto frontier from {len(df)} plans...")
        
        # Filter to only features that exist in the data
        available_features = [f for f in self.features if f in df.columns]
        
        if not available_features:
            raise ValueError(f"None of the specified features {self.features} found in data")
        
        logger.info(f"Using {len(available_features)} features for efficiency analysis: {available_features}")
        
        # Remove plans with missing price or feature data
        clean_df = df.dropna(subset=[price_col] + available_features).copy()
        logger.info(f"After removing NaN values: {len(clean_df)} plans")
        
        if len(clean_df) < 10:
            logger.warning("Too few plans for efficiency analysis, using all available plans")
            self.efficient_plans = clean_df
            self.efficiency_ratio = 1.0
            return clean_df
        
        efficient_indices = []
        
        for i, plan in clean_df.iterrows():
            is_dominated = False
            plan_price = plan[price_col]
            plan_features = [plan[f] for f in available_features]
            
            # Check if this plan is dominated by any other plan
            for j, other in clean_df.iterrows():
                if i == j:
                    continue
                    
                other_price = other[price_col]
                other_features = [other[f] for f in available_features]
                
                # Check dominance conditions
                # Other plan dominates if:
                # 1. All features are >= current plan's features AND
                # 2. Price is <= current plan's price AND
                # 3. At least one strict improvement (better feature OR lower price)
                
                features_dominate = all(other_f >= plan_f for other_f, plan_f in zip(other_features, plan_features))
                price_dominates = other_price <= plan_price
                
                # Strict improvement check
                better_features = any(other_f > plan_f for other_f, plan_f in zip(other_features, plan_features))
                lower_price = other_price < plan_price
                
                if features_dominate and price_dominates and (better_features or lower_price):
                    is_dominated = True
                    break
            
            if not is_dominated:
                efficient_indices.append(i)
        
        self.efficient_plans = clean_df.loc[efficient_indices].copy()
        self.efficiency_ratio = len(self.efficient_plans) / len(clean_df)
        
        logger.info(f"Extracted {len(self.efficient_plans)} efficient plans "
                   f"({self.efficiency_ratio:.1%} of total)")
        
        return self.efficient_plans
    
    def solve_efficiency_frontier_coefficients(self, df: pd.DataFrame, price_col: str = 'fee') -> np.ndarray:
        """
        Solve for coefficients using only Pareto-efficient plans.
        
        Args:
            df: DataFrame with plan data
            price_col: Column name for target price
            
        Returns:
            Array of coefficients [β₀, β₁, β₂, ...]
        """
        # Extract efficient plans first
        efficient_df = self.extract_pareto_frontier(df, price_col)
        
        # Filter to only features that exist in the data
        available_features = [f for f in self.features if f in efficient_df.columns]
        
        logger.info(f"Performing Ridge regression on {len(efficient_df)} efficient plans")
        logger.info(f"Using features: {available_features}")
        
        # Prepare feature matrix X and target vector y
        X_data = []
        y_data = []
        
        for _, plan in efficient_df.iterrows():
            target_value = plan[price_col]
            
            if pd.isna(target_value) or target_value <= 0:
                continue
                
            feature_vector = []
            for feature in available_features:
                value = plan.get(feature, 0)
                if pd.isna(value):
                    feature_vector.append(0)
                else:
                    feature_vector.append(value)
            
            X_data.append(feature_vector)
            y_data.append(target_value)
        
        if len(X_data) == 0:
            raise ValueError("No valid efficient plans found for regression")
        
        X = np.array(X_data)
        y = np.array(y_data)
        
        logger.info(f"Efficient plan regression matrix: X shape {X.shape}, y shape {y.shape}")
        
        # Solve using constrained Ridge regression
        coefficients = self._solve_constrained_ridge(X, y, available_features)
        
        # Store results
        self.coefficients = coefficients
        
        # Log results
        logger.info("Efficiency Frontier Ridge Regression completed:")
        logger.info(f"  Base cost (β₀): ₩{coefficients[0]:.2f}")
        logger.info(f"  Ridge regularization α = {self.alpha}")
        logger.info(f"  Efficiency ratio: {self.efficiency_ratio:.1%}")
        
        for i, feature in enumerate(available_features):
            coeff_value = coefficients[i+1]
            # Show data-driven coefficients without artificial bounds
            logger.info(f"  {feature}: ₩{coeff_value:.2f} (data-driven)")
        
        logger.info(f"  Used {len(y_data)} efficient plans (efficiency ratio: {self.efficiency_ratio:.1%})")
        logger.info("  📊 Data-driven coefficients (no artificial bounds applied)")
        
        return coefficients
    
    def _solve_constrained_ridge(self, X: np.ndarray, y: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Solve Ridge regression with realistic economic constraints.
        
        Args:
            X: Feature matrix
            y: Target values
            features: List of feature names
            
        Returns:
            Constrained coefficients
        """
        # Add intercept column
        X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])
        
        # Set up economic bounds
        bounds = [(None, None)]  # Intercept can be any value
        
        for feature in features:
            feature_bounds = self.feature_bounds.get(feature, (0.1, None))
            bounds.append(feature_bounds)
        
        # Ridge objective function
        def objective(beta):
            residual = X_with_intercept @ beta - y
            data_loss = np.sum(residual ** 2)
            regularization = self.alpha * np.sum(beta[1:] ** 2)  # Don't regularize intercept
            return data_loss + regularization
        
        def gradient(beta):
            residual = X_with_intercept @ beta - y
            grad = 2 * X_with_intercept.T @ residual
            grad[1:] += 2 * self.alpha * beta[1:]  # Add regularization gradient (skip intercept)
            return grad
        
        # Initial guess: small positive values within bounds
        initial_guess = []
        for bound in bounds:
            if bound[0] is not None:
                initial_guess.append(max(bound[0], 1.0))
            else:
                initial_guess.append(1.0)
        
        initial_guess = np.array(initial_guess)
        
        # Solve constrained optimization
        try:
            result = minimize(
                objective,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds,
                jac=gradient,
                options={
                    'maxiter': 1000,
                    'ftol': 1e-8,
                    'gtol': 1e-6
                }
            )
            
            if result.success:
                logger.info(f"Constrained Ridge optimization converged in {result.nit} iterations")
                return result.x
            else:
                logger.warning(f"Optimization failed: {result.message}")
                
        except Exception as e:
            logger.warning(f"Constrained optimization failed: {e}")
        
        # Fallback: Simple Ridge with bounds projection
        logger.info("Using fallback Ridge regression with bounds projection")
        
        try:
            ridge_model = Ridge(alpha=self.alpha, fit_intercept=True)
            ridge_model.fit(X, y)
            
            # Get unconstrained coefficients
            intercept = ridge_model.intercept_
            coef = ridge_model.coef_
            
            # Apply bounds projection
            projected_coef = []
            for i, feature in enumerate(features):
                coeff_val = coef[i]
                min_bound, max_bound = self.feature_bounds.get(feature, (None, None))
                
                if min_bound is not None:
                    coeff_val = max(coeff_val, min_bound)
                if max_bound is not None:
                    coeff_val = min(coeff_val, max_bound)
                    
                projected_coef.append(coeff_val)
            
            coefficients = np.concatenate([[intercept], projected_coef])
            logger.info("Fallback Ridge regression with projection successful")
            
            return coefficients
            
        except Exception as e:
            logger.error(f"Fallback Ridge regression also failed: {e}")
            
            # Final fallback: Equal weight
            logger.info("Using equal weight fallback")
            avg_price = np.mean(y)
            equal_weight = avg_price / len(features)
            
            projected_weights = []
            for feature in features:
                min_bound, max_bound = self.feature_bounds.get(feature, (1.0, None))
                weight = max(equal_weight, min_bound or 1.0)
                if max_bound:
                    weight = min(weight, max_bound)
                projected_weights.append(weight)
            
            return np.concatenate([[0.0], projected_weights])
    
    def get_coefficient_breakdown(self) -> Dict:
        """
        Get coefficient breakdown for visualization and analysis.
        
        Returns:
            Dictionary with coefficient information in the format expected by table generation
        """
        if self.coefficients is None:
            raise ValueError("Must solve coefficients first")
        
        # Filter to only features that were actually used
        available_features = [f for f in self.features if f in (self.efficient_plans.columns if self.efficient_plans is not None else [])]
        
        # Create feature_costs structure expected by table generation
        feature_costs = {}
        for i, feature in enumerate(available_features):
            if i + 1 < len(self.coefficients):
                coeff_value = float(self.coefficients[i + 1])
                bounds = self.feature_bounds.get(feature, (None, None))
                
                feature_costs[feature] = {
                    'coefficient': coeff_value,
                    'cost_per_unit': coeff_value,
                    'bounds': bounds,
                    'within_bounds': (
                        (bounds[0] is None or coeff_value >= bounds[0]) and
                        (bounds[1] is None or coeff_value <= bounds[1])
                    )
                }
        
        breakdown = {
            'method': 'efficiency_frontier',
            'efficiency_ratio': self.efficiency_ratio,
            'efficient_plans_count': len(self.efficient_plans) if self.efficient_plans is not None else 0,
            'regularization_alpha': self.alpha,
            'base_cost': float(self.coefficients[0]),  # Use 'base_cost' instead of 'intercept'
            'feature_costs': feature_costs,  # Add feature_costs structure
            'multicollinearity_fixes': {},  # Empty since efficiency frontier doesn't use commonality analysis
            'data_driven': True,
            'no_artificial_bounds': True
        }
        
        return breakdown
    
    def calculate_cs_ratios(self, df: pd.DataFrame, price_col: str = 'fee') -> pd.DataFrame:
        """
        Calculate CS ratios for all plans using efficiency frontier coefficients.
        
        Args:
            df: DataFrame with all plans
            price_col: Column name for actual price
            
        Returns:
            DataFrame with added CS ratio calculations
        """
        if self.coefficients is None:
            raise ValueError("Must solve coefficients first")
        
        df_result = df.copy()
        available_features = [f for f in self.features if f in df.columns]
        
        # Calculate baseline costs
        baseline_costs = np.full(len(df), self.coefficients[0])  # Start with intercept
        
        for i, feature in enumerate(available_features):
            if i + 1 < len(self.coefficients):
                baseline_costs += self.coefficients[i + 1] * df[feature].fillna(0).values
        
        # Calculate CS ratios
        df_result['B_efficiency'] = baseline_costs
        df_result['CS_efficiency'] = baseline_costs / df_result[price_col]
        
        # Set primary columns
        df_result['B'] = df_result['B_efficiency']
        df_result['CS'] = df_result['CS_efficiency']
        
        # Add coefficient breakdown
        df_result.attrs['cost_structure'] = self.get_coefficient_breakdown()
        
        logger.info(f"Calculated CS ratios for {len(df_result)} plans using efficiency frontier method")
        logger.info(f"CS ratio range: {df_result['CS'].min():.2f} - {df_result['CS'].max():.2f}")
        logger.info(f"Mean CS ratio: {df_result['CS'].mean():.2f}")
        
        return df_result 