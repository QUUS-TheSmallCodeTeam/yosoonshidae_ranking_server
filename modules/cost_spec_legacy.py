"""
Cost-Spec Ratio implementation for MVNO plan ranking.

This module calculates plan rankings based on a cost-spec ratio approach
which compares each plan's actual fee with a theoretical baseline cost derived
from a monotonic frontier of feature costs.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Callable
import logging
from scipy.optimize import minimize

# Import refactored regression classes
from .regression.full_dataset import FullDatasetMultiFeatureRegression
from .regression.multi_feature import MultiFeatureFrontierRegression

# Import refactored frontier functions
from .frontier.core import (
    create_robust_monotonic_frontier,
    calculate_feature_frontiers,
    estimate_frontier_value,
    calculate_plan_baseline_cost
)

# Import refactored cost-spec ratio functions
from .cost_spec.ratio import (
    calculate_cs_ratio,
    rank_plans_by_cs,
    calculate_cs_ratio_enhanced,
    rank_plans_by_cs_enhanced
)

# Configure logging
logger = logging.getLogger(__name__)

# Import feature definitions from config
from .config import FEATURE_SETS, UNLIMITED_FLAGS, CORE_FEATURES

class LinearDecomposition:
    """
    Linear decomposition approach to extract true marginal costs for individual features.
    
    This class solves the double-counting problem in Cost-Spec ratios by decomposing
    plan costs into constituent feature costs using constrained optimization:
    
    plan_cost = β₀ + β₁×data + β₂×voice + β₃×SMS + β₄×tethering + β₅×5G + ...
    
    Where β coefficients represent true marginal costs for each feature.
    """
    
    def __init__(self, tolerance=500, features=None):
        """
        Initialize the linear decomposition analyzer.
        
        Args:
            tolerance: Tolerance for constraint violations (KRW)
            features: List of features to include in decomposition
        """
        self.tolerance = tolerance
        self.features = features or ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb', 'is_5g']
        self.coefficients = None
        self.representative_plans = None
        
    def extract_representative_plans(self, df: pd.DataFrame, 
                                   selection_method: str = 'frontier_points') -> pd.DataFrame:
        """
        Extract frontier plans that represent optimal cost efficiency points.
        
        Args:
            df: DataFrame with plan data
            selection_method: Method for selecting representative plans ('frontier_points')
            
        Returns:
            DataFrame with selected frontier plans
        """
        if selection_method == 'frontier_points':
            # Use the same logic as the original frontier calculation
            # to identify the optimal plans for each feature
            
            # Calculate frontiers for each feature
            frontiers = calculate_feature_frontiers(df, self.features, UNLIMITED_FLAGS, 'original_fee')
            
            # Collect all unique plans that contribute to any frontier
            frontier_plan_indices = set()
            
            for feature in self.features:
                if feature not in frontiers:
                    continue
                    
                frontier = frontiers[feature]
                
                # For each frontier point, find the plan that contributes to it
                for feature_value, cost in frontier.items():
                    # Find plans with this exact feature value and cost
                    matching_plans = df[
                        (df[feature] == feature_value) & 
                        (df['original_fee'] == cost)
                    ]
                    
                    if not matching_plans.empty:
                        # Add the first matching plan's index
                        frontier_plan_indices.add(matching_plans.index[0])
            
            # Extract the frontier plans
            if frontier_plan_indices:
                rep_df = df.loc[list(frontier_plan_indices)].copy()
            else:
                # Fallback: if no frontier plans found, use a small diverse sample
                logger.warning("No frontier plans found, using fallback selection")
                rep_df = df.sample(min(5, len(df))).copy()
                
        elif selection_method == 'diverse_segments':
            # Keep the old method as fallback
            representatives = []
            
            # Budget segment (lowest cost plans)
            budget_plans = df[df['fee'] <= df['fee'].quantile(0.3)].copy()
            if not budget_plans.empty:
                representatives.append(budget_plans.loc[budget_plans['fee'].idxmin()])
            
            # Entry level (low-mid cost with some features)
            entry_plans = df[(df['fee'] > df['fee'].quantile(0.3)) & 
                           (df['fee'] <= df['fee'].quantile(0.6))].copy()
            if not entry_plans.empty:
                representatives.append(entry_plans.loc[entry_plans['fee'].idxmin()])
            
            # Premium segment (high cost plans)
            premium_plans = df[df['fee'] > df['fee'].quantile(0.7)].copy()
            if not premium_plans.empty:
                representatives.append(premium_plans.loc[premium_plans['fee'].idxmin()])
            
            # Data-heavy strategy (plans with high data allowances)
            if 'basic_data_clean' in df.columns:
                data_heavy = df.nlargest(min(3, len(df)), 'basic_data_clean')
                if not data_heavy.empty:
                    representatives.append(data_heavy.iloc[0])
            
            # Voice-heavy strategy (plans with high voice minutes)
            if 'voice_clean' in df.columns:
                voice_heavy = df.nlargest(min(3, len(df)), 'voice_clean')
                if not voice_heavy.empty:
                    representatives.append(voice_heavy.iloc[0])
            
            # Remove duplicates by index
            unique_representatives = []
            seen_indices = set()
            for plan in representatives:
                if plan.name not in seen_indices:
                    unique_representatives.append(plan)
                    seen_indices.add(plan.name)
            
            rep_df = pd.DataFrame(unique_representatives)
            
        else:
            raise ValueError(f"Unknown selection method: {selection_method}")
        
        logger.info(f"Selected {len(rep_df)} representative plans using {selection_method} method")
        self.representative_plans = rep_df
        return rep_df
    
    def solve_coefficients(self, representative_plans: pd.DataFrame, 
                          fee_column: str = 'fee') -> np.ndarray:
        """
        Solve for marginal cost coefficients using constrained optimization.
        
        Args:
            representative_plans: DataFrame with selected representative plans
            fee_column: Column containing plan fees
            
        Returns:
            Array of solved coefficients [β₀, β₁, β₂, ...]
        """
        # Prepare feature matrix X and cost vector c
        X = np.column_stack([
            np.ones(len(representative_plans)),  # β₀ (base cost)
            representative_plans[self.features].values  # β₁, β₂, β₃, ...
        ])
        
        c = representative_plans[fee_column].values
        
        logger.info(f"Solving linear system with {len(representative_plans)} plans and {X.shape[1]} coefficients")
        
        # Objective function: minimize sum of squared residuals
        def objective(beta):
            residuals = X @ beta - c
            return np.sum(residuals ** 2)
        
        # Constraint: frontier respect (solutions must be >= actual costs - tolerance)
        def frontier_constraints(beta):
            predicted_costs = X @ beta
            return predicted_costs - (c - self.tolerance)
        
        # Setup constraints
        constraints = [
            {'type': 'ineq', 'fun': frontier_constraints}
        ]
        
        # Calculate minimum base cost to ensure mathematical feasibility
        max_features = np.max(X[:, 1:], axis=0)
        min_cost = np.min(c)
        min_base_cost = max(0, min_cost - np.sum(max_features) * 10)  # Conservative bound
        
        # Bounds: sample-data-driven only
        bounds = [(min_base_cost, None)]  # β₀: base cost
        bounds.extend([(0, None)] * len(self.features))  # β₁, β₂, ...: non-negative marginal costs
        
        # Initial guess using simple regression
        try:
            beta_ols = np.linalg.lstsq(X, c, rcond=None)[0]
            initial_guess = np.maximum(beta_ols, 0)
            initial_guess[0] = max(initial_guess[0], min_base_cost)
        except:
            avg_cost = np.mean(c)
            initial_guess = [avg_cost * 0.4] + [avg_cost * 0.1] * len(self.features)
        
        # Solve optimization
        result = minimize(
            objective,
            x0=initial_guess,
            method='trust-constr',
            constraints=constraints,
            bounds=bounds,
            options={'disp': False, 'maxiter': 1000}
        )
        
        if not result.success:
            logger.error(f"Linear decomposition optimization failed: {result.message}")
            raise RuntimeError("Failed to solve linear decomposition")
        
        self.coefficients = result.x
        
        # Validation logging
        predicted = X @ self.coefficients
        actual = c
        mae = np.mean(np.abs(predicted - actual))
        max_error = np.max(np.abs(predicted - actual))
        
        logger.info(f"Linear decomposition solved successfully:")
        logger.info(f"  Base cost (β₀): ₩{self.coefficients[0]:,.0f}")
        for i, feature in enumerate(self.features):
            logger.info(f"  {feature} cost: ₩{self.coefficients[i+1]:,.2f}")
        logger.info(f"  Mean absolute error: ₩{mae:,.0f}")
        logger.info(f"  Max absolute error: ₩{max_error:,.0f}")
        
        return self.coefficients
    
    def calculate_decomposed_baselines(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculate baselines using linear decomposition coefficients.
        
        Args:
            df: DataFrame with plan data
            
        Returns:
            Array of decomposed baseline costs
        """
        if self.coefficients is None:
            raise ValueError("Must solve coefficients first using solve_coefficients()")
        
        # Calculate baselines: β₀ + Σ(βᵢ × featureᵢ)
        baselines = np.full(len(df), self.coefficients[0])  # Start with base cost
        
        for i, feature in enumerate(self.features):
            if feature in df.columns:
                baselines += self.coefficients[i+1] * df[feature].values
