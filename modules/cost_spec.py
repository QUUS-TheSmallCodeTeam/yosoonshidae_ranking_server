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
        # REMOVED: only 'data_stops_after_quota' per user request
    ]
}

# Unlimited flag mappings
UNLIMITED_FLAGS = {
    'basic_data_clean': 'basic_data_unlimited',
    'daily_data_clean': 'daily_data_unlimited',
    'voice_clean': 'voice_unlimited',
    'message_clean': 'message_unlimited',
    'speed_when_exhausted': 'has_unlimited_speed'
}

# Use all features for frontier calculation, not just core ones
CORE_FEATURES = FEATURE_SETS['basic']

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
            else:
                logger.warning(f"Feature {feature} not found in data, treating as 0")
        
        return baselines

def create_robust_monotonic_frontier(df_feature_specific: pd.DataFrame, 
                                   feature_col: str, 
                                   cost_col: str) -> pd.Series:
    """
    Create a robust, strictly increasing monotonic frontier for a given feature.
    This ensures the frontier "crawls the bottom" of optimal points.
    
    Args:
        df_feature_specific: DataFrame filtered for non-unlimited values of the specific feature.
        feature_col: The name of the feature column (e.g., 'basic_data_clean').
        cost_col: The name of the column representing the cost for that feature (e.g., 'fee' or 'contribution_feature').
                  For calculate_feature_frontiers, this will typically be the overall plan 'fee'.
        
    Returns:
        A pandas Series where the index is the feature value and the values are the frontier costs.
    """
    if df_feature_specific.empty:
        return pd.Series(dtype=float)

    # Step 1: Identify candidate points with tie-breaking logic
    # For each unique cost, find the point with maximum feature value
    # This ensures we pick higher spec points when costs are the same
    cost_to_max_feature = {}
    for _, row in df_feature_specific.iterrows():
        cost = row[cost_col]
        feature_val = row[feature_col]
        
        if cost not in cost_to_max_feature or feature_val > cost_to_max_feature[cost]['feature_val']:
            cost_to_max_feature[cost] = {
                'feature_val': feature_val,
                'row_index': row.name
            }
    
    # Now for each feature value, find the minimum cost among the selected high-spec points
    feature_to_min_cost = {}
    for cost_info in cost_to_max_feature.values():
        row = df_feature_specific.loc[cost_info['row_index']]
        feature_val = row[feature_col]
        cost = row[cost_col]
        
        if feature_val not in feature_to_min_cost or cost < feature_to_min_cost[feature_val]['cost']:
            feature_to_min_cost[feature_val] = {
                'cost': cost,
                'row_index': cost_info['row_index']
            }
    
    # Create candidate_points_df from the selected indices
    selected_indices = [cost_info['row_index'] for cost_info in feature_to_min_cost.values()]
    candidate_points_df = df_feature_specific.loc[selected_indices]
    candidate_points_df = candidate_points_df.sort_values(by=[feature_col, cost_col])
    
    # Calculate the smallest feature value unit increase in the dataset
    sorted_feature_values = sorted(df_feature_specific[feature_col].unique())
    min_feature_increment = float('inf')
    for i in range(1, len(sorted_feature_values)):
        increment = sorted_feature_values[i] - sorted_feature_values[i-1]
        if increment > 0 and increment < min_feature_increment:
            min_feature_increment = increment
    
    # If no valid increment found (e.g., only one unique value), use a small default
    if min_feature_increment == float('inf'):
        min_feature_increment = 0.1
        
    logger.info(f"Smallest feature increment for {feature_col}: {min_feature_increment}")
    
    candidate_details = []
    for _, row in candidate_points_df.iterrows():
        candidate_details.append({
            'value': row[feature_col],
            'cost': row[cost_col]
            # We don't need plan_name here for cost_spec.py's frontier generation
        })

    # Step 2: Build the true monotonic frontier with minimum 1 KRW cost increase rule
    actual_frontier_stack = []
    should_add_zero_point = True
    
    for candidate in candidate_details:
        current_value = candidate['value']
        current_cost = candidate['cost']

        # Allow the addition of the candidate if it completely dominates the frontier so far
        while actual_frontier_stack:
            last_frontier_point = actual_frontier_stack[-1]
            last_value = last_frontier_point['value']
            last_cost = last_frontier_point['cost']

            # If the candidate is more optimal, we remove points and recheck conditions
            if current_value > last_value and current_cost < last_cost:
                actual_frontier_stack.pop()
                should_add_zero_point = True  # We need to reconsider adding the (0,0) point
            else:
                break

        # Check if the candidate can be added based on monotonic increase rule
        if actual_frontier_stack:
            # Remove points from the end of the frontier that conflict with adding this candidate
            while actual_frontier_stack:
                last_frontier_point = actual_frontier_stack[-1]
                last_value = last_frontier_point['value']
                last_cost = last_frontier_point['cost']

                # Skip this candidate if it has same or lower feature value
                if current_value <= last_value:
                    break  # Cannot add this candidate
                
                # Skip this candidate if it has same or lower cost
                if current_cost <= last_cost:
                    break  # Cannot add this candidate
                
                cost_per_unit = (current_cost - last_cost) / (current_value - last_value)
                if cost_per_unit >= 1.0:
                    # This candidate can be added - it meets all criteria
                    break
                else:
                    # Remove the last point and try again with the previous point
                    actual_frontier_stack.pop()
                    should_add_zero_point = True  # Reconsider adding zero point
            
            # If we still have points in the frontier, check one more time if we can add the candidate
            if actual_frontier_stack:
                last_frontier_point = actual_frontier_stack[-1]
                last_value = last_frontier_point['value']
                last_cost = last_frontier_point['cost']
                
                if (current_value > last_value and 
                    current_cost > last_cost and
                    (current_cost - last_cost) / (current_value - last_value) >= 1.0):
                    actual_frontier_stack.append(candidate)
                    if current_value > 0:
                        should_add_zero_point = False
                # If criteria not met, skip this candidate
            else:
                # Frontier is empty, add this as first point
                actual_frontier_stack.append(candidate)
                if current_value > 0:  # Only disable zero point if we have a non-zero value
                    should_add_zero_point = False
        else:
            # First candidate point
            actual_frontier_stack.append(candidate)
            if current_value > 0:  # Only disable zero point if we have a non-zero value
                should_add_zero_point = False
            
    if not actual_frontier_stack:
        return pd.Series(dtype=float)
    
    # Add (0,0) as the starting point if conditions are met
    if should_add_zero_point:
        # Create a synthetic starting point at (0,0)
        zero_point = {'value': 0, 'cost': 0}
        # Insert at the beginning
        actual_frontier_stack.insert(0, zero_point)
        logger.info(f"Added (0,0) starting point to frontier for {feature_col}")

    # Check for the max feature value and see if we need to add a proper endpoint
    all_feature_values = df_feature_specific[feature_col].values
    max_feature_value = max(all_feature_values) if len(all_feature_values) > 0 else 0
    
    # If the highest feature value is not in our frontier, find the best cost for it
    if max_feature_value > 0 and (not actual_frontier_stack or max_feature_value > actual_frontier_stack[-1]['value']):
        max_value_rows = df_feature_specific[df_feature_specific[feature_col] == max_feature_value]
        if not max_value_rows.empty:
            min_cost_for_max = max_value_rows[cost_col].min()
            max_point = {'value': max_feature_value, 'cost': min_cost_for_max}
            
            # Only add if it maintains monotonicity and 1.0 KRW minimum increase
            if not actual_frontier_stack:
                actual_frontier_stack.append(max_point)
                logger.info(f"Added endpoint ({max_feature_value},{min_cost_for_max}) to frontier for {feature_col}")
            else:
                last_point = actual_frontier_stack[-1]
                cost_per_unit = (min_cost_for_max - last_point['cost']) / (max_feature_value - last_point['value'])
                if (min_cost_for_max > last_point['cost'] and cost_per_unit >= 1.0):
                    actual_frontier_stack.append(max_point)
                    logger.info(f"Added endpoint ({max_feature_value},{min_cost_for_max}) to frontier for {feature_col}")

    # Convert stack to pandas Series
    frontier_s = pd.Series({p['value']: p['cost'] for p in actual_frontier_stack})
    frontier_s = frontier_s.sort_index() # Ensure it's sorted by feature value
    return frontier_s

def calculate_feature_frontiers(df: pd.DataFrame, features: List[str], 
                              unlimited_flags: Dict[str, str], 
                              fee_column: str = 'fee') -> Dict[str, pd.Series]:
    """
    Compute cost frontiers for each feature using the robust monotonicity logic.
    The `fee_column` here is the overall plan fee, used to establish the initial feature cost frontiers.
    """
    frontiers = {}
    
    # Define the cost column to be used for creating feature frontiers.
    # This should be 'original_fee' as per the new requirement for B calculation.
    cost_col_for_frontier_creation = 'original_fee'

    for feature in features:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in dataframe for frontier calculation, skipping")
            continue

        if cost_col_for_frontier_creation not in df.columns:
            logger.error(f"Cost column '{cost_col_for_frontier_creation}' not found in DataFrame. Cannot calculate frontiers.")
            continue

        if feature in unlimited_flags.values():
            continue

        unlimited_flag = unlimited_flags.get(feature)
        
        if unlimited_flag and unlimited_flag in df.columns:
            # Process non-unlimited plans for this feature
            df_non_unlimited = df[(df[unlimited_flag] == 0) & df[cost_col_for_frontier_creation].notna()].copy()
            if not df_non_unlimited.empty:
                robust_frontier = create_robust_monotonic_frontier(df_non_unlimited, feature, cost_col_for_frontier_creation)
                if not robust_frontier.empty:
                    frontiers[feature] = robust_frontier
                    logger.info(f"Created ROBUST monotonic frontier for {feature} with {len(robust_frontier)} points using '{cost_col_for_frontier_creation}'")
                else:
                    logger.warning(f"Robust frontier for {feature} (using '{cost_col_for_frontier_creation}') is empty.")
            else: # This else corresponds to `if not df_non_unlimited.empty:`
                logger.info(f"No non-unlimited plans with valid '{cost_col_for_frontier_creation}' for {feature} to build its main frontier.")

            # Process unlimited plans for this feature
            unlimited_plans_df = df[(df[unlimited_flag] == 1) & df[cost_col_for_frontier_creation].notna()]
            if not unlimited_plans_df.empty:
                min_cost_unlimited = unlimited_plans_df[cost_col_for_frontier_creation].min()
                frontiers[unlimited_flag] = pd.Series([min_cost_unlimited])
                logger.info(f"Added unlimited case for {feature} (as {unlimited_flag}) with '{cost_col_for_frontier_creation}' {min_cost_unlimited}")
            else: # This else corresponds to `if not unlimited_plans_df.empty:` for unlimited_plans_df
                logger.info(f"No unlimited plans with valid '{cost_col_for_frontier_creation}' found for {feature} (flag: {unlimited_flag})")

        else:  # This else corresponds to `if unlimited_flag and unlimited_flag in df.columns:`
            # Feature does not have an unlimited flag
            df_feature_specific = df[df[cost_col_for_frontier_creation].notna()].copy()
            if not df_feature_specific.empty:
                robust_frontier = create_robust_monotonic_frontier(df_feature_specific, feature, cost_col_for_frontier_creation)
                if not robust_frontier.empty:
                    frontiers[feature] = robust_frontier
                    logger.info(f"Created ROBUST monotonic frontier for {feature} (no unlimited option, using '{cost_col_for_frontier_creation}') with {len(robust_frontier)} points")
                else:  # This else corresponds to `if not robust_frontier.empty:` for robust_frontier
                    logger.warning(f"Robust frontier for {feature} (no unlimited option, using '{cost_col_for_frontier_creation}') is empty.")
            else:  # This else corresponds to `if not df_feature_specific.empty:`
                logger.warning(f"No plans with valid '{cost_col_for_frontier_creation}' for {feature} (no unlimited option) to build frontier.")
    
    return frontiers

def estimate_frontier_value(feature_value: float, frontier: pd.Series) -> float:
    """
    Estimate the frontier value for a given feature value.
    
    Args:
        feature_value: The feature value to estimate
        frontier: The frontier series indexed by feature values
        
    Returns:
        The estimated frontier value
    """
    if frontier.empty:
        logger.warning(f"Attempting to estimate value from an empty frontier for feature value {feature_value}. Returning 0.0.")
        return 0.0

    if feature_value in frontier.index:
        # Exact match
        return frontier[feature_value]
    
    # np.searchsorted finds the insertion point to maintain order.
    # The feature values in frontier.index are sorted.
    idx = np.searchsorted(frontier.index, feature_value)
    
    if idx == 0:
        # Feature value is lower than any in frontier, use the cost of the smallest feature value.
        # This is extrapolation using the first point.
        return frontier.iloc[0]
    elif idx == len(frontier.index): # Corrected to len(frontier.index) as idx can be equal to length
        # Feature value is higher than any in frontier, use the cost of the largest feature value.
        # This is extrapolation using the last point.
        return frontier.iloc[-1]
    else:
        # Feature value is between two frontier points. Perform linear interpolation.
        x1 = frontier.index[idx-1]
        y1 = frontier.iloc[idx-1]
        x2 = frontier.index[idx]
        y2 = frontier.iloc[idx]

        # Prevent division by zero if feature values are identical (should not happen with strictly monotonic frontier)
        if x2 == x1:
            logger.warning(f"Frontier points for interpolation have same feature value ({x1}). Returning cost of first point ({y1}).")
            return y1 
        
        # Linear interpolation formula: y = y1 + (x - x1) * (y2 - y1) / (x2 - x1)
        interpolated_cost = y1 + (feature_value - x1) * (y2 - y1) / (x2 - x1)
        return interpolated_cost

def calculate_plan_baseline_cost(row: pd.Series, frontiers: Dict[str, pd.Series],
                               unlimited_flags: Dict[str, str]) -> float:
    """
    Calculate the theoretical baseline cost for a single plan using feature frontiers.
    
    Args:
        row: Series containing a single plan's data
        frontiers: Dictionary mapping features to their cost frontiers
        unlimited_flags: Mapping of feature columns to their unlimited flag columns
        
    Returns:
        Total baseline cost for the plan
    """
    total_cost = 0.0
    
    # Calculate cost for each feature (excluding unlimited flags which are handled with their features)
    for feature in [f for f in CORE_FEATURES if f not in unlimited_flags.values()]:
        # Skip if feature not available
        if feature not in row:
            continue
            
        # Check if this feature has an unlimited flag
        unlimited_flag = unlimited_flags.get(feature)
        
        if unlimited_flag and unlimited_flag in row and row[unlimited_flag] == 1:
            # This feature is unlimited for this plan
            # Use the minimum fee among unlimited plans for this feature
            if unlimited_flag in frontiers:
                total_cost += frontiers[unlimited_flag].iloc[0]
        else:
            # This feature is not unlimited or doesn't have an unlimited option
            feature_value = row[feature]
            
            # Get the frontier-based cost for this feature value
            if feature in frontiers:
                # Check if the index is numeric using pandas-version-agnostic approach
                if pd.api.types.is_numeric_dtype(frontiers[feature].index):
                    # Numeric index - use estimation
                    frontier_value = estimate_frontier_value(feature_value, frontiers[feature])
                    total_cost += frontier_value
                else:
                    # Categorical index - direct lookup
                    if feature_value in frontiers[feature].index:
                        total_cost += frontiers[feature][feature_value]
    
    return total_cost

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
        method: Calculation method ('frontier', 'linear_decomposition', 'multi_frontier', or 'fixed_rates')
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
        logger.info("Starting fixed rates method using pure coefficients")
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
            # Use FullDatasetMultiFeatureRegression to get pure coefficients
            regressor = FullDatasetMultiFeatureRegression(features=analysis_features)
            
            # Solve for pure marginal costs using entire dataset (no filtering)
            coefficients = regressor.solve_full_dataset_coefficients(df)
            logger.info(f"Successfully solved fixed rate coefficients: {coefficients}")
            
            # Calculate baselines using pure coefficients for ALL plans (no base cost)
            baselines = np.zeros(len(df))  # Start from 0 (no base cost)
            
            for i, feature in enumerate(analysis_features):
                if feature in df.columns:
                    # Use the actual feature values (continuous features already zeroed out for unlimited plans, 
                    # unlimited flags have their own coefficients)
                    baselines += coefficients[i+1] * df[feature].values
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
            
            logger.info(f"Created fixed rates breakdown: {coefficient_breakdown}")
            logger.info(f"Processed {len(df_result)} plans using fixed marginal rates")
            
            return df_result
            
        except Exception as e:
            logger.error(f"Fixed rates method failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            # Fallback to frontier method
            logger.info("Falling back to frontier method")
            return calculate_cs_ratio(df, feature_set, fee_column)

    elif method == 'linear_decomposition':
        # Use linear decomposition method
        logger.info("Starting linear decomposition method")
        df_result = df.copy()
        
        # Get features for this feature set
        if feature_set in FEATURE_SETS:
            features = [f for f in FEATURE_SETS[feature_set] if f not in UNLIMITED_FLAGS.values()]
        else:
            raise ValueError(f"Unknown feature set: {feature_set}")
        
        logger.info(f"Using features for decomposition: {features}")
        
        # Initialize linear decomposition
        tolerance = method_kwargs.get('tolerance', 500)
        # Get all features for this feature set (excluding unlimited flags)
        if feature_set in FEATURE_SETS:
            all_features = [f for f in FEATURE_SETS[feature_set] if f not in UNLIMITED_FLAGS.values()]
        else:
            # Fallback to safe features if feature_set not found
            all_features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb', 'is_5g']
        
        # Only use features that actually exist in the dataframe
        decomp_features = [f for f in all_features if f in df.columns]
        if len(decomp_features) < 3:
            logger.warning(f"Not enough valid features for decomposition: {decomp_features}")
            logger.info("Falling back to frontier method due to insufficient features")
            return calculate_cs_ratio(df, feature_set, fee_column)
        decomp_features = method_kwargs.get('features', decomp_features)
        
        logger.info(f"Decomposition features: {decomp_features}")
        
        try:
            decomposer = LinearDecomposition(tolerance=tolerance, features=decomp_features)
            
            # Extract representative plans using frontier points
            selection_method = method_kwargs.get('selection_method', 'frontier_points')
            representative_plans = decomposer.extract_representative_plans(df, selection_method)
            logger.info(f"Extracted {len(representative_plans)} representative plans")
            
            # Solve for coefficients using 'original_fee' to match frontier selection
            coefficients = decomposer.solve_coefficients(representative_plans, 'original_fee')
            logger.info(f"Successfully solved coefficients: {coefficients}")
            
            # Calculate decomposed baselines
            baselines = decomposer.calculate_decomposed_baselines(df)
            logger.info(f"Calculated baselines for {len(baselines)} plans")
            
            # Add results to dataframe
            df_result['B_decomposed'] = baselines
            df_result['CS_decomposed'] = baselines / df_result[fee_column]
            
            # Also calculate traditional method for comparison
            df_traditional = calculate_cs_ratio(df, feature_set, fee_column)
            df_result['B_frontier'] = df_traditional['B']
            df_result['CS_frontier'] = df_traditional['CS']
            
            # Set primary columns to decomposed values
            df_result['B'] = df_result['B_decomposed']
            df_result['CS'] = df_result['CS_decomposed']
            
            # Add coefficient information as metadata (both formats for compatibility)
            cost_structure = {
                'base_cost': float(coefficients[0]),
                'feature_costs': {
                    feature: {
                        'coefficient': float(coef),
                        'min_increment': 1,
                        'cost_per_unit': float(coef)
                    } for feature, coef in zip(decomp_features, coefficients[1:])
                }
            }
            df_result.attrs['decomposition_coefficients'] = cost_structure
            df_result.attrs['cost_structure'] = cost_structure
            
            logger.info(f"Created cost_structure: {cost_structure}")
            
            return df_result
            
        except Exception as e:
            logger.error(f"Linear decomposition failed: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            # Fallback to frontier method
            logger.info("Falling back to frontier method")
            return calculate_cs_ratio(df, feature_set, fee_column)
    
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
        raise ValueError(f"Unknown method: {method}. Supported methods: 'frontier', 'linear_decomposition', 'multi_frontier', 'fixed_rates'")

def rank_plans_by_cs_enhanced(df: pd.DataFrame, method: str = 'frontier',
                            feature_set: str = 'basic', fee_column: str = 'fee',
                            top_n: Optional[int] = None, **method_kwargs) -> pd.DataFrame:
    """
    Enhanced plan ranking supporting multiple CS calculation methods.
    
    Args:
        df: DataFrame with plan data
        method: Calculation method ('frontier' or 'linear_decomposition')
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

class MultiFeatureFrontierRegression:
    """
    Multi-Feature Frontier Regression implementation.
    
    Solves the cross-contamination problem by:
    1. Collecting plans from all feature frontiers
    2. Performing multi-feature regression on complete feature vectors
    3. Extracting pure marginal costs for each feature
    """
    
    def __init__(self, features=None):
        """
        Initialize the multi-feature frontier regression analyzer.
        
        Args:
            features: List of features to analyze. If None, uses CORE_FEATURES.
        """
        self.features = features or CORE_FEATURES
        self.frontier_plans = None
        self.coefficients = None
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
    
    def solve_multi_feature_coefficients(self, df: pd.DataFrame) -> np.ndarray:
        """
        Solve for pure marginal costs using multi-feature regression.
        
        Args:
            df: DataFrame with plan data
            
        Returns:
            Array of coefficients [β₀, β₁, β₂, ...]
        """
        # Step 1: Collect frontier plans
        frontier_plans = self.collect_all_frontier_plans(df)
        
        if len(frontier_plans) < len(self.features) + 1:
            raise ValueError(f"Insufficient frontier plans ({len(frontier_plans)}) for {len(self.features)} features")
        
        # Step 2: Calculate minimum increments
        self.calculate_min_increments(df)
        
        # Step 3: Build feature matrix (exclude unlimited flags)
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
        
        # Step 4: Solve constrained regression (Ridge disabled per user request)
        if self.multicollinearity_detected:
            logger.info("Multicollinearity detected but using constrained regression (Ridge disabled per user request)")
        else:
            logger.info("Using constrained least squares (no multicollinearity)")
        
        # Always use constrained regression
        coefficients = self._solve_constrained_regression(X, y, analysis_features)
        
        # MULTICOLLINEARITY FIX: Handle high correlations by redistributing coefficients
        if self.multicollinearity_detected and hasattr(self, 'correlation_matrix'):
            coefficients = self._fix_multicollinearity_coefficients(coefficients, analysis_features)
        
        self.coefficients = coefficients
        
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
            from scipy.optimize import minimize
            
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

    def _fix_multicollinearity_coefficients(self, coefficients: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Fix multicollinearity by redistributing coefficients for highly correlated features.
        
        Args:
            coefficients: Original coefficients [β₀, β₁, β₂, ...]
            features: List of feature names
            
        Returns:
            Adjusted coefficients with redistributed values
        """
        if not hasattr(self, 'correlation_matrix') or self.correlation_matrix is None:
            return coefficients
        
        fixed_coefficients = coefficients.copy()
        
        # Store multicollinearity fixes for detailed reporting
        if not hasattr(self, 'multicollinearity_fixes'):
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
    
    def get_coefficient_breakdown(self) -> dict:
        """
        Get coefficient breakdown for visualization.
        
        Returns:
            Dictionary with coefficient information including both raw and constrained values
        """
        if self.coefficients is None:
            raise ValueError("Must solve coefficients first")
            
        analysis_features = [f for f in self.features if f in self.frontier_plans.columns]
        
        breakdown = {
            'base_cost': 0.0,  # No base cost - regression through origin
            'feature_costs': {},
            'total_plans_used': len(self.frontier_plans) if self.frontier_plans is not None else 0,
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

    def validate_optimization_quality(self, X, y, coefficients, bounds):
        """
        최적화가 local minima에 빠졌는지 검증
        """
        from scipy.optimize import minimize
        
        def objective(beta):
            return np.sum((X @ beta - y) ** 2)
        
        # 여러 다른 초기값으로 최적화 재실행
        initial_guesses = [
            np.ones(len(coefficients)-1) * 1.0,    # 작은 값 (intercept 제외)
            np.ones(len(coefficients)-1) * 100.0,  # 중간 값
            np.ones(len(coefficients)-1) * 1000.0, # 큰 값
            np.random.uniform(1, 1000, len(coefficients)-1),  # 랜덤
            np.random.uniform(10, 500, len(coefficients)-1),  # 다른 랜덤
        ]
        
        results = []
        convergence_info = []
        for i, guess in enumerate(initial_guesses):
            try:
                result = minimize(
                    objective, 
                    guess, 
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-6}
                )
                if result.success:
                    results.append(result.fun)  # 목적함수 값
                    convergence_info.append({
                        'initial_guess': i,
                        'objective_value': result.fun,
                        'converged': True,
                        'iterations': result.nit
                    })
                else:
                    convergence_info.append({
                        'initial_guess': i,
                        'objective_value': float('inf'),
                        'converged': False,
                        'message': result.message
                    })
            except Exception as e:
                convergence_info.append({
                    'initial_guess': i,
                    'objective_value': float('inf'),
                    'converged': False,
                    'error': str(e)
                })
        
        # 모든 결과가 비슷하면 global optimum 가능성 높음
        if len(results) > 1:
            objective_values = np.array(results)
            consistency = np.std(objective_values) / np.mean(objective_values) if np.mean(objective_values) > 0 else float('inf')
        else:
            consistency = float('inf')
        
        return {
            'is_consistent': consistency < 0.01,  # 1% 이내 변동
            'objective_std': consistency,
            'all_results': results,
            'convergence_details': convergence_info,
            'successful_optimizations': len(results)
        }

    def validate_economic_logic(self, df, coefficients, features):
        """
        계수들이 경제적으로 말이 되는지 검증
        """
        validation_results = {}
        
        try:
            # 1. 스케일 검증: 데이터 1GB vs 5G 지원 비용 비교
            data_coeff = None
            fiveg_coeff = None
            
            if 'basic_data_clean' in features:
                data_idx = features.index('basic_data_clean')
                data_coeff = coefficients[data_idx + 1]  # Skip intercept
            
            if 'is_5g' in features:
                fiveg_idx = features.index('is_5g')
                fiveg_coeff = coefficients[fiveg_idx + 1]  # Skip intercept
            
            if data_coeff is not None and fiveg_coeff is not None:
                validation_results['scale_check'] = {
                    'data_per_gb': data_coeff,
                    'fiveg_premium': fiveg_coeff,
                    'ratio': fiveg_coeff / data_coeff if data_coeff != 0 else float('inf'),
                    'makes_sense': fiveg_coeff > data_coeff * 10,  # 5G가 10GB 데이터보다 비싸야 함
                    'economic_reasoning': '5G 지원 비용이 데이터 1GB 비용의 10배 이상이어야 경제적으로 타당함'
                }
            
            # 2. 순서 검증: 기본 기능 < 프리미엄 기능
            voice_coeff = None
            tethering_coeff = None
            
            if 'voice_clean' in features:
                voice_idx = features.index('voice_clean')
                voice_coeff = coefficients[voice_idx + 1]
            
            if 'tethering_gb' in features:
                tethering_idx = features.index('tethering_gb')
                tethering_coeff = coefficients[tethering_idx + 1]
            
            if voice_coeff is not None and tethering_coeff is not None:
                validation_results['premium_check'] = {
                    'voice_per_min': voice_coeff,
                    'tethering_per_gb': tethering_coeff,
                    'ratio': tethering_coeff / voice_coeff if voice_coeff != 0 else float('inf'),
                    'makes_sense': tethering_coeff > voice_coeff,  # 테더링이 음성보다 비싸야 함
                    'economic_reasoning': '테더링이 일반 음성통화보다 단위당 비용이 높아야 함'
                }
            
            # 3. 양수 검증: 모든 계수가 경제적으로 의미있는 값인지
            positive_check = {}
            negative_coefficients = []
            zero_coefficients = []
            
            for i, feature in enumerate(features):
                coeff = coefficients[i + 1]  # Skip intercept
                if coeff < 0:
                    negative_coefficients.append((feature, coeff))
                elif coeff == 0:
                    zero_coefficients.append((feature, coeff))
            
            positive_check = {
                'negative_count': len(negative_coefficients),
                'zero_count': len(zero_coefficients),
                'negative_features': negative_coefficients,
                'zero_features': zero_coefficients,
                'all_positive': len(negative_coefficients) == 0 and len(zero_coefficients) == 0,
                'economic_reasoning': '모든 기능은 비용을 증가시켜야 하므로 양수 계수를 가져야 함'
            }
            
            validation_results['positive_check'] = positive_check
            
        except Exception as e:
            validation_results['error'] = f"Economic logic validation failed: {str(e)}"
        
        return validation_results

    def validate_prediction_power(self, df, features):
        """
        모델이 실제로 시장 가격을 잘 예측하는지 검증 (Cross-Validation)
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, mean_absolute_error
        
        try:
            X = df[features].values
            y = df['fee'].values
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores = []
            mae_scores = []
            fold_details = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 학습 데이터로 계수 계산
                train_coeffs = self._solve_constrained_regression(X_train, y_train, features)
                
                # 테스트 데이터로 예측
                y_pred = X_test @ train_coeffs[1:]  # Skip intercept
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                r2_scores.append(r2)
                mae_scores.append(mae)
                
                fold_details.append({
                    'fold': fold + 1,
                    'r2_score': r2,
                    'mae': mae,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                })
            
            return {
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'mean_mae': np.mean(mae_scores),
                'std_mae': np.std(mae_scores),
                'is_stable': np.std(r2_scores) < 0.1,  # R² 표준편차가 0.1 미만
                'fold_details': fold_details,
                'all_r2_scores': r2_scores,
                'all_mae_scores': mae_scores
            }
            
        except Exception as e:
            return {
                'error': f"Prediction power validation failed: {str(e)}",
                'mean_r2': 0.0,
                'std_r2': 0.0,
                'mean_mae': float('inf'),
                'std_mae': 0.0,
                'is_stable': False
            }

    def analyze_residuals(self, df, predicted, actual):
        """
        잔차 패턴을 분석하여 모델의 적합성 검증
        """
        try:
            residuals = actual - predicted
            
            # 1. 잔차의 패턴 검사
            import scipy.stats as stats
            
            # 정규성 검정
            normality_test = stats.jarque_bera(residuals)
            
            # 이분산성 검정 (가격 구간별로 잔차 분산이 다른지)
            price_quartiles = np.quantile(actual, [0.25, 0.5, 0.75])
            residual_vars = []
            quartile_info = []
            
            for i in range(len(price_quartiles) + 1):
                if i == 0:
                    mask = actual <= price_quartiles[0]
                    quartile_name = f"Q1 (≤ ₩{price_quartiles[0]:,.0f})"
                elif i == len(price_quartiles):
                    mask = actual > price_quartiles[-1]
                    quartile_name = f"Q4 (> ₩{price_quartiles[-1]:,.0f})"
                else:
                    mask = (actual > price_quartiles[i-1]) & (actual <= price_quartiles[i])
                    quartile_name = f"Q{i+1} (₩{price_quartiles[i-1]:,.0f} - ₩{price_quartiles[i]:,.0f})"
                
                if np.sum(mask) > 0:
                    quartile_residuals = residuals[mask]
                    residual_var = np.var(quartile_residuals)
                    residual_vars.append(residual_var)
                    quartile_info.append({
                        'quartile': quartile_name,
                        'count': np.sum(mask),
                        'residual_variance': residual_var,
                        'residual_std': np.std(quartile_residuals),
                        'mean_residual': np.mean(quartile_residuals)
                    })
            
            # 잔차 분산의 일관성
            heteroscedasticity = np.std(residual_vars) / np.mean(residual_vars) if np.mean(residual_vars) > 0 else float('inf')
            
            # 이상치 잔차 분석
            residual_std = np.std(residuals)
            outlier_threshold = 3 * residual_std
            outlier_residuals = np.sum(np.abs(residuals) > outlier_threshold)
            
            return {
                'mean_residual': np.mean(residuals),
                'residual_std': residual_std,
                'residual_normality': {
                    'statistic': normality_test.statistic,
                    'p_value': normality_test.pvalue,
                    'is_normal': normality_test.pvalue > 0.05
                },
                'heteroscedasticity': {
                    'coefficient': heteroscedasticity,
                    'is_homoscedastic': heteroscedasticity < 0.5,
                    'quartile_analysis': quartile_info
                },
                'outlier_analysis': {
                    'count': outlier_residuals,
                    'percentage': (outlier_residuals / len(residuals)) * 100,
                    'threshold': outlier_threshold
                }
            }
            
        except Exception as e:
            return {
                'error': f"Residual analysis failed: {str(e)}",
                'mean_residual': 0.0,
                'residual_normality': {'is_normal': False},
                'heteroscedasticity': {'is_homoscedastic': False},
                'outlier_analysis': {'count': 0, 'percentage': 0.0}
            }

    def calculate_overall_validation_score(self, validation_report):
        """
        종합 검증 점수 계산 (0-100점)
        """
        score = 0
        detailed_scoring = {}
        
        # 최적화 일관성 (25점)
        optimization_score = 0
        if validation_report.get('optimization', {}).get('is_consistent', False):
            optimization_score = 25
        elif validation_report.get('optimization', {}).get('successful_optimizations', 0) >= 3:
            optimization_score = 15  # 부분 점수
        detailed_scoring['optimization'] = optimization_score
        score += optimization_score
        
        # 경제적 타당성 (25점)
        economic_score = 0
        econ = validation_report.get('economic_logic', {})
        
        # 스케일 검증
        scale_ok = econ.get('scale_check', {}).get('makes_sense', False)
        premium_ok = econ.get('premium_check', {}).get('makes_sense', False)
        positive_ok = econ.get('positive_check', {}).get('all_positive', False)
        
        if scale_ok and premium_ok and positive_ok:
            economic_score = 25
        elif (scale_ok and premium_ok) or (scale_ok and positive_ok) or (premium_ok and positive_ok):
            economic_score = 17
        elif scale_ok or premium_ok or positive_ok:
            economic_score = 8
        detailed_scoring['economic_logic'] = economic_score
        score += economic_score
        
        # 예측력 (30점)
        prediction_score = 0
        pred = validation_report.get('prediction_power', {})
        mean_r2 = pred.get('mean_r2', 0)
        is_stable = pred.get('is_stable', False)
        
        if mean_r2 > 0.8 and is_stable:
            prediction_score = 30
        elif mean_r2 > 0.6 and is_stable:
            prediction_score = 22
        elif mean_r2 > 0.6:
            prediction_score = 20
        elif mean_r2 > 0.4:
            prediction_score = 10
        elif mean_r2 > 0.2:
            prediction_score = 5
        detailed_scoring['prediction_power'] = prediction_score
        score += prediction_score
        
        # 잔차 품질 (20점)
        residual_score = 0
        resid = validation_report.get('residual_analysis', {})
        is_normal = resid.get('residual_normality', {}).get('is_normal', False)
        is_homoscedastic = resid.get('heteroscedasticity', {}).get('is_homoscedastic', False)
        outlier_pct = resid.get('outlier_analysis', {}).get('percentage', 100)
        
        if is_normal and is_homoscedastic and outlier_pct < 5:
            residual_score = 20
        elif (is_normal and is_homoscedastic) or (is_normal and outlier_pct < 5) or (is_homoscedastic and outlier_pct < 5):
            residual_score = 12
        elif is_normal or is_homoscedastic or outlier_pct < 10:
            residual_score = 6
        detailed_scoring['residual_quality'] = residual_score
        score += residual_score
        
        return {
            'total_score': score,
            'grade': 'A' if score >= 85 else 'B' if score >= 70 else 'C' if score >= 55 else 'D' if score >= 40 else 'F',
            'detailed_scoring': detailed_scoring,
            'score_breakdown': {
                'optimization_consistency': f"{optimization_score}/25",
                'economic_logic': f"{economic_score}/25", 
                'prediction_power': f"{prediction_score}/30",
                'residual_quality': f"{residual_score}/20"
            }
        }

    def comprehensive_model_validation(self, df, X, y, coefficients, features, bounds):
        """
        종합적인 모델 검증
        """
        validation_report = {}
        
        logger.info("Starting comprehensive model validation...")
        
        # 1. 최적화 품질
        try:
            validation_report['optimization'] = self.validate_optimization_quality(X, y, coefficients, bounds)
            logger.info("✓ Optimization quality validation completed")
        except Exception as e:
            logger.error(f"✗ Optimization validation failed: {e}")
            validation_report['optimization'] = {'error': str(e), 'is_consistent': False}
        
        # 2. 경제적 타당성
        try:
            validation_report['economic_logic'] = self.validate_economic_logic(df, coefficients, features)
            logger.info("✓ Economic logic validation completed")
        except Exception as e:
            logger.error(f"✗ Economic logic validation failed: {e}")
            validation_report['economic_logic'] = {'error': str(e)}
        
        # 3. 예측력
        try:
            validation_report['prediction_power'] = self.validate_prediction_power(df, features)
            logger.info("✓ Prediction power validation completed")
        except Exception as e:
            logger.error(f"✗ Prediction power validation failed: {e}")
            validation_report['prediction_power'] = {'error': str(e), 'mean_r2': 0.0, 'is_stable': False}
        
        # 4. 잔차 분석
        try:
            predicted = X @ coefficients[1:]  # Skip intercept
            validation_report['residual_analysis'] = self.analyze_residuals(df, predicted, y)
            logger.info("✓ Residual analysis completed")
        except Exception as e:
            logger.error(f"✗ Residual analysis failed: {e}")
            validation_report['residual_analysis'] = {'error': str(e)}
        
        # 5. 종합 점수
        try:
            validation_report['overall_score'] = self.calculate_overall_validation_score(validation_report)
            logger.info(f"✓ Overall validation score: {validation_report['overall_score']['total_score']}/100 ({validation_report['overall_score']['grade']})")
        except Exception as e:
            logger.error(f"✗ Overall scoring failed: {e}")
            validation_report['overall_score'] = {'error': str(e), 'total_score': 0, 'grade': 'F'}
        
        return validation_report

class FullDatasetMultiFeatureRegression:
    """
    Alternative to MultiFeatureFrontierRegression that uses ALL plans in the dataset
    instead of just frontier plans. This captures market-wide pricing patterns
    but requires careful outlier handling and multicollinearity detection.
    """
    
    def __init__(self, features=None, outlier_threshold=3.0, alpha=1.0):
        # Use the correct feature set from FEATURE_SETS
        if features is None:
            features = FEATURE_SETS['basic']
        self.features = features
        self.outlier_threshold = outlier_threshold  # Standard deviations for outlier detection
        self.alpha = alpha  # Ridge regression regularization parameter
        self.coefficients = None
        self.all_plans = None
        self.outliers_removed = 0
        self.correlation_matrix = None
        self.multicollinearity_detected = False
        
    def detect_multicollinearity(self, X: np.ndarray, feature_names: List[str], threshold: float = 0.8) -> Dict:
        """
        Detect multicollinearity using correlation matrix and variance inflation factor.
        
        Args:
            X: Feature matrix (without intercept)
            feature_names: List of feature names
            threshold: Correlation threshold for detecting multicollinearity
            
        Returns:
            Dictionary with multicollinearity analysis results
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)
        self.correlation_matrix = pd.DataFrame(corr_matrix, 
                                             index=feature_names, 
                                             columns=feature_names)
        
        # Find high correlations
        high_correlations = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > threshold:
                    high_correlations.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': corr_val
                    })
        
        self.multicollinearity_detected = len(high_correlations) > 0
        
        analysis = {
            'high_correlations': high_correlations,
            'multicollinearity_detected': self.multicollinearity_detected,
            'correlation_matrix': self.correlation_matrix
        }
        
        if self.multicollinearity_detected:
            logger.warning(f"Multicollinearity detected: {len(high_correlations)} high correlations found")
            for hc in high_correlations:
                logger.warning(f"  {hc['feature1']} ↔ {hc['feature2']}: {hc['correlation']:.3f}")
        
        return analysis

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
        Solve for coefficients using ALL plans in the dataset with multicollinearity handling.
        
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
        
        # Step 3: Detect multicollinearity
        multicollinearity_analysis = self.detect_multicollinearity(X, analysis_features)
        
        # Step 4: Always use constrained regression (Ridge disabled per user request)
        if self.multicollinearity_detected:
            logger.info("Multicollinearity detected but using constrained regression (Ridge disabled per user request)")
        else:
            logger.info("Using constrained least squares (no multicollinearity)")
        
        # Always use constrained regression
        coefficients = self._solve_constrained_regression(X, y, analysis_features)
        
        # MULTICOLLINEARITY FIX: Handle high correlations by redistributing coefficients
        if self.multicollinearity_detected and hasattr(self, 'correlation_matrix'):
            coefficients = self._fix_multicollinearity_coefficients(coefficients, analysis_features)
        
        self.coefficients = coefficients
        
        # Validation logging (no intercept)
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
            from scipy.optimize import minimize
            
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

    def _fix_multicollinearity_coefficients(self, coefficients: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Fix multicollinearity by redistributing coefficients for highly correlated features.
        
        Args:
            coefficients: Original coefficients [β₀, β₁, β₂, ...]
            features: List of feature names
            
        Returns:
            Adjusted coefficients with redistributed values
        """
        if not hasattr(self, 'correlation_matrix') or self.correlation_matrix is None:
            return coefficients
        
        fixed_coefficients = coefficients.copy()
        
        # Store multicollinearity fixes for detailed reporting
        if not hasattr(self, 'multicollinearity_fixes'):
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
            
            # Add multicollinearity fix information if available
            if hasattr(self, 'multicollinearity_fixes') and feature in self.multicollinearity_fixes:
                multicollinearity_info = self.multicollinearity_fixes[feature]
                feature_data['multicollinearity_fix'] = {
                    'paired_with': multicollinearity_info['paired_with'],
                    'correlation': multicollinearity_info['correlation'],
                    'original_value': multicollinearity_info['original_value'],
                    'partner_original_value': multicollinearity_info['partner_original_value'],
                    'calculation_formula': multicollinearity_info['calculation_formula'],
                    'redistributed_value': multicollinearity_info['redistributed_value']
                }
            
            breakdown['feature_costs'][feature] = feature_data
        
        return breakdown

    def validate_optimization_quality(self, X, y, coefficients, bounds):
        """
        최적화가 local minima에 빠졌는지 검증
        """
        from scipy.optimize import minimize
        
        def objective(beta):
            return np.sum((X @ beta - y) ** 2)
        
        # 여러 다른 초기값으로 최적화 재실행
        initial_guesses = [
            np.ones(len(coefficients)-1) * 1.0,    # 작은 값 (intercept 제외)
            np.ones(len(coefficients)-1) * 100.0,  # 중간 값
            np.ones(len(coefficients)-1) * 1000.0, # 큰 값
            np.random.uniform(1, 1000, len(coefficients)-1),  # 랜덤
            np.random.uniform(10, 500, len(coefficients)-1),  # 다른 랜덤
        ]
        
        results = []
        convergence_info = []
        for i, guess in enumerate(initial_guesses):
            try:
                result = minimize(
                    objective, 
                    guess, 
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-6}
                )
                if result.success:
                    results.append(result.fun)  # 목적함수 값
                    convergence_info.append({
                        'initial_guess': i,
                        'objective_value': result.fun,
                        'converged': True,
                        'iterations': result.nit
                    })
                else:
                    convergence_info.append({
                        'initial_guess': i,
                        'objective_value': float('inf'),
                        'converged': False,
                        'message': result.message
                    })
            except Exception as e:
                convergence_info.append({
                    'initial_guess': i,
                    'objective_value': float('inf'),
                    'converged': False,
                    'error': str(e)
                })
        
        # 모든 결과가 비슷하면 global optimum 가능성 높음
        if len(results) > 1:
            objective_values = np.array(results)
            consistency = np.std(objective_values) / np.mean(objective_values) if np.mean(objective_values) > 0 else float('inf')
        else:
            consistency = float('inf')
        
        return {
            'is_consistent': consistency < 0.01,  # 1% 이내 변동
            'objective_std': consistency,
            'all_results': results,
            'convergence_details': convergence_info,
            'successful_optimizations': len(results)
        }

    def validate_economic_logic(self, df, coefficients, features):
        """
        계수들이 경제적으로 말이 되는지 검증
        """
        validation_results = {}
        
        try:
            # 1. 스케일 검증: 데이터 1GB vs 5G 지원 비용 비교
            data_coeff = None
            fiveg_coeff = None
            
            if 'basic_data_clean' in features:
                data_idx = features.index('basic_data_clean')
                data_coeff = coefficients[data_idx + 1]  # Skip intercept
            
            if 'is_5g' in features:
                fiveg_idx = features.index('is_5g')
                fiveg_coeff = coefficients[fiveg_idx + 1]  # Skip intercept
            
            if data_coeff is not None and fiveg_coeff is not None:
                validation_results['scale_check'] = {
                    'data_per_gb': data_coeff,
                    'fiveg_premium': fiveg_coeff,
                    'ratio': fiveg_coeff / data_coeff if data_coeff != 0 else float('inf'),
                    'makes_sense': fiveg_coeff > data_coeff * 10,  # 5G가 10GB 데이터보다 비싸야 함
                    'economic_reasoning': '5G 지원 비용이 데이터 1GB 비용의 10배 이상이어야 경제적으로 타당함'
                }
            
            # 2. 순서 검증: 기본 기능 < 프리미엄 기능
            voice_coeff = None
            tethering_coeff = None
            
            if 'voice_clean' in features:
                voice_idx = features.index('voice_clean')
                voice_coeff = coefficients[voice_idx + 1]
            
            if 'tethering_gb' in features:
                tethering_idx = features.index('tethering_gb')
                tethering_coeff = coefficients[tethering_idx + 1]
            
            if voice_coeff is not None and tethering_coeff is not None:
                validation_results['premium_check'] = {
                    'voice_per_min': voice_coeff,
                    'tethering_per_gb': tethering_coeff,
                    'ratio': tethering_coeff / voice_coeff if voice_coeff != 0 else float('inf'),
                    'makes_sense': tethering_coeff > voice_coeff,  # 테더링이 음성보다 비싸야 함
                    'economic_reasoning': '테더링이 일반 음성통화보다 단위당 비용이 높아야 함'
                }
            
            # 3. 양수 검증: 모든 계수가 경제적으로 의미있는 값인지
            positive_check = {}
            negative_coefficients = []
            zero_coefficients = []
            
            for i, feature in enumerate(features):
                coeff = coefficients[i + 1]  # Skip intercept
                if coeff < 0:
                    negative_coefficients.append((feature, coeff))
                elif coeff == 0:
                    zero_coefficients.append((feature, coeff))
            
            positive_check = {
                'negative_count': len(negative_coefficients),
                'zero_count': len(zero_coefficients),
                'negative_features': negative_coefficients,
                'zero_features': zero_coefficients,
                'all_positive': len(negative_coefficients) == 0 and len(zero_coefficients) == 0,
                'economic_reasoning': '모든 기능은 비용을 증가시켜야 하므로 양수 계수를 가져야 함'
            }
            
            validation_results['positive_check'] = positive_check
            
        except Exception as e:
            validation_results['error'] = f"Economic logic validation failed: {str(e)}"
        
        return validation_results

    def validate_prediction_power(self, df, features):
        """
        모델이 실제로 시장 가격을 잘 예측하는지 검증 (Cross-Validation)
        """
        from sklearn.model_selection import KFold
        from sklearn.metrics import r2_score, mean_absolute_error
        
        try:
            X = df[features].values
            y = df['fee'].values
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores = []
            mae_scores = []
            fold_details = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 학습 데이터로 계수 계산
                train_coeffs = self._solve_constrained_regression(X_train, y_train, features)
                
                # 테스트 데이터로 예측
                y_pred = X_test @ train_coeffs[1:]  # Skip intercept
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                r2_scores.append(r2)
                mae_scores.append(mae)
                
                fold_details.append({
                    'fold': fold + 1,
                    'r2_score': r2,
                    'mae': mae,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                })
            
            return {
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'mean_mae': np.mean(mae_scores),
                'std_mae': np.std(mae_scores),
                'is_stable': np.std(r2_scores) < 0.1,  # R² 표준편차가 0.1 미만
                'fold_details': fold_details,
                'all_r2_scores': r2_scores,
                'all_mae_scores': mae_scores
            }
            
        except Exception as e:
            return {
                'error': f"Prediction power validation failed: {str(e)}",
                'mean_r2': 0.0,
                'std_r2': 0.0,
                'mean_mae': float('inf'),
                'std_mae': 0.0,
                'is_stable': False
            }

    def analyze_residuals(self, df, predicted, actual):
        """
        잔차 패턴을 분석하여 모델의 적합성 검증
        """
        try:
            residuals = actual - predicted
            
            # 1. 잔차의 패턴 검사
            import scipy.stats as stats
            
            # 정규성 검정
            normality_test = stats.jarque_bera(residuals)
            
            # 이분산성 검정 (가격 구간별로 잔차 분산이 다른지)
            price_quartiles = np.quantile(actual, [0.25, 0.5, 0.75])
            residual_vars = []
            quartile_info = []
            
            for i in range(len(price_quartiles) + 1):
                if i == 0:
                    mask = actual <= price_quartiles[0]
                    quartile_name = f"Q1 (≤ ₩{price_quartiles[0]:,.0f})"
                elif i == len(price_quartiles):
                    mask = actual > price_quartiles[-1]
                    quartile_name = f"Q4 (> ₩{price_quartiles[-1]:,.0f})"
                else:
                    mask = (actual > price_quartiles[i-1]) & (actual <= price_quartiles[i])
                    quartile_name = f"Q{i+1} (₩{price_quartiles[i-1]:,.0f} - ₩{price_quartiles[i]:,.0f})"
                
                if np.sum(mask) > 0:
                    quartile_residuals = residuals[mask]
                    residual_var = np.var(quartile_residuals)
                    residual_vars.append(residual_var)
                    quartile_info.append({
                        'quartile': quartile_name,
                        'count': np.sum(mask),
                        'residual_variance': residual_var,
                        'residual_std': np.std(quartile_residuals),
                        'mean_residual': np.mean(quartile_residuals)
                    })
            
            # 잔차 분산의 일관성
            heteroscedasticity = np.std(residual_vars) / np.mean(residual_vars) if np.mean(residual_vars) > 0 else float('inf')
            
            # 이상치 잔차 분석
            residual_std = np.std(residuals)
            outlier_threshold = 3 * residual_std
            outlier_residuals = np.sum(np.abs(residuals) > outlier_threshold)
            
            return {
                'mean_residual': np.mean(residuals),
                'residual_std': residual_std,
                'residual_normality': {
                    'statistic': normality_test.statistic,
                    'p_value': normality_test.pvalue,
                    'is_normal': normality_test.pvalue > 0.05
                },
                'heteroscedasticity': {
                    'coefficient': heteroscedasticity,
                    'is_homoscedastic': heteroscedasticity < 0.5,
                    'quartile_analysis': quartile_info
                },
                'outlier_analysis': {
                    'count': outlier_residuals,
                    'percentage': (outlier_residuals / len(residuals)) * 100,
                    'threshold': outlier_threshold
                }
            }
            
        except Exception as e:
            return {
                'error': f"Residual analysis failed: {str(e)}",
                'mean_residual': 0.0,
                'residual_normality': {'is_normal': False},
                'heteroscedasticity': {'is_homoscedastic': False},
                'outlier_analysis': {'count': 0, 'percentage': 0.0}
            }

    def calculate_overall_validation_score(self, validation_report):
        """
        종합 검증 점수 계산 (0-100점)
        """
        score = 0
        detailed_scoring = {}
        
        # 최적화 일관성 (25점)
        optimization_score = 0
        if validation_report.get('optimization', {}).get('is_consistent', False):
            optimization_score = 25
        elif validation_report.get('optimization', {}).get('successful_optimizations', 0) >= 3:
            optimization_score = 15  # 부분 점수
        detailed_scoring['optimization'] = optimization_score
        score += optimization_score
        
        # 경제적 타당성 (25점)
        economic_score = 0
        econ = validation_report.get('economic_logic', {})
        
        # 스케일 검증
        scale_ok = econ.get('scale_check', {}).get('makes_sense', False)
        premium_ok = econ.get('premium_check', {}).get('makes_sense', False)
        positive_ok = econ.get('positive_check', {}).get('all_positive', False)
        
        if scale_ok and premium_ok and positive_ok:
            economic_score = 25
        elif (scale_ok and premium_ok) or (scale_ok and positive_ok) or (premium_ok and positive_ok):
            economic_score = 17
        elif scale_ok or premium_ok or positive_ok:
            economic_score = 8
        detailed_scoring['economic_logic'] = economic_score
        score += economic_score
        
        # 예측력 (30점)
        prediction_score = 0
        pred = validation_report.get('prediction_power', {})
        mean_r2 = pred.get('mean_r2', 0)
        is_stable = pred.get('is_stable', False)
        
        if mean_r2 > 0.8 and is_stable:
            prediction_score = 30
        elif mean_r2 > 0.6 and is_stable:
            prediction_score = 22
        elif mean_r2 > 0.6:
            prediction_score = 20
        elif mean_r2 > 0.4:
            prediction_score = 10
        elif mean_r2 > 0.2:
            prediction_score = 5
        detailed_scoring['prediction_power'] = prediction_score
        score += prediction_score
        
        # 잔차 품질 (20점)
        residual_score = 0
        resid = validation_report.get('residual_analysis', {})
        is_normal = resid.get('residual_normality', {}).get('is_normal', False)
        is_homoscedastic = resid.get('heteroscedasticity', {}).get('is_homoscedastic', False)
        outlier_pct = resid.get('outlier_analysis', {}).get('percentage', 100)
        
        if is_normal and is_homoscedastic and outlier_pct < 5:
            residual_score = 20
        elif (is_normal and is_homoscedastic) or (is_normal and outlier_pct < 5) or (is_homoscedastic and outlier_pct < 5):
            residual_score = 12
        elif is_normal or is_homoscedastic or outlier_pct < 10:
            residual_score = 6
        detailed_scoring['residual_quality'] = residual_score
        score += residual_score
        
        return {
            'total_score': score,
            'grade': 'A' if score >= 85 else 'B' if score >= 70 else 'C' if score >= 55 else 'D' if score >= 40 else 'F',
            'detailed_scoring': detailed_scoring,
            'score_breakdown': {
                'optimization_consistency': f"{optimization_score}/25",
                'economic_logic': f"{economic_score}/25", 
                'prediction_power': f"{prediction_score}/30",
                'residual_quality': f"{residual_score}/20"
            }
        }

    def comprehensive_model_validation(self, df, X, y, coefficients, features, bounds):
        """
        종합적인 모델 검증
        """
        validation_report = {}
        
        logger.info("Starting comprehensive model validation...")
        
        # 1. 최적화 품질
        try:
            validation_report['optimization'] = self.validate_optimization_quality(X, y, coefficients, bounds)
            logger.info("✓ Optimization quality validation completed")
        except Exception as e:
            logger.error(f"✗ Optimization validation failed: {e}")
            validation_report['optimization'] = {'error': str(e), 'is_consistent': False}
        
        # 2. 경제적 타당성
        try:
            validation_report['economic_logic'] = self.validate_economic_logic(df, coefficients, features)
            logger.info("✓ Economic logic validation completed")
        except Exception as e:
            logger.error(f"✗ Economic logic validation failed: {e}")
            validation_report['economic_logic'] = {'error': str(e)}
        
        # 3. 예측력
        try:
            validation_report['prediction_power'] = self.validate_prediction_power(df, features)
            logger.info("✓ Prediction power validation completed")
        except Exception as e:
            logger.error(f"✗ Prediction power validation failed: {e}")
            validation_report['prediction_power'] = {'error': str(e), 'mean_r2': 0.0, 'is_stable': False}
        
        # 4. 잔차 분석
        try:
            predicted = X @ coefficients[1:]  # Skip intercept
            validation_report['residual_analysis'] = self.analyze_residuals(df, predicted, y)
            logger.info("✓ Residual analysis completed")
        except Exception as e:
            logger.error(f"✗ Residual analysis failed: {e}")
            validation_report['residual_analysis'] = {'error': str(e)}
        
        # 5. 종합 점수
        try:
            validation_report['overall_score'] = self.calculate_overall_validation_score(validation_report)
            logger.info(f"✓ Overall validation score: {validation_report['overall_score']['total_score']}/100 ({validation_report['overall_score']['grade']})")
        except Exception as e:
            logger.error(f"✗ Overall scoring failed: {e}")
            validation_report['overall_score'] = {'error': str(e), 'total_score': 0, 'grade': 'F'}
        
        return validation_report 

def calculate_multiple_coefficient_sets(df: pd.DataFrame, method: str = 'fixed_rates',
                                      feature_set: str = 'basic', fee_column: str = 'fee',
                                      **method_kwargs) -> dict:
    """
    여러 다른 방법으로 coefficient 세트를 계산하고 비교 검증
    
    Args:
        df: 전처리된 DataFrame
        method: 기본 방법
        feature_set: 사용할 feature 세트
        fee_column: 비용 컬럼
        **method_kwargs: 추가 파라미터
        
    Returns:
        여러 coefficient 세트와 검증 결과들
    """
    logger.info("🔬 Starting multiple coefficient sets calculation and validation...")
    
    multiple_results = {
        'method_comparisons': {},
        'validation_comparisons': {},
        'best_method': None,
        'consensus_coefficients': {},
        'reliability_analysis': {}
    }
    
    # 여러 방법으로 coefficient 계산
    methods_to_test = [
        {'name': 'fixed_rates_conservative', 'alpha': 10.0, 'tolerance': 1e-8},
        {'name': 'fixed_rates_standard', 'alpha': 1.0, 'tolerance': 1e-6},
        {'name': 'fixed_rates_aggressive', 'alpha': 0.1, 'tolerance': 1e-4},
        {'name': 'fixed_rates_random_init_1', 'alpha': 1.0, 'random_seed': 42},
        {'name': 'fixed_rates_random_init_2', 'alpha': 1.0, 'random_seed': 123},
    ]
    
    coefficient_results = {}
    validation_results = {}
    
    for method_config in methods_to_test:
        method_name = method_config['name']
        logger.info(f"🧮 Testing method: {method_name}")
        
        try:
            # FullDatasetMultiFeatureRegression 인스턴스 생성
            if 'alpha' in method_config:
                regressor = FullDatasetMultiFeatureRegression(
                    features=None, 
                    outlier_threshold=3.0, 
                    alpha=method_config['alpha']
                )
            else:
                regressor = FullDatasetMultiFeatureRegression()
            
            # 계수 계산
            coefficients = regressor.solve_full_dataset_coefficients(df)
            analysis_features = [f for f in regressor.features if f in df.columns]
            
            # X, y 준비
            X = df[analysis_features].values
            y = df[fee_column].values
            
            # 검증 수행
            validation_report = regressor.comprehensive_model_validation(
                df, X, y, coefficients, analysis_features, regressor.coefficient_bounds
            )
            
            # 결과 저장
            coefficient_results[method_name] = {
                'coefficients': coefficients,
                'features': analysis_features,
                'method_config': method_config,
                'cost_breakdown': regressor.get_coefficient_breakdown()
            }
            
            validation_results[method_name] = validation_report
            
            score = validation_report.get('overall_score', {}).get('total_score', 0)
            logger.info(f"  ✓ {method_name}: Score {score}/100 ({validation_report.get('overall_score', {}).get('grade', 'F')})")
            
        except Exception as e:
            logger.error(f"  ✗ {method_name} failed: {str(e)}")
            coefficient_results[method_name] = {'error': str(e)}
            validation_results[method_name] = {'error': str(e), 'overall_score': {'total_score': 0, 'grade': 'F'}}
    
    # 최고 방법 선택
    best_score = 0
    best_method = None
    for method_name, validation in validation_results.items():
        score = validation.get('overall_score', {}).get('total_score', 0)
        if score > best_score:
            best_score = score
            best_method = method_name
    
    multiple_results['best_method'] = best_method
    multiple_results['method_comparisons'] = coefficient_results
    multiple_results['validation_comparisons'] = validation_results
    
    # Consensus coefficients 계산 (성공한 방법들의 평균)
    successful_methods = [name for name, result in coefficient_results.items() 
                         if 'coefficients' in result and 'error' not in result]
    
    if len(successful_methods) > 1:
        logger.info(f"📊 Calculating consensus coefficients from {len(successful_methods)} successful methods")
        
        # 각 feature별로 coefficient들의 평균과 표준편차 계산
        consensus_coeffs = {}
        reliability_analysis = {}
        
        if successful_methods:
            # 첫 번째 성공한 방법의 feature 리스트 가져오기
            first_method = coefficient_results[successful_methods[0]]
            features = first_method['features']
            
            for i, feature in enumerate(features):
                coeffs_for_feature = []
                for method_name in successful_methods:
                    method_result = coefficient_results[method_name]
                    if 'coefficients' in method_result:
                        coeff = method_result['coefficients'][i + 1]  # Skip intercept
                        coeffs_for_feature.append(coeff)
                
                if coeffs_for_feature:
                    mean_coeff = np.mean(coeffs_for_feature)
                    std_coeff = np.std(coeffs_for_feature)
                    cv = std_coeff / abs(mean_coeff) if abs(mean_coeff) > 1e-6 else float('inf')
                    
                    consensus_coeffs[feature] = {
                        'mean': mean_coeff,
                        'std': std_coeff,
                        'coefficient_of_variation': cv,
                        'individual_values': coeffs_for_feature,
                        'is_reliable': cv < 0.1  # 10% 이내 변동이면 신뢰할만함
                    }
                    
                    reliability_analysis[feature] = {
                        'reliability_score': max(0, 100 - cv * 100),  # 변동계수가 낮을수록 높은 점수
                        'agreement_level': 'High' if cv < 0.05 else 'Medium' if cv < 0.15 else 'Low'
                    }
        
        multiple_results['consensus_coefficients'] = consensus_coeffs
        multiple_results['reliability_analysis'] = reliability_analysis
        
        # 전체 신뢰도 점수
        if reliability_analysis:
            overall_reliability = np.mean([score['reliability_score'] for score in reliability_analysis.values()])
            multiple_results['overall_reliability_score'] = overall_reliability
        else:
            multiple_results['overall_reliability_score'] = 0
            
        logger.info(f"📈 Overall coefficient reliability score: {multiple_results.get('overall_reliability_score', 0):.1f}/100")
    
    # 방법간 상관관계 분석
    if len(successful_methods) > 1:
        logger.info("🔗 Analyzing correlations between methods...")
        correlation_matrix = {}
        
        for i, method1 in enumerate(successful_methods):
            correlation_matrix[method1] = {}
            for j, method2 in enumerate(successful_methods):
                if i <= j:  # 상삼각행렬만 계산
                    coeffs1 = coefficient_results[method1]['coefficients'][1:]  # Skip intercept
                    coeffs2 = coefficient_results[method2]['coefficients'][1:]  # Skip intercept
                    
                    correlation = np.corrcoef(coeffs1, coeffs2)[0, 1]
                    correlation_matrix[method1][method2] = correlation
                    
                    if i != j:
                        correlation_matrix.setdefault(method2, {})[method1] = correlation
        
        multiple_results['method_correlations'] = correlation_matrix
    
    logger.info(f"🏆 Best performing method: {best_method} (Score: {best_score}/100)")
    
    return multiple_results