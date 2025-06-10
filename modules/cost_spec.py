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
        'tethering_gb', 'has_throttled_data',
        'has_unlimited_speed', 'speed_when_exhausted'
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
                if current_value > 0:
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
        method: Calculation method ('frontier' or 'linear_decomposition')
        feature_set: Name of the feature set to use
        fee_column: Column containing the fee to use
        **method_kwargs: Additional arguments passed to specific methods
        
    Returns:
        DataFrame with added CS ratio calculations
    """
    if method == 'frontier':
        # Use existing frontier-based method
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
        # Use safe, commonly available features for decomposition
        safe_features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb', 'is_5g']
        # Only use features that actually exist in the dataframe
        decomp_features = [f for f in safe_features if f in df.columns]
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
                'feature_costs': {feature: float(coef) for feature, coef in zip(decomp_features, coefficients[1:])}
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
    
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: 'frontier', 'linear_decomposition'")

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