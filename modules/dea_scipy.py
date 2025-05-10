"""
DEA implementation using SciPy's linear programming solver with parallel processing.
"""

import numpy as np
import pandas as pd
import logging
import warnings
import os
import concurrent.futures
from multiprocessing import Pool, cpu_count
from scipy.optimize import linprog
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Helper function for efficiency calculation, defined at module level for pickling
def compute_efficiency(dmu_idx, inputs_matrix, outputs_matrix, n_dmu, rts, weight_constraints, non_discretionary, output_cols=None):
    """
    Compute efficiency score for a single DMU using SciPy's linear programming.
    
    Args:
        dmu_idx: Index of the DMU to process
        inputs_matrix: Matrix of input values
        outputs_matrix: Matrix of output values
        n_dmu: Number of DMUs
        rts: Returns to scale ('crs' or 'vrs')
        weight_constraints: Dict of weight restrictions for each feature
        non_discretionary: List of non-discretionary variables (currently ignored)
        output_cols: List of output column names (needed for weight constraints)
        
    Returns:
        Tuple of (efficiency_score, success_flag)
    """
    try:
        # Set up the linear program
        c = np.zeros(n_dmu + 1 + outputs_matrix.shape[1])  # Add variables for weights
        c[0] = 1.0  # Maximize efficiency
        
        # Constraints
        A_ub = []  # Inequality constraints (A_ub * x <= b_ub)
        b_ub = []  # Right-hand side of inequality constraints
        
        # Input constraint
        A_ub_row_input = np.zeros(n_dmu + 1 + outputs_matrix.shape[1])
        A_ub_row_input[0] = -inputs_matrix[dmu_idx, 0] if inputs_matrix.ndim > 1 else -inputs_matrix[dmu_idx]
        A_ub_row_input[1:n_dmu+1] = inputs_matrix.flatten() if inputs_matrix.ndim == 1 else inputs_matrix[:, 0]
        A_ub.append(A_ub_row_input)
        b_ub.append(0.0)
        
        # Output constraints with explicit weights
        for k in range(outputs_matrix.shape[1]):
            A_ub_row_output = np.zeros(n_dmu + 1 + outputs_matrix.shape[1])
            A_ub_row_output[1:n_dmu+1] = -outputs_matrix[:, k]
            A_ub_row_output[0] = outputs_matrix[dmu_idx, k]
            # Set weight variable coefficient
            A_ub_row_output[n_dmu+1+k] = 1.0
            A_ub.append(A_ub_row_output)
            b_ub.append(0.0)
        
        # Weight constraints if provided
        if weight_constraints and output_cols:
            # Normalize weights to sum to 1
            A_eq_weights = np.zeros((1, n_dmu + 1 + outputs_matrix.shape[1]))
            A_eq_weights[0, n_dmu+1:] = 1.0
            b_eq_weights = np.ones(1)
            
            # Apply minimum and maximum weight constraints
            for i, col in enumerate(output_cols):
                if col in weight_constraints:
                    # Minimum weight constraint
                    if 'min' in weight_constraints[col]:
                        min_weight = weight_constraints[col]['min']
                        A_ub_min = np.zeros(n_dmu + 1 + outputs_matrix.shape[1])
                        A_ub_min[n_dmu+1+i] = -1.0  # -weight <= -min_weight
                        A_ub.append(A_ub_min)
                        b_ub.append(-min_weight)
                    
                    # Maximum weight constraint
                    if 'max' in weight_constraints[col]:
                        max_weight = weight_constraints[col]['max']
                        A_ub_max = np.zeros(n_dmu + 1 + outputs_matrix.shape[1])
                        A_ub_max[n_dmu+1+i] = 1.0  # weight <= max_weight
                        A_ub.append(A_ub_max)
                        b_ub.append(max_weight)
        
        # VRS constraint if applicable
        if rts == 'vrs':
            A_eq = np.zeros((1, n_dmu + 1 + outputs_matrix.shape[1]))
            A_eq[0, 1:n_dmu+1] = 1.0
            b_eq = np.ones(1)
        else:
            A_eq = None
            b_eq = None
        
        # Bounds for variables
        bounds = [(0, None) for _ in range(n_dmu + 1 + outputs_matrix.shape[1])]  # All variables >= 0
        
        # Solve the linear program
        result = linprog(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method='highs'  # Use the HiGHS solver for better performance
        )
        
        if result.success:
            return result.x[0], True  # Return the efficiency score and success flag
        else:
            return 1.0, False  # Default to 1.0 if optimization fails
    except Exception as e:
        logger.warning(f"Error computing efficiency for DMU {dmu_idx}: {e}")
        return 1.0, False  # Default to 1.0 on error

def run_scipy_dea(
    df: pd.DataFrame,
    feature_set: str,
    target_variable: str,
    rts: str = 'vrs',  # Changed default to VRS for better discrimination
    weight_constraints: dict = None,
    non_discretionary: list = None,
    sample_size: int = None
) -> pd.DataFrame:
    """
    Run DEA using SciPy's linear programming solver.
    
    Args:
        df: DataFrame containing the data
        feature_set: Set of features to use ('basic' or 'extended')
        target_variable: The target variable (input)
        rts: Returns to scale assumption ('crs' or 'vrs')
        
    Returns:
        DataFrame with DEA results
        
    Raises:
        ValueError: If invalid parameters are provided
        RuntimeError: If DEA calculation fails
    """
    try:
        logger.info("Starting DEA calculation with SciPy")
        
        # Handle sampling if specified
        if sample_size is not None and sample_size < len(df):
            df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            df_sample = df.copy()
        
        # Define feature sets based on input
        if isinstance(feature_set, str):
            # Predefined feature sets - reduced dimensionality to improve discrimination
            feature_sets = {
                'basic': ['data', 'voice', 'message'],  # Core features only
                'extended': ['data', 'voice', 'message', 'data_speed'],  # Reduced from original
                'full': [col for col in df.columns if col != target_variable and col != 'plan_id' and col != 'provider']
            }
            
            if feature_set in feature_sets:
                output_cols = feature_sets[feature_set]
            else:
                logger.warning(f"Unknown feature set '{feature_set}', using 'basic'")
                output_cols = feature_sets['basic']
        elif isinstance(feature_set, list):
            # Custom list of features
            output_cols = feature_set
        else:
            logger.warning(f"Invalid feature_set type {type(feature_set)}, using 'basic'")
            output_cols = ['data', 'voice', 'message']
        
        # Add default weight constraints if none provided
        # This prevents plans from assigning zero weights to features they're weak in
        if weight_constraints is None:
            weight_constraints = {}
            for feature in output_cols:
                if feature in df.columns:
                    # Set minimum weights to at least 10% of maximum possible weight
                    # This forces plans to consider all features
                    weight_constraints[feature] = {'min': 0.1, 'max': 1.0}
        
        logger.info(f"Using feature columns: {output_cols}")
        
        # Check for NaN or inf values in data
        if df_sample[target_variable].isnull().any():
            logger.warning("NaN values detected in input variable; replacing with 0 for robustness.")
            df_sample[target_variable].fillna(0, inplace=True)
            
        # Check and clean feature columns
        for col in output_cols:
            if col not in df_sample.columns:
                raise ValueError(f"Column '{col}' not found in data")
            if df_sample[col].isnull().any() or np.any(np.isnan(df_sample[col])) or np.any(np.isinf(df_sample[col])):
                logger.warning(f"NaN or infinite values detected in column {col}; replacing with 0.")
                df_sample[col] = df_sample[col].fillna(0).replace([np.inf, -np.inf], 0)
        
        n_dmu = len(df_sample)
        inputs = df_sample[target_variable].values.reshape(-1, 1)  # Input data, 2D array (n_dmu x 1)
        outputs = df_sample[feature_columns].values  # Output data, 2D array (n_dmu x n_output)
        
        # Precompute shared arrays outside the per-DMU loop
        inputs_matrix = inputs.copy()
        outputs_matrix = outputs.copy()
        
        # Log weight constraints being applied
        if weight_constraints:
            logger.info(f"Applying weight constraints: {weight_constraints}")
        
        # Use parallel processing if more than 10 DMUs
        if n_dmu > 10 and cpu_count() > 1:
            with Pool(processes=min(cpu_count(), 4)) as pool:
                results = pool.starmap(
                    compute_efficiency,
                    [(i, inputs_matrix, outputs_matrix, n_dmu, rts, weight_constraints, non_discretionary, output_cols) for i in range(n_dmu)]
                )
                efficiencies = [res[0] for res in results]
                lp_results = [res[1] for res in results]
        else:
            # Sequential processing for small datasets
            for i in range(n_dmu):
                eff, success = compute_efficiency(i, inputs_matrix, outputs_matrix, n_dmu, rts, weight_constraints, non_discretionary, output_cols)
                efficiencies.append(eff)
                lp_results.append(success)
        
        # Add results to DataFrame
        df_result = df_sample.copy()
        df_result['dea_efficiency'] = efficiencies
        df_result['dea_defaulted'] = [eff == 1.0 and not success for eff, success in zip(efficiencies, lp_results)]  # Flag for defaulted efficiencies
        df_result['dea_score'] = 1.0 / df_result['dea_efficiency']
        
        # Calculate super-efficiency for efficient DMUs to break ties
        efficient_dmus = df_result[df_result['dea_efficiency'] == 1.0].index.tolist()
        logger.info(f"Found {len(efficient_dmus)} efficient DMUs. Calculating super-efficiency to break ties.")
        
        if len(efficient_dmus) > 1:
            # For each efficient DMU, recalculate efficiency excluding it from the reference set
            super_efficiencies = {}
            
            for dmu_idx in efficient_dmus:
                # Create a subset without this DMU
                subset_indices = [i for i in range(n_dmu) if i != dmu_idx]
                subset_inputs = inputs_matrix[subset_indices]
                subset_outputs = outputs_matrix[subset_indices]
                
                try:
                    # Calculate super-efficiency
                    A_ub = []
                    b_ub = []
                    
                    # Input constraint
                    A_ub_row_input = np.zeros(len(subset_indices) + 1)
                    A_ub_row_input[0] = -inputs_matrix[dmu_idx, 0] if inputs_matrix.ndim > 1 else -inputs_matrix[dmu_idx]
                    A_ub_row_input[1:] = subset_inputs.flatten() if subset_inputs.ndim == 1 else subset_inputs[:, 0]
                    A_ub.append(A_ub_row_input)
                    b_ub.append(0.0)
                    
                    # Output constraints
                    for k in range(outputs_matrix.shape[1]):
                        A_ub_row_output = np.zeros(len(subset_indices) + 1)
                        A_ub_row_output[1:] = -subset_outputs[:, k]
                        A_ub_row_output[0] = outputs_matrix[dmu_idx, k]
                        A_ub.append(A_ub_row_output)
                        b_ub.append(0.0)
                    
                    # VRS constraint
                    if rts == 'vrs':
                        A_eq = np.zeros((1, len(subset_indices) + 1))
                        A_eq[0, 1:] = 1.0
                        b_eq = np.ones(1)
                    else:
                        A_eq = None
                        b_eq = None
                    
                    # Solve LP
                    c = np.zeros(len(subset_indices) + 1)
                    c[0] = 1.0
                    bounds = [(0, None) for _ in range(len(subset_indices) + 1)]
                    
                    result = linprog(
                        c=c,
                        A_ub=A_ub,
                        b_ub=b_ub,
                        A_eq=A_eq,
                        b_eq=b_eq,
                        bounds=bounds,
                        method='highs'
                    )
                    
                    if result.success:
                        super_efficiencies[dmu_idx] = result.x[0]
                    else:
                        # If infeasible, this DMU is super-efficient (can't be represented by others)
                        super_efficiencies[dmu_idx] = 2.0  # Assign a high value
                except Exception as e:
                    logger.warning(f"Error calculating super-efficiency for DMU {dmu_idx}: {e}")
                    super_efficiencies[dmu_idx] = 1.0  # Default
            
            # Apply super-efficiencies to break ties
            for dmu_idx, super_eff in super_efficiencies.items():
                df_result.loc[dmu_idx, 'dea_super_efficiency'] = super_eff
                # Update the score with super-efficiency for efficient DMUs
                df_result.loc[dmu_idx, 'dea_score'] = super_eff
        
        # Calculate final rank
        df_result['dea_rank'] = df_result['dea_score'].rank(ascending=False, method='min')
        
        if sample_size is not None:
            logger.warning("Results based on sampled data; may not represent full dataset.")
        
        return df_result
    
    except Exception as e:
        logger.error(f"Error in SciPy DEA implementation: {e}")
        raise
