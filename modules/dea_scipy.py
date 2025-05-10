"""
DEA implementation using SciPy's linear programming solver with parallel processing.
"""

import numpy as np
from scipy.optimize import linprog
import pandas as pd
import concurrent.futures
import os
import warnings
import logging

# Setup logging
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Helper function for efficiency calculation, defined at module level for pickling
def compute_efficiency(dmu_idx, inputs_matrix, outputs_matrix, n_dmu, rts, weight_constraints, non_discretionary):
    """
    Compute efficiency score for a single DMU using SciPy's linear programming.
    
    Args:
        dmu_idx: Index of the DMU to process
        inputs_matrix: Matrix of input values
        outputs_matrix: Matrix of output values
        n_dmu: Number of DMUs
        rts: Returns to scale ('crs' or 'vrs')
        weight_constraints: Dict of weight restrictions (currently ignored)
        non_discretionary: List of non-discretionary variables (currently ignored)
        
    Returns:
        Tuple of (efficiency_score, success_flag)
    """
    try:
        # Use precomputed matrices, focus on efficient constraint building
        c = np.zeros(n_dmu + 1)
        c[0] = 1.0
        A_ub = []
        b_ub = []
        # Input constraint row
        A_ub_row_input = np.zeros(n_dmu + 1)
        A_ub_row_input[0] = -inputs_matrix[dmu_idx]
        A_ub_row_input[1:] = inputs_matrix
        A_ub.append(A_ub_row_input)
        b_ub.append(0.0)
        # Output constraints
        for k in range(outputs_matrix.shape[1]):
            A_ub_row_output = np.zeros(n_dmu + 1)
            A_ub_row_output[1:] = -outputs_matrix[:, k]
            A_ub.append(A_ub_row_output)
            b_ub.append(-outputs_matrix[dmu_idx, k])
        # Weight and non-discretionary constraints (currently ignored)
        if weight_constraints:
            warnings.warn("Weight restrictions not fully supported; ignoring.")
        if non_discretionary:
            warnings.warn("Non-discretionary variables not implemented; ignoring.")
        # Equality constraints for VRS
        if rts.lower() == 'vrs':
            A_eq = np.zeros((1, n_dmu + 1))
            A_eq[0, 1:] = 1.0
            b_eq = np.array([1.0])
        else:  # CRS
            A_eq = None
            b_eq = None
        
        # Solve the linear programming problem
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method='highs')
        if result.success:
            return 1.0 / result.x[0], True
        else:
            logger.warning(f"LP solver failed for DMU {dmu_idx}: {result.message}")
            return 1.0, False
    except Exception as e:
        logger.error(f"Error in compute_efficiency for DMU {dmu_idx}: {e}")
        return 1.0, False

def run_scipy_dea(
    df: pd.DataFrame,
    feature_set: str,
    target_variable: str,
    rts: str = 'crs',
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
        
        # Check for NaN or inf values in data
        if df_sample[target_variable].isnull().any() or np.any(np.isnan(df_sample[feature_set])) or np.any(np.isinf(df_sample[feature_set])):
            logger.warning("NaN or infinite values detected in input data; replacing with 0 for robustness.")
            df_sample[target_variable].fillna(0, inplace=True)
            df_sample[feature_set] = df_sample[feature_set].fillna(0).replace([np.inf, -np.inf], 0)
        
        n_dmu = len(df_sample)
        inputs = df_sample[target_variable].values  # Input data, 1D array
        outputs = df_sample[feature_set].values  # Output data, 2D array (n_dmu x n_output)
        
        # Precompute shared arrays outside the per-DMU loop
        inputs_matrix = inputs.copy()
        outputs_matrix = outputs.copy()
        
        # Parallel execution using ProcessPoolExecutor
        max_workers = min(64, (os.cpu_count() or 4))  # Cap at 64 to prevent overload, use system CPU count
        efficiencies = []
        lp_results = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for all DMUs
            future_to_index = {executor.submit(compute_efficiency, i, inputs_matrix, outputs_matrix, n_dmu, rts, weight_constraints, non_discretionary): i for i in range(n_dmu)}
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    efficiency, success = future.result()
                    efficiencies.append(efficiency)
                    lp_results.append(success)
                    logger.info(f"Processed DMU {index+1}/{n_dmu} with efficiency {efficiency:.4f}")
                except Exception as e:
                    logger.error(f"Error processing DMU {index+1}: {e}")
                    efficiencies.append(1.0)  # Default to efficient on error
                    lp_results.append(False)
        
        # Add results to DataFrame
        df_result = df_sample.copy()
        df_result['dea_efficiency'] = efficiencies
        df_result['dea_defaulted'] = [eff == 1.0 and not success for eff, success in zip(efficiencies, lp_results)]  # Flag for defaulted efficiencies
        df_result['dea_score'] = 1.0 / df_result['dea_efficiency']
        df_result['dea_rank'] = df_result['dea_score'].rank(ascending=False, method='min')
        
        if sample_size is not None:
            logger.warning("Results based on sampled data; may not represent full dataset.")
        
        return df_result
    
    except Exception as e:
        logger.error(f"Error in SciPy DEA implementation: {e}")
        raise
