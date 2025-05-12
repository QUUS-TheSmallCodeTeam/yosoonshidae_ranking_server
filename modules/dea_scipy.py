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
            # Predefined feature sets with comprehensive mobile plan features
            feature_sets = {
                'basic': [
                    # Core data features
                    'basic_data_clean',      # Monthly data allowance in GB
                    'basic_data_unlimited',  # Flag for unlimited monthly data
                    'daily_data_clean',      # Daily data allowance in GB
                    'daily_data_unlimited',  # Flag for unlimited daily data
                    
                    # Voice and messaging features
                    'voice_clean',           # Voice minutes
                    'voice_unlimited',       # Flag for unlimited voice
                    'message_clean',         # SMS count
                    'message_unlimited',     # Flag for unlimited SMS
                    'additional_call',       # Extra voice minutes
                    
                    # Network and tethering features
                    'is_5g',                 # Flag for 5G network support
                    'tethering_gb',          # Tethering allowance in GB
                    
                    # Throttling and speed features
                    'has_throttled_data',    # Flag for throttling after allowance
                    'has_unlimited_speed',   # Flag for full-speed after allowance
                    'speed_when_exhausted'   # Speed in Mbps when throttled
                ],
                
                'extended': [
                    # All basic features
                    'basic_data_clean', 'basic_data_unlimited', 'daily_data_clean', 'daily_data_unlimited',
                    'voice_clean', 'voice_unlimited', 'message_clean', 'message_unlimited', 'additional_call',
                    'is_5g', 'tethering_gb', 'has_throttled_data', 'has_unlimited_speed', 'speed_when_exhausted',
                    
                    # Additional features
                    'throttle_speed_normalized',
                    'data_sharing',
                    'roaming_support',
                    'micro_payment',
                    'is_esim'
                    # Note: unlimited_type_numeric removed as it's redundant with the individual flag features
                ],
                
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
            output_cols = ['basic_data_clean', 'voice_clean', 'message_clean']
        
        # No weight constraints
        weight_constraints = None
        
        logger.info(f"Using feature columns: {output_cols}")
        
        # Check for NaN or inf values in data
        if df_sample[target_variable].isnull().any():
            logger.warning("NaN values detected in input variable; replacing with 0 for robustness.")
            df_sample[target_variable].fillna(0, inplace=True)
            
        # Pre-process unlimited specs
        spec_pairs = [
            ("basic_data_clean", "basic_data_unlimited"),
            ("daily_data_clean", "daily_data_unlimited"),
            ("voice_clean", "voice_unlimited"),
            ("message_clean", "message_unlimited")
        ]
        
        logger.info("Pre-processing unlimited specs:")
        for num_col, flag_col in spec_pairs:
            if num_col in df_sample.columns and flag_col in df_sample.columns:
                cap = df_sample[num_col].replace(0, np.nan).max()
                if not pd.isna(cap) and cap > 0:
                    df_sample.loc[df_sample[flag_col] == 1, num_col] = cap  # set to max value
                    logger.info(f"  - {flag_col} → set {num_col} to max value: {cap:.2f}")
            elif flag_col in df_sample.columns:  # fallback: create numeric col
                df_sample[num_col] = 0
                cap = 1
                df_sample.loc[df_sample[flag_col] == 1, num_col] = cap
                logger.info(f"  - {flag_col} → created {num_col} with values 0/1")
        
        # Remove unlimited_type_numeric if present as it's redundant with individual flag features
        if 'unlimited_type_numeric' in output_cols:
            logger.info("Removing unlimited_type_numeric as it's redundant with individual flag features")
            output_cols.remove('unlimited_type_numeric')
        
        # Identify and properly handle categorical features
        categorical_features = [
            'basic_data_unlimited', 'daily_data_unlimited', 'voice_unlimited', 'message_unlimited',
            'is_5g', 'has_throttled_data', 'has_unlimited_speed', 'data_sharing',
            'roaming_support', 'micro_payment', 'is_esim'
        ]
        
        for col in categorical_features:
            if col in output_cols and col in df_sample.columns:
                # Ensure categorical features are properly represented as 0/1 values
                df_sample[col] = df_sample[col].apply(lambda x: 1.0 if x else 0.0)
                logger.info(f"  - Normalized categorical feature: {col}")
        
        # Check and clean feature columns
        # Define which features are binary flags (already normalized)
        binary_features = [
            'basic_data_unlimited', 'daily_data_unlimited', 'voice_unlimited', 'message_unlimited',
            'is_5g', 'has_throttled_data', 'has_unlimited_speed', 'data_sharing',
            'roaming_support', 'micro_payment', 'is_esim'
        ]
        
        # Don't use normalized versions of features
        normalized_feature_map = {}
        
        # Process each feature
        for col in list(output_cols):  # Use list() to allow modifying output_cols during iteration
            # Check if column exists
            if col not in df_sample.columns:
                raise ValueError(f"Column '{col}' not found in data")
                
            # Replace with normalized version if available
            if col in normalized_feature_map and normalized_feature_map[col] in df_sample.columns:
                normalized_col = normalized_feature_map[col]
                logger.info(f"Replacing {col} with normalized version {normalized_col}")
                # Replace in output_cols list
                output_cols[output_cols.index(col)] = normalized_col
                continue
                
            # Clean NaN and inf values
            if df_sample[col].isnull().any() or np.any(np.isnan(df_sample[col])) or np.any(np.isinf(df_sample[col])):
                logger.warning(f"NaN or infinite values detected in column {col}; replacing with 0.")
                df_sample[col] = df_sample[col].fillna(0).replace([np.inf, -np.inf], 0)
                
            # Log the feature range for reference
            if col not in binary_features and df_sample[col].nunique() > 2:
                logger.info(f"Feature {col} range: min={df_sample[col].min():.2f}, max={df_sample[col].max():.2f}")
        
        n_dmu = len(df_sample)
        
        # Use the original input values (not normalized)
        inputs = df_sample[target_variable].values.reshape(-1, 1)  # Input data, 2D array (n_dmu x 1)
            
        outputs = df_sample[output_cols].values  # Output data, 2D array (n_dmu x n_output)
        
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
        
        # Calculate super-efficiency for efficient DMUs to break ties
        # Standard approach in DEA literature
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
            
            # Apply super-efficiencies to break ties using standard DEA methodology
            for dmu_idx, super_eff in super_efficiencies.items():
                # Store the super-efficiency value
                df_result.loc[dmu_idx, 'dea_super_efficiency'] = super_eff
                
                # Update the score with super-efficiency for efficient DMUs
                df_result.loc[dmu_idx, 'dea_score'] = super_eff
        
        # Calculate final rank based on DEA score (standard approach)
        # Use method='min' to ensure ties get the same rank and the next rank is skipped
        # This ensures the best plan always gets rank 1
        df_result['dea_rank'] = df_result['dea_score'].rank(ascending=False, method='min')
        
        # Log the top plans to verify ranking
        top_plans = df_result.sort_values('dea_score', ascending=False).head(10)
        logger.info(f"Top 10 plans by DEA score:\n{top_plans[['dea_score', 'dea_rank']].to_string()}")
        
        # Verify we have at least one plan with rank 1
        if 1.0 not in df_result['dea_rank'].values:
            logger.warning("No plan with rank 1 found! This indicates a ranking issue.")
            # Force at least one plan to have rank 1 if there's an issue
            top_idx = df_result['dea_score'].idxmax()
            df_result.loc[top_idx, 'dea_rank'] = 1.0
            
        # Create a sequential rank column that doesn't skip numbers
        # This ensures we have plans with every rank from 1 to N without skipping
        df_result['dea_rank_sequential'] = df_result['dea_score'].rank(ascending=False, method='dense')
        
        # Create a sequential rank for display purposes that doesn't skip numbers
        df_result['dea_rank_display'] = df_result['dea_rank_sequential']
        
        # Replace any potential inf, -inf, or NaN values with appropriate finite values
        # This ensures JSON serialization will work
        df_result.replace([np.inf, -np.inf], np.finfo(np.float64).max, inplace=True)
        df_result.replace(np.nan, 0, inplace=True)
        
        if sample_size is not None:
            logger.warning("Results based on sampled data; may not represent full dataset.")
        
        return df_result
    
    except Exception as e:
        logger.error(f"Error in SciPy DEA implementation: {e}")
        raise
