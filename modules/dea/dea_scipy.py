"""
DEA implementation using SciPy's linear programming solver.
"""

import numpy as np
from scipy.optimize import linprog
import pandas as pd

def run_scipy_dea(
    df: pd.DataFrame,
    feature_set: str,
    target_variable: str,
    rts: str = 'crs'
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
        
        # Validate parameters
        if feature_set not in ['basic', 'extended']:
            logger.error(f"Invalid feature set: {feature_set}")
            raise ValueError(f"Invalid feature set: {feature_set}")
        if rts not in ['crs', 'vrs']:
            logger.error(f"Invalid RTS: {rts}")
            raise ValueError(f"Invalid RTS: {rts}")
            
        # Get input and output variables
        logger.info("Preparing input and output variables")
        inputs = df[target_variable].values.reshape(-1, 1)
        outputs = df[feature_set].values
        
        # Validate data
        if inputs is None or outputs is None:
            logger.error("Missing input or output data")
            raise ValueError("Missing input or output data")
            
        if len(inputs) != len(outputs):
            logger.error("Input and output data must have same number of observations")
            raise ValueError("Input and output data must have same number of observations")
            
        # Number of DMUs (Decision Making Units)
        n_dmus = len(df)
        
        # Create efficiency scores array
        efficiency_scores = np.zeros(n_dmus)
        
        # For each DMU, solve the DEA problem
        logger.info(f"Solving DEA for {n_dmus} DMUs")
        for i in range(n_dmus):
            try:
                # Create the objective function (minimize sum of lambdas)
                c = np.ones(n_dmus)
                
                # Create the inequality constraints
                A_ub = []
                b_ub = []
                
                # Input constraint: sum(lambda_j * x_j) <= x_i
                A_ub.append(inputs.T)
                b_ub.append(inputs[i])
                
                # Output constraints: sum(lambda_j * y_j) >= y_i
                for output in outputs.T:
                    A_ub.append(-output)
                    b_ub.append(-outputs[i, :])
                
                # Add returns to scale constraint if VRS
                if rts == 'vrs':
                    A_ub.append(np.ones(n_dmus))
                    b_ub.append(1)
                
                # Convert lists to arrays
                A_ub = np.vstack(A_ub)
                b_ub = np.array(b_ub).flatten()
                
                # Solve the linear programming problem
                res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
                
                if res.success:
                    efficiency_scores[i] = 1 / res.fun
                else:
                    logger.warning(f"DEA failed for DMU {i}: {res.message}")
                    efficiency_scores[i] = 0
                    
            except Exception as e:
                logger.error(f"Error solving DEA for DMU {i}: {str(e)}")
                efficiency_scores[i] = 0
                
        # Create results DataFrame
        results = df.copy()
        results['efficiency_score'] = efficiency_scores
        
        logger.info("DEA calculation completed successfully")
        return results
        
    except Exception as e:
        logger.error(f"Error in DEA calculation: {str(e)}", exc_info=True)
        raise
