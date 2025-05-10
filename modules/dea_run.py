"""
DEA analysis runner.
"""

import pandas as pd
from .dea_scipy import run_scipy_dea

def run_dea_analysis(df: pd.DataFrame, feature_set: str, target_variable: str,
                     rts: str = 'crs', weight_constraints: dict = None,
                     non_discretionary: list = None, sample_size: int = None) -> pd.DataFrame:
    """
    Run DEA analysis using SciPy implementation.
    
    Args:
        df: DataFrame containing the data
        feature_set: List of features to use
        target_variable: The target variable (input)
        rts: Returns to scale assumption ('crs' or 'vrs')
        weight_constraints: Optional dictionary of weight constraints
        non_discretionary: Optional list of non-discretionary variables
        sample_size: Optional sample size for large datasets
        
    Returns:
        DataFrame with DEA results
    """
    return run_scipy_dea(
        df=df, 
        feature_set=feature_set, 
        target_variable=target_variable, 
        rts=rts,
        weight_constraints=weight_constraints,
        non_discretionary=non_discretionary,
        sample_size=sample_size
    )
