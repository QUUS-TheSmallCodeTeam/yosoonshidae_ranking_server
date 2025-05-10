"""
DEA analysis runner.
"""

import pandas as pd
from .dea_scipy import run_scipy_dea

def run_dea_analysis(df: pd.DataFrame, feature_set: str, target_variable: str,
                     rts: str = 'crs') -> pd.DataFrame:
    """
    Run DEA analysis using SciPy implementation.
    
    Args:
        df: DataFrame containing the data
        feature_set: List of features to use
        target_variable: The target variable (input)
        rts: Returns to scale assumption ('crs' or 'vrs')
        
    Returns:
        DataFrame with DEA results
    """
    return run_scipy_dea(df, feature_set, target_variable, rts)
