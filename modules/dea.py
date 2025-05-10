"""
Data Envelopment Analysis (DEA) implementation for mobile plan ranking.
Uses SciPy's linear programming solver to calculate efficiency scores.
"""

import numpy as np
import pandas as pd
from scripts.dea.dea_scipy import run_scipy_dea
from scripts.dea.dea_run import run_dea_analysis
from modules.models import get_basic_feature_list

logger = logging.getLogger(__name__)

def calculate_rankings_with_dea(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate rankings using DEA method.
    
    Args:
        df: DataFrame containing plan data
        
    Returns:
        DataFrame with DEA rankings
        
    Raises:
        ValueError: If required columns are missing
        RuntimeError: If DEA calculation fails
    """
    try:
        # Validate required columns
        required_columns = ['fee'] + get_basic_feature_list()
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Run DEA analysis
        result_df = run_dea_analysis(
            df,
            feature_set='basic',
            target_variable='fee',
            implementation='scipy',
            rts='crs'
        )
        
        # Calculate rankings
        result_df['dea_rank'] = result_df['efficiency_score'].rank(ascending=False)
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error in DEA calculation: {str(e)}")
        raise

def calculate_rankings_with_ties(df: pd.DataFrame, value_column: str, ascending: bool = False) -> pd.DataFrame:
    """
    Calculate rankings with proper handling of ties.
    Adds "공동" prefix to tied rankings.
    
    Args:
        df: DataFrame containing the values to rank
        value_column: Column name to rank by
        ascending: Whether to rank in ascending order
        
    Returns:
        DataFrame with rankings and tied rankings
    """
    df = df.sort_values(by=value_column, ascending=ascending)
    current_rank = 1
    
    for _, indices in df.groupby(value_column).groups.items():
        if len(indices) > 1:  # If there's a tie
            tied_rank = current_rank
            for idx in indices:
                df.at[idx, 'rank_with_ties'] = f"공동 {tied_rank}위"
            current_rank += len(indices)
        else:
            idx = indices[0]
            df.at[idx, 'rank_with_ties'] = f"{current_rank}위"
            current_rank += 1
    
    return df
