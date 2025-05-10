"""
Data Envelopment Analysis (DEA) implementation for mobile plan ranking.
Uses SciPy's linear programming solver to calculate efficiency scores.
"""

import numpy as np
import pandas as pd
from modules.dea.dea_scipy import run_scipy_dea
from modules.dea.dea_run import run_dea_analysis
from modules.models import get_basic_feature_list

logger = logging.getLogger(__name__)

def calculate_rankings_with_dea(
    df: pd.DataFrame,
    feature_set: str = 'basic',
    target_variable: str = 'fee',
    rts: str = 'crs'
) -> pd.DataFrame:
    """
    Calculate rankings using DEA method.
    
    Args:
        df: DataFrame containing plan data
        feature_set: Set of features to use ('basic' or 'extended')
        target_variable: The target variable (input)
        rts: Returns to scale assumption ('crs' or 'vrs')
        
    Returns:
        DataFrame with DEA rankings
        
    Raises:
        ValueError: If required columns are missing or invalid parameters
        RuntimeError: If DEA calculation fails
    """
    try:
        logger.info("Starting DEA calculation")
        
        # Validate required columns
        required_columns = [target_variable] + get_basic_feature_list()
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        # Validate parameters
        if feature_set not in ['basic', 'extended']:
            logger.error(f"Invalid feature set: {feature_set}")
            raise ValueError(f"Invalid feature set: {feature_set}")
        if rts not in ['crs', 'vrs']:
            logger.error(f"Invalid RTS: {rts}")
            raise ValueError(f"Invalid RTS: {rts}")
            
        # Run DEA analysis
        logger.info(f"Running DEA with feature_set={feature_set}, target_variable={target_variable}, rts={rts}")
        result_df = run_dea_analysis(
            df,
            feature_set=feature_set,
            target_variable=target_variable,
            rts=rts
        )
        
        if result_df is None or result_df.empty:
            logger.error("DEA calculation returned empty results")
            raise RuntimeError("DEA calculation returned empty results")
            
        # Calculate rankings
        result_df['dea_rank'] = result_df['efficiency_score'].rank(ascending=False)
        logger.info("DEA calculation completed successfully")
        
        return result_df
    
    except Exception as e:
        logger.error(f"Error in DEA calculation: {str(e)}", exc_info=True)
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
