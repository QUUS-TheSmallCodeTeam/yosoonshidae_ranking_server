"""
Spearman Correlation Ranking Module

This module implements the Spearman correlation ranking algorithm for mobile plans.
It calculates worth estimates based on feature correlations with plan prices.
"""

import pandas as pd
import numpy as np
import logging
from scipy.stats import spearmanr

# Configure logging
logger = logging.getLogger(__name__)

def calculate_rankings_with_spearman(df, use_log_transform=True, rank_method='relative'):
    """
    Generate worth estimates and rankings using Spearman correlation method with hybrid normalization.
    
    Args:
        df: DataFrame with preprocessed plan data
        use_log_transform: Whether to apply log transformation to non-categorical features
        rank_method: Method to use for ranking ('relative', 'absolute', or 'net')
        
    Returns:
        DataFrame with added worth estimates and rankings
    """
    logger.info(f"Calculating rankings with Spearman hybrid method (rank method: {rank_method}, log transform: {use_log_transform})")
    
    # Check if we have fee (discounted price) in the data
    has_fee = 'fee' in df.columns
    if has_fee:
        logger.info("Found 'fee' column - will calculate rankings for both original_fee and fee")
    else:
        logger.info("No 'fee' column found - using only original_fee for rankings")
        # Create a copy of original_fee as fee to simplify subsequent code
        df = df.copy()
        df['fee'] = df['original_fee']
    
    # Define basic features to include
    features = [
        'is_5g',
        'basic_data_clean', 
        'basic_data_unlimited',
        'daily_data_clean', 
        'daily_data_unlimited',
        'voice_clean', 
        'voice_unlimited',
        'message_clean',
        'message_unlimited',
        'throttle_speed_normalized', 
        'tethering_gb', 
        'unlimited_type_numeric', 
        'additional_call'
    ]
    
    # Only keep features that are actually in the dataframe
    available_features = [f for f in features if f in df.columns]
    logger.info(f"Using {len(available_features)}/{len(features)} available basic features")
        
    # Apply log transformation to non-categorical features if requested
    if use_log_transform:
        # Features to transform - non-binary, non-normalized numeric features
        features_to_transform = [
            'basic_data_clean',
            'daily_data_clean',
            'voice_clean',
            'message_clean', 
            'tethering_gb',
            'additional_call'
        ]
        
        # Filter to only include available features
        features_to_transform = [f for f in features_to_transform if f in available_features]
        
        # Apply log transformation
        df_analysis = df.copy()
        for feature in features_to_transform:
            if feature in df.columns:
                df_analysis[feature] = np.log1p(df[feature])
        logger.info("Log transformation applied to non-categorical features")
    else:
        df_analysis = df.copy()
        logger.info("Using original feature values without log transformation")
    
    # Compute Spearman ρ for each feature vs. original_fee
    rhos = {}
    logger.info("Computing Spearman correlations")
    
    # First check which features are constant (have only one unique value)
    constant_features = []
    for feature in available_features:
        if df_analysis[feature].nunique() <= 1:
            logger.warning(f"Feature '{feature}' is constant with value {df_analysis[feature].iloc[0]} - skipping")
            constant_features.append(feature)
    
    # Remove constant features from available_features
    valid_features = [f for f in available_features if f not in constant_features]
    
    if not valid_features:
        logger.error("No valid features found after removing constant features!")
        return df  # Return original dataframe if no valid features
    
    # Calculate correlations for valid features
    for feature in valid_features:
        # Drop rows with NaN values in either feature or original_fee
        sub = df_analysis.dropna(subset=[feature, 'original_fee'])
        
        try:
            # Compute Spearman correlation
            rho, pval = spearmanr(sub[feature], sub['original_fee'])
            
            # Handle possible NaN from correlation
            if np.isnan(rho):
                logger.warning(f"Feature '{feature}' has NaN correlation - skipping")
                continue
                
            # Store the signed correlation value
            rhos[feature] = rho
            logger.info(f"Feature '{feature}': ρ = {rho:.4f} (p = {pval:.4f})")
        except Exception as e:
            logger.error(f"Error calculating correlation for {feature}: {e} - skipping")
    
    # Check if we have any valid correlations
    if not rhos:
        logger.error("No valid correlations found!")
        return df  # Return original dataframe if no valid correlations
    
    # Use absolute values for normalization
    total_abs_rho = sum(abs(rho) for rho in rhos.values())
    
    # Check if total_abs_rho is valid
    if np.isnan(total_abs_rho) or total_abs_rho == 0:
        logger.error("Sum of absolute correlations is invalid (NaN or zero)!")
        return df
    
    # Calculate absolute-value weights
    weights = {feat: abs(rhos[feat]) / total_abs_rho for feat in rhos.keys()}
    
    # Store correlation signs for later use
    rho_signs = {feat: np.sign(rhos[feat]) for feat in rhos.keys()}
    
    # Identify binary/dummy features vs continuous features
    binary_features = []
    continuous_features = []
    
    for feature in weights.keys():
        # Check if feature is binary or categorical (0/1 or stepped values)
        if feature in df_analysis.columns:
            unique_values = df_analysis[feature].dropna().unique()
            # If feature has only 0/1 values or has small number of unique integers
            if set(unique_values).issubset({0, 1}) or (
                len(unique_values) <= 5 and all(float(x).is_integer() for x in unique_values if not pd.isna(x))
            ):
                binary_features.append(feature)
            else:
                continuous_features.append(feature)
    
    logger.info(f"Identified {len(binary_features)} binary/categorical features and {len(continuous_features)} continuous features")
    
    # Normalize features using hybrid approach
    norm = {}  # a dict of arrays
    
    # 1. For binary/categorical features: keep as is
    for feature in binary_features:
        if feature in df_analysis.columns:
            # Create a mask for non-NaN values
            valid_mask = ~np.isnan(df_analysis[feature].values)
            
            if valid_mask.sum() == 0:
                logger.warning(f"Feature '{feature}' has all NaN values - skipping")
                continue
            
            # Use the feature values directly without normalization
            norm_values = np.full(len(df_analysis), np.nan)
            norm_values[valid_mask] = df_analysis[feature].loc[valid_mask]
            norm[feature] = norm_values
            logger.info(f"Binary feature '{feature}' kept as is without normalization")
    
    # 2. For continuous features: use z-score normalization
    for feature in continuous_features:
        if feature in df_analysis.columns:
            # Create a mask for non-NaN values
            valid_mask = ~np.isnan(df_analysis[feature].values)
            
            if valid_mask.sum() == 0:
                logger.warning(f"Feature '{feature}' has all NaN values - skipping")
                continue
            
            # Get valid values only
            valid_values = df_analysis[feature].loc[valid_mask]
            
            # Check if the feature has variation
            if valid_values.std() == 0:
                logger.info(f"Feature '{feature}' has zero standard deviation, setting normalized values to 0")
                norm_values = np.full(len(df_analysis), 0.0)
                norm_values[~valid_mask] = np.nan
                norm[feature] = norm_values
            else:
                # Apply z-score normalization: (x - mean) / std
                mean_value = valid_values.mean()
                std_value = valid_values.std()
                
                norm_values = np.full(len(df_analysis), np.nan)
                norm_values[valid_mask] = (valid_values - mean_value) / std_value
                norm[feature] = norm_values
                logger.info(f"Continuous feature '{feature}' normalized using z-score")
    
    # Apply sign(ρ) in scoring
    # Compute each plan's raw score S_j = Σ_i sign(ρ_i) * w_i * n_ij
    scores = []
    for j in range(len(df_analysis)):
        S_j = 0
        for feature in weights.keys():
            if feature in norm and not np.isnan(norm[feature][j]):
                # Apply sign of correlation in the scoring
                S_j += rho_signs[feature] * weights[feature] * norm[feature][j]
        scores.append(S_j)
    
    # Calculate delta_krw using standard deviation instead of price range
    # ΔP_j = S_j × σ(price)
    price_std = df["original_fee"].std()
    delta_krw = [S_j * price_std for S_j in scores]
    
    # Calculate worth estimate using the mean price as a baseline
    price_mean = df["original_fee"].mean()
    worth = [price_mean + delta for delta in delta_krw]
    
    # Attach estimates back to the DataFrame
    df_result = df.copy()
    df_result["predicted_price"] = worth  # Use predicted_price for consistency with model-based approach
    df_result["delta_krw"] = delta_krw
    
    # Calculate value metrics for original fee
    df_result["value_ratio_original"] = df_result["predicted_price"] / df_result["original_fee"]
    df_result["net_value_original"] = df_result["delta_krw"] - df_result["original_fee"]
    
    # Calculate value metrics for discounted fee
    df_result["value_ratio_fee"] = df_result["predicted_price"] / df_result["fee"]
    df_result["net_value_fee"] = df_result["delta_krw"] - df_result["fee"]
    
    # Calculate standard value ratio (for compatibility with existing code)
    df_result["value_ratio"] = df_result["value_ratio_fee"]
    
    # Calculate ALL ranking types regardless of specified rank_method
    # This ensures we have data for all possible views in the UI
    
    # Absolute value ranking (same for all fee types)
    df_result["rank_absolute"] = df_result["delta_krw"].rank(ascending=False)
    
    # Relative value rankings
    df_result["rank_relative_original"] = df_result["value_ratio_original"].rank(ascending=False)
    df_result["rank_relative_fee"] = df_result["value_ratio_fee"].rank(ascending=False)
    
    # Net value rankings
    df_result["rank_net_original"] = df_result["net_value_original"].rank(ascending=False)
    df_result["rank_net_fee"] = df_result["net_value_fee"].rank(ascending=False)
    
    # Set the primary rank column based on method and fee type
    if rank_method == 'absolute':
        df_result["rank"] = df_result["rank_absolute"]
        ranking_column = "delta_krw"
    elif rank_method == 'net':
        if 'fee_type' in locals() and fee_type == 'fee':
            df_result["rank"] = df_result["rank_net_fee"]
            ranking_column = "net_value_fee"
        else:
            df_result["rank"] = df_result["rank_net_original"]
            ranking_column = "net_value_original"
    else:  # relative (default)
        if 'fee_type' in locals() and fee_type == 'fee':
            df_result["rank"] = df_result["rank_relative_fee"]
            ranking_column = "value_ratio_fee"
        else:
            df_result["rank"] = df_result["rank_relative_original"]
            ranking_column = "value_ratio_original"
    
    # Apply proper ranking with ties for display
    from . import calculate_rankings_with_ties
    df_result = calculate_rankings_with_ties(df_result, value_column=ranking_column, ascending=(rank_method == 'absolute'))
    
    # Calculate feature-level contribution with sign
    for feature in weights.keys():
        if feature not in norm:
            logger.warning(f"Feature '{feature}' not in normalized features, skipping contribution calculation")
            continue
            
        contribution_col = f"contribution_{feature}"
        try:
            # Calculate contributions with sign
            contribution = np.zeros(len(df_result))
            contribution[:] = np.nan  # Start with all NaN
            
            # Create mask for valid values
            valid_mask = ~np.isnan(norm[feature])
            
            if valid_mask.sum() > 0:
                # Use standard deviation for contribution calculation
                contribution[valid_mask] = rho_signs[feature] * weights[feature] * norm[feature][valid_mask] * price_std
            
            df_result[contribution_col] = contribution
        except Exception as e:
            logger.error(f"Error calculating contribution for {feature}: {e}")
            df_result[contribution_col] = np.nan
    
    logger.info(f"Completed Spearman hybrid ranking for {len(df_result)} plans with all ranking types")
    
    # Store the feature weights and normalization information in the dataframe attributes
    df_result.attrs['used_features'] = list(weights.keys())
    df_result.attrs['feature_weights'] = weights
    df_result.attrs['feature_signs'] = rho_signs
    df_result.attrs['ranking_method'] = rank_method
    df_result.attrs['use_log_transform'] = use_log_transform
    df_result.attrs['binary_features'] = binary_features
    df_result.attrs['continuous_features'] = continuous_features
    df_result.attrs['normalization_method'] = 'hybrid_z_score'
    df_result.attrs['price_std'] = price_std
    df_result.attrs['price_mean'] = price_mean
    
    return df_result
