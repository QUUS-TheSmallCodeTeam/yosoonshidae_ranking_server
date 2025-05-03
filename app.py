from fastapi import FastAPI, HTTPException, Request, Response, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import os
import json
import sys
import logging
import time
import uuid
import gc
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Union, List
from scipy.stats import spearmanr

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define Paths (relative to container root /app)
APP_DIR = Path(__file__).parent  # Should resolve to /app
DATA_DIR = APP_DIR / "data"
REPORT_DIR_BASE = Path("/tmp/reports")  # Use /tmp for reports in container

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules
from modules.data import load_data_from_json  # For loading test data if needed
from modules import prepare_features, ensure_directories, save_raw_data, save_processed_data
from modules.models import get_basic_feature_list  # Import directly to avoid circular imports

# Import the calculate_rankings function for Spearman method
from modules.ranking import calculate_rankings, generate_html_report, save_report

# Initialize FastAPI
app = FastAPI(title="Moyo Plan Ranking Model Server - Spearman Method")

# Global variables for storing data
df_with_rankings = None  # Global variable to store the latest rankings
latest_logical_test_results_cache = None  # For storing logical test results

# Data model for plan input 
class PlanData(BaseModel):
    id: int
    plan_name: str
    network: str
    mvno: str
    mno: str
    basic_data: Union[float, str]  # Accept both float and string
    daily_data: Optional[Union[float, str]] = None  # Optional and accept both types
    data_exhaustion: Optional[str] = None
    voice: int  # Integer as seen in test.json
    message: int  # Integer as seen in test.json
    additional_call: int  # Integer as seen in test.json
    data_sharing: bool
    roaming_support: bool
    micro_payment: bool
    is_esim: bool
    signup_minor: bool
    signup_foreigner: bool
    has_usim: Optional[bool] = None
    has_nfc_usim: Optional[bool] = None
    tethering_gb: Union[float, str]  # Accept both float and string
    tethering_status: str
    tethering_data_unit: Optional[str] = None  # Added field for tethering unit information
    esim_fee: Optional[int] = None
    esim_fee_status: Optional[str] = None
    usim_delivery_fee: Optional[int] = None
    usim_delivery_fee_status: Optional[str] = None
    nfc_usim_delivery_fee: Optional[int] = None
    nfc_usim_delivery_fee_status: Optional[str] = None
    fee: float
    original_fee: float
    discount_fee: float
    discount_period: Optional[int] = None
    post_discount_fee: float
    agreement: bool
    agreement_period: Optional[int] = None
    agreement_type: Optional[str] = None
    num_of_signup: int
    mvno_rating: Union[float, str]  # Accept both float and string
    monthly_review_score: Union[float, str]  # Accept both float and string
    discount_percentage: Union[float, str]  # Accept both float and string

class PlanInput(BaseModel):
    """A simplified model for plan input data based on the PlanData model."""
    id: int
    plan_name: str
    network: str
    mvno: str
    mno: str
    basic_data: Union[float, str]
    daily_data: Optional[Union[float, str]] = None
    data_exhaustion: Optional[str] = None
    voice: int
    message: int
    additional_call: int
    data_sharing: bool
    roaming_support: bool
    micro_payment: bool
    is_esim: bool
    signup_minor: bool
    signup_foreigner: bool
    has_usim: Optional[bool] = None
    has_nfc_usim: Optional[bool] = None
    tethering_gb: Union[float, str]
    tethering_status: str
    tethering_data_unit: Optional[str] = None  # Added field for tethering unit information
    fee: float
    original_fee: float
    discount_fee: float
    discount_period: Optional[int] = None
    post_discount_fee: float

# Spearman Method Implementation
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

# Function to calculate rankings with proper tie handling (공동 notation)
def calculate_rankings_with_ties(df, value_column='value_ratio', ascending=False):
    """
    Calculate rankings with proper handling of ties.
    For tied ranks, uses '공동 X위' (joint X rank) notation
    and ensures the next rank after ties is correctly incremented.
    All tied plans should receive the '공동' label.
    
    Args:
        df: DataFrame containing the data
        value_column: Column to rank by
        ascending: Whether to rank in ascending order
        
    Returns:
        DataFrame with new columns: 'rank' (numeric) and 'rank_display' (with 공동 notation)
    """
    logger.info(f"Calculating rankings based on {value_column} (ascending={ascending})")
    
    # Make a copy to avoid modifying the original
    df_result = df.copy()
    
    # Calculate numeric ranks
    df_result['rank'] = df_result[value_column].rank(ascending=ascending, method='min')
    
    # Sort by the value column
    df_sorted = df_result.sort_values(by=value_column, ascending=ascending).copy()
    
    # Initialize variables for tracking
    current_rank = 1
    previous_value = None
    tied_count = 0
    ranks = []
    rank_displays = []
    
    # Calculate ranks with ties
    for idx, row in df_sorted.iterrows():
        current_value = row[value_column]
        
        # Check if this is a tie with the previous value
        if previous_value is not None and abs(current_value - previous_value) < 1e-10:  # Use small epsilon for float comparison
            tied_count += 1
            # Keep the same rank number but mark as tied (공동)
            ranks.append(current_rank - tied_count)
            rank_displays.append(f"공동 {current_rank - tied_count}위")
        else:
            # New rank, accounting for any previous ties
            current_rank += tied_count
            ranks.append(current_rank)
            rank_displays.append(f"{current_rank}위")
            tied_count = 0
            current_rank += 1
            
        previous_value = current_value
    
    # Add ranks back to the dataframe
    df_sorted['rank'] = ranks
    df_sorted['rank_display'] = rank_displays
    
    # Merge back to the original order
    df_result = df_result.merge(
        df_sorted[['rank_display']], 
        left_index=True, 
        right_index=True, 
        how='left',
        suffixes=('', '_new')
    )
    
    return df_result

# Function to generate HTML report
def generate_html_report(df, timestamp):
    """Generate an HTML report of the rankings.
    
    Args:
        df: DataFrame with ranking data
        timestamp: Timestamp for the report
        
    Returns:
        HTML content as string
    """
    # Get ranking method and log transform from the dataframe attributes if available
    ranking_method = df.attrs.get('ranking_method', 'relative')
    use_log_transform = df.attrs.get('use_log_transform', False)
    
    # Get the features used for Spearman calculation
    used_features = df.attrs.get('used_features', [])
    
    # Get current timestamp
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mobile Plan Rankings - {timestamp_str}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .good-value {{ color: green; }}
            .bad-value {{ color: red; }}
            .container {{ max-width: 100%; overflow-x: auto; }}
            .note {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; margin-bottom: 20px; }}
            .button-group {{ margin-bottom: 15px; }}
            button {{ padding: 10px 15px; background-color: #007bff; color: white; border: none; 
                     border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px; }}
            button:hover {{ background-color: #0056b3; }}
            button.active {{ background-color: #28a745; }}
            .hidden {{ display: none; }}
        </style>
    </head>
    <body>
        <h1>Mobile Plan Rankings (Spearman Method)</h1>
        <p>Generated: {timestamp_str}</p>
        
        <div class="note">
            <strong>Instructions:</strong> Use the buttons below to toggle between different ranking methods,
            fee types, and log transformation options.
        </div>
        
        <h2>Control Panel</h2>
        <div class="button-group">
            <strong>Ranking Method:</strong><br>
            <button id="relative-btn" {"class='active'" if ranking_method == 'relative' else ""} onclick="changeRankMethod('relative')">Relative Value (ΔP/fee)</button>
            <button id="absolute-btn" {"class='active'" if ranking_method == 'absolute' else ""} onclick="changeRankMethod('absolute')">Absolute Value (ΔP)</button>
            <button id="net-btn" {"class='active'" if ranking_method == 'net' else ""} onclick="changeRankMethod('net')">Net Value (ΔP-fee)</button>
        </div>
        
        <div class="button-group">
            <strong>Fee Type:</strong><br>
            <button id="original-fee-btn" class="active" onclick="changeFeeType('original')">Original Fee</button>
            <button id="discounted-fee-btn" onclick="changeFeeType('discounted')">Discounted Fee</button>
        </div>
        
        <div class="button-group">
            <strong>Log Transform:</strong><br>
            <button id="log-transform-on-btn" {"class='active'" if use_log_transform else ""} onclick="toggleLogTransform(true)">On</button>
            <button id="log-transform-off-btn" {"class='active'" if not use_log_transform else ""} onclick="toggleLogTransform(false)">Off</button>
        </div>
    """
    
    # Get feature weights
    html += """
        <h2>Feature Weights</h2>
        <div id="feature-weights-container">
        <table>
            <tr>
                <th>Feature</th>
                <th>Weight</th>
                <th>Average Contribution (KRW)</th>
            </tr>
    """
    
    # Get the feature weights from dataframe attributes
    weights = df.attrs.get('feature_weights', {})
    
    # Get contribution columns
    contribution_cols = [col for col in df.columns if col.startswith("contribution_")]
    
    # Sort contribution columns by average contribution (descending)
    sorted_contribution_cols = sorted(
        contribution_cols,
        key=lambda x: df[x].mean() if not pd.isna(df[x].mean()) else -float('inf'),
        reverse=True
    )
    
    for col in sorted_contribution_cols:
        feature_name = col.replace("contribution_", "")
        avg_contrib = df[col].mean()
        
        # Get the corresponding weight for this feature
        feature_weight = weights.get(feature_name, np.nan)
        
        if pd.isna(avg_contrib):
            if pd.isna(feature_weight):
                html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>
        """
            else:
                html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>{feature_weight:.4f}</td>
            <td>N/A</td>
        </tr>
        """
        else:
            if pd.isna(feature_weight):
                html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>N/A</td>
            <td>{int(avg_contrib):,} KRW</td>
        </tr>
        """
            else:
                html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>{feature_weight:.4f}</td>
            <td>{int(avg_contrib):,} KRW</td>
        </tr>
        """
    
    html += """
        </table>
        </div>
    """
    
    # Add main data table
    html += """
        <h2>Plan Rankings</h2>
        <div class="container" id="main-table-container">
        <table id="main-table">
            <tr>
                <th>Rank</th>
                <th>Plan Name</th>
                <th>Operator</th>
                <th>Original Fee</th>
                <th>Discounted Fee</th>
                <th>Worth Estimate</th>
                <th>Value Ratio</th>
    """
    
    # Add headers for all features used in the calculation
    for feature in used_features:
        # Clean up feature name for display
        display_name = feature.replace('_clean', '').replace('_', ' ').title()
        html += f"<th>{display_name}</th>"
    
    html += """
            </tr>
    """
    
    # Add rows for each plan
    for i, (_, row) in enumerate(df.sort_values('rank').iterrows()):
        plan_name = str(row.get('plan_name', f"Plan {row.get('id', i)}"))
        if len(plan_name) > 30:
            plan_name = plan_name[:27] + "..."
            
        original_fee = f"{int(row.get('original_fee', 0)):,}"
        discounted_fee = f"{int(row.get('fee', 0)):,}"
        predicted_price = f"{int(row.get('predicted_price', 0)):,}"
        
        # Value ratio
        value_ratio = row.get('value_ratio_original', row.get('value_ratio', 0))
        if pd.isna(value_ratio):
            value_ratio_str = "N/A"
            value_class = ""
        else:
            value_ratio_str = f"{value_ratio:.2f}"
            value_class = "good-value" if value_ratio > 1.1 else ("bad-value" if value_ratio < 0.9 else "")
            
        operator = row.get('mvno', "Unknown")
        
        # Don't try to convert rank_display to int since it may contain Korean characters
        rank_display = row.get('rank_display', f"{i+1}")
        
        html += f"""
        <tr>
            <td>{rank_display}</td>
            <td>{plan_name}</td>
            <td>{operator}</td>
            <td>{original_fee}</td>
            <td>{discounted_fee}</td>
            <td>{predicted_price}</td>
            <td class="{value_class}">{value_ratio_str}</td>
        """
        
        # Add values for all features
        for feature in used_features:
            if feature in row:
                # Format the feature value based on its type
                if isinstance(row[feature], bool):
                    value = "Yes" if row[feature] else "No"
                elif isinstance(row[feature], (int, float)):
                    if feature in ['is_5g', 'basic_data_unlimited', 'daily_data_unlimited', 'voice_unlimited', 'message_unlimited']:
                        value = "Yes" if row[feature] == 1 else "No"
                    elif feature == 'unlimited_type_numeric':
                        # Map unlimited type numeric to descriptive text
                        unlimited_types = {
                            0: "Limited",
                            1: "Throttled",
                            2: "Throttled+",
                            3: "Unlimited"
                        }
                        value = unlimited_types.get(row[feature], str(row[feature]))
                    else:
                        # Format with commas if it's a whole number
                        if row[feature] == int(row[feature]):
                            value = f"{int(row[feature]):,}"
                        else:
                            value = f"{row[feature]:.2f}"
                else:
                    value = str(row[feature])
                html += f"<td>{value}</td>"
            else:
                html += "<td>N/A</td>"
        
        html += "</tr>"
    
    html += """
        </table>
        </div>
    """
    
    # Add JavaScript for interactive controls
    html += """
    <script>
    /* Current state */
    let currentState = {
        rankMethod: "relative",
        feeType: "original",
        logTransform: true
    };
    
    /* Store all table containers */
    let tableContainers = {};
    
    /* Initialize on page load */
    document.addEventListener('DOMContentLoaded', function() {
        /* Find the main table in the document */
        const mainTable = document.getElementById('main-table');
        if (!mainTable) return;
        
        /* Create container divs for different views if they don't exist */
        createTableContainers();
        
        /* Set up initial view */
        setTimeout(function() {
            updateVisibleContainer();
        }, 200);
    });
    
    /* Create containers for different ranking views */
    function createTableContainers() {
        /* Clone the table for each ranking method and fee type */
        const rankMethods = ['relative', 'absolute', 'net'];
        const feeTypes = ['original', 'discounted'];
        
        /* Get the parent of the main table */
        const mainTableContainer = document.getElementById('main-table-container');
        
        /* Create container for all tables */
        const container = document.createElement('div');
        container.className = 'rankings-container';
        mainTableContainer.parentNode.insertBefore(container, mainTableContainer);
        
        /* Hide the original table */
        mainTableContainer.style.display = 'none';
        
        /* For each combination, create a container with a cloned table */
        rankMethods.forEach(method => {
            feeTypes.forEach(feeType => {
                const containerId = `${method}-${feeType}`;
                const newContainer = document.createElement('div');
                newContainer.id = containerId;
                newContainer.className = 'container hidden';
                newContainer.innerHTML = mainTableContainer.innerHTML;
                container.appendChild(newContainer);
                
                tableContainers[containerId] = newContainer;
            });
        });
        
        /* Set the default view to visible */
        const defaultContainer = document.getElementById('relative-original');
        if (defaultContainer) {
            defaultContainer.classList.remove('hidden');
        }
    }
    
    /* Change ranking method */
    function changeRankMethod(method) {
        /* Update buttons */
        document.getElementById('relative-btn').classList.remove('active');
        document.getElementById('absolute-btn').classList.remove('active');
        document.getElementById('net-btn').classList.remove('active');
        
        /* Update button styles */
        document.getElementById(method + '-btn').classList.add('active');
        
        /* Update state */
        currentState.rankMethod = method;
        
        /* Update visible container */
        updateVisibleContainer();
    }
    
    /* Change fee type */
    function changeFeeType(type) {
        /* Update buttons */
        document.getElementById('original-fee-btn').classList.remove('active');
        document.getElementById('discounted-fee-btn').classList.remove('active');
        
        /* Update button styles */
        document.getElementById(type + '-fee-btn').classList.add('active');
        
        /* Update state */
        currentState.feeType = type;
        
        /* Update visible container */
        updateVisibleContainer();
    }
    
    /* Toggle log transform */
    function toggleLogTransform(enabled) {
        document.getElementById('log-transform-on-btn').classList.remove('active');
        document.getElementById('log-transform-off-btn').classList.remove('active');
        
        if (enabled) {
            document.getElementById('log-transform-on-btn').classList.add('active');
        } else {
            document.getElementById('log-transform-off-btn').classList.add('active');
        }
        
        currentState.logTransform = enabled;
        
        // Note: In a real implementation, this would trigger a recalculation
        alert("Changing log transform would require recalculating all rankings. This feature is shown for UI demonstration only.");
    }
    
    /* Update visible container based on current state */
    function updateVisibleContainer() {
        /* Hide all containers */
        const containers = document.querySelectorAll('.rankings-container .container');
        containers.forEach(container => {
            container.classList.add('hidden');
        });
        
        /* Show the selected container */
        const containerId = `${currentState.rankMethod}-${currentState.feeType}`;
        const containerElement = document.getElementById(containerId);
        if (containerElement) {
            containerElement.classList.remove('hidden');
        } else {
            /* Fallback to relative-original if the selected container doesn't exist */
            document.getElementById('relative-original').classList.remove('hidden');
            /* Update state and buttons to match */
            currentState.rankMethod = 'relative';
            currentState.feeType = 'original';
            document.getElementById('relative-btn').classList.add('active');
            document.getElementById('original-fee-btn').classList.add('active');
        }
    }
    </script>
    </body>
    </html>
    """
    
    return html

# Function to save a report
def save_report(html_content, timestamp):
    """Save an HTML report to the reports directory."""
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Ensure report directory exists
    for report_dir in [Path("./reports"), REPORT_DIR_BASE]:
        os.makedirs(report_dir, exist_ok=True)
        report_path = report_dir / f"plan_rankings_spearman_{timestamp_str}.html"
        
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Report saved to {report_path}")
            return str(report_path)
        except Exception as e:
            logger.error(f"Failed to save report to {report_path}: {e}")
    
    # If we get here, all save attempts failed
    logger.error("Failed to save report to any location")
    return None

# Define FastAPI endpoints
@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Serve the latest ranking HTML report if available.
    Similar to the original app, but without logical test functionality.
    """
    # Look for the latest HTML report in all potential directories
    report_dirs = [Path("./reports"), Path("/tmp/reports"), Path("/tmp")]
    
    html_files = []
    for reports_dir in report_dirs:
        if reports_dir.exists():
            html_files.extend(list(reports_dir.glob("plan_rankings_*.html")))
    
    if not html_files:
        # No reports found, return welcome message similar to original
        return """
        <html>
            <head>
                <title>Moyo Ranking Model API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                    h1 { color: #2c3e50; }
                    .method-info { background-color: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin-bottom: 20px; }
                    .button-group { margin-bottom: 15px; }
                    button { padding: 10px 15px; background-color: #007bff; color: white; border: none; 
                             border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px; }
                    button:hover { background-color: #0056b3; }
                    button.active { background-color: #28a745; }
                    .hidden { display: none; }
                </style>
            </head>
            <body>
                <h1>Welcome to the Moyo Ranking Model API</h1>
                
                <div class="method-info">
                    <h2>Spearman Correlation Ranking Method</h2>
                    <p>This API uses the Spearman correlation method to estimate plan worth based on feature importance:</p>
                    <ol>
                        <li>Calculate Spearman correlation between each feature and the original plan fee</li>
                        <li>Apply log(1+x) transformation to non-categorical features</li>
                        <li>Normalize correlations to create feature weights</li>
                        <li>Normalize each feature to [0,1] range</li>
                        <li>Calculate weighted score for each plan with correlation signs</li>
                        <li>Scale scores to KRW range</li>
                        <li>Rank by value ratio (predicted price / fee)</li>
                    </ol>
                </div>
                
                <p>No ranking reports are available yet. Use the <code>/process</code> endpoint to analyze data and generate rankings.</p>
                
                <h2>Ranking Options</h2>
                <div class="button-group">
                    <strong>Ranking Method:</strong><br>
                    <button id="relative-btn" class="active" onclick="changeRankMethod('relative')">Relative Value (ΔP/fee)</button>
                    <button id="absolute-btn" onclick="changeRankMethod('absolute')">Absolute Value (ΔP)</button>
                    <button id="net-btn" onclick="changeRankMethod('net')">Net Value (ΔP-fee)</button>
                </div>
                
                <div class="button-group">
                    <strong>Fee Type:</strong><br>
                    <button id="original-fee-btn" class="active" onclick="changeFeeType('original')">Original Fee</button>
                    <button id="discounted-fee-btn" onclick="changeFeeType('discounted')">Discounted Fee</button>
                </div>
                
                <div class="button-group">
                    <strong>Log Transform:</strong><br>
                    <button id="log-transform-on-btn" class="active" onclick="toggleLogTransform(true)">On</button>
                    <button id="log-transform-off-btn" onclick="toggleLogTransform(false)">Off</button>
                </div>
                
                <p class="method-info">Note: These options will be applied when you generate a new report using the <code>/process</code> endpoint.</p>
                
                <hr>
                <h3>Endpoints</h3>
                <ul>
                    <li><code>POST /process</code>: Submit plan data (JSON list) to preprocess, rank using Spearman method, and generate a report.</li>
                    <li><code>POST /test</code>: Echo back the request body (for debugging).</li>
                </ul>
                <hr>
                <p><i>Navigate to /docs for API documentation (Swagger UI).</i></p>
                
                <script>
                /* Current state */
                let currentState = {
                    rankMethod: "relative",
                    feeType: "original",
                    logTransform: true
                };
                
                /* Change ranking method */
                function changeRankMethod(method) {
                    /* Update buttons */
                    document.getElementById('relative-btn').classList.remove('active');
                    document.getElementById('absolute-btn').classList.remove('active');
                    document.getElementById('net-btn').classList.remove('active');
                    document.getElementById(method + '-btn').classList.add('active');
                    
                    /* Update state */
                    currentState.rankMethod = method;
                    console.log("Ranking method changed to: " + method);
                }
                
                /* Change fee type */
                function changeFeeType(type) {
                    /* Update buttons */
                    document.getElementById('original-fee-btn').classList.remove('active');
                    document.getElementById('discounted-fee-btn').classList.remove('active');
                    document.getElementById(type + '-fee-btn').classList.add('active');
                    
                    /* Update state */
                    currentState.feeType = type;
                    console.log("Fee type changed to: " + type);
                }
                
                /* Toggle log transform */
                function toggleLogTransform(enabled) {
                    /* Update buttons */
                    document.getElementById('log-transform-on-btn').classList.remove('active');
                    document.getElementById('log-transform-off-btn').classList.remove('active');
                    
                    if (enabled) {
                        document.getElementById('log-transform-on-btn').classList.add('active');
                    } else {
                        document.getElementById('log-transform-off-btn').classList.add('active');
                    }
                    
                    /* Update state */
                    currentState.logTransform = enabled;
                    console.log("Log transform set to: " + enabled);
                }
                </script>
            </body>
        </html>
        """
    
    # Get the latest report by modification time
    latest_report = max(html_files, key=lambda x: x.stat().st_mtime)
    print(f"Serving latest report: {latest_report}")
    
    # Set the latest_report_path variable 
    latest_report_path = f"/reports/{latest_report.name}"
    
    # Read and return the HTML content
    try:
        with open(latest_report, "r", encoding="utf-8") as f:
            html_content = f.read()
            
        # Insert additional UI controls before the closing </body> tag
        if '</body>' in html_content:
            interactive_controls = """
            <hr>
            <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0;">
                <h3>Spearman Correlation Ranking Method</h3>
                <p>This report uses Spearman correlation coefficients to estimate plan value based on feature importance:</p>
                <ol>
                    <li>Calculate Spearman correlation between each feature and the original plan fee</li>
                    <li>Apply log(1+x) transformation to non-categorical features</li>
                    <li>Normalize correlations to create feature weights</li>
                    <li>Normalize each feature to [0,1] range</li>
                    <li>Calculate weighted score with correlation signs for each plan</li>
                    <li>Scale scores to KRW range</li>
                    <li>Rank by value ratio (predicted price / fee)</li>
                </ol>
                
                <div style="margin-top: 20px;">
                    <h3>Ranking Options</h3>
                    <div style="margin-bottom: 15px;">
                        <strong>Ranking Method:</strong><br>
                        <button id="relative-btn" class="active" style="padding: 10px 15px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px;" onclick="changeRankMethod('relative')">Relative Value (ΔP/fee)</button>
                        <button id="absolute-btn" style="padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px;" onclick="changeRankMethod('absolute')">Absolute Value (ΔP)</button>
                        <button id="net-btn" style="padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px;" onclick="changeRankMethod('net')">Net Value (ΔP-fee)</button>
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <strong>Fee Type:</strong><br>
                        <button id="original-fee-btn" class="active" style="padding: 10px 15px; background-color: #28a745; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px;" onclick="changeFeeType('original')">Original Fee</button>
                        <button id="discounted-fee-btn" style="padding: 10px 15px; background-color: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px;" onclick="changeFeeType('discounted')">Discounted Fee</button>
                    </div>
                </div>
            </div>
            
            <script>
            /* Current state */
            let currentState = {
                rankMethod: "relative",
                feeType: "original"
            };
            
            /* Store all table containers */
            let tableContainers = {};
            
            /* Initialize on page load */
            document.addEventListener('DOMContentLoaded', function() {
                /* Find the main table in the document */
                const mainTable = document.querySelector('table');
                if (!mainTable) return;
                
                /* Create container divs for different views if they don't exist */
                createRankingContainers(mainTable);
                
                /* Set up initial view */
                updateVisibleContainer();
            });
            
            /* Create containers for different ranking views */
            function createRankingContainers(mainTable) {
                /* Clone the table for each ranking method and fee type */
                const rankMethods = ['relative', 'absolute', 'net'];
                const feeTypes = ['original', 'discounted'];
                
                /* Get the parent of the main table */
                const tableParent = mainTable.parentNode;
                
                /* Create container for all tables */
                const rankingsContainer = document.createElement('div');
                rankingsContainer.className = 'rankings-container';
                tableParent.insertBefore(rankingsContainer, mainTable);
                
                /* Hide the original table */
                mainTable.style.display = 'none';
                
                /* For each combination, create a container with a cloned table */
                for (const method of rankMethods) {
                    for (const feeType of feeTypes) {
                        const containerId = `${method}-${feeType}`;
                        const container = document.createElement('div');
                        container.id = containerId;
                        container.className = 'container';
                        container.style.display = 'none'; /* Hide all initially */
                        
                        /* Clone the table for this view */
                        const tableClone = mainTable.cloneNode(true);
                        container.appendChild(tableClone);
                        
                        /* Add to the rankings container */
                        rankingsContainer.appendChild(container);
                        
                        /* Store reference */
                        tableContainers[containerId] = container;
                    }
                }
                
                /* Set the default view to visible */
                if (tableContainers['relative-original']) {
                    tableContainers['relative-original'].style.display = 'block';
                }
            }
            
            /* Change ranking method */
            function changeRankMethod(method) {
                /* Update buttons */
                document.getElementById('relative-btn').classList.remove('active');
                document.getElementById('absolute-btn').classList.remove('active');
                document.getElementById('net-btn').classList.remove('active');
                document.getElementById(method + '-btn').classList.add('active');
                
                /* Update button styles */
                document.getElementById('relative-btn').style.backgroundColor = '#007bff';
                document.getElementById('absolute-btn').style.backgroundColor = '#007bff';
                document.getElementById('net-btn').style.backgroundColor = '#007bff';
                document.getElementById(method + '-btn').style.backgroundColor = '#28a745';
                
                /* Update state */
                currentState.rankMethod = method;
                
                /* Update visible container */
                updateVisibleContainer();
            }
            
            /* Change fee type */
            function changeFeeType(type) {
                /* Update buttons */
                document.getElementById('original-fee-btn').classList.remove('active');
                document.getElementById('discounted-fee-btn').classList.remove('active');
                document.getElementById(type + '-fee-btn').classList.add('active');
                
                /* Update button styles */
                document.getElementById('original-fee-btn').style.backgroundColor = '#007bff';
                document.getElementById('discounted-fee-btn').style.backgroundColor = '#007bff';
                document.getElementById(type + '-fee-btn').style.backgroundColor = '#28a745';
                
                /* Update state */
                currentState.feeType = type;
                
                /* Update visible container */
                updateVisibleContainer();
            }
            
            /* Update visible container based on current state */
            function updateVisibleContainer() {
                /* Hide all containers */
                for (const containerId in tableContainers) {
                    tableContainers[containerId].style.display = 'none';
                }
                
                /* Show the selected container */
                const containerId = `${currentState.rankMethod}-${currentState.feeType}`;
                if (tableContainers[containerId]) {
                    tableContainers[containerId].style.display = 'block';
                } else {
                    /* Fallback to relative-original if the selected container doesn't exist */
                    if (tableContainers['relative-original']) {
                        tableContainers['relative-original'].style.display = 'block';
                        
                        /* Update state and buttons to match */
                        currentState.rankMethod = 'relative';
                        currentState.feeType = 'original';
                        
                        document.getElementById('relative-btn').classList.add('active');
                        document.getElementById('original-fee-btn').classList.add('active');
                        
                        document.getElementById('relative-btn').style.backgroundColor = '#28a745';
                        document.getElementById('original-fee-btn').style.backgroundColor = '#28a745';
                    }
                }
            }
            </script>
            """
            insert_pos = html_content.find('</body>')
            html_content = html_content[:insert_pos] + interactive_controls + html_content[insert_pos:]
            logger.info(f"Added interactive ranking controls to HTML report")
            
        return html_content
    except Exception as e:
        logger.error(f"Error reading HTML report: {e}")
        return f"""
        <html>
            <head>
                <title>Moyo Ranking Model API - Error</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #e74c3c; }}
                    .method-info {{ background-color: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin-bottom: 20px; }}
                    .button-group {{ margin-bottom: 15px; }}
                    button {{ padding: 10px 15px; background-color: #007bff; color: white; border: none; 
                             border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px; }}
                    button:hover {{ background-color: #0056b3; }}
                    button.active {{ background-color: #28a745; }}
                    .hidden {{ display: none; }}
                </style>
            </head>
            <body>
                <h1>Error Reading Report</h1>
                
                <div class="method-info">
                    <h2>Spearman Correlation Ranking Method</h2>
                    <p>This API uses the Spearman correlation method to estimate plan worth based on feature importance:</p>
                    <ol>
                        <li>Calculate Spearman correlation between each feature and the original plan fee</li>
                        <li>Apply log(1+x) transformation to non-categorical features</li>
                        <li>Normalize correlations to create feature weights</li>
                        <li>Normalize each feature to [0,1] range</li>
                        <li>Calculate weighted score for each plan with correlation signs</li>
                        <li>Scale scores to KRW range</li>
                        <li>Rank by value ratio (predicted price / fee)</li>
                    </ol>
                </div>
                
                <p>Error reading report: {str(e)}</p>
                <p>Please try generating a new report using the <code>/process</code> endpoint.</p>
                
                <h2>Ranking Options</h2>
                <div class="button-group">
                    <strong>Ranking Method:</strong><br>
                    <button id="relative-btn" class="active" onclick="changeRankMethod('relative')">Relative Value (ΔP/fee)</button>
                    <button id="absolute-btn" onclick="changeRankMethod('absolute')">Absolute Value (ΔP)</button>
                    <button id="net-btn" onclick="changeRankMethod('net')">Net Value (ΔP-fee)</button>
                </div>
                
                <div class="button-group">
                    <strong>Fee Type:</strong><br>
                    <button id="original-fee-btn" class="active" onclick="changeFeeType('original')">Original Fee</button>
                    <button id="discounted-fee-btn" onclick="changeFeeType('discounted')">Discounted Fee</button>
                </div>
                
                <div class="button-group">
                    <strong>Log Transform:</strong><br>
                    <button id="log-transform-on-btn" class="active" onclick="toggleLogTransform(true)">On</button>
                    <button id="log-transform-off-btn" onclick="toggleLogTransform(false)">Off</button>
                </div>
                
                <p class="method-info">Note: These options will be applied when you generate a new report using the <code>/process</code> endpoint.</p>
                
            <hr>
            <h3>Endpoints</h3>
            <ul>
                    <li><code>POST /process</code>: Submit plan data (JSON list) to preprocess, rank using Spearman method, and generate a report.</li>
                    <li><code>POST /test</code>: Echo back the request body (for debugging).</li>
            </ul>
            
            <script>
            /* Current state */
            let currentState = {{
                rankMethod: "relative",
                feeType: "original",
                logTransform: true
            }};
            
            /* Change ranking method */
            function changeRankMethod(method) {{
                /* Update buttons */
                document.getElementById('relative-btn').classList.remove('active');
                document.getElementById('absolute-btn').classList.remove('active');
                document.getElementById('net-btn').classList.remove('active');
                document.getElementById(method + '-btn').classList.add('active');
                
                /* Update state */
                currentState.rankMethod = method;
                console.log("Ranking method changed to: " + method);
            }}
            
            /* Change fee type */
            function changeFeeType(type) {{
                /* Update buttons */
                document.getElementById('original-fee-btn').classList.remove('active');
                document.getElementById('discounted-fee-btn').classList.remove('active');
                document.getElementById(type + '-fee-btn').classList.add('active');
                
                /* Update state */
                currentState.feeType = type;
                console.log("Fee type changed to: " + type);
            }}
            
            /* Toggle log transform */
            function toggleLogTransform(enabled) {{
                /* Update buttons */
                document.getElementById('log-transform-on-btn').classList.remove('active');
                document.getElementById('log-transform-off-btn').classList.remove('active');
                
                if (enabled) {{
                    document.getElementById('log-transform-on-btn').classList.add('active');
                }} else {{
                    document.getElementById('log-transform-off-btn').classList.add('active');
                }}
                
                /* Update state */
                currentState.logTransform = enabled;
                console.log("Log transform set to: " + enabled);
            }}
            </script>
        </body>
    </html>
    """

@app.post("/process")
async def process_data(request: Request):
    """Process plan data using the Spearman method."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received /process request")
    
    try:
        # Step 1: Ensure directories exist
        ensure_directories()
        
        # Step 2: Parse request data and options
        request_json = await request.json()
        
        # Check if the request includes data and/or options
        if isinstance(request_json, dict):
            # Structure with options and data
            options = request_json.get('options', {})
            data = request_json.get('data', [])
            
            # If data is not in the expected format, assume the entire body is the data
            if not isinstance(data, list):
                data = request_json
                options = {}
        else:
            # Assume the entire body is the data array
            data = request_json
            options = {}
        
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Expected a list of plan data")

        logger.info(f"[{request_id}] Received {len(data)} plans")
        
        # Extract ranking options with defaults
        rank_method = options.get('rankMethod', 'relative')
        use_log_transform = options.get('logTransform', True)
        fee_type = options.get('feeType', 'original')
        
        logger.info(f"[{request_id}] Using ranking options: method={rank_method}, fee_type={fee_type}, log_transform={use_log_transform}")

        # Step 3: Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_path = save_raw_data(data, timestamp)
        
        # Step 4: Preprocess data
        df = pd.DataFrame(data)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data to process")
        
        processed_df = prepare_features(df)
        logger.info(f"[{request_id}] Processed DataFrame shape: {processed_df.shape}")
        
        # Free memory
        del df
        gc.collect()
        
        # Step 5: Save processed data
        processed_data_paths = save_processed_data(processed_df)
        latest_processed_path = processed_data_paths[1] if len(processed_data_paths) > 1 else processed_data_paths[0]
        
        # Step 6: Apply Spearman ranking method with options
        # Note: calculate_rankings_with_spearman now calculates ALL ranking types internally
        df_ranked = calculate_rankings_with_spearman(
            processed_df,
            use_log_transform=use_log_transform,
            rank_method=rank_method
        )
        
        logger.info(f"[{request_id}] Ranked DataFrame shape: {df_ranked.shape}")
        
        # Save to global variable for later use
        global df_with_rankings
        df_with_rankings = df_ranked.copy()
        
        # Step 7: Generate HTML report
        timestamp_now = datetime.now()
        html_report = generate_html_report(df_ranked, timestamp_now)
        report_path = save_report(html_report, timestamp_now)
        
        # Step 8: Prepare response with complete ranking data
        # Include all ranking types in the response
        all_rankings = {}
        
        # Group plans by each ranking method
        ranking_methods = {
            'absolute': ('rank_absolute', 'delta_krw'),
            'relative_original': ('rank_relative_original', 'value_ratio_original'),
            'relative_fee': ('rank_relative_fee', 'value_ratio_fee'),
            'net_original': ('rank_net_original', 'net_value_original'),
            'net_fee': ('rank_net_fee', 'net_value_fee')
        }
        
        # Get all plans for all ranking types
        for rank_type, (rank_col, value_col) in ranking_methods.items():
            if rank_col in df_ranked.columns and value_col in df_ranked.columns:
                columns_to_include = ["id", "plan_name", "mvno", "fee", "original_fee", 
                                     "predicted_price", rank_col, "rank_display", value_col]
                available_columns = [col for col in columns_to_include if col in df_ranked.columns]
                
                # Sort by the ranking column and convert to records
                all_rankings[rank_type] = df_ranked.sort_values(rank_col)[available_columns].to_dict(orient="records")
                logger.info(f"[{request_id}] Included {rank_type} rankings with {len(all_rankings[rank_type])} plans")
        
        # Get top 10 by the primary requested method
        top_10_value_col = "value_ratio_original"
        if rank_method == 'absolute':
            top_10_value_col = "delta_krw"
        elif rank_method == 'net':
            top_10_value_col = f"net_value_{fee_type}"
        elif rank_method == 'relative':
            top_10_value_col = f"value_ratio_{fee_type}"
        
        # Get top 10 plans
        top_10_plans = []
        try:
            columns_to_include = ["id", "plan_name", "mvno", "fee", "original_fee", 
                               "predicted_price", "rank_display", "rank", top_10_value_col]
            available_columns = [col for col in columns_to_include if col in df_ranked.columns]
            
            top_10_plans = df_ranked.sort_values(top_10_value_col, ascending=False).head(10)[available_columns].to_dict(orient="records")
            logger.info(f"[{request_id}] Extracted top 10 plans based on {top_10_value_col}")
        except Exception as e:
            logger.error(f"[{request_id}] Error extracting top plans: {e}")
            
        # Create all_ranked_plans (structured for edge function compatibility)
        all_ranked_plans = []
        try:
            # Use the same columns as top_10_plans but include value_ratio explicitly for DB upsert
            columns_to_include = ["id", "plan_name", "mvno", "fee", "original_fee", 
                               "predicted_price", "rank_display", "rank"]
                               
            # If we have a value_ratio column available, include it
            if top_10_value_col in df_ranked.columns:
                columns_to_include.append(top_10_value_col)
            elif "value_ratio" in df_ranked.columns:
                columns_to_include.append("value_ratio")
                
            available_columns = [col for col in columns_to_include if col in df_ranked.columns]
            
            # Get all plans, sorted by the current ranking method
            all_ranked_plans = df_ranked.sort_values(top_10_value_col, ascending=False)[available_columns].to_dict(orient="records")
            
            # Ensure each plan has a value_ratio field (required for edge function DB upsert)
            for plan in all_ranked_plans:
                # If we don't already have value_ratio, add it from the appropriate column
                if "value_ratio" not in plan and top_10_value_col in plan:
                    plan["value_ratio"] = plan[top_10_value_col]
            
            logger.info(f"[{request_id}] Prepared all_ranked_plans with {len(all_ranked_plans)} plans")
        except Exception as e:
            logger.error(f"[{request_id}] Error preparing all_ranked_plans: {e}")
        
        # Calculate timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        response = {
            "request_id": request_id,
            "message": "Data processing complete using Spearman correlation method",
            "status": "success",
            "processing_time_seconds": round(processing_time, 4),
            "options": {
                "rankMethod": rank_method,
                "feeType": fee_type,
                "logTransform": use_log_transform
            },
            "results": {
                "raw_data_path": raw_data_path,
                "processed_data_path": latest_processed_path,
                "report_path": report_path,
                "report_url": f"/reports/{Path(report_path).name}" if report_path else None
            },
            "ranking_method": "spearman",
            "top_10_plans": top_10_plans,
            "all_ranked_plans": all_ranked_plans
        }
        
        return response
    except Exception as e:
        logger.exception(f"[{request_id}] Error in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.post("/test")
def test(request: dict = Body(...)):
    """Simple echo endpoint for testing (returns the provided data)."""
    return {"received": request}

# Run the application
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=7860)