from fastapi import FastAPI, HTTPException, Request, Response
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
    fee: float
    original_fee: float
    discount_fee: float
    discount_period: Optional[int] = None
    post_discount_fee: float

# Spearman Method Implementation
def calculate_rankings_with_spearman(df, use_log_transform=True, rank_method='relative'):
    """
    Generate worth estimates and rankings using Spearman correlation method.
    
    Args:
        df: DataFrame with preprocessed plan data
        use_log_transform: Whether to apply log transformation to non-categorical features
        rank_method: Method to use for ranking ('relative', 'absolute', or 'net')
        
    Returns:
        DataFrame with added worth estimates and rankings
    """
    logger.info(f"Calculating rankings with Spearman method (rank method: {rank_method}, log transform: {use_log_transform})")
    
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
    
    # Normalize each feature to [0,1]
    norm = {}  # a dict of arrays
    for feature in weights.keys():
        # Create a mask for non-NaN values
        valid_mask = ~np.isnan(df_analysis[feature].values)
        
        if valid_mask.sum() == 0:
            logger.warning(f"Feature '{feature}' has all NaN values - skipping")
            continue
        
        # Get min and max from valid values only
        min_v = df_analysis[feature].loc[valid_mask].min()
        max_v = df_analysis[feature].loc[valid_mask].max()
        
        # Handle the case where min and max are the same
        if max_v == min_v:
            logger.info(f"Feature '{feature}' has min=max={min_v}, setting normalized values to 0.5")
            # Create array with 0.5 for valid values, NaN for invalid
            norm_values = np.full(len(df_analysis), 0.5)
            norm_values[~valid_mask] = np.nan
            norm[feature] = norm_values
        else:
            # Create normalized array, preserving NaN
            norm_values = np.full(len(df_analysis), np.nan)
            norm_values[valid_mask] = (df_analysis[feature].loc[valid_mask] - min_v) / (max_v - min_v)
            norm[feature] = norm_values
    
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
    
    # Calculate delta_krw directly from the score
    # ΔP_j = S_j × (P_max - P_min)
    P_min = df["original_fee"].min()
    P_max = df["original_fee"].max()
    price_range = P_max - P_min
    
    delta_krw = [S_j * price_range for S_j in scores]
    
    # Linearly scale scores back into KRW range for worth_estimate
    # worth_estimate = P_min + ΔP_j
    worth = [P_min + delta for delta in delta_krw]
    
    # Attach estimates back to the DataFrame
    df_result = df.copy()
    df_result["predicted_price"] = worth  # Use predicted_price for consistency with model-based approach
    df_result["delta_krw"] = delta_krw
    
    # Calculate metrics using original_fee
    df_result["value_ratio_original"] = df_result["predicted_price"] / df_result["original_fee"]
    df_result["net_value_original"] = df_result["delta_krw"] - df_result["original_fee"]
    
    # Calculate metrics using fee (discounted price)
    df_result["value_ratio_fee"] = df_result["predicted_price"] / df_result["fee"]
    df_result["net_value_fee"] = df_result["delta_krw"] - df_result["fee"]
    
    # Calculate standard value ratio (for compatibility with existing code)
    df_result["value_ratio"] = df_result["predicted_price"] / df_result["fee"]
    
    # Calculate rankings based on both original and discounted fees
    # Absolute value ranking
    df_result["rank_absolute"] = df_result["delta_krw"].rank(ascending=False)
    
    # Relative value rankings
    df_result["rank_relative_original"] = df_result["value_ratio_original"].rank(ascending=False)
    df_result["rank_relative_fee"] = df_result["value_ratio_fee"].rank(ascending=False)
    
    # Net value rankings
    df_result["rank_net_original"] = df_result["net_value_original"].rank(ascending=False)
    df_result["rank_net_fee"] = df_result["net_value_fee"].rank(ascending=False)
    
    # Set the primary rank_number based on method and fee type
    if rank_method == 'absolute':
        df_result["rank"] = df_result["rank_absolute"]
        ranking_column = "delta_krw"
    elif rank_method == 'net':
        df_result["rank"] = df_result["rank_net_original"]
        ranking_column = "net_value_original"
    else:  # relative (default)
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
                # Direct contribution calculation with sign
                contribution[valid_mask] = rho_signs[feature] * weights[feature] * norm[feature][valid_mask] * price_range
            
            df_result[contribution_col] = contribution
        except Exception as e:
            logger.error(f"Error calculating contribution for {feature}: {e}")
            df_result[contribution_col] = np.nan
    
    logger.info(f"Completed Spearman ranking for {len(df_result)} plans")
    
    # Store the feature weights in the dataframe attributes
    df_result.attrs['used_features'] = list(weights.keys())
    df_result.attrs['feature_weights'] = weights
    df_result.attrs['feature_signs'] = rho_signs
    df_result.attrs['ranking_method'] = 'spearman'
    
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
    # Sort the dataframe by the value column
    df_sorted = df.sort_values(by=value_column, ascending=ascending).copy()
    
    # Initialize variables for tracking
    ranks = []
    rank_displays = []
    current_rank = 1
    i = 0
    n = len(df_sorted)
    while i < n:
        current_value = df_sorted.iloc[i][value_column]
        # Find all tied indices
        tie_indices = [i]
        j = i + 1
        while j < n and df_sorted.iloc[j][value_column] == current_value:
            tie_indices.append(j)
            j += 1
        tie_count = len(tie_indices)
        if tie_count > 1:
            for _ in tie_indices:
                ranks.append(current_rank)
                rank_displays.append(f"공동 {current_rank}위")
        else:
            ranks.append(current_rank)
            rank_displays.append(f"{current_rank}위")
        current_rank += tie_count
        i += tie_count
    # Add ranks back to the dataframe
    df_sorted['rank'] = ranks
    df_sorted['rank_display'] = rank_displays
    # Return to original order
    df_sorted = df_sorted.reindex(df.index)
    return df_sorted

# Function to generate HTML report
def generate_html_report(df, timestamp):
    """Generate an HTML report of the Spearman rankings."""
    
    # Get feature weights from dataframe attributes
    feature_weights = df.attrs.get('feature_weights', {})
    feature_signs = df.attrs.get('feature_signs', {})
    used_features = df.attrs.get('used_features', [])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mobile Plan Rankings - Spearman Method</title>
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
        </style>
    </head>
    <body>
        <h1>Mobile Plan Rankings (Spearman Correlation Method)</h1>
        <p>Generated: {timestamp}</p>
        
        <div class="note">
            <h3>Ranking Methodology</h3>
            <p>This report uses Spearman correlation coefficients to estimate plan value based on feature importance.</p>
            <ol>
                <li>Calculate Spearman correlation between each feature and the original plan fee</li>
                <li>Apply log(1+x) transformation to non-categorical features</li>
                <li>Normalize correlations to create feature weights</li>
                <li>Normalize each feature to [0,1] range</li>
                <li>Calculate weighted score with correlation signs for each plan</li>
                <li>Scale scores to KRW range</li>
                <li>Rank by value ratio (predicted price / fee)</li>
            </ol>
        </div>
        
        <h2>Feature Weights</h2>
        <div class="container">
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Weight</th>
                    <th>Direction</th>
                </tr>
    """
    
    # Add feature weights to the table
    for feature, weight in sorted(feature_weights.items(), key=lambda x: abs(x[1]), reverse=True):
        sign = feature_signs.get(feature, 0)
        direction = "Positive (↑)" if sign > 0 else "Negative (↓)" if sign < 0 else "Neutral"
        direction_class = "good-value" if sign > 0 else "bad-value" if sign < 0 else ""
        
        html += f"""
                <tr>
                    <td>{feature}</td>
                    <td>{abs(weight):.4f}</td>
                    <td class="{direction_class}">{direction}</td>
                </tr>
        """
    
    html += """
            </table>
        </div>
        
        <h2>Plan Rankings</h2>
        <div class="container">
            <table>
                <tr>
                    <th>Rank</th>
                    <th>Plan Name</th>
                    <th>MVNO</th>
                    <th>Original Fee</th>
                    <th>Fee</th>
                    <th>Predicted Worth</th>
                    <th>Value Ratio</th>
    """
    
    # Add feature columns
    for feature in used_features:
        html += f"<th>{feature}</th>"
    
    html += """
                </tr>
    """
    
    # Add plan rows
    for i, row in df.sort_values('rank').iterrows():
        plan_name = row.get('plan_name', f"Plan {row.get('id', i)}")
        mvno = row.get('mvno', 'N/A')
        original_fee = f"{int(row['original_fee']):,}" if 'original_fee' in row else 'N/A'
        fee = f"{int(row['fee']):,}" if 'fee' in row else original_fee
        predicted_price = f"{int(row['predicted_price']):,}" if 'predicted_price' in row else 'N/A'
        
        value_ratio = row.get('value_ratio', 0)
        value_class = "good-value" if value_ratio > 1.1 else "bad-value" if value_ratio < 0.9 else ""
        
        html += f"""
                <tr>
                    <td>{row.get('rank_display', i+1)}</td>
                    <td>{plan_name}</td>
                    <td>{mvno}</td>
                    <td>{original_fee}</td>
                    <td>{fee}</td>
                    <td>{predicted_price}</td>
                    <td class="{value_class}">{value_ratio:.2f}</td>
        """
        
        # Add feature values
        for feature in used_features:
            if feature in row:
                value = row[feature]
                if isinstance(value, bool):
                    value = "Yes" if value else "No"
                elif isinstance(value, (int, float)):
                    if feature in ['is_5g', 'basic_data_unlimited', 'daily_data_unlimited', 'voice_unlimited', 'message_unlimited']:
                        value = "Yes" if value == 1 else "No"
                    else:
                        value = f"{value:.1f}" if value % 1 != 0 else f"{int(value)}"
                html += f"<td>{value}</td>"
            else:
                html += "<td>N/A</td>"
        
        html += """
                </tr>
        """
    
    html += """
            </table>
        </div>
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
                <hr>
                <h3>Endpoints</h3>
                <ul>
                    <li><code>POST /process</code>: Submit plan data (JSON list) to preprocess, rank using Spearman method, and generate a report.</li>
                    <li><code>POST /predict</code>: Get price predictions and rankings for plan features using Spearman method.</li>
                    <li><code>GET /rankings</code>: Get the latest list of ranked plans.</li>
                    <li><code>GET /features</code>: Get the list of features used for ranking.</li>
                    <li><code>POST /test</code>: Echo back the request body (for debugging).</li>
                </ul>
                <hr>
                <p><i>Navigate to /docs for API documentation (Swagger UI).</i></p>
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
            
        # Insert additional info before the closing </body> tag
        if '</body>' in html_content:
            method_info = """
            <hr>
            <div style="background-color: #f8f9fa; padding: 10px; border-left: 4px solid #28a745; margin-bottom: 20px;">
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
            </div>
            """
            insert_pos = html_content.find('</body>')
            html_content = html_content[:insert_pos] + method_info + html_content[insert_pos:]
            logger.info(f"Added Spearman method information to HTML report")
            
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
                
                <hr>
                <h3>Endpoints</h3>
                <ul>
                    <li><code>POST /process</code>: Submit plan data (JSON list) to preprocess, rank using Spearman method, and generate a report.</li>
                    <li><code>POST /predict</code>: Get price predictions and rankings for plan features using Spearman method.</li>
                    <li><code>GET /rankings</code>: Get the latest list of ranked plans.</li>
                    <li><code>GET /features</code>: Get the list of features used for ranking.</li>
                    <li><code>POST /test</code>: Echo back the request body (for debugging).</li>
                </ul>
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
        
        # Step 2: Receive and validate data
        data = await request.json()
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Expected a list of plan data")
        
        logger.info(f"[{request_id}] Received {len(data)} plans")
        
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
        
        # Step 6: Apply Spearman ranking method
        df_ranked = calculate_rankings_with_spearman(
            processed_df,
            use_log_transform=True,  # Enable log transform
            rank_method='relative'    # Use relative ranking (value ratio)
        )
        
        logger.info(f"[{request_id}] Ranked DataFrame shape: {df_ranked.shape}")
        
        # Save to global variable for later use
        global df_with_rankings
        df_with_rankings = df_ranked.copy()
        
        # Step 7: Generate HTML report
        timestamp_now = datetime.now()
        html_report = generate_html_report(df_ranked, timestamp_now)
        report_path = save_report(html_report, timestamp_now)
        
        # Step 8: Prepare response
        # Get top 10 plans
        top_10_plans = []
        try:
            columns_to_include = ["plan_name", "mvno", "fee", "value_ratio", "predicted_price", "rank_display", "id"]
            available_columns = [col for col in columns_to_include if col in df_ranked.columns]
            
            top_10_plans = df_ranked.sort_values("value_ratio", ascending=False).head(10)[available_columns].to_dict(orient="records")
            
            # Get all ranked plans
            rank_columns = ["id", "predicted_price", "rank", "rank_display", "value_ratio"]
            available_rank_columns = [col for col in rank_columns if col in df_ranked.columns]
            
            all_ranked_plans = df_ranked.sort_values("rank")[available_rank_columns].to_dict(orient="records")
            
            logger.info(f"[{request_id}] Extracted top 10 plans and all {len(all_ranked_plans)} ranked plans")
        except Exception as e:
            logger.error(f"[{request_id}] Error extracting top plans: {e}")
            all_ranked_plans = []
        
        # Calculate timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        response = {
            "request_id": request_id,
            "message": "Data processing complete using Spearman correlation method",
            "status": "success",
            "processing_time_seconds": round(processing_time, 4),
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

@app.post("/predict")
async def predict_plans(plans: List[PlanInput]):
    """Predict and rank plans using the Spearman method."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received {len(plans)} plans for prediction using Spearman method")
    
    try:
        # Convert input to list of dictionaries
        input_data = [plan.dict() for plan in plans]
        
        # Convert to DataFrame
        df_input = pd.DataFrame(input_data)
        if df_input.empty:
            raise HTTPException(status_code=400, detail="Input data is empty")
        
        # Preprocess data
        df_processed = prepare_features(df_input)
        logger.info(f"[{request_id}] Processed {len(df_processed)} plans")
        
        # Apply Spearman ranking
        df_ranked = calculate_rankings_with_spearman(
            df_processed,
            use_log_transform=True,
            rank_method='relative'
        )
        
        logger.info(f"[{request_id}] Ranked {len(df_ranked)} plans using Spearman method")
        
        # Format results
        results = []
        for _, row in df_ranked.sort_values("rank").iterrows():
            plan_id = row["id"] if "id" in row else "unknown"
            plan_id = int(plan_id) if isinstance(plan_id, (int, float)) else plan_id
            
            results.append({
                "plan_id": plan_id,
                "predicted_price": float(row["predicted_price"]),
                "rank": row["rank_display"],
                "value_ratio": float(row["value_ratio"])
            })
        
        end_time = time.time()
        logger.info(f"[{request_id}] Prediction completed in {end_time - start_time:.4f} seconds")
        
        return results
    except Exception as e:
        logger.exception(f"[{request_id}] Error in /predict: {e}")
        raise HTTPException(status_code=500, detail=f"Error predicting prices: {str(e)}")

@app.get("/rankings")
async def get_rankings():
    """Return the complete list of ranked plans."""
    # Check if we have rankings
    global df_with_rankings
    
    if df_with_rankings is None:
        raise HTTPException(status_code=404, detail="No rankings available. Run /process endpoint first")
    
    try:
        # Format response data
        rankings_list = []
        for _, row in df_with_rankings.sort_values("rank").iterrows():
            plan_id = row["id"] if "id" in row else "unknown"
            plan_id = int(plan_id) if isinstance(plan_id, (int, float)) else plan_id
            
            rankings_list.append({
                "plan_id": plan_id,
                "predicted_price": float(row["predicted_price"]),
                "rank": row["rank_display"],
                "value_ratio": float(row["value_ratio"])
            })
        
        # Add ranking method information
        response = {
            "ranking_method": "spearman",
            "rankings": rankings_list
        }
        
        return response
    except Exception as e:
        logger.exception(f"Error generating rankings list: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating rankings: {str(e)}")

@app.post("/test")
def test(request: dict = Body(...)):
    """Simple echo endpoint for testing (returns the provided data)."""
    return request

@app.get("/features")
def get_features():
    """Return the list of features used in the ranking model."""
    try:
        # Try to get features from the processed data
        if df_with_rankings is not None and not df_with_rankings.empty:
            numeric_features = list(df_with_rankings.select_dtypes(include=['number']).columns)
            features = [f for f in numeric_features if f not in ['id', 'rank', 'predicted_price', 'value_ratio']]
            return {"features": features}
        
        # If no data is available, return supported features
        supported_features = [
            "basic_data", "daily_data", "voice", "message", "additional_call", 
            "data_sharing", "roaming_support", "micro_payment", "is_esim", 
            "signup_minor", "signup_foreigner", "tethering_gb", "fee", 
            "original_fee", "discount_fee", "discount_period", "post_discount_fee", 
            "agreement", "agreement_period", "num_of_signup", "mvno_rating", 
            "monthly_review_score", "discount_percentage"
        ]
        return {"features": supported_features}
    except Exception as e:
        logger.error(f"Error retrieving features: {e}")
        return {"error": str(e)}

# Run the application
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=7860)