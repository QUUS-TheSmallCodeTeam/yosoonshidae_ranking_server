from fastapi import FastAPI, HTTPException, Request, Response, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import json
import uuid
import gc
import time
import subprocess
import threading
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Union, List
import os
import psutil
import logging

# Import configuration
from modules.config import config, logger

# Import necessary modules
# Legacy imports (cleaned up - now using consolidated import below)
from fastapi import UploadFile, File
from modules import (
    prepare_features, 
    calculate_rankings_with_ties, 
    generate_html_report, 
    save_report, 
    ensure_directories,
    cleanup_all_datasets,
    rank_plans_by_cs_enhanced
)

# Initialize FastAPI
app = FastAPI(title="Moyo Plan Ranking Model Server - Enhanced Cost-Spec Method")

# Log monitoring functionality
def start_log_monitoring():
    """Start log monitoring in background if script exists"""
    try:
        script_path = Path("./simple_log_monitor.sh")
        if script_path.exists():
            # Check if monitoring is already running
            result = subprocess.run(
                ["ps", "aux"], 
                capture_output=True, 
                text=True
            )
            if "simple_log_monitor.sh" not in result.stdout:
                # Start monitoring in background
                subprocess.Popen(
                    ["sh", str(script_path)], 
                    stdout=subprocess.DEVNULL, 
                    stderr=subprocess.DEVNULL
                )
                logger.info("Started log monitoring script")
            else:
                logger.info("Log monitoring already running")
        else:
            logger.warning("Log monitoring script not found")
    except Exception as e:
        logger.error(f"Failed to start log monitoring: {e}")

# Start monitoring on app startup
@app.on_event("startup")
async def startup_event():
    """Initialize monitoring on app startup"""
    # Wait a moment for the server to fully start
    def delayed_start():
        time.sleep(3)  # Wait 3 seconds for server to be ready
        start_log_monitoring()
    
    # Start in background thread
    threading.Thread(target=delayed_start, daemon=True).start()
    logger.info("FastAPI app started - log monitoring will start in 3 seconds")

# Global variables for storing data
df_with_rankings = None  # Global variable to store the latest rankings
latest_logical_test_results_cache = None  # For storing logical test results

# Async chart calculation globals
chart_calculation_status = {
    'is_calculating': False,
    'last_calculation_time': None,
    'calculation_progress': 0,
    'error_message': None
}
chart_calculation_task = None  # Store the background task

def generate_basic_html_report(df):
    """
    Generate a basic HTML report without expensive chart calculations.
    Used when chart calculation fails or is in progress.
    """
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Get top 10 plans
    top_10 = df.head(10)
    
    # Create simple table HTML
    table_rows = ""
    for idx, row in top_10.iterrows():
        table_rows += f"""
        <tr>
            <td>{row.get('rank_number', idx+1)}</td>
            <td>{row.get('plan_name', 'N/A')}</td>
            <td>{row.get('mvno', 'N/A')}</td>
            <td>₩{row.get('original_fee', 0):,.0f}</td>
            <td>{row.get('CS', 0):.2f}</td>
        </tr>
        """
    
    basic_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mobile Plan Rankings - Basic View</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .note {{ background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <h1>📱 Mobile Plan Rankings</h1>
        <p><strong>Generated:</strong> {timestamp_str}</p>
        
        <div class="note">
            <h3>⚠️ Basic View</h3>
            <p>Advanced charts are being generated in the background. Refresh the page in a few moments to see the full report with visualizations.</p>
        </div>
        
        <h2>Top 10 Plans by Cost-Spec Ratio</h2>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Plan Name</th>
                    <th>Provider</th>
                    <th>Price</th>
                    <th>CS Ratio</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>
        
        <hr>
        <p><em>For the complete analysis with advanced charts, please refresh this page in a few moments.</em></p>
    </body>
    </html>
    """
    
    return basic_html

async def calculate_charts_async(df_ranked, method, cost_structure, request_id):
    """
    Asynchronously calculate charts and generate HTML report.
    This runs in the background after the API response is sent.
    """
    global chart_calculation_status
    
    try:
        chart_calculation_status['is_calculating'] = True
        chart_calculation_status['calculation_progress'] = 10
        chart_calculation_status['error_message'] = None
        
        logger.info(f"[{request_id}] Starting async chart calculation...")
        
        # Set up HTML report info
        timestamp_now = datetime.now()
        method_suffix = "decomp" if method == "linear_decomposition" else "frontier"
        report_filename = f"cs_ranking_{method_suffix}_{timestamp_now.strftime('%Y%m%d_%H%M%S')}.html"
        report_path = config.cs_report_dir / report_filename
        
        chart_calculation_status['calculation_progress'] = 30
        
        # Generate HTML content with method-specific title
        method_name = {
            "linear_decomposition": "Linear Decomposition",
            "multi_frontier": "Multi-Feature Frontier Regression",
            "frontier": "Frontier-Based"
        }.get(method, "Enhanced Cost-Spec")
        title = f"Enhanced Cost-Spec Rankings ({method_name})"
        
        chart_calculation_status['calculation_progress'] = 50
        logger.info(f"[{request_id}] Generating HTML report with charts...")
        
        html_report = generate_html_report(
            df_ranked, 
            timestamp_now, 
            is_cs=True, 
            title=title,
            method=method,
            cost_structure=cost_structure
        )
        
        chart_calculation_status['calculation_progress'] = 80
        
        # Write HTML content to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        chart_calculation_status['calculation_progress'] = 100
        chart_calculation_status['is_calculating'] = False
        chart_calculation_status['last_calculation_time'] = timestamp_now
        
        logger.info(f"[{request_id}] Async chart calculation completed successfully. Report saved to {report_path}")
        
    except Exception as e:
        chart_calculation_status['is_calculating'] = False
        chart_calculation_status['error_message'] = str(e)
        chart_calculation_status['calculation_progress'] = 0
        logger.error(f"[{request_id}] Error in async chart calculation: {str(e)}")

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

# HTML report generation and saving is now handled by the modules.report module

# Define FastAPI endpoints
@app.get("/")
async def root(basic: bool = False):
    """
    Root endpoint that shows either:
    1. Visual status indicators if charts are calculating or failed
    2. Full HTML report if charts are ready
    3. Basic report if basic=true parameter is provided
    """
    global df_with_rankings, chart_calculation_status
    
    try:
        # If basic report requested, generate simple HTML without charts
        if basic:
            if df_with_rankings is not None:
                html_report = generate_html_report(
                    df_with_rankings, 
                    datetime.now(), 
                    is_cs=True, 
                    title="Basic Cost-Spec Rankings (No Charts)",
                    method="basic",
                    cost_structure=None
                )
                return HTMLResponse(content=html_report)
            else:
                return HTMLResponse(content="<h1>No data available. Please process data first via /process endpoint.</h1>")
        
        # Check if we have data
        if df_with_rankings is None:
            return HTMLResponse(content="<h1>No data available. Please process data first via /process endpoint.</h1>")
        
        # Check chart calculation status
        if chart_calculation_status['is_calculating']:
            # Show loading status
            progress_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Charts Calculating...</title>
                <style>
                    body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                    .status-card {{ 
                        background: #f0f8ff; 
                        border: 2px solid #4CAF50; 
                        border-radius: 10px; 
                        padding: 30px; 
                        margin: 20px auto; 
                        max-width: 500px; 
                    }}
                    .loading-icon {{ font-size: 48px; margin-bottom: 20px; }}
                    .progress {{ font-size: 18px; color: #666; }}
                </style>
            </head>
            <body>
                <div class="status-card">
                    <div class="loading-icon">⚙️</div>
                    <h2>Multi-Feature Frontier Regression Analysis</h2>
                    <p class="progress">Calculating charts... {chart_calculation_status['calculation_progress']}%</p>
                    <p><a href="/">Refresh to check progress</a> | <a href="/?basic=true">View basic report</a></p>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=progress_html)
        
        elif chart_calculation_status['error_message']:
            # Show error status
            error_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Chart Generation Failed</title>
                <style>
                    body {{ font-family: Arial, sans-serif; text-align: center; padding: 50px; }}
                    .status-card {{ 
                        background: #fff0f0; 
                        border: 2px solid #f44336; 
                        border-radius: 10px; 
                        padding: 30px; 
                        margin: 20px auto; 
                        max-width: 500px; 
                    }}
                    .error-icon {{ font-size: 48px; margin-bottom: 20px; }}
                    .error-msg {{ font-size: 14px; color: #666; background: #f5f5f5; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="status-card">
                    <div class="error-icon">❌</div>
                    <h2>Chart Generation Failed</h2>
                    <p>There was an error generating the charts.</p>
                    <div class="error-msg">{chart_calculation_status['error_message']}</div>
                    <p><a href="/">Try again</a> | <a href="/?basic=true">View basic report</a></p>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html)
        
        else:
            # Charts are ready - generate fresh HTML report
            method = getattr(df_with_rankings, 'method', 'multi_frontier')
            cost_structure = getattr(df_with_rankings, 'cost_structure', None) or getattr(df_with_rankings, 'multi_frontier_breakdown', None)
            
            method_name = {
                "linear_decomposition": "Linear Decomposition",
                "multi_frontier": "Multi-Feature Frontier Regression",
                "frontier": "Frontier-Based"
            }.get(method, "Enhanced Cost-Spec")
            title = f"Enhanced Cost-Spec Rankings ({method_name})"
            
            html_report = generate_html_report(
                df_with_rankings, 
                datetime.now(), 
                is_cs=True, 
                title=title,
                method=method,
                cost_structure=cost_structure
            )
            return HTMLResponse(content=html_report)
            
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.post("/process")
async def process_data(request: Request):
    """
    Main data processing endpoint - analyzes mobile plans using Cost-Spec enhanced analysis.
    """
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    # Configure logging to ensure cleanup logs are visible
    logging.getLogger().setLevel(logging.INFO)
    
    # Log memory usage before processing
    process = psutil.Process()
    initial_memory_mb = process.memory_info().rss / 1024 / 1024
    logger.info(f"[{request_id}] Initial memory usage: {initial_memory_mb:.2f} MB")
    
    try:
        # Step 1: Ensure directories exist and cleanup old files
        ensure_directories()
        
        # Cleanup old datasets to prevent disk space issues
        # Since pipeline recalculates everything from scratch, keep only 1 previous dataset
        logger.info(f"[{request_id}] Cleaning up old dataset files...")
        cleanup_stats = cleanup_all_datasets(max_files=1, max_age_days=1)
        logger.info(f"[{request_id}] Cleanup completed: {cleanup_stats}")
        if cleanup_stats['total'] > 0:
            logger.info(f"[{request_id}] Successfully cleaned up {cleanup_stats['total']} old files")
        else:
            logger.info(f"[{request_id}] No old files found to clean up")
        
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
        
        # Extract options
        feature_set = options.get('featureSet', 'basic')
        fee_column = options.get('feeColumn', 'fee')  # Fee column to use for comparison
        method = options.get('method', 'multi_frontier')  # Default to multi-frontier method
        tolerance = options.get('tolerance', 500)  # Optimization tolerance
        include_comparison = options.get('includeComparison', False)  # Include frontier comparison
        
        logger.info(f"[{request_id}] Using Enhanced Cost-Spec method: {method}, feature_set={feature_set}, fee_column={fee_column}")
        
        # Step 3: Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_path = config.cs_raw_dir / f"raw_data_{timestamp}.json"
        
        with open(raw_data_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
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
        processed_data_path = config.cs_processed_dir / f"processed_data_{timestamp}.csv"
        
        processed_df.to_csv(processed_data_path, index=False, encoding='utf-8')
        
        # Step 6: Apply Enhanced Cost-Spec ranking method with options
        df_ranked = rank_plans_by_cs_enhanced(
            processed_df,
            method=method,
            feature_set=feature_set,
            fee_column=fee_column,
            tolerance=tolerance,
            include_comparison=include_comparison
        )
        
        logger.info(f"[{request_id}] Ranked DataFrame shape: {df_ranked.shape}")
        
        # Store the results in global state for access by other endpoints
        logger.info(f"Storing {len(df_ranked)} plans in global state for HTML report")
        
        # Log the top 10 plans by rank to verify all are included
        rank_column = 'rank' if 'rank' in df_ranked.columns else 'rank_number'
        if rank_column in df_ranked.columns:
            top_10_by_rank = df_ranked.sort_values(rank_column).head(10)
            logger.info(f"Top 10 plans by rank to be stored:\n{top_10_by_rank[['plan_name', 'CS', rank_column]].to_string()}")
        
        # Store the complete dataframe
        config.df_with_rankings = df_ranked.copy()
        
        # Store the global dataframe for the root endpoint
        global df_with_rankings
        df_with_rankings = df_ranked.copy()
        
        # Step 7: Prepare response data first before HTML generation
        # First, ensure all float values are JSON-serializable
        # Replace inf, -inf, and NaN with appropriate values
        df_for_response = df_ranked.copy()
        df_for_response = df_for_response.replace([np.inf, -np.inf], np.finfo(np.float64).max)
        df_for_response = df_for_response.replace(np.nan, 0)
        
        # Convert all numpy types to Python native types for JSON serialization
        def convert_numpy_types(obj):
            """Recursively convert numpy types to Python native types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Apply conversion to all DataFrame columns
        for col in df_for_response.columns:
            if df_for_response[col].dtype.kind in ['i', 'u']:  # integer types
                df_for_response[col] = df_for_response[col].astype(int)
            elif df_for_response[col].dtype.kind == 'f':  # float types
                df_for_response[col] = df_for_response[col].astype(float)
            elif df_for_response[col].dtype.kind == 'b':  # boolean types
                df_for_response[col] = df_for_response[col].astype(bool)
        
        # Extract cost structure if available (linear decomposition)
        cost_structure = {}
        logger.info(f"[{request_id}] Checking for cost_structure in DataFrame attrs...")
        logger.info(f"[{request_id}] DataFrame has attrs: {hasattr(df_ranked, 'attrs')}")
        if hasattr(df_ranked, 'attrs'):
            logger.info(f"[{request_id}] DataFrame attrs keys: {list(df_ranked.attrs.keys())}")
            if 'cost_structure' in df_ranked.attrs:
                cost_structure = df_ranked.attrs['cost_structure']
                logger.info(f"[{request_id}] Cost structure discovered: {cost_structure}")
            else:
                logger.info(f"[{request_id}] No cost_structure found in attrs")
        else:
            logger.info(f"[{request_id}] DataFrame has no attrs")
        
        # Create all_ranked_plans for the response
        columns_to_include = ["id", "plan_name", "mvno", "fee", "original_fee", "rank", "B", "CS"]
        
        # Include comparison columns if available
        if include_comparison and 'B_frontier' in df_for_response.columns:
            columns_to_include.extend(["B_frontier", "CS_frontier"])
        
        available_columns = [col for col in columns_to_include if col in df_for_response.columns]
        
        # Sort by CS ratio (descending) and add value_ratio field for compatibility
        all_ranked_plans = df_for_response.sort_values("CS", ascending=False)[available_columns].to_dict(orient="records")
        
        # Ensure each plan has a value_ratio field (required for edge function DB upsert)
        for plan in all_ranked_plans:
            # Add value_ratio field using CS for compatibility
            plan["value_ratio"] = plan.get("CS", 0)
        
        logger.info(f"[{request_id}] Prepared all_ranked_plans with {len(all_ranked_plans)} plans")
        
        # Get top 10 plans for the response
        top_10_plans = all_ranked_plans[:10] if len(all_ranked_plans) >= 10 else all_ranked_plans
        
        # Calculate timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Set up HTML report info
        timestamp_now = datetime.now()
        method_suffix = "decomp" if method == "linear_decomposition" else "frontier"
        report_filename = f"cs_ranking_{method_suffix}_{timestamp_now.strftime('%Y%m%d_%H%M%S')}.html"
        report_path = config.cs_report_dir / report_filename
        
        # Prepare response and convert all numpy types
        response = {
            "request_id": request_id,
            "message": f"Data processing complete using Enhanced Cost-Spec method ({method})",
            "status": "success",
            "processing_time_seconds": round(processing_time, 4),
            "options": {
                "method": method,
                "featureSet": feature_set,
                "feeColumn": fee_column,
                "tolerance": tolerance,
                "includeComparison": include_comparison
            },
            "ranking_method": "enhanced_cs",
            "cost_structure": convert_numpy_types(cost_structure),
            "results": {
                "raw_data_path": str(raw_data_path),
                "processed_data_path": str(processed_data_path),
                "report_path": str(report_path),
                "report_url": f"/reports/cs_reports/{report_filename}"
            },
            "top_10_plans": convert_numpy_types(top_10_plans),
            "all_ranked_plans": convert_numpy_types(all_ranked_plans)
        }
        
        # Start chart calculation in the background (non-blocking)
        import asyncio
        try:
            # Create background task for chart calculation
            loop = asyncio.get_event_loop()
            chart_calculation_task = loop.create_task(
                calculate_charts_async(df_ranked, method, cost_structure, request_id)
            )
            logger.info(f"[{request_id}] Started background chart calculation task")
            response["chart_status"] = "calculating"
        except Exception as e:
            logger.error(f"[{request_id}] Failed to start background chart calculation: {str(e)}")
            response["chart_status"] = "failed_to_start"
            response["chart_error"] = str(e)
        
        return response
    except Exception as e:
        logger.exception(f"[{request_id}] Error in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/chart-status")
def get_chart_status():
    """
    Get the current status of chart calculation.
    """
    global chart_calculation_status
    
    status = chart_calculation_status.copy()
    
    # Add human-readable status
    if status['is_calculating']:
        status['status_text'] = "Calculating charts..."
    elif status['error_message']:
        status['status_text'] = f"Error: {status['error_message']}"
    elif status['last_calculation_time']:
        status['status_text'] = "Charts ready"
    else:
        status['status_text'] = "No calculations performed yet"
    
    return status

@app.get("/status", response_class=HTMLResponse)
def get_status_page():
    """
    Get a simple HTML page showing the current system status.
    """
    global chart_calculation_status
    
    status = chart_calculation_status.copy()
    
    if status['is_calculating']:
        status_icon = "⚙️"
        status_text = f"Calculating charts... {status['calculation_progress']}%"
        status_color = "#007bff"
    elif status['error_message']:
        status_icon = "❌"
        status_text = f"Error: {status['error_message']}"
        status_color = "#dc3545"
    elif status['last_calculation_time']:
        status_icon = "✅"
        status_text = "Charts ready"
        status_color = "#28a745"
    else:
        status_icon = "⏳"
        status_text = "No calculations performed yet"
        status_color = "#6c757d"
    
    html = f"""
    <html>
        <head>
            <title>System Status - Moyo Plan Rankings</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; background-color: #f8f9fa; }}
                .status-card {{ max-width: 400px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .status-icon {{ font-size: 64px; margin: 20px 0; }}
                .status-text {{ font-size: 18px; color: {status_color}; margin: 20px 0; }}
                .btn {{ background-color: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; text-decoration: none; display: inline-block; }}
                .btn:hover {{ background-color: #0056b3; }}
            </style>
        </head>
        <body>
            <div class="status-card">
                <h1>System Status</h1>
                <div class="status-icon">{status_icon}</div>
                <div class="status-text">{status_text}</div>
                <a href="/" class="btn">🏠 Home</a>
                <a href="/chart-status" class="btn">📊 API Status</a>
                <button onclick="window.location.reload()" class="btn">🔄 Refresh</button>
            </div>
        </body>
    </html>
    """
    
    return HTMLResponse(content=html)

@app.post("/test")
def test(request: dict = Body(...)):
    """Simple echo endpoint for testing (returns the provided data)."""
    return {"received": request}

# The /upload-csv endpoint has been removed
# All functionality is now consolidated in the /process endpoint