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

# Individual chart calculation statuses with threading
from concurrent.futures import ThreadPoolExecutor
import threading

chart_types = [
    'feature_frontier',
    'marginal_cost_frontier', 
    'multi_frontier_analysis',
    'plan_efficiency'
]

chart_calculation_statuses = {
    chart_type: {
        'is_calculating': False,
        'last_calculation_time': None,
        'calculation_progress': 0,
        'error_message': None,
        'chart_data': None,
        'status': 'idle'  # idle, calculating, ready, error
    } for chart_type in chart_types
}

chart_calculation_lock = threading.Lock()
chart_executor = ThreadPoolExecutor(max_workers=len(chart_types), thread_name_prefix="chart_worker")

def update_chart_status(chart_type, status=None, progress=None, error=None, data=None):
    """Thread-safe update of chart status"""
    with chart_calculation_lock:
        if status is not None:
            chart_calculation_statuses[chart_type]['status'] = status
            chart_calculation_statuses[chart_type]['is_calculating'] = (status == 'calculating')
        if progress is not None:
            chart_calculation_statuses[chart_type]['calculation_progress'] = progress
        if error is not None:
            chart_calculation_statuses[chart_type]['error_message'] = str(error)
            chart_calculation_statuses[chart_type]['status'] = 'error'
        if data is not None:
            chart_calculation_statuses[chart_type]['chart_data'] = data
        if status == 'ready':
            chart_calculation_statuses[chart_type]['last_calculation_time'] = datetime.now()
            chart_calculation_statuses[chart_type]['calculation_progress'] = 100

def calculate_single_chart(chart_type, df_ranked, method, cost_structure, request_id):
    """Calculate a single chart type in a separate thread"""
    try:
        update_chart_status(chart_type, status='calculating', progress=10)
        logger.info(f"[{request_id}] Started calculating {chart_type} chart")
        
        chart_data = None
        
        if chart_type == 'feature_frontier':
            update_chart_status(chart_type, progress=30)
            from modules.report_charts import prepare_feature_frontier_data
            core_features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
            chart_data = prepare_feature_frontier_data(df_ranked, core_features)
            
        elif chart_type == 'marginal_cost_frontier':
            update_chart_status(chart_type, progress=30)
            from modules.report_charts import prepare_marginal_cost_frontier_data, prepare_granular_marginal_cost_frontier_data
            if hasattr(df_ranked, 'attrs') and 'multi_frontier_breakdown' in df_ranked.attrs:
                multi_frontier_breakdown = df_ranked.attrs['multi_frontier_breakdown']
                core_features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
                
                # Use granular method with unlimited intercepts
                chart_data = prepare_granular_marginal_cost_frontier_data(df_ranked, multi_frontier_breakdown, core_features)
                logger.info(f"[{request_id}] Using GRANULAR marginal cost frontier with unlimited intercepts")
            else:
                logger.warning(f"[{request_id}] No multi_frontier_breakdown found for {chart_type}")
                
        elif chart_type == 'multi_frontier_analysis':
            update_chart_status(chart_type, progress=30)
            from modules.report_charts import prepare_multi_frontier_chart_data
            if hasattr(df_ranked, 'attrs') and 'multi_frontier_breakdown' in df_ranked.attrs:
                multi_frontier_breakdown = df_ranked.attrs['multi_frontier_breakdown']
                chart_data = prepare_multi_frontier_chart_data(df_ranked, multi_frontier_breakdown)
            else:
                logger.warning(f"[{request_id}] No multi_frontier_breakdown found for {chart_type}")
                
        # Linear decomposition chart removed per user request
            
        elif chart_type == 'plan_efficiency':
            update_chart_status(chart_type, progress=30)
            from modules.report_html import prepare_plan_efficiency_data
            chart_data = prepare_plan_efficiency_data(df_ranked, method)
        
        update_chart_status(chart_type, progress=70)
        
        # Convert numpy types to ensure JSON serialization
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
        
        if chart_data:
            chart_data = convert_numpy_types(chart_data)
            
        update_chart_status(chart_type, status='ready', progress=100, data=chart_data)
        logger.info(f"[{request_id}] Successfully calculated {chart_type} chart")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error calculating {chart_type} chart: {str(e)}")
        update_chart_status(chart_type, error=str(e), progress=0)

async def calculate_charts_async(df_ranked, method, cost_structure, request_id):
    """
    Calculate all charts asynchronously using thread pool.
    Each chart type is calculated in parallel.
    """
    try:
        logger.info(f"[{request_id}] Starting parallel chart calculation...")
        
        # Reset all chart statuses
        for chart_type in chart_types:
            update_chart_status(chart_type, status='idle', progress=0, error=None, data=None)
        
        # Submit all chart calculations to thread pool
        futures = []
        for chart_type in chart_types:
            future = chart_executor.submit(
                calculate_single_chart, 
                chart_type, 
                df_ranked, 
                method, 
                cost_structure, 
                request_id
            )
            futures.append((chart_type, future))
        
        # Wait for all calculations to complete (non-blocking for API response)
        import asyncio
        loop = asyncio.get_event_loop()
        
        def wait_for_charts():
            for chart_type, future in futures:
                try:
                    future.result(timeout=300)  # 5 minute timeout per chart
                except Exception as e:
                    logger.error(f"[{request_id}] Chart {chart_type} failed: {str(e)}")
                    update_chart_status(chart_type, error=str(e))
        
        # Run in background without blocking the API response
        await loop.run_in_executor(None, wait_for_charts)
        
        logger.info(f"[{request_id}] All chart calculations completed")
        
    except Exception as e:
        logger.error(f"[{request_id}] Error in chart calculation orchestration: {str(e)}")
        # Mark all charts as error if orchestration fails
        for chart_type in chart_types:
            update_chart_status(chart_type, error=str(e))

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
    global df_with_rankings, chart_calculation_statuses
    
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
        
        # Check chart calculation status - show individual chart progress
        with chart_calculation_lock:
            statuses = chart_calculation_statuses.copy()
        
        total_charts = len(chart_types)
        ready_charts = sum(1 for status in statuses.values() if status['status'] == 'ready')
        calculating_charts = sum(1 for status in statuses.values() if status['status'] == 'calculating')
        
        if calculating_charts > 0 or ready_charts < total_charts:
            # Generate individual chart status HTML
            chart_status_items = []
            for chart_type, status in statuses.items():
                chart_name = {
                    'feature_frontier': 'Feature Frontier Charts',
                    'marginal_cost_frontier': 'Marginal Cost Frontier Charts',
                    'multi_frontier_analysis': 'Multi-Frontier Analysis',
                    # 'linear_decomposition': 'Linear Decomposition Charts', # Removed per user request
                    'plan_efficiency': 'Plan Efficiency Matrix'
                }.get(chart_type, chart_type.replace('_', ' ').title())
                
                if status['status'] == 'calculating':
                    icon = '⚙️'
                    text = f"Calculating... {status['calculation_progress']}%"
                    color = '#007bff'
                elif status['status'] == 'ready':
                    icon = '✅'
                    text = 'Ready'
                    color = '#28a745'
                elif status['status'] == 'error':
                    icon = '❌'
                    text = f"Error: {status['error_message'][:50]}..."
                    color = '#dc3545'
                else:
                    icon = '⏳'
                    text = 'Waiting...'
                    color = '#6c757d'
                
                chart_status_items.append(f"""
                    <div class="chart-status-item" style="border-left: 4px solid {color};">
                        <span class="chart-icon">{icon}</span>
                        <span class="chart-name">{chart_name}</span>
                        <span class="chart-status" style="color: {color};">{text}</span>
                    </div>
                """)
            
            overall_progress = (ready_charts / total_charts) * 100 if total_charts > 0 else 0
            
            progress_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Charts Processing...</title>
                <meta http-equiv="refresh" content="5">
                <style>
                    body {{ 
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
                        margin: 0; padding: 20px; background: #f8f9fa; 
                    }}
                    .container {{ max-width: 800px; margin: 0 auto; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .progress-bar {{ 
                        width: 100%; height: 20px; background: #e9ecef; 
                        border-radius: 10px; overflow: hidden; margin: 20px 0; 
                    }}
                    .progress-fill {{ 
                        height: 100%; background: linear-gradient(90deg, #007bff, #28a745); 
                        width: {overall_progress}%; transition: width 0.5s ease; 
                    }}
                    .chart-status-grid {{ 
                        display: grid; gap: 15px; margin: 30px 0; 
                    }}
                    .chart-status-item {{ 
                        background: white; padding: 20px; border-radius: 8px; 
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; 
                        align-items: center; gap: 15px; 
                    }}
                    .chart-icon {{ font-size: 24px; }}
                    .chart-name {{ flex: 1; font-weight: 600; }}
                    .chart-status {{ font-size: 14px; }}
                    .actions {{ text-align: center; margin-top: 30px; }}
                    .btn {{ 
                        display: inline-block; padding: 10px 20px; 
                        background: #007bff; color: white; text-decoration: none; 
                        border-radius: 5px; margin: 0 10px; 
                    }}
                    .btn:hover {{ background: #0056b3; }}
                    .btn-secondary {{ background: #6c757d; }}
                    .btn-secondary:hover {{ background: #5a6268; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Chart Generation in Progress</h1>
                        <p>Processing multi-threaded chart calculations...</p>
                        <div class="progress-bar">
                            <div class="progress-fill"></div>
                        </div>
                        <p><strong>{ready_charts}/{total_charts} charts ready</strong> ({overall_progress:.1f}%)</p>
                    </div>
                    
                    <div class="chart-status-grid">
                        {''.join(chart_status_items)}
                    </div>
                    
                    <div class="actions">
                        <a href="/" class="btn">🔄 Refresh Status</a>
                        <a href="/?basic=true" class="btn btn-secondary">📄 View Basic Report</a>
                        <a href="/chart-status" class="btn btn-secondary">🔍 API Status</a>
                    </div>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=progress_html)
        
        elif any(status['error_message'] for status in chart_calculation_statuses.values()):
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
                    <div class="error-msg">{', '.join(status['error_message'] for status in chart_calculation_statuses.values() if status['error_message'])}</div>
                    <p><a href="/">Try again</a> | <a href="/?basic=true">View basic report</a></p>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=error_html)
        
        else:
            # Charts are ready - generate fresh HTML report with FRESH coefficient calculation
            method = getattr(df_with_rankings, 'method', 'fixed_rates')  # Use fixed_rates for fresh calculation
            
            # Force fresh coefficient calculation by re-running the ranking with fixed_rates method
            from modules import rank_plans_by_cs_enhanced
            
            # Extract the original DataFrame without rankings to recalculate coefficients
            df_for_recalc = df_with_rankings.copy()
            
            # Remove any existing ranking/coefficient columns to force fresh calculation
            columns_to_remove = ['rank', 'rank_number', 'B', 'CS', 'coefficient_breakdown']
            for col in columns_to_remove:
                if col in df_for_recalc.columns:
                    df_for_recalc = df_for_recalc.drop(columns=[col])
            
            # Force fresh coefficient calculation with fixed_rates method (no caching)
            df_fresh = rank_plans_by_cs_enhanced(
                df_for_recalc,
                method='fixed_rates',  # Always use fixed_rates for consistent, fresh results
                feature_set='basic',
                fee_column='fee',
                tolerance=500,
                include_comparison=False
            )
            
            # Update the cost structure from fresh calculation
            cost_structure = getattr(df_fresh, 'attrs', {}).get('cost_structure', None)
            
            method_name = "Fixed Rates (Fresh Calculation)"
            title = f"Enhanced Cost-Spec Rankings ({method_name})"
            
            html_report = generate_html_report(
                df_fresh, 
                datetime.now(), 
                is_cs=True, 
                title=title,
                method='fixed_rates',
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
        method = options.get('method', 'fixed_rates')  # Default to fixed rates method for ranking table
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
        
        # Extract cost structure from DataFrame attrs and store as object attributes
        cost_structure = {}
        if hasattr(df_ranked, 'attrs') and 'cost_structure' in df_ranked.attrs:
            cost_structure = df_ranked.attrs['cost_structure']
            logger.info(f"[{request_id}] Cost structure found in DataFrame attrs: {cost_structure}")
        elif hasattr(df_ranked, 'attrs') and 'multi_frontier_breakdown' in df_ranked.attrs:
            cost_structure = df_ranked.attrs['multi_frontier_breakdown']
            logger.info(f"[{request_id}] Multi-frontier breakdown found in DataFrame attrs: {cost_structure}")
        
        # Set cost_structure as object attributes for HTML report access
        df_with_rankings.cost_structure = cost_structure
        df_with_rankings.method = method
        config.df_with_rankings.cost_structure = cost_structure
        config.df_with_rankings.method = method
        
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
    Get the current status of all individual chart calculations.
    """
    with chart_calculation_lock:
        statuses = chart_calculation_statuses.copy()
    
    # Add summary information
    total_charts = len(chart_types)
    ready_charts = sum(1 for status in statuses.values() if status['status'] == 'ready')
    calculating_charts = sum(1 for status in statuses.values() if status['status'] == 'calculating')
    error_charts = sum(1 for status in statuses.values() if status['status'] == 'error')
    
    overall_progress = (ready_charts / total_charts) * 100 if total_charts > 0 else 0
    
    return {
        "individual_charts": statuses,
        "summary": {
            "total_charts": total_charts,
            "ready_charts": ready_charts,
            "calculating_charts": calculating_charts,
            "error_charts": error_charts,
            "overall_progress": round(overall_progress, 1),
            "all_ready": ready_charts == total_charts,
            "any_calculating": calculating_charts > 0,
            "any_errors": error_charts > 0
        }
    }

@app.get("/chart-status/{chart_type}")
def get_single_chart_status(chart_type: str):
    """
    Get the status of a specific chart type.
    """
    if chart_type not in chart_types:
        raise HTTPException(status_code=404, detail=f"Chart type '{chart_type}' not found")
    
    with chart_calculation_lock:
        status = chart_calculation_statuses[chart_type].copy()
    
    # Add human-readable status text
    if status['status'] == 'calculating':
        status['status_text'] = f"Calculating {chart_type} chart... {status['calculation_progress']}%"
    elif status['status'] == 'error':
        status['status_text'] = f"Error in {chart_type}: {status['error_message']}"
    elif status['status'] == 'ready':
        status['status_text'] = f"{chart_type} chart ready"
    else:
        status['status_text'] = f"{chart_type} chart idle"
    
    return status

@app.get("/chart-data/{chart_type}")
def get_chart_data(chart_type: str):
    """
    Get the calculated data for a specific chart type.
    """
    if chart_type not in chart_types:
        raise HTTPException(status_code=404, detail=f"Chart type '{chart_type}' not found")
    
    with chart_calculation_lock:
        status = chart_calculation_statuses[chart_type].copy()
    
    if status['status'] != 'ready':
        raise HTTPException(
            status_code=409, 
            detail=f"Chart '{chart_type}' is not ready. Current status: {status['status']}"
        )
    
    return {
        "chart_type": chart_type,
        "status": status['status'],
        "data": status['chart_data'],
        "last_calculation_time": status['last_calculation_time']
    }

@app.get("/status", response_class=HTMLResponse)
def get_status_page():
    """
    Get a simple HTML page showing the current system status.
    """
    global chart_calculation_statuses
    
    status = {chart_type: status.copy() for chart_type, status in chart_calculation_statuses.items()}
    
    if any(status['is_calculating'] for status in status.values()):
        status_icon = "⚙️"
        status_text = f"Calculating charts... {sum(status['calculation_progress'] for status in status.values())}%"
        status_color = "#007bff"
    elif any(status['error_message'] for status in status.values()):
        status_icon = "❌"
        status_text = f"Error: {', '.join(status['error_message'] for status in status.values() if status['error_message'])}"
        status_color = "#dc3545"
    elif any(status['last_calculation_time'] for status in status.values()):
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