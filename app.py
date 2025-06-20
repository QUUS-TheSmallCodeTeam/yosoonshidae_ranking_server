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
from typing import Optional, Union, List, Any
import os
import psutil
import logging

# Import configuration
from modules.config import config, logger
from modules import data_storage
from modules.performance import profiler, cache_manager, get_system_info

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

# Validation imports removed

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
    # Load existing chart data if available
    try:
        from modules import data_storage
        _, _, _, existing_charts = data_storage.load_rankings_data()
        if existing_charts:
            # Update chart_calculation_statuses with existing data
            for chart_type in chart_types:
                if chart_type in existing_charts and existing_charts[chart_type] is not None:
                    update_chart_status(chart_type, status='ready', progress=100, data=existing_charts[chart_type])
                    logger.info(f"Loaded existing chart data for {chart_type}")
            logger.info(f"Successfully loaded existing charts: {[k for k, v in existing_charts.items() if v is not None]}")
        else:
            logger.info("No existing chart data found")
    except Exception as e:
        logger.error(f"Failed to load existing chart data: {e}")
    
    # Wait a moment for the server to fully start
    def delayed_start():
        time.sleep(3)  # Wait 3 seconds for server to be ready
        start_log_monitoring()
    
    # Start in background thread
    threading.Thread(target=delayed_start, daemon=True).start()
    logger.info("FastAPI app started - log monitoring will start in 3 seconds")

# Global variables for storing data (using config module instead of global variables)
latest_logical_test_results_cache = None  # For storing logical test results
# Validation cache removed

# Individual chart calculation statuses with threading
from concurrent.futures import ThreadPoolExecutor
import threading

chart_types = [
    'feature_frontier',
    # 'marginal_cost_frontier',  # 복잡한 전체 데이터셋 분석 - 너무 무거워서 비활성화
    # 'multi_frontier_analysis',  # 다중 프론티어 분석 - 너무 무거워서 비활성화  
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

# Validation status tracking removed

chart_calculation_lock = threading.Lock()
chart_executor = ThreadPoolExecutor(max_workers=len(chart_types), thread_name_prefix="chart_worker")
# Validation executor removed

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

# Validation status update function removed

def calculate_and_save_charts_background(df_ranked, method, cost_structure, request_id):
    """
    Calculate charts in background and save to files when complete
    """
    try:
        logger.info(f"[{request_id}] Starting background chart calculations...")
        
        # Start calculating each chart type and update their statuses
        chart_types_list = ['feature_frontier', 'plan_efficiency']
        
        for i, chart_type in enumerate(chart_types_list):
            try:
                # Update status to calculating
                update_chart_status(chart_type, status='calculating', progress=0)
                logger.info(f"[{request_id}] Starting {chart_type} calculation...")
                
                # Calculate this specific chart
                if chart_type == 'feature_frontier':
                    from modules.report_charts import prepare_feature_frontier_data
                    core_features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
                    chart_data = prepare_feature_frontier_data(df_ranked, core_features)
                elif chart_type == 'plan_efficiency':
                    from modules.report_html import prepare_plan_efficiency_data
                    chart_data = prepare_plan_efficiency_data(df_ranked, method)
                else:
                    chart_data = None
                
                # Update progress
                progress = int((i + 1) / len(chart_types_list) * 100)
                update_chart_status(chart_type, status='ready', progress=progress, data=chart_data)
                logger.info(f"[{request_id}] ✓ {chart_type} calculation completed")
                
            except Exception as e:
                logger.error(f"[{request_id}] Failed to calculate {chart_type}: {str(e)}")
                update_chart_status(chart_type, status='error', error=str(e))
        
        # Calculate all charts data for file storage
        charts_data = data_storage.calculate_and_save_charts(df_ranked, method, cost_structure)
        
        # Load existing data and update with charts
        existing_df, existing_cost_structure, existing_method, _ = data_storage.load_rankings_data()
        
        if existing_df is not None:
            # Save updated data with charts
            save_success = data_storage.save_rankings_data(existing_df, existing_cost_structure, existing_method, charts_data)
            if save_success:
                logger.info(f"[{request_id}] Successfully saved charts data to files")
            else:
                logger.error(f"[{request_id}] Failed to save charts data to files")
        else:
            logger.error(f"[{request_id}] No existing data found to update with charts")
            
    except Exception as e:
        logger.error(f"[{request_id}] Error in background chart calculation: {str(e)}")
        # Update all chart statuses to error
        for chart_type in ['feature_frontier', 'plan_efficiency']:
            update_chart_status(chart_type, status='error', error=str(e))

# Validation system removed - only chart calculation in background

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
def root(basic: bool = False):
    """
    Root endpoint that always shows the full HTML report.
    Individual chart sections show loading status if not ready.
    """
    global chart_calculation_statuses
    
    try:
        # Load data from files instead of using config module
        df_to_use, cost_structure, method, charts_data = data_storage.load_rankings_data()
        
        logger.info(f"Root endpoint - loaded data is None: {df_to_use is None}")
        if df_to_use is not None:
            logger.info(f"Root endpoint - loaded data shape: {df_to_use.shape}")
        
        # Use loaded data or defaults
        if method is None:
            method = 'fixed_rates'
        if cost_structure is None:
            cost_structure = {}
        
        method_name = "Fixed Rates"
        title = f"Enhanced Cost-Spec Rankings ({method_name})"
        
        # Use pre-calculated charts data instead of calculating live
        # Charts are now calculated synchronously in /process endpoint
        
        # Use existing data and include pre-calculated charts
        html_report = generate_html_report(
            df_to_use, 
            datetime.now(), 
            is_cs=True, 
            title=title,
            method=method,
            cost_structure=cost_structure,
            charts_data=charts_data  # Pass pre-calculated charts data
        )
        return HTMLResponse(content=html_report)
            
    except Exception as e:
        logger.error(f"Error in root endpoint: {str(e)}")
        return HTMLResponse(content=f"<h1>Error: {str(e)}</h1>", status_code=500)

@app.post("/process")
def process_data(data: Any = Body(...)):
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
        request_json = data
        
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
        
        # Step 6: Apply Enhanced Cost-Spec ranking method (IMMEDIATELY - no validation delay)
        logger.info(f"[{request_id}] Starting IMMEDIATE ranking calculation...")
        
        df_ranked = rank_plans_by_cs_enhanced(
            processed_df,
            method=method,
            feature_set=feature_set,
            fee_column=fee_column,
            tolerance=tolerance,
            include_comparison=include_comparison
        )
        
        logger.info(f"[{request_id}] ✓ Ranking calculation completed immediately")
        logger.info(f"[{request_id}] Ranked DataFrame shape: {df_ranked.shape}")
        
        # Store the results in global state for access by other endpoints
        logger.info(f"Storing {len(df_ranked)} plans in global state for HTML report")
        
        # Log the top 10 plans by rank to verify all are included
        rank_column = 'rank' if 'rank' in df_ranked.columns else 'rank_number'
        if rank_column in df_ranked.columns:
            top_10_by_rank = df_ranked.sort_values(rank_column).head(10)
            logger.info(f"Top 10 plans by rank to be stored:\n{top_10_by_rank[['plan_name', 'CS', rank_column]].to_string()}")
        
        # Extract cost structure from DataFrame attrs
        cost_structure = {}
        if hasattr(df_ranked, 'attrs') and 'cost_structure' in df_ranked.attrs:
            cost_structure = df_ranked.attrs['cost_structure']
            logger.info(f"[{request_id}] Cost structure found in DataFrame attrs: {cost_structure}")
        elif hasattr(df_ranked, 'attrs') and 'multi_frontier_breakdown' in df_ranked.attrs:
            cost_structure = df_ranked.attrs['multi_frontier_breakdown']
            logger.info(f"[{request_id}] Multi-frontier breakdown found in DataFrame attrs: {cost_structure}")
        
        # Save ranking data immediately (without charts - they will be calculated in background)
        save_success = data_storage.save_rankings_data(df_ranked, cost_structure, method)
        if save_success:
            logger.info(f"[{request_id}] Successfully saved rankings data to files")
        else:
            logger.error(f"[{request_id}] Failed to save rankings data to files")
        
        # Also store in config for backward compatibility (will be removed later)
        config.df_with_rankings = df_ranked.copy()
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
        method_suffix = method if method in ["fixed_rates", "multi_frontier"] else "frontier"
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
            "all_ranked_plans": convert_numpy_types(all_ranked_plans),
            "chart_status": "calculating"  # Charts will be calculated in background
        }
        
        # Start background chart calculation AFTER response is prepared
        try:
            # Start chart calculation in background and save to files when complete
            chart_calculation_task = chart_executor.submit(
                calculate_and_save_charts_background,
                df_ranked.copy(),
                method,
                cost_structure,
                request_id
            )
            logger.info(f"[{request_id}] Started background chart calculation task")
            
        except Exception as e:
            logger.error(f"[{request_id}] Failed to start background chart task: {str(e)}")
            response["chart_status"] = "failed_to_start"
            response["background_error"] = str(e)
        
        return response
    except Exception as e:
        logger.exception(f"[{request_id}] Error in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

# Validation endpoints removed

@app.get("/chart-status")
def get_chart_status():
    """
    Get the current status of all individual chart calculations.
    """
    logger.info("get_chart_status function called")
    try:
        logger.info(f"chart_calculation_lock: {chart_calculation_lock}")
        logger.info(f"chart_calculation_statuses keys: {list(chart_calculation_statuses.keys())}")
        
        with chart_calculation_lock:
            statuses = chart_calculation_statuses.copy()
        
        logger.info(f"Copied statuses: {[k for k in statuses.keys()]}")
        
        # Convert datetime objects to strings for JSON serialization
        for chart_type, status in statuses.items():
            logger.info(f"Processing {chart_type}: {status}")
            if status.get('last_calculation_time') and hasattr(status['last_calculation_time'], 'isoformat'):
                statuses[chart_type]['last_calculation_time'] = status['last_calculation_time'].isoformat()
        
        # Add summary information
        total_charts = len(chart_types)
        ready_charts = sum(1 for status in statuses.values() if status['status'] == 'ready')
        calculating_charts = sum(1 for status in statuses.values() if status['status'] == 'calculating')
        error_charts = sum(1 for status in statuses.values() if status['status'] == 'error')
        
        overall_progress = (ready_charts / total_charts) * 100 if total_charts > 0 else 0
        
        result = {
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
        
        logger.info(f"Returning result: {type(result)}")
        return result
        
    except Exception as e:
        logger.error(f"Error in get_chart_status: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "individual_charts": {},
            "summary": {
                "total_charts": 0,
                "ready_charts": 0,
                "calculating_charts": 0,
                "error_charts": 0,
                "overall_progress": 0,
                "all_ready": False,
                "any_calculating": False,
                "any_errors": True
            },
            "error": str(e)
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
    
    if any(status['status'] == 'calculating' for status in status.values()):
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
    
    html = """
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
            <script>
                function checkSystemStatus() {{
                    console.log('Checking system status...');
                    fetch('/chart-status')
                        .then(response => response.json())
                        .then(data => {{
                            console.log('System status:', data);
                            const summary = data.summary;
                            let message = '시스템 상태:\\n';
                            message += '전체 차트: ' + summary.total_charts + '개\\n';
                            message += '완료된 차트: ' + summary.ready_charts + '개\\n';
                            message += '계산 중인 차트: ' + summary.calculating_charts + '개\\n';
                            message += '오류 차트: ' + summary.error_charts + '개\\n';
                            message += '전체 진행률: ' + summary.overall_progress + '%';
                            
                            if (summary.any_calculating) {{
                                message += '\\n\\n차트 계산이 진행 중입니다.';
                            }} else if (summary.all_ready) {{
                                message += '\\n\\n모든 차트가 준비되었습니다!';
                            }}
                            
                            alert(message);
                        }})
                        .catch(error => {{
                            console.error('Error checking status:', error);
                            alert('상태 확인 중 오류가 발생했습니다.');
                        }});
                }}
            </script>
        </head>
        <body>
            <div class="status-card">
                <h1>System Status</h1>
                <div class="status-icon">{status_icon}</div>
                <div class="status-text">{status_text}</div>
                <a href="/" class="btn">🏠 Home</a>
                <a href="/chart-status" class="btn">📊 API Status</a>
                <button onclick="checkSystemStatus()" class="btn">🔄 상태 확인</button>
            </div>
        </body>
    </html>
    """.format(status_color=status_color, status_icon=status_icon, status_text=status_text)
    
    return HTMLResponse(content=html)

@app.post("/test")
def test(request: dict = Body(...)):
    """Simple echo endpoint for testing (returns the provided data)."""
    return {"received": request}

@app.get("/test-reload")
def test_reload():
    """Test endpoint to check if server is reloading code changes."""
    return {"message": "Server is working and reloading changes!", "timestamp": "2025-06-19 10:07:00"}

@app.get("/debug-global")
def debug_global_state():
    """Debug endpoint to check file-based data storage state"""
    # Check file-based storage
    file_info = data_storage.get_data_info()
    
    # Also check config for comparison
    config_info = {
        "config_df_with_rankings_is_none": config.df_with_rankings is None,
        "config_df_with_rankings_type": str(type(config.df_with_rankings)),
        "config_df_with_rankings_shape": config.df_with_rankings.shape if config.df_with_rankings is not None else None,
        "config_df_with_rankings_columns": list(config.df_with_rankings.columns) if config.df_with_rankings is not None else None,
        "has_method_attr": hasattr(config.df_with_rankings, 'method') if config.df_with_rankings is not None else False,
        "has_cost_structure_attr": hasattr(config.df_with_rankings, 'cost_structure') if config.df_with_rankings is not None else False
    }
    
    return {
        "file_storage": file_info,
        "config_storage": config_info,
        "storage_method": "file_based_with_config_backup"
    }

# Performance monitoring endpoints
@app.get("/performance")
def get_performance_stats():
    """Get performance monitoring statistics"""
    try:
        return {
            "status": "success",
            "system_info": get_system_info(),
            "profiler_summary": profiler.get_summary(),
            "cache_stats": cache_manager.get_stats(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Performance endpoint error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/performance/clear-cache")
def clear_performance_cache(pattern: Optional[str] = None):
    """Clear performance cache"""
    try:
        cache_manager.clear(pattern)
        return {
            "status": "success",
            "message": f"Cache cleared{f' (pattern: {pattern})' if pattern else ''}",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache clear error: {str(e)}")
        return {"status": "error", "message": str(e)}

@app.post("/performance/save-report")
def save_performance_report():
    """Save performance report to file"""
    try:
        profiler.save_report()
        return {
            "status": "success",
            "message": "Performance report saved",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Performance report save error: {str(e)}")
        return {"status": "error", "message": str(e)}

# The /upload-csv endpoint has been removed
# All functionality is now consolidated in the /process endpoint