from fastapi import FastAPI, HTTPException, Request, Response, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import json
import uuid
import gc
import time
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

# HTML report generation and saving is now handled by the modules.report module

# Define FastAPI endpoints
@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Serve the latest ranking HTML report if available.
    """
    # Check if we have rankings in memory first
    if config.df_with_rankings is not None:
        # Generate HTML report from the in-memory rankings
        try:
            # Check if this is a CS ranking based on column names
            is_cs = any(col for col in config.df_with_rankings.columns if col == 'CS')
            
            # Create a copy to avoid modifying the original
            df_for_html = config.df_with_rankings.copy()
            
            if is_cs:
                # Sort by CS ratio descending to get the correct order
                df_for_html = df_for_html.sort_values('CS', ascending=False)
                
                # Check if rank 1 exists for error detection
                if 'rank_number' in df_for_html.columns:
                    has_rank_one = 1 in df_for_html['rank_number'].values
                    if not has_rank_one:
                        logger.warning("No rank 1 found in original dataframe! This is unexpected.")
                
                # Extract method and cost_structure from the stored data
                method = "linear_decomposition"  # Default
                cost_structure = {}
                
                # Try to get method and cost_structure from DataFrame attrs
                if hasattr(df_for_html, 'attrs'):
                    if 'cost_structure' in df_for_html.attrs:
                        cost_structure = df_for_html.attrs['cost_structure']
                        method = "linear_decomposition"
                    else:
                        method = "frontier"
                else:
                    method = "frontier"
                
                # Generate report with CS method
                html_content = generate_html_report(
                    df_for_html, 
                    datetime.now(), 
                    is_cs=True, 
                    title="Cost-Spec Mobile Plan Rankings",
                    method=method,
                    cost_structure=cost_structure
                )
            else:
                html_content = generate_html_report(config.df_with_rankings, datetime.now())
                
            return HTMLResponse(content=html_content)
        except Exception as e:
            logger.error(f"Error generating report from in-memory rankings: {e}")
            # Fall back to looking for files
    
    # Look for the latest HTML report in all potential directories
    report_dirs = [
        config.cs_report_dir,        # CS reports
        Path("./reports"), 
        Path("/tmp/reports"), 
        Path("/tmp")
    ]
    
    html_files = []
    for reports_dir in report_dirs:
        if reports_dir.exists():
            # Look for all ranking reports
            html_files.extend(list(reports_dir.glob("*ranking_*.html")))
    
    if not html_files:
        # No reports found, return welcome message
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
                    <h2>Enhanced Cost-Spec Ratio Method</h2>
                    <p>This API offers advanced Cost-Spec analysis with two methods:</p>
                    
                    <h3>🔬 Linear Decomposition Method (Default, Recommended)</h3>
                    <ul>
                        <li><strong>Advanced Analysis</strong>: Extracts true marginal costs for individual features</li>
                        <li><strong>Fair Baselines</strong>: Eliminates double-counting artifacts</li>
                        <li><strong>Realistic Ratios</strong>: CS ratios typically 0.8-1.5x (realistic efficiency)</li>
                        <li><strong>Cost Discovery</strong>: Reveals actual marginal costs for strategic insights</li>
                        <li><strong>Mathematical Model</strong>: plan_cost = β₀ + β₁×data + β₂×voice + β₃×SMS + ...</li>
                    </ul>
                    
                    <h3>📈 Frontier-Based Method (Legacy)</h3>
                    <ul>
                        <li><strong>Traditional Approach</strong>: Identifies minimum costs for each feature level</li>
                        <li><strong>Simple Logic</strong>: Sums frontier costs to create baselines</li>
                        <li><strong>Note</strong>: May show inflated CS ratios (4-7x) due to double-counting</li>
                        <li><strong>Compatibility</strong>: Maintained for comparison and legacy support</li>
                    </ul>
                </div>
                
                <p>No ranking reports are available yet. Use the <code>/process</code> endpoint to analyze data and generate rankings.</p>
                
                <div class="method-info">
                    <h3>API Usage</h3>
                    <p>Submit plan data to generate rankings using the Enhanced Cost-Spec method:</p>
                    <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">
{
  "options": {
    "method": "linear_decomposition",  // or "frontier"
    "featureSet": "basic",
    "feeColumn": "fee",
    "tolerance": 500,
    "includeComparison": false
  },
  "data": [
    { "id": 1, "plan_name": "Plan A", "fee": 10000, ... },
    { "id": 2, "plan_name": "Plan B", "fee": 15000, ... }
  ]
}
                    </pre>
                    <p><strong>Options:</strong></p>
                    <ul>
                        <li><code>method</code>: "linear_decomposition" (default) or "frontier"</li>
                        <li><code>tolerance</code>: Optimization tolerance for linear decomposition (default: 500)</li>
                        <li><code>includeComparison</code>: Include both methods in results (default: false)</li>
                        <li><code>featureSet</code>: Feature set to use (default: "basic")</li>
                        <li><code>feeColumn</code>: Fee column for analysis (default: "fee")</li>
                    </ul>
                </div>

                <h2>Method Selection</h2>
                <div class="button-group">
                    <strong>Analysis Method:</strong><br>
                    <button id="decomp-btn" class="active" onclick="changeMethod('linear_decomposition')">Linear Decomposition (Recommended)</button>
                    <button id="frontier-btn" onclick="changeMethod('frontier')">Frontier-Based (Legacy)</button>
                </div>
                
                <div class="button-group">
                    <strong>Fee Type:</strong><br>
                    <button id="original-fee-btn" class="active" onclick="changeFeeType('original')">Original Fee</button>
                    <button id="discounted-fee-btn" onclick="changeFeeType('discounted')">Discounted Fee</button>
                </div>
                
                <p class="method-info">Note: These options will be applied when you generate a new report using the <code>/process</code> endpoint.</p>
                
                <hr>
                <h3>Endpoints</h3>
                <ul>
                    <li><code>POST /process</code>: Submit plan data to analyze using Enhanced Cost-Spec method</li>
                    <li><code>POST /test</code>: Echo back the request body (for debugging)</li>
                </ul>
                <hr>
                <p><i>Navigate to /docs for API documentation (Swagger UI).</i></p>
                
                <script>
                /* Current state */
                let currentState = {
                    method: "linear_decomposition",
                    feeType: "original"
                };
                
                /* Change method */
                function changeMethod(method) {
                    /* Update buttons */
                    document.getElementById('decomp-btn').classList.remove('active');
                    document.getElementById('frontier-btn').classList.remove('active');
                    document.getElementById(method === 'linear_decomposition' ? 'decomp-btn' : 'frontier-btn').classList.add('active');
                    
                    /* Update state */
                    currentState.method = method;
                    console.log("Method changed to: " + method);
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
                </script>
            </body>
        </html>
        """
    
    # Get the latest report by modification time
    latest_report = max(html_files, key=lambda x: x.stat().st_mtime)
    
    # Check if this is a CS report based on the filename
    is_cs_report = 'cs' in latest_report.name.lower()
    
    # If it's a file from the CS reports directory, it's definitely a CS report
    if latest_report.parent == config.cs_report_dir:
        is_cs_report = True
    
    # Read the HTML file content
    with open(latest_report, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Set the latest_report_path variable for reference
    latest_report_path = f"/reports/{latest_report.name}"
    
    # Read and return the HTML content
    try:
        # Insert additional UI controls before the closing </body> tag
        if '</body>' in html_content:
            interactive_controls = """
            <hr>
            <div style="background-color: #f8f9fa; padding: 15px; border-left: 4px solid #28a745; margin: 20px 0;">
                <h3>Cost-Spec Ratio Method</h3>
                <p>This report uses the Cost-Spec ratio method to evaluate plan value:</p>
                <ol>
                    <li>Calculate baseline costs for each feature value</li>
                    <li>Sum baseline costs to get a theoretical baseline cost (B) for each plan</li>
                    <li>Calculate Cost-Spec ratio (CS = B / fee)</li>
                    <li>Rank plans by CS ratio (higher is better)</li>
                </ol>
                
                <div style="margin-top: 20px;">
                    <h3>Ranking Options</h3>
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
                feeType: "original"
            };
            
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
                console.log("Fee type changed to: " + type);
            }
            </script>
            """
            insert_pos = html_content.find('</body>')
            html_content = html_content[:insert_pos] + interactive_controls + html_content[insert_pos:]
            
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
                    <h2>Cost-Spec Ratio Method</h2>
                    <p>This API uses the Cost-Spec method to evaluate plan value:</p>
                    <ol>
                        <li>Calculate baseline costs for each feature value</li>
                        <li>Sum baseline costs to get a theoretical baseline cost (B) for each plan</li>
                        <li>Calculate Cost-Spec ratio (CS = B / fee)</li>
                        <li>Rank plans by CS ratio (higher is better)</li>
                    </ol>
                </div>
                
                <p>Error reading report: {str(e)}</p>
                <p>Please try generating a new report using the <code>/process</code> endpoint.</p>
                
                <h2>Ranking Options</h2>
                <div class="button-group">
                    <strong>Fee Type:</strong><br>
                    <button id="original-fee-btn" class="active" onclick="changeFeeType('original')">Original Fee</button>
                    <button id="discounted-fee-btn" onclick="changeFeeType('discounted')">Discounted Fee</button>
                </div>
                
                <p class="method-info">Note: These options will be applied when you generate a new report using the <code>/process</code> endpoint.</p>
                
            <hr>
            <h3>Endpoints</h3>
            <ul>
                    <li><code>POST /process</code>: Submit plan data (JSON list) to preprocess, rank using the Cost-Spec method, and generate a report.</li>
                    <li><code>POST /test</code>: Echo back the request body (for debugging).</li>
            </ul>
            
            <script>
            /* Current state */
            let currentState = {{
                feeType: "original"
            }};
            
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
            </script>
        </body>
    </html>
    """

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
        method = options.get('method', 'linear_decomposition')  # Default to linear decomposition
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
        
        # Step 7: Prepare response data first before HTML generation
        # First, ensure all float values are JSON-serializable
        # Replace inf, -inf, and NaN with appropriate values
        df_for_response = df_ranked.copy()
        df_for_response = df_for_response.replace([np.inf, -np.inf], np.finfo(np.float64).max)
        df_for_response = df_for_response.replace(np.nan, 0)
        
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
        
        # Prepare response
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
            "cost_structure": cost_structure,
            "results": {
                "raw_data_path": str(raw_data_path),
                "processed_data_path": str(processed_data_path),
                "report_path": str(report_path),
                "report_url": f"/reports/cs_reports/{report_filename}"
            },
            "top_10_plans": top_10_plans,
            "all_ranked_plans": all_ranked_plans
        }
        
        # Try to generate HTML report, but don't block API response if it fails
        try:
            # Generate HTML content with method-specific title
            method_name = "Linear Decomposition" if method == "linear_decomposition" else "Frontier-Based"
            title = f"Enhanced Cost-Spec Rankings ({method_name})"
            
            html_report = generate_html_report(
                df_ranked, 
                timestamp_now, 
                is_cs=True, 
                title=title,
                method=method,
                cost_structure=cost_structure
            )
            
            # Write HTML content to file
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_report)
            
            logger.info(f"[{request_id}] HTML report successfully generated and saved to {report_path}")
        except Exception as e:
            logger.error(f"[{request_id}] Error generating HTML report: {str(e)}")
            response["message"] = "Data processing complete, but HTML report generation failed"
            response["status"] = "partial_success"
            response["error"] = f"HTML report generation failed: {str(e)}"
            # We still return the ranking data even if HTML generation fails
        
        return response
    except Exception as e:
        logger.exception(f"[{request_id}] Error in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.post("/test")
def test(request: dict = Body(...)):
    """Simple echo endpoint for testing (returns the provided data)."""
    return {"received": request}

# The /upload-csv endpoint has been removed
# All functionality is now consolidated in the /process endpoint

# Run the application
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with Uvicorn...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)