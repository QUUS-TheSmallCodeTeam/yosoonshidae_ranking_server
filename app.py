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
from scipy.stats import spearmanr
import os

# Import configuration
from modules.config import config, logger

# Import necessary modules
from modules.data import load_data_from_json
from modules.preprocess import prepare_features
from modules.utils import ensure_directories, save_raw_data, save_processed_data
from modules.ranking import calculate_rankings_with_ties
from modules.report import generate_html_report
from modules.cost_spec import calculate_cs_ratio, rank_plans_by_cs
from modules.models import get_basic_feature_list
from modules.spearman import calculate_rankings_with_spearman
from fastapi import UploadFile, File

# Initialize FastAPI
app = FastAPI(title="Moyo Plan Ranking Model Server - Cost-Spec Method")

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

# Spearman ranking calculation is now imported from the modules.spearman module
# calculate_rankings_with_ties function is now imported from the modules.ranking module

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
            
            # Log the dataframe info before generating the report
            logger.info(f"Generating report from in-memory rankings with {len(config.df_with_rankings)} plans")
            
            # Create a copy to avoid modifying the original
            df_for_html = config.df_with_rankings.copy()
            
            if is_cs:
                # Sort by CS ratio descending to get the correct order
                df_for_html = df_for_html.sort_values('CS', ascending=False)
                
                # Log some information for debugging
                if 'rank_number' in df_for_html.columns:
                    unique_ranks = sorted(df_for_html['rank_number'].unique())
                    logger.info(f"Unique CS ranks in dataframe before HTML generation: {unique_ranks[:10]}")
                    
                    # Check if rank 1 exists
                    has_rank_one = 1 in df_for_html['rank_number'].values
                    if not has_rank_one:
                        logger.warning("No rank 1 found in original dataframe! This is unexpected.")
                    
                    # Count plans per rank
                    rank_counts = df_for_html['rank_number'].value_counts().sort_index()
                    logger.info(f"Plans per rank before HTML generation: {rank_counts.head(10).to_dict()}")
                
                # Log top plans for debugging
                top_plans = df_for_html.sort_values('CS', ascending=False).head(5)
                logger.info(f"Top 5 plans with ranks for HTML report:\n{top_plans[['plan_name', 'CS', 'rank_number']].to_string()}")
                
                # Generate report with CS method
                logger.info("Generating CS report for main endpoint")
                html_content = generate_html_report(
                    df_for_html, 
                    datetime.now(), 
                    is_cs=True, 
                    title="Cost-Spec Mobile Plan Rankings"
                )
            else:
                # For non-CS reports (e.g. Spearman)
                logger.info("Generating Spearman report for main endpoint")
                html_content = generate_html_report(config.df_with_rankings, datetime.now())
                
            return HTMLResponse(content=html_content)
        except Exception as e:
            logger.error(f"Error generating report from in-memory rankings: {e}")
            # Fall back to looking for files
    
    # Look for the latest HTML report in all potential directories
    report_dirs = [
        config.spearman_report_dir,  # Spearman reports
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
                    <h2>Cost-Spec Ratio Method</h2>
                    <p>This API uses the Cost-Spec method to estimate plan value and rank mobile plans:</p>
                    <ol>
                        <li>Calculate baseline costs for each feature value</li>
                        <li>Sum baseline costs to get a theoretical baseline cost (B) for each plan</li>
                        <li>Calculate Cost-Spec ratio (CS = B / fee)</li>
                        <li>Rank plans by CS ratio (higher is better)</li>
                        <li>Generate comprehensive reports with detailed metrics</li>
                    </ol>
                </div>
                
                <p>No ranking reports are available yet. Use the <code>/process</code> endpoint to analyze data and generate rankings.</p>
                
                <h2>Ranking Method Details</h2>
                <div class="method-info">
                    <h3>Cost-Spec Ratio Method</h3>
                    <p>The Cost-Spec method offers several advantages for ranking mobile plans:</p>
                    <ul>
                        <li><strong>Intuitive approach</strong>: Compares plans to the theoretical minimum cost for features</li>
                        <li><strong>Value measurement</strong>: Higher CS ratio means better value for money</li>
                        <li><strong>Transparent</strong>: Easy to understand why one plan ranks higher than another</li>
                        <li><strong>Objective</strong>: Based on actual minimum costs in the market</li>
                        <li><strong>Flexible configuration</strong>: Supports different feature sets</li>
                    </ul>
                </div>

                <div class="method-info">
                    <h3>API Usage</h3>
                    <p>Submit plan data to the <code>/process</code> endpoint to generate rankings using the Cost-Spec method.</p>
                    <p>Required columns: 'fee' and basic feature columns (basic_data_clean, voice_clean, message_clean, etc.)</p>
                    <p>Use the <code>/process</code> endpoint to submit plan data in JSON format:</p>
                    <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">
{
  "options": {
    "featureSet": "basic",
    "feeColumn": "fee"
  },
  "data": [
    { "id": 1, "plan_name": "Plan A", ... },
    { "id": 2, "plan_name": "Plan B", ... }
  ]
}
                    </pre>
                </div>

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
                <hr>
                <p><i>Navigate to /docs for API documentation (Swagger UI).</i></p>
                
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
    logger.info(f"Serving latest report: {latest_report}")
    
    # Check if this is a CS report based on the filename
    is_cs_report = 'cs' in latest_report.name.lower()
    logger.info(f"Report identified as CS report: {is_cs_report}")
    
    # If it's a file from the CS reports directory, it's definitely a CS report
    if latest_report.parent == config.cs_report_dir:
        is_cs_report = True
        logger.info("Report confirmed as CS report based on directory")
    
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
    """Process plan data using the Cost-Spec ratio method."""
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
        
        # Extract options
        feature_set = options.get('featureSet', 'basic')
        fee_column = options.get('feeColumn', 'fee')  # Fee column to use for comparison
        
        logger.info(f"[{request_id}] Using Cost-Spec method with feature_set={feature_set}, fee_column={fee_column}")
        
        # Step 3: Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_path = config.spearman_raw_dir / f"raw_data_{timestamp}.json"
        
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
        processed_data_path = config.spearman_processed_dir / f"processed_data_{timestamp}.csv"
        
        processed_df.to_csv(processed_data_path, index=False, encoding='utf-8')
        
        # Step 6: Apply Cost-Spec ranking method with options
        df_ranked = rank_plans_by_cs(
            processed_df,
            feature_set=feature_set,
            fee_column=fee_column
        )
        
        logger.info(f"[{request_id}] Ranked DataFrame shape: {df_ranked.shape}")
        
        # Store the results in global state for access by other endpoints
        logger.info(f"Storing {len(df_ranked)} plans in global state for HTML report")
        
        # Log the top 10 plans by rank to verify all are included
        top_10_by_rank = df_ranked.sort_values('rank_number').head(10)
        logger.info(f"Top 10 plans by rank to be stored:\n{top_10_by_rank[['plan_name', 'CS', 'rank_number']].to_string()}")
        
        # Store the complete dataframe
        config.df_with_rankings = df_ranked.copy()
        
        # Step 7: Generate HTML report
        timestamp_now = datetime.now()
        report_filename = f"cs_ranking_{timestamp_now.strftime('%Y%m%d_%H%M%S')}.html"
        report_path = config.cs_report_dir / report_filename
        
        # Generate HTML content
        # Pass is_cs=True to indicate this is a Cost-Spec report
        html_report = generate_html_report(df_ranked, timestamp_now, is_cs=True, title="Cost-Spec Mobile Plan Rankings")
        
        # Write HTML content to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Step 8: Prepare response with CS ranking data
        # First, ensure all float values are JSON-serializable
        # Replace inf, -inf, and NaN with appropriate values
        df_ranked = df_ranked.replace([np.inf, -np.inf], np.finfo(np.float64).max)
        df_ranked = df_ranked.replace(np.nan, 0)
        
        # Create all_ranked_plans for the response
        columns_to_include = ["id", "plan_name", "mvno", "fee", "original_fee", 
                             "rank_number", "rank_display", "B", "CS"]
        available_columns = [col for col in columns_to_include if col in df_ranked.columns]
        
        # Sort by CS ratio (descending) and add value_ratio field for compatibility
        all_ranked_plans = df_ranked.sort_values("CS", ascending=False)[available_columns].to_dict(orient="records")
        
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
        
        # Prepare response
        response = {
            "request_id": request_id,
            "message": "Data processing complete using Cost-Spec method",
            "status": "success",
            "processing_time_seconds": round(processing_time, 4),
            "options": {
                "featureSet": feature_set,
                "feeColumn": fee_column
            },
            "ranking_method": "cs",
            "results": {
                "raw_data_path": str(raw_data_path),
                "processed_data_path": str(processed_data_path),
                "report_path": str(report_path),
                "report_url": f"/reports/cs_reports/{report_filename}"
            },
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

# The /upload-csv endpoint has been removed
# All functionality is now consolidated in the /process endpoint

# Run the application
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with Uvicorn...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)