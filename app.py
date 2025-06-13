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
cached_html_content = None  # Cache for HTML content to avoid regeneration
cached_html_timestamp = None  # Timestamp of cached content

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
    global cached_html_content, cached_html_timestamp, chart_calculation_status
    
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
        
        # Update cache
        cached_html_content = html_report
        cached_html_timestamp = timestamp_now
        
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
@app.get("/", response_class=HTMLResponse)
def read_root(basic: bool = False):
    """
    Serve the latest ranking HTML report if available, or show calculation status.
    """
    global cached_html_content, cached_html_timestamp, chart_calculation_status
    
    # If basic report is requested and we have data, show it
    if basic and config.df_with_rankings is not None:
        try:
            df_for_html = config.df_with_rankings.copy()
            is_cs = any(col for col in df_for_html.columns if col == 'CS')
            
            if is_cs:
                df_for_html = df_for_html.sort_values('CS', ascending=False)
                basic_html = generate_basic_html_report(df_for_html)
                return HTMLResponse(content=basic_html)
        except Exception as e:
            logger.error(f"Error generating basic report: {e}")
    
    # Check if we have rankings in memory first
    if config.df_with_rankings is not None:
        # Check if we have cached content that's still fresh (less than 5 minutes old)
        current_time = datetime.now()
        if (cached_html_content is not None and 
            cached_html_timestamp is not None and 
            (current_time - cached_html_timestamp).total_seconds() < 300):  # 5 minutes
            return HTMLResponse(content=cached_html_content)
        
        # If charts are currently being calculated, show status page
        if chart_calculation_status['is_calculating']:
            progress = chart_calculation_status['calculation_progress']
            status_html = f"""
            <html>
                <head>
                    <title>Generating Charts - Moyo Plan Rankings</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; background-color: #f8f9fa; }}
                        .status-container {{ max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                        .progress-container {{ width: 100%; margin: 30px 0; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
                        .progress-bar {{ height: 20px; background: linear-gradient(90deg, #007bff, #0056b3); border-radius: 10px; transition: width 0.3s ease; }}
                        .status {{ margin: 20px 0; font-size: 18px; color: #495057; }}
                        .loading-icon {{ font-size: 48px; margin: 20px 0; animation: spin 2s linear infinite; }}
                        .refresh-btn {{ background-color: #007bff; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; margin-top: 20px; }}
                        .refresh-btn:hover {{ background-color: #0056b3; }}
                        @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
                        .eta {{ font-size: 14px; color: #6c757d; margin-top: 10px; }}
                    </style>
                </head>
                <body>
                    <div class="status-container">
                        <h1>📊 Multi-Feature Frontier Regression Analysis</h1>
                        <div class="loading-icon">⚙️</div>
                        <div class="status">Processing advanced visualizations...</div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width: {progress}%"></div>
                        </div>
                        <p><strong>{progress}% Complete</strong></p>
                        <div class="eta">
                            {'Estimated time remaining: 30-45 seconds' if progress < 50 else 'Almost done! 10-15 seconds remaining'}
                        </div>
                        <button class="refresh-btn" onclick="window.location.reload()">🔄 Check Progress</button>
                        <hr style="margin: 30px 0;">
                        <p style="font-size: 14px; color: #6c757d;">
                            The system is generating advanced multi-frontier regression charts and cost structure analysis.<br>
                            <strong>Manual refresh recommended</strong> - Click "Check Progress" or refresh the page to see updates.
                        </p>
                    </div>
                </body>
            </html>
            """
            return HTMLResponse(content=status_html)
        
        # If there was an error in chart calculation, show error page
        if chart_calculation_status['error_message']:
            error_html = f"""
            <html>
                <head>
                    <title>Chart Generation Failed - Moyo Plan Rankings</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 40px; text-align: center; background-color: #f8f9fa; }}
                        .error-container {{ max-width: 600px; margin: 0 auto; background: white; padding: 40px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); border-left: 5px solid #dc3545; }}
                        .error-icon {{ font-size: 48px; margin: 20px 0; color: #dc3545; }}
                        .error-title {{ color: #dc3545; margin-bottom: 20px; }}
                        .error-message {{ background-color: #f8d7da; color: #721c24; padding: 15px; border-radius: 5px; margin: 20px 0; font-family: monospace; text-align: left; }}
                        .retry-btn {{ background-color: #28a745; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }}
                        .retry-btn:hover {{ background-color: #218838; }}
                        .basic-btn {{ background-color: #6c757d; color: white; border: none; padding: 12px 24px; border-radius: 5px; cursor: pointer; font-size: 16px; margin: 10px; }}
                        .basic-btn:hover {{ background-color: #5a6268; }}
                    </style>
                </head>
                <body>
                    <div class="error-container">
                        <h1 class="error-title">❌ Chart Generation Failed</h1>
                        <div class="error-icon">⚠️</div>
                        <p>The advanced chart generation encountered an error:</p>
                        <div class="error-message">{chart_calculation_status['error_message']}</div>
                        <p>You can try the following options:</p>
                        <button class="retry-btn" onclick="window.location.href='/process'">🔄 Process New Data</button>
                        <button class="basic-btn" onclick="generateBasicReport()">📋 Show Basic Report</button>
                        <hr style="margin: 30px 0;">
                        <p style="font-size: 14px; color: #6c757d;">
                            The ranking calculations completed successfully, but chart visualization failed.<br>
                            You can still access the ranking data through the API or process new data.
                        </p>
                    </div>
                    <script>
                        function generateBasicReport() {{
                            // Redirect to a basic report endpoint
                            window.location.href = '/?basic=true';
                        }}
                    </script>
                </body>
            </html>
            """
            return HTMLResponse(content=error_html)
        
        # If there was an error in chart calculation, show basic report without charts
        if chart_calculation_status['error_message']:
            try:
                # Generate a simple report without expensive chart calculations
                df_for_html = config.df_with_rankings.copy()
                is_cs = any(col for col in df_for_html.columns if col == 'CS')
                
                if is_cs:
                    df_for_html = df_for_html.sort_values('CS', ascending=False)
                    
                    # Generate basic HTML without charts
                    basic_html = generate_basic_html_report(df_for_html)
                    return HTMLResponse(content=basic_html)
                    
            except Exception as e:
                logger.error(f"Error generating basic report: {e}")
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
                    <button id="multi-frontier-btn" class="active" onclick="changeMethod('multi_frontier')">Multi-Frontier Regression (Recommended)</button>
                    <button id="decomp-btn" onclick="changeMethod('linear_decomposition')">Linear Decomposition</button>
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
                    method: "multi_frontier",
                    feeType: "original"
                };
                
                /* Change method */
                function changeMethod(method) {
                    /* Update buttons */
                    document.getElementById('multi-frontier-btn').classList.remove('active');
                    document.getElementById('decomp-btn').classList.remove('active');
                    document.getElementById('frontier-btn').classList.remove('active');
                    
                    if (method === 'multi_frontier') {
                        document.getElementById('multi-frontier-btn').classList.add('active');
                    } else if (method === 'linear_decomposition') {
                        document.getElementById('decomp-btn').classList.add('active');
                    } else {
                        document.getElementById('frontier-btn').classList.add('active');
                    }
                    
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
    latest_report_path = f"/reports/cs_reports/{latest_report.name}"
    
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
        
        # Invalidate HTML cache when new data is processed
        global cached_html_content, cached_html_timestamp
        cached_html_content = None
        cached_html_timestamp = None
        
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