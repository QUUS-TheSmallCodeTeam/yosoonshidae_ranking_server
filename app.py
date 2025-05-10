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
from modules import (
    prepare_features, ensure_directories, save_raw_data, save_processed_data,
    calculate_rankings_with_spearman, calculate_rankings_with_ties,
    generate_html_report, save_report
)
from modules.dea import calculate_rankings_with_dea
from fastapi import UploadFile, File
from modules.models import get_basic_feature_list  # Import directly to avoid circular imports

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

# Spearman ranking calculation is now imported from the modules.spearman module
# calculate_rankings_with_ties function is now imported from the modules.ranking module

# HTML report generation and saving is now handled by the modules.report module

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
                
                <h2>Ranking Methods</h2>
                <div class="method-info">
                    <h3>Spearman Correlation</h3>
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

                <div class="method-info">
                    <h3>Data Envelopment Analysis (DEA)</h3>
                    <p>Upload a CSV file with plan data to perform DEA ranking analysis. The file should have the same structure as the 'latest processed data' CSV.</p>
                    <p>DEA will calculate efficiency scores for each plan based on:</p>
                    <ul>
                        <li>Input: Plan fee</li>
                        <li>Outputs: Plan features (data, voice, message, etc.)</li>
                    </ul>
                    <p>Plans with higher efficiency scores will be ranked higher.</p>
                </div>

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
                <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const form = document.querySelector('.dea-form');
                    const fileInput = document.getElementById('csv_file');
                    const fileError = document.getElementById('file-error');
                    const processingStatus = document.getElementById('processing-status');
                    const resultMessage = document.getElementById('result-message');
                    const progressBar = document.querySelector('.progress');
                    
                    form.addEventListener('submit', async function(e) {
                        e.preventDefault();
                        
                        // Reset error message
                        fileError.textContent = '';
                        fileError.classList.add('hidden');
                        
                        // Validate file
                        if (!fileInput.files[0]) {
                            fileError.textContent = 'Please select a file';
                            fileError.classList.remove('hidden');
                            return;
                        }
                        
                        // Show processing status
                        processingStatus.classList.remove('hidden');
                        progressBar.style.width = '0%';
                        
                        try {
                            // Simulate processing (in real implementation, this would be async)
                            for (let i = 0; i <= 100; i += 10) {
                                progressBar.style.width = i + '%';
                                await new Promise(resolve => setTimeout(resolve, 200));
                            }
                            
                            // Submit form
                            form.submit();
                        } catch (error) {
                            fileError.textContent = 'Error processing file: ' + error.message;
                            fileError.classList.remove('hidden');
                            processingStatus.classList.add('hidden');
                        }
                    });
                    
                    // Handle form submission response
                    window.addEventListener('message', function(e) {
                        if (e.data.type === 'dea-result') {
                            resultMessage.textContent = e.data.message;
                            resultMessage.classList.remove('hidden');
                            processingStatus.classList.add('hidden');
                        }
                    });
                });
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

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """
    Upload CSV file and process with DEA ranking.
    
    Args:
        file: CSV file containing plan data
        
    Returns:
        HTML report with DEA rankings
        
    Raises:
        HTTPException: If file upload or processing fails
    """
    try:
        # Validate file
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
            
        # Save uploaded file
        csv_path = DATA_DIR / "dea_input" / file.filename
        if not csv_path.parent.exists():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(csv_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Validate CSV content
        try:
            df = pd.read_csv(csv_path)
            if df.empty:
                raise ValueError("Uploaded CSV file is empty")
            
            # Check for required columns
            required_columns = ['fee'] + get_basic_feature_list()
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
                
        except Exception as e:
            logger.error(f"Error reading or validating CSV file: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Process DEA
        try:
            result_df = calculate_rankings_with_dea(df)
            if result_df is None or result_df.empty:
                raise ValueError("DEA calculation returned empty results")
                
        except Exception as e:
            logger.error(f"Error in DEA calculation: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error in DEA calculation: {str(e)}")
        
        # Generate report
        try:
            report_path = REPORT_DIR_BASE / f"dea_ranking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            generate_html_report(result_df, report_path)
            if not report_path.exists():
                raise ValueError("Failed to generate HTML report")
                
            return FileResponse(report_path)
            
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in upload_csv: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Run the application
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with Uvicorn...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)