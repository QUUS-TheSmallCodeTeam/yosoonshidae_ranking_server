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
from modules.dea import calculate_rankings_with_dea
from modules.models import get_basic_feature_list
from modules.spearman import calculate_rankings_with_spearman
from fastapi import UploadFile, File

# Initialize FastAPI
app = FastAPI(title="Moyo Plan Ranking Model Server - DEA Method")

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
    # Check if we have rankings in memory first
    if config.df_with_rankings is not None:
        # Generate HTML report from the in-memory rankings
        try:
            # Check if this is a DEA ranking based on column names
            is_dea = any(col for col in config.df_with_rankings.columns if col.startswith('dea_'))
            
            # Generate report with appropriate parameters
            if is_dea:
                logger.info("Generating DEA report for main endpoint")
                html_content = generate_html_report(
                    config.df_with_rankings, 
                    datetime.now(), 
                    is_dea=True, 
                    title="DEA Mobile Plan Rankings"
                )
            else:
                logger.info("Generating Spearman report for main endpoint")
                html_content = generate_html_report(config.df_with_rankings, datetime.now())
                
            return HTMLResponse(content=html_content)
        except Exception as e:
            logger.error(f"Error generating report from in-memory rankings: {e}")
            # Fall back to looking for files
    
    # Look for the latest HTML report in all potential directories
    report_dirs = [
        config.spearman_report_dir,  # Spearman reports
        config.dea_report_dir,       # DEA reports
        Path("./reports"), 
        Path("/tmp/reports"), 
        Path("/tmp")
    ]
    
    html_files = []
    for reports_dir in report_dirs:
        if reports_dir.exists():
            # Look for both DEA and Spearman reports
            html_files.extend(list(reports_dir.glob("*ranking_*.html")))
    
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
                    <h2>Data Envelopment Analysis (DEA) Ranking Method</h2>
                    <p>This API uses the DEA method to estimate plan efficiency and rank mobile plans:</p>
                    <ol>
                        <li>Calculate efficiency scores using Data Envelopment Analysis</li>
                        <li>Use plan features as outputs (basic_data, voice, message, etc.)</li>
                        <li>Use plan fee as input (cost)</li>
                        <li>Apply Variable Returns to Scale (VRS) for better discrimination</li>
                        <li>Set unlimited features to maximum observed values</li>
                        <li>Rank plans by DEA efficiency score</li>
                        <li>Generate comprehensive reports with detailed metrics</li>
                    </ol>
                </div>
                
                <p>No ranking reports are available yet. Use the <code>/process</code> endpoint to analyze data and generate rankings.</p>
                
                <h2>Ranking Method Details</h2>
                <div class="method-info">
                    <h3>Data Envelopment Analysis (DEA)</h3>
                    <p>The DEA method offers several advantages for ranking mobile plans:</p>
                    <ul>
                        <li><strong>Data-driven approach</strong>: No subjective weights needed</li>
                        <li><strong>Efficiency measurement</strong>: Evaluates how efficiently plans convert cost to features</li>
                        <li><strong>Multiple input/output handling</strong>: Considers all plan features simultaneously</li>
                        <li><strong>Non-parametric</strong>: No assumptions about the underlying distribution</li>
                        <li><strong>Flexible configuration</strong>: Supports different returns to scale and feature sets</li>
                    </ul>
                </div>

                <div class="method-info">
                    <h3>API Usage</h3>
                    <p>Submit plan data to the <code>/process</code> endpoint to generate rankings using Data Envelopment Analysis (DEA).</p>
                    <p>DEA uses SciPy's linear programming solver to calculate efficiency scores, which are then converted to rankings.</p>
                    <p>Required columns: 'fee' and basic feature columns (basic_data_clean, voice_clean, message_clean, etc.)</p>
                    <p>Use the <code>/process</code> endpoint to submit plan data in JSON format:</p>
                    <pre style="background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto;">
{
  "options": {
    "featureSet": "basic",
    "targetVariable": "fee",
    "rts": "vrs"
  },
  "data": [
    { "id": 1, "plan_name": "Plan A", ... },
    { "id": 2, "plan_name": "Plan B", ... }
  ]
}
                    </pre>
                </div>

                <style>
                    .upload-form {
                        margin-top: 20px;
                        padding: 15px;
                        background-color: #f8f9fa;
                        border-radius: 4px;
                    }
                    .upload-form input[type="file"] {
                        margin-bottom: 10px;
                        padding: 8px;
                        border: 1px solid #ddd;
                        border-radius: 4px;
                    }
                    .upload-btn {
                        padding: 10px 20px;
                        background-color: #007bff;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        cursor: pointer;
                    }
                    .upload-btn:hover {
                        background-color: #0056b3;
                    }
                </style>

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
                    <li><code>POST /process</code>: Submit plan data (JSON list) to preprocess, rank using DEA method, and generate a report.</li>
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
    logger.info(f"Serving latest report: {latest_report}")
    
    # Check if this is a DEA report based on the filename
    is_dea_report = 'dea' in latest_report.name.lower()
    logger.info(f"Report identified as DEA report: {is_dea_report}")
    
    # If it's a file from the DEA reports directory, it's definitely a DEA report
    if latest_report.parent == config.dea_report_dir:
        is_dea_report = True
        logger.info("Report confirmed as DEA report based on directory")
    
    # Read the HTML file content
    with open(latest_report, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # If it's a DEA report but doesn't have the proper is_dea parameter in the HTML,
    # regenerate the report with the proper parameters
    if is_dea_report and '<h2>DEA Plan Rankings</h2>' in html_content:
        try:
            # Try to load the DataFrame from the file
            # This is a simplified approach - in a real implementation, you might want to
            # extract the DataFrame from the HTML or use a cached version
            if config.df_with_rankings is not None:
                logger.info("Regenerating DEA report with proper parameters")
                html_content = generate_html_report(
                    config.df_with_rankings,
                    datetime.now(),
                    is_dea=True,
                    title="DEA Mobile Plan Rankings"
                )
        except Exception as e:
            logger.error(f"Error regenerating DEA report: {e}")
    
    # Set the latest_report_path variable for reference
    latest_report_path = f"/reports/{latest_report.name}"
    
    # Read and return the HTML content
    # Note: We've already read the content above for DEA detection, so we don't need to read it again
    # unless we didn't regenerate it
    try:
        # html_content is already set from above
        
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
    """Process plan data using the DEA method."""
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
        
        # Extract DEA options with defaults
        feature_set = options.get('featureSet', 'basic')
        target_variable = options.get('targetVariable', 'fee')
        rts = options.get('rts', 'vrs')  # Default to VRS for better discrimination
        
        logger.info(f"[{request_id}] Using DEA options: feature_set={feature_set}, target_variable={target_variable}, rts={rts}")

        # Step 3: Save raw data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_data_path = config.spearman_raw_dir / f"raw_data_{timestamp}.json"
        if not raw_data_path.parent.exists():
            raw_data_path.parent.mkdir(parents=True, exist_ok=True)
        
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
        if not processed_data_path.parent.exists():
            processed_data_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_df.to_csv(processed_data_path, index=False, encoding='utf-8')
        
        # Step 6: Apply DEA ranking method with options
        df_ranked = calculate_rankings_with_dea(
            processed_df,
            feature_set=feature_set,
            target_variable=target_variable,
            rts=rts
        )
        
        logger.info(f"[{request_id}] Ranked DataFrame shape: {df_ranked.shape}")
        
        # Store the results in global state for access by other endpoints
        config.df_with_rankings = df_ranked
        
        # Step 7: Generate HTML report
        timestamp_now = datetime.now()
        report_filename = f"dea_ranking_{timestamp_now.strftime('%Y%m%d_%H%M%S')}.html"
        report_path = config.dea_report_dir / report_filename
        if not report_path.parent.exists():
            report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate HTML content
        # Pass is_dea=True to indicate this is a DEA report
        html_report = generate_html_report(df_ranked, timestamp_now, is_dea=True, title="DEA Mobile Plan Rankings")
        
        # Write HTML content to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_report)
        
        # Step 8: Prepare response with DEA ranking data
        # First, ensure all float values are JSON-serializable
        # Replace inf, -inf, and NaN with appropriate values
        df_ranked = df_ranked.replace([np.inf, -np.inf], np.finfo(np.float64).max)
        df_ranked = df_ranked.replace(np.nan, 0)
        
        all_rankings = {}
        
        # For DEA, we only have one ranking method
        rank_col = 'dea_rank'
        value_col = 'dea_score'
        
        if rank_col in df_ranked.columns and value_col in df_ranked.columns:
            columns_to_include = ["id", "plan_name", "mvno", "fee", "original_fee", 
                                rank_col, "dea_efficiency", value_col]
            available_columns = [col for col in columns_to_include if col in df_ranked.columns]
            
            # Sort by the ranking column and convert to records
            all_rankings['dea'] = df_ranked.sort_values(rank_col)[available_columns].to_dict(orient="records")
            logger.info(f"[{request_id}] Included DEA rankings with {len(all_rankings['dea'])} plans")
        
        # Get top 10 plans by DEA score
        top_10_value_col = "dea_score"
        
        # Get top 10 plans
        top_10_plans = []
        try:
            columns_to_include = ["id", "plan_name", "mvno", "fee", "original_fee", 
                               "dea_rank", "dea_efficiency", top_10_value_col]
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
                               "dea_rank", "dea_efficiency", top_10_value_col]
                               
            available_columns = [col for col in columns_to_include if col in df_ranked.columns]
            
            # Get all plans, sorted by DEA score
            all_ranked_plans = df_ranked.sort_values(top_10_value_col, ascending=False)[available_columns].to_dict(orient="records")
            
            # Ensure each plan has a value_ratio field (required for edge function DB upsert)
            for plan in all_ranked_plans:
                # Add value_ratio field using dea_score for compatibility
                plan["value_ratio"] = plan.get(top_10_value_col, 0)
            
            logger.info(f"[{request_id}] Prepared all_ranked_plans with {len(all_ranked_plans)} plans")
        except Exception as e:
            logger.error(f"[{request_id}] Error preparing all_ranked_plans: {e}")
        
        # Calculate timing
        end_time = time.time()
        processing_time = end_time - start_time
        
        response = {
            "request_id": request_id,
            "message": "Data processing complete using DEA method",
            "status": "success",
            "processing_time_seconds": round(processing_time, 4),
            "options": {
                "featureSet": feature_set,
                "targetVariable": target_variable,
                "rts": rts
            },
            "results": {
                "raw_data_path": raw_data_path,
                "processed_data_path": processed_data_path,
                "report_path": report_path,
                "report_url": f"/reports/{Path(report_path).name}" if report_path else None
            },
            "ranking_method": "dea",
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