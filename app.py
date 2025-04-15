from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc  # Import garbage collector for memory cleanup
from fastapi.responses import HTMLResponse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from typing import Optional, Union, List
from fastapi.templating import Jinja2Templates
import sys
import logging
import uuid # Import uuid for request IDs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Define Paths (relative to container root /app) ---
APP_DIR = Path(__file__).parent # Should resolve to /app
MODEL_DIR = APP_DIR / "model_files" # Assuming model files are copied here
METADATA_PATH = MODEL_DIR / "xgboost_basic_without_domain_standard_model_metadata.json" # Specific metadata file
LOGICAL_TEST_DATA_PATH = APP_DIR / "data" / "test" / "logical_model_test_set.json" # Path to logical test data
REPORT_DIR_BASE = Path("/tmp/reports") # Use /tmp for reports in container

# Add the project root to the Python path (Less relevant in container, but keep for now)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules - Keep data loader, remove others if only used by /logical-test
from modules.models import XGBoostModel
# from modules.utils import setup_logging # Remove this import
from modules.data import load_data_from_json # Keep for potential use
# from modules.preprocess import preprocess_input_data # REMOVE this import
# Need to re-import original processing/ranking functions if they were removed
from modules import (
    prepare_features,
    get_model,
    calculate_rankings,
    generate_html_report,
    save_report,
    ensure_directories,
    save_raw_data,
    save_processed_data,
    get_basic_feature_list,
    format_model_config,
    save_model_config
    # preprocess_input_data was already removed here
)
# Keep data models
from modules.data_models import PlanInput, FeatureDefinitions

app = FastAPI(title="Moyo Plan Ranking Model Server")
templates = Jinja2Templates(directory="templates")

# Define a model for the incoming data - adjusted to match test.json structure
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

model = None
model_metadata = {}

# Global variable to cache latest results
latest_logical_test_results_cache = None

def load_model_and_metadata():
    global model, model_metadata
    logger.info("Attempting to load model and metadata...")
    model = None # Default to None
    model_metadata = {} # Default to empty
    try:
        # Try loading metadata first
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Loaded metadata from {METADATA_PATH}: {model_metadata.get('model_type', 'N/A')}, Features: {len(model_metadata.get('feature_names', []))}")
        else:
            logger.warning(f"Metadata file not found at {METADATA_PATH}. Proceeding without pre-loaded metadata.")
            model_metadata = {'feature_names': []} # Ensure feature_names exists but is empty

        # Try loading the model - Check if MODEL_DIR exists first
        if os.path.exists(MODEL_DIR) and any(fname.endswith(('.pkl', '.json', '.txt', '.keras', '.h5', '.cbm')) for fname in os.listdir(MODEL_DIR)):
            # Attempt to load only if directory exists and seems to contain model files
            loaded_model_instance = XGBoostModel.load(model_path=MODEL_DIR) # Pass directory path
            if loaded_model_instance:
                model = loaded_model_instance # Assign to global variable only if successful
                logger.info(f"XGBoost model loaded successfully from {MODEL_DIR}.")
            else:
                # This case might happen if .load() returns None for other reasons
                logger.warning(f"XGBoostModel.load returned None when attempting to load from {MODEL_DIR}. Proceeding without pre-loaded model.")
        else:
            logger.warning(f"Model directory {MODEL_DIR} not found or empty. Proceeding without pre-loaded model.")

    except FileNotFoundError:
        # This specific exception might be caught by the os.path.exists checks now, but kept for safety
        logger.warning(f"Model or metadata file not found during startup loading attempt. This is expected if no model was bundled. Proceeding without pre-loaded model.")
    except Exception as e:
        # Log other unexpected errors but don't crash the server
        logger.exception(f"Unexpected error during optional model loading at startup: {e}")
        logger.warning("Proceeding without pre-loaded model due to unexpected error.")

@app.on_event("startup")
async def startup_event():
    load_model_and_metadata()

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Serve the latest ranking HTML report if available,
    and display latest logical test failure count from memory cache.
    """
    # Look for the latest HTML report in the reports directory
    # Try both the regular app directory and the /tmp fallback
    report_dirs = [Path("./reports"), Path("/tmp/reports"), Path("/tmp")]
    
    html_files = []
    for reports_dir in report_dirs:
        if reports_dir.exists():
            html_files.extend(list(reports_dir.glob("plan_rankings_*.html")))
    
    if not html_files:
        # No reports found, return welcome message
        return """
        <html>
            <head>
                <title>Moyo Ranking Model API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                    h1 { color: #2c3e50; }
                </style>
            </head>
            <body>
                <h1>Welcome to the Moyo Ranking Model API</h1>
                <p>No ranking reports are available yet. Use the <code>/process</code> endpoint to analyze data and generate rankings.</p>
            </body>
        </html>
        """
    
    # Get the latest report by modification time
    latest_report = max(html_files, key=lambda x: x.stat().st_mtime)
    print(f"Serving latest report: {latest_report}")
    
    # Set the latest_report_path variable (was missing)
    latest_report_path = f"/reports/{latest_report.name}"
    
    # Read and return the HTML content
    try:
        with open(latest_report, "r", encoding="utf-8") as f:
            html_content = f.read()
        # --- Load and format logical test results FROM MEMORY CACHE --- 
        logical_test_failure_count = "N/A" # Default value
        
        # --> ADDED LOG: Log the cache value *before* trying to read from it
        logger.info(f"Accessing '/' route. Current cache content: {latest_logical_test_results_cache}")
        
        if latest_logical_test_results_cache:
            try:
                summary = latest_logical_test_results_cache.get("summary", {})
                # Get the failure count, default to 0 if not found
                logical_test_failure_count = summary.get('failures', 0) 
            except Exception as e:
                logger.error(f"Error processing cached logical test results: {e}")
                logical_test_failure_count = "Error loading cache"
                
        # Format the simple failure count line
        logical_test_html = f"<p><b>Logical Test Failures:</b> {logical_test_failure_count}</p>"
        
        # Insert the logical test info into the HTML before the closing </body> tag
        if '</body>' in html_content:
            insert_pos = html_content.find('</body>')
            html_content = html_content[:insert_pos] + f"<hr><h3>Model Quality Metrics</h3>{logical_test_html}" + html_content[insert_pos:]
            logger.info(f"Added logical test failure count ({logical_test_failure_count}) to HTML report")
            
        return html_content
    except Exception as e:
        print(f"Error reading HTML report: {e}")
        return """
        <html>
            <head>
                <title>Moyo Ranking Model API - Error</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                    h1 { color: #e74c3c; }
                </style>
            </head>
            <body>
                <h1>Error Reading Report</h1>
                <p>There was an error reading the latest report. Please try generating a new report using the <code>/process</code> endpoint.</p>
            </body>
        </html>
        """

    # For cases where we don't have HTML files (used the early return above)
    # Simple status check
    model_status = "Loaded" if model else "Not Loaded (Run /process)"
    num_features = len(model_metadata.get('feature_names', []))
    
    # --- Load and format logical test results FROM MEMORY CACHE --- 
    logical_test_failure_count = "N/A" # Default value
    
    # --> ADDED LOG: Log the cache value *before* trying to read from it
    logger.info(f"Accessing '/' route. Current cache content: {latest_logical_test_results_cache}")
    
    if latest_logical_test_results_cache:
        try:
            summary = latest_logical_test_results_cache.get("summary", {})
            # Get the failure count, default to 0 if not found
            logical_test_failure_count = summary.get('failures', 0) 
        except Exception as e:
            logger.error(f"Error processing cached logical test results: {e}")
            logical_test_failure_count = "Error loading cache"
            
    # Format the simple failure count line
    logical_test_html = f"<p><b>Logical Test Failures:</b> {logical_test_failure_count}</p>"
    # --- End logical test results --- 
    
    # Report info for welcome page (no reports case)
    report_info = "<p>No reports available yet.</p>"
    
    return f"""
    <html>
        <head><title>Moyo Plan Ranker</title></head>
        <body>
            <h1>Moyo Mobile Plan Ranking Model Server</h1>
            <p>Status: <b>Ready</b></p>
            <p>Model Type: XGBoost</p>
            <p>Model Status: {model_status}</p>
            <p>Expected Features: {num_features}</p>
            {report_info}
            <hr>
            <h3>Logical Test Status</h3>
            {logical_test_html}
            <hr>
            <h3>Endpoints</h3>
            <ul>
                <li><code>POST /process</code>: Submit plan data (JSON list) to preprocess, train, rank, and generate a report.</li>
                <li><code>POST /predict</code>: Get price predictions for plan features (expects preprocessed features).</li>
                <li><code>GET /features</code>: Get the list of features the model expects for /predict.</li>
            </ul>
            <hr>
            <p><i>Navigate to /docs for API documentation (Swagger UI).</i></p>
        </body>
    </html>
    """

@app.post("/test")
async def test_endpoint(request: Request):
    # Echo back the request body
    data = await request.json()
    return data

@app.post("/process")
async def process_data(request: Request):
    start_process_time = time.time()
    request_id = str(uuid.uuid4()) # Generate unique request ID
    logger.info(f"[{request_id}] Received /process request.")
    try:
        # Step 0: Ensure all directories exist
        ensure_directories()
        
        # Step 1: Receive and validate data
        data = await request.json()
        if not isinstance(data, list):
            logger.error(f"[{request_id}] Invalid input data type: {type(data)}. Expected list.")
            raise HTTPException(status_code=400, detail="Expected a list of plan data")

        logger.info(f"[{request_id}] Received {len(data)} plans.")

        # Step 2: Save the received data (important to keep for audit trail)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = save_raw_data(data, timestamp)
        
        # Step 3: Preprocess the data
        try:
            df = pd.DataFrame(data)
            if df.empty:
                raise HTTPException(status_code=400, detail="No data to process")
            
            # Process the data
            processed_df = prepare_features(df)
            
            # Free memory from original dataframe
            del df
            gc.collect()
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=422, detail=f"Error preprocessing data: {str(e)}")
        
        # Save processed data (only latest version)
        latest_path = save_processed_data(processed_df)[1]  # Use the second returned path (latest)
        
        # Step 4: Train XGBoost model
        # Get basic feature list
        basic_features = get_basic_feature_list()
        
        # Filter for features in the processed dataframe
        features_to_use = [f for f in basic_features if f in processed_df.columns]
        X = processed_df[features_to_use]
        y = processed_df['original_fee']
        
        # Record start time for training
        training_start_time = time.time()
        
        # Get XGBoost model - with domain knowledge DISABLED for now
        model = get_model(
            'xgboost',
            use_domain_knowledge=False,  # Disable domain knowledge until monotonic constraints are fixed
            feature_names=features_to_use
        )
        
        # Train the model
        model.train(X, y)
        
        # Calculate training time
        training_time = time.time() - training_start_time
        
        # Calculate model metrics
        try:
            # Make predictions on the same data to evaluate model performance
            y_pred = model.predict(X)
            
            # Calculate various metrics
            metrics = {
                "rmse": np.sqrt(mean_squared_error(y, y_pred)),
                "mae": mean_absolute_error(y, y_pred),
                "r2": r2_score(y, y_pred),
                "explained_variance": explained_variance_score(y, y_pred),
                "training_time": training_time,
                "num_features": len(features_to_use),
                "num_samples": len(X)
            }
            
            # Calculate MAPE but handle zeros appropriately
            # Avoid division by zero by excluding zero values
            non_zero_idx = y != 0
            if non_zero_idx.sum() > 0:
                metrics["mean_absolute_percentage_error"] = mean_absolute_percentage_error(
                    y[non_zero_idx], y_pred[non_zero_idx]
                )
            else:
                metrics["mean_absolute_percentage_error"] = 0.0
                
            print(f"Model metrics: {metrics}")
        except Exception as e:
            print(f"Error calculating metrics: {e}")
            metrics = {
                "note": "Failed to calculate metrics",
                "error": str(e),
                "training_time": training_time,
                "num_features": len(features_to_use),
                "num_samples": len(X)
            }
        
        # Save the model
        model_path = model.save()
        
        # Save model config
        config = format_model_config(
            model,
            features_to_use,
            {
                "input_source": "api",
                "training_examples": len(X),
                "feature_count": len(features_to_use),
                "timestamp": timestamp,
                "metrics": metrics
            }
        )
        config_path = save_model_config(config)
        
        # Step 5: Calculate rankings using the updated logic from update_rankings.py
        df_with_rankings = calculate_rankings(processed_df, model)
        
        # Apply proper ranking with ties
        df_with_rankings = calculate_rankings_with_ties(df_with_rankings, value_column='value_ratio')
        
        # Free memory from processed dataframe
        del processed_df
        gc.collect()
        
        # Step 6: Generate report with the enhanced format
        timestamp_now = time.strftime("%Y-%m-%d_%H-%M-%S")
        html_report = generate_html_report(df_with_rankings, "xgboost", timestamp_now, metrics)
        report_path = save_report(html_report, "xgboost", "standard", "basic", timestamp_now)
        
        logger.info(f"[{request_id}] HTML report saved to {report_path}")

        # --- Step 6: Run Logical Test --- 
        logger.info(f"[{request_id}] Starting logical pricing test...")
        logical_test_results = []
        logical_test_error = None
        try:
            if not os.path.exists(LOGICAL_TEST_DATA_PATH):
                 logical_test_error = f"Logical test data file not found at {LOGICAL_TEST_DATA_PATH}"
                 logger.error(f"[{request_id}] {logical_test_error}")
            else:
                df_logical_test = load_data_from_json(LOGICAL_TEST_DATA_PATH)
                if df_logical_test is None:
                    logical_test_error = f"Failed to load or parse logical test data from {LOGICAL_TEST_DATA_PATH}"
                    logger.error(f"[{request_id}] {logical_test_error}")
                else:
                    logger.info(f"[{request_id}] Loaded {len(df_logical_test)} plans for logical test.")
                    # Prepare features using the model's expected features (from THIS training run)
                    required_model_features = features_to_use # CORRECT: Use features from this request's training
                    if not required_model_features:
                        logical_test_error = "Cannot run logical test: Feature list for current model is empty."
                        logger.error(f"[{request_id}] {logical_test_error}")
                    else:
                        # --- MODIFIED LOGIC: Do NOT re-preprocess logical test data --- 
                        X_logical_test = None
                        logical_predictions = None
                        try:
                            # Verify that logical test data contains all required features
                            missing_logical_features = [f for f in required_model_features if f not in df_logical_test.columns]
                            if missing_logical_features:
                                raise ValueError(f"Logical test data is missing required model features: {missing_logical_features}")
                                
                            # Select required features directly from the loaded logical test data
                            X_logical_test = df_logical_test[required_model_features].copy()
                            logger.info(f"[{request_id}] Selected required features directly from logical test data.")
                            
                            # Predict using the LOADED model
                            logical_predictions = model.predict(X_logical_test)
                            # Add predictions back to the original logical test df for comparison context
                            df_logical_test['predicted_price'] = logical_predictions 
                            logger.info(f"[{request_id}] Generated predictions for logical test data using loaded model.")

                        except Exception as logical_predict_err:
                             # Catch errors specifically during feature selection or prediction of logical test data
                             logical_test_error = f"Error during logical test feature selection/prediction: {type(logical_predict_err).__name__} - {str(logical_predict_err)}"
                             logger.error(f"[{request_id}] {logical_test_error}")
                             import traceback
                             logger.error(f"[{request_id}] Traceback for logical test selection/predict error:\n{traceback.format_exc()}")
                        # --- END MODIFIED LOGIC --- 

                        # Only proceed to comparison if prediction succeeded
                        if logical_predictions is not None and logical_test_error is None:
                            # Compare (using the same logic as before)
                            base_case_row = df_logical_test[df_logical_test['id'] == 'base']
                            if base_case_row.empty:
                                logical_test_error = "Base case id='base' not found in logical test data."
                                logger.error(f"[{request_id}] {logical_test_error}")
                            else:
                                base_plan = base_case_row.iloc[0]
                                # Explicitly convert predicted prices to Python floats
                                base_price = float(base_plan['predicted_price'])
                                rules = {
                                    'basic_data_clean': 1, 'daily_data_clean': 1, 'voice_clean': 1,
                                    'message_clean': 1, 'throttle_speed_normalized': 1, 'tethering_gb': 1,
                                    'is_5g': 1 
                                }
                                tolerance = 1e-6
                                
                                for _, variant in df_logical_test[df_logical_test['id'] != 'base'].iterrows():
                                    # Explicitly convert variant price to Python float
                                    variant_price = float(variant['predicted_price'])
                                    price_diff = variant_price - base_price # Difference is now between Python floats
                                    comparison = {
                                        "id": variant['id'], "name": variant['name'],
                                        # Ensure rounded values are also Python floats
                                        "base_price": round(base_price, 2),
                                        "variant_price": round(variant_price, 2),
                                        "difference": round(price_diff, 2),
                                        "expected_change": "Unknown",
                                        "status": "?", "reason": ""
                                    }
                                    changed_feature = None
                                    expected_direction = 0
                                    for feature, direction in rules.items():
                                        if feature in base_plan.index and feature in variant.index:
                                            base_val = base_plan[feature]
                                            variant_val = variant[feature]
                                            if not np.isclose(base_val, variant_val):
                                                if variant_val > base_val:
                                                    changed_feature = f"{feature} increased"
                                                    expected_direction = direction
                                                    break
                                                elif variant_val < base_val:
                                                    changed_feature = f"{feature} decreased"
                                                    expected_direction = -direction
                                                    break
                                    
                                    if changed_feature:
                                        comparison['expected_change'] = "Increase" if expected_direction == 1 else "Decrease" if expected_direction == -1 else "No Change/Unknown"
                                        if expected_direction == 1:
                                            if price_diff > tolerance: comparison['status'], comparison['reason'] = "✅", f"Passed: {changed_feature}, price increased."
                                            elif abs(price_diff) <= tolerance: comparison['status'], comparison['reason'] = "⚠️", f"Warn: {changed_feature}, price did not change."
                                            else: comparison['status'], comparison['reason'] = "❌", f"Failed: {changed_feature}, price decreased."
                                        elif expected_direction == -1:
                                            if price_diff < -tolerance: comparison['status'], comparison['reason'] = "✅", f"Passed: {changed_feature}, price decreased."
                                            elif abs(price_diff) <= tolerance: comparison['status'], comparison['reason'] = "⚠️", f"Warn: {changed_feature}, price did not change."
                                            else: comparison['status'], comparison['reason'] = "❌", f"Failed: {changed_feature}, price increased."
                                        else: comparison['status'], comparison['reason'] = "?", f"Info: {changed_feature}, expected direction unknown."
                                    else: comparison['status'], comparison['reason'], comparison['expected_change'] = "?", "No significant feature change detected.", "-"
                                    logical_test_results.append(comparison)
        except Exception as e:
            # Log the specific error occurring during the logical test (outer catch-all)
            logger.exception(f"[{request_id}] Outer error during logical test execution: {type(e).__name__} - {str(e)}")
            # Avoid overwriting more specific error from inner block if it exists
            if not logical_test_error:
                 logical_test_error = f"Outer error during logical test: {type(e).__name__} - {str(e)}"
            # Add detailed traceback log for debugging
            import traceback
            logger.error(f"[{request_id}] Traceback for logical test error:\n{traceback.format_exc()}")
            
        logger.info(f"[{request_id}] Logical pricing test finished.")
        # --- Update logical test results CACHE --- 
        global latest_logical_test_results_cache # Declare modification of global
        try:
            latest_logical_test_results_cache = {
                "summary": {
                    "status": "Error" if logical_test_error else "Completed",
                    "error_message": logical_test_error,
                    "tests_run": len(logical_test_results),
                    "passes": sum(1 for r in logical_test_results if r['status'] == "✅"),
                    "failures": sum(1 for r in logical_test_results if r['status'] == "❌"),
                    "warnings": sum(1 for r in logical_test_results if r['status'] == "⚠️")
                },
                "failures_details": [r for r in logical_test_results if r['status'] == "❌"]
            }
            logger.info(f"[{request_id}] Updated in-memory cache for logical test results.")
            # --> ADDED LOG: Log the actual cached value
            logger.info(f"[{request_id}] Cache content after update: {latest_logical_test_results_cache}")
        except Exception as cache_err:
            logger.error(f"[{request_id}] Failed to update logical test results cache: {cache_err}")
            latest_logical_test_results_cache = None # Reset cache on error
        # --- End cache update ---
            
        # --- End of Logical Test Step --- 

        # Clean up memory (REMOVED fragile del statement)
        # Consider which dataframes are needed for the response
        # del processed_df, df_with_rankings, X, y # REMOVED
        # if 'df_logical_test' in locals(): del df_logical_test
        # if 'df_logical_test_processed' in locals(): del df_logical_test_processed
        # gc.collect()
        # logger.info(f"[{request_id}] Memory cleanup performed.")

        # Step 7: Prepare final response
        end_process_time = time.time()
        total_time = end_process_time - start_process_time
        logger.info(f"[{request_id}] Total processing time: {total_time:.4f} seconds.")
        
        # Get top 10 plans based on value_ratio
        top_10_plans = []
        try:
            if 'value_ratio' in df_with_rankings.columns:
                # Convert relevant columns to standard types before creating dict
                cols_to_convert = {'fee': float, 'value_ratio': float, 'predicted_price': float}
                for col, dtype in cols_to_convert.items():
                    if col in df_with_rankings.columns:
                        df_with_rankings[col] = df_with_rankings[col].astype(dtype)
                         
                top_10_plans = df_with_rankings.sort_values("value_ratio", ascending=False).head(10)[
                    ["plan_name", "mvno", "fee", "value_ratio", "predicted_price", "rank_display", "id"]
                ].to_dict(orient="records")
                logger.info(f"[{request_id}] Successfully extracted top 10 plans.")
            else:
                logger.warning(f"[{request_id}] Report CSV not found at {report_path}, cannot extract top 10 plans.")
        except Exception as e:
             logger.exception(f"[{request_id}] Error reading report CSV for top 10 plans.")

        response = {
            "request_id": request_id,
            "message": "Data processing, model training, ranking, and logical test complete.",
            "status": "Success",
            "processing_time_seconds": round(total_time, 4),
            "results": {
                "raw_data_path": file_path,
                "processed_data_path": latest_path,
                "model_path": model_path,
                "config_path": config_path,
                "report_path": report_path,
                "report_url": f"/reports/{Path(report_path).name}" # Relative URL
            },
            "model_metrics": metrics,
            "logical_test_summary": latest_logical_test_results_cache.get("summary") if latest_logical_test_results_cache else {"status": "Cache not populated"},
            "logical_test_details": latest_logical_test_results_cache.get("failures_details", []) if latest_logical_test_results_cache else [],
            "top_10_plans": top_10_plans
        }
        return response
    except Exception as e:
        import traceback
        # Log the type and message of the exception caught by the outer block
        logger.exception(f"[{request_id}] Unhandled error in /process endpoint: {type(e).__name__} - {str(e)}")
        # Log the full traceback for the outer exception
        logger.error(f"[{request_id}] Traceback for unhandled error:\n{traceback.format_exc()}")
        # Raise the HTTPException, embedding the specific error message
        raise HTTPException(status_code=500, detail=f"Error processing data: {type(e).__name__} - {str(e)}")

@app.post("/predict")
async def predict_plans(plans: List[PlanInput]):
    """Predict prices for a list of mobile plans."""
    start_time = time.time()
    # Check if model is loaded (might be None if /process hasn't run yet)
    if not model:
        logger.error("Model not available for /predict. Run /process first.")
        raise HTTPException(status_code=503, detail="Model not yet trained/available. Run /process first.")

    logger.info(f"Received {len(input_data)} plans for prediction.") # Moved log after check

    # Get features from metadata (should be loaded or defaulted by startup)
    feature_names = model_metadata.get('feature_names')
    if not feature_names:
        logger.error("Feature names not found in model metadata (model might not be loaded/trained).")
        raise HTTPException(status_code=500, detail="Model metadata incomplete (missing feature names). Run /process first.")

    # Convert Pydantic models to list of dicts
    input_data = [plan.dict() for plan in plans]
    # logger.info(f"Received {len(input_data)} plans for prediction.") # Moved log up

    # Preprocess data using prepare_features
    # Convert list of dicts to DataFrame
    df_input = pd.DataFrame(input_data)
    if df_input.empty:
        raise HTTPException(status_code=400, detail="Input data is empty.")
    
    # Use prepare_features (designed for raw data)
    df_processed = prepare_features(df_input) 
    
    # Select only the features the model was trained on
    # Ensure all required features exist after preprocessing
    missing_features = [f for f in feature_names if f not in df_processed.columns]
    if missing_features:
        logger.error(f"Preprocessing did not generate required features: {missing_features}")
        raise HTTPException(status_code=500, detail=f"Preprocessing failed to generate required features: {missing_features}")
        
    df_features = df_processed[feature_names].copy()

    # df_features = preprocess_input_data(input_data, feature_names) # OLD: Incorrect call
    if df_features is None:
        # This check might be redundant now if prepare_features raises errors
        raise HTTPException(status_code=400, detail="Invalid input data or missing features after preprocessing.")

    # Predict
    try:
        predictions = model.predict(df_features)
        logger.info(f"Generated {len(predictions)} predictions.")

        # Format results
        results = []
        for i, plan_dict in enumerate(input_data):
            # Include original input features + prediction
            result_item = plan_dict.copy()
            result_item['predicted_price'] = round(float(predictions[i]), 2) # Round prediction
            results.append(result_item)

        end_time = time.time()
        logger.info(f"Prediction completed in {end_time - start_time:.4f} seconds.")
        return {"predictions": results}

    except Exception as e:
        logger.exception(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")

@app.get("/features")
async def get_features():
    """Return the list of features the loaded model expects."""
    # Check if model/metadata is available
    if not model or not model_metadata or not model_metadata.get('feature_names'):
        logger.error("Model/Metadata not available for /features. Run /process first.")
        raise HTTPException(status_code=503, detail="Model/Features not available. Run /process first.")
        
    feature_names = model_metadata.get('feature_names')
    return {"expected_features": feature_names}

# Add function to calculate proper rankings with ties
def calculate_rankings_with_ties(df, value_column='value_ratio', ascending=False):
    """
    Calculate rankings with proper handling of ties.
    For tied ranks, uses '공동 X위' (joint X rank) notation
    and ensures the next rank after ties is correctly incremented.
    
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
    current_rank = 1
    previous_value = None
    tied_count = 0
    ranks = []
    rank_displays = []
    
    # Calculate ranks
    for idx, row in df_sorted.iterrows():
        current_value = row[value_column]
        
        # Check if this is a tie with the previous value
        if previous_value is not None and current_value == previous_value:
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
    
    # Return to original order
    df_sorted = df_sorted.reindex(df.index)
    return df_sorted

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting server with Uvicorn...")
    # Ensure model is loaded before starting server if run directly
    if model is None:
        try:
            load_model_and_metadata()
        except HTTPException as e:
            logger.critical(f"CRITICAL: Failed to load model on startup: {e.detail}. Server might not function correctly.")
            # Optionally exit if model loading is critical
            # sys.exit(1)
        except Exception as e:
             logger.critical(f"CRITICAL: Unexpected error loading model on startup: {e}. Server might not function correctly.")
             # sys.exit(1)

    uvicorn.run(app, host="0.0.0.0", port=7860)