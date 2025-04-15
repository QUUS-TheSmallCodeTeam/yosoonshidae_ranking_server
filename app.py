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

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import necessary modules - Keep data loader, remove others if only used by /logical-test
from modules.models import XGBoostModel
# from modules.utils import setup_logging # Remove this import
from modules.data import load_data_from_json # Keep for potential use
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

def load_model_and_metadata():
    global model, model_metadata
    logger.info("Attempting to load model and metadata...")
    try:
        # Load metadata first
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, 'r') as f:
                model_metadata = json.load(f)
            logger.info(f"Loaded metadata: {model_metadata.get('model_type', 'N/A')}, Features: {len(model_metadata.get('feature_names', []))}")
        else:
            logger.warning(f"Metadata file not found at {METADATA_PATH}")
            model_metadata = {'feature_names': []} # Default empty

        # Load model
        model = XGBoostModel.load(model_path=MODEL_DIR)
        if model:
            logger.info("XGBoost model loaded successfully.")
        else:
            logger.error("Failed to load XGBoost model. XGBoostModel.load returned None.")
            raise HTTPException(status_code=500, detail="Model could not be loaded.")

    except FileNotFoundError:
        logger.error(f"Model or metadata file not found in {MODEL_DIR}. Ensure model is trained and saved.")
        raise HTTPException(status_code=500, detail="Model file not found. Train the model first.")
    except Exception as e:
        logger.exception(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during model loading: {e}")

@app.on_event("startup")
async def startup_event():
    load_model_and_metadata()

@app.get("/", response_class=HTMLResponse)
def read_root():
    """
    Serve the latest ranking HTML report if available,
    otherwise return a simple welcome message.
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
    
    # Read and return the HTML content
    try:
        with open(latest_report, "r", encoding="utf-8") as f:
            html_content = f.read()
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

@app.post("/test")
async def test_endpoint(request: Request):
    # Echo back the request body
    data = await request.json()
    return data

@app.post("/process")
async def process_data(request: Request):
    try:
        # Step 0: Ensure all directories exist
        ensure_directories()
        
        # Step 1: Receive and validate data
        data = await request.json()
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Expected a list of plan data")

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
                    # Prepare features using the *current* model's expected features
                    required_model_features = features_to_use
                    if not required_model_features:
                        logical_test_error = "Cannot run logical test: Model metadata missing feature names."
                        logger.error(f"[{request_id}] {logical_test_error}")
                    else:
                        comparison_features = [
                            'basic_data_clean', 'daily_data_clean', 'voice_clean', 
                            'message_clean', 'throttle_speed_normalized', 'is_5g', 'tethering_gb'
                        ]
                        features_to_ensure = list(set(required_model_features + comparison_features))
                        
                        for feature in features_to_ensure:
                            if feature not in df_logical_test.columns:
                                 logger.warning(f"[{request_id}] Logical test data missing '{feature}', adding with 0.")
                                 df_logical_test[feature] = 0 # Add missing with 0
                                 
                        X_logical_test = df_logical_test[required_model_features].copy()
                        
                        # Predict using the model that was just trained
                        logical_predictions = model.predict(X_logical_test)
                        df_logical_test['predicted_price'] = logical_predictions
                        logger.info(f"[{request_id}] Generated predictions for logical test data.")
                        
                        # Compare (using the same logic as before)
                        base_case_row = df_logical_test[df_logical_test['id'] == 'base']
                        if base_case_row.empty:
                            logical_test_error = "Base case id='base' not found in logical test data."
                            logger.error(f"[{request_id}] {logical_test_error}")
                        else:
                            base_plan = base_case_row.iloc[0]
                            base_price = base_plan['predicted_price']
                            rules = {
                                'basic_data_clean': 1, 'daily_data_clean': 1, 'voice_clean': 1,
                                'message_clean': 1, 'throttle_speed_normalized': 1, 'tethering_gb': 1,
                                'is_5g': 1 
                            }
                            tolerance = 1e-6
                            
                            for _, variant in df_logical_test[df_logical_test['id'] != 'base'].iterrows():
                                variant_price = variant['predicted_price']
                                price_diff = variant_price - base_price
                                comparison = {
                                    "id": variant['id'], "name": variant['name'],
                                    "base_price": round(base_price, 2), "variant_price": round(variant_price, 2),
                                    "difference": round(price_diff, 2), "expected_change": "Unknown",
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
            logger.exception(f"[{request_id}] Error during logical test execution.")
            logical_test_error = f"Error during logical test: {str(e)}"
            
        logger.info(f"[{request_id}] Logical pricing test finished.")
        # --- End of Logical Test Step --- 

        # Clean up memory
        del processed_df, df_with_rankings, X, y
        # Remove y_pred if it exists from metrics calculation
        if 'y_pred' in locals():
            del y_pred
        if 'df_logical_test' in locals(): del df_logical_test
        gc.collect()
        logger.info(f"[{request_id}] Memory cleanup performed.")

        # Step 7: Prepare final response
        end_process_time = time.time()
        total_time = end_process_time - start_process_time
        logger.info(f"[{request_id}] Total processing time: {total_time:.4f} seconds.")
        
        # Get top 10 plans based on value_ratio
        # Need to load the rankings back from the saved report CSV to get top plans
        top_10_plans = []
        try:
            report_csv_path = report_path.replace(".html", ".csv") # Assuming CSV is saved alongside HTML
            if os.path.exists(report_csv_path):
                top_10_plans_df = pd.read_csv(report_csv_path)
                top_10_plans = top_10_plans_df.sort_values("value_ratio", ascending=False).head(10)[
                    ["plan_name", "mvno", "fee", "value_ratio", "predicted_price"]
                ].to_dict(orient="records")
                logger.info(f"[{request_id}] Successfully extracted top 10 plans from report CSV.")
            else:
                logger.warning(f"[{request_id}] Report CSV not found at {report_csv_path}, cannot extract top 10 plans.")
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
            "logical_test_summary": {
                "status": "Error" if logical_test_error else "Completed",
                "error_message": logical_test_error,
                "tests_run": len(logical_test_results),
                "passes": sum(1 for r in logical_test_results if r['status'] == "✅"),
                "failures": sum(1 for r in logical_test_results if r['status'] == "❌"),
                "warnings": sum(1 for r in logical_test_results if r['status'] == "⚠️")
            },
            "logical_test_details": logical_test_results, # Include detailed results
            "top_10_plans": top_10_plans
        }
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def root():
    """ Basic HTML interface. """
    # Look for the latest HTML report
    report_dirs = [REPORT_DIR_BASE, Path("/tmp/reports"), Path("/tmp")]
    html_files = []
    for reports_dir in report_dirs:
        if reports_dir.exists():
            html_files.extend(list(reports_dir.glob("plan_rankings_*.html")))

    latest_report_path = None
    if html_files:
        latest_report = max(html_files, key=lambda x: x.stat().st_mtime)
        # Generate a relative URL path for the report
        try:
            # Construct path relative to potential base path or just filename
            latest_report_path = f"/reports/{latest_report.name}" # Simple relative path
        except Exception:
            latest_report_path = None # Fallback if path generation fails

    # Simple status check
    model_status = "Loaded" if model else "Not Loaded"
    num_features = len(model_metadata.get('feature_names', []))
    report_info = f'<p>Latest Report: <a href="{latest_report_path}">{latest_report.name}</a></p>' if latest_report_path else "<p>No reports available yet.</p>"

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
            <p>Endpoints:</p>
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

@app.post("/predict")
async def predict_plans(plans: List[PlanInput]):
    """Predict prices for a list of mobile plans."""
    start_time = time.time()
    if not model:
        logger.error("Model not loaded during prediction request.")
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    feature_names = model_metadata.get('feature_names')
    if not feature_names:
        logger.error("Feature names not found in model metadata.")
        raise HTTPException(status_code=500, detail="Model metadata incomplete (missing feature names).")

    # Convert Pydantic models to list of dicts
    input_data = [plan.dict() for plan in plans]
    logger.info(f"Received {len(input_data)} plans for prediction.")

    # Preprocess data
    df_features = preprocess_input_data(input_data, feature_names)
    if df_features is None:
        raise HTTPException(status_code=400, detail="Invalid input data or missing features.")

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
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    feature_names = model_metadata.get('feature_names')
    if not feature_names:
        raise HTTPException(status_code=500, detail="Feature names not available in model metadata.")
    return {"expected_features": feature_names}

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