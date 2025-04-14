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
from typing import Optional, Union

# Import modules
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

app = FastAPI()

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
        
        # Extract the top 10 plans for response, now sorting by value_ratio instead of ranking_score
        top_10_plans = df_with_rankings.sort_values("value_ratio", ascending=False).head(10)[
            ["plan_name", "mvno", "fee", "value_ratio", "predicted_price"]
        ].to_dict(orient="records")
        
        # Free remaining memory
        del df_with_rankings, model, X, y
        gc.collect()
        
        # Prepare response with all information
        response = {
            "message": "Data processing complete",
            "processing_steps": {
                "data_received": len(data),
                "data_saved": file_path,
                "preprocessing": {
                    "processed_data_saved": latest_path
                },
                "model_training": {
                    "model_type": "xgboost",
                    "features_used": features_to_use,
                    "training_examples": len(data),
                    "model_saved": model_path,
                    "metrics": metrics
                },
                "ranking": {
                    "report_generated": report_path,
                    "report_url": "/"  # Root endpoint now serves the report
                }
            },
            "top_10_plans": top_10_plans
        }
        
        return response
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)