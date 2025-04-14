from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import time
import gc  # Import garbage collector for memory cleanup
from fastapi.responses import HTMLResponse

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

# Define a model for the incoming data
class PlanData(BaseModel):
    id: int
    plan_name: str
    network: str
    mvno: str
    mno: str
    basic_data: float
    daily_data: float
    data_exhaustion: str
    voice: str
    message: str
    additional_call: str
    data_sharing: bool
    roaming_support: bool
    micro_payment: bool
    is_esim: bool
    signup_minor: bool
    signup_foreigner: bool
    has_usim: bool
    has_nfc_usim: bool
    tethering_gb: float
    tethering_status: str
    esim_fee: int
    esim_fee_status: str
    usim_delivery_fee: int
    usim_delivery_fee_status: str
    nfc_usim_delivery_fee: int
    nfc_usim_delivery_fee_status: str
    fee: float
    original_fee: float
    discount_fee: float
    discount_period: int
    post_discount_fee: float
    agreement: bool
    agreement_period: int
    agreement_type: str
    num_of_signup: int
    mvno_rating: float
    monthly_review_score: float
    discount_percentage: float

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
        
        # Get XGBoost model
        model = get_model(
            'xgboost',
            use_domain_knowledge=True,
            feature_names=features_to_use
        )
        
        # Train the model
        model.train(X, y)
        
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
                "timestamp": timestamp
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
        html_report = generate_html_report(df_with_rankings, "xgboost", timestamp_now)
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
                    "model_saved": model_path
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