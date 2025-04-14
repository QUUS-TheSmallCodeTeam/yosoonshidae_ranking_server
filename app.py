from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import os
import json
from datetime import datetime
import pandas as pd
from pathlib import Path
from preprocess import prepare_features
import uvicorn

app = FastAPI()

# Define a model for the incoming data (adjust based on actual data structure from crawl-plans)
class PlanData(BaseModel):
    id: int
    plan_name: str
    network: str
    mvno: str
    mno: str
    basic_data: float
    daily_data: float
    data_exhaustion: str
    voice: float
    message: float
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the Moyo Ranking Model API"}

@app.post("/receive-data")
async def receive_data(request: Request):
    try:
        data = await request.json()
        # Validate data structure (adjust based on actual data from crawl-plans)
        if not isinstance(data, list):
            raise HTTPException(status_code=400, detail="Expected a list of plan data")

        # Save the received data to a file for preprocessing
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_dir = os.path.join(os.path.dirname(__file__), "../data/raw")
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, f"received_data_{timestamp}.json")
        
        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        return {"message": f"Received {len(data)} records, saved to {file_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

@app.post("/preprocess-and-train")
async def preprocess_and_train():
    try:
        # Define paths
        data_dir = Path(os.path.dirname(__file__)) / "../data"
        raw_dir = data_dir / "raw"
        processed_dir = data_dir / "processed"

        # Create directories if they don't exist
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(processed_dir, exist_ok=True)

        # Find the latest raw data file
        raw_files = list(raw_dir.glob("received_data_*.json"))
        if not raw_files:
            raise HTTPException(status_code=404, detail="No raw data files found for preprocessing")

        latest_file = max(raw_files, key=os.path.getctime)
        with open(latest_file, "r") as f:
            data = json.load(f)

        # Convert data to DataFrame
        df = pd.DataFrame(data)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data to preprocess")

        # Preprocess the data using logic from preprocess.py
        processed_df = prepare_features(df)

        # Save processed data
        timestamp = datetime.now().strftime("%Y%m%d")
        output_path = processed_dir / f"processed_data_{timestamp}.csv"
        processed_df.to_csv(output_path, index=False)

        # Also save as latest for easy access
        latest_path = processed_dir / "latest_processed_data.csv"
        processed_df.to_csv(latest_path, index=False)

        # Placeholder for training logic
        # In a real implementation, you would call your training function here
        training_message = "Training process started (placeholder). In a real implementation, training would occur here."

        return {
            "message": f"Preprocessing complete. Saved to {output_path}",
            "also_saved_to": str(latest_path),
            "total_records": len(processed_df),
            "features": len(processed_df.columns),
            "training": training_message
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during preprocessing or training: {str(e)}")

@app.post("/test")
async def test_endpoint():
    return {"message": "Test successful. Endpoint is reachable."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
