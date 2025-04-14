#!/bin/bash

# Create necessary directories
mkdir -p data/raw data/processed trained_models/xgboost reports

# Install requirements
pip install -r requirements.txt

# Run the server
python -m uvicorn app:app --reload --host 0.0.0.0 --port 7860 