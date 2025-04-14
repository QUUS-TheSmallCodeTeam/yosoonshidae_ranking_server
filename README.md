---
title: Mvno Plan Ranking Model
emoji: 🌖
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Moyo Mobile Plan Ranking API

This FastAPI application provides an API for preprocessing, training, and ranking mobile phone plans based on various features.

## API Endpoints

- **GET `/`**: Welcome message
- **POST `/test`**: Test endpoint - echoes back the request body
- **POST `/process`**: Main endpoint - processes plan data through the complete ML pipeline:
  1. Processes and saves raw data
  2. Applies feature engineering
  3. Trains XGBoost model with domain knowledge
  4. Ranks plans and generates reports

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- pandas
- numpy
- xgboost
- scikit-learn

## Directory Structure

```
hf_server/
├── app.py               # Main FastAPI application
├── Dockerfile           # For Hugging Face Spaces deployment
├── modules/             # Modular code organization
│   ├── __init__.py      # Module exports
│   ├── models.py        # ML model implementation
│   ├── preprocess.py    # Feature engineering
│   ├── ranking.py       # Plan ranking logic
│   └── utils.py         # Utility functions
└── requirements.txt     # Dependencies
```

## How to Use

### Run Locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

### Deploy to Hugging Face Spaces

1. Create a new Space on Hugging Face with Docker template
2. Push this code to the Space repository
3. The Space will automatically build and deploy the API

### Using the API

Send a POST request to the `/process` endpoint with your mobile plan data in JSON format:

```json
[
  {
    "id": 1,
    "plan_name": "Example Plan",
    "network": "5G",
    "mvno": "Example Provider",
    "basic_data": 5,
    "daily_data": 0,
    ...
  },
  ...
]
```

The API will respond with processed data, model details, and top ranked plans.
