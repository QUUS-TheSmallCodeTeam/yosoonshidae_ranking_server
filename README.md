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

This FastAPI application provides an API for preprocessing and ranking mobile phone plans based on various features using the Cost-Spec ratio method.

## API Endpoints

- **GET `/`**: Welcome message
- **POST `/test`**: Test endpoint - echoes back the request body
- **POST `/process`**: Main endpoint - processes plan data through the complete pipeline:
  1. Processes and saves raw data
  2. Applies feature engineering
  3. Calculates Cost-Spec ratios
  4. Ranks plans and generates reports

## Requirements

- Python 3.9+
- FastAPI
- Uvicorn
- pandas
- numpy
- scikit-learn

## Directory Structure

```
hf_server/
├── app.py               # Main FastAPI application
├── Dockerfile           # For Hugging Face Spaces deployment
├── modules/             # Modular code organization
│   ├── __init__.py      # Module exports
│   ├── data.py          # Data loading functions
│   ├── models.py        # Feature definitions
│   ├── preprocess.py    # Feature engineering
│   ├── ranking.py       # Ranking display logic
│   ├── report.py        # Report generation
│   ├── cost_spec.py     # Cost-Spec ratio calculation
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

The API will respond with processed data, ranking details, and top ranked plans.

## Ranking Algorithm Explained

This section details how the Moyo GPTs ranking model processes data and calculates rankings.

### 1. Data Flow Overview

The ranking model follows this general workflow:

1. Receive plan data via API request
2. Preprocess the data to prepare features for analysis
3. Apply the Cost-Spec ratio method
4. Calculate value metrics and rankings
5. Generate reports and return results

### 2. Data Reception and Parsing

When a request is sent to the `/process` endpoint:

- The API accepts JSON data containing a list of plan objects
- Each plan object contains details like plan name, carrier, data allowance, voice, messaging, fees, etc.
- Optional ranking parameters can be specified:
  - `featureSet`: 'basic' (default) or 'extended'
  - `feeColumn`: 'original_fee' or 'fee'

### 3. Data Preprocessing

The `prepare_features()` function transforms raw plan data into model-ready features:

#### 3.1 Numeric Conversion

- Ensures fields like `basic_data`, `daily_data`, `tethering_gb` are numeric
- Converts any MB tethering values to GB for consistency

#### 3.2 Network Type Processing

- Creates `is_5g` binary feature from `network` field

#### 3.3 Data Throttling Analysis

- Extracts speed values from `data_exhaustion` field (e.g., "1Mbps" → 1)
- Creates `speed_when_exhausted` and `has_throttled_data` features

#### 3.4 Data Allowance Processing

- Identifies unlimited plans (values 999, 9999)
- Creates `basic_data_unlimited` and `daily_data_unlimited` binary features
- Replaces unlimited markers with maximum finite values observed
- Calculates `total_data` by combining basic and daily allowances

#### 3.5 Unlimited Data Classification

- `has_unlimited_data`: Plans with unlimited basic or daily data
- `has_unlimited_speed`: Plans with unlimited data and no throttling
- `any_unlimited_data`: Any type of unlimited data (including throttled)
- `unlimited_type`: Detailed classification (text)
- `unlimited_type_numeric`: Numeric encoding (0-3)
  - 3: unlimited_speed (unlimited data AND speed)
  - 2: throttled_unlimited (full speed until quota, then throttled)
  - 1: unlimited_with_throttling (always throttled despite unlimited data)
  - 0: limited (service stops after quota)

#### 3.6 Throttled Speed Normalization

- Creates `throttle_speed_normalized` (0-1 range)
- Maximum throttle speed is set to 10.0 Mbps

#### 3.7 Voice and Message Processing

- Identifies unlimited voice/messages (values containing "999")
- Creates `voice_clean`, `voice_unlimited`, `message_clean`, `message_unlimited`
- Handles special values (-1 = not applicable)

#### 3.8 SIM Card Feature Processing

- Creates features for USIM/eSIM support and pricing

#### 3.9 Price Features

- Calculates `price_per_gb` for plans with finite data
- Computes `price_unlimited_adjusted` for unlimited plans
- Derives `discount_ratio` from original and discounted fees

#### 3.10 Data Cleaning

- Replaces infinities with NaN
- Imputes missing values with column medians

### 4. Cost-Spec Ratio Method

The Cost-Spec ratio method calculates plan value as follows:

#### 4.1 Feature Selection

- Uses a predefined set of basic features like data allowances, voice, messaging, etc.
- Features are selected based on their core value contribution to plans

#### 4.2 Baseline Cost Calculation

- For each feature (e.g., data, voice, messages), identifies the minimum cost in the market
- Creates a feature frontier based on value vs. cost
- Calculates baseline cost (B) by estimating the minimum market cost for each feature

#### 4.3 Cost-Spec Ratio Calculation

- For each plan, calculates the Cost-Spec ratio (CS = B / fee)
- CS ratio > 1 indicates good value (baseline cost higher than actual cost)
- CS ratio < 1 indicates poor value (baseline cost lower than actual cost)

#### 4.4 Ranking by CS Ratio

- Plans are ranked by CS ratio in descending order
- Higher CS ratio = better value for money

### 5. Value Metrics and Rankings

The model calculates multiple value metrics:

#### 5.1 Value Metrics

- **CS Ratio**: Baseline cost / actual fee
- **Value Differential**: Baseline cost - actual fee

#### 5.2 Ranking Calculation

- Plans are ranked by CS ratio in descending order
- The function `calculate_rankings_with_ties()` handles proper ranking with ties
- For tied ranks, uses '공동 X위' (joint X rank) notation
- All plans in a tied group receive the same numeric rank
- Ranks are properly incremented after tied groups

#### 5.3 Feature Contributions

- For each feature, the model calculates its contribution to the final score
- Contribution = sign(ρ) * weight * normalized_value * price_std

### 6. Response Structure

The API response includes:

- Processing metadata (request ID, processing time)
- Used ranking options (method, fee type, log transform)
- Path information (raw data, processed data, report)
- Top 10 plans by the selected ranking method
- All ranked plans in a format compatible with database upsert operations

The `all_ranked_plans` array contains:
- Plan identifiers and basic info
- Original and discounted fee
- Predicted price
- Rank and rank display (with tie notation)
- Value ratio for comparing plan value

### 7. HTML Report Generation

The model generates a comprehensive HTML report with:
- Summary statistics (total plans, average value ratio)
- Plan rankings with detailed metrics
- Value distribution visualization
- Complete feature contribution analysis

This report serves as a visual representation of the ranking results for analysis.
