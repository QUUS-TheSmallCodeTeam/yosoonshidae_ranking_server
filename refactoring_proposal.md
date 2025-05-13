# Mobile Plan Ranking Model Implementation Proposal

## Project Evolution

### Original State
The codebase initially used Spearman correlation for ranking mobile plans.

### Current State
The codebase currently uses the DEA (Data Envelopment Analysis) method as the ranking approach. This method evaluates the efficiency of plans by measuring their outputs (features) against their inputs (price).

### Next Phase
Our goal is to replace the DEA method with a Cost-Spec (CS) ratio method as the primary ranking approach. This new method will provide a more intuitive and transparent way to evaluate mobile plans.

## Current Codebase Structure

### Directory Structure
```
Moyo_GPTs/src/moyo_ranking_model/hf_server/
├── app.py                   # Main FastAPI application
├── modules/                 # Core functionality modules
│   ├── __init__.py          # Module exports
│   ├── config.py            # Configuration settings
│   ├── data.py              # Data loading utilities
│   ├── dea.py               # DEA implementation wrapper
│   ├── dea_run.py           # DEA runner functions
│   ├── dea_scipy.py         # Low-level DEA implementation using SciPy
│   ├── preprocess.py        # Data preprocessing functions
│   ├── ranking.py           # Ranking utilities
│   ├── report.py            # HTML report generation
│   ├── spearman.py          # Legacy Spearman method (reduced)
│   └── utils.py             # General utility functions
├── requirements.txt         # Dependencies
├── Dockerfile               # Container definition
└── data/                    # Data storage directory
```

### Key Files and Their Purpose

1. **app.py**: Main FastAPI application that defines the API endpoints:
   - `/` - Root endpoint that serves the latest HTML report
   - `/process` - Endpoint for processing plan data using DEA
   - `/test` - Simple test endpoint for health checks

2. **modules/dea.py**: Primary DEA implementation wrapper with:
   - `calculate_rankings_with_dea()` - Main function for DEA-based ranking

3. **modules/dea_scipy.py**: Low-level DEA implementation that:
   - Uses SciPy's linear programming solver
   - Calculates efficiency scores
   - Supports VRS and CRS models
   - Handles weight constraints

4. **modules/preprocess.py**: Data preprocessing with:
   - `prepare_features()` - Transforms raw data into features for analysis
   - Handles special values, unit conversions, derived features

5. **modules/report.py**: HTML report generation:
   - `generate_html_report()` - Creates interactive HTML reports
   - Formats data tables and visualizations

### Current Logical Flow

1. **Request Handling**:
   - Client sends data to `/process` endpoint with options
   - Request data is parsed and validated

2. **Data Processing**:
   - Raw data converted to DataFrame
   - `prepare_features()` cleans and transforms data
   - Feature engineering is applied (handling unlimited values, etc.)

3. **DEA Analysis**:
   - `calculate_rankings_with_dea()` is called with processed data
   - DEA model (VRS or CRS) is applied via SciPy
   - Efficiency scores are calculated
   - Plans are ranked by DEA score (higher is better)

4. **Report Generation**:
   - Results stored in global state for access by other endpoints
   - HTML report generated with explanation and interactive controls
   - Tables include plan details, efficiency scores, and rankings

5. **Response Formation**:
   - API response includes request ID, timing information
   - Top plans and all ranked plans included in response
   - Method-specific parameters included

### Key Data Structures

1. **Input Data Format**:
   - JSON array of plan objects with features
   - Options object with configuration parameters

2. **Plan Features**:
   - `basic_data` / `daily_data`: Data allowances
   - `voice` / `message`: Voice minutes and SMS counts
   - `fee` / `original_fee`: Price information
   - Various feature flags (5G, throttling, etc.)

3. **DEA Output Format**:
   - Original features plus:
   - `dea_efficiency`: Raw efficiency score (0-1)
   - `dea_score`: Ranking score (higher is better)
   - `dea_rank`: Numerical rank

## Cost-Spec Ratio Method Overview

The Cost-Spec ratio method calculates plan value through these steps:

1. **Baseline Feature Costs**: For each feature value, find the minimum fee among plans with that value
2. **Plan Baseline Cost (B)**: Sum the baseline costs for all features in the plan
3. **Cost-Spec Ratio**: Calculate CS = B / actual_fee (higher is better)
4. **Ranking**: Sort plans by CS ratio (descending)

This method objectively measures each plan's value by comparing its cost to a theoretical minimum cost for its features.

## Implementation Plan

### 1. Create Cost-Spec Ratio Module

Create a new module `modules/cost_spec.py` that implements the CS ratio calculation method, replacing the current DEA approach.

```python
"""
Cost-Spec Ratio implementation for MVNO plan ranking.

This module calculates plan rankings based on a cost-spec ratio approach
which compares each plan's actual fee with a theoretical baseline cost derived
from minimum costs of individual features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple

# Feature set definitions
FEATURE_SETS = {
    'basic': [
        'basic_data_clean', 'basic_data_unlimited',
        'daily_data_clean', 'daily_data_unlimited',
        'voice_clean', 'voice_unlimited',
        'message_clean', 'message_unlimited',
        'additional_call', 'is_5g',
        'tethering_gb', 'has_throttled_data',
        'has_unlimited_speed', 'speed_when_exhausted'
    ]
}

# Unlimited flag mappings
UNLIMITED_FLAGS = {
    'basic_data_clean': 'basic_data_unlimited',
    'daily_data_clean': 'daily_data_unlimited',
    'voice_clean': 'voice_unlimited',
    'message_clean': 'message_unlimited',
    'speed_when_exhausted': 'has_unlimited_speed'
}

def calculate_baseline_costs(df: pd.DataFrame, features: List[str], 
                           unlimited_flags: Dict[str, str], 
                           fee_column: str = 'monthly_price') -> Dict[str, pd.Series]:
    """
    Compute baseline minimum costs for each feature value.
    
    For continuous features with 'unlimited' options, we exclude those plans
    when calculating the baseline. For other features, we use all plans.
    """
    # Implementation here...

def calculate_plan_baseline_cost(row: pd.Series, features: List[str], 
                               baseline_costs: Dict[str, pd.Series]) -> float:
    """
    Calculate the theoretical baseline cost for a single plan.
    """
    # Implementation here...

def calculate_cs_ratio(df: pd.DataFrame, feature_set: str = 'basic', 
                      fee_column: str = 'monthly_price') -> pd.DataFrame:
    """
    Calculate Cost-Spec ratio for each plan in the DataFrame.
    """
    # Implementation here...

def rank_plans_by_cs(df: pd.DataFrame, feature_set: str = 'basic',
                    fee_column: str = 'monthly_price', 
                    top_n: Optional[int] = None) -> pd.DataFrame:
    """
    Rank plans by Cost-Spec ratio.
    """
    # Implementation here...
```

### 2. Modify the `/process` Endpoint to Use the CS Method

Update the `/process` endpoint in `app.py` to use the new CS ratio method instead of DEA.

```python
@app.post("/process")
async def process_data(request: Request):
    """Process plan data using the Cost-Spec ratio method."""
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
        
        # Extract options
        feature_set = options.get('featureSet', 'basic')
        fee_column = options.get('feeColumn', 'fee')  # Fee column to use for comparison
        
        # Save and preprocess data
        df = pd.DataFrame(data)
        if df.empty:
            raise HTTPException(status_code=400, detail="No data to process")
        
        processed_df = prepare_features(df)
        logger.info(f"[{request_id}] Processed DataFrame shape: {processed_df.shape}")
        
        # Apply Cost-Spec ratio method
        logger.info(f"[{request_id}] Using Cost-Spec method with feature_set={feature_set}, fee_column={fee_column}")
        df_ranked = rank_plans_by_cs(
            processed_df,
            feature_set=feature_set,
            fee_column=fee_column
        )
        
        # Store results in global state and generate report
        config.df_with_rankings = df_ranked.copy()
        
        # Generate HTML report
        timestamp_now = datetime.now()
        html_report = generate_html_report(
            df_ranked, 
            timestamp_now, 
            title="Cost-Spec Mobile Plan Rankings"
        )
        
        # Prepare response
        columns_to_include = ["id", "plan_name", "mvno", "fee", "original_fee", 
                           "rank_number", "B", "CS"]
        
        # Add ranked plans to response
        available_columns = [col for col in columns_to_include if col in df_ranked.columns]
        all_ranked_plans = df_ranked.sort_values("CS", ascending=False)[available_columns].to_dict(orient="records")
        
        response = {
            "request_id": request_id,
            "message": "Data processing complete using Cost-Spec method",
            "status": "success",
            "processing_time_seconds": round(time.time() - start_time, 4),
            "options": {
                "featureSet": feature_set,
                "feeColumn": fee_column
            },
            "ranking_method": "cs",
            "top_10_plans": all_ranked_plans[:10] if len(all_ranked_plans) >= 10 else all_ranked_plans,
            "all_ranked_plans": all_ranked_plans
        }
        
        return response
    except Exception as e:
        logger.exception(f"[{request_id}] Error in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
```

### 3. Update Report Generation

Modify the report generation to handle CS ratio specific metrics:

```python
def generate_html_report(df, timestamp, title="Mobile Plan Rankings"):
    # Existing implementation...
    
    # Add method explanation section for CS method
    html += """
    <h2>Cost-Spec Ratio Explanation</h2>
    <div class="note">
        <p><strong>Cost-Spec Ratio</strong> is a method that evaluates the value of mobile plans by comparing their fees to a theoretical baseline cost.</p>
        <p>In this analysis:</p>
        <ul>
            <li><strong>Baseline Feature Cost (E):</strong> For each feature value, the minimum fee among plans with that value</li>
            <li><strong>Plan Baseline Cost (B):</strong> Sum of baseline costs for all features in a plan</li>
            <li><strong>Cost-Spec Ratio (CS):</strong> B / fee - the ratio of theoretical cost to actual fee</li>
        </ul>
        <p>Plans are ranked based on their CS Ratio (higher is better).</p>
    </div>
    """
    
    # CS-specific columns
    html += """
    <tr>
        <th>Rank</th>
        <th>Plan Name</th>
        <th>MVNO</th>
        <th>Fee (KRW)</th>
        <th>Baseline Cost (B)</th>
        <th>CS Ratio</th>
        <!-- Other columns... -->
    </tr>
    """
    
    # Generate table rows for CS method
    for _, row in df.sort_values('rank_number').iterrows():
        # Format row data for CS method...
        html += f"""
        <tr>
            <td>{rank_str}</td>
            <td>{plan_name}</td>
            <td>{mvno}</td>
            <td>{fee:,}</td>
            <td>{baseline_cost:,}</td>
            <td class="good-value">{cs_ratio:.2f}</td>
            <!-- Other columns... -->
        </tr>
        """
```

### 4. Update the `__init__.py` to Export CS Functions

Update the module exports in `modules/__init__.py`:

```python
from .cost_spec import rank_plans_by_cs, calculate_cs_ratio
from .report import generate_html_report
from .preprocess import prepare_features
from .utils import ensure_directories, save_raw_data, save_processed_data

__all__ = [
    'rank_plans_by_cs',
    'calculate_cs_ratio',
    'generate_html_report',
    'prepare_features',
    'ensure_directories',
    'save_raw_data',
    'save_processed_data'
]
```

### 5. Update App Title and Documentation

Update the FastAPI app title and documentation to reflect the new method:

```python
# Initialize FastAPI
app = FastAPI(title="Moyo Plan Ranking Model Server")
```

## Implementation Process

1. Create `modules/cost_spec.py` with CS ratio implementation
2. Update `app.py` to replace DEA with the CS ratio method
3. Update report generation to handle CS ratio metrics
4. Update `__init__.py` to export the new functions
5. Remove DEA-specific code and references
6. Test with sample data

## Expected Results

1. Complete replacement of the DEA method with the Cost-Spec ratio method
2. Consistent API interface for all ranking operations
3. Clear explanations of the CS method's approach in HTML reports
4. Improved interpretability of ranking results

## Implementation Challenges

1. **Handling Unlimited Features**: The CS method requires special handling for unlimited features, ensuring that unlimited plans are excluded when calculating baseline costs for continuous features.

2. **Response Format Changes**: We need to update the response format to reflect the CS method metrics instead of DEA metrics.

3. **Visualization Updates**: Different metrics will be displayed for the CS method, requiring appropriate changes to the HTML report templates.

## Verification Steps

After implementation, we will verify:

1. That the CS method produces mathematically correct rankings
2. That the HTML reports accurately present the results
3. That the API endpoints maintain expected functionality
4. That error handling is robust


