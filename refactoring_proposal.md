# DEA Implementation Proposal

## Current State Analysis

The codebase currently uses Spearman correlation for ranking. We need to replace it completely with DEA (Data Envelopment Analysis) as the primary ranking method.

## Implementation Plan

### 1. Replace Spearman with DEA in the `/process` Endpoint

The current `/process` endpoint uses the Spearman method for ranking. We'll modify it to use the DEA method instead, leveraging the existing DEA implementation in `modules/dea.py` and `modules/dea_scipy.py`.

#### Modifications to `app.py`:

```python
@app.post("/process")
async def process_data(request: Request):
    """Process plan data using the DEA method."""
    start_time = time.time()
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received /process request")
    
    try:
        # Parse request data and options
        request_json = await request.json()
        
        # Extract data and options
        if isinstance(request_json, dict):
            options = request_json.get('options', {})
            data = request_json.get('data', [])
            if not isinstance(data, list):
                data = request_json
                options = {}
        else:
            data = request_json
            options = {}
        
        # Extract DEA options with defaults
        feature_set = options.get('featureSet', 'basic')
        target_variable = options.get('targetVariable', 'fee')
        rts = options.get('rts', 'vrs')  # Default to VRS for better discrimination
        
        # Preprocess data
        df = pd.DataFrame(data)
        processed_df = prepare_features(df)
        
        # Run DEA analysis
        df_ranked = calculate_rankings_with_dea(
            processed_df,
            feature_set=feature_set,
            target_variable=target_variable,
            rts=rts
        )
        
        # Generate report and prepare response
        # ...
    
        return response
    except Exception as e:
        logger.exception(f"[{request_id}] Error in /process: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")
```

### 2. Remove the CSV Upload Endpoint

Since we're completely replacing the Spearman method with DEA, we'll remove the `/upload-csv` endpoint and consolidate all functionality into the `/process` endpoint. This will simplify the API and provide a consistent interface for all ranking operations.

### 3. Update the App Title and Documentation

Update the FastAPI app title and documentation to reflect the change to DEA:

```python
# Initialize FastAPI
app = FastAPI(title="Moyo Plan Ranking Model Server - DEA Method")
```

### 4. Update Report Generation

Modify the report generation to ensure it properly displays DEA-specific metrics:

```python
# Generate HTML content
html_report = generate_html_report(df_ranked, timestamp_now)

# Write HTML content to file
report_path = config.dea_report_dir / f"dea_ranking_{timestamp_now.strftime('%Y%m%d_%H%M%S')}.html"
with open(report_path, 'w', encoding='utf-8') as f:
    f.write(html_report)
```
    return FileResponse(report_path)
```

### Updated HTML Template
Update the root endpoint to include CSV upload form:

```html
<form action="/upload-csv" method="post" enctype="multipart/form-data">
    <input type="file" name="csv_file" accept=".csv">
    <button type="submit">Upload CSV for DEA Analysis</button>
</form>

<!-- Existing report display code -->
```

## Implementation Process

1. Create `modules/dea.py` with DEA implementation
2. Add CSV upload endpoint to `app.py`
3. Update HTML template
4. Test DEA ranking functionality

## Expected Results

1. New DEA ranking method available alongside Spearman
2. CSV upload functionality for DEA analysis
3. Consistent report generation with DEA results

## Files to Remove or Significantly Reduce

The following XGBoost-related code should be removed:

1. Any XGBoost model loading/saving functions in `app.py`
2. XGBoost-specific ranking functions not used by the Spearman method
3. Model training code that's not related to the Spearman method

Only the code absolutely necessary to maintain the current functionality of the '/' and '/process' endpoints should remain.
