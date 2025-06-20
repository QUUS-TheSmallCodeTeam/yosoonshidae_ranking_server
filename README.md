---
title: Mvno Plan Ranking Model
emoji: 🌖
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
---

# Moyo Mobile Plan Ranking API

This FastAPI application provides a comprehensive API for preprocessing and ranking mobile phone plans using advanced Cost-Spec ratio methodology with file-based multiprocessing architecture.

## API Endpoints

### Core Endpoints
- **GET `/`**: Interactive HTML interface with ranking tables, charts, and feature coefficient analysis
- **POST `/process`**: Main data processing endpoint - handles complete mobile plan analysis pipeline
- **POST `/test`**: Test endpoint for API validation

### Chart & Status Endpoints  
- **GET `/chart-status`**: Overall chart calculation status
- **GET `/chart-status/{chart_type}`**: Individual chart calculation status
- **GET `/chart-data/{chart_type}`**: Retrieve specific chart data
- **GET `/status`**: System status page with processing information

### Debug Endpoints
- **GET `/test-reload`**: Test system reload functionality
- **GET `/debug-global`**: Debug global state and file-based storage

## Key Features

### Advanced Ranking Algorithm
- **Fixed Rates Method**: Uses pure marginal coefficients from entire dataset analysis
- **File-Based Storage**: Multiprocessing-compatible data sharing via `/app/data/shared/` directory
- **Async Chart Generation**: Background chart calculation with real-time status indicators
- **Comprehensive Feature Analysis**: 16+ features including unlimited plans, 5G support, and data throttling

### Data Processing Pipeline
1. **Raw Data Ingestion**: JSON plan data with comprehensive feature extraction
2. **Feature Engineering**: Advanced preprocessing with unlimited flag handling
3. **Coefficient Calculation**: FullDatasetMultiFeatureRegression with multicollinearity handling
4. **CS Ratio Computation**: Baseline cost vs actual fee analysis
5. **Ranking Generation**: Tie-aware ranking with Korean notation support
6. **Report Generation**: HTML reports with interactive charts and coefficient tables

## Requirements

```
fastapi==0.115.12
uvicorn[standard]==0.34.1
pydantic==2.11.3
pandas==2.2.3
numpy==1.26.4
matplotlib==3.10.1
scikit-learn==1.6.1
jinja2
python-multipart==0.0.6
psutil==6.1.1
```

## Directory Structure

```
app/
├── app.py                   # Main FastAPI application with async processing
├── Dockerfile               # Hugging Face Spaces deployment
├── requirements.txt         # Python dependencies
├── simple_log_monitor.sh    # Log monitoring script
├── memory.md               # System status and context
├── todolist.md            # Task management
├── modules/                # Modular code organization
│   ├── __init__.py         # Module exports
│   ├── config.py           # Configuration management
│   ├── data_storage.py     # File-based data persistence (multiprocessing)
│   ├── data.py             # Data loading functions
│   ├── data_models.py      # Pydantic data models
│   ├── preprocess.py       # Advanced feature engineering
│   ├── cost_spec.py        # Cost-Spec ratio calculation (121KB, 2746 lines)
│   ├── ranking.py          # Ranking display logic
│   ├── report_html.py      # HTML report generation (96KB, 2058 lines)
│   ├── report_charts.py    # Chart generation (86KB, 1825 lines)
│   ├── report_tables.py    # Table generation
│   ├── report_utils.py     # Report utilities
│   ├── report.py           # Report coordination
│   ├── piecewise_regression.py  # Piecewise linear modeling
│   ├── categorical_handlers.py # Categorical data processing
│   ├── models.py           # Feature definitions
│   └── utils.py            # Utility functions
├── data/
│   ├── raw/                # Raw input data files
│   ├── processed/          # Processed data cache
│   ├── shared/             # File-based multiprocessing storage
│   └── test/               # Test datasets
├── trained_models/         # Model persistence
├── results/                # Analysis results
│   ├── latest/             # Current results
│   └── archive/            # Historical results
└── reports/                # Generated reports
```

## Deployment

### Hugging Face Spaces (Production)
1. **Automatic Deployment**: Push to Space repository triggers Docker build
2. **Container Setup**: Python 3.11 with optimized dependencies
3. **Directory Creation**: Automatic setup of data, model, and report directories
4. **Port Configuration**: Uvicorn server on port 7860
5. **Log Monitoring**: Automated background log monitoring with filtering

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app:app --reload --host 0.0.0.0 --port 7860

# Monitor logs (if script available)
chmod +x simple_log_monitor.sh
./simple_log_monitor.sh &
```

## API Usage

### Processing Mobile Plan Data

Send POST request to `/process` with plan data:

```json
[
  {
    "id": 1,
    "plan_name": "Example Plan",
    "network": "5G",
    "mvno": "Example Provider",
    "mno": "SKT",
    "basic_data": 5,
    "daily_data": 0,
    "data_exhaustion": "1Mbps",
    "voice": 300,
    "message": 100,
    "additional_call": 0,
    "data_sharing": false,
    "roaming_support": true,
    "micro_payment": false,
    "is_esim": true,
    "signup_minor": false,
    "signup_foreigner": false,
    "has_usim": true,
    "has_nfc_usim": false,
    "tethering_gb": 2,
    "tethering_status": "included",
    "esim_fee": 0,
    "usim_delivery_fee": 3000,
    "fee": 29000,
    "original_fee": 35000,
    "discount_fee": 6000,
    "discount_period": 12,
    "post_discount_fee": 35000,
    "agreement": true,
    "agreement_period": 24,
    "agreement_type": "standard",
    "num_of_signup": 1,
    "mvno_rating": 4.2,
    "monthly_review_score": 3.8,
    "discount_percentage": 17.1
  }
]
```

### Response Structure

The API returns:
- **Immediate JSON response** with ranked plans and CS ratios
- **Background chart generation** with real-time status updates
- **File-based storage** for persistent data sharing

```json
{
  "request_id": "uuid-string",
  "processing_time": 0.45,
  "method": "fixed_rates",
  "fee_type": "original_fee",
  "paths": {
    "raw_data": "/app/data/raw/data_20241219_164523.json",
    "processed_data": "/app/data/processed/processed_20241219_164523.json",
    "report": "/app/reports/report_20241219_164523.html"
  },
  "top_plans": [...],
  "all_ranked_plans": [...]
}
```

## Mathematical Foundation & Algorithm Design

### Core Mathematical Principles

The system implements advanced mathematical modeling for mobile plan value assessment based on established economic and statistical principles.

#### 1. Marginal Cost Theory

**Economic Foundation**:
- **Frontier Purpose**: Trend learning by selecting minimum cost at each feature level, removing overpriced plans
- **Piecewise Beta Coefficients**: Reflects economies of scale (cost per 1GB ≠ cost per 100th GB)
- **No Interaction Effects**: Maintains interpretability and prevents model complexity
- **1 KRW/Feature Rule**: Ensures monotonicity in cost-feature relationships

**Mathematical Formulation**:
```
Marginal_Cost(feature_level) = {
  β₁, if 0 ≤ feature_level < threshold₁
  β₂, if threshold₁ ≤ feature_level < threshold₂
  β₃, if feature_level ≥ threshold₂
}
```

Where thresholds are determined by gradient change points in the cost curve.

#### 2. Feature Frontier Analysis

**Monotonic Frontier Construction**:
For each feature f, construct frontier F_f where:
```
F_f = {(x, min_cost(x)) | x ∈ feature_values, cost(x) is monotonically increasing}
```

**Cross-Contamination Problem**:
- **Issue**: Frontier point prices contain value from multiple features
- **Solution**: Multi-feature simultaneous regression on entire dataset
- **Improvement**: Pure independent feature value estimation

#### 3. FullDatasetMultiFeatureRegression Algorithm

**Regression Formulation**:
```
Price = β₀ + Σᵢ βᵢ × featureᵢ + ε

Where:
- β₀: Base cost (intercept)
- βᵢ: Marginal cost coefficient for feature i
- featureᵢ: Normalized feature value
- ε: Error term
```

**Constraint Optimization**:
```
minimize: ||Xβ - y||² + α||β||²
subject to: βᵢ ≥ 0 for economic features
           βⱼ ∈ ℝ for unlimited flags
```

**Multicollinearity Handling**:
When correlation(fᵢ, fⱼ) > 0.8:
```
β'ᵢ = β'ⱼ = (βᵢ + βⱼ) / 2
```

#### 4. Cost-Spec Ratio Calculation

**Fixed Rates Method**:
```
Baseline_Cost = β₀ + Σᵢ βᵢ × feature_valueᵢ

CS_Ratio = Baseline_Cost / Actual_Fee

Value_Score = {
  "Excellent": CS_Ratio > 1.2
  "Good": 1.0 < CS_Ratio ≤ 1.2
  "Fair": 0.8 < CS_Ratio ≤ 1.0
  "Poor": CS_Ratio ≤ 0.8
}
```

#### 5. Unlimited Feature Modeling

**Boolean Flag Approach**:
```
Unlimited_Value = {
  βᵢ × multiplier, if unlimited_flagᵢ = 1
  0, if unlimited_flagᵢ = 0
}

Continuous_Value = {
  βᵢ × actual_value, if unlimited_flagᵢ = 0
  0, if unlimited_flagᵢ = 1  # Prevent double counting
}
```

**Unlimited Type Classification**:
```
Unlimited_Type = {
  3: unlimited_speed (data AND speed unlimited)
  2: throttled_unlimited (quota → throttled)
  1: unlimited_with_throttling (always throttled)
  0: limited (service stops after quota)
}
```

### Advanced Implementation Details

#### 1. Categorical Feature Processing

**CategoricalFeatureHandler** (`categorical_handlers.py`):
- **Dummy Variable Encoding**: One-hot encoding for categorical features
- **Effect Coding**: Sum-to-zero constraints for balanced coefficients
- **Cost-Based Encoding**: Premium calculation using actual cost differences
- **Mixed Modeling**: Unlimited as very high continuous values

#### 2. Piecewise Linear Regression

**PiecewiseLinearRegression** (`piecewise_regression.py`):
- **Automatic Breakpoint Detection**: Gradient analysis for change points
- **Segment-Based Coefficients**: Different marginal costs per feature range
- **Economies of Scale**: Decreasing marginal costs at higher usage levels

#### 3. Korean Tie Ranking System

**Ranking Algorithm** (`ranking.py`):
```python
def calculate_rankings_with_ties(df, value_column='CS'):
    """
    Korean tie notation: '공동 X위' (joint X rank)
    Proper rank incrementing after tied groups
    """
    # Group by rounded CS ratio values
    value_groups = df.groupby(df[value_column].round(10)).indices
    
    # Assign ranks with tie handling
    for value, indices in sorted(value_groups.items(), reverse=True):
        if len(indices) > 1:  # Tied plans
            for idx in indices:
                display_ranks[idx] = f"공동 {current_rank}위"
        else:  # Single plan
            display_ranks[indices[0]] = f"{current_rank}위"
        
        current_rank += len(indices)  # Skip positions for ties
```

#### 4. File-Based Multiprocessing Architecture

**Data Storage System** (`data_storage.py`):
```python
# Storage configuration
STORAGE_DIR = Path("/app/data/shared")
FILES = {
    'rankings': "rankings.json",
    'cost_structure': "cost_structure.json", 
    'metadata': "metadata.json",
    'charts': "charts.json"
}

# Save/load operations with error handling
def save_rankings_data(df, cost_structure, method, charts_data):
    """Serialize DataFrame and metadata to JSON files"""
    
def load_rankings_data():
    """Load DataFrame and metadata from JSON files"""
    return df_with_rankings, cost_structure, method, charts_data
```

#### 6. Piecewise Linear Modeling

**Cumulative Cost Calculation**:
```
Cumulative_Cost(x) = Σᵢ₌₁ⁿ βᵢ × segment_widthᵢ

Where:
segment_widthᵢ = min(x, threshold_{i+1}) - threshold_i
```

**Change Point Detection**:
Uses gradient analysis to identify natural breakpoints:
```
Change_Point = argmax |∇Cost(x)|
```

#### 7. Statistical Validation Framework

**Multicollinearity Detection**:
```
VIF_i = 1 / (1 - R²_i)

Where R²_i is from regression of feature_i on all other features
```

**Ridge Regression Activation**:
```
Regularization = {
  Ridge(α=10.0), if max(correlation_matrix) > 0.8
  LinearRegression(), otherwise
}
```

**Outlier Detection**:
```
Z_Score = |x - μ| / σ
Outlier = Z_Score > 3.0
```

### Advanced Features

#### 1. Coefficient Transparency

**Mathematical Steps Display**:
```
Raw_OLS → Constraint_Application → Multicollinearity_Redistribution
β_raw → β_constrained → β_final

Example: "(70.2 + 49.8) / 2 = 60.0" for correlated features
```

#### 2. Economic Bounds

**Feature-Specific Constraints**:
```
Data_Features: β ≥ 0 (positive marginal cost)
Unlimited_Flags: β ∈ [-∞, +∞] (can represent discounts)
Network_Features: β ≥ 0 (premium for better service)
```

#### 3. Ranking Algorithm

**Tie-Aware Ranking**:
```
Rank(plan_i) = |{plan_j : CS_Ratio_j > CS_Ratio_i}| + 1

For ties: Rank_Display = "공동 " + min(tied_ranks) + "위"
```

## Advanced Ranking Algorithm

### Cost-Spec Ratio Method (Enhanced)

The system uses a sophisticated **Fixed Rates** methodology:

#### 1. Feature Engineering Pipeline

**Data Preprocessing** (`prepare_features()`):
- **Network Analysis**: 5G support detection and binary encoding
- **Data Throttling**: Speed extraction from `data_exhaustion` fields
- **Unlimited Classification**: Multi-tier unlimited data categorization
- **Voice/Message Processing**: Unlimited flag creation and value cleaning
- **Price Feature Engineering**: Per-GB pricing and discount ratio calculation

**Unlimited Data Classification**:
- `unlimited_speed` (3): Unlimited data AND speed
- `throttled_unlimited` (2): Full speed until quota, then throttled  
- `unlimited_with_throttling` (1): Always throttled despite unlimited data
- `limited` (0): Service stops after quota

#### 2. Coefficient Calculation

**FullDatasetMultiFeatureRegression**:
- **Comprehensive Analysis**: Uses entire dataset (2000+ plans) instead of frontier points
- **Multicollinearity Handling**: Automatic Ridge regression when correlations > 0.8
- **Constraint Optimization**: Economic bounds with positive coefficient enforcement
- **Outlier Removal**: Z-score based outlier detection (threshold: 3.0)

**Feature Set** (16 features):
```
['basic_data_clean', 'basic_data_unlimited', 'daily_data_clean', 'daily_data_unlimited',
 'voice_clean', 'voice_unlimited', 'message_clean', 'message_unlimited', 
 'additional_call', 'is_5g', 'tethering_gb', 'speed_when_exhausted',
 'data_throttled_after_quota', 'data_unlimited_speed', 'has_unlimited_speed']
```

#### 3. CS Ratio Calculation

**Fixed Rates Method**:
```
baseline_cost = β₀ + β₁×data + β₂×voice + β₃×SMS + β₄×tethering + β₅×5G + ...
cs_ratio = baseline_cost / actual_fee
```

Where β coefficients represent pure marginal costs from regression analysis.

#### 4. Ranking System

**Tie-Aware Ranking**:
- Plans ranked by CS ratio (descending)
- Korean tie notation: '공동 X위' (joint X rank)
- Proper rank incrementing after tied groups
- Value ratio calculation for comparative analysis

### File-Based Architecture

**Multiprocessing Compatibility**:
- **Storage Location**: `/app/data/shared/` directory
- **Data Files**: `rankings.json`, `cost_structure.json`, `metadata.json`
- **Process Flow**: `/process` saves → `/` loads → Background charts update
- **Serialization**: pandas DataFrame → JSON with metadata preservation

### Chart Generation System

**Available Chart Types**:
1. **Feature Frontier Charts**: Market-based cost trends by feature
2. **Plan Efficiency Charts**: Value ratio distribution and rankings

**Async Processing**:
- **Immediate API Response**: Rankings available instantly
- **Background Calculation**: Charts computed asynchronously  
- **Status Tracking**: Real-time progress indicators with individual chart monitoring
- **Manual Refresh**: No auto-polling, user-controlled updates

**Chart Data Processing**:
```python
# Synchronous chart calculation for file storage
charts_data = calculate_and_save_charts(df_ranked, method, cost_structure)

# Chart status tracking
chart_statuses = {
    'feature_frontier': {'status': 'calculating', 'progress': 45},
    'plan_efficiency': {'status': 'completed', 'timestamp': '2025-01-19T...'}
}
```

### HTML Report Features

**Interactive Interface**:
- **Ranking Table**: Complete plan rankings with CS ratios and value metrics
- **Coefficient Table**: Feature marginal costs with unconstrained vs constrained comparison
- **Chart Visualization**: Interactive charts with hover tooltips and zoom
- **Status Indicators**: Loading (⚙️) and error (❌) states for chart sections
- **Refresh Controls**: Manual page refresh button (🔄 새로고침)

**Technical Implementation**:
- **No Caching**: Fresh content generation on each request
- **Error Resilience**: Graceful degradation when files unavailable
- **Mobile Responsive**: Adaptive design for different screen sizes
- **Korean Localization**: Korean text and formatting throughout interface

**Report Generation** (`report_html.py`):
```python
# Main report generation with chart status integration
generate_html_report(df, timestamp, report_title, is_cs=True, 
                    method=method, cost_structure=cost_structure,
                    chart_statuses=chart_statuses, charts_data=charts_data)

# Feature coefficient table with transparency
generate_feature_rates_table_html(cost_structure)

# Plan efficiency data for visualizations  
prepare_plan_efficiency_data(df, method)
```

## System Architecture

### Multiprocessing Design
- **File-Based Storage**: Eliminates global variable sharing issues
- **Process Isolation**: Each FastAPI worker operates independently
- **Data Persistence**: Reliable inter-process communication via filesystem
- **Error Recovery**: Graceful handling of missing or corrupted data files

### Performance Optimizations
- **Async Background Tasks**: Chart calculations don't block API responses
- **Efficient Serialization**: Optimized JSON handling with numpy type conversion via `NumpyEncoder`
- **Memory Management**: Garbage collection and resource cleanup
- **Log Monitoring**: Automated log management with HF Space polling filter

### Development Environment
- **Docker Container**: Python 3.11 with optimized dependency installation
- **Dev Mode Support**: Hugging Face Spaces development mode compatibility
- **Log Monitoring**: Background log monitoring with SSH polling filtration
- **Debug Endpoints**: Comprehensive debugging tools for development

## Testing

### End-to-End Testing
```bash
# Test with latest raw data file
curl -X POST http://localhost:7860/process \
  -H "Content-Type: application/json" \
  -d @$(ls -t data/raw/*.json | head -1)

# Check processing status
curl http://localhost:7860/chart-status

# View results
open http://localhost:7860/
```

### Log Monitoring
```bash
# Monitor filtered logs (recommended)
./simple_log_monitor.sh &
tail -f error.log

# Monitor raw server logs (debugging)
PID=$(ps aux | grep "python.*uvicorn" | grep -v grep | awk '{print $2}' | head -1)
cat /proc/$PID/fd/1
```

## Troubleshooting

### Common Issues

**Chart Status "Calculating"**:
- Charts compute asynchronously after API response
- Use manual refresh button to check progress
- Check `/chart-status` endpoint for detailed status

**Empty Ranking Table**:
- Verify `/process` endpoint was called successfully
- Check file permissions in `/app/data/shared/` directory
- Use `/debug-global` endpoint to inspect storage state

**Memory Issues**:
- Monitor with `psutil` integration
- Check for outlier data causing processing issues
- Verify garbage collection effectiveness

### Performance Monitoring

**Key Metrics**:
- API response time < 500ms (excluding charts)
- Chart generation < 30 seconds
- Memory usage < 2GB for typical datasets
- Error rate < 1% for valid input

**Health Checks**:
- `/status` endpoint for system overview
- `/chart-status` for background task monitoring
- Log analysis for error patterns and performance issues

## Contributing

### Development Workflow
1. **Local Setup**: Clone repository and install dependencies
2. **Feature Development**: Use provided debug endpoints for testing
3. **End-to-End Testing**: Test with actual mobile plan data
4. **Documentation**: Update this README with any architectural changes

### Code Organization
- **Modular Design**: Each module has specific responsibility
- **Type Hints**: Comprehensive typing throughout codebase
- **Error Handling**: Graceful degradation and informative logging
- **Performance**: Async operations and efficient data structures

### Key Modules Overview

**Core Analysis** (`cost_spec.py`):
- `LinearDecomposition`: Constrained optimization approach
- `MultiFeatureFrontierRegression`: Frontier-based analysis
- `FullDatasetMultiFeatureRegression`: Comprehensive dataset regression

**Data Processing** (`preprocess.py`):
- Feature engineering pipeline with unlimited flag handling
- Data type conversion and validation
- Missing value imputation and outlier detection

**Visualization** (`report_charts.py`, `report_html.py`):
- Interactive chart generation with Chart.js integration
- HTML report templates with Korean localization
- Async chart calculation with status tracking

**Utilities**:
- `ranking.py`: Korean tie notation ranking algorithm
- `data_storage.py`: File-based multiprocessing data sharing
- `categorical_handlers.py`: Advanced categorical feature processing
- `piecewise_regression.py`: Piecewise linear modeling capabilities

This system provides a production-ready solution for mobile plan ranking with advanced mathematical modeling, robust multiprocessing architecture, and comprehensive user interface capabilities.
