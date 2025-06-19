# 🧠 Memory & Context

## 📊 Current System Status
- **File-based data storage**: Implemented complete solution for multiprocessing memory sharing using `/app/data/shared/` directory
- **Ranking table display**: Fixed "데이터 처리 대기 중" issue - now shows actual ranking data
- **Multiprocessing compatibility**: Process-to-process data sharing via file system instead of memory
- **Refresh button functionality**: Fixed AttributeError when df_with_rankings is None, now works in all states
- **Async chart calculation**: Implemented to eliminate continuous calculations triggered by root endpoint
- **Visual status indicators**: Loading icons (⚙️) for in-progress, error icons (❌) for failed calculations
- **Manual refresh system**: No auto-polling, users manually refresh to check progress
- **No caching**: All HTML content generated fresh on each request for immediate status updates
- **Multi-frontier regression methodology**: Successfully implemented and fully operational
- **Chart visualization**: Advanced charts now calculated asynchronously in background
- **API response time**: Immediate response from /process endpoint, charts calculated separately
- **Default method**: Changed to `fixed_rates` for consistent coefficient calculation
- **Feature coefficient calculation**: Successfully includes voice_unlimited and message_unlimited in regression analysis
- **Chart data format**: Fixed JavaScript functions to handle nested cost structure objects properly
- **Marginal Cost Frontier Charts**: Successfully implemented feature-level trend visualization using pure marginal costs from multi-frontier regression
- **✅ PIECEWISE LINEAR MODEL IMPLEMENTED**: Replaced simple linear model with realistic piecewise segments showing economies of scale
- **✅ MONOTONIC FILTERING APPLIED**: Same robust monotonic frontier logic with 1 KRW/feature rule as original system
- **✅ UNLIMITED HANDLING COMPLETE**: Separate processing of unlimited plans with proper endpoints
- **✅ UNLIMITED AS FLAGS ONLY**: Unlimited features processed as boolean flags, not continuous data points in marginal cost trendlines
- **✅ DOUBLE FILTERING FIXED**: Eliminated double filtering - monotonicity applied only to trendline, not raw market data
- **✅ FULL DATASET ANALYSIS IMPLEMENTED**: Switched from frontier points to entire dataset regression for comprehensive analysis
- **✅ 5G FEATURE ADDED**: Added is_5g to core_continuous_features for complete feature coverage
- **✅ LINEAR DECOMPOSITION COMPLETELY REMOVED**: Removed all Linear Decomposition Analysis sections, functions, and references
- **✅ CUMULATIVE COST CALCULATION FIXED**: Charts now plot cumulative costs instead of fixed marginal rates
- **✅ PIECEWISE SEGMENTS PROPERLY IMPLEMENTED**: Using fit_cumulative_piecewise_linear for realistic cost accumulation
- **✅ MULTI-FEATURE FRONTIER REGRESSION ANALYSIS REMOVED**: Deleted entire section from HTML and related calculation code per user request
- **✅ FIXED RATES METHOD IMPLEMENTED**: New ranking calculation using pure coefficients for entire dataset without filtering
- **✅ UNLIMITED FLAGS IN REGRESSION**: voice_unlimited and message_unlimited now properly included in Feature Marginal Cost Coefficients table
- **✅ MULTICOLLINEARITY ISSUE RESOLVED**: Removed problematic features from coefficient calculation pipeline ('data_stops_after_quota' and other highly correlated features), switched from Ridge to LinearRegression, enforced positive coefficient bounds for key features
- **✅ ENHANCED COEFFICIENT TABLE**: Added unconstrained (raw OLS) vs constrained (bounded) coefficient comparison with color-coded adjustments
- **✅ ASYNC PROCESSING SEQUENCE VERIFIED**: Response returned immediately after ranking calculation, validation and charts run in background only
- **✅ DETAILED CALCULATION FORMULAS**: HTML coefficient table now shows exact mathematical steps including multicollinearity redistribution formulas
- **F-string backslash error fixed**: HTML JavaScript code in f-string was using backslashes directly, moved to variable for proper syntax
- **✅ MULTIPROCESSING MEMORY SHARING SOLVED**: Implemented file-based data storage system to replace global variable sharing
- **✅ FILE-BASED STORAGE ARCHITECTURE**: Created data_storage.py module with save/load functions for DataFrame and cost structure
- **✅ PROCESS-TO-PROCESS DATA SHARING**: Uses file system (/app/data/shared/) for reliable data exchange between FastAPI processes
- **Process endpoint**: ✅ Working correctly - returns 1000+ ranked plans with CS ratios (JSON response successful) + saves to files
- **Root endpoint**: ✅ Fixed - loads data from files instead of relying on global variables
- **✅ ERROR LOG ANALYSIS COMPLETE**: 500+ line error.log contains only 1 actual error (empty data processing) and 500+ normal HF Space keep-alive polling logs
- **✅ ENDPOINT LOGIC ANALYSIS COMPLETE**: Detailed code flow understanding for both / and /process endpoints for system documentation
- **✅ PERFORMANCE OPTIMIZATION COMPLETE**: /process에서 랭킹 계산 즉시 완료 후 응답, 차트는 백그라운드에서 비동기 계산
- **✅ BACKGROUND CHART CALCULATION**: Charts calculated asynchronously after response, saved to files when complete
- **✅ FILE-BASED BACKGROUND SHARING**: Background tasks use file storage for data persistence and sharing
- **✅ ALWAYS LATEST FILE ACCESS**: / endpoint always loads most recent files, never caches, always shows current data
- **✅ DOCKER DIRECTORY SETUP**: Added /app/data/shared directory creation in Dockerfile for storage reliability

## 🎯 Key Achievements
- **Cross-contamination problem solved**: Marginal Cost Frontier Charts show pure feature trends without contamination
- **Feature-level visualization**: Charts display how pure marginal costs vary across different feature levels
- **Data integration**: Successfully combines multi-frontier regression coefficients with feature-level trend analysis
- **Chart rendering**: All chart types (traditional frontier, marginal cost frontier) working correctly
- **✅ PIECEWISE IMPLEMENTATION**: Real economies of scale reflected in marginal cost trends with automatic change point detection
- **✅ REFACTORING PROPOSAL FULLY IMPLEMENTED**: All key mathematical concepts from refactoring_proposal.md now working in production
- **✅ QUALITY ASSURANCE**: Same filtering standards as original frontier charts (monotonicity + 1KRW rule)
- **✅ DATA INTEGRITY**: Proper unlimited plan handling with separate endpoints
- **✅ CLEAN TRENDLINES**: Unlimited features stored as flags, not mixed into continuous marginal cost calculations
- **✅ CONSISTENT DATA POINTS**: Traditional and marginal frontier charts now show same number of actual market plans
- **✅ COMPREHENSIVE DATASET USAGE**: Full dataset regression provides more accurate coefficients than frontier-only analysis
- **✅ COMPLETE FEATURE COVERAGE**: All 5 core features (data, voice, messages, tethering, 5G) now analyzed
- **✅ UI SIMPLIFICATION**: Both Linear Decomposition Analysis and Multi-Feature Frontier Regression Analysis sections removed for cleaner interface
- **✅ PROPER COST ACCUMULATION**: Charts show cumulative costs building up through piecewise segments
- **✅ REALISTIC MARGINAL COST STRUCTURE**: Piecewise segments displayed in coefficient table instead of fixed rates
- **✅ FIXED RATES RANKING**: Ranking table now uses pure marginal coefficients from entire dataset for CS calculation
- **✅ COMPREHENSIVE COEFFICIENT INVESTIGATION**: Systematic analysis of negative coefficient causes completed with definitive root cause identification
- **✅ COEFFICIENT COMPARISON ENHANCEMENT**: Feature coefficient table now shows both unconstrained (raw) and constrained (bounded) values with difference calculation
- **✅ MATHEMATICAL TRANSPARENCY**: Coefficient table displays exact calculation steps including multicollinearity redistribution with formulas like "(70.2 + 49.8) / 2 = 60.0"
- **✅ MULTIPROCESSING ARCHITECTURE SOLVED**: File-based storage eliminates global variable sharing issues in FastAPI multiprocessing environment

## 🔌 Endpoint Architecture
**/ endpoint (Root HTML Interface)**:
- **Purpose**: Visual interface for users to view processed rankings and charts
- **Data Source**: Loads from files using data_storage.load_rankings_data() instead of global variables
- **Content**: Complete HTML report with ranking table, charts, and feature coefficient table
- **Chart Status**: Shows individual loading states for each chart section if still calculating
- **Response**: Always returns HTML (immediate, never blocks for calculations)
- **File Dependencies**: rankings.json, cost_structure.json, metadata.json in /app/data/shared/

**/process endpoint (Data Processing API)**:
- **Purpose**: Processes raw mobile plan data and calculates rankings using Cost-Spec analysis
- **Input**: JSON array of mobile plan data
- **Processing**: Preprocessing → Feature extraction → Coefficient calculation → Ranking → Store results
- **File Storage**: Saves results to /app/data/shared/ directory using data_storage.save_rankings_data()
- **Chart Calculation**: Triggers background async chart calculations (non-blocking)
- **Response**: Immediate JSON with ranked plans and CS ratios
- **Side Effect**: Populates file-based storage for / endpoint to display

**Testing Workflow**: `/process` for data processing → `/` for visual verification of results
**Development Pattern**: Use `/process` endpoint for testing core functionality, check HTML results via `/` endpoint

## 🔧 Technical Implementation
- **File-based storage**: data_storage.py module handles save/load operations for DataFrame and cost structure
- **Storage location**: /app/data/shared/ directory with rankings.json, cost_structure.json, metadata.json
- **Multiprocessing compatibility**: File system provides reliable inter-process communication
- **Data preparation**: `prepare_granular_marginal_cost_frontier_data()` function uses entire dataset for regression analysis
- **Chart creation**: `createMarginalCostFrontierCharts()` JavaScript function renders interactive charts with full dataset results
- **HTML integration**: Marginal Cost Frontier Analysis section displays comprehensive analysis results
- **Method detection**: System uses FullDatasetMultiFeatureRegression for more accurate coefficient extraction
- **✅ FULL DATASET ALGORITHMS**: Uses entire dataset instead of frontier points for regression analysis
- **✅ REALISTIC MARGINAL COSTS**: Variable marginal costs across feature ranges with comprehensive market data
- **✅ FRONTIER CONSISTENCY**: Maintains quality filtering while using full dataset for coefficient calculation
- **✅ UNLIMITED PROCESSING**: Separate handling of unlimited plans with proper categorical treatment
- **✅ FLAG-BASED UNLIMITED**: Unlimited features stored separately from continuous analysis
- **✅ COMPREHENSIVE FILTERING**: Uses entire dataset for analysis while maintaining data quality standards
- **✅ CLEAN CODEBASE**: All Linear Decomposition and Multi-Feature Regression functions and references removed from codebase
- **✅ CUMULATIVE PIECEWISE CALCULATION**: Uses fit_cumulative_piecewise_linear for proper cost accumulation
- **✅ CHART Y-AXIS FIX**: Charts plot cumulative_cost instead of marginal_cost for proper visualization
- **✅ FIXED RATES CS CALCULATION**: New method calculates CS ratios using pure coefficients without frontier filtering
- **✅ DATA PIPELINE ANALYSIS**: Comprehensive investigation framework for diagnosing coefficient calculation issues
- **✅ COEFFICIENT ENHANCEMENT**: `generate_feature_rates_table_html()` function shows unconstrained vs constrained coefficients with color-coded adjustment indicators
- **✅ FILE-BASED DATA SHARING**: Eliminates global variable dependencies and multiprocessing memory sharing issues

## 🚨 Current Issues
- **None currently**: File-based storage system resolved all major multiprocessing memory sharing issues

## 📝 Feature Enhancement Details
- **File storage structure**: JSON format for DataFrame serialization with metadata preservation
- **Error handling**: Graceful degradation when files don't exist (returns None)
- **Backward compatibility**: Maintains config module storage alongside file storage during transition
- **Debug capabilities**: Enhanced debug-global endpoint shows both file and config storage status
- **Unconstrained coefficients**: Raw OLS regression results without bounds constraints
- **Constrained coefficients**: Final values after applying economic bounds (non-negative, minimum values)
- **Adjustment display**: Green for upward adjustments, red for downward adjustments, gray for minimal changes
- **Comparison format**: Side-by-side table with separate columns for before/after values and difference

## 🔍 Information Sources
- **User feedback**: Request for coefficient table with both raw and adjusted values for comparison
- **Code enhancement**: Modified `_solve_constrained_regression()` to store unconstrained coefficients
- **UI improvement**: Enhanced `generate_feature_rates_table_html()` with expanded table format
- **Architecture decision**: User preference for file-based storage over multithreading conversion
- **Problem diagnosis**: Identified multiprocessing as root cause of global variable sharing issues

## 📈 Chart Types Available
1. **Traditional Feature Frontier Charts**: Market-based trends (with contamination)
2. **Marginal Cost Frontier Charts**: Full dataset coefficient-based trends (contamination-free) ⭐ NOW USING CUMULATIVE PIECEWISE COSTS
3. **Plan Efficiency Charts**: Value ratio analysis

## 🎨 User Experience
- **Clear explanations**: Each chart section includes Korean explanations of methodology and interpretation
- **Visual distinction**: Blue lines for cumulative cost trends, red points for market comparison
- **Responsive design**: Charts adapt to different screen sizes and data volumes
- **Interactive features**: Hover tooltips and zoom capabilities for detailed analysis
- **Manual refresh system**: No auto-polling, users manually refresh to check progress using refresh button
- **✅ REFRESH BUTTON**: Added 🔄 새로고침 button in header for manual page refresh to load latest data
- **✅ FULL DATASET VISUALIZATION**: Charts show comprehensive analysis results from entire dataset
- **✅ COMPLETE FEATURE SET**: All 5 core features visualized including 5G support
- **✅ SIMPLIFIED INTERFACE**: Both Linear Decomposition Analysis and Multi-Feature Frontier Regression Analysis removed for better focus
- **✅ PROPER COST VISUALIZATION**: Charts show realistic cumulative cost accumulation
- **✅ PIECEWISE SEGMENT DISPLAY**: Coefficient table shows segment ranges instead of fixed rates
- **✅ CLEAN RANKING TABLE**: Ranking now uses pure fixed rates from entire dataset analysis
- **✅ RELIABLE DATA DISPLAY**: File-based storage ensures consistent ranking table display across all processes

## 🎯 User Requirements & Preferences
- **File-based storage preferred**: User chose file-based solution over multithreading conversion for multiprocessing memory sharing
- **No auto-refresh**: Manual refresh only, no constant polling
- **Visual feedback**: Clear status indicators for chart calculation progress
- **Immediate API response**: /process endpoint returns instantly, charts calculated separately
- **Fresh content**: No caching, all content generated on-demand
- **Comprehensive analysis**: Marginal cost frontier analysis using entire dataset
- **No Linear Decomposition**: Linear Decomposition Analysis section completely removed per user request
- **No Multi-Feature Frontier**: Multi-Feature Frontier Regression Analysis section completely removed per user request
- **5G Feature Inclusion**: 5G support feature added to analysis scope
- **Entire Dataset Usage**: Full dataset regression instead of frontier-only analysis
- **Cumulative Cost Visualization**: Charts show proper cost accumulation, not fixed rates
- **Piecewise Segment Structure**: Coefficient table displays segment ranges with varying rates
- **Fixed Rates Ranking**: Ranking table uses pure marginal coefficients for entire dataset
- **Root cause investigation**: User prefers thorough analysis of underlying issues rather than quick workarounds

## 🔧 Technical Implementation Details
- **File-based architecture**: data_storage.py module with save_rankings_data() and load_rankings_data() functions
- **Storage format**: JSON serialization of pandas DataFrame with metadata preservation
- **Inter-process communication**: File system provides reliable data sharing between FastAPI processes
- **Error resilience**: Graceful handling of missing files with fallback to None values
- **Infinite loop fix**: Added safety counters and division-by-zero checks in `prepare_feature_frontier_data`
- **Logging optimization**: Reduced verbose logging to prevent SSH polling spam
- **Chart data handling**: JavaScript functions handle full dataset analysis results
- **Background processing**: Chart calculations run asynchronously without blocking API responses
- **Full dataset regression**: FullDatasetMultiFeatureRegression provides comprehensive coefficient analysis
- **Code cleanup**: All Linear Decomposition and Multi-Feature Frontier Regression functions and references removed from codebase
- **Cumulative cost calculation**: Fixed chart plotting to use cumulative_cost instead of marginal_cost
- **Piecewise segment implementation**: Using fit_cumulative_piecewise_linear for realistic cost accumulation
- **Fixed rates method**: New 'fixed_rates' method in calculate_cs_ratio_enhanced using FullDatasetMultiFeatureRegression
- **Data preprocessing pipeline**: Raw data requires preprocessing via prepare_features() to create expected feature columns

## 🎯 Working Methods
- **File-based data persistence**: Eliminates multiprocessing memory sharing issues through file system storage
- **Fixed rates regression**: Uses FullDatasetMultiFeatureRegression for pure coefficient extraction on entire dataset
- **Feature frontier charts**: Original logic maintained as requested
- **Safety measures**: Infinite loop prevention implemented and working
- **Numpy type conversion**: Comprehensive serialization fix for all data types
- **Async processing**: Chart calculations run in background, API responds immediately
- **Cumulative piecewise calculation**: Proper cost accumulation through segments
- **Investigation methodology**: Systematic analysis of data pipeline issues using parallel comparisons

## 🔧 Implementation Patterns
- **File-based storage pattern**: Save on process, load on display - eliminates global variable dependencies
- **Async chart calculation**: Background tasks for expensive visualizations
- **Progressive status display**: Real-time progress indicators for chart generation
- **Fallback mechanisms**: Basic HTML reports when charts fail or are in progress
- **Method integration**: Fixed rates methods integrated into existing cost_spec.py structure
- **Error handling**: Robust type conversion and safety measures
- **Testing workflow**: Using raw data files from /data/raw/ directory
- **Clean server startup**: Direct uvicorn command in Dockerfile, log monitoring via app.py startup event
- **Root cause analysis**: Comprehensive investigation of technical issues before implementing solutions

## 📈 Data Flow
- Raw data → Fixed rates multi-feature regression → CS ratio calculation → **File storage** → Immediate API response
- Background: Chart generation → HTML report with visualizations → Cache update
- **File-based persistence**: Process endpoint saves to files, root endpoint loads from files
- Feature analysis for each core feature (data, voice, messages, tethering, 5G)
- Comprehensive dataset utilization for accurate coefficient extraction without filtering
- Cross-contamination eliminated through full dataset regression approach using entire dataset
- Cumulative cost calculation through piecewise segments for realistic visualization
- Pure coefficient calculation for ranking table using fixed marginal rates
- **Critical**: Raw data requires preprocessing to create expected feature columns before coefficient calculation
- **Multiprocessing compatible**: File system provides reliable inter-process data sharing

## 시스템 정보
- 운영체제: Linux 5.10.237-230.949.amzn2.x86_64
- 워크스페이스: vscode-remote://ssh-remote%2Bssh.hf.space.mvno/app
- 쉘: /bin/sh

## Marginal Calculation 수학적 원리 ⭐ 명확화 완료
- **프론티어 목적**: 트렌드 학습용, 각 feature 레벨에서 최저가만 선택하여 overpriced 요금제 제거
- **구간별 beta**: 규모의 경제 반영 (첫 1GB ≠ 100GB에서 1GB)
- **상호작용 제외**: 복잡성 방지, 해석 가능성 유지
- **핵심 문제 발견**: 프론티어 포인트 가격에 다른 feature들 가치가 혼재됨
- **해결책**: 다중 Feature 동시 회귀 (프론티어 선택 + 전체 다중 회귀)
- **개선 방향**: 순수한 각 feature의 독립적 가치 추정
- **추천 방법**: 기울기 변화점 기반 구간 설정 + 1KRW/feature 제약 유지
- **실행 계획**: 4단계 점진적 개선 (기존 시스템 보존하면서 새 방법 추가)
- **✅ 누적 비용 계산**: 구간별 한계비용을 누적하여 실제 총 비용 트렌드 시각화
- **✅ 구간별 변화**: 고정 요율 대신 구간별로 다른 한계비용 적용
- **✅ 고정 요율 랭킹**: 전체 데이터셋에서 순수 한계비용 계수를 사용한 랭킹 테이블

## Hugging Face Dev Mode 환경 ⭐ 중요
- **현재 환경**: Hugging Face Space에서 Dev Mode 활성화 상태
- **서버 상태**: localhost:7860에서 상시 실행 중 (절대 종료 금지)
- **로그 모니터링**: simple_log_monitor.sh 스크립트 정상 작동 중
- **자동화 완료**: Dockerfile 수정으로 서버 시작 후 로그 모니터링 자동 실행
- **쉘 호환성**: sh 쉘 사용으로 Docker 환경 호환성 확보
- **실행 순서**: 서버 먼저 시작 → 3초 대기 → 로그 모니터링 시작 (PID 찾기 문제 해결)
- **최근 상태**: 로그 모니터링 시스템 완전 복구 완료
- **코드 반영**: 파일 수정 시 서버에 즉시 반영됨 (재시작 불필요)
- **Git 상태**: Dev Mode에서의 변경사항은 자동으로 Git에 저장되지 않음
- **중요사항**: 서버 종료 시 Dev Mode 비활성화될 위험 있음 → 절대 프로세스 kill 금지
- **참고**: [Hugging Face Dev Mode 문서](https://huggingface.co/docs/hub/spaces-dev-mode)

## 무한 루프 문제 해결 ⭐ 해결 완료
- **문제 발생**: 2025-06-12 05:48:03~05:49:38 동안 modules.report_charts에서 무한 반복
- **원인**: prepare_feature_frontier_data 함수의 이중 while 루프 (113-138번 줄)
- **해결책**: 반복 횟수 제한, 0으로 나누기 방지, 안전장치 추가
- **결과**: 05:49:43 이후 정상 작동, 무한 루프 완전 해결

## 연속 계산 문제 해결 ⭐ 해결 완료
- **문제**: SSH 원격 연결 폴링으로 인한 "/" 엔드포인트 연속 호출
- **원인**: 루트 엔드포인트에서 매번 generate_html_report 호출로 차트 계산 트리거
- **해결책**: 비동기 차트 계산 시스템 구현
  - /process 엔드포인트: 즉시 API 응답 반환
  - 백그라운드: 차트 계산 비동기 실행
  - 루트 엔드포인트: 캐시된 콘텐츠 제공 또는 진행 상태 표시
- **결과**: 연속 계산 완전 제거, 응답 시간 대폭 개선

## 🔍 **Negative Coefficient Investigation** ⭐ **ROOT CAUSE IDENTIFIED**

### **Comprehensive Investigation Results**
- **Primary Cause**: Data preprocessing pipeline mismatch (NOT economic modeling issues)
- **Raw Data Status**: Only 2/16 expected features available (`additional_call`, `tethering_gb`)
- **Processed Data Status**: All 16/16 expected features created by preprocessing pipeline
- **Economic Logic**: Features present in data show positive correlations with price (economically correct)
- **Multicollinearity**: Detected in processed data but separate issue from missing features
- **Coefficient Stability**: Stable across regularization levels for available features

### **Investigation Methodology Applied**
- **Feature Distribution Analysis**: Examined data quality, missing values, outliers
- **Correlation Analysis**: Checked for multicollinearity and economic logic violations
- **Economic Relationship Analysis**: Verified features correlate positively with prices
- **Feature Engineering Analysis**: Identified preprocessing requirements
- **Regression Diagnostics**: Tested coefficient stability across models
- **Pipeline Comparison**: Raw vs processed data coefficient calculation

### **Confirmed NOT the Cause**
❌ **Multicollinearity**: Low correlations in raw data, manageable in processed data
❌ **Economic Logic Violations**: Present features show positive price correlation
❌ **Overfitting**: Adequate sample-to-feature ratio (1149:1 for available features)
❌ **Coefficient Instability**: Stable results across different regularization levels
❌ **Data Quality Issues**: No significant outliers or data corruption

### **Investigation Results - PREPROCESSING PIPELINE WORKING CORRECTLY**
✅ **Data Flow Verified**: 
- Raw data (40 columns) → prepare_features() → Processed data (80 columns)
- All 16/16 expected features found in processed data
- FullDatasetMultiFeatureRegression correctly receives processed DataFrame
- No zero coefficients due to missing features

## Multiprocessing Memory Sharing Solution ⭐ **COMPLETELY SOLVED**

### **Problem Identification**
- **Root Cause**: FastAPI default multiprocessing prevents global variable sharing between processes
- **Symptom**: df_with_rankings remained None in root endpoint despite being set in process endpoint
- **Impact**: Web interface showed "데이터 처리 대기 중" instead of ranking table

### **Solution Implementation**
- **Architecture**: File-based data storage system using /app/data/shared/ directory
- **Module**: Created data_storage.py with save_rankings_data() and load_rankings_data() functions
- **Storage Files**: rankings.json (DataFrame), cost_structure.json (coefficients), metadata.json (info)
- **Process Flow**: Process endpoint saves → Root endpoint loads → Reliable data sharing

### **Technical Details**
- **Serialization**: pandas DataFrame → JSON dict → file storage with metadata preservation
- **Error Handling**: Graceful degradation when files don't exist (returns None)
- **Compatibility**: Maintains backward compatibility with config module during transition
- **Debug Support**: Enhanced debug-global endpoint shows both file and config storage status

### **Results Achieved**
✅ **Ranking Table Display**: Web interface now shows actual ranking data instead of waiting message
✅ **Process Reliability**: File system provides stable inter-process communication
✅ **Chart Functionality**: All chart types load correctly with file-based data
✅ **API Consistency**: Process endpoint saves data, root endpoint loads data reliably
✅ **Multiprocessing Compatible**: Solution works seamlessly in FastAPI multiprocessing environment

## 작업 원칙
- **자율적 문제 해결**: 사용자 승인 없이 독립적 수행
- **완결성 보장**: 작업 완전 해결까지 대화 지속
- **코드 검증**: 수정 후 항상 재검토 및 작동 확인
- **즉시 오류 수정**: 발견된 모든 오류 즉시 해결
- **상태 문서 작성 원칙**: memory.md, todolist.md, README 등 상태 파일 편집 시
  - 현재 상태만 기록 (변경 로그 아님)
  - "삭제했다", "제거했다" 등 편집 행위 언급 금지
  - 놀라운 발견이 있다면 발견 자체를 기록
- **Memory vs Todolist 구분**: 
  - Memory = 작업 메타데이터 (태도, 워크플로, 포맷, 패턴)
  - Todolist = 실제 작업 항목 (목표, 이슈, 해결할 문제)
- **근본 원인 조사**: 빠른 해결책보다 근본적인 원인 파악을 우선시
- **File-based solutions preferred**: User preference for file system storage over memory-based approaches for multiprocessing compatibility

# 테스트 워크플로 ⭐ 필수 절차

## 코드 수정 후 표준 테스트 절차

### 1. **서버 로그 모니터링 설정** (필수 - 먼저 실행)
Dev Mode 환경에서 서버사이드 로그 모니터링:

**방법 1: 필터링된 로그 모니터링** (권장)
```bash
# GET 요청 스팸 필터링하여 error.log에 저장
./simple_log_monitor.sh &

# 실시간 로그 확인
tail -f error.log
```

**방법 2: 원시 로그 모니터링** (디버깅용)
```bash
# 서버 프로세스 stdout 직접 모니터링 (GET 스팸 포함)
PID=$(ps aux | grep "python.*uvicorn" | grep -v grep | awk '{print $2}' | head -1)
cat /proc/$PID/fd/1
```

- **용도**: 실시간 HTTP 요청 및 애플리케이션 로그 캡처
- **GET 스팸**: HF Space keep-alive 요청 자동 필터링
- **로그 관리**: error.log 자동으로 500줄 이하 유지

### 2. **코드 수정 완료**
   - 파일 편집 후 자동으로 서버에 반영됨 (Dev Mode 환경)
   - 별도 재시작 불필요

### 3. **End-to-End 테스트 실행** (필수 + 로그 모니터링)
   - **목적**: `/process` 엔드포인트가 전체 코드베이스의 핵심 기능
   - **⚠️ 로그 모니터링 동시 실행**: 테스트하면서 반드시 서버 로그 확인
   
   **방법 1** (선호): 로컬 실제 데이터 테스트
   ```bash
   # 터미널 1: 필터링된 로그 모니터링 시작했는지 체크 (필수!)
   ./simple_log_monitor.sh &
   # 1개만 실행하도록!

   # 터미널 2: 최신 raw 데이터 파일로 테스트 (동적으로 가장 최근 파일 사용)
   curl -X POST http://localhost:7860/process -H "Content-Type: application/json" -d @$(ls -t data/raw/*.json | head -1)
   ```

   **방법 2**: Supabase 함수 사용 (service_role 인증 필요)
   ```bash
   # .env.local에서 service_role 키 사용
   curl -X POST https://zqoybuhwasuppzjqnllm.supabase.co/functions/v1/submit-data \
     -H "Authorization: Bearer $(grep service_role .env.local | cut -d'=' -f2)"
   ```

### 4. **웹 인터페이스 확인** (필수)
   - **브라우저**: `http://localhost:7860` 접속
   - **확인 사항**: 
     - 페이지 로딩 정상
     - 차트 렌더링 정상
     - JavaScript 오류 없음 (개발자 도구 콘솔 확인)

### 5. **로그 분석** (필수)
   - **서버 로그**: error.log 파일에서 오류 메시지 확인
   - **HTTP 로그**: uvicorn 요청 로그에서 응답 코드 확인
   - **JavaScript 오류**: 브라우저 개발자 도구에서 콘솔 오류 확인

## 🚨 주의사항
- **로그 모니터링 필수**: 코드 수정 후 반드시 로그 모니터링 상태에서 테스트
- **서버 종료 금지**: Dev Mode 환경에서 서버 프로세스 절대 종료하지 말 것
- **동시 실행**: 로그 모니터링과 테스트를 동시에 실행하여 실시간 피드백 확보
- **완전한 테스트**: 단순 API 응답뿐만 아니라 웹 인터페이스까지 전체 확인

# 현재 상태

## 작업된 주요 기능
- **File-based data storage**: Complete multiprocessing memory sharing solution implemented
- Cross-contamination 문제 해결: 순수 계수(pure coefficients) 기반 CS 비율 계산
- Multi-Feature Frontier Regression Analysis 섹션 완전 제거
- Fixed rates 방식으로 전체 데이터셋 기반 CS 계산 구현
- Plan Value Efficiency Analysis 섹션이 ranking table과 동일한 fixed_rates 방식 사용 확인
- 기능별 한계비용 계수 테이블 추가: 랭킹 테이블 위에 각 기능의 한계비용 표시
- **Double counting 문제 해결**: 무제한 기능의 연속값을 0으로 설정하여 이중 계산 방지
- **Unlimited type flags 구현**: 3가지 데이터 소진 후 상태를 별도 플래그로 분리
- **Negative coefficient 근본 원인 식별**: 데이터 전처리 파이프라인 불일치 확인

## 기술적 구현
- **File-based storage architecture**: data_storage.py module with save/load functions
- **Multiprocessing compatibility**: File system provides reliable inter-process data sharing
- calculate_cs_ratio_enhanced()에 'fixed_rates' 방식 추가
- FullDatasetMultiFeatureRegression으로 전체 데이터셋에서 순수 계수 추출
- prepare_plan_efficiency_data() 함수가 모든 계산 방식(linear_decomposition, frontier, fixed_rates, multi_frontier) 올바르게 처리
- app.py의 기본 방식을 'fixed_rates'로 변경
- generate_feature_rates_table_html() 함수로 기능별 한계비용 테이블 생성
- **무제한 기능 전처리 수정**: unlimited 플래그가 1인 경우 연속값을 0으로 설정
- **Unlimited type flags**: data_stops_after_quota, data_throttled_after_quota, data_unlimited_speed
- **계수 문제 진단 도구**: 체계적인 근본 원인 분석 프레임워크 구현

## 데이터 처리 방식
- **File-based persistence**: Process endpoint saves to files, root endpoint loads from files
- 무제한 기능: 불린 플래그와 3배 승수 값으로 처리
- **Double counting 방지**: 무제한 플래그가 있는 기능의 연속값은 0으로 설정
- 필터링 없이 전체 데이터셋 처리
- 순수 계수 기반 baseline cost / original fee로 CS 비율 계산
- 계수 분석 결과를 시각화와 호환되도록 저장
- **데이터 파이프라인 요구사항**: 원시 데이터는 prepare_features()를 통한 전처리 필요

## 기능별 한계비용 현황 (최신 데이터 기준)
- 데이터 소진 후 속도제한: ₩10,838 (고정)
- 5G 지원: ₩6,627 (고정)
- Daily Data: ₩4,628/unit
- 소진 후 속도: ₩2,292/Mbps
- 테더링: ₩84.31/GB
- 데이터: ₩75.86/GB (무제한 시 0으로 설정)
- 추가 통화: 계수값/unit
- 음성통화: ₩0.0000/분 (무제한 시 0으로 설정)
- 데이터 소진 후 중단: 계수값 (기준)
- 데이터 무제한: 계수값 (고정)
- 문자메시지: ₩3.19/건 (무제한 시 0으로 설정)

## 테스트 환경
- **File-based storage**: Uses /app/data/shared/ directory for reliable data persistence
- 로컬 테스트 시 data/raw 폴더의 최신 JSON 파일 사용
- curl -X POST http://localhost:7860/process -H "Content-Type: application/json" -d @$(ls -t data/raw/*.json | head -1)
- 모든 기능이 정상 작동 중
- Double counting 문제 해결 완료
- Unlimited type flags 정상 작동
- Negative coefficient 근본 원인 식별 완료
- **Multiprocessing memory sharing**: Completely resolved with file-based storage system

## 🔬 Model Validation & Quality Assurance System - REMOVED

### **Validation System Completely Removed**
- **Background validation**: Removed all automatic validation calculations
- **HTML validation section**: Removed "🔬 Model Validation & Reliability Analysis" section
- **JavaScript validation functions**: Removed all validation display functions
- **Validation endpoints**: Removed `/validation-status` and `/validation-results` endpoints
- **Multi-method reliability**: Removed 5-method coefficient comparison system
- **Economic logic validation**: Removed bias-based validation criteria
- **Statistical validation**: Removed cross-validation and residual analysis
- **Validation scoring**: Removed 0-100 point scoring system

### **Current Background Processing**
- **Charts only**: Background tasks now calculate only chart visualizations
- **No validation overhead**: Eliminated time-consuming validation calculations
- **Simplified workflow**: `/process` → immediate response → background charts only
- **Clean architecture**: Removed validation executors, status tracking, and result caching

### **Rationale for Removal**
- **Bias-based criteria**: Economic logic validation was based on subjective assumptions
- **Arbitrary parameters**: Multi-method validation used market-irrelevant parameter sets
- **Statistical inadequacy**: High R² doesn't guarantee correct coefficients
- **False precision**: Complex scoring system created illusion of accuracy
- **Performance overhead**: Validation calculations added unnecessary complexity

## 최근 해결된 주요 문제

### 차트 표시 문제 (2025-06-19 완료)
- **문제**: HTML에서 차트가 표시되지 않음 (display:none으로 숨겨짐)
- **원인**: HTML 템플릿에서 `get_chart_status_html()` 함수가 실행되지 않고 문자열로 출력됨
- **해결**: 
  1. 차트 상태 변수를 사전에 계산하여 HTML 템플릿에 변수로 전달
  2. replace() 메서드로 변수 치환 처리 추가
  3. 차트 표시/숨김 로직을 올바르게 수정
- **결과**: Feature Frontier와 Plan Efficiency 차트 모두 정상 표시

### 차트 상태 API 문제 (이전에 해결됨)
- **문제**: `/chart-status` API에서 500 Internal Server Error
- **원인**: datetime 직렬화 오류, 필드명 불일치, 앱 시작 시 차트 데이터 로딩 누락
- **해결**: 
  1. datetime 안전 처리 로직 추가
  2. 필드명 수정 (`is_calculating` → `status == 'calculating'`)
  3. startup event에 차트 데이터 로딩 로직 추가

## 현재 상태
- **차트 시스템**: ✅ 완전히 정상 작동
- **API 엔드포인트**: ✅ 모든 엔드포인트 정상
- **데이터 로딩**: ✅ 앱 시작 시 자동 로딩
- **HTML 표시**: ✅ 차트 정상 렌더링

## 주의사항
- HTML 템플릿 수정 시 변수 replace 처리 확인 필요
- 차트 상태 함수 수정 시 HTML 변수 동기화 확인
- datetime 객체 JSON 직렬화 시 안전 처리 적용