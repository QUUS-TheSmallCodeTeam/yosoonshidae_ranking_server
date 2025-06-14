# 🧠 Memory & Context

## 📊 Current System Status
- **Async chart calculation**: Implemented to eliminate continuous calculations triggered by root endpoint
- **Visual status indicators**: Loading icons (⚙️) for in-progress, error icons (❌) for failed calculations
- **Manual refresh system**: No auto-polling, users manually refresh to check progress
- **No caching**: All HTML content generated fresh on each request for immediate status updates
- **Multi-frontier regression methodology**: Successfully implemented and fully operational
- **Chart visualization**: Advanced charts now calculated asynchronously in background
- **API response time**: Immediate response from /process endpoint, charts calculated separately
- **Default method**: Changed to `fixed_rates` for new ranking table calculation approach
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

## 🔧 Technical Implementation
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
- **✅ CLEAN CODEBASE**: All Linear Decomposition and Multi-Feature Frontier Regression functions and references removed from codebase
- **✅ CUMULATIVE PIECEWISE CALCULATION**: Uses fit_cumulative_piecewise_linear for proper cost accumulation
- **✅ CHART Y-AXIS FIX**: Charts plot cumulative_cost instead of marginal_cost for proper visualization
- **✅ FIXED RATES CS CALCULATION**: New method calculates CS ratios using pure coefficients without frontier filtering

## 📈 Chart Types Available
1. **Traditional Feature Frontier Charts**: Market-based trends (with contamination)
2. **Marginal Cost Frontier Charts**: Full dataset coefficient-based trends (contamination-free) ⭐ NOW USING CUMULATIVE PIECEWISE COSTS
3. **Plan Efficiency Charts**: Value ratio analysis

## 🎨 User Experience
- **Clear explanations**: Each chart section includes Korean explanations of methodology and interpretation
- **Visual distinction**: Blue lines for cumulative cost trends, red points for market comparison
- **Responsive design**: Charts adapt to different screen sizes and data volumes
- **Interactive features**: Hover tooltips and zoom capabilities for detailed analysis
- **✅ FULL DATASET VISUALIZATION**: Charts show comprehensive analysis results from entire dataset
- **✅ COMPLETE FEATURE SET**: All 5 core features visualized including 5G support
- **✅ SIMPLIFIED INTERFACE**: Both Linear Decomposition Analysis and Multi-Feature Frontier Regression Analysis removed for better focus
- **✅ PROPER COST VISUALIZATION**: Charts show realistic cumulative cost accumulation
- **✅ PIECEWISE SEGMENT DISPLAY**: Coefficient table shows segment ranges instead of fixed rates
- **✅ CLEAN RANKING TABLE**: Ranking now uses pure fixed rates from entire dataset analysis

## 🎯 User Requirements & Preferences
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

## 🔧 Technical Implementation Details
- **Infinite loop fix**: Added safety counters and division-by-zero checks in `prepare_feature_frontier_data`
- **Logging optimization**: Reduced verbose logging to prevent SSH polling spam
- **Chart data handling**: JavaScript functions handle full dataset analysis results
- **Background processing**: Chart calculations run asynchronously without blocking API responses
- **Full dataset regression**: FullDatasetMultiFeatureRegression provides comprehensive coefficient analysis
- **Code cleanup**: All Linear Decomposition and Multi-Feature Frontier Regression functions and references removed from codebase
- **Cumulative cost calculation**: Fixed chart plotting to use cumulative_cost instead of marginal_cost
- **Piecewise segment implementation**: Using fit_cumulative_piecewise_linear for realistic cost accumulation
- **Fixed rates method**: New 'fixed_rates' method in calculate_cs_ratio_enhanced using FullDatasetMultiFeatureRegression

## 🎯 Working Methods
- **Fixed rates regression**: Uses FullDatasetMultiFeatureRegression for pure coefficient extraction on entire dataset
- **Feature frontier charts**: Original logic maintained as requested
- **Safety measures**: Infinite loop prevention implemented and working
- **Numpy type conversion**: Comprehensive serialization fix for all data types
- **Async processing**: Chart calculations run in background, API responds immediately
- **Cumulative piecewise calculation**: Proper cost accumulation through segments

## 🔧 Implementation Patterns
- **Async chart calculation**: Background tasks for expensive visualizations
- **Progressive status display**: Real-time progress indicators for chart generation
- **Fallback mechanisms**: Basic HTML reports when charts fail or are in progress
- **Method integration**: Fixed rates methods integrated into existing cost_spec.py structure
- **Error handling**: Robust type conversion and safety measures
- **Testing workflow**: Using raw data files from /data/raw/ directory
- **Clean server startup**: Direct uvicorn command in Dockerfile, log monitoring via app.py startup event

## 📈 Data Flow
- Raw data → Fixed rates multi-feature regression → CS ratio calculation → Immediate API response
- Background: Chart generation → HTML report with visualizations → Cache update
- Feature analysis for each core feature (data, voice, messages, tethering, 5G)
- Comprehensive dataset utilization for accurate coefficient extraction without filtering
- Cross-contamination eliminated through full dataset regression approach using entire dataset
- Cumulative cost calculation through piecewise segments for realistic visualization
- Pure coefficient calculation for ranking table using fixed marginal rates

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
- Cross-contamination 문제 해결: 순수 계수(pure coefficients) 기반 CS 비율 계산
- Multi-Feature Frontier Regression Analysis 섹션 완전 제거
- Fixed rates 방식으로 전체 데이터셋 기반 CS 계산 구현
- Plan Value Efficiency Analysis 섹션이 ranking table과 동일한 fixed_rates 방식 사용 확인
- 기능별 한계비용 계수 테이블 추가: 랭킹 테이블 위에 각 기능의 한계비용 표시
- **Double counting 문제 해결**: 무제한 기능의 연속값을 0으로 설정하여 이중 계산 방지
- **Unlimited type flags 구현**: 3가지 데이터 소진 후 상태를 별도 플래그로 분리

## 기술적 구현
- calculate_cs_ratio_enhanced()에 'fixed_rates' 방식 추가
- FullDatasetMultiFeatureRegression으로 전체 데이터셋에서 순수 계수 추출
- prepare_plan_efficiency_data() 함수가 모든 계산 방식(linear_decomposition, frontier, fixed_rates, multi_frontier) 올바르게 처리
- app.py의 기본 방식을 'fixed_rates'로 변경
- generate_feature_rates_table_html() 함수로 기능별 한계비용 테이블 생성
- **무제한 기능 전처리 수정**: unlimited 플래그가 1인 경우 연속값을 0으로 설정
- **Unlimited type flags**: data_stops_after_quota, data_throttled_after_quota, data_unlimited_speed

## 데이터 처리 방식
- 무제한 기능: 불린 플래그와 3배 승수 값으로 처리
- **Double counting 방지**: 무제한 플래그가 있는 기능의 연속값은 0으로 설정
- 필터링 없이 전체 데이터셋 처리
- 순수 계수 기반 baseline cost / original fee로 CS 비율 계산
- 계수 분석 결과를 시각화와 호환되도록 저장

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
- 로컬 테스트 시 data/raw 폴더의 최신 JSON 파일 사용
- curl -X POST http://localhost:7860/process -H "Content-Type: application/json" -d @$(ls -t data/raw/*.json | head -1)
- 모든 기능이 정상 작동 중
- Double counting 문제 해결 완료
- Unlimited type flags 정상 작동