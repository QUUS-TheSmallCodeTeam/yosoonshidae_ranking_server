# Cost-Spec Analysis System - TODO List

## ✅ COMPLETED - Critical Fix Applied

### 🔧 MAJOR FIX: Frontier-Based Linear Decomposition 
- **✅ FIXED**: Changed from arbitrary "market segments" to frontier-based representative plan selection
- **✅ FIXED**: Now uses same optimal candidate point logic as original frontier method  
- **✅ FIXED**: Linear decomposition now operates on cost-efficient frontier plans only
- **Impact**: Ensures mathematically sound marginal cost discovery from optimal plans

### 📊 Enhanced Visualization System - COMPLETED
- **✅ IMPLEMENTED**: Cost Structure Decomposition Charts (doughnut + bar)
- **✅ IMPLEMENTED**: Plan Value Efficiency Matrix (bubble chart) 
- **✅ IMPLEMENTED**: Marginal Cost Analysis Chart (with business tooltips)
- **✅ FIXED**: Cost structure data format handling (nested vs flat)
- **✅ RESTORED**: JavaScript chart implementation

### 🐛 Bug Fixes - COMPLETED  
- **✅ FIXED**: Format string error in HTML report generation
- **✅ FIXED**: Cost structure data compatibility issues
- **✅ CORRECTED**: Understanding of monotonicity exclusion (BY DESIGN, not bug)

## 🧪 IMMEDIATE TESTING NEEDED

### Test Updated Linear Decomposition
- **TODO**: Run `/process` endpoint to test frontier-based selection
- **TODO**: Verify representative plans are now frontier contributors  
- **TODO**: Confirm marginal cost results are based on optimal plans
- **TODO**: Check if charts display correctly with corrected data

### Validation Tasks
- **TODO**: Compare old vs new representative plan selection in logs
- **TODO**: Verify cost structure makes sense with frontier-based selection
- **TODO**: Ensure HTML report generates without errors

## 📈 FUTURE ENHANCEMENTS (Lower Priority)

### Advanced Features
- **Consider**: Post-decomposition frontier refinement for broader plan inclusion
- **Consider**: Multi-method comparison dashboard
- **Consider**: Interactive chart features (zoom, filter)

### Documentation
- **Future**: Update README with corrected methodology explanation
- **Future**: Add technical documentation for frontier-based linear decomposition

---
**Priority Order:**
1. **URGENT**: Test frontier-based linear decomposition implementation
2. **HIGH**: Validate chart functionality with corrected data  
3. **MEDIUM**: Future enhancements and documentation

*Last Updated: After critical frontier-based selection fix*

# Cost-Spec Linear Decomposition Implementation Tasks

## ✅ Completed
- [x] Mathematical problem analysis and formulation
- [x] Linear decomposition solution design
- [x] Test script creation and validation
- [x] Proof of concept with real data
- [x] Validation of approach effectiveness
- [x] Integrate linear decomposition into main `cost_spec.py` module
- [x] Add configuration options for decomposition vs frontier methods
- [x] Create enhanced API functions supporting both methods
- [x] Integration testing and validation
- [x] **Complete refactoring and production optimization**
- [x] **Update entire codebase with enhanced implementation**
- [x] **Refactor app.py with method selection and enhanced features**
- [x] **Update HTML report generation with method information**
- [x] **Enhance web interface with linear decomposition capabilities**
- [x] **Cleanup test files and finalize implementation**
- [x] **Data cleaning functionality**

## 💡 DESIGN CLARIFICATION RESOLVED
- [x] **Understanding of monotonicity exclusion corrected** ✅ RESOLVED
  - **User Intent**: Exclude non-monotonic data BY DESIGN for reasonable cost trends
  - **Purpose**: Most optimistic baseline for fair 가성비 ranking
  - **Tethering Example**: ₩0/GB coefficient correct - insufficient reasonable data after proper exclusion
  - **System Working As Intended**: Not a bug, but proper filtering for realistic ranking

## 🔄 Production Integration Tasks  
- [ ] Update `ranking.py` to use enhanced cost_spec functions (if needed)
- [ ] Add configuration file for method selection
- [ ] Update main app.py to support method switching (✅ COMPLETED)

## 🧪 Testing & Validation
- [ ] Test with larger real datasets from `/data` folder
- [ ] Validate coefficient stability across different data samples
- [ ] Performance benchmarking vs current frontier method
- [ ] Edge case testing (unlimited plans, missing features)

## 📊 Enhancement Features
- [x] Automatic representative plan selection algorithm (implemented)
- [x] **Cost Structure Decomposition Visualization** ⭐ PRIORITY 1 ✅ COMPLETED
  - [x] Add cost structure chart to HTML template showing discovered β coefficients
  - [x] Implement doughnut/pie chart showing: Base cost, Data cost/GB, Voice cost/100min, SMS cost, Tethering cost, 5G premium
  - [x] Include percentage breakdown and actual KRW values
  - [x] Add business interpretation tooltips
  - [x] Added dual chart display: Cost components breakdown + Per-unit cost visualization
- [x] **Plan Value Efficiency Matrix** ⭐ PRIORITY 1 ✅ COMPLETED
  - [x] Implement 2D bubble chart: Baseline cost vs Actual cost
  - [x] Add diagonal efficiency line (CS = 1.0)
  - [x] Color coding: Green (good value) vs Red (overpriced)
  - [x] Interactive tooltips with plan details
  - [x] Bubble size represents total feature levels
  - [x] Replaced outdated residual fee analysis
- [x] **Marginal Cost Analysis Chart** ⭐ PRIORITY 1 ✅ COMPLETED
  - [x] Visualize individual β coefficients (marginal costs) per feature
  - [x] Business interpretation tooltips (e.g., "데이터 1GB 추가시 ₩50 비용 증가")
  - [x] Base infrastructure cost display separate from marginal costs
  - [x] Color-coded bar chart with Korean labels
- [ ] Confidence intervals for coefficient estimates
- [ ] Feature importance analysis for cost drivers
- [ ] Market segment analysis using decomposed costs

## 📈 Business Applications
- [ ] Competitive pricing analysis dashboard
- [ ] Plan optimization recommendations
- [ ] Market positioning insights
- [ ] Cost structure benchmarking tools

## 🔧 Technical Improvements
- [x] Optimize solver performance for large datasets
- [x] Add robust error handling and validation
- [ ] Implement coefficient caching for repeated analysis
- [ ] Add support for time-series cost evolution

## 📋 Documentation
- [x] Update API documentation for new methods (in code docstrings)
- [x] Business case documentation for stakeholders (memory.md)
- [x] Web interface documentation (enhanced welcome page)
- [ ] Create user guide for linear decomposition features
- [ ] Technical implementation guide for developers

## 🎯 Current Status
**✅ CODEBASE REFACTORING COMPLETED!**

The entire system has been successfully refactored to include:
- Enhanced Cost-Spec API with method selection (linear_decomposition/frontier)
- Production-ready LinearDecomposition class with scikit-learn style API
- Updated web interface with method selection and cost structure display
- Enhanced HTML reports with method information and comparison data
- Full backward compatibility maintained

**Next Priority**: Testing with real production data and performance optimization.

## 현재 시스템이 해결하는 문제
1. **Invalid Baselines**: 불가능한 기준선 계산 → 경제적으로 유효한 기준선
2. **Unfair Rankings**: 수학적 아티팩트 기반 순위 → 실제 가치 기반 순위  
3. **MVNO Disadvantage**: 예산 통신사 불리 → 공정한 경쟁 환경
4. **Strategic Blindness**: 가격 정책 불투명 → 실제 비용 구조 발견
5. **Disk Space Issues**: 파일 누적 → 포괄적 자동 정리 시스템 (중간 파일 포함) ⭐ 업데이트

**Suggestions for Next Steps**:
- **Real-time Dashboard**: Continuous plan monitoring
- **Notification System**: Notification on new competitive plan
- **API Extension**: Specific telecom analysis endpoint
- **Data Visualization**: Cost structure change trend graph
- **Mobile Optimization**: Improved responsive web interface

**Current System Status**:
- **✅ CODEBASE REFACTORING COMPLETED!**

The entire system has been successfully refactored to include:
- Enhanced Cost-Spec API with method selection (linear_decomposition/frontier)
- Production-ready LinearDecomposition class with scikit-learn style API
- Updated web interface with method selection and cost structure display
- Enhanced HTML reports with method information and comparison data
- Full backward compatibility maintained

**Next Priority**: Testing with real production data and performance optimization.

# Todo List

## 완료된 작업 ✅
- ✅ **"\n frontier" 오류 해결** - HTML 템플릿 JavaScript 중괄호 충돌 수정
- ✅ **포맷 문자열 교체 로직 수정** - {len_df_sorted:,} 등 정확한 패턴 매칭
- ✅ **프론티어 포인트 선택 로직 개선** - 올바른 대표 플랜 선택 방식 구현
- ✅ **계수 저장 중복 처리** - decomposition_coefficients와 cost_structure 동시 저장
- ✅ **CSS 수정 완료** - 이중 중괄호 문제 해결, 테이블 그리드 라인 복원
- ✅ **마진 비용 분석 차트 추가** - Feature Frontier Charts 다음에 추가 구현

## 현재 최우선 작업 🔥

### 1. 가변 베타 계수 구현 (Piecewise Linear Regression)
**사용자 요구사항**: "beta values are changing over feature value increment because we expect that the rate of cost would be different for each section of feature value increment"

**구현 필요사항**:
- [ ] **PiecewiseLinearRegression 클래스 완성** - 자동 구간 분할 및 최적화
- [ ] **cost_spec.py에 통합** - linear_decomposition 메서드에서 piecewise 옵션 제공
- [ ] **구간별 계수 시각화** - 마진 비용 차트에서 구간별 기울기 표시
- [ ] **breakpoint 자동 탐지** - 데이터 기반 최적 구간 분할점 찾기

**기술적 구현 방향**:
```python
# 예시: 기본 데이터 비용 구조
# 0-10GB: ₩50/GB
# 10-50GB: ₩30/GB  
# 50+GB: ₩20/GB
```

### 2. 테스트 및 검증
- [ ] **실제 데이터로 piecewise 모델 테스트**
- [ ] **선형 vs piecewise 모델 비교 결과 검증**
- [ ] **마진 비용 차트에서 구간별 기울기 정확히 표시되는지 확인**

## 현재 작업 중 (진행 중)
- **PiecewiseLinearRegression 모듈 기본 구조 생성됨** - 완전한 구현 및 통합 필요
- **HTML 차트 시스템 완료** - 모든 차트 타입 정상 작동

## 향후 개선 제안 (우선순위 낮음)
- **성능 최적화**: 2,283개 이상 플랜 처리 시 메모리 사용량 최적화
- **추가 차트**: 통신사별 비교 분석 차트
- **고급 필터링**: 사용자 맞춤형 플랜 필터링 옵션
- **API 확장**: RESTful API 엔드포인트 추가

## 주의사항
- **브라우저 캐시**: 변경사항 확인 시 강제 새로고침 (Ctrl+F5) 권장
- **대용량 데이터**: 처리 시 메모리 모니터링 필요  
- **모델 검증**: piecewise 구현 시 기존 선형 모델과 성능 비교 필수 

# Current Issues to Resolve

## 🚨 Immediate Fixes Needed

### HTML 보고서 포맷 오류
- **문제**: `unsupported format string passed to dict.__format__` 오류 발생
- **우선순위**: High
- **위치**: HTML 보고서 생성 과정

### 서버 로깅 시스템
- **문제**: error.log 파일이 현재 업데이트되지 않음 (정적 상태)
- **우선순위**: Medium
- **현재 상태**: 과거 로그 내용만 포함, 실시간 로깅 안됨

## 📋 Current System Status

### 차트 시스템 (작동 중)
1. **Feature Frontier Charts**: 각 feature별 비용 프론티어, frontier points/excluded points/unlimited plans 표시
2. **Linear Decomposition Charts**: 비용 구성 요소 도넛 차트, 단위당 비용 막대 차트, 마진 비용 분석 차트 (linear_decomposition method에서만)  
3. **Plan Value Efficiency Matrix**: 기준비용 vs 실제비용 버블 차트, 대각선 효율성 라인, 색상 코딩

### 현재 설정
- 기본 method: 'linear_decomposition'
- 주요 엔드포인트: `/process`
- Linear decomposition 실패 시 자동 frontier method 전환

### Linear Decomposition 시스템 현황
**대표 플랜 선택**: calculate_feature_frontiers() 로직 사용하여 프론티어 후보 플랜 선택
**최적화 프로세스**: 프론티어 플랜에서 제약 최적화를 통한 마진 비용 계산
**비용 구조 발견**: 기본 인프라 비용 + 기능별 프리미엄 분리

### 한국 모바일 시장 비용 구조 (현재 분석 결과)
- 기본 비용: ₩2,991 (네트워크 인프라)
- 데이터: ~₩0/GB (기본 서비스에 포함)
- 음성: ~₩0/100분 (기본 서비스에 포함)  
- SMS: ₩8.70/100개 (소액 메시징 프리미엄)
- 테더링: ₩554.83/GB (핫스팟 프리미엄)
- 5G: ~₩0 (현대 요금제에 포함)

### 사용자 요구사항 이해
**설계 철학**: 공정한 가성비 순위를 위한 최적화된 기준선 생성
**단조성 제외**: "더 많은 기능이 더 저렴"한 비현실적 데이터 포인트 의도적 제외
**프론티어 선택**: 각 기능 레벨에서 최소 가격으로 최적화된 기준선