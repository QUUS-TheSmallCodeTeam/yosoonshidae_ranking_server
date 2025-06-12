# 프로젝트 메모리

## 시스템 정보
- 운영체제: Linux 5.10.236-228.935.amzn2.x86_64
- 워크스페이스: vscode-remote://ssh-remote%2Bssh.hf.space/app
- 쉘: /bin/sh

## Hugging Face Dev Mode 환경 ⭐ 중요
- **현재 환경**: Hugging Face Space에서 Dev Mode 활성화 상태
- **서버 상태**: localhost:7860에서 상시 실행 중 (절대 종료 금지)
- **코드 반영**: 파일 수정 시 서버에 즉시 반영됨 (재시작 불필요)
- **Git 상태**: Dev Mode에서의 변경사항은 자동으로 Git에 저장되지 않음
- **중요사항**: 서버 종료 시 Dev Mode 비활성화될 위험 있음 → 절대 프로세스 kill 금지
- **참고**: [Hugging Face Dev Mode 문서](https://huggingface.co/docs/hub/spaces-dev-mode)

## 프로젝트 개요
- **핵심 기능**: Enhanced Cost-Spec Ratio를 사용한 모바일 요금제 가성비 분석 및 순위 시스템
- **주요 기술**: FastAPI, pandas, scikit-learn, Linear Decomposition
- **기본 방법**: Linear Decomposition (기본값, 권장) vs Frontier-based (기존 방법)

## 시스템 아키텍처

### API 엔드포인트
- `POST /process`: 메인 데이터 처리 엔드포인트 (Enhanced Cost-Spec 분석)
- `GET /`: HTML 보고서 서빙 및 웰컴 페이지
- `POST /test`: 간단한 에코 테스트 엔드포인트

### 데이터 플로우
1. **요청 수신**: JSON 형태의 요금제 데이터 수신
2. **전처리**: prepare_features()를 통한 특성 정규화 및 정제
3. **분석**: Linear Decomposition 또는 Frontier 방법 적용
4. **순위 계산**: Cost-Spec 비율 기반 순위 산정
5. **결과 저장**: 전역 상태 및 파일 저장
6. **HTML 보고서**: 웹 인터페이스를 통한 결과 시각화

### 데이터 관리 시스템 ⭐ 신규 추가
- **자동 정리**: 새 데이터 처리 시 이전 데이터셋 자동 삭제
- **보존 정책**: 최대 1개 파일, 최대 1일 보존 (파이프라인이 처음부터 재계산하므로)
- **포괄적 정리 대상**: 
  - Raw data files (`raw_data_*.json`, `received_data_*.json`)
  - Processed data files (`processed_data_*.csv`, `latest_processed_data.csv`)
  - HTML reports (`*ranking_*.html`)
  - Intermediate files (`results/*.csv`, `results/*.json`)
- **메모리 모니터링**: psutil을 사용한 메모리 사용량 추적
- **정리 로직**: 새 요청마다 이전 중간 파일들이 무의미해지므로 적극적 정리

## 알고리즘 상세

### Linear Decomposition 방법 (기본값)
- **목적**: 개별 특성의 실제 비용 구조 분해
- **수학적 모델**: `plan_cost = β₀ + β₁×data + β₂×voice + β₃×SMS + β₄×tethering`
- **제약 조건**: 
  - 음수 비용 금지 (βⱼ ≥ 0)
  - Frontier 제약 (모델 예측 ≥ 실제 비용 - 허용 오차)
  - 데이터 기반 최소 기본 비용
- **비즈니스 가치**: 공정한 가성비 평가, 실제 마진 비용 파악

### Frontier 방법 (기존)
- **목적**: 각 특성별 최소 비용 경계선 발견
- **문제점**: 서로 다른 요금제의 완전 비용을 합산하여 불가능한 기준선 생성
- **결과**: 4-7배 인플레이션된 CS 비율

## 파일 구조

### 주요 모듈
- `modules/cost_spec.py`: LinearDecomposition 클래스 및 Enhanced Cost-Spec 함수
- `modules/preprocess.py`: 데이터 전처리 및 특성 정규화
- `modules/report_html.py`: HTML 보고서 생성 (방법별 정보 포함)
- `modules/utils.py`: 유틸리티 함수 (정리 기능 포함) ⭐ 신규 추가
- `modules/config.py`: 전역 설정 및 디렉토리 구조
- `app.py`: FastAPI 애플리케이션 및 엔드포인트

### 데이터 디렉토리
- `data/raw/`: 원시 JSON 데이터 저장
- `data/processed/`: 전처리된 CSV 데이터 저장
- `reports/`: HTML 보고서 저장
- `results/`: 결과 아카이브

## 기술적 특징

### 성능 최적화
- **메모리 관리**: 처리 후 DataFrame 정리 및 가비지 컬렉션
- **파일 정리**: 자동 이전 파일 삭제로 디스크 공간 절약 ⭐ 신규 추가
- **병렬 처리**: scikit-learn 기반 최적화된 선형 대수 연산

### 오류 처리
- **견고성**: HTML 보고서 생성 실패 시에도 API 응답 유지
- **로깅**: 상세한 로그 기록 및 오류 추적
- **Fallback**: 메모리 내 데이터 우선, 파일 기반 보조

### 웹 인터페이스
- **방법 선택**: Linear Decomposition vs Frontier 방법 선택 버튼
- **실시간 정보**: 비용 구조 발견 정보 표시
- **비교 모드**: 두 방법 간 결과 비교 기능
- **비용 구조 시각화**: ⭐ 신규 추가
  - 도넛 차트: 전체 비용 구성 요소 분해 (기본 인프라, 데이터, 음성, SMS, 테더링, 5G)
  - 막대 차트: 단위당 마진 비용 (₩/GB, ₩/100분, ₩/100SMS 등)
  - 한국어 라벨 및 툴팁으로 비즈니스 해석 제공
- **요금제 가성비 매트릭스**: ⭐ 신규 추가
  - 버블 차트: 기준 비용 vs 실제 비용 2D 분석
  - 대각선 효율성 기준선 (CS = 1.0)
  - 색상 코딩: 녹색(가성비 좋음) vs 빨간색(과가격)
  - 버블 크기: 총 기능 수준 표시
  - 구식 잔여 분석 테이블 제거 및 대체
- **마진 비용 분석 차트**: ⭐ 신규 추가
  - 기능별 마진 비용 계수 시각화 (β₁, β₂, β₃...)
  - 각 기능의 단위당 추가 비용 분석
  - 비즈니스 해석 툴팁 (예: "데이터 1GB 추가시 ₩50 비용 증가")
  - 기본 인프라 비용과 분리된 순수 마진 비용 표시

## 비즈니스 로직

### 💡 DESIGN UNDERSTANDING CLARIFIED: Monotonicity Exclusion is Intentional
- **User Intent**: Exclude non-monotonic data points BY DESIGN to ensure reasonable cost trends
- **Purpose**: Create most optimistic baseline (lowest possible cost) for fair 가성비 ranking
- **Logic**: More features should cost more (common sense) - exclude contradictory data
- **Frontier Selection**: Minimum price at each feature level for lowest possible baseline
- **Tethering Example**: Plans where more tethering costs less are correctly excluded as unreasonable
- **Result**: Tethering coefficient ≈ ₩0/GB because insufficient reasonable tethering data exists
- **System Working As Intended**: This is not a bug but a feature for realistic ranking

### 비용 구조 발견
- **기본 비용**: 네트워크 인프라 및 고객 서비스 비용
- **데이터 비용**: GB당 마진 비용 (보통 ₩10-50/GB)
- **음성 비용**: 100분당 비용 (스펙트럼 및 교환 비용)
- **테더링 프리미엄**: 프리미엄 기능 추가 비용

### 경쟁 분석
- **공정한 순위**: 인위적 인플레이션 제거
- **MVNO 경쟁력**: 예산 통신사의 실제 경쟁력 반영
- **전략적 통찰**: 경쟁사 가격 정책 분석

## 설정 옵션

### API 옵션
- `method`: 'linear_decomposition' (기본값) 또는 'frontier'
- `featureSet`: 사용할 특성 세트 ('basic' 기본값)
- `feeColumn`: 비교할 요금 컬럼 ('fee' 기본값)
- `tolerance`: 최적화 허용 오차 (500 기본값)
- `includeComparison`: Frontier 방법과 비교 포함 여부

### 정리 설정 ⭐ 신규 추가
- `max_files`: 보존할 최대 파일 수 (3개 기본값)
- `max_age_days`: 파일 보존 기간 (5일 기본값)

## 의존성
- fastapi==0.115.12: 웹 프레임워크
- pandas==2.2.3: 데이터 처리
- scikit-learn==1.6.1: 기계학습 및 최적화
- psutil==6.1.1: 시스템 모니터링 ⭐ 신규 추가
- numpy, matplotlib, jinja2: 보조 라이브러리

## 작업 원칙
- **자율적 문제 해결**: 사용자 승인 없이 독립적 수행
- **완결성 보장**: 작업 완전 해결까지 대화 지속
- **코드 검증**: 수정 후 항상 재검토 및 작동 확인
- **즉시 오류 수정**: 발견된 모든 오류 즉시 해결

# 현재 작업 상황

## 해결된 문제
- ✅ **Cost Structure Chart 표시 문제**: Linear decomposition 방법의 `attrs['decomposition_coefficients']`가 `attrs['cost_structure']`로도 저장되도록 수정하여 HTML 리포트와 `/process` 엔드포인트에서 차트가 정상 표시되도록 해결
- ✅ **차트 표시 조건 개선**: Cost Structure Charts가 `method="linear_decomposition"`일 때만 표시되도록 수정

## 현재 디버깅 중인 문제
- 🔍 **Linear Decomposition 실행 실패**: 차트가 아예 표시되지 않는 문제 발견
- 🔍 **Feature 존재 문제**: linear decomposition에서 사용하는 features가 실제 data에 존재하지 않을 가능성
- ✅ **로깅 추가**: 상세한 디버깅 로그와 fallback 로직 추가
- ✅ **Feature 안전성 개선**: 실제 DataFrame에 존재하는 features만 사용하도록 수정

## 구현된 안전장치
- Exception handling과 fallback to frontier method
- Feature 존재 확인 및 최소 3개 feature 요구사항
- 상세한 로깅으로 실행 과정 추적
- cost_structure를 float로 변환하여 JSON 직렬화 안전성 확보

## 현재 차트 구성
1. **Feature Frontier Charts** (모든 method에서 표시):
   - 각 feature별 비용 프론티어 차트
   - Frontier points, excluded points, unlimited plans 표시
   - 사용자가 언급한 "frontier data point selection처럼 feature별 차트"가 이미 구현됨

2. **Linear Decomposition Charts** (linear_decomposition method에서만 표시):
   - 비용 구성 요소 도넛 차트
   - 단위당 비용 막대 차트  
   - 발견된 마진 비용 분석 차트

3. **Plan Value Efficiency Matrix** (모든 method에서 표시):
   - 기준비용 vs 실제비용 버블 차트
   - 대각선 효율성 라인 (CS = 1.0)
   - 색상 코딩: 초록(좋은 가치) vs 빨강(비쌈)

## 현재 상태 확인
- 사용자는 `/process` 엔드포인트만 사용하며, 기본값이 `method='linear_decomposition'`로 설정됨
- Debug 정보 섹션이 method 및 cost_structure 상태를 보여줌
- Feature별 frontier 차트는 이미 모든 method에서 표시되고 있음
- Linear decomposition 실패 시 automatic fallback to frontier method

## 기술적 세부사항
- Chart.js 사용한 시각화
- 한국어 지원
- 기존 인터페이스와 일치하는 반응형 디자인
- JSON을 통한 JavaScript로 데이터 전달
- JavaScript 콘솔 로깅으로 디버깅 지원
- 상세한 서버 로깅으로 실행 과정 추적

## 다음 단계
- 사용자의 실제 데이터로 테스트하여 로그 확인 및 linear decomposition이 실제 작동하는지 검증

# 테스트 워크플로 ⭐ 필수 절차

## 코드 수정 후 표준 테스트 절차
1. **코드 수정 완료**
   - 파일 편집 후 자동으로 서버에 반영됨 (Dev Mode 환경)
   - 별도 재시작 불필요

2. **End-to-End 테스트 실행** (필수)
   - **목적**: `/process` 엔드포인트가 전체 코드베이스의 핵심 기능
   - **방법 1** (선호): `/data/raw` 폴더의 JSON 데이터 사용
     ```bash
     curl -X POST localhost:7860/process \
          -H "Content-Type: application/json" \
          -d @data/raw/[JSON_FILE_NAME].json
     ```
   - **방법 2** (백업): Supabase 함수 사용
     ```bash
     curl -X POST https://zqoybuhwasuppzjqnllm.supabase.co/functions/v1/submit-data \
          -H "Content-Type: application/json" \
          -d "{}"
     ```
     → 이 방법은 유사한 데이터셋으로 자동으로 서버의 `/process` 엔드포인트 호출

3. **응답 검증**
   - HTTP 상태 코드 확인 (200 OK 기대)
   - 응답 JSON 구조 및 데이터 검증
   - 로그 메시지 확인 (특히 linear decomposition 실행 여부)

4. **웹 인터페이스 확인** (추가 검증)
   - `http://localhost:7860/` 접속
   - HTML 보고서 정상 생성 확인
   - 차트 표시 상태 확인

## 테스트 데이터 관리
- **우선순위**: `/data/raw` 폴더 내 JSON 파일 사용
- **백업 방법**: Supabase 외부 엔드포인트 (동일한 효과)
- **데이터 구조**: 요금제 정보가 포함된 JSON 배열 형태

## 문제 해결 체크리스트
- [ ] 서버가 7860 포트에서 실행 중인가?
- [ ] `/process` 엔드포인트 응답이 정상인가?
- [ ] Linear decomposition이 실행되는가? (로그 확인)
- [ ] HTML 보고서가 생성되는가?
- [ ] 차트가 정상 표시되는가?
- [ ] 메모리 사용량이 정상 범위인가?

## 중요한 제약사항
- ⚠️ **절대 서버 종료 금지**: Dev Mode 비활성화 위험
- ⚠️ **Git 수동 커밋 필요**: 변경사항은 자동 저장되지 않음
- ⚠️ **테스트 필수**: 코드 수정 후 반드시 `/process` 엔드포인트 테스트 

## Current Status: Marginal Cost System CORRECTED

### ✅ CRITICAL FIX APPLIED: Frontier-Based Linear Decomposition
**Issue**: Previously used arbitrary "market segments" for linear decomposition representative plan selection
**Fix**: Now uses same frontier candidate point logic as original frontier method
**Impact**: Ensures linear decomposition uses optimal cost-efficient plans rather than random samples

### Enhanced Visualization System - IMPLEMENTED
**Three Chart Types Successfully Added:**
1. **Cost Structure Decomposition Charts** - doughnut and bar charts showing β coefficients from linear decomposition
2. **Plan Value Efficiency Matrix** - 2D bubble chart comparing baseline vs actual costs with diagonal efficiency line  
3. **Marginal Cost Analysis Chart** - individual feature marginal costs with business interpretation tooltips

### User Requirement Clarification - CORRECTED UNDERSTANDING
**Previous Error**: Thought monotonicity exclusion was a bug
**Corrected Understanding**: Excluding non-monotonic data points is BY DESIGN for creating optimistic baseline ranking
**Business Logic**: Plans where "more features cost less" should be excluded as unrealistic for fair 가성비 ranking

### Technical Implementation Details
**Linear Decomposition Process:**
- Now extracts frontier points using `calculate_feature_frontiers()` logic
- Selects plans that contribute to optimal cost frontiers for each feature
- Solves constrained optimization on these frontier plans only
- Discovers true marginal costs: base infrastructure + per-feature premiums

**Current Results (Realistic Korean Market Structure):**
- Base cost: ₩2,991 (covers basic network infrastructure)
- Data: ~₩0/GB (commoditized, bundled into base)
- Voice: ~₩0/100min (bundled into base service)  
- SMS: ₩8.70/100msg (small messaging premium)
- Tethering: ₩554.83/GB (significant hotspot premium)
- 5G: ~₩0 (included in modern plans)

### Chart Display Status
**Current**: All chart functionality implemented and should display correctly
**Fixed Issues**: 
- Cost structure data format compatibility (nested vs flat)
- JavaScript chart implementation restored
- Frontier-based representative plan selection implemented

### Next Steps
**Immediate**: Test updated frontier-based linear decomposition
**Future**: Consider post-decomposition frontier refinement for broader plan inclusion

---
*Last Updated: After critical frontier-based selection fix*

## 완료된 작업
- ✅ **Cost-Spec 시스템 기본 구현** - 선형 분해 및 프론티어 기반 방법론
- ✅ **프론티어 포인트 선택 로직 수정** - calculate_feature_frontiers() 로직 사용하여 올바른 대표 플랜 선택
- ✅ **HTML 템플릿 오류 해결** - "\n frontier" KeyError 수정 (JavaScript 중괄호 충돌 문제)
- ✅ **CSS 수정 완료** - 이중 중괄호 문제 해결, 테이블 그리드 라인 정상 표시
- ✅ **마진 비용 분석 차트 추가** - 프론티어 차트와 동일한 형태로 마진 비용 적용 버전 구현
- ✅ **마진 비용 차트 데이터 구조 수정** - feature_costs 및 base_cost를 cost structure 데이터에 포함

## 해결된 주요 이슈

### CSS 표시 문제 (2025-06-10)
**문제**: CSS가 적용되지 않고 이중 중괄호 `{{` 로 표시됨
**원인**: .format() 에서 .replace() 로 변경하면서 CSS의 이중 중괄호가 그대로 유지됨
**해결책**: 모든 CSS 규칙을 단일 중괄호 `{` 로 수정
```css
/* 기존 (오류): body {{ margin: 0; }} */
/* 수정 (정상): body { margin: 0; } */
```

### HTML 템플릿 포맷팅 오류 (2025-06-10)
**문제**: HTML 보고서 생성 시 `KeyError: '\n frontier'` 오류 발생
**원인**: JavaScript 코드의 중괄호 `{}` 가 Python .format() 메서드에서 포맷 플레이스홀더로 해석됨
**해결책**: .format() 대신 개별 .replace() 호출로 변경

### 프론티어 포인트 선택 로직 개선 (2025-06-09)
**문제**: 임의의 시장 세그먼트 대신 실제 프론티어 후보 선택 로직 필요
**해결책**: `calculate_feature_frontiers()` 와 동일한 'frontier_points' 선택 방식 구현

## 현재 구현된 차트 시스템
1. **Cost Structure Decomposition Charts** - 선형 분해 계수 시각화
2. **Feature Frontier Charts** - 시장 최소 비용 프론티어 표시
3. **Marginal Cost Analysis Charts** - 마진 비용 계수를 적용한 이론적 비용 라인 vs 시장 데이터
4. **Plan Value Efficiency Matrix** - 가성비 분석 버블 차트

## 사용자 요구사항 - 구현 대기 중
### 가변 베타 계수 (Piecewise Linear Regression)
**요구사항**: "beta values are changing over feature value increment because we expect that the rate of cost would be different for each section of feature value increment"

**의미**: 기능 값 범위에 따라 다른 마진 비용 적용
- 예: 기본 데이터 0-10GB: ₩50/GB
- 예: 기본 데이터 10-50GB: ₩30/GB  
- 예: 기본 데이터 50+GB: ₩20/GB

**현재 상태**: PiecewiseLinearRegression 모듈 생성됨, 코어 로직에 통합 필요

## 현재 시스템 상태
- **방법론**: 선형 분해 (기본), 프론티어 기반 (비교용)
- **한국 시장 비용 구조**: 기본 ₩2,991, 데이터 ~₩0, 음성 ~₩0, SMS ₩8.70, 테더링 ₩554.83, 5G ~₩0
- **차트 시스템**: 4가지 차트 타입 완전 구현
- **HTML 보고서**: CSS 및 JavaScript 정상 작동, 완전 자동화

## 기술 스택
- **백엔드**: FastAPI, pandas, scipy.optimize, numpy
- **프론트엔드**: Chart.js, 반응형 HTML/CSS  
- **데이터 처리**: 2,283개 플랜 실시간 분석 지원
- **차트**: 동적 생성, 인터랙티브 툴팁, 다중 데이터셋 지원 

### 마진 비용 차트 렌더링 문제 (2025-06-10)
**문제**: Marginal Cost Analysis Charts가 렌더링되지 않음
**원인**: JavaScript가 `costStructureData.feature_costs`를 기대했지만 cost structure 데이터에 포함되지 않음
**해결책**: `prepare_cost_structure_chart_data()` 함수 수정하여 원시 `feature_costs`와 `base_cost` 포함
```javascript
// 기존: costStructureData = { overall: {...}, unit_costs: {...} }
// 수정: costStructureData = { overall: {...}, unit_costs: {...}, feature_costs: {...}, base_cost: 3000 }
```
**결과**: 마진 비용 차트가 정상적으로 이론적 비용 라인과 시장 데이터 포인트 표시 