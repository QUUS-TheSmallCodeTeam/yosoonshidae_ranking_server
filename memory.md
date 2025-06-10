# 프로젝트 메모리

## 시스템 정보
- 운영체제: Linux 5.10.236-228.935.amzn2.x86_64
- 워크스페이스: vscode-remote://ssh-remote%2Bssh.hf.space/app
- 쉘: /bin/sh

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

## 비즈니스 로직

### 🚨 CRITICAL ISSUE DISCOVERED: Frontier Point Exclusion Timing
- **Problem**: Frontier points excluded BEFORE decomposition based on bundled costs
- **Impact**: 15-25% of potentially valid plans wrongly excluded
- **Location**: `create_robust_monotonic_frontier()` in `modules/cost_spec.py` lines 238-418
- **Current Logic**: Plans excluded if cost increase < ₩1,000/feature unit using bundled costs
- **Issue**: Plans that appear inefficient in bundled form might be efficient after marginal cost decomposition
- **Solution Required**: Post-decomposition frontier refinement

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