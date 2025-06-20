# 🔧 MVNO Plan Ranking Model - Refactoring Plan (세부 검토 완료)

## 🎯 **Refactoring 목표**

### **시스템 현대화 및 최적화**
- **Performance**: 응답 시간 단축 및 메모리 사용량 최적화
- **Scalability**: 대용량 데이터셋 처리 능력 향상
- **Maintainability**: 코드 구조 개선 및 모듈화 강화
- **Reliability**: 오류 처리 및 복구 메커니즘 개선
- **Extensibility**: 새로운 기능 추가 용이성 확보

## 🏗️ **Phase 0: Code Modularization (Priority 1) - **상세 분석 완료**

### 🔬 **심층 분석 결과**

#### **1. cost_spec.py (2,746 lines) - 완전 분석**

**클래스 기반 분해 우선 순위:**

**🔴 1순위: FullDatasetMultiFeatureRegression (784 lines, 1795-2579)**
- **기능**: 전체 데이터셋 회귀 분석 (현재 활성 사용 중)
- **독립성**: 높음 - self.features, self.coefficients, 명확한 인터페이스
- **타겟**: `modules/regression/full_dataset.py`
- **메서드 수**: 15개 (init, solve_full_dataset_coefficients, detect_multicollinearity 등)
- **의존성**: sklearn, numpy만 필요

**🔴 2순위: MultiFeatureFrontierRegression (776 lines, 1018-1794)**
- **기능**: 다중 특성 프론티어 회귀 (레거시 코드)
- **독립성**: 높음 - 프론티어 수집 + 회귀 로직
- **타겟**: `modules/regression/multi_feature.py`
- **메서드 수**: 12개 (collect_all_frontier_plans, solve_multi_feature_coefficients 등)
- **의존성**: CORE_FEATURES, UNLIMITED_FLAGS

**🟡 3순위: LinearDecomposition (232 lines, 43-275)**
- **기능**: 선형 분해 분석 (미사용)
- **독립성**: 높음
- **타겟**: `modules/regression/linear_decomposition.py` (레거시 보관)

**함수 기반 분해:**

**🔴 4순위: Frontier Functions (337 lines, 276-613)**
- create_robust_monotonic_frontier(), calculate_feature_frontiers(), estimate_frontier_value()
- **타겟**: `modules/frontier/core.py`

**🔴 5순위: CS Ratio Functions (403 lines, 614-1017)**  
- calculate_cs_ratio(), rank_plans_by_cs(), calculate_cs_ratio_enhanced()
- **타겟**: `modules/cost_spec/ratio.py`

**🟢 6순위: Helper Functions (166 lines, 2580-2746)**
- **타겟**: `modules/cost_spec/utils.py`

#### **2. report_html.py (2,058 lines) - 핵심 문제 발견**

**🚨 핵심 문제: generate_html_report() 함수 1,507 lines (551-2058)**

**분해 전략:**
- **HTML 템플릿 추출**: 1,000+ lines → `modules/templates/main_template.py`
- **JavaScript 코드 분리**: 300+ lines → `modules/templates/chart_scripts.py`
- **CSS 스타일 분리**: 200+ lines → `modules/templates/styles.py`
- **내부 함수 추출**: get_chart_status_html() → `modules/report/status.py`

**기존 함수 분리:**
- prepare_cost_structure_chart_data() (133 lines) → `modules/report/chart_data.py`
- prepare_plan_efficiency_data() (68 lines) → `modules/report/efficiency.py`
- generate_feature_rates_table_html() (330 lines) → `modules/report/tables.py`

#### **3. report_charts.py (1,825 lines) - 함수별 분석**

**🔴 1순위: Marginal Cost Group (915 lines, 909-1824)**
- prepare_marginal_cost_frontier_data() (524 lines)
- prepare_granular_marginal_cost_frontier_data() (392 lines)
- **타겟**: `modules/charts/marginal_cost.py`

**🔴 2순위: Feature Frontier Group (347 lines, 14-361)**
- prepare_feature_frontier_data() (347 lines)
- **타겟**: `modules/charts/feature_frontier.py`

**🟡 3순위: Multi-Frontier Group (407 lines, 501-908)**
- prepare_multi_frontier_chart_data(), prepare_contamination_comparison_data()
- **타겟**: `modules/charts/multi_frontier.py` (레거시)

**🟡 4순위: Piecewise Utilities (221 lines, 686-907)**
- detect_change_points(), fit_piecewise_linear(), fit_piecewise_linear_segments()
- **타겟**: `modules/charts/piecewise_utils.py`

**🟢 5순위: Residual Analysis (138 lines, 362-500)**
- **타겟**: `modules/charts/residual.py`

### 🔄 **순환 의존성 분석 (Critical Issue)**

**현재 순환 의존성:**
```
report_html.py → report_charts.py → cost_spec.py
data_storage.py → report_charts.py + report_html.py  
report_charts.py → cost_spec.py (3곳에서 import)
```

**해결 방안:**
1. **Interface Layer 도입**: `modules/interfaces/chart_interface.py`
2. **Dependency Injection**: 함수 파라미터로 전달
3. **Configuration Registry**: 중앙 설정 관리

### 📊 **수정된 현실적 모듈화 계획**

#### **Week 1: cost_spec.py 분해 (우선순위 1)**

**Day 1-2: 회귀 클래스 추출**
```python
# modules/regression/full_dataset.py
class FullDatasetMultiFeatureRegression:
    # 784 lines → 독립 모듈

# modules/regression/multi_feature.py  
class MultiFeatureFrontierRegression:
    # 776 lines → 독립 모듈

# modules/regression/__init__.py
from .full_dataset import FullDatasetMultiFeatureRegression
from .multi_feature import MultiFeatureFrontierRegression
```

**Day 3-4: 함수 그룹 분리**
```python
# modules/frontier/core.py
def create_robust_monotonic_frontier(...)
def calculate_feature_frontiers(...)
def estimate_frontier_value(...)

# modules/cost_spec/ratio.py
def calculate_cs_ratio(...)
def calculate_cs_ratio_enhanced(...)
def rank_plans_by_cs(...)
```

**Day 5: 브리지 함수 및 호환성**
```python
# modules/cost_spec.py (축소됨 - 500 lines 이하)
from .regression import FullDatasetMultiFeatureRegression
from .frontier.core import create_robust_monotonic_frontier
from .ratio import calculate_cs_ratio_enhanced

# 기존 import 경로 유지 (6개월간)
```

#### **Week 2: report_html.py 분해**

**Day 1-2: 템플릿 분리**
```python
# modules/templates/main_template.py
def get_main_html_template() -> str:
    # 1,000+ lines HTML template

# modules/templates/chart_scripts.py  
def get_chart_javascript() -> str:
    # 300+ lines JavaScript

# modules/templates/styles.py
def get_report_styles() -> str:
    # 200+ lines CSS
```

**Day 3-4: 함수 추출**
```python
# modules/report/html_generator.py
def generate_html_report(...):
    # 300 lines 이하로 축소
    template = get_main_html_template()
    # 조립 로직만

# modules/report/status.py
def get_chart_status_html(...)

# modules/report/chart_data.py
def prepare_cost_structure_chart_data(...)
```

#### **Week 3: report_charts.py 분해**

**Day 1-2: 핵심 차트 그룹**
```python
# modules/charts/marginal_cost.py
def prepare_marginal_cost_frontier_data(...)  # 524 lines
def prepare_granular_marginal_cost_frontier_data(...)  # 392 lines

# modules/charts/feature_frontier.py
def prepare_feature_frontier_data(...)  # 347 lines
```

**Day 3-4: 유틸리티 분리**
```python
# modules/charts/piecewise_utils.py
def detect_change_points(...)
def fit_piecewise_linear(...)

# modules/charts/residual.py
def prepare_residual_analysis_data(...)
```

#### **Week 4: 의존성 정리 및 통합 테스트**

**Day 1-2: 순환 의존성 해결**
```python
# modules/interfaces/chart_interface.py
class ChartDataInterface:
    @abstractmethod
    def prepare_feature_frontier_data(...)
    @abstractmethod  
    def prepare_marginal_cost_data(...)

# 의존성 주입으로 순환 참조 제거
```

**Day 3-4: 브리지 레이어 구현**
```python
# modules/legacy_bridge.py
import warnings
from .regression import FullDatasetMultiFeatureRegression

def calculate_cs_ratio_enhanced(*args, **kwargs):
    warnings.warn("Import path deprecated, use modules.cost_spec.ratio", DeprecationWarning)
    from .cost_spec.ratio import calculate_cs_ratio_enhanced
    return calculate_cs_ratio_enhanced(*args, **kwargs)
```

**Day 5: 통합 테스트**
- 모든 기존 import 경로 호환성 확인
- /process 엔드포인트 완전 테스트
- 차트 생성 파이프라인 검증

### 📈 **수정된 성공 기준**

**모듈 크기:**
- ⭐ **개별 파일**: < 800 lines (기존 500에서 현실적으로 조정)
- ⭐ **핵심 함수**: < 200 lines (generate_html_report 1,507 lines 해결)
- ⭐ **클래스 메서드**: < 100 lines

**의존성:**
- ⭐ **순환 의존성**: 0개 (현재 3개)
- ⭐ **계층 구조**: 최대 3 depth
- ⭐ **인터페이스**: 모든 모듈 간 명확한 계약

**호환성:**
- ⭐ **기존 import**: 6개월간 100% 지원
- ⭐ **API 변화**: 0개 (모든 기존 함수 시그니처 유지)
- ⭐ **성능**: 10% 이내 오버헤드

### ⚠️ **위험 관리**

**고위험 요소:**
1. **generate_html_report() 1,507 lines** - 템플릿 분리 중 문법 오류 가능성
2. **순환 의존성** - 잘못된 추출 시 import 에러 
3. **클래스 상태 관리** - self.coefficients 등 상태 분리

**완화 전략:**
1. **단계별 복사본 생성** - 기존 파일 보존하며 새 구조 구축
2. **자동화된 테스트** - 각 단계마다 /process 엔드포인트 전체 테스트
3. **롤백 계획** - Git 브랜치 전략으로 즉시 되돌리기 가능

### 🎯 **Phase 0 완료 후 예상 구조**

```
modules/
├── regression/
│   ├── __init__.py
│   ├── full_dataset.py (780 lines)
│   ├── multi_feature.py (770 lines) 
│   └── linear_decomposition.py (230 lines)
├── frontier/
│   ├── __init__.py
│   ├── core.py (250 lines)
│   └── utils.py (80 lines)
├── cost_spec/
│   ├── __init__.py
│   ├── ratio.py (400 lines)
│   └── constants.py (50 lines)
├── report/
│   ├── __init__.py
│   ├── html_generator.py (300 lines)
│   ├── chart_data.py (130 lines)
│   ├── status.py (100 lines)
│   └── tables.py (330 lines)
├── templates/
│   ├── __init__.py
│   ├── main_template.py (600 lines)
│   ├── chart_scripts.py (300 lines)
│   └── styles.py (200 lines)
├── charts/
│   ├── __init__.py
│   ├── marginal_cost.py (650 lines)
│   ├── feature_frontier.py (350 lines)
│   ├── piecewise_utils.py (220 lines)
│   └── residual.py (140 lines)
└── interfaces/
    ├── __init__.py
    └── chart_interface.py (150 lines)
```

**총 파일 수**: 28개 (기존 3개 → 25개 증가)
**최대 파일 크기**: 780 lines (목표 달성)
**순환 의존성**: 0개 (인터페이스 레이어로 해결)

---

## Phase 1-6: 후속 계획 (Phase 0 완료 후 진행)

### Phase 1: Core Algorithm Optimization (Month 2-3)
- 회귀 알고리즘 성능 최적화
- 캐싱 전략 구현
- 병렬 처리 개선

### Phase 2: Data Processing Modernization (Month 4-5)  
- Pandas 최적화
- 메모리 사용량 개선
- 스트리밍 데이터 처리

### Phase 3: Performance Enhancement (Month 6-7)
- 비동기 처리 확장
- 데이터베이스 통합
- API 응답 시간 최적화

### Phase 4: Intelligence & Analytics (Month 8-9)
- 머신러닝 모델 통합
- 예측 분석 기능
- 이상치 탐지 개선

### Phase 5: Data Architecture Modernization (Month 10-11)
- 마이크로서비스 아키텍처
- 실시간 데이터 파이프라인
- 스케일링 전략

### Phase 6: Security & Monitoring (Month 12)
- 보안 강화
- 로깅 개선
- 모니터링 대시보드

---

## 🚀 **즉시 실행 가능한 첫 단계**

1. **regression 폴더 생성** 및 클래스 복사
2. **FullDatasetMultiFeatureRegression** 추출 및 테스트
3. **기존 import 경로** 브리지 함수 생성
4. **/process 엔드포인트** 완전 동작 확인

**예상 소요 시간**: Phase 0 완료까지 4주
**리스크 레벨**: 중간 (체계적 접근으로 관리 가능)
**백워드 호환성**: 100% 보장