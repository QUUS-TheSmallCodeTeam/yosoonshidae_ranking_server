# 🧠 Memory & Context

## 📊 Current System Status
- **Async chart calculation**: Implemented to eliminate continuous calculations triggered by root endpoint
- **Visual status indicators**: Loading icons (⚙️) for in-progress, error icons (❌) for failed calculations
- **Manual refresh system**: No auto-polling, users manually refresh to check progress
- **No caching**: All HTML content generated fresh on each request for immediate status updates
- **Multi-frontier regression methodology**: Successfully implemented and fully operational
- **Chart visualization**: Advanced charts now calculated asynchronously in background
- **API response time**: Immediate response from /process endpoint, charts calculated separately
- **Default method**: Changed to `multi_frontier` for new analysis approach
- **Chart data format**: Fixed JavaScript chart functions to handle nested cost structure objects
- **Linear decomposition charts**: Now properly extracting coefficient values from nested data structures

## 🎯 User Requirements & Preferences
- **No auto-refresh**: Manual refresh only, no constant polling
- **Visual feedback**: Clear status indicators for chart calculation progress
- **Immediate API response**: /process endpoint returns instantly, charts calculated separately
- **Fresh content**: No caching, all content generated on-demand
- **Comprehensive analysis**: Both frontier and linear decomposition methods displayed together

## 🔧 Technical Implementation Details
- **Infinite loop fix**: Added safety counters and division-by-zero checks in `prepare_feature_frontier_data`
- **Logging optimization**: Reduced verbose logging to prevent SSH polling spam
- **Dual method display**: Shows both multi-frontier and linear decomposition results simultaneously
- **Chart data handling**: JavaScript functions now properly parse nested coefficient objects
- **Background processing**: Chart calculations run asynchronously without blocking API responses

## 🎯 Working Methods
- **Multi-frontier regression**: Eliminates cross-contamination by using complete feature vectors
- **Feature frontier charts**: Original logic maintained as requested
- **Safety measures**: Infinite loop prevention implemented and working
- **Numpy type conversion**: Comprehensive serialization fix for all data types
- **Async processing**: Chart calculations run in background, API responds immediately

## 🔧 Implementation Patterns
- **Async chart calculation**: Background tasks for expensive visualizations
- **Progressive status display**: Real-time progress indicators for chart generation
- **Fallback mechanisms**: Basic HTML reports when charts fail or are in progress
- **Method integration**: New methods added to existing cost_spec.py structure
- **Error handling**: Robust type conversion and safety measures
- **Testing workflow**: Using raw data files from /data/raw/ directory
- **Clean server startup**: Direct uvicorn command in Dockerfile, log monitoring via app.py startup event

## 📈 Data Flow
- Raw data → Multi-frontier regression → CS ratio calculation → Immediate API response
- Background: Chart generation → HTML report with visualizations → Cache update
- Feature frontier analysis for each core feature (data, voice, messages, tethering, 5G)
- Proper frontier point selection (single cheapest plan per feature level)
- Cross-contamination eliminated through multi-feature regression approach

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
- **상태**: API 및 웹 인터페이스 정상 작동 확인

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
   
   **방법 1** (선호): Supabase 함수 사용 (실제 데이터셋 테스트)
   ```bash
   # 터미널 1: 필터링된 로그 모니터링 시작했는지 체크 (필수!)
   ./simple_log_monitor.sh &
   # 1개만 실행하도록!

   # 터미널 2: 테스트 실행 (환경변수 참조)
   source .env.local && curl -X POST https://zqoybuhwasuppzjqnllm.supabase.co/functions/v1/submit-data \
        -H "Authorization: Bearer $service_role" \
        -H "Content-Type: application/json" \
        -d "{}"
   
   # 터미널 3: 로그 확인
   tail -f error.log
   ```
   → 이 방법은 유사한 데이터셋으로 자동으로 서버의 `/process` 엔드포인트 호출
   → **GET 스팸 필터링**: 무한 keep-alive 요청은 제외하고 실제 로그만 저장
   → **환경변수**: .env.local 파일의 service_role 키 자동 참조

   **방법 2** (로컬 데이터): `/data/raw` 폴더의 JSON 데이터 사용
   ```bash
   curl -X POST localhost:7860/process \
        -H "Content-Type: application/json" \
        -d @data/raw/[JSON_FILE_NAME].json
   ```

### 4. **비동기 차트 계산 검증** (새로 추가)
   - **차트 상태 확인**: `curl localhost:7860/chart-status`
   - **진행 상황 모니터링**: 차트 계산 진행률 및 상태 확인
   - **웹 인터페이스**: 루트 페이지에서 진행 상태 또는 완성된 차트 확인

### 5. **서버사이드 로그 검증** (핵심)
   - **비동기 차트 계산 시작** 로그 확인
   - **Cost Structure 계산 과정** 추적
   - **오류 메시지** 발생 여부
   - **메모리 사용량** 및 **처리 시간** 확인
   - **Feature 존재 여부** 및 **계수 계산** 성공 확인

### 6. **응답 검증**
   - HTTP 상태 코드 확인 (200 OK 기대)
   - 응답 JSON 구조 및 데이터 검증
   - `cost_structure` 키 존재 및 값 확인
   - `chart_status` 필드 확인 (calculating/ready/error)

### 7. **웹 인터페이스 확인** (추가 검증)
   - `http://localhost:7860/` 접속
   - 진행 상태 페이지 또는 완성된 HTML 보고서 확인
   - 차트 표시 상태 확인 (비동기 완료 후)

## 테스트 데이터 관리
- **우선순위**: `/data/raw` 폴더 내 JSON 파일 사용
- **백업 방법**: Supabase 외부 엔드포인트 (동일한 효과)
- **데이터 구조**: 요금제 정보가 포함된 JSON 배열 형태

## 문제 해결 체크리스트

### 서버 상태 확인
- [ ] 서버가 7860 포트에서 실행 중인가? (`ps aux | grep python`)
- [ ] 서버 프로세스 ID 확인 (일반적으로 PID 9)
- [ ] 로그 모니터링이 설정되어 있는가?

### API 테스트 
- [ ] `/process` 엔드포인트 응답이 정상인가? (HTTP 200)
- [ ] 응답 JSON에 `cost_structure` 키가 존재하는가?
- [ ] `chart_status` 필드가 "calculating"으로 설정되는가?
- [ ] Supabase 외부 엔드포인트 테스트가 성공하는가?

### 비동기 차트 계산 검증 ⭐ 새로 추가
- [ ] `/chart-status` 엔드포인트가 정상 응답하는가?
- [ ] 차트 계산 진행률이 0→10→30→50→80→100으로 진행되는가?
- [ ] 차트 계산 완료 후 캐시된 HTML이 제공되는가?
- [ ] 차트 계산 중 루트 페이지에서 진행 상태가 표시되는가?

### 서버사이드 로그 검증 ⭐ 핵심
- [ ] 비동기 차트 계산 시작 로그가 나타나는가?
- [ ] Cost structure 계산 과정이 로그에 기록되는가?
- [ ] Feature 존재 확인 메시지가 있는가?
- [ ] 오류나 예외 메시지가 발생하지 않는가?
- [ ] 연속 계산 로그가 더 이상 발생하지 않는가?

### 웹 인터페이스 확인
- [ ] 진행 상태 페이지가 정상 표시되는가?
- [ ] 차트 계산 완료 후 HTML 보고서가 생성되는가?
- [ ] 차트가 정상 표시되는가?
- [ ] 메모리 사용량이 정상 범위인가?

## 중요한 제약사항
- ⚠️ **절대 서버 종료 금지**: Dev Mode 비활성화 위험
- ⚠️ **Git 수동 커밋 필요**: 변경사항은 자동 저장되지 않음
- ⚠️ **테스트 필수**: 코드 수정 후 반드시 `/process` 엔드포인트 테스트
- ⚠️ **비동기 검증**: 차트 계산 상태 및 완료 여부 확인 필수