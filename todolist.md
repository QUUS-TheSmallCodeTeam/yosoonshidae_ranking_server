# 📋 할 일 목록

## 🔥 High Priority

### **Multicollinearity Resolution in Regression Analysis**
- **Problem**: `message_unlimited` and `additional_call` coefficients are ₩0.0000 due to severe multicollinearity
- **Impact**: Missing ₩18,482 value for unlimited messaging and ₩16,261 value for additional calls in cost model
- **Solutions to implement**:
  - [ ] Add Ridge/Lasso regularization to handle multicollinearity
  - [ ] Implement variance inflation factor (VIF) checking
  - [ ] Create composite features (e.g., `unlimited_communication_bundle`)
  - [ ] Add feature selection algorithm to automatically handle correlated features
- **Priority**: High - affects accuracy of cost model for major features

## ✅ 완료된 작업
- Multi-Feature Frontier Regression Analysis 섹션 완전 제거
- Fixed rates 방식으로 전체 요금제 랭킹 테이블 계산 변경
- Plan Value Efficiency Analysis가 ranking table과 동일한 방식 사용하도록 수정
- 기능별 한계비용 계수 테이블 추가 (랭킹 테이블 위에 표시)
- **Double counting 문제 해결**: 무제한 기능의 연속값을 0으로 설정하여 이중 계산 방지
- **Unlimited type flags 구현**: 3가지 데이터 소진 후 상태를 별도 플래그로 분리
  - data_stops_after_quota (서비스 중단)
  - data_throttled_after_quota (속도 제한)
  - data_unlimited_speed (무제한 속도)

## 🎯 현재 상태
- 모든 주요 기능이 정상 작동 중
- Cross-contamination 문제 해결 완료
- Fixed rates 방식으로 순수 한계비용 기반 CS 계산 구현
- Double counting 문제 해결로 더 정확한 CS 비율 계산
- 사용자 인터페이스에서 계산 기준이 명확히 표시됨
- 11개 기능의 한계비용이 정확히 계산되고 표시됨

## 💡 제안사항 (향후 개선)
- 구글/페이스북 로그인 추가 (다음 단계)
- 추가 기능 분석 (예: 국제로밍, 부가서비스)
- 모바일 최적화 개선
- 데이터 시각화 차트 추가 개선

## 🚨 Issues to Monitor

### System Health
- 🔍 **Memory Usage**: Monitor memory consumption with full dataset processing
- 🔍 **Response Times**: Ensure API remains responsive under load
- 🔍 **Error Rates**: Track any calculation errors or edge cases
- 🔍 **Log Quality**: Ensure logging provides adequate debugging information

### Data Quality
- 🔍 **Coefficient Stability**: Monitor coefficient consistency across different datasets
- 🔍 **Outlier Detection**: Watch for data quality issues affecting calculations
- 🔍 **Edge Cases**: Handle plans with unusual feature combinations

## 📊 Success Metrics

### Performance Targets
- ✅ API response time < 500ms (excluding chart generation)
- ✅ Chart generation completion < 30 seconds
- ✅ Memory usage < 2GB for typical datasets
- ✅ Error rate < 1% for valid input data

### User Experience Goals
- ✅ Intuitive ranking system using pure coefficients
- ✅ Clean, simplified interface without overpriced plan analysis
- ✅ Accurate CS ratios reflecting true feature values
- ✅ Comprehensive dataset coverage without filtering bias

### **Voice & Message Unlimited Flag Integration**
- [x] Fixed preprocessing to include unlimited flags in regression analysis
- [x] Updated `fixed_rates` method to include voice_unlimited and message_unlimited
- [x] Verified data preprocessing creates correct unlimited flags (1,227 message_unlimited plans)
- [x] Confirmed features are included in regression analysis (16 total features)
- [x] Voice unlimited coefficient now shows ₩3,115 (working correctly)

### **System Architecture & Performance**
- [x] Async chart calculation implementation
- [x] Visual status indicators for calculation progress
- [x] Manual refresh system (no auto-polling)
- [x] Background chart processing
- [x] Immediate API response with separate chart calculation

## 💡 Future Enhancements

### **Regression Model Improvements**
- [ ] Implement cross-validation for coefficient stability
- [ ] Add model diagnostics and residual analysis
- [ ] Consider ensemble methods for robust coefficient estimation
- [ ] Add confidence intervals for coefficient estimates

### **Data Quality & Validation**
- [ ] Add automated data quality checks
- [ ] Implement outlier detection and handling
- [ ] Add feature importance analysis
- [ ] Create data drift monitoring