"""
Tables Module

This module handles HTML table generation for reports.
"""

def generate_feature_rates_table_html(cost_structure):
    """
    Generate HTML table showing feature marginal cost rates.
    
    Args:
        cost_structure: Dictionary containing feature costs and coefficients
        
    Returns:
        str: HTML string for the feature rates table
    """
    if not cost_structure:
        return ""
    
    def format_coefficient(value):
        """Format coefficient values with appropriate precision and units"""
        if value is None:
            return "N/A"
        
        try:
            num_val = float(value)
            if abs(num_val) < 0.01:
                return f"₩{num_val:.4f}"
            elif abs(num_val) < 1:
                return f"₩{num_val:.2f}"
            elif abs(num_val) < 1000:
                return f"₩{num_val:.1f}"
            else:
                return f"₩{num_val:,.0f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Get feature costs from cost structure
    feature_costs = cost_structure.get('feature_costs', {})
    if not feature_costs:
        feature_costs = {k: v for k, v in cost_structure.items() if k != 'base_cost'}
    
    if not feature_costs:
        return ""
    
    # Feature display information
    feature_display_info = {
        'basic_data_clean': {'name': '데이터 (Data)', 'unit': '₩/GB'},
        'voice_clean': {'name': '음성통화 (Voice)', 'unit': '₩/100분'},
        'message_clean': {'name': 'SMS 문자', 'unit': '₩/100건'},
        'tethering_gb': {'name': '테더링 (Tethering)', 'unit': '₩/GB'},
        'is_5g': {'name': '5G 기술료', 'unit': '₩/요금제'},
        'additional_call': {'name': '추가 통화', 'unit': '₩/unit'},
        'speed_when_exhausted': {'name': '소진 후 속도', 'unit': '₩/Mbps'},
        'daily_data_clean': {'name': 'Daily Data', 'unit': '₩/unit'},
        'data_throttled_after_quota': {'name': '데이터 소진 후 속도제한', 'unit': '₩ (고정)'},
        'data_unlimited_speed': {'name': '데이터 무제한', 'unit': '₩ (고정)'},
        'voice_unlimited': {'name': '음성통화 무제한', 'unit': '₩ (플래그)'},
        'message_unlimited': {'name': '문자메시지 무제한', 'unit': '₩ (플래그)'}
    }
    
    html = """
    <div class="summary">
        <h3>🔢 기능별 한계비용 계수 (Feature Marginal Cost Coefficients)</h3>
        <p>각 기능의 한계비용 계수입니다. 이 값들이 CS 비율 계산의 기준이 됩니다.</p>
        <table style="width: auto; margin: 10px 0;">
            <thead>
                <tr>
                    <th style="text-align: left;">기능 (Feature)</th>
                    <th style="text-align: center;">한계비용 계수</th>
                    <th style="text-align: center;">단위 (Unit)</th>
                    <th style="text-align: left;">설명</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add base cost if available
    base_cost = cost_structure.get('base_cost', 0)
    if base_cost and base_cost != 0:
        html += f"""
                <tr style="background-color: #e8f4fd;">
                    <td style="font-weight: bold;">기본 인프라 (Base Cost)</td>
                    <td style="text-align: center; font-weight: bold;">{format_coefficient(base_cost)}</td>
                    <td style="text-align: center;">₩/요금제</td>
                    <td>네트워크 유지, 청구 시스템 등 기본 비용</td>
                </tr>
        """
    
    # Process feature costs
    for feature, cost_data in feature_costs.items():
        if feature not in feature_display_info:
            continue
            
        info = feature_display_info[feature]
        
        # Extract coefficient value
        if isinstance(cost_data, dict):
            coefficient = cost_data.get('coefficient', cost_data.get('cost_per_unit', 0))
        else:
            coefficient = cost_data
        
        coeff_display = format_coefficient(coefficient)
        
        # Generate detailed calculation description based on feature type and coefficient
        if isinstance(cost_data, dict):
            # Extract calculation details from cost_data
            calculation_method = cost_data.get('method', 'regression')
            r_squared = cost_data.get('r_squared', None)
            samples_used = cost_data.get('samples_used', None)
            
            base_description = ""
            if feature.endswith('_unlimited'):
                base_description = "무제한 기능 활성화 시 적용되는 가치"
            elif feature in ['data_throttled_after_quota', 'data_unlimited_speed']:
                base_description = "데이터 소진 후 처리 방식에 따른 고정 가치"
            elif feature == 'is_5g':
                base_description = "5G 네트워크 지원에 따른 기술 프리미엄"
            else:
                base_description = f"{info['name']} 1단위 증가 시 추가되는 한계비용"
            
            # Add calculation details
            calc_details = []
            if calculation_method:
                calc_details.append(f"방법: {calculation_method}")
            if r_squared is not None:
                calc_details.append(f"R²: {r_squared:.3f}")
            if samples_used is not None:
                calc_details.append(f"샘플수: {samples_used}")
            
            if calc_details:
                description = f"{base_description}<br><small>계산상세: {', '.join(calc_details)}</small>"
            else:
                description = base_description
        else:
            # Simple coefficient value - show calculation formula
            if feature.endswith('_unlimited'):
                description = "무제한 기능 활성화 시 적용되는 가치"
            elif feature in ['data_throttled_after_quota', 'data_unlimited_speed']:
                description = "데이터 소진 후 처리 방식에 따른 고정 가치"
            elif feature == 'is_5g':
                description = "5G 네트워크 지원에 따른 기술 프리미엄"
            else:
                description = f"{info['name']} 1단위 증가 시 추가되는 한계비용<br><small>계산: {coefficient:.4f} × 기능값 = 기여분</small>"
        
        html += f"""
                <tr>
                    <td>{info['name']}</td>
                    <td style="text-align: center;">{coeff_display}</td>
                    <td style="text-align: center;">{info['unit']}</td>
                    <td style="font-size: 0.9em; color: #666;">{description}</td>
                </tr>
        """
    
    html += """
            </tbody>
        </table>
        <p style="font-size: 0.9em; color: #666; margin-top: 15px;">
            <strong>참고:</strong> 이 계수들은 전체 데이터셋 회귀 분석을 통해 도출된 각 기능의 순수 한계비용입니다. 
            CS 비율은 이 계수들을 사용하여 계산된 기준 비용과 실제 요금제 가격의 비율입니다.
        </p>
    </div>
    """
    
    return html 