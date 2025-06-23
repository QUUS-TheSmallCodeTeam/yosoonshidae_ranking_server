"""
Tables Module

This module handles HTML table generation for reports.
"""

def generate_feature_rates_table_html(cost_structure):
    """
    Generate HTML table showing feature marginal cost rates with mathematical formulas.
    
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
    
    def get_mathematical_formula(feature, coefficient, cost_data=None):
        """Generate mathematical formula for each coefficient calculation"""
        coeff_val = coefficient
        
        # Extract bounds information if available
        bounds_info = ""
        if isinstance(cost_data, dict) and 'bounds' in cost_data:
            bounds = cost_data['bounds']
            lower = bounds.get('lower', 0)
            upper = bounds.get('upper', '∞')
            if upper is None:
                upper = '∞'
            bounds_info = f" subject to {lower} ≤ β ≤ {upper}"
        
        # Feature-specific mathematical formulas with Ridge regularization information
        base_formula = f"<code>min ||Xβ - y||² + α||β||²{bounds_info}</code>"
        ridge_info = "<small>Ridge 정규화: α||β||² 항으로 다중공선성 해결</small><br>"
        hessian_info = "<small>정확한 헤시안 H = 2X'X + 2αI (잘 조건화됨)</small><br>"
        
        if 'data' in feature.lower():
            unit_desc = "데이터GB"
        elif 'voice' in feature.lower():
            unit_desc = "음성분" if 'unlimited' not in feature.lower() else "무제한플래그(0/1)"
        elif 'message' in feature.lower() or 'sms' in feature.lower():
            unit_desc = "SMS건수" if 'unlimited' not in feature.lower() else "무제한플래그(0/1)"
        elif 'tethering' in feature.lower():
            unit_desc = "테더링GB"
        elif '5g' in feature.lower():
            unit_desc = "5G여부(0/1)"
        elif 'speed' in feature.lower():
            unit_desc = "속도Mbps"
        elif 'unlimited' in feature.lower() or 'throttled' in feature.lower():
            unit_desc = "처리방식플래그(0/1)"
        else:
            unit_desc = f"{feature}"
        
        coefficient_info = f"β = {coeff_val:,.4f}, 기여분 = β × {unit_desc}"
        
        return f"{base_formula}<br>{ridge_info}{hessian_info}{coefficient_info}"
    
    # Get feature costs from cost structure
    feature_costs = cost_structure.get('feature_costs', {})
    if not feature_costs:
        feature_costs = {k: v for k, v in cost_structure.items() if k != 'base_cost'}
    
    # Get multicollinearity information if available
    multicollinearity_fixes = cost_structure.get('multicollinearity_fixes', {})
    has_multicollinearity = len(multicollinearity_fixes) > 0
    
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
        <p>각 기능의 한계비용 계수와 실제 수학적 계산식입니다. 이 값들이 CS 비율 계산의 기준이 됩니다.</p>
    """
    
    # Add multicollinearity warning if detected
    if has_multicollinearity:
        html += """
        <div style="background-color: #fff3cd; border: 1px solid #ffeaa7; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="color: #856404; margin: 0 0 5px 0;">⚠️ 다중공선성 처리 적용됨</h4>
            <p style="margin: 0; font-size: 0.9em; color: #856404;">
                높은 상관관계를 가진 기능들 간에 계수 재분배가 적용되었습니다. 
                아래 표에서 원본 계수와 재분배된 계수를 확인할 수 있습니다.
            </p>
        </div>
        """
    
    # Determine table headers based on whether multicollinearity fixes exist
    if has_multicollinearity:
        table_headers = """
            <thead>
                <tr>
                    <th style="text-align: left;">기능 (Feature)</th>
                    <th style="text-align: center;">원본 계수</th>
                    <th style="text-align: center;">재분배 계수</th>
                    <th style="text-align: center;">단위 (Unit)</th>
                    <th style="text-align: left;">수학적 계산식 & 다중공선성 처리</th>
                </tr>
            </thead>
        """
    else:
        table_headers = """
            <thead>
                <tr>
                    <th style="text-align: left;">기능 (Feature)</th>
                    <th style="text-align: center;">한계비용 계수</th>
                    <th style="text-align: center;">단위 (Unit)</th>
                    <th style="text-align: left;">수학적 계산식</th>
                </tr>
            </thead>
        """
    
    html += f"""
        <table style="width: auto; margin: 10px 0;">
            {table_headers}
            <tbody>
    """
    
    # Add base cost if available
    base_cost = cost_structure.get('base_cost', 0)
    if base_cost and base_cost != 0:
        if has_multicollinearity:
            base_formula = f"<code>β₀ = {base_cost}</code><br><small>고정 기본비용</small>"
            base_description = f"""
                <div>
                    <strong style="color: #2c3e50;">수학적 계산식:</strong><br>
                    <div style="font-size: 0.9em; color: #666; margin-left: 10px;">
                        {base_formula}
                    </div>
                    <div style="margin-top: 8px; padding-top: 5px; border-top: 1px solid #eee;">
                        <small style="color: #6c757d;">기본비용은 다중공선성 영향 없음</small>
                    </div>
                </div>
            """
            html += f"""
                    <tr style="background-color: #e8f4fd;">
                        <td style="font-weight: bold;">기본 인프라 (Base Cost)</td>
                        <td style="text-align: center; font-weight: bold;">{format_coefficient(base_cost)}</td>
                        <td style="text-align: center; font-weight: bold;">{format_coefficient(base_cost)}</td>
                        <td style="text-align: center;">₩/요금제</td>
                        <td>{base_description}</td>
                    </tr>
            """
        else:
            html += f"""
                    <tr style="background-color: #e8f4fd;">
                        <td style="font-weight: bold;">기본 인프라 (Base Cost)</td>
                        <td style="text-align: center; font-weight: bold;">{format_coefficient(base_cost)}</td>
                        <td style="text-align: center;">₩/요금제</td>
                        <td><code>β₀ = {base_cost}</code><br><small>고정 기본비용</small></td>
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
        
        # Check if this feature has multicollinearity fixes
        if feature in multicollinearity_fixes:
            fix_info = multicollinearity_fixes[feature]
            original_coeff = fix_info['original_value']
            redistributed_coeff = fix_info['redistributed_value']
            paired_feature = fix_info['paired_with']
            correlation = fix_info['correlation']
            formula = fix_info['calculation_formula']
            
            # Generate mathematical formula for the feature
            base_formula = get_mathematical_formula(feature, redistributed_coeff, cost_data)
            
            # Create combined description with both mathematical formula and multicollinearity process
            combined_description = f"""
                <div style="margin-bottom: 10px;">
                    <strong style="color: #2c3e50;">수학적 계산식:</strong><br>
                    <div style="font-size: 0.9em; color: #666; margin-left: 10px;">
                        {base_formula}
                    </div>
                </div>
                <div style="border-top: 1px solid #ddd; padding-top: 8px;">
                    <strong style="color: #d63384;">다중공선성 처리:</strong><br>
                    <div style="font-size: 0.85em; margin-left: 10px;">
                        <strong>상관관계:</strong> {paired_feature} (r={correlation:.3f})<br>
                        <strong>재분배 계산:</strong> <code>{formula}</code><br>
                        <small style="color: #6c757d;">균등분배 = (원본₁ + 원본₂) ÷ 2</small>
                    </div>
                </div>
            """
            
            html += f"""
                    <tr style="background-color: #fef7e0;">
                        <td>{info['name']}</td>
                        <td style="text-align: center; color: #dc3545;">{format_coefficient(original_coeff)}</td>
                        <td style="text-align: center; color: #0d6efd; font-weight: bold;">{format_coefficient(redistributed_coeff)}</td>
                        <td style="text-align: center;">{info['unit']}</td>
                        <td>{combined_description}</td>
                    </tr>
            """
        else:
            # Generate mathematical formula
            formula = get_mathematical_formula(feature, coefficient, cost_data)
            
            if has_multicollinearity:
                # For features without multicollinearity, still show the mathematical formula
                combined_description = f"""
                    <div>
                        <strong style="color: #2c3e50;">수학적 계산식:</strong><br>
                        <div style="font-size: 0.9em; color: #666; margin-left: 10px;">
                            {formula}
                        </div>
                        <div style="margin-top: 8px; padding-top: 5px; border-top: 1px solid #eee;">
                            <small style="color: #6c757d;">다중공선성 영향 없음</small>
                        </div>
                    </div>
                """
                
                html += f"""
                        <tr>
                            <td>{info['name']}</td>
                            <td style="text-align: center;">{coeff_display}</td>
                            <td style="text-align: center;">{coeff_display}</td>
                            <td style="text-align: center;">{info['unit']}</td>
                            <td>{combined_description}</td>
                        </tr>
                """
            else:
                html += f"""
                        <tr>
                            <td>{info['name']}</td>
                            <td style="text-align: center;">{coeff_display}</td>
                            <td style="text-align: center;">{info['unit']}</td>
                            <td style="font-size: 0.9em; color: #666;">{formula}</td>
                        </tr>
                """
    
    html += """
            </tbody>
        </table>
    """
    
    # Add explanation based on whether multicollinearity was detected
    if has_multicollinearity:
        html += """
        <div style="background-color: #f8f9fa; padding: 15px; margin-top: 15px; border-radius: 5px;">
            <h4 style="margin: 0 0 10px 0; color: #495057;">📊 다중공선성 처리 상세 과정</h4>
            <ol style="margin: 0; padding-left: 20px; font-size: 0.9em; color: #495057;">
                <li><strong>Ridge 정규화</strong>: <code>min ||Xβ - y||² + α||β||²</code> (α=10.0) 적용</li>
                <li><strong>상관관계 분석</strong>: |correlation| > 0.8인 기능 쌍 감지</li>
                <li><strong>계수 재분배</strong>: 높은 상관관계 기능들의 계수를 균등 분배</li>
                <li><strong>경제적 제약</strong>: 음수 계수 및 비현실적 값 보정</li>
            </ol>
            <p style="margin: 10px 0 0 0; font-size: 0.85em; color: #6c757d;">
                <strong>재분배 공식:</strong> β₁_new = β₂_new = (β₁_original + β₂_original) ÷ 2<br>
                <strong>목적:</strong> 다중공선성으로 인한 계수 불안정성 해결 및 해석 가능성 향상
            </p>
        </div>
        """
    else:
        html += """
        <p style="font-size: 0.9em; color: #666; margin-top: 15px;">
            <strong>수식 설명:</strong> 
            <code>min ||Xβ - y||² + α||β||²</code> = Ridge 정규화된 제약 최적화로 다중공선성 문제 해결<br>
            <strong>Ridge 정규화</strong>: α||β||² 항이 계수 크기를 제한하여 안정적인 해 도출<br>
            <strong>정확한 헤시안</strong>: H = 2X'X + 2αI (잘 조건화된 행렬)<br>
            <strong>X</strong> = 기능 행렬, <strong>β</strong> = 계수 벡터, <strong>y</strong> = 실제 요금, <strong>α</strong> = 정규화 강도<br>
            각 기능의 기여분은 <strong>β × 기능값</strong>으로 계산되어 총 예상 요금에 합산됩니다.
        </p>
        """
    
    html += """
    </div>
    """
    
    return html 