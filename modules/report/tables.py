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
        
        # Feature-specific descriptions for commonality analysis
        if 'data' in feature.lower():
            unit_desc = "데이터GB"
            feature_type = "연속형 사용량"
        elif 'voice' in feature.lower():
            unit_desc = "음성분" if 'unlimited' not in feature.lower() else "무제한플래그(0/1)"
            feature_type = "음성통화" if 'unlimited' not in feature.lower() else "무제한 서비스"
        elif 'message' in feature.lower() or 'sms' in feature.lower():
            unit_desc = "SMS건수" if 'unlimited' not in feature.lower() else "무제한플래그(0/1)"
            feature_type = "문자메시지" if 'unlimited' not in feature.lower() else "무제한 서비스"
        elif 'tethering' in feature.lower():
            unit_desc = "테더링GB"
            feature_type = "연결 서비스"
        elif '5g' in feature.lower():
            unit_desc = "5G여부(0/1)"
            feature_type = "네트워크 기술"
        elif 'speed' in feature.lower():
            unit_desc = "속도Mbps"
            feature_type = "속도 제어"
        elif 'unlimited' in feature.lower() or 'throttled' in feature.lower():
            unit_desc = "처리방식플래그(0/1)"
            feature_type = "정책 제어"
        else:
            unit_desc = f"{feature}"
            feature_type = "기타 기능"
        
        # Commonality Analysis based formula
        base_formula = f"<code>공통분산분석: R² = 고유효과 + 공통효과{bounds_info}</code>"
        commonality_info = f"<small>분산분해: 다중공선성을 고유기여분과 공통기여분으로 정량화</small><br>"
        decomposition_info = f"<small>최종계수 = 고유기여 × α + 공통기여 × β (α,β는 분배비율)</small><br>"
        
        coefficient_info = f"β = {coeff_val:,.4f}, 기여분 = β × {unit_desc}"
        feature_context = f"<small style='color: #666;'>{feature_type}: {unit_desc} 단위당 한계비용</small>"
        
        return f"{base_formula}<br>{commonality_info}{decomposition_info}{coefficient_info}<br>{feature_context}"
    
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
        'voice_clean': {'name': '음성통화 (Voice)', 'unit': '₩/분'},
        'message_clean': {'name': 'SMS 문자', 'unit': '₩/건'},
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
    
    # Add commonality analysis info if multicollinearity detected
    if has_multicollinearity:
        html += """
        <div style="background-color: #e3f2fd; border: 1px solid #90caf9; padding: 10px; margin: 10px 0; border-radius: 5px;">
            <h4 style="color: #1976d2; margin: 0 0 5px 0;">📊 공통분산분석 (Commonality Analysis) 적용됨</h4>
            <p style="margin: 0; font-size: 0.9em; color: #1976d2;">
                다중공선성을 고유효과와 공통효과로 분해하여 각 변수의 실제 기여도를 정량화했습니다. 
                아래 표에서 원본 계수와 분산분해 기반 재분배 계수를 확인할 수 있습니다.
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
                    <th style="text-align: center;">분산분해 계수</th>
                    <th style="text-align: center;">단위 (Unit)</th>
                    <th style="text-align: left;">공통분산분석 결과 & 분산분해 과정</th>
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
                    <th style="text-align: left;">공통분산분석 결과</th>
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
            base_formula = f"<code>β₀ = {base_cost}</code><br><small>기본 인프라 고정비용 (절편항)</small>"
            base_description = f"""
                <div>
                    <strong style="color: #2c3e50;">공통분산분석 결과:</strong><br>
                    <div style="font-size: 0.9em; color: #666; margin-left: 10px;">
                        {base_formula}
                    </div>
                    <div style="margin-top: 8px; padding-top: 5px; border-top: 1px solid #eee;">
                        <strong style="color: #17a2b8;">기본 인프라:</strong> 모든 요금제 공통 기본비용<br>
                        <small style="color: #6c757d;">특정 기능과 무관한 운영비용 (다중공선성 분석 대상 외)</small>
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
            
            # Extract Enhanced Commonality Analysis details
            unique_effect = fix_info.get('unique_effect', 0)
            common_effect = fix_info.get('common_effect', 0)
            redistribution_method = fix_info.get('redistribution_method', 'simple_averaging')
            variance_breakdown = fix_info.get('variance_breakdown', '')
            method_type = fix_info.get('method', 'simple_averaging')
            
            # Create enhanced description with new commonality analysis information
            if method_type == 'enhanced_commonality_analysis':
                combined_description = f"""
                    <div style="margin-bottom: 10px;">
                        <strong style="color: #2c3e50;">🔬 Enhanced Commonality Analysis 결과:</strong><br>
                        <div style="font-size: 0.9em; color: #666; margin-left: 10px;">
                            {base_formula}
                        </div>
                    </div>
                    <div style="border-top: 1px solid #ddd; padding-top: 8px;">
                        <strong style="color: #d63384;">분산분해 상세 정보:</strong><br>
                                                 <div style="font-size: 0.85em; margin-left: 10px;">
                             <strong>상관변수:</strong> {paired_feature} (r={correlation:.3f})<br>
                             <strong>분산분해:</strong> <code style="background-color: #fff3cd; padding: 2px;">{variance_breakdown}</code><br>
                             <strong>재분배 방법:</strong> <code style="background-color: #d1ecf1; padding: 2px;">{redistribution_method}</code><br>
                             <strong>계산공식:</strong> <code style="background-color: #f8d7da; padding: 2px;">{formula}</code><br>
                             <small style="color: #6c757d;">
                                 R²(Total) = R²(고유효과) + R²(공통효과)<br>
                                 🔬 고유효과: <span style="color: #0066cc; font-weight: bold;">{unique_effect:.4f}</span> | 공통효과: <span style="color: #cc6600; font-weight: bold;">{common_effect:.4f}</span>
                             </small>
                         </div>
                    </div>
                """
            else:
                # Fallback to original description for simple averaging
                combined_description = f"""
                    <div style="margin-bottom: 10px;">
                        <strong style="color: #2c3e50;">공통분산분석 결과:</strong><br>
                        <div style="font-size: 0.9em; color: #666; margin-left: 10px;">
                            {base_formula}
                        </div>
                    </div>
                    <div style="border-top: 1px solid #ddd; padding-top: 8px;">
                        <strong style="color: #d63384;">다중공선성 분해:</strong><br>
                        <div style="font-size: 0.85em; margin-left: 10px;">
                            <strong>상관변수:</strong> {paired_feature} (r={correlation:.3f})<br>
                            <strong>공통분산 처리:</strong> <code>{formula}</code><br>
                            <strong>분산분해 원리:</strong> 겹치는 설명력을 두 변수가 균등분배<br>
                            <small style="color: #6c757d;">
                                R²({feature} + {paired_feature}) = R²({feature} 고유) + R²({paired_feature} 고유) + R²(공통)
                            </small>
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
                        <strong style="color: #2c3e50;">공통분산분석 결과:</strong><br>
                        <div style="font-size: 0.9em; color: #666; margin-left: 10px;">
                            {formula}
                        </div>
                        <div style="margin-top: 8px; padding-top: 5px; border-top: 1px solid #eee;">
                            <strong style="color: #28a745;">독립적 기여:</strong> 다른 변수와 공통분산 없음<br>
                            <small style="color: #6c757d;">R²(총) = R²(고유효과) (공통효과 = 0)</small>
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
            <h4 style="margin: 0 0 10px 0; color: #495057;">🔬 Enhanced Commonality Analysis 상세 과정</h4>
            <ol style="margin: 0; padding-left: 20px; font-size: 0.9em; color: #495057;">
                <li><strong>All Possible Subsets Regression</strong>: 2^n개 조합에서 모든 R² 계산</li>
                <li><strong>완전한 분산분해</strong>: R² = Σ(고유효과) + Σ(공통효과)</li>
                <li><strong>지능적 재분배</strong>: 경제적 제약조건과 Commonality 결과의 블렌딩</li>
                <li><strong>투명한 분산분해</strong>: 각 변수의 고유/공통 기여도 정량화</li>
            </ol>
            <div style="margin: 15px 0; padding: 10px; background-color: #e3f2fd; border-radius: 3px;">
                <strong style="color: #1976d2;">🧠 지능적 재분배 로직:</strong><br>
                <code style="background-color: #fff; padding: 2px 4px; display: block; margin: 5px 0;">
                    if commonality_coeff < min_bound:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;final_coeff = 0.7 × min_bound + 0.3 × original_coeff<br>
                    elif commonality_coeff > max_bound:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;final_coeff = 0.7 × max_bound + 0.3 × original_coeff<br>
                    else:<br>
                    &nbsp;&nbsp;&nbsp;&nbsp;final_coeff = commonality_coeff
                </code><br>
                <small style="color: #1565c0;">
                    극단적 결과는 원본 계수와 블렌딩하여 경제적 타당성과 통계적 정확성 양립
                </small>
            </div>
            <p style="margin: 10px 0 0 0; font-size: 0.85em; color: #6c757d;">
                <strong>✅ 핵심 개선사항:</strong> ① 의미있는 분산분해 + 지능적 재분배 ② 경제적 타당성 보장<br>
                <strong>📊 결과 투명성:</strong> 고유효과(%), 공통효과(%), 재분배 방법, 최종 계수까지 완전 공개<br>
                <strong>🎯 목표 달성:</strong> 단순 보존이 아닌 실제 분산분해 결과를 활용한 의미있는 계수 조정
            </p>
        </div>
        """
    else:
        html += """
        <p style="font-size: 0.9em; color: #666; margin-top: 15px;">
            <strong>공통분산분석 (Commonality Analysis):</strong> 
            각 feature의 고유효과와 공통효과를 분리하여 다중공선성을 정량화<br>
            <strong>고유효과</strong>: 다른 변수와 독립적인 설명력, <strong>공통효과</strong>: 다른 변수와 공유하는 설명력<br>
            <strong>분산분해 공식</strong>: R² = Σ(고유효과) + Σ(2변수 공통효과) + Σ(3변수 공통효과) + ...<br>
            <strong>최종 계수</strong>: 고유기여분과 공통기여분을 합리적으로 분배하여 경제적 해석력 확보<br>
            현재 데이터에서는 모든 변수가 독립적으로 작동하여 공통효과가 미미합니다.
        </p>
        """
    
    html += """
    </div>
    """
    
    return html 