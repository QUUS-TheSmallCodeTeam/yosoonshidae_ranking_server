"""
Report HTML Module

This module handles the main HTML report generation.
"""

import logging
import json
import pandas as pd
from datetime import datetime
from .report_utils import NumpyEncoder, FEATURE_DISPLAY_NAMES, FEATURE_UNITS, UNLIMITED_FLAGS
from .report_charts import prepare_feature_frontier_data, prepare_residual_analysis_data, prepare_granular_marginal_cost_frontier_data
from .report_tables import generate_all_plans_table_html, generate_residual_analysis_table_html

# Configure logging
logger = logging.getLogger(__name__)

def prepare_cost_structure_chart_data(cost_structure):
    """
    Prepare chart data for cost structure visualization.
    
    Args:
        cost_structure: Dictionary with cost structure from linear decomposition
        
    Returns:
        Dictionary with chart data for JavaScript rendering
    """
    if not cost_structure:
        return None
    
    # Chart 1: Overall cost breakdown (pie chart)
    overall_data = {
        'labels': [],
        'data': [],
        'colors': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    }
    
    # Chart 2: Per-unit costs (bar chart)
    unit_cost_data = {
        'labels': [],
        'data': [],
        'units': [],
        'colors': ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    }
    
    # Base cost (infrastructure)
    base_cost = cost_structure.get('base_cost', 0)
    if base_cost > 0:
        overall_data['labels'].append('기본 인프라 (Base Infrastructure)')
        overall_data['data'].append(base_cost)
    
    # Feature costs with Korean labels and units
    feature_labels = {
        'basic_data_clean': {'label': '데이터 비용 (Data Cost)', 'unit': 'per GB'},
        'voice_clean': {'label': '음성 비용 (Voice Cost)', 'unit': 'per 100min'},
        'message_clean': {'label': 'SMS 비용 (SMS Cost)', 'unit': 'per 100msg'},
        'tethering_gb': {'label': '테더링 프리미엄 (Tethering Premium)', 'unit': 'per GB'},
        'is_5g': {'label': '5G 기술료 (5G Technology)', 'unit': 'per plan'}
    }
    
    # Handle both flat and nested cost structure formats
    feature_costs = cost_structure.get('feature_costs', {})
    if not feature_costs:
        # Fallback to flat structure (direct keys excluding base_cost)
        feature_costs = {k: v for k, v in cost_structure.items() if k != 'base_cost'}
    
    for feature, cost in feature_costs.items():
        if feature in feature_labels:
            info = feature_labels[feature]
            
            # Extract numeric value from cost (handle both dict and numeric formats)
            if isinstance(cost, dict):
                # Multi-frontier method returns nested structure
                cost_value = cost.get('coefficient', cost.get('cost_per_unit', 0))
            else:
                # Simple numeric value
                cost_value = cost
            
            # For overall breakdown - use normalized values for comparison
            overall_data['labels'].append(info['label'])
            overall_data['data'].append(abs(cost_value))  # Use absolute value for visualization
            
            # For unit costs - only meaningful marginal costs
            if cost_value > 0:  # Only positive marginal costs
                unit_cost_data['labels'].append(info['label'])
                unit_cost_data['data'].append(cost_value)
                unit_cost_data['units'].append(info['unit'])
    
    # Chart 3: Detailed marginal cost analysis with business interpretation
    marginal_analysis = {
        'features': [],
        'coefficients': [],
        'interpretations': [],
        'colors': ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e'],
        'base_cost': base_cost
    }
    
    # Feature analysis with business context
    feature_analysis = {
        'basic_data_clean': {
            'name': '데이터 (Data)', 
            'unit': '₩/GB',
            'interpretation': lambda x: f"데이터 1GB 추가시 ₩{x:.0f} 비용 증가"
        },
        'voice_clean': {
            'name': '음성통화 (Voice)', 
            'unit': '₩/100분',
            'interpretation': lambda x: f"음성 100분 추가시 ₩{x:.0f} 비용 증가"
        },
        'message_clean': {
            'name': 'SMS 문자', 
            'unit': '₩/100건',
            'interpretation': lambda x: f"SMS 100건 추가시 ₩{x:.0f} 비용 증가"
        },
        'tethering_gb': {
            'name': '테더링 (Tethering)', 
            'unit': '₩/GB',
            'interpretation': lambda x: f"테더링 1GB 추가시 ₩{x:.0f} 비용 증가"
        },
        'is_5g': {
            'name': '5G 기술료', 
            'unit': '₩/요금제',
            'interpretation': lambda x: f"5G 지원시 ₩{x:.0f} 추가 비용"
        }
    }
    
    # Use the same feature_costs handling for marginal analysis
    for feature, cost in feature_costs.items():
        if feature in feature_analysis:
            analysis = feature_analysis[feature]
            
            # Extract numeric value from cost (handle both dict and numeric formats)
            if isinstance(cost, dict):
                # Multi-frontier method returns nested structure
                cost_value = cost.get('coefficient', cost.get('cost_per_unit', 0))
            else:
                # Simple numeric value
                cost_value = cost
            
            marginal_analysis['features'].append(analysis['name'])
            marginal_analysis['coefficients'].append(cost_value)
            marginal_analysis['interpretations'].append(analysis['interpretation'](cost_value))
    
    return {
        'overall': overall_data,
        'unit_costs': unit_cost_data,
        'marginal_analysis': marginal_analysis,
        'feature_costs': feature_costs,  # Add raw feature costs for marginal cost charts
        'base_cost': base_cost           # Add base cost for marginal cost charts
    }

def prepare_plan_efficiency_data(df, method):
    """
    Prepare data for Plan Value Efficiency Matrix visualization.
    
    Args:
        df: DataFrame with plan data including CS ratios and baselines
        method: Method used ('linear_decomposition', 'frontier', 'fixed_rates', or 'multi_frontier')
        
    Returns:
        Dictionary with chart data for JavaScript rendering
    """
    if df.empty:
        return None
    
    efficiency_data = {
        'plans': [],
        'diagonal': {'min': 0, 'max': 0}
    }
    
    # Get baseline and actual cost columns based on method
    if method == 'linear_decomposition' and 'B_decomposed' in df.columns:
        baseline_col = 'B_decomposed'
        cs_col = 'CS_decomposed'
    else:
        # For frontier, fixed_rates, multi_frontier methods, use standard columns
        baseline_col = 'B'
        cs_col = 'CS'
    
    actual_col = 'fee'
    
    if baseline_col not in df.columns or actual_col not in df.columns or cs_col not in df.columns:
        return None
    
    # Prepare plan data
    for _, row in df.iterrows():
        baseline = row[baseline_col] if pd.notna(row[baseline_col]) else 0
        actual = row[actual_col] if pd.notna(row[actual_col]) else 0
        cs_ratio = row[cs_col] if pd.notna(row[cs_col]) else 0
        
        # Calculate total features for bubble size
        feature_total = 0
        feature_cols = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
        for col in feature_cols:
            if col in row and pd.notna(row[col]):
                feature_total += row[col]
        
        plan_data = {
            'baseline': baseline,
            'actual': actual,
            'cs_ratio': cs_ratio,
            'plan_name': row.get('plan_name', 'Unknown'),
            'mvno': row.get('mvno', 'Unknown'),
            'feature_total': feature_total,
            'is_good_value': cs_ratio > 1.0
        }
        
        efficiency_data['plans'].append(plan_data)
    
    # Calculate diagonal line range
    all_costs = []
    for plan in efficiency_data['plans']:
        all_costs.extend([plan['baseline'], plan['actual']])
    
    if all_costs:
        efficiency_data['diagonal']['min'] = min(all_costs)
        efficiency_data['diagonal']['max'] = max(all_costs)
    
    return efficiency_data

def generate_feature_rates_table_html(cost_structure):
    """
    Generate HTML table showing feature rates/coefficients used in CS calculations.
    
    Args:
        cost_structure: Dictionary containing feature_costs and other cost data
        
    Returns:
        HTML string for the feature rates table
    """
    if not cost_structure or not cost_structure.get('feature_costs'):
        return ""
    
    feature_costs = cost_structure.get('feature_costs', {})
    
    # Handle different feature_costs structures
    if isinstance(feature_costs, list):
        # Convert list format to dict format
        features_data = {
            item['feature']: {
                'coefficient': item.get('coefficient', 0),
                'display_name': item.get('display_name', item['feature']),
                'unit': item.get('unit', '')
            }
            for item in feature_costs
        }
    elif isinstance(feature_costs, dict):
        # Check if feature_costs has nested structure
        if feature_costs and isinstance(list(feature_costs.values())[0], dict):
            features_data = feature_costs
        else:
            # Flat structure - convert to nested
            features_data = {
                feature: {
                    'coefficient': coeff,
                    'display_name': feature.replace('_clean', '').replace('_', ' ').title(),
                    'unit': 'KRW/unit'
                }
                for feature, coeff in feature_costs.items()
            }
    else:
        return ""
    
    # Create table HTML
    table_html = """
    <div class="metrics">
        <h3>기능별 한계비용 계수 (Feature Marginal Cost Coefficients)</h3>
        <p>아래 표는 CS 비율 계산에 사용되는 각 기능의 한계비용을 보여줍니다.</p>
        <table style="width: 100%; max-width: 800px; margin: 0 auto;">
            <thead>
                <tr>
                    <th>기능 (Feature)</th>
                    <th>한계비용 (Marginal Cost)</th>
                    <th>단위 (Unit)</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Feature display names mapping
    feature_names = {
        'basic_data_clean': '데이터 (Data)',
        'voice_clean': '음성통화 (Voice)',
        'message_clean': '문자메시지 (Messages)',
        'tethering_gb': '테더링 (Tethering)',
        'is_5g': '5G 지원 (5G Support)',
        'data_stops_after_quota': '데이터 소진 후 중단 (Data Stops)',
        'data_throttled_after_quota': '데이터 소진 후 속도제한 (Data Throttled)',
        'data_unlimited_speed': '데이터 무제한 (Data Unlimited)',
        'basic_data_unlimited': '기본 데이터 무제한 (Basic Data Unlimited)',
        'voice_unlimited': '음성 무제한 (Voice Unlimited)',
        'message_unlimited': '문자 무제한 (Message Unlimited)',
        'has_throttled_data': '속도제한 데이터 (Throttled Data)',
        'additional_call': '추가 통화 (Additional Call)',
        'speed_when_exhausted': '소진 후 속도 (Speed When Exhausted)'
    }
    
    # Feature units mapping
    feature_units = {
        'basic_data_clean': 'KRW/GB',
        'voice_clean': 'KRW/분',
        'message_clean': 'KRW/건',
        'tethering_gb': 'KRW/GB',
        'is_5g': 'KRW (고정)',
        'data_stops_after_quota': 'KRW (기준)',
        'data_throttled_after_quota': 'KRW (고정)',
        'data_unlimited_speed': 'KRW (고정)',
        'basic_data_unlimited': 'KRW (고정)',
        'voice_unlimited': 'KRW (고정)',
        'message_unlimited': 'KRW (고정)',
        'has_throttled_data': 'KRW (고정)',
        'additional_call': 'KRW/unit',
        'speed_when_exhausted': 'KRW/Mbps'
    }
    
    # Sort features by coefficient value (highest first)
    sorted_features = sorted(features_data.items(), 
                           key=lambda x: x[1].get('coefficient', 0), 
                           reverse=True)
    
    for feature, data in sorted_features:
        coefficient = data.get('coefficient', 0)
        display_name = feature_names.get(feature, feature.replace('_clean', '').replace('_', ' ').title())
        unit = feature_units.get(feature, 'KRW/unit')
        
        # Format coefficient with proper number formatting
        if coefficient >= 1000:
            coeff_str = f"₩{coefficient:,.0f}"
        elif coefficient >= 1:
            coeff_str = f"₩{coefficient:.2f}"
        else:
            coeff_str = f"₩{coefficient:.4f}"
        
        table_html += f"""
                <tr>
                    <td style="text-align: left; font-weight: bold;">{display_name}</td>
                    <td style="text-align: right; font-family: monospace;">{coeff_str}</td>
                    <td style="text-align: center;">{unit}</td>
                </tr>
        """
    
    table_html += """
            </tbody>
        </table>
        <p style="font-size: 0.9em; color: #666; margin-top: 10px;">
            * 이 계수들은 전체 데이터셋에서 추출된 순수 한계비용으로, cross-contamination이 제거된 값입니다.<br>
            * CS 비율 = 기준비용(이 계수들로 계산) / 실제 요금
        </p>
    </div>
    """
    
    return table_html

def generate_html_report(df, timestamp=None, report_title="Mobile Plan Rankings", is_cs=True, title=None, method=None, cost_structure=None):
    """
    Generate a full HTML report with plan rankings and feature frontier charts.
    
    Args:
        df: DataFrame with ranking data
        timestamp: Timestamp for the report (defaults to current time)
        report_title: Title for the report
        is_cs: Whether this is a Cost-Spec report (for backward compatibility)
        title: Alternative title (for backward compatibility)
        method: Cost-Spec method used ('linear_decomposition' or 'frontier')
        cost_structure: Cost structure dictionary from linear decomposition
        
    Returns:
        HTML string for the complete report
    """
    # Use title parameter if provided (for backward compatibility)
    if title:
        report_title = title
        
    # Add method information to title if available
    if method:
        method_name = "Linear Decomposition" if method == "linear_decomposition" else "Frontier-Based"
        report_title = f"{report_title} ({method_name})"
        
    # Info logging disabled for frequent polling requests
    # logger.info(f"Generating HTML report with title: {report_title}")
    
    # Set timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Sort DataFrame by rank (CS ratio)
    df_sorted = df.copy()
    if 'CS' in df_sorted.columns:
        df_sorted = df_sorted.sort_values(by='CS', ascending=False)
    
    # Add method and cost structure information to summary
    method_info_html = ""
    if method:
        if method == "linear_decomposition":
            method_info_html = f"""
            <div class="summary">
                <h3>🔬 Linear Decomposition Method</h3>
                <p>This report uses the advanced linear decomposition method to extract true marginal costs for individual features, 
                   eliminating double-counting artifacts present in traditional frontier-based approaches.</p>
                <ul>
                    <li><strong>Mathematical Model:</strong> plan_cost = β₀ + β₁×data + β₂×voice + β₃×SMS + β₄×tethering + β₅×5G</li>
                    <li><strong>Advantages:</strong> Fair baselines, realistic CS ratios (0.8-1.5x), true cost structure discovery</li>
                    <li><strong>Baseline Calculation:</strong> Uses decomposed marginal costs instead of summing complete plan costs</li>
                </ul>
            """
            
            if cost_structure:
                method_info_html += """
                <h4>📊 Discovered Cost Structure</h4>
                <table style="width: auto; margin: 10px 0;">
                    <tr><th>Component</th><th>Cost</th><th>Business Interpretation</th></tr>
                """
                
                # Base cost
                base_cost = cost_structure.get('base_cost', 0)
                method_info_html += f"<tr><td>Base Infrastructure</td><td>₩{base_cost:,.0f}</td><td>Network maintenance, billing systems</td></tr>"
                
                # Feature costs
                feature_interpretations = {
                    'basic_data_clean': 'Per-GB data cost (spectrum, backhaul)',
                    'voice_clean': 'Per-100min voice cost (switching, interconnect)',
                    'message_clean': 'Per-100SMS cost (usually bundled)',
                    'tethering_gb': 'Per-GB tethering premium',
                    'is_5g': '5G technology premium'
                }
                
                # Handle feature costs from nested structure
                feature_costs = cost_structure.get('feature_costs', {})
                if not feature_costs:
                    # Fallback: iterate through direct keys if feature_costs not found
                    feature_costs = {k: v for k, v in cost_structure.items() if k != 'base_cost'}
                
                for feature, cost in feature_costs.items():
                    if feature in feature_interpretations:
                        interpretation = feature_interpretations.get(feature, 'Feature-specific cost')
                        # Handle both simple float and nested dictionary cost structures
                        try:
                            if isinstance(cost, dict):
                                # Multi-frontier method returns nested structure
                                coefficient = cost.get('coefficient', cost.get('cost_per_unit', 0))
                                cost_str = f"₩{float(coefficient):.2f}"
                            else:
                                # Simple float value
                                cost_str = f"₩{float(cost):.2f}"
                        except (ValueError, TypeError):
                            cost_str = str(cost)
                        method_info_html += f"<tr><td>{feature}</td><td>{cost_str}</td><td>{interpretation}</td></tr>"
                
                method_info_html += "</table>"
            
            method_info_html += "</div>"
            
        else:  # frontier method
            method_info_html = f"""
            <div class="summary">
                <h3>📈 Frontier-Based Method</h3>
                <p>This report uses the traditional frontier-based method that identifies minimum costs for each feature level 
                   and sums them to create plan baselines.</p>
                <ul>
                    <li><strong>Approach:</strong> Find minimum cost plans for each feature level</li>
                    <li><strong>Baseline Calculation:</strong> Sum frontier costs for all features</li>
                    <li><strong>Note:</strong> May exhibit double-counting effects in CS ratios</li>
                </ul>
            </div>
            """
    
    # Check for comparison data (both methods included)
    comparison_info_html = ""
    if 'CS_frontier' in df_sorted.columns and 'CS' in df_sorted.columns:
        # Calculate correlation between methods
        correlation = df_sorted['CS'].corr(df_sorted['CS_frontier'])
        comparison_info_html = f"""
        <div class="note">
            <h4>📊 Method Comparison</h4>
            <p>This report includes both linear decomposition and frontier-based results for comparison.</p>
            <ul>
                <li><strong>Correlation between methods:</strong> {correlation:.3f}</li>
                <li><strong>Linear Decomposition CS:</strong> Primary ranking (realistic baselines)</li>
                <li><strong>Frontier CS:</strong> Comparison data (may show artifacts)</li>
            </ul>
        </div>
        """
    
    # Generate advanced analysis charts (show both methods when data is available)
    advanced_analysis_chart_html = ""
    advanced_analysis_json = "null"  # Default value
    
    # Check if we have multi-frontier breakdown data
    multi_frontier_breakdown = getattr(df, 'attrs', {}).get('multi_frontier_breakdown')
    linear_decomp_cost_structure = getattr(df, 'attrs', {}).get('cost_structure')
    
    # Multi-Feature Frontier Regression Analysis section removed per user request
    
    # Linear decomposition analysis removed per user request
    linear_decomp_json = "null"
    
    # Import CORE_FEATURES from cost_spec to use all 14 features
    from modules.cost_spec import CORE_FEATURES
    
    # Use all features from FEATURE_SETS['basic'] for comprehensive analysis
    core_continuous_features = CORE_FEATURES
    
    # Prepare data for feature frontier charts
    feature_frontier_data, all_chart_data, visual_frontiers_for_residual_table = prepare_feature_frontier_data(df, core_continuous_features)
    
    # Prepare marginal cost frontier charts (using pure coefficients from multi-frontier regression)
    marginal_cost_frontier_data = {}
    if cost_structure and cost_structure.get('feature_costs'):
        # Create a mock multi_frontier_breakdown from cost_structure for compatibility
        # Handle both list and dict feature_costs structures
        feature_costs_raw = cost_structure.get('feature_costs', {})
        
        # Convert feature_costs to dictionary format expected by prepare_marginal_cost_frontier_data
        if isinstance(feature_costs_raw, list):
            # Convert list format to dict format
            simplified_feature_costs = {
                item['feature']: {
                    'coefficient': item.get('coefficient', 0),
                    'display_name': item.get('display_name', item['feature']),
                    'unit': item.get('unit', '')
                }
                for item in feature_costs_raw
            }
        elif isinstance(feature_costs_raw, dict):
            # Check if feature_costs has nested structure (from multi-frontier method)
            if feature_costs_raw and isinstance(list(feature_costs_raw.values())[0], dict):
                # Extract coefficients from nested structure
                simplified_feature_costs = feature_costs_raw
            else:
                # Already flat structure (from linear decomposition)
                simplified_feature_costs = {
                    feature: {'coefficient': coeff}
                    for feature, coeff in feature_costs_raw.items()
                }
        else:
            simplified_feature_costs = {}
            
        mock_breakdown = {
            'feature_costs': simplified_feature_costs,
            'base_cost': cost_structure.get('base_cost', 0)
        }
        marginal_cost_frontier_data = prepare_granular_marginal_cost_frontier_data(df, mock_breakdown, core_continuous_features)
    
    # Convert to JSON for JavaScript
    feature_frontier_json = json.dumps(feature_frontier_data, cls=NumpyEncoder)
    marginal_cost_frontier_json = json.dumps(marginal_cost_frontier_data, cls=NumpyEncoder)
    
    # Generate feature rates table HTML
    feature_rates_table_html = generate_feature_rates_table_html(cost_structure)
    
    # Generate table HTML
    all_plans_html = generate_all_plans_table_html(df_sorted)
    
    # Prepare Plan Value Efficiency Matrix data
    plan_efficiency_data = prepare_plan_efficiency_data(df_sorted, method)
    plan_efficiency_json = json.dumps(plan_efficiency_data, cls=NumpyEncoder)
    
    # Main HTML template  
    html_template = """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.9em;
                text-align: center;
            }
            table, th, td {
                border: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                color: #333;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
                text-align: center;
            }
            tr:nth-child(even) {
                background-color: #f9f9f9;
            }
            tr:hover {
                background-color: #f1f1f1;
            }
            th, td {
                padding: 8px 12px;
                overflow-wrap: break-word;
                word-break: break-all;
                text-align: center;
            }
            .highlight-high {
                color: #27ae60;
                font-weight: bold;
            }
            .highlight-low {
                color: #e74c3c;
                font-weight: bold;
            }
            .good-value {
                color: #27ae60;
                font-weight: bold;
            }
            .bad-value {
                color: #e74c3c;
                font-weight: bold;
            }
            .metric-good {
                color: #27ae60;
            }
            .metric-average {
                color: #f39c12;
            }
            .metric-poor {
                color: #e74c3c;
            }
            .container {
                max-width: 100%;
                margin: 0 auto;
            }
            .summary {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .metrics {
                background-color: #eaf7fd;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }
            .note {
                background-color: #f8f9fa;
                padding: 10px;
                border-left: 4px solid #007bff;
                margin-bottom: 20px;
            }
            
            /* Content wrapper with padding */
            .content-wrapper {
                padding: 20px;
            }
            
            /* Feature charts wrapper - no padding for full width */
            .charts-wrapper {
                width: 100%;
            }
            
            /* Feature charts grid */
            .chart-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                width: 100%;
            }
            
            .chart-container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 15px;
                position: relative;
                width: 100%;
                height: 400px;
            }
            
            @media print {
                body {
                    font-size: 10pt;
                }
                table {
                    font-size: 9pt;
                }
                .no-print {
                    display: none;
                }
            }
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>모바일 요금제 랭킹</h1>
            <h2>Cost-Spec Ratio 모델</h2>
            <p>생성일: {timestamp_str}</p>
            
            <div class="summary">
                <h2>요약 통계</h2>
                <ul>
                    <li>분석된 요금제 수: <strong>{len_df_sorted:,}개</strong></li>
                    <li>평균 CS 비율: <strong>{avg_cs:.2f}배</strong></li>
                    <li>고평가 요금제 (CS ≥ 1): <strong>{high_cs_count:,}개</strong> ({high_cs_pct:.1%})</li>
                    <li>저평가 요금제 (CS < 1): <strong>{low_cs_count:,}개</strong> ({low_cs_pct:.1%})</li>
                </ul>
            </div>
            
            {method_info_html}
            {comparison_info_html}
        
            <!-- Multi-Frontier Analysis Charts -->
            {multi_frontier_chart_html}
            
            <!-- Linear Decomposition Analysis removed per user request -->
        
            <div class="note">
                <p>이 보고서는 Cost-Spec Ratio 방법론을 기반으로 한 모바일 플랜 랭킹을 보여줍니다. CS 비율이 높을수록 사양 대비 더 좋은 가치를 제공합니다.</p>
                <p>모든 비용은 한국 원화(KRW)로 표시됩니다.</p>
            </div>

            <!-- Feature Frontier Charts -->
            <div class="charts-wrapper">
                <h2>Feature Frontier Charts</h2>
                <div class="note">
                    <p>이 차트는 각 기능에 대한 비용 프론티어를 보여줍니다. 프론티어에 있는 플랜은 다양한 수준에서 해당 기능에 대한 최상의 가치를 제공합니다.</p>
                </div>
                <div id="featureCharts" class="chart-grid"></div>
            </div>
            
            <!-- Marginal Cost Frontier Charts -->
            <div class="charts-wrapper">
                <h2>📈 Marginal Cost Frontier Analysis</h2>
                <div class="note">
                    <p><strong>Pure Marginal Cost Trends:</strong> 이 차트는 Multi-Feature Frontier Regression에서 추출된 순수 한계비용을 사용하여 각 기능의 비용 트렌드를 보여줍니다.</p>
                    <p><strong>핵심 개선사항:</strong> 기존 프론티어 차트의 교차 오염 문제를 해결하여, 각 기능의 실제 가치만을 반영한 순수 한계비용을 시각화합니다.</p>
                    <p><strong>해석:</strong> 파란색 선은 순수 한계비용 트렌드, 빨간색 점은 실제 시장 요금제와의 비교를 보여줍니다.</p>
                </div>
                <div id="marginalCostFrontierCharts" class="chart-grid"></div>
            </div>
            
            <!-- Plan Value Efficiency Matrix -->
            <div class="charts-wrapper">
                <h2>💰 Plan Value Efficiency Analysis</h2>
                <div class="note">
                    <p>이 차트는 각 요금제의 실제 비용 대비 계산된 기준 비용을 보여줍니다. 대각선 아래(녹색 영역)는 가성비가 좋은 요금제, 위(빨간색 영역)는 과가격 요금제입니다.</p>
                </div>
                <div class="chart-container" style="width: 100%; height: 600px;">
                    <canvas id="planEfficiencyChart"></canvas>
                </div>
                <p style="text-align: center; margin-top: 10px; color: #666; font-size: 0.9em;">
                    🟢 녹색 = 가성비 좋은 요금제 (CS > 1.0) | 🔴 빨간색 = 과가격 요금제 (CS < 1.0)<br>
                    대각선 = 완벽한 효율성 기준선 | 버블 크기 = 총 기능 수준
                </p>
            </div>

            {feature_rates_table_html}
            
            <h2>전체 요금제 랭킹</h2>
            {all_plans_html}
        </div>

        <!-- Add Chart.js implementation -->
        <script>
            // Feature frontier data from Python
            const featureFrontierData = __FEATURE_FRONTIER_JSON__;
            
            // Marginal cost frontier data from Python (pure coefficients)
            const marginalCostFrontierData = __MARGINAL_COST_FRONTIER_JSON__;
            
            // Cost structure data from Python (multi-frontier method)
            const advancedAnalysisData = __ADVANCED_ANALYSIS_JSON__;
            
            // Linear decomposition data removed per user request
            const linearDecompData = null;
            
            // Plan efficiency data from Python
            const planEfficiencyData = __PLAN_EFFICIENCY_JSON__;
            
            // Chart color scheme
            const chartColors = {
                frontier: 'rgba(54, 162, 235, 1)',      // Blue for frontier
                frontierFill: 'rgba(54, 162, 235, 0.2)', // Light blue fill
                unlimited: 'rgba(255, 159, 64, 1)',      // Orange for unlimited
                excluded: 'rgba(255, 99, 132, 0.6)',     // Red for excluded
                otherPoints: 'rgba(201, 203, 207, 0.6)'  // Gray for other
            };
            
            // Create charts for each feature
            document.addEventListener('DOMContentLoaded', () => {
                const chartsContainer = document.getElementById('featureCharts');
                
                // Track created charts for potential later use (e.g., responsiveness)
                const charts = [];
                
                // For each feature in the data, create a chart
                for (const [feature, data] of Object.entries(featureFrontierData)) {
                    // Create chart container
                    const chartContainer = document.createElement('div');
                    chartContainer.className = 'chart-container';
                    chartContainer.style.width = '100%';  // Full viewport width
                    chartContainer.style.maxWidth = '100%';  // Prevent horizontal overflow
                    chartContainer.style.margin = '0 0 20px 0';  // Add bottom margin
                    chartContainer.style.padding = '15px';  // Small padding inside container
                    chartContainer.style.boxSizing = 'border-box'; // Include padding in width
                    chartContainer.style.height = '500px';  // Taller charts
                    
                    // Create feature title (h3)
                    const title = document.createElement('h3');
                    title.textContent = feature.replace('_clean', '').replace('_', ' ') + ' Frontier';
                    title.style.marginTop = '0';
                    title.style.textAlign = 'center';
                    chartContainer.appendChild(title);
                    
                    // Create canvas for Chart.js
                    const canvas = document.createElement('canvas');
                    chartContainer.appendChild(canvas);
                    chartsContainer.appendChild(chartContainer);
                    
                    // Prepare data for Chart.js
                    // Create datasets for frontier points (line) and excluded points (scatter)
                    const frontierDataset = {
                        label: 'Cost Frontier',
                        data: data.frontier_values.map((val, i) => ({
                            x: val,
                            y: data.frontier_contributions[i],
                            plan: data.frontier_plan_names[i]
                        })),
                        borderColor: chartColors.frontier,
                        backgroundColor: chartColors.frontierFill,
                        pointBackgroundColor: chartColors.frontier,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        showLine: true
                    };
                    
                    const excludedDataset = {
                        label: 'Excluded Plans',
                        data: data.excluded_values.map((val, i) => ({
                            x: val,
                            y: data.excluded_contributions[i],
                            plan: data.excluded_plan_names[i]
                        })),
                        backgroundColor: chartColors.excluded,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        showLine: false
                    };
                    
                    // Create a dataset for unlimited point if present
                    let unlimitedDataset = null;
                    if (data.has_unlimited) {
                        unlimitedDataset = {
                            label: 'Unlimited Plan',
                            data: [{
                                x: null, // Will be rendered on right edge
                                y: data.unlimited_value,
                                plan: data.unlimited_plan
                            }],
                            backgroundColor: chartColors.unlimited,
                            pointRadius: 7,
                            pointHoverRadius: 9,
                            pointStyle: 'triangle',
                            rotation: 90,
                            showLine: false
                        };
                    }
                    
                    // Combine datasets
                    const datasets = [frontierDataset, excludedDataset];
                    if (unlimitedDataset) datasets.push(unlimitedDataset);
                    
                    // Create Chart.js chart
                    const chartConfig = {
                        type: 'scatter',
                        data: {
                            datasets: datasets
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: feature.includes('data') ? 'GB' : 
                                              feature.includes('voice') ? 'Minutes' : 
                                              feature.includes('message') ? 'Messages' : 'Value'
                                    },
                                    beginAtZero: true,
                                    suggestedMin: 0
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Cost (KRW)'
                                    },
                                    beginAtZero: true,
                                    suggestedMin: 0
                                }
                            },
                            plugins: {
                                legend: {
                                    position: 'top',
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            const point = context.raw;
                                            const planName = point.plan ? point.plan : 'Unknown';
                                            const xValue = point.x !== null ? point.x.toLocaleString() : 'Unlimited';
                                            const yValue = point.y.toLocaleString();
                                            return `${planName}: ${xValue} ${feature.includes('data') ? 'GB' : 
                                                            feature.includes('voice') ? 'min' : 
                                                            feature.includes('message') ? 'SMS' : ''} - ${yValue} KRW`;
                                        }
                                    }
                                }
                            }
                        }
                    };
                    
                    const chart = new Chart(canvas, chartConfig);
                    charts.push(chart);
                }
                
                // Debug: Log the data to console
                console.log('Cost Structure Data:', costStructureData);
                console.log('Plan Efficiency Data:', planEfficiencyData);
                
                // Create cost structure charts if data is available
                if (costStructureData && costStructureData !== null) {
                    console.log('Creating cost structure charts...');
                    createCostStructureCharts(costStructureData);
                } else {
                    console.log('No cost structure data available');
                }
                
                // Create plan efficiency chart if data is available
                if (planEfficiencyData && planEfficiencyData !== null) {
                    console.log('Creating plan efficiency chart...');
                    createPlanEfficiencyChart(planEfficiencyData);
                } else {
                    console.log('No plan efficiency data available');
                }
                
                // Create marginal cost charts if cost structure data is available
                // Multi-frontier charts removed per user request
            });
            
            // Function to create cost structure charts
            function createCostStructureCharts(data) {
                console.log('createCostStructureCharts called with data:', data);
                
                // Chart 1: Cost structure breakdown (doughnut chart)
                const costStructureCanvas = document.getElementById('costStructureChart');
                console.log('Cost structure canvas element:', costStructureCanvas);
                console.log('Data.overall:', data.overall);
                
                if (costStructureCanvas && data.overall) {
                    console.log('Creating doughnut chart...');
                    new Chart(costStructureCanvas, {
                        type: 'doughnut',
                        data: {
                            labels: data.overall.labels,
                            datasets: [{
                                data: data.overall.data,
                                backgroundColor: data.overall.colors.slice(0, data.overall.data.length),
                                borderWidth: 2,
                                borderColor: '#fff'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    position: 'bottom',
                                    labels: {
                                        boxWidth: 15,
                                        padding: 15
                                    }
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            const value = context.parsed;
                                            const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                            const percentage = ((value / total) * 100).toFixed(1);
                                            return context.label + ': ₩' + value.toLocaleString() + ' (' + percentage + '%)';
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Chart 2: Per-unit costs (horizontal bar chart)
                const unitCostCanvas = document.getElementById('unitCostChart');
                if (unitCostCanvas && data.unit_costs) {
                    new Chart(unitCostCanvas, {
                        type: 'bar',
                        data: {
                            labels: data.unit_costs.labels,
                            datasets: [{
                                label: '단위당 비용 (Cost per Unit)',
                                data: data.unit_costs.data,
                                backgroundColor: data.unit_costs.colors.slice(0, data.unit_costs.data.length),
                                borderWidth: 1,
                                borderColor: '#333'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            indexAxis: 'y',  // Horizontal bars
                            plugins: {
                                legend: {
                                    display: false
                                },
                                tooltip: {
                                    callbacks: {
                                        label: function(context) {
                                            const value = context.parsed.x;
                                            const unit = data.unit_costs.units[context.dataIndex] || '';
                                            return '₩' + value.toLocaleString() + ' ' + unit;
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: '비용 (KRW)'
                                    },
                                    beginAtZero: true
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: '기능 (Features)'
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Chart 3: Detailed marginal cost analysis
                const marginalCostCanvas = document.getElementById('marginalCostChart');
                if (marginalCostCanvas && data.marginal_analysis) {
                    const marginalData = data.marginal_analysis;
                    
                    new Chart(marginalCostCanvas, {
                        type: 'bar',
                        data: {
                            labels: marginalData.features,
                            datasets: [{
                                label: '마진 비용 계수 (Marginal Cost Coefficient)',
                                data: marginalData.coefficients,
                                backgroundColor: marginalData.colors.slice(0, marginalData.coefficients.length),
                                borderWidth: 1,
                                borderColor: '#333'
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: {
                                    display: false
                                },
                                title: {
                                    display: true,
                                    text: `기본 인프라 비용: ₩${marginalData.base_cost.toLocaleString()} | 분석된 기능별 마진 비용`
                                },
                                tooltip: {
                                    callbacks: {
                                        title: function(context) {
                                            return marginalData.features[context[0].dataIndex];
                                        },
                                        label: function(context) {
                                            const index = context.dataIndex;
                                            const value = context.parsed.y;
                                            const interpretation = marginalData.interpretations[index];
                                            return [
                                                `마진 비용: ₩${value.toLocaleString()}`,
                                                `해석: ${interpretation}`
                                            ];
                                        }
                                    }
                                }
                            },
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: '기능 (Features)'
                                    }
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: '마진 비용 계수 (Marginal Cost Coefficient, ₩)'
                                    },
                                    beginAtZero: true
                                }
                            }
                        }
                    });
                }
            }
            
            // Function to create plan efficiency chart
            function createPlanEfficiencyChart(data) {
                const canvas = document.getElementById('planEfficiencyChart');
                if (!canvas || !data || !data.plans) return;
                
                // Prepare datasets
                const goodValuePlans = [];
                const poorValuePlans = [];
                
                data.plans.forEach(plan => {
                    const point = {
                        x: plan.baseline,
                        y: plan.actual,
                        r: Math.max(5, Math.min(20, plan.feature_total / 20)), // Bubble size based on features
                        plan_name: plan.plan_name,
                        mvno: plan.mvno,
                        cs_ratio: plan.cs_ratio,
                        feature_total: plan.feature_total
                    };
                    
                    if (plan.is_good_value) {
                        goodValuePlans.push(point);
                    } else {
                        poorValuePlans.push(point);
                    }
                });
                
                // Create diagonal line data
                const diagonalData = [
                    {x: data.diagonal.min, y: data.diagonal.min},
                    {x: data.diagonal.max, y: data.diagonal.max}
                ];
                
                new Chart(canvas, {
                    type: 'bubble',
                    data: {
                        datasets: [
                            {
                                label: '가성비 좋은 요금제 (Good Value)',
                                data: goodValuePlans,
                                backgroundColor: 'rgba(46, 204, 113, 0.6)',
                                borderColor: 'rgba(46, 204, 113, 1)',
                                borderWidth: 2
                            },
                            {
                                label: '과가격 요금제 (Overpriced)',
                                data: poorValuePlans,
                                backgroundColor: 'rgba(231, 76, 60, 0.6)',
                                borderColor: 'rgba(231, 76, 60, 1)',
                                borderWidth: 2
                            },
                            {
                                label: '효율성 기준선 (Perfect Efficiency)',
                                data: diagonalData,
                                type: 'line',
                                borderColor: 'rgba(52, 73, 94, 0.8)',
                                borderWidth: 2,
                                borderDash: [5, 5],
                                pointRadius: 0,
                                showLine: true,
                                fill: false
                            }
                        ]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                title: {
                                    display: true,
                                    text: '계산된 기준 비용 (Calculated Baseline Cost, ₩)'
                                },
                                beginAtZero: true
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: '실제 요금제 비용 (Actual Plan Cost, ₩)'
                                },
                                beginAtZero: true
                            }
                        },
                        plugins: {
                            legend: {
                                position: 'top'
                            },
                            tooltip: {
                                callbacks: {
                                    title: function(context) {
                                        return context[0].raw.plan_name;
                                    },
                                    label: function(context) {
                                        const point = context.raw;
                                        return [
                                            `통신사: ${point.mvno}`,
                                            `CS 비율: ${point.cs_ratio.toFixed(2)}`,
                                            `기준 비용: ₩${point.x.toLocaleString()}`,
                                            `실제 비용: ₩${point.y.toLocaleString()}`,
                                            `총 기능 점수: ${point.feature_total.toFixed(1)}`
                                        ];
                                    }
                                }
                            }
                        }
                    }
                });
            }
            
            // Multi-Feature chart functions removed per user request
            
            // Linear Decomposition Chart Functions removed per user request
            
            // Function to create traditional feature frontier charts
            function createFeatureFrontierCharts() {
                console.log('Creating traditional feature frontier charts');
                
                if (!featureFrontierData || Object.keys(featureFrontierData).length === 0) {
                    console.log('No feature frontier data available');
                    return;
                }
                
                const chartsContainer = document.getElementById('featureCharts');
                if (!chartsContainer) {
                    console.log('Feature charts container not found');
                    return;
                }
                
                // Create charts for each feature
                for (const [feature, data] of Object.entries(featureFrontierData)) {
                    console.log(`Creating traditional frontier chart for ${feature}`);
                    
                    // Create chart container
                    const chartContainer = document.createElement('div');
                    chartContainer.className = 'chart-container';
                    chartContainer.style.width = '100%';
                    chartContainer.style.height = '500px';
                    chartContainer.style.margin = '0 0 20px 0';
                    chartContainer.style.padding = '15px';
                    chartContainer.style.boxSizing = 'border-box';
                    
                    // Create feature title
                    const title = document.createElement('h3');
                    title.textContent = `${feature.replace('_clean', '').replace('_', ' ')} Frontier`;
                    title.style.marginTop = '0';
                    title.style.textAlign = 'center';
                    title.style.color = '#2c3e50';
                    chartContainer.appendChild(title);
                    
                    // Create canvas for Chart.js
                    const canvas = document.createElement('canvas');
                    chartContainer.appendChild(canvas);
                    chartsContainer.appendChild(chartContainer);
                    
                    // Prepare datasets for traditional frontier
                    const frontierDataset = {
                        label: 'Cost Frontier',
                        data: data.frontier_values.map((val, i) => ({
                            x: val,
                            y: data.frontier_contributions[i],
                            plan: data.frontier_plan_names[i]
                        })),
                        borderColor: 'rgba(54, 162, 235, 1)',
                        backgroundColor: 'rgba(54, 162, 235, 0.2)',
                        pointBackgroundColor: 'rgba(54, 162, 235, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        showLine: true
                    };
                    
                    const excludedDataset = {
                        label: 'Excluded Plans',
                        data: data.excluded_values.map((val, i) => ({
                            x: val,
                            y: data.excluded_contributions[i],
                            plan: data.excluded_plan_names[i]
                        })),
                        backgroundColor: 'rgba(255, 99, 132, 0.6)',
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        showLine: false
                    };
                    
                    // Create Chart.js chart
                    new Chart(canvas, {
                        type: 'scatter',
                        data: {
                            datasets: [frontierDataset, excludedDataset]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: feature.includes('data') ? 'GB' : 
                                              feature.includes('voice') ? 'Minutes' : 
                                              feature.includes('message') ? 'Messages' : 'Value'
                                    },
                                    beginAtZero: true
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Cost (KRW)'
                                    },
                                    beginAtZero: true
                                }
                            },
                            plugins: {
                                legend: {
                                    position: 'top'
                                },
                                tooltip: {
                                    callbacks: {
                                        title: function(context) {
                                            return context[0].raw.plan || 'Plan';
                                        },
                                        label: function(context) {
                                            return [
                                                `Feature Value: ${context.parsed.x}`,
                                                `Cost: ₩${context.parsed.y.toLocaleString()}`
                                            ];
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
            }
            
            // Function to create full dataset marginal cost frontier charts
            function createMarginalCostFrontierCharts(marginalCostData) {
                console.log('Creating FULL DATASET marginal cost frontier charts');
                console.log('marginalCostData:', marginalCostData);
                
                if (!marginalCostData || Object.keys(marginalCostData).length === 0) {
                    console.log('No marginal cost frontier data available');
                    return;
                }
                
                const chartsContainer = document.getElementById('marginalCostFrontierCharts');
                if (!chartsContainer) {
                    console.log('Marginal cost frontier charts container not found');
                    return;
                }
                
                // Display method metadata if available
                if (marginalCostData.method_info) {
                    const methodInfo = document.createElement('div');
                    methodInfo.className = 'method-info';
                    methodInfo.style.background = '#e8f4fd';
                    methodInfo.style.padding = '15px';
                    methodInfo.style.margin = '0 0 20px 0';
                    methodInfo.style.borderRadius = '8px';
                    methodInfo.style.border = '1px solid #3498db';
                    
                    methodInfo.innerHTML = `
                        <h4 style="margin: 0 0 10px 0; color: #2c3e50;">📊 ${marginalCostData.method_info.name}</h4>
                        <p style="margin: 0; color: #34495e;">
                            <strong>Method:</strong> ${marginalCostData.method_info.description}<br>
                            <strong>Total Plans Analyzed:</strong> ${marginalCostData.method_info.total_plans_analyzed.toLocaleString()}<br>
                            <strong>Features Analyzed:</strong> ${marginalCostData.method_info.features_analyzed}<br>
                            <strong>Approach:</strong> Entire dataset regression (not just frontier points)
                        </p>
                    `;
                    chartsContainer.appendChild(methodInfo);
                }
                
                // Create charts for each feature (skip metadata entries)
                for (const [feature, data] of Object.entries(marginalCostData)) {
                    if (feature === 'method_info' || feature === 'cost_breakdown' || feature === 'coefficient_comparison') continue; // Skip metadata
                    
                    console.log(`Creating full dataset marginal cost chart for ${feature}`);
                    
                    // Create chart container
                    const chartContainer = document.createElement('div');
                    chartContainer.className = 'chart-container';
                    chartContainer.style.width = '100%';
                    chartContainer.style.height = '600px';
                    chartContainer.style.margin = '0 0 30px 0';
                    chartContainer.style.padding = '20px';
                    chartContainer.style.boxSizing = 'border-box';
                    chartContainer.style.border = '1px solid #ddd';
                    chartContainer.style.borderRadius = '8px';
                    chartContainer.style.background = '#fafafa';
                    
                    // Create feature title
                    const title = document.createElement('h3');
                    title.textContent = `${data.display_name} - Full Dataset Analysis`;
                    title.style.marginTop = '0';
                    title.style.textAlign = 'center';
                    title.style.color = '#2c3e50';
                    chartContainer.appendChild(title);
                    
                    // Create subtitle with analysis info
                    const subtitle = document.createElement('p');
                    subtitle.innerHTML = `
                        <strong>Coefficient:</strong> ₩${data.pure_coefficient.toFixed(2)} ${data.unit}<br>
                        <strong>Feature Range:</strong> ${data.feature_range.min} - ${data.feature_range.max} (${data.feature_range.unique_values} unique values)<br>
                        <strong>Frontier Points:</strong> ${data.feature_range.filtered_frontier_points}<br>
                        ${data.unlimited_info ? `<strong>Unlimited Plans:</strong> ${data.unlimited_info.count} plans, cheapest: ₩${data.unlimited_info.min_cost.toLocaleString()}` : '<strong>No unlimited plans</strong> for this feature'}
                    `;
                    subtitle.style.textAlign = 'center';
                    subtitle.style.color = '#7f8c8d';
                    subtitle.style.fontSize = '0.9em';
                    subtitle.style.margin = '0 0 15px 0';
                    chartContainer.appendChild(subtitle);
                    
                    // Create canvas for Chart.js
                    const canvas = document.createElement('canvas');
                    chartContainer.appendChild(canvas);
                    chartsContainer.appendChild(chartContainer);
                    
                    // Prepare datasets
                    const datasets = [];
                    
                    // Dataset 1: Frontier points (cumulative costs from piecewise segments)
                    if (data.frontier_points && data.frontier_points.length > 0) {
                        const frontierDataset = {
                            label: `Cumulative Cost Trend (Piecewise)`,
                            data: data.frontier_points.map(point => ({
                                x: point.feature_value,
                                y: point.cumulative_cost,  // Plot CUMULATIVE cost, not marginal rate
                                actual_cost: point.actual_cost,
                                marginal_rate: point.marginal_cost,
                                predicted_cost: point.cumulative_cost,
                                segment: point.segment_info,
                                plan_count: point.plan_count
                            })),
                            borderColor: 'rgba(52, 152, 219, 1)',      // Blue line
                            backgroundColor: 'rgba(52, 152, 219, 0.1)', // Light blue fill
                            pointBackgroundColor: 'rgba(52, 152, 219, 1)',
                            pointRadius: 5,
                            pointHoverRadius: 8,
                            borderWidth: 3,
                            fill: false,
                            tension: 0.1
                        };
                        datasets.push(frontierDataset);
                    }
                    
                    // Dataset 2: Actual market plans (scatter)
                    if (data.actual_plan_points && data.actual_plan_points.length > 0) {
                        const actualPlansDataset = {
                            label: 'Actual Market Plans',
                            data: data.actual_plan_points.map(point => ({
                                x: point.feature_value,
                                y: point.actual_cost,
                                plan_name: point.plan_name,
                                segment: point.segment_info
                            })),
                            backgroundColor: 'rgba(231, 76, 60, 0.6)',  // Red dots
                            pointBackgroundColor: 'rgba(231, 76, 60, 0.8)',
                            pointRadius: 3,
                            pointHoverRadius: 6,
                            showLine: false
                        };
                        datasets.push(actualPlansDataset);
                    }
                    
                    // Dataset 3: Unlimited plans (if available)
                    if (data.unlimited_info && data.unlimited_info.has_unlimited) {
                        const unlimitedDataset = {
                            label: `Unlimited Plans`,
                            data: [{
                                x: data.feature_range.max * 1.1, // Position at right edge
                                y: data.unlimited_info.min_cost,
                                plan_name: data.unlimited_info.plan_name,
                                unlimited_count: data.unlimited_info.count
                            }],
                            borderColor: 'rgba(255, 159, 64, 1)',      // Orange
                            backgroundColor: 'rgba(255, 159, 64, 0.8)', // Orange dot
                            pointBackgroundColor: 'rgba(255, 159, 64, 1)',
                            pointRadius: 8,
                            pointHoverRadius: 12,
                            borderWidth: 3,
                            showLine: false,
                            pointStyle: 'triangle'
                        };
                        datasets.push(unlimitedDataset);
                    }
                    
                    // Create Chart.js chart
                    new Chart(canvas, {
                        type: 'line',
                        data: { datasets: datasets },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                x: {
                                    title: {
                                        display: true,
                                        text: data.display_name
                                    },
                                    beginAtZero: true
                                },
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Cost (KRW)'
                                    },
                                    beginAtZero: true,
                                    grace: '10%'
                                }
                            },
                            plugins: {
                                legend: {
                                    position: 'top'
                                },
                                tooltip: {
                                    callbacks: {
                                        title: function(context) {
                                            const point = context[0];
                                            if (point.raw.plan_name) {
                                                return point.raw.plan_name;
                                            }
                                            return `${data.display_name}: ${point.parsed.x}`;
                                        },
                                        label: function(context) {
                                            const point = context.raw;
                                            const dataset = context.dataset;
                                            
                                            if (dataset.label.includes('Cumulative Cost Trend')) {
                                                return [
                                                    `Cumulative Cost: ₩${context.parsed.y.toLocaleString()}`,
                                                    `Marginal Rate: ₩${point.marginal_rate.toFixed(2)} ${data.unit}`,
                                                    `Actual Market Cost: ₩${point.actual_cost.toLocaleString()}`,
                                                    `Plans at this level: ${point.plan_count}`,
                                                    `Segment: ${point.segment}`
                                                ];
                                            } else if (dataset.label.includes('Actual Market Plans')) {
                                                return [
                                                    `Plan: ${point.plan_name}`,
                                                    `Actual Cost: ₩${context.parsed.y.toLocaleString()}`,
                                                    `Feature Value: ${context.parsed.x}`,
                                                    `Segment: ${point.segment}`
                                                ];
                                            } else if (dataset.label.includes('Unlimited')) {
                                                return [
                                                    `Unlimited Plan: ${point.plan_name}`,
                                                    `Cost: ₩${context.parsed.y.toLocaleString()}`,
                                                    `Total Unlimited Plans: ${point.unlimited_count}`
                                                ];
                                            }
                                            return `Cost: ₩${context.parsed.y.toLocaleString()}`;
                                        }
                                    }
                                }
                            }
                        }
                    });
                }
                
                // Display coefficient comparison if available
                if (marginalCostData.coefficient_comparison) {
                    const comparisonContainer = document.createElement('div');
                    comparisonContainer.className = 'coefficient-comparison';
                    comparisonContainer.style.background = '#f8f9fa';
                    comparisonContainer.style.padding = '20px';
                    comparisonContainer.style.margin = '20px 0';
                    comparisonContainer.style.borderRadius = '8px';
                    comparisonContainer.style.border = '1px solid #dee2e6';
                    
                    const comparisonTitle = document.createElement('h4');
                    comparisonTitle.textContent = '📊 Piecewise Marginal Cost Structure';
                    comparisonTitle.style.margin = '0 0 15px 0';
                    comparisonTitle.style.color = '#2c3e50';
                    comparisonContainer.appendChild(comparisonTitle);
                    
                    const coeffTable = document.createElement('table');
                    coeffTable.style.width = '100%';
                    coeffTable.style.borderCollapse = 'collapse';
                    coeffTable.style.fontSize = '0.9em';
                    
                    let tableHTML = `
                        <thead>
                            <tr style="background: #e9ecef;">
                                <th style="padding: 10px; border: 1px solid #dee2e6; text-align: left;">Feature</th>
                                <th style="padding: 10px; border: 1px solid #dee2e6; text-align: left;">Piecewise Segments</th>
                                <th style="padding: 10px; border: 1px solid #dee2e6; text-align: center;">Unit</th>
                            </tr>
                        </thead>
                        <tbody>
                    `;
                    
                    const comparison = marginalCostData.coefficient_comparison;
                    for (let i = 0; i < comparison.features.length; i++) {
                        const feature = comparison.features[i];
                        const segments = comparison.piecewise_segments[i];
                        const unit = comparison.units[i];
                        
                        // Format segments as a list
                        const segmentsList = segments.map(seg => `<div style="margin: 2px 0; font-size: 0.9em;">${seg}</div>`).join('');
                        
                        tableHTML += `
                            <tr>
                                <td style="padding: 8px; border: 1px solid #dee2e6; vertical-align: top;">${feature}</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6; vertical-align: top;">${segmentsList}</td>
                                <td style="padding: 8px; border: 1px solid #dee2e6; text-align: center; vertical-align: top;">${unit}</td>
                            </tr>
                        `;
                    }
                    
                    tableHTML += '</tbody>';
                    coeffTable.innerHTML = tableHTML;
                    comparisonContainer.appendChild(coeffTable);
                    
                    chartsContainer.appendChild(comparisonContainer);
                }
            }
            
            // Initialize all charts when DOM is ready
            document.addEventListener('DOMContentLoaded', () => {
                // Create traditional feature frontier charts
                createFeatureFrontierCharts();
                
                // Create marginal cost frontier charts
                createMarginalCostFrontierCharts(marginalCostFrontierData);
                
                // Create plan efficiency chart
                createPlanEfficiencyChart(planEfficiencyData);
                
                // Create multi-frontier analysis charts if available
                if (advancedAnalysisData && advancedAnalysisData !== null) {
                    createMultiFrontierCharts(advancedAnalysisData);
                }
                
                // Linear decomposition charts removed per user request
            });

        </script>
    </body>
</html>"""

    # Use string replacement instead of .format() to avoid issues with JavaScript braces
    html = html_template.replace('{report_title}', report_title)
    html = html.replace('{timestamp_str}', timestamp_str)
    html = html.replace('{len_df_sorted:,}', f"{len(df_sorted):,}")
    html = html.replace('{avg_cs:.2f}', f"{df_sorted['CS'].mean():.2f}")
    html = html.replace('{high_cs_count:,}', f"{(df_sorted['CS'] >= 1).sum():,}")
    html = html.replace('{high_cs_pct:.1%}', f"{(df_sorted['CS'] >= 1).sum()/len(df_sorted):.1%}")
    html = html.replace('{low_cs_count:,}', f"{(df_sorted['CS'] < 1).sum():,}")
    html = html.replace('{low_cs_pct:.1%}', f"{(df_sorted['CS'] < 1).sum()/len(df_sorted):.1%}")
    html = html.replace('{method_info_html}', method_info_html)
    html = html.replace('{comparison_info_html}', comparison_info_html)
    html = html.replace('{multi_frontier_chart_html}', advanced_analysis_chart_html)
    html = html.replace('{feature_rates_table_html}', feature_rates_table_html)
    # Linear decomposition chart removed per user request
    html = html.replace('{all_plans_html}', all_plans_html)

    # Replace JSON placeholders safely
    html = html.replace('__FEATURE_FRONTIER_JSON__', feature_frontier_json)
    html = html.replace('__MARGINAL_COST_FRONTIER_JSON__', marginal_cost_frontier_json)
    html = html.replace('__ADVANCED_ANALYSIS_JSON__', advanced_analysis_json)
    # Linear decomposition JSON removed per user request
    html = html.replace('__PLAN_EFFICIENCY_JSON__', plan_efficiency_json)

    return html
