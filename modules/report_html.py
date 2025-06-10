"""
Report HTML Module

This module handles the main HTML report generation.
"""

import logging
import json
import pandas as pd
from datetime import datetime
from .report_utils import NumpyEncoder, FEATURE_DISPLAY_NAMES, FEATURE_UNITS, UNLIMITED_FLAGS
from .report_charts import prepare_feature_frontier_data, prepare_residual_analysis_data
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
    
    for feature, cost in cost_structure.items():
        if feature != 'base_cost' and feature in feature_labels:
            info = feature_labels[feature]
            
            # For overall breakdown - use normalized values for comparison
            overall_data['labels'].append(info['label'])
            overall_data['data'].append(abs(cost))  # Use absolute value for visualization
            
            # For unit costs - only meaningful marginal costs
            if cost > 0:  # Only positive marginal costs
                unit_cost_data['labels'].append(info['label'])
                unit_cost_data['data'].append(cost)
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
    
    for feature, cost in cost_structure.items():
        if feature != 'base_cost' and feature in feature_analysis:
            analysis = feature_analysis[feature]
            marginal_analysis['features'].append(analysis['name'])
            marginal_analysis['coefficients'].append(cost)
            marginal_analysis['interpretations'].append(analysis['interpretation'](cost))
    
    return {
        'overall': overall_data,
        'unit_costs': unit_cost_data,
        'marginal_analysis': marginal_analysis
    }

def prepare_plan_efficiency_data(df, method):
    """
    Prepare data for Plan Value Efficiency Matrix visualization.
    
    Args:
        df: DataFrame with plan data including CS ratios and baselines
        method: Method used ('linear_decomposition' or 'frontier')
        
    Returns:
        Dictionary with chart data for JavaScript rendering
    """
    if df.empty:
        return None
    
    efficiency_data = {
        'plans': [],
        'diagonal': {'min': 0, 'max': 0}
    }
    
    # Get baseline and actual cost columns
    baseline_col = 'B_decomposed' if method == 'linear_decomposition' and 'B_decomposed' in df.columns else 'B'
    actual_col = 'fee'
    cs_col = 'CS_decomposed' if method == 'linear_decomposition' and 'CS_decomposed' in df.columns else 'CS'
    
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
        
    logger.info(f"Generating HTML report with title: {report_title}")
    
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
                
                for feature, cost in cost_structure.items():
                    if feature != 'base_cost':
                        interpretation = feature_interpretations.get(feature, 'Feature-specific cost')
                        method_info_html += f"<tr><td>{feature}</td><td>₩{cost:.2f}</td><td>{interpretation}</td></tr>"
                
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
    
    # Generate cost structure chart (only for linear decomposition)
    cost_structure_chart_html = ""
    cost_structure_json = "null"  # Default value
    
    if method == "linear_decomposition":
        # Debug: Always show the chart section for linear decomposition
        # Prepare debug info for cost structure keys
        debug_keys_info = ""
        if cost_structure:
            keys_list = list(cost_structure.keys())
            debug_keys_info = f'<p>Cost Structure Keys: {keys_list}</p>'
        else:
            debug_keys_info = '<p>Cost Structure Keys: None</p>'
            
        cost_structure_chart_html = f"""
        <div class="charts-wrapper">
            <h2>📊 Linear Decomposition Charts</h2>
            <div class="note">
                <p><strong>Debug Info:</strong> Method = {method}, Cost Structure Available = {bool(cost_structure)}</p>
                {debug_keys_info}
            </div>
            <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: center;">
                <div class="chart-container" style="width: 500px; height: 400px;">
                    <h3 style="text-align: center; margin-top: 0;">비용 구성 요소 (Cost Components)</h3>
                    <canvas id="costStructureChart" style="max-height: 300px;"></canvas>
                </div>
                <div class="chart-container" style="width: 500px; height: 400px;">
                    <h3 style="text-align: center; margin-top: 0;">단위당 비용 (Per-Unit Costs)</h3>
                    <canvas id="unitCostChart" style="max-height: 300px;"></canvas>
                </div>
                <div class="chart-container" style="width: 1000px; height: 400px;">
                    <h3 style="text-align: center; margin-top: 0;">발견된 마진 비용 분석 (Discovered Marginal Costs)</h3>
                    <canvas id="marginalCostChart" style="max-height: 300px;"></canvas>
                </div>
            </div>
        </div>
        """
        
        if cost_structure:
            # Prepare chart data
            cost_structure_data = prepare_cost_structure_chart_data(cost_structure)
            cost_structure_json = json.dumps(cost_structure_data, cls=NumpyEncoder)
    
    # Define continuous features for visualization (5 most important)
    core_continuous_features = [
        'basic_data_clean', 
        'daily_data_clean',
        'voice_clean',
        'message_clean',
        'tethering_gb'
    ]
    
    # Prepare data for feature frontier charts
    feature_frontier_data, all_chart_data, visual_frontiers_for_residual_table = prepare_feature_frontier_data(df, core_continuous_features)
    
    # Convert to JSON for JavaScript
    feature_frontier_json = json.dumps(feature_frontier_data, cls=NumpyEncoder)
    
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
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
                font-size: 0.9em;
                text-align: center;
            }}
            table, th, td {{
                border: 1px solid #ddd;
            }}
            th {{
                background-color: #f2f2f2;
                color: #333;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
                text-align: center;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            th, td {{
                padding: 8px 12px;
                overflow-wrap: break-word;
                word-break: break-all;
                text-align: center;
            }}
            .highlight-high {{
                color: #27ae60;
                font-weight: bold;
            }}
            .highlight-low {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .good-value {{
                color: #27ae60;
                font-weight: bold;
            }}
            .bad-value {{
                color: #e74c3c;
                font-weight: bold;
            }}
            .metric-good {{
                color: #27ae60;
            }}
            .metric-average {{
                color: #f39c12;
            }}
            .metric-poor {{
                color: #e74c3c;
            }}
            .container {{
                max-width: 100%;
                margin: 0 auto;
            }}
            .summary {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .metrics {{
                background-color: #eaf7fd;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
            }}
            .note {{
                background-color: #f8f9fa;
                padding: 10px;
                border-left: 4px solid #007bff;
                margin-bottom: 20px;
            }}
            
            /* Content wrapper with padding */
            .content-wrapper {{
                padding: 20px;
            }}
            
            /* Feature charts wrapper - no padding for full width */
            .charts-wrapper {{
                width: 100%;
            }}
            
            /* Feature charts grid */
            .chart-grid {{
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                width: 100%;
            }}
            
            .chart-container {{
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 15px;
                position: relative;
                width: 100%;
                height: 400px;
            }}
            
            @media print {{
                body {{
                    font-size: 10pt;
                }}
                table {{
                    font-size: 9pt;
                }}
                .no-print {{
                    display: none;
                }}
            }}
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
                    <li>분석된 요금제 수: <strong>{len_df_sorted:,}</strong></li>
                    <li>평균 CS 비율: <strong>{avg_cs:.2f}배</strong></li>
                    <li>고평가 요금제 (CS ≥ 1): <strong>{high_cs_count:,}개</strong> ({high_cs_pct:.1%})</li>
                    <li>저평가 요금제 (CS < 1): <strong>{low_cs_count:,}개</strong> ({low_cs_pct:.1%})</li>
                </ul>
            </div>
            
            {method_info_html}
            {comparison_info_html}
        
            <!-- Cost Structure Decomposition Chart (only for linear decomposition) -->
            {cost_structure_chart_html}
        
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

            <h2>전체 요금제 랭킹</h2>
            {all_plans_html}
        </div>

        <!-- Add Chart.js implementation -->
        <script>
            // Feature frontier data from Python
            const featureFrontierData = __FEATURE_FRONTIER_JSON__;
            
            // Cost structure data from Python (only for linear decomposition)
            const costStructureData = __COST_STRUCTURE_JSON__;
            
            // Plan efficiency data from Python
            const planEfficiencyData = __PLAN_EFFICIENCY_JSON__;
            
            console.log('Charts disabled for debugging - data loaded successfully');
        </script>
    </body>
</html>"""

    # Use the f-string approach but with safe JSON replacement
    html = html_template.format(
        report_title=report_title,
        timestamp_str=timestamp_str,
        len_df_sorted=len(df_sorted),
        avg_cs=df_sorted['CS'].mean(),
        high_cs_count=(df_sorted['CS'] >= 1).sum(),
        high_cs_pct=(df_sorted['CS'] >= 1).sum()/len(df_sorted),
        low_cs_count=(df_sorted['CS'] < 1).sum(),
        low_cs_pct=(df_sorted['CS'] < 1).sum()/len(df_sorted),
        method_info_html=method_info_html,
        comparison_info_html=comparison_info_html,
        cost_structure_chart_html=cost_structure_chart_html,
        all_plans_html=all_plans_html
    )

    # Replace JSON placeholders safely
    html = html.replace('__FEATURE_FRONTIER_JSON__', feature_frontier_json)
    html = html.replace('__COST_STRUCTURE_JSON__', cost_structure_json)
    html = html.replace('__PLAN_EFFICIENCY_JSON__', plan_efficiency_json)

    return html
