"""
Main HTML Template Module

This module contains the main HTML template structure for the report.
"""

def get_main_html_template():
    """
    Get the main HTML template structure.
    
    Returns:
        str: Complete HTML template as string with placeholders
    """
    return """<!DOCTYPE html>
<html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{report_title}</title>
        <style>
            {css_styles}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    </head>
    <body>
        <div class="container">
            <h1>모바일 요금제 랭킹</h1>
            <h2>Cost-Spec Ratio 모델</h2>
            <div style="display: flex; justify-content: space-between; align-items: center; margin: 10px 0;">
                <p style="margin: 0;">생성일: {timestamp_str}</p>
                <button onclick="refreshPage()" style="background-color: #007bff; color: white; border: none; padding: 8px 16px; border-radius: 5px; cursor: pointer; font-size: 14px;">
                    🔄 새로고침
                </button>
            </div>
            
            {no_data_message}
            
            <div class="summary" style="{'display:none;' if no_data_message else ''}">
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
                {feature_frontier_status_html}
                <div id="featureCharts" class="chart-grid" style="{feature_frontier_display_style}"></div>
            </div>
            
            <!-- Plan Value Efficiency Matrix -->
            <div class="charts-wrapper">
                <h2>💰 Plan Value Efficiency Analysis</h2>
                <div class="note">
                    <p>이 차트는 각 요금제의 실제 비용 대비 계산된 기준 비용을 보여줍니다. 대각선 아래(녹색 영역)는 가성비가 좋은 요금제, 위(빨간색 영역)는 과가격 요금제입니다.</p>
                </div>
                {plan_efficiency_status_html}
                <div class="chart-container" style="width: 100%; height: 600px; {plan_efficiency_display_style}">
                    <canvas id="planEfficiencyChart"></canvas>
                </div>
                <p style="text-align: center; margin-top: 10px; color: #666; font-size: 0.9em; {plan_efficiency_display_style}">
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
            {javascript_code}
        </script>
    </body>
</html>""" 