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

def generate_html_report(df, report_title="Mobile Plan Rankings", timestamp=None):
    """
    Generate a full HTML report with plan rankings and feature frontier charts.
    
    Args:
        df: DataFrame with ranking data
        report_title: Title for the report
        timestamp: Timestamp for the report (defaults to current time)
        
    Returns:
        HTML string for the complete report
    """
    logger.info(f"Generating HTML report with title: {report_title}")
    
    # Set timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now()
    
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Sort DataFrame by rank (CS ratio)
    df_sorted = df.copy()
    if 'CS' in df_sorted.columns:
        df_sorted = df_sorted.sort_values(by='CS', ascending=False)
    
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
    
    # Prepare residual analysis data
    residual_analysis_data = prepare_residual_analysis_data(
        all_chart_data, 
        visual_frontiers_for_residual_table,
        core_continuous_features,
        FEATURE_DISPLAY_NAMES,
        FEATURE_UNITS,
        UNLIMITED_FLAGS
    )
    
    # Generate table HTML
    all_plans_html = generate_all_plans_table_html(df_sorted)
    residual_table_html = generate_residual_analysis_table_html(residual_analysis_data)
    
    # Main HTML template
    html = f"""<!DOCTYPE html>
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
                    <li>분석된 요금제 수: <strong>{len(df_sorted):,}</strong></li>
                    <li>평균 CS 비율: <strong>{df_sorted['CS'].mean():.2f}배</strong></li>
                    <li>고평가 요금제 (CS ≥ 1): <strong>{(df_sorted['CS'] >= 1).sum():,}개</strong> ({(df_sorted['CS'] >= 1).sum()/len(df_sorted):.1%})</li>
                    <li>저평가 요금제 (CS &lt; 1): <strong>{(df_sorted['CS'] < 1).sum():,}개</strong> ({(df_sorted['CS'] < 1).sum()/len(df_sorted):.1%})</li>
                </ul>
            </div>
        
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
            
            <!-- Residual analysis table -->
            <h2>Residual Fee Analysis</h2>
            <div class="container">
                <div class="note">
                    <p>이 표는 기능 비용 프론티어에 있는 플랜의 경우, 분석된 기능과 다른 기능에 귀속될 수 있는 플랜 수수료 비율을 보여줍니다.</p>
                </div>
                <table class="residual-table">
                    <tr>
                        <th>Analyzed Feature</th>
                        <th>Sample Plan</th>
                        <th>Plan Core Specs</th>
                        <th>Fee Breakdown</th>
                    </tr>
                    
                    <!-- Generate table rows for residual analysis -->
                    {residual_table_html}
                </table>
            </div>

            <!-- All plans table -->
            <h2>통합 요금제 랭킹</h2>
            <div class="container">
                {all_plans_html}
            </div>
        </div>

        <!-- Add Chart.js implementation -->
        <script>
            // Feature frontier data from Python
            const featureFrontierData = {feature_frontier_json};
            
            // Color configuration
            const chartColors = {{
                frontier: 'rgba(54, 162, 235, 1)',       // Blue line
                frontierFill: 'rgba(54, 162, 235, 0.1)', // Light blue fill
                unlimited: 'rgba(255, 159, 64, 1)',      // Orange for unlimited
                excluded: 'rgba(255, 99, 132, 0.6)',     // Red for excluded
                otherPoints: 'rgba(201, 203, 207, 0.6)'  // Gray for other
            }};
            
            // Create charts for each feature
            document.addEventListener('DOMContentLoaded', () => {{
                const chartsContainer = document.getElementById('featureCharts');
                
                // Track created charts for potential later use (e.g., responsiveness)
                const charts = [];
                
                // For each feature in the data, create a chart
                for (const [feature, data] of Object.entries(featureFrontierData)) {{
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
                    const frontierDataset = {{
                        label: 'Cost Frontier',
                        data: data.frontier_values.map((val, i) => ({{
                            x: val,
                            y: data.frontier_contributions[i],
                            plan: data.frontier_plan_names[i]
                        }})),
                        borderColor: chartColors.frontier,
                        backgroundColor: chartColors.frontierFill,
                        pointBackgroundColor: chartColors.frontier,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        borderWidth: 2,
                        fill: true,
                        tension: 0.1,
                        showLine: true
                    }};
                    
                    const excludedDataset = {{
                        label: 'Excluded Plans',
                        data: data.excluded_values.map((val, i) => ({{
                            x: val,
                            y: data.excluded_contributions[i],
                            plan: data.excluded_plan_names[i]
                        }})),
                        backgroundColor: chartColors.excluded,
                        pointRadius: 5,
                        pointHoverRadius: 7,
                        showLine: false
                    }};
                    
                    // Create a dataset for unlimited point if present
                    let unlimitedDataset = null;
                    if (data.has_unlimited) {{
                        unlimitedDataset = {{
                            label: 'Unlimited Plan',
                            data: [{{
                                x: null, // Will be rendered on right edge
                                y: data.unlimited_value,
                                plan: data.unlimited_plan
                            }}],
                            backgroundColor: chartColors.unlimited,
                            pointRadius: 7,
                            pointHoverRadius: 9,
                            pointStyle: 'triangle',
                            rotation: 90,
                            showLine: false
                        }};
                    }}
                    
                    // Combine datasets
                    const datasets = [frontierDataset, excludedDataset];
                    if (unlimitedDataset) datasets.push(unlimitedDataset);
                    
                    // Create Chart.js chart
                    const chartConfig = {{
                        type: 'scatter',
                        data: {{
                            datasets: datasets
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: feature.includes('data') ? 'GB' : 
                                              feature.includes('voice') ? 'Minutes' : 
                                              feature.includes('message') ? 'Messages' : 'Value'
                                    }},
                                    beginAtZero: true,
                                    suggestedMin: 0
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'Cost (KRW)'
                                    }},
                                    beginAtZero: true,
                                    suggestedMin: 0
                                }}
                            }},
                            plugins: {{
                                legend: {{
                                    position: 'top',
                                }},
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            const point = context.raw;
                                            const planName = point.plan ? point.plan : 'Unknown';
                                            const xValue = point.x !== null ? point.x.toLocaleString() : 'Unlimited';
                                            const yValue = point.y.toLocaleString();
                                            return `${{planName}}: ${{xValue}} ${{feature.includes('data') ? 'GB' : 
                                                            feature.includes('voice') ? 'min' : 
                                                            feature.includes('message') ? 'SMS' : ''}} - ${{yValue}} KRW`;
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }};
                    
                    const chart = new Chart(canvas, chartConfig);
                    charts.push(chart);
                }}
            }});
        </script>
    </body>
</html>"""

    return html 