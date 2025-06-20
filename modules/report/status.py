"""
Status Module

This module handles chart status HTML generation and display logic.
"""

def get_chart_status_html(chart_type, chart_div_id, df=None, charts_data=None, chart_statuses=None):
    """
    Generate loading/error status HTML for individual chart sections.
    
    Args:
        chart_type: Type of chart ('feature_frontier', 'plan_efficiency', etc.)
        chart_div_id: HTML div ID for the chart container
        df: DataFrame with data (None if no data available)
        charts_data: Pre-calculated charts data from file storage
        chart_statuses: Dictionary with individual chart statuses
        
    Returns:
        str: HTML string for chart status overlay or empty string if chart is ready
    """
    # If no data, always show waiting for data message
    if df is None or df.empty:
        return f"""
        <div class="chart-waiting-overlay" id="{chart_div_id}_waiting">
            <div class="waiting-content">
                <div class="waiting-icon">📊</div>
                <p>데이터 처리 대기 중...</p>
                <p style="font-size: 0.9em; color: #666;">
                    <code>/process</code> 엔드포인트를 통해 데이터를 처리하면 차트가 표시됩니다.
                </p>
            </div>
        </div>
        <style>
        .chart-waiting-overlay {{
            position: relative;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8f9fa;
            border: 1px dashed #dee2e6;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .waiting-content {{
            text-align: center;
            padding: 40px;
            color: #6c757d;
        }}
        .waiting-icon {{
            font-size: 48px;
            margin-bottom: 20px;
        }}
        </style>
        """
    
    # Check if pre-calculated charts data is available
    if charts_data and chart_type in charts_data and charts_data[chart_type] is not None:
        return ""  # Chart data is ready, show chart normally
    
    # If no charts data but have ranking data, show calculating message
    if not charts_data:
        return f"""
        <div class="chart-calculating-overlay" id="{chart_div_id}_calculating">
            <div class="calculating-content">
                <div class="calculating-icon">⚙️</div>
                <p>차트 계산 중...</p>
                <p style="font-size: 0.9em; color: #666;">
                    백그라운드에서 차트를 생성하고 있습니다. 잠시 후 새로고침해주세요.
                </p>
            </div>
        </div>
        <style>
        .chart-calculating-overlay {{
            position: relative;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8f9fa;
            border: 1px dashed #ffc107;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .calculating-content {{
            text-align: center;
            padding: 40px;
            color: #856404;
        }}
        .calculating-icon {{
            font-size: 48px;
            animation: spin 2s linear infinite;
            margin-bottom: 20px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>
        """
    
    # Fallback to chart_statuses if no pre-calculated data
    if not chart_statuses:
        return ""  # No status info, show chart normally
        
    status_info = chart_statuses.get(chart_type, {})
    status = status_info.get('status', 'ready')
    
    if status == 'calculating':
        progress = status_info.get('calculation_progress', 0)
        return f"""
        <div class="chart-loading-overlay" id="{chart_div_id}_loading">
            <div class="loading-content">
                <div class="spinner">⚙️</div>
                <p>차트 계산 중... {progress}%</p>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {progress}%"></div>
                </div>
            </div>
        </div>
        <style>
        .chart-loading-overlay {{
            position: relative;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8f9fa;
            border: 1px dashed #dee2e6;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .loading-content {{
            text-align: center;
            padding: 40px;
        }}
        .spinner {{
            font-size: 48px;
            animation: spin 2s linear infinite;
            margin-bottom: 20px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        .progress-bar {{
            width: 200px;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            overflow: hidden;
            margin: 10px auto;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #007bff, #28a745);
            transition: width 0.3s ease;
        }}
        </style>
        """
    elif status == 'error':
        error_msg = status_info.get('error_message', 'Unknown error')
        return f"""
        <div class="chart-error-overlay" id="{chart_div_id}_error">
            <div class="error-content">
                <div class="error-icon">❌</div>
                <p>차트 생성 실패</p>
                <details>
                    <summary>오류 세부사항</summary>
                    <pre>{error_msg[:200]}...</pre>
                </details>
                <button onclick="checkChartStatus()">상태 확인</button>
            </div>
        </div>
        <style>
        .chart-error-overlay {{
            position: relative;
            min-height: 300px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #fff5f5;
            border: 1px solid #fed7d7;
            border-radius: 8px;
            margin: 20px 0;
        }}
        .error-content {{
            text-align: center;
            padding: 40px;
            color: #e53e3e;
        }}
        .error-icon {{
            font-size: 48px;
            margin-bottom: 20px;
        }}
        </style>
        """
    else:
        return ""  # Chart is ready, show normally 