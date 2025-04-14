import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime

def calculate_rankings(df, model):
    """
    Calculate rankings for mobile plans based on model predictions and feature weights,
    using the value ratio approach from update_rankings.py.
    
    Args:
        df: DataFrame containing plan data with features
        model: Trained model for price predictions
        
    Returns:
        DataFrame with added ranking scores and rankings
    """
    # Make a copy to avoid modifying the original
    ranking_df = df.copy()
    
    # Extract relevant features for prediction
    if hasattr(model, 'feature_names') and model.feature_names:
        required_features = model.feature_names
        # Check if all required features are in the dataframe
        missing_features = [f for f in required_features if f not in ranking_df.columns]
        if missing_features:
            # Allow missing 'additional_call' for now as it might be missing in some datasets
            if missing_features == ['additional_call']:
                print("Warning: Feature 'additional_call' missing from input data. Proceeding without it.")
                required_features = [f for f in required_features if f != 'additional_call']
            else:
                print(f"Input data is missing features required by the model: {missing_features}")
                # Use whatever features are available
                available_features = [f for f in required_features if f in ranking_df.columns]
                required_features = available_features
                
        # Select only the features the model was trained on
        X = ranking_df[required_features].copy()
    else:
        # Fallback: If feature names aren't loaded with the model, 
        # use all available columns
        X = ranking_df.copy()

    # Get predictions from the model
    predicted_prices = model.predict(X)
    ranking_df['predicted_price'] = predicted_prices
    
    # --- Updated Value Calculation (using value ratio approach from update_rankings.py) ---
    
    # Store both original and discounted fees for reference
    ranking_df['original_price'] = ranking_df['original_fee'] if 'original_fee' in ranking_df.columns else ranking_df['fee']
    ranking_df['discounted_price'] = ranking_df['fee'] if 'fee' in ranking_df.columns else ranking_df['original_price']
    
    # Calculate discount amount per month
    ranking_df['monthly_discount'] = ranking_df['original_price'] - ranking_df['discounted_price']
    
    # Now use the discounted price for value comparison
    comparison_price_col = 'discounted_price'
    
    # 1. Calculate price difference (backward compatibility)
    ranking_df['price_difference'] = ranking_df['predicted_price'] - ranking_df[comparison_price_col]
    
    # 2. Calculate value ratio (predicted_price / discounted_price)
    # Value ratio > 1 means good value, < 1 means poor value
    ranking_df['value_ratio'] = ranking_df.apply(
        lambda row: row['predicted_price'] / row[comparison_price_col] 
                  if row[comparison_price_col] > 0 
                  else (float('inf') if row['predicted_price'] > 0 else 1.0), 
        axis=1
    )
    
    # For backward compatibility
    ranking_df['price_difference_percentage'] = np.where(
        ranking_df[comparison_price_col] > 0,  # Condition: comparison price > 0?
        np.clip((ranking_df['price_difference'] / ranking_df[comparison_price_col]), -1.0, 5.0), # Value if True, capped at reasonable range
        # Value if False (comparison price is 0): use nested np.where
        np.where(ranking_df['price_difference'] > 0, 5.0, # If diff > 0, use max percentage
                np.where(ranking_df['price_difference'] < 0, -1.0, 0.0)) # If diff < 0, use min percentage, else 0
    )
    
    # Also calculate the old value score for compatibility with other code
    ranking_df['value_score'] = np.clip(ranking_df['value_ratio'] - 1, -1, 1)
    
    # --- End Updated Value Calculation ---

    # Sort by value_ratio (higher is better)
    ranking_df = ranking_df.sort_values(by='value_ratio', ascending=False)
    ranking_df['rank'] = range(1, len(ranking_df) + 1)
    
    return ranking_df

def format_number_with_commas(value):
    """Format a numeric value with commas."""
    if pd.isna(value) or value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if value == int(value):  # Check if it's a whole number
            return f"{int(value):,}"
        return f"{value:,.2f}"
    return str(value)

def shorten_plan_name(name, max_length=40):
    """Shorten plan name if it's too long."""
    if not name:
        return ""
    if len(name) <= max_length:
        return name
    return name[:max_length-3] + "..."

def generate_html_report(df, model_name, timestamp, model_metrics=None):
    """
    Generate a full HTML report with styling using the enhanced format from update_rankings.py.
    
    Args:
        df: DataFrame with ranking results
        model_name: Name of the model used for prediction
        timestamp: Timestamp for the report
        model_metrics: Optional dictionary containing model performance metrics
        
    Returns:
        HTML string containing the report
    """
    html_head = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>모바일 요금제 랭킹</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }
        h1, h2 {
            color: #2c3e50;
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
        .metrics table {
            width: auto;
            margin: 0;
        }
        .metrics td, .metrics th {
            padding: 4px 8px;
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
        .metric-good {
            color: #27ae60;
        }
        .metric-average {
            color: #f39c12;
        }
        .metric-poor {
            color: #e74c3c;
        }
        .chart {
            width: 100%;
            max-width: 800px;
            height: 300px;
            margin: 20px auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            background-color: #f9f9f9;
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
</head>
<body>
    <div class="container">
"""
    
    html_foot = """
    </div>
</body>
</html>"""
    
    # Summary statistics
    good_value_count = (df['value_ratio'] >= 1).sum()
    poor_value_count = (df['value_ratio'] < 1).sum()
    total_plans = len(df)
    avg_ratio = df['value_ratio'].mean()
    best_value_plan = df.iloc[0] if not df.empty else None
    worst_value_plan = df.iloc[-1] if not df.empty else None
    
    summary_html = f"""
        <h1>모바일 요금제 랭킹</h1>
        <h2>모델: {model_name}</h2>
        <p>생성일: {timestamp}</p>
        
        <div class="summary">
            <h2>요약 통계</h2>
            <ul>
                <li>분석된 요금제 수: <strong>{format_number_with_commas(total_plans)}</strong></li>
                <li>평균 가성비 (예측가/할인가): <strong>{avg_ratio:.2f}배</strong></li>
                <li>저평가 요금제 (가성비 &lt; 1): <strong>{format_number_with_commas(poor_value_count)}개</strong> ({poor_value_count/total_plans:.1%})</li>
                <li>고평가 요금제 (가성비 ≥ 1): <strong>{format_number_with_commas(good_value_count)}개</strong> ({good_value_count/total_plans:.1%})</li>
    """
    
    if best_value_plan is not None:
        summary_html += f"""
                <li>최고 가성비 요금제: <span class="highlight-high">{best_value_plan.get('plan_name', '')}</span> (가성비: {best_value_plan['value_ratio']:.2f}배)</li>
                <li>최저 가성비 요금제: <span class="highlight-low">{worst_value_plan.get('plan_name', '')}</span> (가성비: {worst_value_plan['value_ratio']:.2f}배)</li>
        """
        
    summary_html += """
            </ul>
        </div>
    """
    
    # Add model metrics section if provided
    metrics_html = ""
    if model_metrics:
        metrics_html = """
        <div class="metrics">
            <h2>모델 성능 지표</h2>
            <table>
                <tr>
                    <th>지표 이름</th>
                    <th>값</th>
                    <th>설명</th>
                </tr>
        """
        
        # Format metrics with descriptions
        metrics_descriptions = {
            "rmse": "Root Mean Squared Error - 실제 가격과 예측 가격의 평균 제곱근 오차",
            "mae": "Mean Absolute Error - 실제 가격과 예측 가격의 평균 절대 오차",
            "r2": "R² Score - 모델이 데이터 변동성을 설명하는 정도 (1에 가까울수록 좋음)",
            "explained_variance": "Explained Variance - 모델이 예측한 가격의 분산이 실제 가격의 분산과 일치하는 정도",
            "mean_absolute_percentage_error": "MAPE - 실제 가격 대비 예측 오차의 평균 백분율",
            "training_time": "모델 학습 시간 (초)",
            "num_features": "사용된 특성 수",
            "num_samples": "학습 샘플 수"
        }
        
        # Define thresholds for metric colors (customize based on your domain)
        def get_metric_class(metric_name, value):
            if metric_name == 'r2':
                if value > 0.8: return "metric-good"
                if value > 0.5: return "metric-average"
                return "metric-poor"
            elif metric_name in ['rmse', 'mae', 'mean_absolute_percentage_error']:
                # For error metrics, lower is better
                if metric_name == 'mean_absolute_percentage_error':
                    if value < 0.1: return "metric-good" 
                    if value < 0.2: return "metric-average"
                    return "metric-poor"
                elif metric_name == 'mae':
                    if value < 5000: return "metric-good" 
                    if value < 10000: return "metric-average"
                    return "metric-poor"
                else:  # rmse
                    if value < 8000: return "metric-good" 
                    if value < 15000: return "metric-average"
                    return "metric-poor"
            return ""  # Default, no color
        
        # Add each metric
        for metric_name, metric_value in model_metrics.items():
            # Format the metric value
            if isinstance(metric_value, float):
                if metric_name == 'mean_absolute_percentage_error':
                    formatted_value = f"{metric_value:.2%}"  # Format as percentage
                elif metric_name in ['training_time', 'r2', 'explained_variance']:
                    formatted_value = f"{metric_value:.4f}"  # 4 decimal places
                else:
                    formatted_value = format_number_with_commas(metric_value)  # Standard number format
            else:
                formatted_value = str(metric_value)
            
            # Get the metric description
            description = metrics_descriptions.get(metric_name, "")
            
            # Get CSS class for color-coding
            metric_class = get_metric_class(metric_name, metric_value if isinstance(metric_value, (int, float)) else 0)
            
            # Add to table
            metrics_html += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td class="{metric_class}">{formatted_value}</td>
                    <td>{description}</td>
                </tr>
            """
        
        metrics_html += """
            </table>
        </div>
        """
        
        # Add basic visualization - Prediction vs Actual scatter plot placeholder
        # This would normally be implemented with JavaScript/D3.js for interactive visualization
        metrics_html += """
        <div class="chart">
            <h3>모델 예측 정확도</h3>
            <p style="text-align: center; margin-top: 120px;">
                이 섹션에서는 실제 가격과 예측 가격의 관계를 시각화합니다. 
                완전한 시각화는 클라이언트 측 JavaScript 구현이 필요합니다.
            </p>
        </div>
        """
    
    # Main section heading
    table_heading = """
        <h2>통합 요금제 랭킹</h2>
    """
    
    # Map for simplified column headers without units
    simplified_headers = {
        'is_5g': '네트워크',
        'basic_data_clean': '기본 데이터',
        'daily_data_clean': '일일 데이터',
        'voice_clean': '음성 통화',
        'message_clean': '문자 메시지',
        'tethering_gb': '테더링',
        'throttle_speed_normalized': '데이터 정책',
        'additional_call': '부가전화'
    }
    
    # Determine the model's features for display
    model_features = []
    if len(df) > 0:
        first_row = df.iloc[0]
        # 열 순서 변경: 네트워크, 기본 데이터, 테더링, 데이터 정책, 일일 데이터, 음성 통화, 문자 메시지
        potential_features = [
            'is_5g',
            'basic_data_clean',
            'tethering_gb',
            'throttle_speed_normalized',
            'daily_data_clean',
            'voice_clean',
            'message_clean',
        ]
        
        model_features = [f for f in potential_features if f in first_row]
    
    # Combine all headers into one table
    all_headers = ["No", "요금제명", "원가", "할인가", "예측가 (가성비)"]
    feature_headers = [simplified_headers.get(feature, feature) for feature in model_features]
    all_headers.extend(feature_headers)
    
    # Start table
    table_html = ['<table>']
    
    # Table header
    table_html.append('<thead><tr>')
    for header in all_headers:
        align = "left" if header in ["요금제명", "네트워크", "데이터 정책"] else "right"
        table_html.append(f'<th align="{align}">{header}</th>')
    table_html.append('</tr></thead>')
    
    # Table body
    table_html.append('<tbody>')
    
    # Add all plans to the table (no limit)
    for _, row in df.iterrows():
        table_html.append('<tr>')
        
        plan_name = row.get('plan_name', f"Plan")
        # Add ID in parentheses after plan name
        plan_name = f"{shorten_plan_name(plan_name, 30)}<br>({row['id']})" if 'id' in row else shorten_plan_name(plan_name, 30)
        
        # Format price values with commas and add units
        original_price = f"{format_number_with_commas(row.get('original_price', row.get('original_fee', 0)))}원"
        discounted_price = f"{format_number_with_commas(row.get('discounted_price', row.get('fee', 0)))}원"
        
        # Calculate value ratio if not already present
        if 'value_ratio' not in row:
            fee = row.get('discounted_price', row.get('fee', 0))
            if fee > 0:
                value_ratio = row.get('predicted_price', 0) / fee
            else:
                value_ratio = float('inf') if row.get('predicted_price', 0) > 0 else 1.0
        else:
            value_ratio = row.get('value_ratio', 1.0)
        
        # Format value display with arrow
        value_arrow = "↑" if value_ratio > 1 else "↓" if value_ratio < 1 else "→"
        value_class = "highlight-high" if value_ratio > 1 else ("highlight-low" if value_ratio < 1 else "")
        
        # Display predicted price and value ratio
        if value_ratio == float('inf'):
            predicted_with_ratio = f"{format_number_with_commas(row.get('predicted_price', 0))}원<br>(<span class='{value_class}'>∞배 {value_arrow}</span>)"
        else:
            predicted_with_ratio = f"{format_number_with_commas(row.get('predicted_price', 0))}원<br>(<span class='{value_class}'>{value_ratio:.2f}배 {value_arrow}</span>)"
        
        # Basic cells
        cells = [
            str(row.get('rank', 'N/A')),
            plan_name, 
            original_price, 
            discounted_price, 
            predicted_with_ratio
        ]
        
        # Add feature cells with appropriate formatting
        for feature in model_features:
            if feature in row:
                value = row[feature]
                
                # Format based on feature type with units
                if feature == 'is_5g':
                    formatted_value = "5G" if value == 1 else "LTE"
                elif feature == 'basic_data_clean':
                    # Check if unlimited data
                    if 'basic_data_unlimited' in row and row['basic_data_unlimited'] == 1:
                        formatted_value = "무제한"
                    else:
                        formatted_value = f"{format_number_with_commas(value)}GB"
                elif feature == 'tethering_gb':
                    if value == 0:
                        formatted_value = "-"
                    else:
                        formatted_value = f"{format_number_with_commas(value)}GB"
                elif feature == 'daily_data_clean':
                    # Check if unlimited daily data
                    if 'daily_data_unlimited' in row and row['daily_data_unlimited'] == 1:
                        formatted_value = "무제한"
                    elif value == 0:
                        formatted_value = "-"
                    else:
                        formatted_value = f"{format_number_with_commas(value)}GB"
                elif feature == 'voice_clean':
                    # Check if unlimited voice and include additional_call if available
                    add_call = row.get('additional_call', 0)
                    if 'voice_unlimited' in row and row['voice_unlimited'] == 1:
                        if add_call > 0:
                            formatted_value = f"무제한<br>(부가: {format_number_with_commas(add_call)}분)"
                        else:
                            formatted_value = "무제한"
                    else:
                        if add_call > 0:
                            formatted_value = f"{format_number_with_commas(value)}분<br>(부가: {format_number_with_commas(add_call)}분)"
                        else:
                            formatted_value = f"{format_number_with_commas(value)}분"
                elif feature == 'message_clean':
                    # Check if unlimited messages
                    if 'message_unlimited' in row and row['message_unlimited'] == 1:
                        formatted_value = "무제한"
                    else:
                        formatted_value = f"{format_number_with_commas(value)}건"
                elif feature == 'throttle_speed_normalized':
                    # Handle throttling display
                    has_unlimited_data = ('basic_data_unlimited' in row and row['basic_data_unlimited'] == 1) or ('daily_data_unlimited' in row and row['daily_data_unlimited'] == 1)
                    unlimited_type = row.get('unlimited_type_numeric', 0)
                    
                    if has_unlimited_data:
                        if value == 1.0:  # 속도 무제한
                            formatted_value = "무제한"
                        elif value > 0:  # 속도 제한 있음
                            formatted_value = f"{value * 10:.1f}Mbps"
                        else:  # 속도 없음
                            formatted_value = "소진시<br>종료"
                    else:
                        # 무제한 데이터가 아님
                        formatted_value = "소진시<br>종료"
                    
                    # Handle different unlimited types
                    if unlimited_type == 2:  # 할당량 후 제한속도
                        formatted_value = f"소진시<br>{value * 10:.1f}Mbps"
                    elif unlimited_type == 1:  # 항상 제한속도
                        formatted_value = f"항상<br>{value * 10:.1f}Mbps"
                    elif unlimited_type == 3:  # 완전 무제한
                        formatted_value = "무제한"
                else:
                    formatted_value = format_number_with_commas(value)
                
                cells.append(formatted_value)
            else:
                cells.append("-")
        
        # Add cells to row
        for idx, cell in enumerate(cells):
            align = "left" if idx == 1 or (idx >= 5 and idx in [5, 8]) else "right"
            table_html.append(f'<td align="{align}">{cell}</td>')
            
        table_html.append('</tr>')
    
    # Close table elements
    table_html.append('</tbody>')
    table_html.append('</table>')
    
    # Combine all HTML parts
    full_html = html_head + summary_html + metrics_html + table_heading + "\n".join(table_html) + html_foot
    
    return full_html

def save_report(html_content, model_name, dataset_type, feature_set, timestamp):
    """Save the HTML report to a file."""
    # Construct the filename
    filename = f"plan_rankings_{model_name}_{dataset_type}_{feature_set}_{timestamp}.html"
    
    # Create reports directory if it doesn't exist
    # Use a path that will work in Hugging Face Spaces
    base_dir = Path("./reports")
    
    try:
        # Try to create directory in the app folder
        base_dir.mkdir(exist_ok=True, parents=True)
        file_path = base_dir / filename
    except PermissionError:
        # Fallback to /tmp if app directory is not writable
        print("Permission error creating reports directory, using /tmp as fallback")
        base_dir = Path("/tmp/reports")
        base_dir.mkdir(exist_ok=True, parents=True)
        file_path = base_dir / filename
    
    # Write the HTML content to the file
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report saved to {file_path}")
    except Exception as e:
        print(f"Error saving report: {e}")
        # Use an even more basic fallback if all else fails
        file_path = Path(f"/tmp/{filename}")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        print(f"Report saved to fallback location: {file_path}")
    
    return str(file_path) 