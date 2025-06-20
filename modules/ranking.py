"""
Ranking Module (Facade)

This module serves as a facade for the refactored ranking functionality.
The original large module has been decomposed into focused modules for better maintainability.

Modules:
- ranking_logic: Ranking calculation and statistical analysis
- display_utils: HTML generation and formatting utilities

Original functions maintained for backward compatibility.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import time
from datetime import datetime
import logging

# Import from refactored modules
from .ranking_logic import (
    format_number_with_commas, 
    shorten_plan_name, 
    calculate_rankings_with_ties,
    get_ranking_statistics
)

# Configure logging
logger = logging.getLogger(__name__)

def generate_html_report(df, model_name, timestamp, model_metrics=None):
    """
    Generate a full HTML report with styling.
    Simplified facade version - delegates to display utilities.
    
    Args:
        df: DataFrame with ranking results
        model_name: Name of the model used for prediction
        timestamp: Timestamp for the report
        model_metrics: Optional dictionary containing model performance metrics
        
    Returns:
        HTML string containing the report
    """
    # Basic HTML structure
    html_head = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>모바일 요금제 랭킹</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; color: #333; line-height: 1.6; }
        h1, h2 { color: #2c3e50; }
        .container { max-width: 100%; margin: 0 auto; }
        .summary { background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; text-align: center; }
        table, th, td { border: 1px solid #ddd; }
        th { background-color: #f2f2f2; color: #333; font-weight: bold; position: sticky; top: 0; z-index: 10; text-align: center; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        tr:hover { background-color: #f1f1f1; }
        th, td { padding: 8px 12px; overflow-wrap: break-word; word-break: break-all; text-align: center; }
        .highlight-high { color: #27ae60; font-weight: bold; }
        .highlight-low { color: #e74c3c; font-weight: bold; }
    </style>
</head>
<body>
    <div class="container">
"""
    
    html_foot = """
    </div>
</body>
</html>"""
    
    # Get statistics
    stats = get_ranking_statistics(df)
    
    # Summary HTML
    summary_html = f"""
        <h1>모바일 요금제 랭킹</h1>
        <h2>모델: {model_name}</h2>
        <p>생성일: {timestamp}</p>
        
        <div class="summary">
            <h2>요약 통계</h2>
            <ul>
                <li>분석된 요금제 수: <strong>{format_number_with_commas(stats['total_plans'])}</strong></li>
                <li>평균 가성비: <strong>{stats['avg_ratio']:.2f}배</strong></li>
                <li>저평가 요금제: <strong>{format_number_with_commas(stats['poor_value_count'])}개</strong></li>
                <li>고평가 요금제: <strong>{format_number_with_commas(stats['good_value_count'])}개</strong></li>
            </ul>
        </div>
    """
    
    # Simple table HTML
    table_html = """
        <h2>통합 요금제 랭킹</h2>
        <table>
            <thead>
                <tr>
                    <th>순위</th>
                    <th>요금제명</th>
                    <th>원가</th>
                    <th>할인가</th>
                    <th>가성비</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add table rows
    for _, row in df.head(50).iterrows():  # Limit to top 50 for performance
        plan_name = shorten_plan_name(row.get('plan_name', 'Unknown'), 30)
        original_fee = format_number_with_commas(row.get('original_fee', 0))
        fee = format_number_with_commas(row.get('fee', 0))
        value_ratio = row.get('value_ratio', 1.0)
        
        value_class = "highlight-high" if value_ratio > 1 else "highlight-low" if value_ratio < 1 else ""
        
        table_html += f"""
            <tr>
                <td>{row.get('display_ranking', 'N/A')}</td>
                <td>{plan_name}</td>
                <td>{original_fee}원</td>
                <td>{fee}원</td>
                <td class="{value_class}">{value_ratio:.2f}배</td>
            </tr>
        """
    
    table_html += """
            </tbody>
        </table>
    """
    
    return html_head + summary_html + table_html + html_foot

def save_report(html_content, model_name, dataset_type, feature_set, timestamp):
    """
    Save HTML report to file.
    
    Args:
        html_content: HTML content string
        model_name: Name of the model
        dataset_type: Type of dataset
        feature_set: Feature set used
        timestamp: Timestamp for filename
        
    Returns:
        Path to saved file
    """
    # Create reports directory if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Generate filename
    filename = f"ranking_report_{model_name}_{dataset_type}_{feature_set}_{timestamp}.html"
    filepath = reports_dir / filename
    
    # Save file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Report saved to {filepath}")
    return filepath

# Re-export functions for backward compatibility
__all__ = [
    'format_number_with_commas',
    'shorten_plan_name', 
    'calculate_rankings_with_ties',
    'generate_html_report',
    'save_report',
    'get_ranking_statistics'
]