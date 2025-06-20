"""
HTML Generator Module

This module contains the main HTML report generation function using templates.
"""

import logging
import json
import pandas as pd
from datetime import datetime

# Import template modules
from ..templates import get_main_html_template, get_chart_javascript, get_main_css_styles

# Import report modules
from .status import get_chart_status_html
from .chart_data import prepare_cost_structure_chart_data, prepare_plan_efficiency_data
from .tables import generate_feature_rates_table_html

# Import other dependencies
from ..report_utils import NumpyEncoder
from ..report_charts import prepare_feature_frontier_data, prepare_granular_marginal_cost_frontier_data
from ..report_tables import generate_all_plans_table_html

# Configure logging
logger = logging.getLogger(__name__)

def generate_html_report(df, timestamp=None, report_title="Mobile Plan Rankings", is_cs=True, title=None, method=None, cost_structure=None, chart_statuses=None, charts_data=None):
    """
    Generate a full HTML report with plan rankings and feature frontier charts.
    
    Args:
        df: DataFrame with ranking data
        timestamp: Timestamp for the report (defaults to current time)
        report_title: Title for the report
        is_cs: Whether this is a Cost-Spec report (for backward compatibility)
        title: Alternative title (for backward compatibility)
        method: Cost-Spec method used ('frontier', 'multi_frontier', or 'fixed_rates')
        cost_structure: Cost structure dictionary from linear decomposition
        chart_statuses: Dictionary with individual chart statuses for loading states
        charts_data: Pre-calculated charts data from file storage
        
    Returns:
        HTML string for the complete report
    """
    # Use title parameter if provided (for backward compatibility)
    if title:
        report_title = title
        
    # Add method information to title if available
    if method:
        method_names = {
            "fixed_rates": "Fixed Rates", 
            "multi_frontier": "Multi-Frontier", 
            "frontier": "Frontier-Based"
        }
        method_name = method_names.get(method, "Frontier-Based")
        report_title = f"{report_title} ({method_name})"
        
    # Set timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now()
    
    # Handle timestamp as string or datetime object
    if isinstance(timestamp, str):
        timestamp_str = timestamp
    else:
        timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Handle case when no data is available
    if df is None or df.empty:
        # Create empty DataFrame with default columns for display
        df_sorted = pd.DataFrame(columns=['plan_name', 'CS', 'B', 'fee'])
        no_data_message = """
        <div class="summary">
            <h3>📊 데이터 처리 대기 중</h3>
            <p>아직 데이터가 처리되지 않았습니다. <code>/process</code> 엔드포인트를 통해 데이터를 처리해주세요.</p>
            <button onclick="checkDataAndRefresh()" style="background-color: #28a745; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 10px;">
                🔄 데이터 확인
            </button>
        </div>
        """
    else:
        # Sort DataFrame by rank (CS ratio)
        df_sorted = df.copy()
        if 'CS' in df_sorted.columns:
            df_sorted = df_sorted.sort_values(by='CS', ascending=False)
        no_data_message = ""
    
    # Calculate summary statistics
    if not df_sorted.empty and 'CS' in df_sorted.columns:
        len_df_sorted = len(df_sorted)
        avg_cs = df_sorted['CS'].mean()
        high_cs_count = (df_sorted['CS'] >= 1).sum()
        low_cs_count = (df_sorted['CS'] < 1).sum()
        high_cs_pct = high_cs_count / len_df_sorted if len_df_sorted > 0 else 0
        low_cs_pct = low_cs_count / len_df_sorted if len_df_sorted > 0 else 0
    else:
        len_df_sorted = 0
        avg_cs = 0
        high_cs_count = 0
        low_cs_count = 0
        high_cs_pct = 0
        low_cs_pct = 0
    
    # Add method and cost structure information to summary
    method_info_html = ""
    comparison_info_html = ""
    multi_frontier_chart_html = ""
    
    # Prepare data for charts
    if df is None or df.empty:
        # Empty data for charts when no data available
        feature_frontier_data = {}
        plan_efficiency_data = None
        feature_rates_table_html = ""
        all_plans_html = "<p style='text-align: center; color: #666; padding: 40px;'>데이터 처리 후 요금제 목록이 여기에 표시됩니다.</p>"
    else:
        # Priority 1: Use pre-calculated charts data from file storage
        if charts_data:
            feature_frontier_data = charts_data.get('feature_frontier', {})
            plan_efficiency_data = charts_data.get('plan_efficiency')
            logger.info("Using pre-calculated charts data from file storage")
        else:
            # Priority 2: Calculate on demand if no pre-calculated data available
            logger.info("Calculating charts data on demand...")
            from ..config import CORE_FEATURES
            core_continuous_features = CORE_FEATURES
            
            feature_frontier_data, all_chart_data, visual_frontiers_for_residual_table = prepare_feature_frontier_data(df, core_continuous_features)
            
            # Prepare Plan Value Efficiency Matrix data
            plan_efficiency_data = prepare_plan_efficiency_data(df_sorted, method)
        
        # Generate feature rates table HTML
        feature_rates_table_html = generate_feature_rates_table_html(cost_structure)
        
        # Generate table HTML
        all_plans_html = generate_all_plans_table_html(df_sorted)
    
    # Convert to JSON for JavaScript
    feature_frontier_json = json.dumps(feature_frontier_data, cls=NumpyEncoder)
    plan_efficiency_json = json.dumps(plan_efficiency_data, cls=NumpyEncoder)
    
    # Prepare chart status variables
    feature_frontier_status_html = get_chart_status_html('feature_frontier', 'featureCharts', df, charts_data, chart_statuses)
    plan_efficiency_status_html = get_chart_status_html('plan_efficiency', 'planEfficiencyChart', df, charts_data, chart_statuses)
    
    # Determine display styles based on chart status
    feature_frontier_display_style = "display:none;" if feature_frontier_status_html else ""
    plan_efficiency_display_style = "display:none;" if plan_efficiency_status_html else ""
    summary_display_style = "display:none;" if no_data_message else ""
    
    # Get template components
    css_styles = get_main_css_styles()
    javascript_code = get_chart_javascript()
    
    # Replace JavaScript placeholders with actual data
    javascript_code = javascript_code.replace('__FEATURE_FRONTIER_JSON__', feature_frontier_json)
    javascript_code = javascript_code.replace('__PLAN_EFFICIENCY_JSON__', plan_efficiency_json)
    javascript_code = javascript_code.replace('__ADVANCED_ANALYSIS_JSON__', 'null')
    javascript_code = javascript_code.replace('__MARGINAL_COST_FRONTIER_JSON__', 'null')
    
    # Get main HTML template
    html_template = get_main_html_template()
    
    # Format the template with all variables
    html = html_template.format(
        report_title=report_title,
        css_styles=css_styles,
        timestamp_str=timestamp_str,
        no_data_message=no_data_message,
        summary_display_style=summary_display_style,
        len_df_sorted=len_df_sorted,
        avg_cs=avg_cs,
        high_cs_count=high_cs_count,
        high_cs_pct=high_cs_pct,
        low_cs_count=low_cs_count,
        low_cs_pct=low_cs_pct,
        method_info_html=method_info_html,
        comparison_info_html=comparison_info_html,
        multi_frontier_chart_html=multi_frontier_chart_html,
        feature_frontier_status_html=feature_frontier_status_html,
        feature_frontier_display_style=feature_frontier_display_style,
        plan_efficiency_status_html=plan_efficiency_status_html,
        plan_efficiency_display_style=plan_efficiency_display_style,
        feature_rates_table_html=feature_rates_table_html,
        all_plans_html=all_plans_html,
        javascript_code=javascript_code
    )
    
    return html 