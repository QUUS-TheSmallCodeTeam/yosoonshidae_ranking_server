"""
Report Module

This module contains report generation functions separated by responsibility.
"""

from .html_generator import generate_html_report
from .status import get_chart_status_html
from .chart_data import prepare_cost_structure_chart_data, prepare_plan_efficiency_data
from .tables import generate_feature_rates_table_html

__all__ = [
    'generate_html_report',
    'get_chart_status_html',
    'prepare_cost_structure_chart_data',
    'prepare_plan_efficiency_data', 
    'generate_feature_rates_table_html'
] 