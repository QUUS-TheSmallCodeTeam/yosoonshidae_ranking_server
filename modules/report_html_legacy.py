"""
Report HTML Module (Refactored)

This module now imports from the modularized report components.
All HTML generation logic has been moved to dedicated modules:
- templates/: HTML templates, CSS styles, JavaScript code
- report/: Report generation functions by responsibility
"""

# Import the main function from the new modular structure
from .report.html_generator import generate_html_report
from .report.chart_data import prepare_cost_structure_chart_data, prepare_plan_efficiency_data
from .report.tables import generate_feature_rates_table_html

# Maintain backward compatibility by exposing the same interface
__all__ = [
    'generate_html_report',
    'prepare_cost_structure_chart_data', 
    'prepare_plan_efficiency_data',
    'generate_feature_rates_table_html'
]
