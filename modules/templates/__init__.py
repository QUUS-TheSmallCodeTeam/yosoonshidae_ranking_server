"""
Templates Module

This module contains HTML template generation functionality.

Classes and Functions:
- main_template: Main HTML template structure
- styles: CSS styling definitions
- chart_scripts: JavaScript chart functionality (facade)
- cost_structure_charts: Cost structure chart JavaScript
- efficiency_charts: Plan efficiency chart JavaScript

The chart_scripts module has been refactored into a facade pattern using focused sub-modules.
"""

from .main_template import get_main_html_template
from .styles import get_main_css_styles
from .chart_scripts import get_chart_javascript

# Export refactored chart sub-modules
from .cost_structure_charts import get_cost_structure_chart_javascript
from .efficiency_charts import get_efficiency_chart_javascript

__all__ = [
    'get_main_html_template',
    'get_main_css_styles', 
    'get_chart_javascript',
    'get_cost_structure_chart_javascript',
    'get_efficiency_chart_javascript'
] 