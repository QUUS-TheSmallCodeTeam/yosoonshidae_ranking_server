"""
Templates Module

This module contains HTML templates, CSS styles, and JavaScript code for report generation.
"""

from .main_template import get_main_html_template
from .chart_scripts import get_chart_javascript
from .styles import get_main_css_styles

__all__ = [
    'get_main_html_template',
    'get_chart_javascript', 
    'get_main_css_styles'
] 