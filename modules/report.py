"""
Report Module (Legacy Wrapper)

This module is a lightweight wrapper around the new modularized report components.
It maintains backward compatibility for existing code that imports functions from this module.
"""

import logging
from datetime import datetime
from .report_html import generate_html_report
from .report_utils import save_report

# Configure logging
logger = logging.getLogger(__name__)

# Re-export generate_html_report and save_report for backward compatibility
__all__ = ['generate_html_report', 'save_report']
