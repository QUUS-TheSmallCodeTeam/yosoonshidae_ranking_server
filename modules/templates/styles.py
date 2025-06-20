"""
CSS Styles Module

This module contains all CSS styles for the HTML report.
"""

def get_main_css_styles():
    """
    Get the main CSS styles for the HTML report.
    
    Returns:
        str: Complete CSS stylesheet as string
    """
    return """
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                color: #333;
                line-height: 1.6;
            }
            h1, h2, h3 {
                color: #2c3e50;
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
            .good-value {
                color: #27ae60;
                font-weight: bold;
            }
            .bad-value {
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
            .note {
                background-color: #f8f9fa;
                padding: 10px;
                border-left: 4px solid #007bff;
                margin-bottom: 20px;
            }
            
            /* Content wrapper with padding */
            .content-wrapper {
                padding: 20px;
            }
            
            /* Feature charts wrapper - no padding for full width */
            .charts-wrapper {
                width: 100%;
            }
            
            /* Feature charts grid */
            .chart-grid {
                display: grid;
                grid-template-columns: 1fr;
                gap: 20px;
                width: 100%;
            }
            
            .chart-container {
                background-color: #fff;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 15px;
                position: relative;
                width: 100%;
                height: 400px;
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
    """ 