"""
Report Tables Module

This module handles the generation of HTML tables for the report.
"""

import logging
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def generate_all_plans_table_html(df_sorted):
    """
    Generate HTML for the complete plan rankings table.
    
    Args:
        df_sorted: DataFrame with sorted ranking data
        
    Returns:
        HTML string for the table
    """
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
    
    # Determine the model's features for display
    model_features = []
    if len(df_sorted) > 0:
        first_row = df_sorted.iloc[0]
        # Column order: network, basic data, tethering, throttle speed, daily data, voice, message
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
    
    # Combine all headers into one table
    all_headers = ["No", "요금제명", "MVNO", "원가", "할인가", "CS값 (가성비)"]
    feature_headers = [simplified_headers.get(feature, feature) for feature in model_features]
    all_headers.extend(feature_headers)
    
    # Start table
    table_html = ['<table>']
    
    # Table header
    table_html.append('<thead><tr>')
    for header in all_headers:
        align = "left" if header in ["요금제명", "MVNO", "네트워크", "데이터 정책"] else "right"
        table_html.append(f'<th align="{align}">{header}</th>')
    table_html.append('</tr></thead>')
    
    # Table body
    table_html.append('<tbody>')
    
    # Add all plans to the table
    for _, row in df_sorted.iterrows():
        table_html.append('<tr>')
        
        # Basic plan information
        plan_name = row.get('plan_name', "Plan")
        # Add ID in parentheses after plan name if available
        plan_name = f"{shorten_plan_name(plan_name, 30)}<br>({row['id']})" if 'id' in row else shorten_plan_name(plan_name, 30)
        
        # Format price values with commas and add units
        original_price = f"{format_number_with_commas(row.get('original_price', row.get('original_fee', 0)))}원"
        discounted_price = f"{format_number_with_commas(row.get('discounted_price', row.get('fee', 0)))}원"
        
        # Format CS ratio with arrow indicator
        cs_ratio = row.get('CS', 0)
        cs_arrow = "↑" if cs_ratio > 1 else "↓" if cs_ratio < 1 else "→"
        cs_class = "highlight-high" if cs_ratio > 1 else ("highlight-low" if cs_ratio < 1 else "")
        
        # Format CS value display
        if cs_ratio == float('inf'):
            cs_with_ratio = f"{format_number_with_commas(row.get('B', 0))}원<br>(<span class='{cs_class}'>∞배 {cs_arrow}</span>)"
        else:
            cs_with_ratio = f"{format_number_with_commas(row.get('B', 0))}원<br>(<span class='{cs_class}'>{cs_ratio:.2f}배 {cs_arrow}</span>)"
        
        # Basic cells
        cells = [
            str(row.get('rank_display', row.get('rank_number', 'N/A'))),
            plan_name,
            row.get('mvno', 'N/A'),
            original_price, 
            discounted_price, 
            cs_with_ratio
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
            align = "left" if idx in [1, 2] or (idx >= 6 and idx in [6, 9]) else "right"
            table_html.append(f'<td align="{align}">{cell}</td>')
            
        table_html.append('</tr>')
    
    # Close table elements
    table_html.append('</tbody>')
    table_html.append('</table>')
    
    return "\n".join(table_html)

def generate_residual_analysis_table_html(residual_analysis_data):
    """
    Generate HTML for the residual fee analysis table.
    
    Args:
        residual_analysis_data: List of dictionaries with residual analysis data
        
    Returns:
        HTML string for the table
    """
    residual_table_html = ""
    
    # Generate the residual analysis table rows
    for row_data in residual_analysis_data:
        residual_table_html += f"""
            <tr>
                <td>{row_data['analyzed_feature_display']}</td>
                <td>{row_data['target_plan_name']}</td>
                <td>{row_data['plan_specs_string']}</td>
                <td>{row_data['fee_breakdown_string']}</td>
            </tr>
        """
    
    # If no data is available, add an empty row with an explanation
    if not residual_table_html:
        residual_table_html = """
            <tr>
                <td colspan="4" style="text-align: center;">No feature frontier data available for residual analysis.</td>
            </tr>
        """
    
    return residual_table_html 