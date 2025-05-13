"""
Report Generation Module

This module handles generating and saving HTML reports for the Moyo Ranking Model.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

def generate_html_report(df, timestamp, is_dea=False, is_cs=False, title="Mobile Plan Rankings"):
    """Generate an HTML report of the rankings.
    
    Args:
        df: DataFrame with ranking data
        timestamp: Timestamp for the report
        is_dea: Whether this is a DEA report (default: False)
        is_cs: Whether this is a Cost-Spec report (default: False)
        title: Title for the report (default: "Mobile Plan Rankings")
        
    Returns:
        HTML content as string
    """
    # Get ranking method and log transform from the dataframe attributes if available
    ranking_method = df.attrs.get('ranking_method', 'relative')
    use_log_transform = df.attrs.get('use_log_transform', False)
    
    # Get the features used for calculation
    used_features = df.attrs.get('used_features', [])
    
    # Get current timestamp
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Set report title based on method
    if is_dea:
        report_title = "DEA Mobile Plan Rankings"
    elif is_cs:
        report_title = "Cost-Spec Mobile Plan Rankings"
    else:
        report_title = title
    
    # Create HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title} - {timestamp_str}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; font-size: 14px; }}
            th, td {{ padding: 8px; text-align: left; border: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; z-index: 10; }}
            tr:hover {{ background-color: #f5f5f5; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .good-value {{ color: green; font-weight: bold; }}
            .bad-value {{ color: red; }}
            .container {{ max-width: 100%; overflow-x: auto; }}
            .note {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; margin-bottom: 20px; }}
            
            /* Feature category colors */
            .core-feature {{ background-color: #e6f7ff; }}
            .dea-metrics {{ background-color: #fff0f6; }}
            .cs-metrics {{ background-color: #f9f0ff; }}
            .input-feature {{ background-color: #f9f0ff; }}
            .output-feature {{ background-color: #f6ffed; }}
            
            /* Collapsible sections */
            .collapsible {{ 
                background-color: #f1f1f1;
                color: #444;
                cursor: pointer;
                padding: 18px;
                width: 100%;
                border: none;
                text-align: left;
                outline: none;
                font-size: 15px;
                margin-bottom: 5px;
            }}
            
            .active, .collapsible:hover {{ background-color: #ccc; }}
            
            .content {{ 
                padding: 0 18px;
                display: none;
                overflow: hidden;
                background-color: #f9f9f9;
                margin-bottom: 10px;
            }}
            .button-group {{ margin-bottom: 15px; }}
            button {{ padding: 10px 15px; background-color: #007bff; color: white; border: none; 
                     border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px; }}
            button:hover {{ background-color: #0056b3; }}
            button.active {{ background-color: #28a745; }}
            .hidden {{ display: none; }}
        </style>
    </head>
    <body>
        <h1>{report_title}</h1>
        <p>Generated: {timestamp_str}</p>
        
        <div class="note">
            <strong>Instructions:</strong> Use the buttons below to toggle between different ranking methods,
            fee types, and log transformation options.
        </div>
        
        <h2>Control Panel</h2>
        <div class="button-group">
            <strong>Ranking Method:</strong><br>
            <button id="relative-btn" {"class='active'" if ranking_method == 'relative' else ""} onclick="changeRankMethod('relative')">Relative Value (ΔP/fee)</button>
            <button id="absolute-btn" {"class='active'" if ranking_method == 'absolute' else ""} onclick="changeRankMethod('absolute')">Absolute Value (ΔP)</button>
            <button id="net-btn" {"class='active'" if ranking_method == 'net' else ""} onclick="changeRankMethod('net')">Net Value (ΔP-fee)</button>
        </div>
        
        <div class="button-group">
            <strong>Fee Type:</strong><br>
            <button id="original-fee-btn" class="active" onclick="changeFeeType('original')">Original Fee</button>
            <button id="discounted-fee-btn" onclick="changeFeeType('discounted')">Discounted Fee</button>
        </div>
        
        <div class="button-group">
            <strong>Log Transform:</strong><br>
            <button id="log-transform-on-btn" {"class='active'" if use_log_transform else ""} onclick="toggleLogTransform(true)">On</button>
            <button id="log-transform-off-btn" {"class='active'" if not use_log_transform else ""} onclick="toggleLogTransform(false)">Off</button>
        </div>
    """
    
    # Add method explanation section based on the method
    if is_dea:
        html += """
        <h2>DEA Calculation Explanation</h2>
        <div class="note">
            <p><strong>Data Envelopment Analysis (DEA)</strong> is a method that evaluates the efficiency of decision-making units (in this case, mobile plans).</p>
            <p>In this analysis:</p>
            <ul>
                <li><strong>Input:</strong> Plan price (fee)</li>
                <li><strong>Outputs:</strong> Data, Voice, SMS, and other features</li>
                <li><strong>DEA Efficiency:</strong> A score between 0 and 1, where 1 means the plan is efficient (on the efficiency frontier)</li>
                <li><strong>DEA Score:</strong> For inefficient plans, this is 1/efficiency. For efficient plans, this is the super-efficiency score</li>
                <li><strong>Super-Efficiency:</strong> A score that helps differentiate between efficient plans (higher is better)</li>
            </ul>
            <p>Plans are ranked based on their DEA Score (higher is better).</p>
        </div>
        """
    elif is_cs:
        html += """
        <h2>Cost-Spec Ratio Explanation</h2>
        <div class="note">
            <p><strong>Cost-Spec Ratio (CS)</strong> is a method that evaluates the value of mobile plans by comparing their fees to a theoretical baseline cost.</p>
            <p>In this analysis:</p>
            <ul>
                <li><strong>Baseline Feature Cost (E):</strong> For each feature value, the minimum fee among plans with that value</li>
                <li><strong>Plan Baseline Cost (B):</strong> Sum of baseline costs for all features in a plan</li>
                <li><strong>Cost-Spec Ratio (CS):</strong> B / fee - the ratio of theoretical cost to actual fee</li>
            </ul>
            <p>Plans are ranked based on their CS Ratio (higher is better).</p>
        </div>
        """
    
    # Add feature weights section
    html += """
        <h2>Feature Weights</h2>
        <div class="container">
        <table>
            <tr>
                <th>Feature</th>
                <th>Weight</th>
                <th>Average Contribution (KRW)</th>
            </tr>
    """
    
    # Get the feature weights from dataframe attributes
    weights = df.attrs.get('feature_weights', {})
    
    # Get contribution columns
    contribution_cols = [col for col in df.columns if col.startswith("contribution_")]
    
    # Sort contribution columns by average contribution (descending)
    sorted_contribution_cols = sorted(
        contribution_cols,
        key=lambda x: df[x].mean() if not pd.isna(df[x].mean()) else -float('inf'),
        reverse=True
    )
    
    for col in sorted_contribution_cols:
        feature_name = col.replace("contribution_", "")
        avg_contrib = df[col].mean()
        
        # Get the corresponding weight for this feature
        feature_weight = weights.get(feature_name, float('nan'))
        
        if pd.isna(avg_contrib):
            if pd.isna(feature_weight):
                html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>N/A</td>
            <td>N/A</td>
        </tr>
        """
            else:
                html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>{feature_weight:.4f}</td>
            <td>N/A</td>
        </tr>
        """
        else:
            if pd.isna(feature_weight):
                html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>N/A</td>
            <td>{int(avg_contrib):,} KRW</td>
        </tr>
        """
            else:
                html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>{feature_weight:.4f}</td>
            <td>{int(avg_contrib):,} KRW</td>
        </tr>
        """
    
    html += """
        </table>
        </div>
    """
    
    # Add Features List
    if used_features:
        html += """
        <h2>Features Used</h2>
        <div class="container">
        <table>
            <tr>
                <th>Feature</th>
                <th>Category</th>
            </tr>
        """
        
        for feature in used_features:
            category = "Output" if feature != "fee" else "Input"
            html += f"""
            <tr>
                <td>{feature}</td>
                <td>{category}</td>
            </tr>
            """
        
        html += """
        </table>
        </div>
        """
    
    # Add rankings table
    html += """
        <h2>Plan Rankings</h2>
        <div class="container">
        <table>
            <tr>
                <th>Rank</th>
                <th>Plan Name</th>
                <th>MVNO</th>
                <th>Fee (KRW)</th>
                <th>Original Fee (KRW)</th>
    """
    
    # Add method-specific columns
    if is_dea:
        html += """
                <th>DEA Efficiency</th>
                <th>DEA Score</th>
        """
    elif is_cs:
        html += """
                <th>Baseline Cost (B)</th>
                <th>CS Ratio</th>
        """
    else:
        html += """
                <th>Value Ratio</th>
                <th>Delta P</th>
                <th>Delta P - Fee</th>
        """
    
    # Continue with feature columns
    html += """
                <th>Data (GB)</th>
                <th>Voice (min)</th>
                <th>Message (SMS)</th>
                <th>Additional Call (min)</th>
                <th>5G</th>
            </tr>
    """
    
    # Generate table rows
    rank_col = 'dea_rank' if is_dea else ('rank_number' if is_cs else 'rank')
    value_col = 'dea_score' if is_dea else ('CS' if is_cs else 'value_ratio')
    
    # Sort by the appropriate ranking column
    if rank_col in df.columns:
        df_sorted = df.sort_values(rank_col)
    else:
        df_sorted = df.sort_values('rank', ascending=True) if 'rank' in df.columns else df
    
    for _, row in df_sorted.iterrows():
        # Format rank
        if is_dea:
            rank = int(row['dea_rank']) if 'dea_rank' in row and not pd.isna(row['dea_rank']) else ""
            rank_str = f"{rank}" if rank else ""
        elif is_cs:
            rank = int(row['rank_number']) if 'rank_number' in row and not pd.isna(row['rank_number']) else ""
            rank_str = f"{rank}" if rank else ""
        else:
            rank_str = row['rank_display'] if 'rank_display' in row else ""
        
        # Get plan data
        plan_name = row['plan_name'] if 'plan_name' in row else ""
        mvno = row['mvno'] if 'mvno' in row else ""
        fee = int(row['fee']) if 'fee' in row and not pd.isna(row['fee']) else 0
        original_fee = int(row['original_fee']) if 'original_fee' in row and not pd.isna(row['original_fee']) else 0
        
        # Method-specific metrics
        if is_dea:
            dea_efficiency = row['dea_efficiency'] if 'dea_efficiency' in row else ""
            dea_score = row['dea_score'] if 'dea_score' in row else ""
            
            method_specific_cols = f"""
                <td>{dea_efficiency:.4f if isinstance(dea_efficiency, float) else dea_efficiency}</td>
                <td class="good-value">{dea_score:.4f if isinstance(dea_score, float) else dea_score}</td>
            """
        elif is_cs:
            baseline_cost = int(row['B']) if 'B' in row and not pd.isna(row['B']) else 0
            cs_ratio = row['CS'] if 'CS' in row else ""
            
            method_specific_cols = f"""
                <td>{baseline_cost:,}</td>
                <td class="good-value">{cs_ratio:.4f if isinstance(cs_ratio, float) else cs_ratio}</td>
            """
        else:
            value_ratio = row['value_ratio'] if 'value_ratio' in row else ""
            delta_p = row.get('delta_p', "")
            delta_p_minus_fee = row.get('delta_p_minus_fee', "")
            
            # Format delta_p and delta_p_minus_fee
            delta_p_str = f"{int(delta_p):,}" if isinstance(delta_p, (int, float)) else str(delta_p)
            delta_p_minus_fee_str = f"{int(delta_p_minus_fee):,}" if isinstance(delta_p_minus_fee, (int, float)) else str(delta_p_minus_fee)
            
            method_specific_cols = f"""
                <td class="good-value">{value_ratio:.4f if isinstance(value_ratio, float) else value_ratio}</td>
                <td>{delta_p_str}</td>
                <td>{delta_p_minus_fee_str}</td>
            """
        
        # Get feature data
        data_gb = row['basic_data_clean'] if 'basic_data_clean' in row else "N/A"
        voice = row['voice_clean'] if 'voice_clean' in row else "N/A"
        message = row['message_clean'] if 'message_clean' in row else "N/A"
        additional_call = row['additional_call'] if 'additional_call' in row else "N/A"
        is_5g = "Yes" if row.get('is_5g') == 1 else "No"
        
        # Handle unlimited values
        if 'basic_data_unlimited' in row and row['basic_data_unlimited'] == 1:
            data_gb = "Unlimited"
        if 'voice_unlimited' in row and row['voice_unlimited'] == 1:
            voice = "Unlimited"
        if 'message_unlimited' in row and row['message_unlimited'] == 1:
            message = "Unlimited"
        
        # Generate the row HTML
        html += f"""
            <tr>
                <td>{rank_str}</td>
                <td>{plan_name}</td>
                <td>{mvno}</td>
                <td>{fee:,}</td>
                <td>{original_fee:,}</td>
                {method_specific_cols}
                <td>{data_gb}</td>
                <td>{voice}</td>
                <td>{message}</td>
                <td>{additional_call}</td>
                <td>{is_5g}</td>
            </tr>
        """
    
    # Close the table and add collapsible details
    html += """
        </table>
        </div>
        
        <h2>Detailed Plan Information</h2>
    """
    
    # Add collapsible sections for each plan
    for _, row in df_sorted.iterrows():
        plan_name = row['plan_name'] if 'plan_name' in row else "Unknown Plan"
        mvno = row['mvno'] if 'mvno' in row else "Unknown Provider"
        
        html += f"""
        <button type="button" class="collapsible">{plan_name} - {mvno}</button>
        <div class="content">
            <table>
                <tr>
                    <th>Feature</th>
                    <th>Value</th>
                </tr>
        """
        
        # Add all features and values to the table
        for col in sorted(row.index):
            # Skip null values and internal columns
            if col in ('index', 'level_0') or pd.isna(row[col]):
                continue
            
            # Format value based on type
            if isinstance(row[col], (int, float)):
                if col.endswith('unlimited') or col.startswith('is_') or col.startswith('has_'):
                    # Boolean-like values
                    value = "Yes" if row[col] == 1 else "No"
                elif col in ('fee', 'original_fee', 'post_discount_fee', 'discount_fee'):
                    # Currency values
                    value = f"{int(row[col]):,} KRW"
                else:
                    # Regular numeric values
                    value = f"{row[col]}"
            else:
                value = str(row[col])
            
            html += f"""
                <tr>
                    <td>{col}</td>
                    <td>{value}</td>
                </tr>
            """
        
        html += """
            </table>
        </div>
        """
    
    # Add JavaScript for interactive elements
    html += """
        <script>
        /* Add collapsible functionality */
        var coll = document.getElementsByClassName("collapsible");
        var i;

        for (i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }
        
        /* Current state */
        let currentState = {
            rankMethod: "relative",
            feeType: "original",
            logTransform: true
        };
        
        /* Change ranking method */
        function changeRankMethod(method) {
            /* Update buttons */
            document.getElementById('relative-btn').classList.remove('active');
            document.getElementById('absolute-btn').classList.remove('active');
            document.getElementById('net-btn').classList.remove('active');
            document.getElementById(method + '-btn').classList.add('active');
            
            /* Update button styles */
            document.getElementById('relative-btn').style.backgroundColor = '#007bff';
            document.getElementById('absolute-btn').style.backgroundColor = '#007bff';
            document.getElementById('net-btn').style.backgroundColor = '#007bff';
            document.getElementById(method + '-btn').style.backgroundColor = '#28a745';
            
            /* Update state */
            currentState.rankMethod = method;
            console.log("Ranking method changed to: " + method);
        }
        
        /* Change fee type */
        function changeFeeType(type) {
            /* Update buttons */
            document.getElementById('original-fee-btn').classList.remove('active');
            document.getElementById('discounted-fee-btn').classList.remove('active');
            document.getElementById(type + '-fee-btn').classList.add('active');
            
            /* Update button styles */
            document.getElementById('original-fee-btn').style.backgroundColor = '#007bff';
            document.getElementById('discounted-fee-btn').style.backgroundColor = '#007bff';
            document.getElementById(type + '-fee-btn').style.backgroundColor = '#28a745';
            
            /* Update state */
            currentState.feeType = type;
            console.log("Fee type changed to: " + type);
        }
        
        /* Toggle log transform */
        function toggleLogTransform(enabled) {
            /* Update buttons */
            document.getElementById('log-transform-on-btn').classList.remove('active');
            document.getElementById('log-transform-off-btn').classList.remove('active');
            
            if (enabled) {
                document.getElementById('log-transform-on-btn').classList.add('active');
                document.getElementById('log-transform-on-btn').style.backgroundColor = '#28a745';
                document.getElementById('log-transform-off-btn').style.backgroundColor = '#007bff';
            } else {
                document.getElementById('log-transform-off-btn').classList.add('active');
                document.getElementById('log-transform-on-btn').style.backgroundColor = '#007bff';
                document.getElementById('log-transform-off-btn').style.backgroundColor = '#28a745';
            }
            
            /* Update state */
            currentState.logTransform = enabled;
            console.log("Log transform set to: " + enabled);
        }
        </script>
    </body>
    </html>
    """
    
    return html

def save_report(html_content, timestamp, directory=None, prefix="ranking", description=None):
    """Save an HTML report to a file.
    
    Args:
        html_content: HTML content as string
        timestamp: Timestamp to use in filename
        directory: Directory to save to (optional)
        prefix: Prefix for the filename (default: "ranking")
        description: Optional description to include in filename
    
    Returns:
        Path object of the saved file
    """
    # Generate filename with timestamp
    filename_parts = [prefix]
    if description:
        filename_parts.append(description)
    
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    filename = f"{'-'.join(filename_parts)}_{timestamp_str}.html"
    
    # Determine directory
    if directory is None:
        directory = Path("./reports")
    
    # Create directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # Create file path
    file_path = Path(directory) / filename
    
    # Write content to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Report saved to {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Error saving report: {e}")
        return None
