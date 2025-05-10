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

def generate_html_report(df, timestamp):
    """Generate an HTML report of the rankings.
    
    Args:
        df: DataFrame with ranking data
        timestamp: Timestamp for the report
        
    Returns:
        HTML content as string
    """
    # Determine if this is a DEA or Spearman report based on columns
    is_dea = 'efficiency_score' in df.columns or 'dea_score' in df.columns
    
    # Get ranking method and log transform from the dataframe attributes if available
    ranking_method = df.attrs.get('ranking_method', 'relative')
    use_log_transform = df.attrs.get('use_log_transform', False)
    
    # Get the features used for calculation
    used_features = df.attrs.get('used_features', [])
    
    # Set report title based on method but keep consistent for user
    report_title = "Mobile Plan Rankings"
    
    # Get current timestamp
    timestamp_str = timestamp.strftime("%Y-%m-%d %H:%M:%S")
    
    # Create HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title} - {timestamp_str}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background-color: #f2f2f2; position: sticky; top: 0; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .good-value {{ color: green; }}
            .bad-value {{ color: red; }}
            .container {{ max-width: 100%; overflow-x: auto; }}
            .note {{ background-color: #f8f9fa; padding: 10px; border-left: 4px solid #007bff; margin-bottom: 20px; }}
            .button-group {{ margin-bottom: 15px; }}
            button {{ padding: 10px 15px; background-color: #007bff; color: white; border: none; 
                     border-radius: 4px; cursor: pointer; margin-right: 10px; margin-bottom: 10px; }}
            button:hover {{ background-color: #0056b3; }}
            button.active {{ background-color: #28a745; }}
            .hidden {{ display: none; }}
        </style>
    </head>
    <body>
        <h1>Mobile Plan Rankings</h1>
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
    
    # Get feature weights
    html += """
        <h2>Feature Weights</h2>
        <div id="feature-weights-container">
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
    
    # Add main data table
    html += """
        <h2>Plan Rankings</h2>
        <div class="container" id="main-table-container">
    """
    
    # Table header with appropriate value column name
    value_column_name = "DEA Score" if is_dea else "Value Ratio"
    
    html += f"""
        <table>
            <tr>
                <th>Rank</th>
                <th>Plan Name</th>
                <th>Operator</th>
                <th>Original Price</th>
                <th>Discounted Price</th>
                <th>{value_column_name}</th>
                <th>Data</th>
                <th>Voice</th>
                <th>Message</th>
                <th>Network</th>
            </tr>
    """
    
    # Add headers for all features used in the calculation
    for feature in used_features:
        # Clean up feature name for display
        display_name = feature.replace('_clean', '').replace('_', ' ').title()
        html += f"<th>{display_name}</th>"
    
    html += """
            </tr>
    """
    
    # Determine which rank column to use
    rank_column = None
    for possible_column in ['rank', 'dea_rank', 'rank_with_ties']:
        if possible_column in df.columns:
            rank_column = possible_column
            break
            
    if rank_column is None:
        # If no rank column is found, create a simple rank based on index
        logger.warning("No rank column found in data, creating a simple rank")
        df['temp_rank'] = range(1, len(df) + 1)
        rank_column = 'temp_rank'
    
    logger.info(f"Using rank column: {rank_column}")
    
    # Add rows for each plan
    for i, (_, row) in enumerate(df.sort_values(rank_column).iterrows()):
        plan_name = str(row.get('plan_name', f"Plan {row.get('id', i)}"))
        if len(plan_name) > 30:
            plan_name = plan_name[:27] + "..."
            
        original_fee = f"{int(row.get('original_fee', 0)):,}"
        discounted_fee = f"{int(row.get('fee', 0)):,}"
        predicted_price = f"{int(row.get('predicted_price', 0)):,}"
        
        # Value ratio or efficiency score for DEA
        if is_dea:
            # For DEA, use efficiency or score metrics
            if 'dea_score' in row:
                value_ratio = row.get('dea_score', 0)
                value_name = "DEA Score"
            elif 'dea_efficiency' in row:
                value_ratio = 1.0 / row.get('dea_efficiency', 1.0)  # Convert efficiency to score
                value_name = "DEA Score"
            else:
                value_ratio = 0
                value_name = "Value"
        else:
            # For Spearman, use value ratio
            value_ratio = row.get('value_ratio_original', row.get('value_ratio', 0))
            value_name = "Value Ratio"
            
        if pd.isna(value_ratio):
            value_ratio_str = "N/A"
            value_class = ""
        else:
            value_ratio_str = f"{value_ratio:.2f}"
            # For DEA, higher score is better
            # For Spearman, ratio > 1 is good, < 1 is bad
            if is_dea:
                value_class = "good-value" if value_ratio > 0.9 else ("bad-value" if value_ratio < 0.5 else "")
            else:
                value_class = "good-value" if value_ratio > 1.1 else ("bad-value" if value_ratio < 0.9 else "")
            
        operator = row.get('mvno', "Unknown")
        
        # Get the rank value using the determined rank column
        if rank_column in row:
            rank_display = row[rank_column]
        else:
            rank_display = i+1
            
        # Format rank display - if it's a number, add "위" (Korean for "rank")
        if isinstance(rank_display, (int, float)):
            rank_display = f"{int(rank_display)}위"
        
        # For DEA reports, we don't need predicted price column
        # Get data, voice, and message values
        basic_data = row.get('basic_data_clean', row.get('basic_data', 0))
        if pd.isna(basic_data):
            basic_data = 0
        if row.get('basic_data_unlimited', 0) == 1:
            data_str = "무제한"
        else:
            data_str = f"{basic_data}GB"
            
        voice = row.get('voice_clean', row.get('voice', 0))
        if pd.isna(voice):
            voice = 0
        if row.get('voice_unlimited', 0) == 1:
            voice_str = "무제한"
        else:
            voice_str = f"{int(voice)}분"
            
        message = row.get('message_clean', row.get('message', 0))
        if pd.isna(message):
            message = 0
        if row.get('message_unlimited', 0) == 1:
            message_str = "무제한"
        else:
            message_str = f"{int(message)}건"
            
        network = row.get('network', "")
        if network == "5G":
            network = "5G"
        elif network == "LTE":
            network = "LTE"
        else:
            network = ""
            
        html += f"""
        <tr>
            <td>{rank_display}</td>
            <td>{plan_name}</td>
            <td>{operator}</td>
            <td>{original_fee}</td>
            <td>{discounted_fee}</td>
            <td class="{value_class}">{value_ratio_str}</td>
            <td>{data_str}</td>
            <td>{voice_str}</td>
            <td>{message_str}</td>
            <td>{network}</td>
        """
        
        # Add values for all features
        for feature in used_features:
            if feature in row:
                # Format the feature value based on its type
                if isinstance(row[feature], bool):
                    value = "Yes" if row[feature] else "No"
                elif isinstance(row[feature], (int, float)):
                    if feature in ['is_5g', 'basic_data_unlimited', 'daily_data_unlimited', 'voice_unlimited', 'message_unlimited']:
                        value = "Yes" if row[feature] == 1 else "No"
                    elif feature == 'unlimited_type_numeric':
                        # Map unlimited type numeric to descriptive text
                        unlimited_types = {
                            0: "Limited",
                            1: "Throttled",
                            2: "Throttled+",
                            3: "Unlimited"
                        }
                        value = unlimited_types.get(row[feature], str(row[feature]))
                    else:
                        # Format with commas if it's a whole number
                        if row[feature] == int(row[feature]):
                            value = f"{int(row[feature]):,}"
                        else:
                            value = f"{row[feature]:.2f}"
                else:
                    value = str(row[feature])
                html += f"<td>{value}</td>"
            else:
                html += "<td>N/A</td>"
        
        html += "</tr>"
    
    html += """
        </table>
        </div>
    """
    
    # Add JavaScript for interactive controls
    html += """
    <script>
    /* Current state */
    let currentState = {
        rankMethod: "relative",
        feeType: "original",
        logTransform: true
    };
    
    /* Store all table containers */
    let tableContainers = {};
    
    /* Initialize on page load */
    document.addEventListener('DOMContentLoaded', function() {
        /* Find the main table in the document */
        const mainTable = document.getElementById('main-table');
        if (!mainTable) return;
        
        /* Create container divs for different views if they don't exist */
        createTableContainers();
        
        /* Set up initial view */
        setTimeout(function() {
            updateVisibleContainer();
        }, 200);
    });
    
    /* Create containers for different ranking views */
    function createTableContainers() {
        /* Clone the table for each ranking method and fee type */
        const rankMethods = ['relative', 'absolute', 'net'];
        const feeTypes = ['original', 'discounted'];
        
        /* Get the parent of the main table */
        const mainTableContainer = document.getElementById('main-table-container');
        
        /* Create container for all tables */
        const container = document.createElement('div');
        container.className = 'rankings-container';
        mainTableContainer.parentNode.insertBefore(container, mainTableContainer);
        
        /* Hide the original table */
        mainTableContainer.style.display = 'none';
        
        /* For each combination, create a container with a cloned table */
        rankMethods.forEach(method => {
            feeTypes.forEach(feeType => {
                const containerId = `${method}-${feeType}`;
                const newContainer = document.createElement('div');
                newContainer.id = containerId;
                newContainer.className = 'container hidden';
                newContainer.innerHTML = mainTableContainer.innerHTML;
                container.appendChild(newContainer);
                
                tableContainers[containerId] = newContainer;
            });
        });
        
        /* Set the default view to visible */
        const defaultContainer = document.getElementById('relative-original');
        if (defaultContainer) {
            defaultContainer.classList.remove('hidden');
        }
    }
    
    /* Change ranking method */
    function changeRankMethod(method) {
        /* Update buttons */
        document.getElementById('relative-btn').classList.remove('active');
        document.getElementById('absolute-btn').classList.remove('active');
        document.getElementById('net-btn').classList.remove('active');
        
        /* Update button styles */
        document.getElementById(method + '-btn').classList.add('active');
        
        /* Update state */
        currentState.rankMethod = method;
        
        /* Update visible container */
        updateVisibleContainer();
    }
    
    /* Change fee type */
    function changeFeeType(type) {
        /* Update buttons */
        document.getElementById('original-fee-btn').classList.remove('active');
        document.getElementById('discounted-fee-btn').classList.remove('active');
        
        /* Update button styles */
        document.getElementById(type + '-fee-btn').classList.add('active');
        
        /* Update state */
        currentState.feeType = type;
        
        /* Update visible container */
        updateVisibleContainer();
    }
    
    /* Toggle log transform */
    function toggleLogTransform(enabled) {
        document.getElementById('log-transform-on-btn').classList.remove('active');
        document.getElementById('log-transform-off-btn').classList.remove('active');
        
        if (enabled) {
            document.getElementById('log-transform-on-btn').classList.add('active');
        } else {
            document.getElementById('log-transform-off-btn').classList.add('active');
        }
        
        currentState.logTransform = enabled;
        
        // Note: In a real implementation, this would trigger a recalculation
        alert("Changing log transform would require recalculating all rankings. This feature is shown for UI demonstration only.");
    }
    
    /* Update visible container based on current state */
    function updateVisibleContainer() {
        /* Hide all containers */
        const containers = document.querySelectorAll('.rankings-container .container');
        containers.forEach(container => {
            container.classList.add('hidden');
        });
        
        /* Show the selected container */
        const containerId = `${currentState.rankMethod}-${currentState.feeType}`;
        const containerElement = document.getElementById(containerId);
        if (containerElement) {
            containerElement.classList.remove('hidden');
        } else {
            /* Fallback to relative-original if the selected container doesn't exist */
            document.getElementById('relative-original').classList.remove('hidden');
            /* Update state and buttons to match */
            currentState.rankMethod = 'relative';
            currentState.feeType = 'original';
            document.getElementById('relative-btn').classList.add('active');
            document.getElementById('original-fee-btn').classList.add('active');
        }
    }
    </script>
    </body>
    </html>
    """
    
    return html

def save_report(html_content, timestamp):
    """Save an HTML report to the reports directory."""
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
    
    # Define report directories
    report_dirs = [Path("./reports"), Path("/tmp/reports")]
    
    # Ensure report directories exist
    saved_paths = []
    for report_dir in report_dirs:
        try:
            os.makedirs(report_dir, exist_ok=True)
            report_path = report_dir / f"plan_rankings_spearman_{timestamp_str}.html"
            
            with open(report_path, "w", encoding="utf-8") as f:
                f.write(html_content)
            logger.info(f"Report saved to {report_path}")
            saved_paths.append(str(report_path))
        except Exception as e:
            logger.error(f"Failed to save report to {report_dir}: {e}")
    
    # If we saved to any location, return the first one
    if saved_paths:
        return saved_paths[0]
    
    # If all save attempts failed, try a fallback location
    try:
        fallback_path = Path(f"/tmp/plan_rankings_spearman_{timestamp_str}.html")
        with open(fallback_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Report saved to fallback location: {fallback_path}")
        return str(fallback_path)
    except Exception as e:
        logger.error(f"Failed to save report to fallback location: {e}")
        return None
