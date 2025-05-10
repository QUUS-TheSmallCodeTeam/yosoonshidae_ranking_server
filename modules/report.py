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

def generate_html_report(df, timestamp, is_dea=False, title="Mobile Plan Rankings"):
    """Generate an HTML report of the rankings.
    
    Args:
        df: DataFrame with ranking data
        timestamp: Timestamp for the report
        is_dea: Whether this is a DEA report (default: False)
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
    report_title = "DEA Mobile Plan Rankings" if is_dea else title
    
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
        <h1>Mobile Plan Rankings (Spearman Method)</h1>
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
    
    # Add DEA explanation section if this is a DEA report
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
    
    # Add main data table
    if is_dea:
        html += """
        <h2>DEA Plan Rankings</h2>
        <div class="container">
        <table id="rankings-table">
            <tr>
                <th>Rank</th>
                <th>Plan Name</th>
                <th>Provider</th>
                <th class="input-feature">Fee (Input)</th>
                <th class="dea-metrics">DEA Score</th>
                <th class="dea-metrics">Efficiency</th>
                <th class="dea-metrics">Super-Efficiency</th>
                
                <!-- Core output features used in DEA calculation -->
                <th class="output-feature">Data (GB)</th>
                <th class="output-feature">Voice (Min)</th>
                <th class="output-feature">SMS</th>
                <th class="core-feature">Network</th>
        """
    else:
        html += """
        <h2>Plan Rankings</h2>
        <div class="container" id="main-table-container">
        <table id="main-table">
            <tr>
                <th>Rank</th>
                <th>Plan Name</th>
                <th>Operator</th>
                <th>Original Fee</th>
                <th>Discounted Fee</th>
                <th>Worth Estimate</th>
                <th>Value Ratio</th>
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
    
    # Check for available rank columns based on the method
    if is_dea:
        # For DEA, first check for dea-specific rank columns
        for possible_column in ['dea_rank', 'dea_efficiency_rank', 'rank']:
            if possible_column in df.columns:
                rank_column = possible_column
                break
    else:
        # For Spearman, look for standard rank columns
        for possible_column in ['rank', 'rank_with_ties']:
            if possible_column in df.columns:
                rank_column = possible_column
                break
            
    if rank_column is None:
        # If no rank column is found, create a simple rank based on index or efficiency
        logger.warning("No rank column found in data, creating a temporary rank")
        
        if is_dea and 'dea_efficiency' in df.columns:
            # For DEA, sort by efficiency (descending) if available
            logger.info("Creating temporary rank based on DEA efficiency")
            df = df.sort_values('dea_efficiency', ascending=False)
            df['temp_rank'] = range(1, len(df) + 1)
        elif is_dea and 'dea_score' in df.columns:
            # For DEA, sort by score (descending) if available
            logger.info("Creating temporary rank based on DEA score")
            df = df.sort_values('dea_score', ascending=False)
            df['temp_rank'] = range(1, len(df) + 1)
        elif not is_dea and 'value_ratio' in df.columns:
            # For Spearman, sort by value ratio (descending) if available
            logger.info("Creating temporary rank based on value ratio")
            df = df.sort_values('value_ratio', ascending=False)
            df['temp_rank'] = range(1, len(df) + 1)
        else:
            # Fallback to simple index-based rank
            logger.info("Creating simple index-based rank")
            df['temp_rank'] = range(1, len(df) + 1)
            
        rank_column = 'temp_rank'
    
    logger.info(f"Using rank column: {rank_column}")
    
    # Add rows for each plan
    for i, (_, row) in enumerate(df.sort_values(rank_column).iterrows()):
        plan_name = str(row.get('plan_name', f"Plan {row.get('id', i)}"))
        if len(plan_name) > 30:
            plan_name = plan_name[:27] + "..."
        
        # Get the rank value using the determined rank column
        if rank_column in row:
            rank_display = row[rank_column]
        else:
            rank_display = i+1
            
        # Format rank display - if it's a number, add "위" (Korean for "rank")
        if isinstance(rank_display, (int, float)):
            rank_display = f"{int(rank_display)}위"
        
        if is_dea:
            # DEA specific values
            fee = int(row.get('fee', 0))
            fee_str = f"{fee:,}"
            
            # DEA metrics
            efficiency = row.get('dea_efficiency', 0)
            if pd.isna(efficiency):
                efficiency_str = "N/A"
            else:
                efficiency_str = f"{efficiency:.4f}"
                
            dea_score = row.get('dea_score', 0)
            if pd.isna(dea_score):
                dea_score_str = "N/A"
                value_class = ""
            else:
                dea_score_str = f"{dea_score:.4f}"
                value_class = "good-value" if dea_score > 1.0 else ""
                
            super_efficiency = row.get('dea_super_efficiency', 0)
            if pd.isna(super_efficiency):
                super_efficiency_str = "N/A"
            else:
                super_efficiency_str = f"{super_efficiency:.4f}"
            
            # Get data, voice, and message values for DEA outputs
            data_value = row.get('basic_data_clean', row.get('basic_data', 0))
            if pd.isna(data_value):
                data_value = 0
            if row.get('basic_data_unlimited', 0) == 1:
                data_str = "무제한"
            else:
                data_str = f"{data_value}GB"
                
            voice_value = row.get('voice_clean', row.get('voice', 0))
            if pd.isna(voice_value):
                voice_value = 0
            if row.get('voice_unlimited', 0) == 1:
                voice_str = "무제한"
            else:
                voice_str = f"{int(voice_value)}분"
                
            message_value = row.get('message_clean', row.get('message', 0))
            if pd.isna(message_value):
                message_value = 0
            if row.get('message_unlimited', 0) == 1:
                message_str = "무제한"
            else:
                message_str = f"{int(message_value)}건"
            
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
                <td>{row.get('mvno', 'Unknown')}</td>
                <td class="input-feature">{fee_str}</td>
                <td class="dea-metrics {value_class}">{dea_score_str}</td>
                <td class="dea-metrics">{efficiency_str}</td>
                <td class="dea-metrics">{super_efficiency_str}</td>
                <td class="output-feature">{data_str}</td>
                <td class="output-feature">{voice_str}</td>
                <td class="output-feature">{message_str}</td>
                <td class="core-feature">{network}</td>
            """
        else:
            # Spearman specific values
            original_fee = f"{int(row.get('original_fee', row.get('fee', 0))):,}"
            discounted_fee = f"{int(row.get('fee', 0)):,}"
            predicted_price = f"{int(row.get('predicted_price', 0)):,}"
            
            # Value ratio
            value_ratio = row.get('value_ratio_original', row.get('value_ratio', 0))
            if pd.isna(value_ratio):
                value_ratio_str = "N/A"
                value_class = ""
            else:
                value_ratio_str = f"{value_ratio:.2f}"
                value_class = "good-value" if value_ratio > 1.1 else ("bad-value" if value_ratio < 0.9 else "")
                
            operator = row.get('mvno', "Unknown")
            
            html += f"""
            <tr>
                <td>{rank_display}</td>
                <td>{plan_name}</td>
                <td>{operator}</td>
                <td>{original_fee}</td>
                <td>{discounted_fee}</td>
                <td>{predicted_price}</td>
                <td class="{value_class}">{value_ratio_str}</td>
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
