"""
Report Generation Module

This module handles generating and saving HTML reports for the Moyo Ranking Model.
"""

import os
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
import json
import numpy as np

# Define unlimited flag mappings (copied from cost_spec.py)
UNLIMITED_FLAGS = {
    'basic_data_clean': 'basic_data_unlimited',
    'daily_data_clean': 'daily_data_unlimited',
    'voice_clean': 'voice_unlimited',
    'message_clean': 'message_unlimited',
    'speed_when_exhausted': 'has_unlimited_speed'
}

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)

# Configure logging
logger = logging.getLogger(__name__)

def generate_html_report(df, timestamp, is_dea=False, is_cs=True, title="Mobile Plan Rankings"):
    """Generate an HTML report of the rankings.
    
    Args:
        df: DataFrame with ranking data
        timestamp: Timestamp for the report
        is_dea: Deprecated parameter, kept for backward compatibility
        is_cs: Whether this is a Cost-Spec report (default: True)
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
    report_title = "Cost-Spec Mobile Plan Rankings"
    
    # Sort the DataFrame based on rank column
    rank_col = 'rank_number'
    df_sorted = df.sort_values(rank_col) if rank_col in df.columns else df
    
    # Calculate feature contributions to baseline costs
    feature_contribution_data = {}
    
    # Get contribution columns (from cost_spec.py)
    contribution_cols = [col for col in df.columns if col.startswith("contribution_")]
    
    # Calculate average, min, max contribution for each feature
    for col in contribution_cols:
        feature_name = col.replace("contribution_", "")
        avg_contrib = df[col].mean() if col in df.columns else 0
        max_contrib = df[col].max() if col in df.columns else 0
        min_contrib = df[col].min() if col in df.columns else 0
        
        # Percentage of baseline cost
        avg_baseline_cost = df['B'].mean() if 'B' in df.columns else 1
        contribution_percentage = (avg_contrib / avg_baseline_cost * 100) if avg_baseline_cost > 0 else 0
        
        feature_contribution_data[feature_name] = {
            'avg_contribution': avg_contrib,
            'max_contribution': max_contrib,
            'min_contribution': min_contrib,
            'percentage': contribution_percentage
        }
    
    # Build calculation summary data
    total_plans = len(df)
    
    # Top plan info
    if len(df_sorted) > 0:
        top_plan_name = str(df_sorted.iloc[0]['plan_name']) if 'plan_name' in df_sorted.iloc[0] else 'N/A'
        top_value = df_sorted.iloc[0]['CS'] if 'CS' in df_sorted.iloc[0] else None
        if isinstance(top_value, float):
            top_plan_value = f"{top_value:.4f}"
        else:
            top_plan_value = "N/A"
    else:
        top_plan_name = 'N/A'
        top_plan_value = 'N/A'
    
    # Prepare feature frontier data
    # This will hold feature values and their corresponding baseline costs
    feature_frontier_data = {}
    
    # Core continuous features to visualize (those that likely have frontiers)
    core_continuous_features = [
        'basic_data_clean', 'daily_data_clean', 'voice_clean', 
        'message_clean', 'speed_when_exhausted', 'additional_call',
        'tethering_gb'
    ]
    
    # Prepare data points for feature-specific charts
    for feature in core_continuous_features:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in dataframe, skipping visualization")
            continue
            
        # For visualization, we will use 'original_fee' as the cost metric for the frontier.
        # The 'contribution_col' (derived from 'fee') is used for backend calculations but not directly for this plot's Y-axis.
        cost_metric_for_visualization = 'original_fee'
        if cost_metric_for_visualization not in df.columns:
            logger.warning(f"'{cost_metric_for_visualization}' not found in dataframe, skipping visualization for {feature}")
            continue
            
        logger.info(f"Preparing frontier chart data for feature: {feature} using '{cost_metric_for_visualization}'")
        
        # Check if this feature has an unlimited flag
        unlimited_flag = UNLIMITED_FLAGS.get(feature)
        has_unlimited_data = False
        unlimited_min_visual_cost = None # Renamed to avoid confusion with 'fee' based unlimited costs
        unlimited_min_plan = None
        
        # If unlimited flag exists, extract unlimited value data
        if unlimited_flag and unlimited_flag in df.columns:
            unlimited_plans_df = df[df[unlimited_flag] == 1] 
            if not unlimited_plans_df.empty and cost_metric_for_visualization in unlimited_plans_df.columns:
                has_unlimited_data = True
                min_visual_cost_idx = unlimited_plans_df[cost_metric_for_visualization].idxmin()
                unlimited_min_visual_cost = unlimited_plans_df.loc[min_visual_cost_idx, cost_metric_for_visualization]
                unlimited_min_plan = unlimited_plans_df.loc[min_visual_cost_idx, 'plan_name'] if 'plan_name' in unlimited_plans_df.columns else "Unknown"
                logger.info(f"Found unlimited {feature} with minimum '{cost_metric_for_visualization}' {unlimited_min_visual_cost} from plan '{unlimited_min_plan}'")
                df_for_frontier = df[(df[unlimited_flag] == 0) & df[cost_metric_for_visualization].notna()].copy()
            else:
                df_for_frontier = df[df[cost_metric_for_visualization].notna()].copy()
        else:
            df_for_frontier = df[df[cost_metric_for_visualization].notna()].copy()
        
        # Step 1: Get all unique feature values and their minimum costs (using cost_metric_for_visualization)
        candidate_points_details = []
        if not df_for_frontier.empty:
            min_cost_indices = df_for_frontier.loc[df_for_frontier.groupby(feature)[cost_metric_for_visualization].idxmin()].index
            min_cost_candidates_df = df_for_frontier.loc[min_cost_indices]
            
            min_cost_candidates_df = min_cost_candidates_df.sort_values(by=[feature, cost_metric_for_visualization])

            for _, row in min_cost_candidates_df.iterrows():
                candidate_points_details.append({
                    'value': row[feature],
                    'cost': row[cost_metric_for_visualization], # Using original_fee here
                    'plan_name': row['plan_name'] if 'plan_name' in row else "Unknown"
                })
        
        # Step 2: Build the true monotonic frontier (strictly increasing cost, based on cost_metric_for_visualization)
        actual_frontier_stack = []
        for candidate in candidate_points_details:
            while actual_frontier_stack and candidate['cost'] < actual_frontier_stack[-1]['cost']:
                actual_frontier_stack.pop()
            
            if not actual_frontier_stack:
                actual_frontier_stack.append(candidate)
            else:
                last_frontier_point = actual_frontier_stack[-1]
                if (candidate['value'] > last_frontier_point['value'] and
                    candidate['cost'] > last_frontier_point['cost'] and
                    (candidate['cost'] - last_frontier_point['cost']) >= 1.0):
                    actual_frontier_stack.append(candidate)

        # Step 3: Classify all points from df_for_frontier based on the visual frontier
        frontier_feature_values = []
        frontier_visual_costs = [] # Renamed to reflect it's original_fee
        frontier_plan_names = []
        
        excluded_feature_values = []
        excluded_visual_costs = [] # Renamed
        excluded_plan_names = []
        
        other_feature_values = []
        other_visual_costs = [] # Renamed
        other_plan_names = []

        true_frontier_set_tuples = set((p['value'], p['cost'], p['plan_name']) for p in actual_frontier_stack)
        
        candidate_tuples_for_exclusion_check = set((p['value'], p['cost'], p['plan_name']) for p in candidate_points_details)

        for candidate in candidate_points_details:
            if (candidate['value'], candidate['cost'], candidate['plan_name']) in true_frontier_set_tuples:
                frontier_feature_values.append(float(candidate['value']))
                frontier_visual_costs.append(float(candidate['cost'])) # Storing original_fee
                frontier_plan_names.append(candidate['plan_name'])
            else:
                excluded_feature_values.append(float(candidate['value']))
                excluded_visual_costs.append(float(candidate['cost'])) # Storing original_fee
                excluded_plan_names.append(candidate['plan_name'])
        
        if not df_for_frontier.empty:
            all_candidate_min_value_cost_pairs = set((p['value'], p['cost']) for p in candidate_points_details)
            for _, row in df_for_frontier.iterrows():
                f_val = row[feature]
                c_cost = row[cost_metric_for_visualization] # Using original_fee for comparison
                p_name = row['plan_name'] if 'plan_name' in row else "Unknown"
                if (f_val, c_cost) not in all_candidate_min_value_cost_pairs:
                    other_feature_values.append(float(f_val))
                    other_visual_costs.append(float(c_cost)) # Storing original_fee
                    other_plan_names.append(p_name)
        
        all_values_for_js = []
        all_visual_costs_for_js = [] # Renamed
        all_plan_names_for_js = []
        all_is_frontier_for_js = []
        all_is_excluded_for_js = []

        for i in range(len(frontier_feature_values)):
            all_values_for_js.append(frontier_feature_values[i])
            all_visual_costs_for_js.append(frontier_visual_costs[i]) # Using original_fee based list
            all_plan_names_for_js.append(frontier_plan_names[i])
            all_is_frontier_for_js.append(True)
            all_is_excluded_for_js.append(False)
        
        for i in range(len(excluded_feature_values)):
            all_values_for_js.append(excluded_feature_values[i])
            all_visual_costs_for_js.append(excluded_visual_costs[i]) # Using original_fee based list
            all_plan_names_for_js.append(excluded_plan_names[i])
            all_is_frontier_for_js.append(False)
            all_is_excluded_for_js.append(True)

        for i in range(len(other_feature_values)):
            all_values_for_js.append(other_feature_values[i])
            all_visual_costs_for_js.append(other_visual_costs[i]) # Using original_fee based list
            all_plan_names_for_js.append(other_plan_names[i])
            all_is_frontier_for_js.append(False)
            all_is_excluded_for_js.append(False)
            
        frontier_points_count = len(frontier_feature_values)
        excluded_points_count = len(excluded_feature_values)
        other_points_count = len(other_feature_values)
        unlimited_count = 1 if has_unlimited_data else 0
        
        logger.info(f"Identified {frontier_points_count} visual frontier points, {excluded_points_count} excluded, {other_points_count} other, {unlimited_count} unlimited for {feature} using '{cost_metric_for_visualization}'")
        
        if frontier_points_count > 0 or excluded_points_count > 0 or other_points_count > 0 or has_unlimited_data:
            feature_frontier_data[feature] = {
                'all_values': all_values_for_js, 
                'all_contributions': all_visual_costs_for_js, # Changed key name to all_costs_for_visualization or similar might be clearer, but JS expects all_contributions
                'all_is_frontier': all_is_frontier_for_js,
                'all_is_excluded': all_is_excluded_for_js,
                'all_is_unlimited': [has_unlimited_data and val == float('inf') for val in all_values_for_js], # This might need review if inf is not used for original_fee unlimited
                'all_plan_names': all_plan_names_for_js,
                
                'frontier_values': frontier_feature_values,
                'frontier_contributions': frontier_visual_costs, # JS expects this key name
                'frontier_plan_names': frontier_plan_names,
                'excluded_values': excluded_feature_values,
                'excluded_contributions': excluded_visual_costs, # JS expects this key name
                'excluded_plan_names': excluded_plan_names,
                'other_values': other_feature_values,
                'other_contributions': other_visual_costs, # JS expects this key name
                'other_plan_names': other_plan_names,
                'has_unlimited': has_unlimited_data,
                'unlimited_value': unlimited_min_visual_cost if has_unlimited_data else None, # This is original_fee based
                'unlimited_plan': unlimited_min_plan if has_unlimited_data else None
            }
            logger.info(f"Added data for {feature} to chart data. Visual Frontier: {frontier_points_count}, Excluded: {excluded_points_count}, Other: {other_points_count}, Unlimited: {unlimited_count}")
        else:
            logger.warning(f"No points found for {feature} (including unlimited) using '{cost_metric_for_visualization}', skipping chart data preparation")
    
    # Serialize feature frontier data to JSON
    try:
        feature_frontier_json = json.dumps(feature_frontier_data, cls=NumpyEncoder)
        logger.info(f"Successfully serialized frontier data: {len(feature_frontier_data)} features included")
    except Exception as e:
        logger.error(f"Error serializing feature frontier data: {e}")
        feature_frontier_json = "{}"
    
    # Check if we have any frontier data to display
    if not feature_frontier_data:
        logger.warning("No feature frontier data available for charts, charts will not be displayed")
    
    # Create HTML
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{report_title} - {timestamp_str}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2, h3 {{ color: #333; }}
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
            .cs-metrics {{ background-color: #f9f0ff; }}
            .input-feature {{ background-color: #f9f0ff; }}
            .output-feature {{ background-color: #f6ffed; }}
            
            /* Bar chart styles */
            .bar-container {{ 
                width: 100%; 
                background-color: #f1f1f1; 
                margin-top: 5px;
                border-radius: 4px;
                overflow: hidden;
            }}
            .bar {{ 
                height: 20px; 
                background-color: #4CAF50; 
                text-align: right; 
                color: white; 
                padding-right: 5px;
                border-radius: 4px;
            }}
            
            /* Feature charts grid */
            .chart-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 10px;
                margin-top: 15px;
            }}
            
            .chart-container {{
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 10px;
                background-color: white;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                height: 250px;
                min-width: 280px;
                max-width: 400px;
                margin: 0 auto;
            }}
            
            .chart-title {{
                font-size: 14px;
                font-weight: bold;
                margin-bottom: 5px;
                text-align: center;
            }}
            
            .hidden {{ display: none; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@2.1.0/dist/chartjs-plugin-annotation.min.js"></script>
    </head>
    <body>
        <h1>{report_title}</h1>
        <p>Generated: {timestamp_str}</p>
        
 
    """
    
    # Add method explanation section - CS method only now
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
    
    # Add feature contributions section (new section)
    html += """
        <h2>Feature Contributions to Baseline Cost</h2>
        <div class="container">
        <div class="note">
            <p>This section shows how each feature contributes to the baseline cost calculation. The baseline cost for each feature is determined by finding the minimum fee among plans with that feature value. Features with higher contributions have a greater impact on the overall ranking.</p>
        </div>
        <table>
            <tr>
                <th>Feature</th>
                <th>Avg Contribution (KRW)</th>
                <th>Min Contribution (KRW)</th>
                <th>Max Contribution (KRW)</th>
                <th>% of Baseline Cost</th>
                <th>Contribution Distribution</th>
            </tr>
    """
    
    # Sort features by average contribution (descending)
    sorted_features = sorted(
        feature_contribution_data.items(),
        key=lambda x: x[1]['avg_contribution'],
        reverse=True
    )
    
    # Add rows for each feature contribution
    for feature_name, data in sorted_features:
        avg_contrib = data['avg_contribution']
        min_contrib = data['min_contribution']
        max_contrib = data['max_contribution']
        percentage = data['percentage']
        
        # Create a simple bar chart for the percentage
        bar_width = min(percentage, 100)  # Cap at 100%
        
        html += f"""
        <tr>
            <td>{feature_name}</td>
            <td>{int(avg_contrib):,} KRW</td>
            <td>{int(min_contrib):,} KRW</td>
            <td>{int(max_contrib):,} KRW</td>
            <td>{percentage:.1f}%</td>
            <td>
                <div class="bar-container">
                    <div class="bar" style="width: {bar_width}%">{percentage:.1f}%</div>
                </div>
            </td>
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
                <th>Baseline Cost (B)</th>
                <th>CS Ratio</th>
                <th>Data (GB)</th>
                <th>Voice (min)</th>
                <th>Message (SMS)</th>
                <th>Additional Call (min)</th>
                <th>Throttled Speed (Mbps)</th>
                <th>5G</th>
            </tr>
    """
    
    # Generate table rows
    for _, row in df_sorted.iterrows():
        # Format rank
        rank = int(row['rank_number']) if 'rank_number' in row and not pd.isna(row['rank_number']) else ""
        rank_str = f"{rank}" if rank else ""
        
        # Get plan data
        plan_name = row['plan_name'] if 'plan_name' in row else ""
        mvno = row['mvno'] if 'mvno' in row else ""
        fee = int(row['fee']) if 'fee' in row and not pd.isna(row['fee']) else 0
        original_fee = int(row['original_fee']) if 'original_fee' in row and not pd.isna(row['original_fee']) else 0
        
        # CS-specific metrics
        baseline_cost = int(row['B']) if 'B' in row and not pd.isna(row['B']) else 0
        cs_ratio = row['CS'] if 'CS' in row else ""
        
        # Format CS ratio with proper handling of types
        formatted_cs_ratio = f"{cs_ratio:.4f}" if isinstance(cs_ratio, float) else str(cs_ratio)
        
        # Get feature data
        data_gb = row['basic_data_clean'] if 'basic_data_clean' in row else "N/A"
        voice = row['voice_clean'] if 'voice_clean' in row else "N/A"
        message = row['message_clean'] if 'message_clean' in row else "N/A"
        additional_call = row['additional_call'] if 'additional_call' in row else "N/A"
        is_5g = "Yes" if row.get('is_5g') == 1 else "No"
        
        # Throttled speed data - using raw speed since that's used in calculations
        raw_speed = row['speed_when_exhausted'] if 'speed_when_exhausted' in row else 0
        
        # Format throttled speed
        if 'has_unlimited_speed' in row and row['has_unlimited_speed'] == 1:
            throttled_speed = "Unlimited"
        elif raw_speed > 0:
            throttled_speed = f"{raw_speed} Mbps"
        else:
            throttled_speed = "N/A"
        
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
                <td>{baseline_cost:,}</td>
                <td class="good-value">{formatted_cs_ratio}</td>
                <td>{data_gb}</td>
                <td>{voice}</td>
                <td>{message}</td>
                <td>{additional_call}</td>
                <td>{throttled_speed}</td>
                <td>{is_5g}</td>
            </tr>
        """
    
    # Close the rankings table
    html += """
        </table>
        </div>
    """
    
    # Add calculation summary table with updated metrics
    html += f"""
        <h2>Calculation Summary</h2>
        <div class="container">
            <div class="note">
                <p><strong>Feature Importance:</strong> This summary shows the overall contribution of features to the baseline cost, which determines the CS ratio and ranking.</p>
            </div>
            
            <div style="display: flex; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 300px; margin-right: 20px;">
                    <table>
                        <tr>
                            <th>Metric</th>
                            <th>Value</th>
                        </tr>
                        <tr>
                            <td>Total Plans Analyzed</td>
                            <td>{total_plans}</td>
                        </tr>
                        <tr>
                            <td>Top Contributing Feature</td>
                            <td>{sorted_features[0][0] if sorted_features else 'N/A'}</td>
                        </tr>
                        <tr>
                            <td>Avg Baseline Cost</td>
                            <td>{int(df['B'].mean()) if 'B' in df.columns else 0:,} KRW</td>
                        </tr>
                        <tr>
                            <td>Top Plan</td>
                            <td>{top_plan_name}</td>
                        </tr>
                        <tr>
                            <td>Top Plan CS Ratio</td>
                            <td>{top_plan_value}</td>
                        </tr>
                    </table>
                </div>
            </div>
        </div>
    """
    
    # Add Feature Frontier Charts section
    html += """
        <h2>Feature Frontier Analysis</h2>
        <div class="container">
            <div class="note">
                <p><strong>Feature Frontier Analysis:</strong> These charts show how each feature contributes to the baseline cost. Points on the frontier (red) represent the minimum cost for each feature value and are used in the baseline calculation. The baseline cost increases as feature value increases, reflecting that higher feature values typically cost more.</p>
            </div>
    """
    
    # Add div for the feature charts grid
    html += """
            <div class="chart-grid" id="feature-charts-container">
                <!-- Feature charts will be generated here -->
            </div>
    """
    
    # Close container
    html += """
        </div>
    """
    
    # Add JavaScript for feature frontier charts
    script_template = """
        <script>
        // Feature frontier data
        const featureFrontierData = FRONTIER_DATA_JSON;
        
        // Create feature frontier charts
        document.addEventListener('DOMContentLoaded', function() {
            console.log('Initializing feature frontier charts...');
            console.log('Feature frontier data:', featureFrontierData);
            
            // Check if we have data
            if (!featureFrontierData || Object.keys(featureFrontierData).length === 0) {
                console.warn('No frontier data available, charts will not be displayed');
                document.getElementById('feature-charts-container').innerHTML = '<p>No frontier data available for visualization.</p>';
                return;
            }
            
            // Feature display names
            const featureDisplayNames = {
                'basic_data_clean': 'Basic Data (GB)',
                'daily_data_clean': 'Daily Data (GB)',
                'voice_clean': 'Voice Minutes',
                'message_clean': 'SMS Messages',
                'additional_call': 'Additional Call Minutes',
                'speed_when_exhausted': 'Throttled Speed (Mbps)',
                'tethering_gb': 'Tethering Data (GB)'
            };
            
            // Get the container
            const container = document.getElementById('feature-charts-container');
            if (!container) {
                console.error('Could not find feature charts container');
                return;
            }
            
            // Create a chart for each feature
            for (const [feature, data] of Object.entries(featureFrontierData)) {
                console.log(`Processing feature: ${feature} with ${data.all_values.length} total data points`);
                
                // Validate data
                if (!data.all_values || !data.all_contributions || data.all_values.length === 0) {
                    console.warn(`Invalid data for feature ${feature}, skipping`);
                    continue;
                }
                
                // Create chart container
                const chartContainer = document.createElement('div');
                chartContainer.className = 'chart-container';
                
                // Create chart title
                const chartTitle = document.createElement('div');
                chartTitle.className = 'chart-title';
                const hasUnlimited = data.has_unlimited ? ' U:1' : '';
                chartTitle.textContent = (featureDisplayNames[feature] || feature) + 
                    ` (F:${data.frontier_values.length} E:${data.excluded_values.length} O:${data.other_values.length}${hasUnlimited})`;
                chartContainer.appendChild(chartTitle);
                
                // Create canvas for chart
                const canvas = document.createElement('canvas');
                canvas.id = "chart-" + feature;
                chartContainer.appendChild(canvas);
                
                // Add to main container
                container.appendChild(chartContainer);
                
                // Prepare chart data points for frontier points
                try {
                const frontierPoints = [];
                    const excludedPoints = [];
                    const otherPoints = [];
                    const unlimitedPoints = [];
                    
                    // Create data points for frontier points
                    for (let i = 0; i < data.frontier_values.length; i++) {
                        const x = data.frontier_values[i];
                        const y = data.frontier_contributions[i];
                        const planName = data.frontier_plan_names[i] || 'Unknown';
                        
                        // Ensure x and y are numbers
                        if (typeof x === 'number' && typeof y === 'number' && !isNaN(x) && !isNaN(y)) {
                            frontierPoints.push({
                                x: x,
                                y: y,
                                plan_name: planName,
                                is_frontier: true,
                                is_excluded: false,
                                is_unlimited: false
                            });
                        }
                    }
                    
                    // Create data points for excluded points
                    for (let i = 0; i < data.excluded_values.length; i++) {
                        const x = data.excluded_values[i];
                        const y = data.excluded_contributions[i];
                        const planName = data.excluded_plan_names[i] || 'Unknown';
                        
                        // Ensure x and y are numbers
                        if (typeof x === 'number' && typeof y === 'number' && !isNaN(x) && !isNaN(y)) {
                            excludedPoints.push({
                                x: x,
                                y: y,
                                plan_name: planName,
                                is_frontier: false,
                                is_excluded: true,
                                is_unlimited: false
                            });
                        }
                    }
                    
                    // Create data points for other points
                    for (let i = 0; i < data.other_values.length; i++) {
                        const x = data.other_values[i];
                        const y = data.other_contributions[i];
                        const planName = data.other_plan_names[i] || 'Unknown';
                        
                        // Ensure x and y are numbers
                        if (typeof x === 'number' && typeof y === 'number' && !isNaN(x) && !isNaN(y)) {
                            otherPoints.push({
                                x: x,
                                y: y,
                                plan_name: planName,
                                is_frontier: false,
                                is_excluded: false,
                                is_unlimited: false
                            });
                        }
                    }
                    
                    // Add unlimited point if available
                    if (data.has_unlimited && data.unlimited_value !== null) {
                        const unlimitedValue = data.unlimited_value;
                        const unlimitedPlan = data.unlimited_plan || 'Unknown';
                        
                        // Create a special mark for unlimited (using the right edge of the chart)
                        unlimitedPoints.push({
                            // Position at the maximum of actual data points plus 20% or at a default position
                            x: data.frontier_values.length > 0 ? Math.max(...data.frontier_values) * 1.2 : 10,
                            y: unlimitedValue,
                            plan_name: unlimitedPlan,
                            is_frontier: true,
                            is_excluded: false,
                            is_unlimited: true,
                            originalValue: "Unlimited"
                        });
                    }
                    
                    const pointCounts = {
                        frontier: frontierPoints.length,
                        excluded: excludedPoints.length,
                        other: otherPoints.length,
                        unlimited: unlimitedPoints.length
                    };
                    
                    console.log(`Created ${frontierPoints.length} frontier points, ${excludedPoints.length} excluded points, ${otherPoints.length} other points, and ${unlimitedPoints.length} unlimited point for ${feature}`);
                    
                    if (frontierPoints.length === 0 && excludedPoints.length === 0 && otherPoints.length === 0 && unlimitedPoints.length === 0) {
                        console.warn(`No valid data points for ${feature}, skipping chart`);
                        continue;
                    }
                    
                    // Create Chart.js datasets
                    const frontierDataset = {
                        label: 'Frontier Points',
                        data: frontierPoints,
                        backgroundColor: 'rgba(255, 0, 0, 1)',
                        borderColor: 'rgba(255, 0, 0, 1)',
                        pointRadius: 5,
                        pointHoverRadius: 8,
                        showLine: true,
                        tension: 0.1,
                        borderWidth: 2,
                        fill: false
                    };
                    
                    // Add a dataset for the frontier line area
                    const frontierAreaDataset = {
                        label: 'Frontier Line',
                        data: frontierPoints,
                        backgroundColor: 'rgba(255, 0, 0, 0.1)',
                        borderColor: 'rgba(255, 0, 0, 0)',
                        pointRadius: 0,
                        showLine: true,
                        tension: 0.1,
                        fill: 'origin',
                        hidden: false,
                        // Hide from legend
                        hidden: false,
                        showLine: true,
                        display: true,
                        spanGaps: true
                    };
                    
                    const excludedDataset = {
                        label: 'Excluded Points',
                        data: excludedPoints,
                        backgroundColor: 'rgba(255, 165, 0, 0.8)',  // Orange for excluded
                        borderColor: 'rgba(255, 165, 0, 0.8)',
                        pointRadius: 4,
                        pointHoverRadius: 7,
                        showLine: false
                    };
                    
                    const otherDataset = {
                        label: 'Other Points',
                        data: otherPoints,
                        backgroundColor: 'rgba(100, 100, 100, 0.5)',  // Gray for others
                        borderColor: 'rgba(100, 100, 100, 0.5)',
                        pointRadius: 3,
                        pointHoverRadius: 6,
                        showLine: false
                    };
                    
                    const unlimitedDataset = {
                        label: 'Unlimited',
                        data: unlimitedPoints,
                        backgroundColor: 'rgba(128, 0, 128, 1)',  // Purple for unlimited
                        borderColor: 'rgba(128, 0, 128, 1)',
                        pointRadius: 6,
                        pointHoverRadius: 9,
                        pointStyle: 'star',
                        showLine: false
                    };
                    
                    // Determine which datasets to include
                    const datasets = [];
                    if (otherPoints.length > 0) datasets.push(otherDataset);
                    if (excludedPoints.length > 0) datasets.push(excludedDataset);
                    if (unlimitedPoints.length > 0) datasets.push(unlimitedDataset);
                    if (frontierPoints.length > 0) {
                        if (frontierPoints.length > 1) datasets.push(frontierAreaDataset);
                        datasets.push(frontierDataset);
                    }
                    
                    // Create Chart.js chart options with possible unlimited annotation
                    const chartOptions = {
                            responsive: true,
                        maintainAspectRatio: true,
                        aspectRatio: 1.8,
                        plugins: {
                            tooltip: {
                                titleFont: {
                                    size: 12
                                },
                                bodyFont: {
                                    size: 11
                                },
                                callbacks: {
                                    label: function(context) {
                                            const point = context.raw;
                                        // Don't show tooltip for area dataset
                                        if (context.dataset.label === 'Frontier Line') return null;
                                        
                                        // Base info
                                        const tooltipLines = [
                                            "Plan: " + point.plan_name,
                                            "Value: " + (point.is_unlimited ? "Unlimited" : point.x),
                                            "Cost: " + point.y.toLocaleString() + " KRW"
                                        ];
                                        
                                        // Add point type explanation
                                        if (point.is_frontier) {
                                            tooltipLines.push("Type: Frontier point - used in baseline cost");
                                        } else if (point.is_excluded) {
                                            tooltipLines.push("Type: Excluded - minimum cost for value but not monotonic");
                                        } else if (point.is_unlimited) {
                                            tooltipLines.push("Type: Unlimited value");
                                        } else {
                                            tooltipLines.push("Type: Other - not minimum cost for value");
                                        }
                                        
                                        return tooltipLines;
                                    }
                                }
                            },
                            legend: {
                                    position: 'top',
                                labels: {
                                    font: {
                                        size: 11
                                    },
                                    boxWidth: 12,
                                    padding: 8,
                                    filter: function(legendItem, chartData) {
                                        // Don't show frontier area in legend
                                        return legendItem.text !== 'Frontier Line';
                                    }
                                }
                            },
                                                            title: {
                                    display: false
                                }
                        },
                        scales: {
                            x: {
                                title: {
                                        display: true,
                                    text: featureDisplayNames[feature] || feature,
                                    font: {
                                        size: 11
                                    },
                                    padding: {
                                        top: 0,
                                        bottom: 0
                                    }
                                },
                                ticks: {
                                    font: {
                                        size: 10
                                    }
                                }
                            },
                            y: {
                                title: {
                                        display: true,
                                    text: 'Original Fee (KRW)',
                                    font: {
                                        size: 11
                                    }
                                },
                                ticks: {
                                    callback: function(value) {
                                            return value.toLocaleString();
                                    },
                                    font: {
                                        size: 10
                                    }
                                }
                            }
                        },
                        layout: {
                            padding: {
                                left: 0,
                                right: 0,
                                top: 0,
                                bottom: 0
                            }
                        }
                    };
                    
                    new Chart(canvas, {
                        type: 'scatter',
                        data: {
                            datasets: datasets
                        },
                        options: chartOptions
                    });
                    
                    // Add text label for unlimited value if present
                    if (data.has_unlimited && data.unlimited_value !== null) {
                        const unlimitedLabel = document.createElement('div');
                        unlimitedLabel.style.position = 'absolute';
                        unlimitedLabel.style.right = '10px';
                        unlimitedLabel.style.top = '50%';
                        unlimitedLabel.style.transform = 'translateY(-50%)';
                        unlimitedLabel.style.backgroundColor = 'rgba(128, 0, 128, 0.1)';
                        unlimitedLabel.style.borderLeft = '3px solid rgba(128, 0, 128, 0.8)';
                        unlimitedLabel.style.padding = '4px 8px';
                        unlimitedLabel.style.fontSize = '11px';
                        unlimitedLabel.style.color = '#333';
                        unlimitedLabel.style.borderRadius = '0 4px 4px 0';
                        unlimitedLabel.innerHTML = `<strong>Unlimited:</strong> ${data.unlimited_value.toLocaleString()} KRW<br>Plan: ${data.unlimited_plan}`;
                        chartContainer.style.position = 'relative';
                        chartContainer.appendChild(unlimitedLabel);
                    }
                    
                    console.log("Chart for " + feature + " created successfully");
                } catch (err) {
                    console.error("Error creating chart for " + feature + ":", err);
                }
            }
        });
        </script>
    """
    
    # Replace the placeholder with actual JSON data
    script_html = script_template.replace('FRONTIER_DATA_JSON', feature_frontier_json)
    html += script_html
    
    # Close HTML
    html += """
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
