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
    
    # Prepare data for frontier chart visualization
    # Determine the value column based on the ranking method
    value_col = 'CS'
    
    # Sort the DataFrame based on rank column
    rank_col = 'rank_number'
    df_sorted = df.sort_values(rank_col) if rank_col in df.columns else df
    
    # Prepare data for chart - create JSON objects for all plans
    chart_data_points = []
    
    # Sort plans by fee to find frontier
    df_sorted_by_fee = df.sort_values('fee')  # Use original df, not df_sorted
    max_value_at_fee = -float('inf')
    
    for _, row in df_sorted_by_fee.iterrows():
        fee = float(row['fee']) if 'fee' in row and not pd.isna(row['fee']) else 0
        value = float(row[value_col]) if value_col in row and not pd.isna(row[value_col]) else 0
        plan_name = row['plan_name'] if 'plan_name' in row else 'Unknown'
        mvno = row['mvno'] if 'mvno' in row else 'Unknown'
        
        # Get rank based on ranking method
        rank = int(row['rank_number']) if 'rank_number' in row and not pd.isna(row['rank_number']) else 0
        
        # Determine if this is a frontier point
        is_frontier = value > max_value_at_fee
        if is_frontier:
            max_value_at_fee = value
        
        # Create data point
        data_point = {
            'fee': fee,
            'value': value,
            'plan_name': str(plan_name),
            'mvno': str(mvno),
            'rank': rank,
            'is_frontier': is_frontier
        }
        chart_data_points.append(data_point)
    
    # Convert to JSON to embed in HTML
    # Ensure JSON is properly formatted by using a safe JSON dump
    try:
        chart_data_json = json.dumps(chart_data_points)
    except Exception as e:
        logger.error(f"Error serializing chart data: {e}")
        # Provide a fallback empty array if serialization fails
        chart_data_json = "[]"
    
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
        top_value = df_sorted.iloc[0][value_col] if value_col in df_sorted.iloc[0] else None
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
        'message_clean', 'speed_when_exhausted', 'additional_call'
    ]
    
    # Prepare data points for feature-specific charts
    for feature in core_continuous_features:
        if feature not in df.columns:
            continue
            
        # Get the corresponding contribution column
        contribution_col = f"contribution_{feature}"
        if contribution_col not in df.columns:
            continue
            
        # Collect all unique feature values and their contributions
        feature_values = []
        contribution_values = []
        is_frontier_points = []
        plan_names = []
        
        # Group plans by feature value and get minimum contribution
        # This simulates how the frontier is calculated
        grouped = df.groupby(feature)[contribution_col].min().reset_index()
        
        # Track the minimum cost seen so far to identify frontier points
        min_cost_so_far = float('inf')
        
        # Sort by feature value (ascending)
        grouped = grouped.sort_values(feature)
        
        # For each unique feature value, identify if it's a frontier point
        for _, row in grouped.iterrows():
            feature_value = row[feature]
            contribution = row[contribution_col]
            
            # If this contribution is lower than previous minimum, it's a frontier point
            is_frontier = contribution <= min_cost_so_far
            if is_frontier:
                min_cost_so_far = contribution
                
            # Add data point
            feature_values.append(feature_value)
            contribution_values.append(contribution)
            is_frontier_points.append(is_frontier)
            
            # Find a plan with this feature value and contribution for display
            matching_plans = df[(df[feature] == feature_value) & 
                               (df[contribution_col] == contribution)]
            plan_name = matching_plans.iloc[0]['plan_name'] if not matching_plans.empty else "Unknown"
            plan_names.append(plan_name)
        
        # Add to feature frontier data
        feature_frontier_data[feature] = {
            'values': feature_values,
            'contributions': contribution_values,
            'is_frontier': is_frontier_points,
            'plan_names': plan_names
        }
    
    # Serialize feature frontier data to JSON
    try:
        feature_frontier_json = json.dumps(feature_frontier_data)
    except Exception as e:
        logger.error(f"Error serializing feature frontier data: {e}")
        feature_frontier_json = "{}"
    
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
                grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
                gap: 20px;
                margin-top: 20px;
            }}
            
            .chart-container {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 15px;
                background-color: white;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            
            .chart-title {{
                font-size: 16px;
                font-weight: bold;
                margin-bottom: 10px;
                text-align: center;
            }}
            
            .hidden {{ display: none; }}
        </style>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
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
    
    # Add frontier chart section with inline chart data
    html += f"""
        <h2>Value Frontier Analysis</h2>
        <div class="container">
            <div style="display: flex; flex-direction: column;">
                <div style="margin-bottom: 20px;">
                    <p>The chart below shows the frontier analysis of plans based on their fee (x-axis) and value metrics (y-axis). 
                    Plans on the frontier represent the best value at their price point.</p>
                </div>
                <div style="height: 500px; width: 100%;">
                    <canvas id="frontierChart"></canvas>
                </div>
            </div>
        </div>
        
        <script>
        // Chart data is embedded directly here
        const chartData = {chart_data_json};
        
        // Create the frontier chart
        document.addEventListener('DOMContentLoaded', function() {{
            try {{
                console.log('Initializing chart...');
                const ctx = document.getElementById('frontierChart');
                if (!ctx) {{
                    console.error('Could not find frontier chart canvas element');
                    return;
                }}
                console.log('Canvas found');
                
                // Check chart data
                if (!Array.isArray(chartData)) {{
                    console.error('Chart data is not an array:', typeof chartData);
                    return;
                }}
                
                console.log('Data loaded - Plans:', chartData.length);
                
                // Split into frontier and non-frontier points
                const frontierPlans = chartData.filter(plan => plan.is_frontier);
                const otherPlans = chartData.filter(plan => !plan.is_frontier);
                
                console.log('Data sorted - Frontier plans:', frontierPlans.length, ', Other plans:', otherPlans.length);
                
                // Create the datasets for Chart.js
                const frontierDataset = {{
                    label: 'Frontier Plans',
                    data: frontierPlans.map(plan => ({{
                        x: plan.fee,
                        y: plan.value,
                        plan_name: plan.plan_name,
                        mvno: plan.mvno,
                        rank: plan.rank
                    }})),
                    backgroundColor: 'rgba(255, 99, 132, 1)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    pointRadius: 8,
                    pointHoverRadius: 10
                }};
                
                const otherDataset = {{
                    label: 'Other Plans',
                    data: otherPlans.map(plan => ({{
                        x: plan.fee,
                        y: plan.value,
                        plan_name: plan.plan_name,
                        mvno: plan.mvno,
                        rank: plan.rank
                    }})),
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 0.5)',
                    pointRadius: 6,
                    pointHoverRadius: 8
                }};
                
                // Create chart
                console.log('Creating chart...');
                try {{
                    const frontierChart = new Chart(ctx, {{
                        type: 'scatter',
                        data: {{
                            datasets: [frontierDataset, otherDataset]
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {{
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            const point = context.raw;
                                            return [
                                                `${{point.plan_name}} (${{point.mvno}})`, 
                                                `Rank: ${{point.rank}}`, 
                                                `Fee: ${{point.x.toLocaleString()}} KRW`, 
                                                `Value: ${{point.y.toFixed(4)}}`
                                            ];
                                        }}
                                    }}
                                }},
                                legend: {{
                                    position: 'top',
                                }},
                                title: {{
                                    display: true,
                                    text: 'Plan Value Frontier Analysis'
                                }}
                            }},
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: 'Fee (KRW)'
                                    }},
                                    ticks: {{
                                        callback: function(value) {{
                                            return value.toLocaleString();
                                        }}
                                    }}
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'CS Ratio'
                                    }}
                                }}
                            }}
                        }}
                    }});
                    console.log('Chart created successfully!');
                }} catch (err) {{
                    console.error('Error creating chart:', err);
                }}
            }} catch (err) {{
                console.error('Error in chart initialization:', err);
            }}
        }});
        </script>
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
    html += f"""
        <script>
        // Feature frontier data
        const featureFrontierData = {feature_frontier_json};
        
        // Create feature frontier charts
        document.addEventListener('DOMContentLoaded', function() {{
            console.log('Initializing feature frontier charts...');
            
            // Feature display names
            const featureDisplayNames = {{
                'basic_data_clean': 'Basic Data (GB)',
                'daily_data_clean': 'Daily Data (GB)',
                'voice_clean': 'Voice Minutes',
                'message_clean': 'SMS Messages',
                'additional_call': 'Additional Call Minutes',
                'speed_when_exhausted': 'Throttled Speed (Mbps)',
                'tethering_gb': 'Tethering Data (GB)'
            }};
            
            // Get the container
            const container = document.getElementById('feature-charts-container');
            if (!container) {{
                console.error('Could not find feature charts container');
                return;
            }}
            
            // Create a chart for each feature
            for (const [feature, data] of Object.entries(featureFrontierData)) {{
                // Create chart container
                const chartContainer = document.createElement('div');
                chartContainer.className = 'chart-container';
                
                // Create chart title
                const chartTitle = document.createElement('div');
                chartTitle.className = 'chart-title';
                chartTitle.textContent = featureDisplayNames[feature] || feature;
                chartContainer.appendChild(chartTitle);
                
                // Create canvas for chart
                const canvas = document.createElement('canvas');
                canvas.id = `chart-${{feature}}`;
                chartContainer.appendChild(canvas);
                
                // Add to main container
                container.appendChild(chartContainer);
                
                // Prepare datasets
                const frontierPoints = [];
                const nonFrontierPoints = [];
                
                // Split data into frontier and non-frontier points
                for (let i = 0; i < data.values.length; i++) {{
                    const point = {{
                        x: data.values[i],
                        y: data.contributions[i],
                        plan_name: data.plan_names[i]
                    }};
                    
                    if (data.is_frontier[i]) {{
                        frontierPoints.push(point);
                    }} else {{
                        nonFrontierPoints.push(point);
                    }}
                }}
                
                // Create Chart.js chart
                try {{
                    const frontierDataset = {{
                        label: 'Frontier Points',
                        data: frontierPoints,
                        backgroundColor: 'rgba(255, 99, 132, 1)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        pointRadius: 6,
                        pointHoverRadius: 10,
                        showLine: true,
                        tension: 0.1
                    }};
                    
                    const nonFrontierDataset = {{
                        label: 'Other Points',
                        data: nonFrontierPoints,
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 0.5)',
                        pointRadius: 4,
                        pointHoverRadius: 8
                    }};
                    
                    const datasets = [frontierDataset];
                    if (nonFrontierPoints.length > 0) {{
                        datasets.push(nonFrontierDataset);
                    }}
                    
                    new Chart(canvas, {{
                        type: 'scatter',
                        data: {{
                            datasets: datasets
                        }},
                        options: {{
                            responsive: true,
                            maintainAspectRatio: false,
                            aspectRatio: 1.5,
                            plugins: {{
                                tooltip: {{
                                    callbacks: {{
                                        label: function(context) {{
                                            const point = context.raw;
                                            return [
                                                `Plan: ${{point.plan_name}}`,
                                                `Value: ${{point.x}}`,
                                                `Cost: ${{point.y.toLocaleString()}} KRW`
                                            ];
                                        }}
                                    }}
                                }},
                                legend: {{
                                    position: 'top',
                                }},
                                title: {{
                                    display: false
                                }}
                            }},
                            scales: {{
                                x: {{
                                    title: {{
                                        display: true,
                                        text: featureDisplayNames[feature] || feature
                                    }}
                                }},
                                y: {{
                                    title: {{
                                        display: true,
                                        text: 'Baseline Cost (KRW)'
                                    }},
                                    ticks: {{
                                        callback: function(value) {{
                                            return value.toLocaleString();
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }});
                    
                    console.log(`Chart for ${{feature}} created successfully`);
                }} catch (err) {{
                    console.error(`Error creating chart for ${{feature}}:`, err);
                }}
            }}
        }});
        </script>
    """
    
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
