"""
Report Charts Module

This module handles chart data preparation for feature frontier visualization.
"""

import logging
import numpy as np
import pandas as pd
from .report_utils import UNLIMITED_FLAGS

# Configure logging
logger = logging.getLogger(__name__)

def prepare_feature_frontier_data(df, core_continuous_features):
    """
    Prepares data for feature frontier charts.

    Args:
        df: DataFrame with ranking data
        core_continuous_features: List of core continuous features to visualize

    Returns:
        Dictionary with feature frontier data for visualization
    """
    # Prepare feature frontier data
    feature_frontier_data = {}

    # Initialize all_chart_data to store comprehensive chart data including full plan series for frontiers
    all_chart_data = {}
    visual_frontiers_for_residual_table = {} # Stores (value, cost) tuples for estimate_value_on_visual_frontier

    # Cost metric for visualization - we use original_fee for the visual frontier
    cost_metric_for_visualization = 'original_fee'

    for feature in core_continuous_features:
        if feature not in df.columns:
            logger.warning(f"Feature {feature} not found in dataframe, skipping visualization")
            continue

        if cost_metric_for_visualization not in df.columns:
            logger.warning(f"'{cost_metric_for_visualization}' not found in dataframe, skipping visualization for {feature}")
            continue

        # Verbose logging disabled to prevent spam from frequent polling
    # logger.info(f"Preparing frontier chart data for feature: {feature} using '{cost_metric_for_visualization}'")

        # Check if this feature has an unlimited flag
        unlimited_flag = UNLIMITED_FLAGS.get(feature)
        has_unlimited_data = False
        unlimited_min_visual_cost = None
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

        # Step 1: Get all unique feature values and their minimum costs
        candidate_points_details_series = [] 
        if not df_for_frontier.empty:
            # For each unique cost, find the point with maximum feature value
            # This ensures we pick higher spec points when costs are the same
            cost_to_max_feature = {}
            for _, row in df_for_frontier.iterrows():
                cost = row[cost_metric_for_visualization]
                feature_val = row[feature]

                if cost not in cost_to_max_feature or feature_val > cost_to_max_feature[cost]['feature_val']:
                    cost_to_max_feature[cost] = {
                        'feature_val': feature_val,
                        'row': row
                    }

            # Now for each feature value, find the minimum cost among the selected high-spec points
            feature_to_min_cost = {}
            for cost_info in cost_to_max_feature.values():
                feature_val = cost_info['feature_val']
                row = cost_info['row']
                cost = row[cost_metric_for_visualization]

                if feature_val not in feature_to_min_cost or cost < feature_to_min_cost[feature_val]['cost']:
                    feature_to_min_cost[feature_val] = {
                        'cost': cost,
                        'row': row
                    }

            # Sort candidates by feature value, then by cost
            sorted_candidates = sorted(feature_to_min_cost.items(), key=lambda x: (x[0], x[1]['cost']))

            for feature_val, cost_info in sorted_candidates:
                candidate_points_details_series.append(cost_info['row'])

            # Verbose logging disabled to prevent spam from frequent polling
    # logger.info(f"Found {len(candidate_points_details_series)} minimum-cost candidate points for feature {feature}")

        # Step 2: Build the true monotonic frontier
        actual_frontier_plans_series_list = []
        should_add_zero_point = True

        for candidate_plan_series in candidate_points_details_series:
            current_value = candidate_plan_series[feature]
            current_cost = candidate_plan_series[cost_metric_for_visualization]

            # Allow the addition of the candidate if it completely dominates the frontier so far
            max_iterations_1 = len(candidate_points_details_series) + 10  # 안전장치
            iteration_count_1 = 0
            while actual_frontier_plans_series_list and iteration_count_1 < max_iterations_1:
                iteration_count_1 += 1
                last_frontier_plan_series = actual_frontier_plans_series_list[-1]
                last_value = last_frontier_plan_series[feature]
                last_cost = last_frontier_plan_series[cost_metric_for_visualization]

                # If the candidate is more optimal, we remove points and recheck conditions
                if current_value > last_value and current_cost < last_cost:
                    actual_frontier_plans_series_list.pop()
                    should_add_zero_point = True  # We need to reconsider adding the (0,0) point
                else:
                    break
                    
            if iteration_count_1 >= max_iterations_1:
                logger.warning(f"Feature {feature}: First while loop reached maximum iterations ({max_iterations_1}), breaking to prevent infinite loop")

            # Check if the candidate can be added based on monotonic increase rule
            if actual_frontier_plans_series_list:
                # Remove points from the end of the frontier that conflict with adding this candidate
                max_iterations_2 = len(actual_frontier_plans_series_list) + 10  # 안전장치
                iteration_count_2 = 0
                while actual_frontier_plans_series_list and iteration_count_2 < max_iterations_2:
                    iteration_count_2 += 1
                    last_frontier_plan_series = actual_frontier_plans_series_list[-1]
                    last_value = last_frontier_plan_series[feature]
                    last_cost = last_frontier_plan_series[cost_metric_for_visualization]

                    # Skip this candidate if it has same or lower feature value
                    if current_value <= last_value:
                        break  # Cannot add this candidate
                    
                    # Skip this candidate if it has same or lower cost
                    if current_cost <= last_cost:
                        break  # Cannot add this candidate
                    
                    # 안전한 cost_per_unit 계산
                    if (current_value - last_value) == 0:
                        logger.warning(f"Feature {feature}: Division by zero avoided in cost_per_unit calculation")
                        break
                        
                    cost_per_unit = (current_cost - last_cost) / (current_value - last_value)
                    if cost_per_unit >= 1.0:
                        # This candidate can be added - it meets all criteria
                        break
                    else:
                        # Remove the last point and try again with the previous point
                        actual_frontier_plans_series_list.pop()
                        should_add_zero_point = True  # Reconsider adding zero point
                        
                if iteration_count_2 >= max_iterations_2:
                    logger.warning(f"Feature {feature}: Second while loop reached maximum iterations ({max_iterations_2}), breaking to prevent infinite loop")
                
                # If we still have points in the frontier, check one more time if we can add the candidate
                if actual_frontier_plans_series_list:
                    last_frontier_plan_series = actual_frontier_plans_series_list[-1]
                    last_value = last_frontier_plan_series[feature]
                    last_cost = last_frontier_plan_series[cost_metric_for_visualization]
                    
                    if (current_value > last_value and 
                        current_cost > last_cost and
                        (current_cost - last_cost) / (current_value - last_value) >= 1.0):
                        actual_frontier_plans_series_list.append(candidate_plan_series)
                        if current_value > 0:
                            should_add_zero_point = False
                    # If criteria not met, skip this candidate
                else:
                    # Frontier is empty, add this as first point
                    actual_frontier_plans_series_list.append(candidate_plan_series)
                    if current_value > 0:
                        should_add_zero_point = False
            else:
                # First candidate point
                actual_frontier_plans_series_list.append(candidate_plan_series)
                if current_value > 0:  # Only disable zero point if we have a non-zero value
                    should_add_zero_point = False

        # Add (0,0) as the starting point if conditions are met
        if should_add_zero_point and not df_for_frontier.empty:
            zero_point_series = actual_frontier_plans_series_list[0].copy() if actual_frontier_plans_series_list else pd.Series({})
            zero_point_series[feature] = 0
            zero_point_series[cost_metric_for_visualization] = 0
            zero_point_series['plan_name'] = "Free Baseline"
            actual_frontier_plans_series_list.insert(0, zero_point_series)
            logger.info(f"Added (0,0) starting point to feature {feature} frontier")

        # Ensure the last points are connected by sorting
        actual_frontier_plans_series_list.sort(key=lambda p: p[feature])

        # Find the maximum feature value in the dataset, and add the lowest cost point for that value to the frontier if not present
        if not df_for_frontier.empty:
            max_feature_value = df_for_frontier[feature].max()
            max_feature_rows = df_for_frontier[df_for_frontier[feature] == max_feature_value]

            if not max_feature_rows.empty:
                # Find minimum cost for the maximum feature value
                min_cost_for_max_value = max_feature_rows[cost_metric_for_visualization].min()
                max_value_min_cost_row = max_feature_rows.loc[max_feature_rows[cost_metric_for_visualization] == min_cost_for_max_value].iloc[0]

                # Check if this max value point is already in our frontier 
                existing_max_values = [p[feature] for p in actual_frontier_plans_series_list if p[feature] == max_feature_value]

                # If not in frontier, or if our frontier is empty, add it
                if not existing_max_values and actual_frontier_plans_series_list:
                    # Only add if it maintains monotonicity and 1.0 KRW minimum increase
                    last_frontier_point = actual_frontier_plans_series_list[-1]
                    cost_per_unit = (min_cost_for_max_value - last_frontier_point[cost_metric_for_visualization]) / (max_feature_value - last_frontier_point[feature])
                    if (max_feature_value > last_frontier_point[feature] and
                        min_cost_for_max_value > last_frontier_point[cost_metric_for_visualization] and
                        cost_per_unit >= 1.0):
                        actual_frontier_plans_series_list.append(max_value_min_cost_row)
                        logger.info(f"Added maximum value point ({max_feature_value}) with minimum cost to feature {feature} frontier")

        # Verbose logging disabled to prevent spam from frequent polling
        # logger.info(f"Built monotonic frontier with {len(actual_frontier_plans_series_list)} points for feature {feature}")

        # Populate visual_frontiers_for_residual_table with (value, original_fee) tuples from these Series
        current_feature_visual_frontier_tuples = [(p[feature], p[cost_metric_for_visualization]) for p in actual_frontier_plans_series_list]
        visual_frontiers_for_residual_table[feature] = current_feature_visual_frontier_tuples

        # Store the list of frontier plan Series in all_chart_data for later use in residual analysis
        if feature not in all_chart_data: all_chart_data[feature] = {}
        all_chart_data[feature]['actual_frontier_plans_series'] = actual_frontier_plans_series_list

        # Step 3: Classify all points from df_for_frontier based on the visual frontier
        # Extract necessary data for JS charts
        frontier_feature_values = [p[feature] for p in actual_frontier_plans_series_list]
        frontier_visual_costs = [p[cost_metric_for_visualization] for p in actual_frontier_plans_series_list]
        frontier_plan_names = [p['plan_name'] if 'plan_name' in p else "Unknown" for p in actual_frontier_plans_series_list]

        excluded_feature_values = []
        excluded_visual_costs = [] 
        excluded_plan_names = []

        other_feature_values = []
        other_visual_costs = []
        other_plan_names = []

        # Create a set of (value, cost, plan_name) for quick lookup of true frontier points
        true_frontier_signature_set = set(
            (p[feature], p[cost_metric_for_visualization], p['plan_name'] if 'plan_name' in p else "Unknown") 
            for p in actual_frontier_plans_series_list
        )

        # Identify excluded points from the initial candidates (min cost for each value)
        for candidate_plan_series in candidate_points_details_series:
            sig = (candidate_plan_series[feature], candidate_plan_series[cost_metric_for_visualization], 
                  candidate_plan_series['plan_name'] if 'plan_name' in candidate_plan_series else "Unknown")
            if sig not in true_frontier_signature_set:
                excluded_feature_values.append(float(candidate_plan_series[feature]))
                excluded_visual_costs.append(float(candidate_plan_series[cost_metric_for_visualization]))
                excluded_plan_names.append(candidate_plan_series['plan_name'] if 'plan_name' in candidate_plan_series else "Unknown")

        # Identify 'other' points (not on frontier, not excluded - i.e., not a min cost for their value)
        if not df_for_frontier.empty:
            # Create a set of (value, cost) for all candidates to quickly identify non-candidate points
            all_candidate_min_value_cost_pairs = set(
                (p_series[feature], p_series[cost_metric_for_visualization]) for p_series in candidate_points_details_series
            )
            for _, row_series in df_for_frontier.iterrows():
                f_val = row_series[feature]
                c_cost = row_series[cost_metric_for_visualization]
                # Check if this point (value, cost) was among the min-cost candidates for its value
                if (f_val, c_cost) not in all_candidate_min_value_cost_pairs:
                    other_feature_values.append(float(f_val))
                    other_visual_costs.append(float(c_cost))
                    other_plan_names.append(row_series['plan_name'] if 'plan_name' in row_series else "Unknown")

        # Count points for logging
        frontier_points_count = len(frontier_feature_values)
        excluded_points_count = len(excluded_feature_values)
        other_points_count = len(other_feature_values)
        unlimited_count = 1 if has_unlimited_data else 0

        # Verbose logging disabled to prevent spam from frequent polling
        # logger.info(f"Feature {feature}: Found {frontier_points_count} frontier, {excluded_points_count} excluded, {unlimited_count} unlimited points")

        # For the JS chart, we will pass frontier and excluded points
        js_chart_values = []
        js_chart_costs = []
        js_chart_plan_names = []
        js_chart_is_frontier = []
        js_chart_is_excluded = []

        for i in range(len(frontier_feature_values)):
            js_chart_values.append(frontier_feature_values[i])
            js_chart_costs.append(frontier_visual_costs[i])
            js_chart_plan_names.append(frontier_plan_names[i])
            js_chart_is_frontier.append(True)
            js_chart_is_excluded.append(False)

        for i in range(len(excluded_feature_values)):
            js_chart_values.append(excluded_feature_values[i])
            js_chart_costs.append(excluded_visual_costs[i])
            js_chart_plan_names.append(excluded_plan_names[i])
            js_chart_is_frontier.append(False)
            js_chart_is_excluded.append(True)

        # Only add to feature_frontier_data if we have frontier or excluded points, or an unlimited point
        if frontier_points_count > 0 or excluded_points_count > 0 or has_unlimited_data:
            # Ensure frontier points are sorted by feature value to guarantee proper line connection
            sorted_frontier_values = []
            sorted_frontier_costs = []
            sorted_frontier_names = []

            # Create sorted lists based on feature values
            sorted_indices = sorted(range(len(frontier_feature_values)), key=lambda i: frontier_feature_values[i])
            for idx in sorted_indices:
                sorted_frontier_values.append(frontier_feature_values[idx])
                sorted_frontier_costs.append(frontier_visual_costs[idx])
                sorted_frontier_names.append(frontier_plan_names[idx])

            # Check if we need to add a (0,0) point for proper charting
            if len(sorted_frontier_values) > 0 and sorted_frontier_values[0] > 0:
                sorted_frontier_values.insert(0, 0)
                sorted_frontier_costs.insert(0, 0)
                sorted_frontier_names.insert(0, "Free Baseline")
                logger.info(f"Added (0,0) point to sorted frontier values for chart rendering")

            feature_frontier_data[feature] = {
                # Pass only the combined frontier/excluded lists to JS under generic 'all_...' keys
                'all_values': js_chart_values, 
                'all_contributions': js_chart_costs, # JS expects all_contributions for costs
                'all_is_frontier': js_chart_is_frontier,
                'all_is_excluded': js_chart_is_excluded,
                # all_is_unlimited will be determined in JS based on the separate unlimited point data
                'all_plan_names': js_chart_plan_names,

                # Use the sorted frontier values for proper line connection
                'frontier_values': sorted_frontier_values,
                'frontier_contributions': sorted_frontier_costs, 
                'frontier_plan_names': sorted_frontier_names,
                'excluded_values': excluded_feature_values,
                'excluded_contributions': excluded_visual_costs, 
                'excluded_plan_names': excluded_plan_names,

                'has_unlimited': has_unlimited_data,
                'unlimited_value': unlimited_min_visual_cost if has_unlimited_data else None, 
                'unlimited_plan': unlimited_min_plan if has_unlimited_data else None
            }
        else:
            logger.info(f"Skipping chart data for feature {feature}: no frontier, excluded, or unlimited points")

    logger.info("Finished populating chart data.")
    return feature_frontier_data, all_chart_data, visual_frontiers_for_residual_table

def prepare_residual_analysis_data(all_chart_data, visual_frontiers_for_residual_table, 
                                 core_continuous_features, feature_display_names, feature_units, unlimited_flags):
    """
    Prepare data for the residual fee analysis table.

    Args:
        all_chart_data: Dictionary with frontier plan series data
        visual_frontiers_for_residual_table: Dictionary with frontier point tuples
        core_continuous_features: List of core continuous features
        feature_display_names: Dictionary mapping features to display names
        feature_units: Dictionary mapping features to units
        unlimited_flags: Dictionary mapping features to unlimited flags

    Returns:
        List of dictionaries with residual analysis data for each feature
    """
    from .report_utils import get_richness_score, estimate_value_on_visual_frontier, format_plan_specs_display_string

    residual_analysis_table_data = []
    logger.info("Starting Residual Original Fee Analysis.")

    cost_metric_for_visualization = 'original_fee'

    for feature_analyzed in core_continuous_features:
        # Skip features missing required data
        if feature_analyzed not in all_chart_data or \
           'actual_frontier_plans_series' not in all_chart_data[feature_analyzed] or \
           not all_chart_data[feature_analyzed]['actual_frontier_plans_series']:
            logger.info(f"Skipping residual analysis for '{feature_analyzed}': missing frontier data.")
            continue
        if feature_analyzed not in feature_display_names:
            logger.info(f"Skipping residual analysis for '{feature_analyzed}': missing display name.")
            continue

        # 1. Get the list of plans on the visual frontier for the current feature_analyzed
        current_feature_frontier_plans = all_chart_data[feature_analyzed]['actual_frontier_plans_series']

        # 2. Find plans in this list with the minimum value for feature_analyzed
        min_val_for_feature_on_frontier = min(p[feature_analyzed] for p in current_feature_frontier_plans)

        candidate_target_plans = [
            p for p in current_feature_frontier_plans if p[feature_analyzed] == min_val_for_feature_on_frontier
        ]

        if not candidate_target_plans:
            logger.info(f"Skipping residual analysis for '{feature_analyzed}': no min-value plans found.")
            continue

        # 3. Tie-breaking:
        #    a) Lowest 'original_fee'
        candidate_target_plans.sort(key=lambda p: p[cost_metric_for_visualization])

        min_original_fee_for_candidates = candidate_target_plans[0][cost_metric_for_visualization]

        # Filter by this min_original_fee for the next tie-breaking step
        plans_tied_on_fee = [
            p for p in candidate_target_plans if p[cost_metric_for_visualization] == min_original_fee_for_candidates
        ]

        target_plan_series = None
        if len(plans_tied_on_fee) == 1:
            target_plan_series = plans_tied_on_fee[0]
        else:
            # b) Highest "richness score" for *other* core features
            other_core_features_for_richness = [f for f in core_continuous_features if f != feature_analyzed]
            best_target_plan_series_richness = None
            highest_richness_score = -1

            for plan_series_item in plans_tied_on_fee:
                current_richness = get_richness_score(plan_series_item, other_core_features_for_richness, 
                                                      core_continuous_features, unlimited_flags)
                if current_richness > highest_richness_score:
                    highest_richness_score = current_richness
                    best_target_plan_series_richness = plan_series_item

            target_plan_series = best_target_plan_series_richness if best_target_plan_series_richness is not None else plans_tied_on_fee[0]

        if target_plan_series is None:
             logger.warning(f"Could not select a target plan from visual frontier for {feature_analyzed}. Skipping.")
             continue

        # 4. Prepare data for the table row
        target_plan_name_display = target_plan_series['plan_name'] if 'plan_name' in target_plan_series else "Unknown"
        plan_specs_string = format_plan_specs_display_string(target_plan_series, core_continuous_features, 
                                                           feature_display_names, feature_units, unlimited_flags)

        # Cost of the analyzed feature is its cost on its own frontier (which is the point we selected)
        cost_of_analyzed_feature_on_frontier = target_plan_series[cost_metric_for_visualization]

        # Estimate combined cost of other core features
        combined_est_cost_others = 0
        all_other_costs_valid_for_sum = True
        other_core_features_list = [f for f in core_continuous_features if f != feature_analyzed]

        for f_other in other_core_features_list:
            if f_other not in visual_frontiers_for_residual_table or not visual_frontiers_for_residual_table[f_other]:
                logger.warning(f"Visual frontier for other feature '{f_other}' not found or empty. Cannot estimate its cost for plan '{target_plan_name_display}'.")
                all_other_costs_valid_for_sum = False
                continue

            val_f_other_in_target_plan = target_plan_series.get(f_other)
            frontier_tuples_f_other = visual_frontiers_for_residual_table[f_other]

            cost_component_f_other = None
            if pd.notna(val_f_other_in_target_plan):
                 cost_component_f_other = estimate_value_on_visual_frontier(val_f_other_in_target_plan, frontier_tuples_f_other)

            if cost_component_f_other is not None:
                combined_est_cost_others += cost_component_f_other
            else:
                logger.info(f"Could not estimate cost component for '{f_other}' (value: {val_f_other_in_target_plan}) in plan '{target_plan_name_display}'.")
                all_other_costs_valid_for_sum = False

        plan_total_original_fee = target_plan_series['original_fee']

        # Format the breakdown string
        combined_others_display = f"{combined_est_cost_others:,.0f}" if combined_est_cost_others is not None else "N/A (estimation incomplete)"
        if not all_other_costs_valid_for_sum and combined_est_cost_others == 0:
             combined_others_display = "N/A (estimation failed for all other features)"
        elif not all_other_costs_valid_for_sum and combined_est_cost_others > 0:
             combined_others_display = f"{combined_est_cost_others:,.0f} (estimation incomplete for some other features)"

        fee_breakdown_str = (
            f"{cost_of_analyzed_feature_on_frontier:,.0f} (Analyzed Feature)"
            f" + {combined_others_display} (Other Features)"
            f" = {plan_total_original_fee:,.0f} KRW (Plan Total)"
        )

        residual_analysis_table_data.append({
            'analyzed_feature_display': feature_display_names[feature_analyzed],
            'target_plan_name': target_plan_name_display,
            'plan_specs_string': plan_specs_string,
            'fee_breakdown_string': fee_breakdown_str
        })
        logger.info(f"Added row for '{feature_analyzed}' to residual table. Plan: {target_plan_name_display}.")

    logger.info(f"Completed residual analysis with {len(residual_analysis_table_data)} feature entries.")
    return residual_analysis_table_data

def prepare_multi_frontier_chart_data(df, multi_frontier_breakdown):
    """
    Prepare chart data for multi-feature frontier regression visualization.
    
    Args:
        df: DataFrame with plan data
        multi_frontier_breakdown: Coefficient breakdown from MultiFeatureFrontierRegression
        
    Returns:
        Dictionary with chart data for visualization
    """
    if not multi_frontier_breakdown:
        return {}
    
    # Feature display names for better visualization
    feature_display_names = {
        'basic_data_clean': 'Data (GB)',
        'voice_clean': 'Voice (min)',
        'message_clean': 'Messages',
        'tethering_gb': 'Tethering (GB)',
        'is_5g': '5G Support'
    }
    
    # Feature units for cost display
    feature_units = {
        'basic_data_clean': '/GB',
        'voice_clean': '/min',
        'message_clean': '/msg',
        'tethering_gb': '/GB',
        'is_5g': '/feature'
    }
    
    chart_data = {
        'method_info': {
            'name': 'Multi-Feature Frontier Regression',
            'description': 'Pure marginal costs extracted from frontier plans',
            'total_frontier_plans': multi_frontier_breakdown.get('total_frontier_plans', 0),
            'features_analyzed': multi_frontier_breakdown.get('features_analyzed', 0)
        },
        'cost_breakdown': {
            'base_cost': multi_frontier_breakdown.get('base_cost', 0),
            'feature_costs': []
        },
        'coefficient_comparison': {
            'features': [],
            'pure_costs': [],
            'display_names': [],
            'units': []
        },
        'frontier_plan_analysis': {
            'plan_count_by_feature': {},
            'cost_range_analysis': {}
        }
    }
    
    # Prepare cost breakdown data
    feature_costs = multi_frontier_breakdown.get('feature_costs', {})
    for feature, cost_info in feature_costs.items():
        display_name = feature_display_names.get(feature, feature)
        unit = feature_units.get(feature, '')
        coefficient = cost_info.get('coefficient', 0)
        
        chart_data['cost_breakdown']['feature_costs'].append({
            'feature': feature,
            'display_name': display_name,
            'coefficient': coefficient,
            'unit': unit,
            'cost_per_unit': coefficient
        })
        
        # Add to coefficient comparison
        chart_data['coefficient_comparison']['features'].append(feature)
        chart_data['coefficient_comparison']['pure_costs'].append(coefficient)
        chart_data['coefficient_comparison']['display_names'].append(display_name)
        chart_data['coefficient_comparison']['units'].append(unit)
    
    # Analyze frontier plan distribution
    if 'frontier_plans' in multi_frontier_breakdown:
        frontier_plans = multi_frontier_breakdown['frontier_plans']
        
        for feature in feature_costs.keys():
            if feature in df.columns:
                feature_values = df[feature].dropna()
                chart_data['frontier_plan_analysis']['plan_count_by_feature'][feature] = {
                    'total_plans': len(feature_values),
                    'unique_values': len(feature_values.unique()),
                    'min_value': float(feature_values.min()),
                    'max_value': float(feature_values.max()),
                    'avg_value': float(feature_values.mean())
                }
    
    return chart_data

def prepare_contamination_comparison_data(df, traditional_frontiers, multi_frontier_breakdown):
    """
    Prepare data to visualize the contamination problem and solution.
    
    Args:
        df: DataFrame with plan data
        traditional_frontiers: Traditional frontier data
        multi_frontier_breakdown: Multi-frontier regression results
        
    Returns:
        Dictionary with comparison data
    """
    comparison_data = {
        'contamination_examples': [],
        'coefficient_comparison': {
            'traditional': {},
            'multi_frontier': {},
            'improvement_metrics': {}
        },
        'prediction_accuracy': {
            'traditional_mae': 0,
            'multi_frontier_mae': 0,
            'improvement_percentage': 0
        }
    }
    
    # Example contamination cases
    feature_display_names = {
        'basic_data_clean': 'Data',
        'voice_clean': 'Voice',
        'message_clean': 'Messages',
        'tethering_gb': 'Tethering'
    }
    
    # Compare coefficients
    multi_feature_costs = multi_frontier_breakdown.get('feature_costs', {})
    
    for feature, cost_info in multi_feature_costs.items():
        display_name = feature_display_names.get(feature, feature)
        pure_cost = cost_info.get('coefficient', 0)
        
        comparison_data['coefficient_comparison']['multi_frontier'][feature] = {
            'display_name': display_name,
            'pure_cost': pure_cost,
            'method': 'Multi-Frontier (Pure)'
        }
    
    # Calculate improvement metrics
    if multi_feature_costs:
        total_features = len(multi_feature_costs)
        comparison_data['coefficient_comparison']['improvement_metrics'] = {
            'features_analyzed': total_features,
            'cross_contamination_eliminated': True,
            'pure_marginal_costs': True
        }
    
    return comparison_data

def prepare_frontier_plan_matrix_data(multi_frontier_breakdown):
    """
    Prepare data for frontier plan matrix visualization.
    
    Args:
        multi_frontier_breakdown: Multi-frontier regression results
        
    Returns:
        Dictionary with matrix data for visualization
    """
    if not multi_frontier_breakdown or 'frontier_plans' not in multi_frontier_breakdown:
        return {}
    
    matrix_data = {
        'plan_matrix': {
            'headers': ['Plan', 'Data (GB)', 'Voice (min)', 'Messages', 'Tethering (GB)', '5G', 'Price (₩)'],
            'rows': [],
            'total_plans': 0
        },
        'feature_diversity': {
            'data_range': {'min': 0, 'max': 0},
            'voice_range': {'min': 0, 'max': 0},
            'message_range': {'min': 0, 'max': 0},
            'tethering_range': {'min': 0, 'max': 0}
        },
        'regression_quality': {
            'total_frontier_plans': multi_frontier_breakdown.get('total_frontier_plans', 0),
            'features_analyzed': multi_frontier_breakdown.get('features_analyzed', 0),
            'base_cost': multi_frontier_breakdown.get('base_cost', 0)
        }
    }
    
    return matrix_data

def detect_change_points(feature_values, costs, min_segment_size=3):
    """
    Detect change points in cost structure using slope analysis.
    
    Args:
        feature_values: Array of feature values (sorted)
        costs: Array of corresponding costs
        min_segment_size: Minimum points per segment
        
    Returns:
        List of change point indices
    """
    if len(feature_values) < min_segment_size * 2:
        return []
    
    # Calculate local slopes using moving windows
    slopes = []
    for i in range(len(feature_values) - 1):
        if feature_values[i+1] != feature_values[i]:  # Avoid division by zero
            slope = (costs[i+1] - costs[i]) / (feature_values[i+1] - feature_values[i])
            slopes.append(slope)
        else:
            slopes.append(0)
    
    if len(slopes) < min_segment_size:
        return []
    
    # Find significant slope changes
    change_points = []
    window_size = max(2, min_segment_size // 2)
    
    for i in range(window_size, len(slopes) - window_size):
        # Calculate average slope before and after this point
        before_slope = np.mean(slopes[i-window_size:i])
        after_slope = np.mean(slopes[i:i+window_size])
        
        # Check if there's a significant change (>20% difference)
        if abs(before_slope) > 0 and abs(after_slope - before_slope) / abs(before_slope) > 0.2:
            change_points.append(i)
    
    # Merge nearby change points
    if len(change_points) > 1:
        filtered_points = [change_points[0]]
        for cp in change_points[1:]:
            if cp - filtered_points[-1] >= min_segment_size:
                filtered_points.append(cp)
        change_points = filtered_points
    
    return change_points

def fit_piecewise_linear(feature_values, costs, change_points):
    """
    Fit piecewise linear model with detected change points.
    
    Args:
        feature_values: Array of feature values
        costs: Array of costs
        change_points: List of change point indices
        
    Returns:
        Dictionary with segment information
    """
    segments = []
    
    # Create segments based on change points
    start_idx = 0
    segment_boundaries = change_points + [len(feature_values)]
    
    for end_idx in segment_boundaries:
        if end_idx <= start_idx:
            continue
            
        # Extract segment data
        seg_features = feature_values[start_idx:end_idx]
        seg_costs = costs[start_idx:end_idx]
        
        if len(seg_features) < 2:
            start_idx = end_idx
            continue
        
        # Fit linear regression for this segment
        try:
            # Calculate slope and intercept
            x_mean = np.mean(seg_features)
            y_mean = np.mean(seg_costs)
            
            numerator = np.sum((seg_features - x_mean) * (seg_costs - y_mean))
            denominator = np.sum((seg_features - x_mean) ** 2)
            
            if denominator > 0:
                slope = numerator / denominator
                intercept = y_mean - slope * x_mean
            else:
                slope = 0
                intercept = y_mean
            
            segments.append({
                'start_feature': float(seg_features[0]),
                'end_feature': float(seg_features[-1]),
                'start_idx': start_idx,
                'end_idx': end_idx - 1,
                'slope': float(slope),
                'intercept': float(intercept),
                'points': len(seg_features),
                'marginal_cost': float(slope)  # Slope is the marginal cost
            })
            
        except Exception as e:
            logger.warning(f"Error fitting segment {start_idx}-{end_idx}: {e}")
        
        start_idx = end_idx
    
    return segments

def prepare_marginal_cost_frontier_data(df, multi_frontier_breakdown, core_continuous_features):
    """
    Prepare feature frontier charts using PIECEWISE LINEAR model for realistic marginal costs.
    This shows economies of scale with different marginal costs across feature ranges.
    
    Args:
        df: DataFrame with plan data
        multi_frontier_breakdown: Pure marginal costs from MultiFeatureFrontierRegression
        core_continuous_features: List of features to visualize
        
    Returns:
        Dictionary with piecewise marginal cost frontier data for visualization
    """
    if not multi_frontier_breakdown or not multi_frontier_breakdown.get('feature_costs'):
        logger.warning("No multi-frontier breakdown data available for marginal cost frontiers")
        return {}

    logger.info("Preparing PIECEWISE LINEAR marginal cost frontier charts with economies of scale")
    
    # Feature display configuration
    feature_display_names = {
        'basic_data_clean': 'Data (GB)',
        'voice_clean': 'Voice (min)', 
        'message_clean': 'Messages',
        'tethering_gb': 'Tethering (GB)',
        'is_5g': '5G Support'
    }
    
    feature_units = {
        'basic_data_clean': 'KRW/GB',
        'voice_clean': 'KRW/min',
        'message_clean': 'KRW/msg', 
        'tethering_gb': 'KRW/GB',
        'is_5g': 'KRW/feature'
    }
    
    marginal_cost_frontier_data = {}
    feature_costs = multi_frontier_breakdown.get('feature_costs', {})
    base_cost = multi_frontier_breakdown.get('base_cost', 0)
    
    for feature in core_continuous_features:
        if feature not in df.columns or feature not in feature_costs:
            logger.warning(f"Feature {feature} not available for marginal cost frontier")
            continue
            
        # Get pure marginal cost coefficient (as baseline)
        cost_info = feature_costs[feature]
        pure_coefficient = cost_info.get('coefficient', 0)
        
        if pure_coefficient <= 0:
            logger.warning(f"Invalid coefficient for {feature}: {pure_coefficient}")
            continue
            
        logger.info(f"Creating PIECEWISE LINEAR frontier for {feature} with base coefficient: {pure_coefficient}")
        
        # Get actual frontier data points using SAME METHODOLOGY as original frontier
        feature_values = df[feature].dropna()
        if feature_values.empty:
            continue
        
        from modules.cost_spec import create_robust_monotonic_frontier, UNLIMITED_FLAGS
        
        # Apply same unlimited handling as original system
        unlimited_flag = UNLIMITED_FLAGS.get(feature)
        
        # Process non-unlimited plans with robust monotonic frontier
        if unlimited_flag and unlimited_flag in df.columns:
            df_non_unlimited = df[(df[unlimited_flag] == 0) & df['original_fee'].notna()].copy()
        else:
            df_non_unlimited = df[df['original_fee'].notna()].copy()
        
        if df_non_unlimited.empty:
            logger.warning(f"No non-unlimited plans found for {feature}")
            continue
        
        # Use SAME monotonic frontier logic with 1 KRW/feature rule
        robust_frontier = create_robust_monotonic_frontier(df_non_unlimited, feature, 'original_fee')
        
        if robust_frontier.empty:
            logger.warning(f"No valid monotonic frontier for {feature}")
            continue
        
        # Convert robust frontier to data points for piecewise analysis
        frontier_data_points = [(val, cost) for val, cost in robust_frontier.items()]
        
        # Handle unlimited plans separately (same as original system)
        unlimited_endpoint = None
        if unlimited_flag and unlimited_flag in df.columns:
            unlimited_plans = df[(df[unlimited_flag] == 1) & df['original_fee'].notna()]
            if not unlimited_plans.empty:
                min_unlimited_cost = unlimited_plans['original_fee'].min()
                # Find max feature value for unlimited endpoint
                max_feature_val = df[feature].max() if not df[feature].empty else 100
                unlimited_endpoint = (max_feature_val * 2, min_unlimited_cost)  # Use 2x max as "unlimited"
                logger.info(f"✅ Found unlimited {feature} plans: cheapest at ₩{min_unlimited_cost}")
        
        logger.info(f"✅ Applied monotonic filtering for {feature}: {len(frontier_data_points)} points (was {len(feature_values.unique())} raw points)")
        
        if len(frontier_data_points) < 3:
            # Fallback to linear model if insufficient data
            logger.warning(f"Insufficient frontier data for {feature}, using linear model")
            min_val = feature_values.min()
            max_val = feature_values.max()
            feature_points = np.linspace(min_val, max_val, 20)
            
            frontier_points = []
            for feature_val in feature_points:
                pure_cost = base_cost + (feature_val * pure_coefficient)
                frontier_points.append({
                    'feature_value': float(feature_val),
                    'pure_cost': float(pure_cost),
                    'marginal_cost': float(pure_coefficient),
                    'cumulative_cost': float(feature_val * pure_coefficient),
                    'segment': 'linear'
                })
        else:
            # PIECEWISE LINEAR ANALYSIS with monotonic + unlimited data
            
            # Add unlimited endpoint if available
            if unlimited_endpoint:
                frontier_data_points.append(unlimited_endpoint)
                logger.info(f"✅ Added unlimited endpoint for {feature}: {unlimited_endpoint}")
            
            frontier_features = np.array([p[0] for p in frontier_data_points])
            frontier_costs = np.array([p[1] for p in frontier_data_points])
            
            # Detect change points in the FILTERED monotonic frontier
            change_points = detect_change_points(frontier_features, frontier_costs)
            logger.info(f"Detected {len(change_points)} change points for {feature} (monotonic data)")
            
            # Fit piecewise linear segments on FILTERED data
            segments = fit_piecewise_linear(frontier_features, frontier_costs, change_points)
            logger.info(f"Fitted {len(segments)} segments for {feature} (with monotonicity + 1KRW rule)")
            
            # Generate smooth piecewise frontier points
            frontier_points = []
            min_val = frontier_features.min()
            max_val = frontier_features.max()
            
            # Create detailed points for visualization
            num_points_per_segment = 15
            
            for seg_idx, segment in enumerate(segments):
                seg_start = segment['start_feature']
                seg_end = segment['end_feature']
                seg_slope = segment['marginal_cost']
                seg_intercept = segment['intercept']
                
                # Generate points for this segment
                if seg_idx == len(segments) - 1:  # Last segment
                    seg_points = np.linspace(seg_start, seg_end, num_points_per_segment)
                else:
                    seg_points = np.linspace(seg_start, seg_end, num_points_per_segment)[:-1]  # Exclude end to avoid duplication
                
                for feature_val in seg_points:
                    # Piecewise linear cost calculation
                    segment_cost = seg_intercept + (feature_val * seg_slope)
                    
                    # Cumulative cost calculation (integral of piecewise function)
                    cumulative_cost = 0
                    for prev_seg in segments:
                        if prev_seg['end_feature'] < feature_val:
                            # Full segment contribution
                            seg_length = prev_seg['end_feature'] - prev_seg['start_feature']
                            avg_cost = prev_seg['marginal_cost']
                            cumulative_cost += seg_length * avg_cost
                        elif prev_seg['start_feature'] <= feature_val <= prev_seg['end_feature']:
                            # Partial segment contribution
                            seg_length = feature_val - prev_seg['start_feature']
                            avg_cost = prev_seg['marginal_cost']
                            cumulative_cost += seg_length * avg_cost
                            break
                    
                    frontier_points.append({
                        'feature_value': float(feature_val),
                        'pure_cost': float(segment_cost),
                        'marginal_cost': float(seg_slope),  # Marginal cost for this segment
                        'cumulative_cost': float(cumulative_cost),
                        'segment': f'segment_{seg_idx}',
                        'segment_info': {
                            'index': seg_idx,
                            'start': float(seg_start),
                            'end': float(seg_end),
                            'slope': float(seg_slope),
                            'range': f"{seg_start:.1f}-{seg_end:.1f}"
                        }
                    })
        
        # Find actual plans for comparison
        actual_plan_points = []
        unique_feature_vals = sorted(feature_values.unique())
        
        sample_points = []
        if len(unique_feature_vals) > 10:
            step = len(unique_feature_vals) // 10
            sample_points = unique_feature_vals[::step]
        else:
            sample_points = unique_feature_vals
            
        for feature_val in sample_points[:15]:
            matching_plans = df[df[feature] == feature_val]
            if not matching_plans.empty:
                cheapest_idx = matching_plans['original_fee'].idxmin()
                cheapest_plan = matching_plans.loc[cheapest_idx]
                
                # Find which segment this point belongs to
                segment_info = "linear"
                if len(segments) > 1:
                    for seg in segments:
                        if seg['start_feature'] <= feature_val <= seg['end_feature']:
                            segment_info = f"Segment {seg['start_feature']:.1f}-{seg['end_feature']:.1f} (₩{seg['marginal_cost']:.0f}/{feature_units.get(feature, 'unit').split('/')[-1]})"
                            break
                
                actual_plan_points.append({
                    'feature_value': float(feature_val),
                    'actual_cost': float(cheapest_plan['original_fee']),
                    'plan_name': cheapest_plan.get('plan_name', 'Unknown'),
                    'segment_info': segment_info
                })
        
        # Calculate comprehensive cost analysis
        if frontier_points:
            marginal_costs = [p['marginal_cost'] for p in frontier_points]
            cumulative_costs = [p['cumulative_cost'] for p in frontier_points]
            
            marginal_cost_frontier_data[feature] = {
                'feature_name': feature,
                'display_name': feature_display_names.get(feature, feature),
                'unit': feature_units.get(feature, ''),
                'pure_coefficient': float(pure_coefficient),
                'base_cost': float(base_cost),
                'frontier_points': frontier_points,
                'actual_plan_points': actual_plan_points,
                'piecewise_info': {
                    'is_piecewise': len(segments) > 1 if 'segments' in locals() else False,
                    'num_segments': len(segments) if 'segments' in locals() else 1,
                    'segments': segments if 'segments' in locals() else [],
                    'change_points_detected': len(change_points) if 'change_points' in locals() else 0
                },
                'feature_range': {
                    'min': float(feature_values.min()),
                    'max': float(feature_values.max()),
                    'unique_values': len(feature_values.unique()),
                    'filtered_frontier_points': len(frontier_data_points)
                },
                'cost_analysis': {
                    'min_marginal_cost': float(min(marginal_costs)),
                    'max_marginal_cost': float(max(marginal_costs)),
                    'avg_marginal_cost': float(np.mean(marginal_costs)),
                    'marginal_cost_range': float(max(marginal_costs) - min(marginal_costs)),
                    'economies_of_scale': float(max(marginal_costs) - min(marginal_costs)) > 0,
                    'total_cost_range': float(max(cumulative_costs) - min(cumulative_costs)) if cumulative_costs else 0
                }
            }
            
            if len(segments) > 1:
                logger.info(f"✅ PIECEWISE model for {feature}: {len(segments)} segments, marginal cost range: ₩{min(marginal_costs):.0f}-₩{max(marginal_costs):.0f}")
            else:
                logger.info(f"📊 Linear model for {feature}: constant marginal cost ₩{pure_coefficient:.0f}")
        
        logger.info(f"Prepared piecewise frontier for {feature}: {len(frontier_points)} points, {len(actual_plan_points)} actual plans")
    
    logger.info(f"✅ Completed PIECEWISE LINEAR frontier preparation for {len(marginal_cost_frontier_data)} features")
    return marginal_cost_frontier_data