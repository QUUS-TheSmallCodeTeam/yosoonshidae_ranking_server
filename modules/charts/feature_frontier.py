"""
Feature Frontier Chart Module

This module handles feature frontier chart data preparation and residual analysis.
Extracted from report_charts_legacy.py for better modularity.

Functions:
- prepare_feature_frontier_data: Main function for feature frontier chart data preparation
- prepare_residual_analysis_data: Residual fee analysis for frontier plans
"""

import logging
import numpy as np
import pandas as pd
from ..report_utils import UNLIMITED_FLAGS

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
    from ..report_utils import get_richness_score, estimate_value_on_visual_frontier, format_plan_specs_display_string

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
            if val_f_other_in_target_plan is None:
                logger.warning(f"Feature '{f_other}' value not found in target plan '{target_plan_name_display}'. Cannot estimate its cost.")
                all_other_costs_valid_for_sum = False
                continue

            est_cost_f_other = estimate_value_on_visual_frontier(val_f_other_in_target_plan, visual_frontiers_for_residual_table[f_other])
            if est_cost_f_other is None:
                logger.warning(f"Could not estimate cost for feature '{f_other}' with value {val_f_other_in_target_plan} on its visual frontier for plan '{target_plan_name_display}'.")
                all_other_costs_valid_for_sum = False
                continue

            combined_est_cost_others += est_cost_f_other

        # Calculate residual cost
        residual_cost = None
        if all_other_costs_valid_for_sum:
            residual_cost = cost_of_analyzed_feature_on_frontier - combined_est_cost_others

        # Add row to the table data
        residual_analysis_table_data.append({
            'feature_analyzed': feature_analyzed,
            'feature_display_name': feature_display_names[feature_analyzed],
            'target_plan_name': target_plan_name_display,
            'plan_specs': plan_specs_string,
            'cost_of_analyzed_feature': cost_of_analyzed_feature_on_frontier,
            'combined_cost_others': combined_est_cost_others if all_other_costs_valid_for_sum else None,
            'residual_cost': residual_cost,
            'residual_cost_valid': all_other_costs_valid_for_sum
        })

        logger.info(f"Residual analysis for '{feature_analyzed}': Plan '{target_plan_name_display}', Cost {cost_of_analyzed_feature_on_frontier}, Others {combined_est_cost_others if all_other_costs_valid_for_sum else 'N/A'}, Residual {residual_cost if residual_cost is not None else 'N/A'}")

    logger.info(f"Completed residual analysis for {len(residual_analysis_table_data)} features.")
    return residual_analysis_table_data 