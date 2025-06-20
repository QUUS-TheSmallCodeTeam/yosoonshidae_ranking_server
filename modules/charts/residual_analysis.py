"""
Residual Analysis Module

This module handles residual fee analysis calculations.
Extracted from feature_frontier.py for better modularity.

Functions:
- prepare_residual_analysis_data: Residual fee analysis for frontier plans
"""

import logging

# Configure logging
logger = logging.getLogger(__name__)

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