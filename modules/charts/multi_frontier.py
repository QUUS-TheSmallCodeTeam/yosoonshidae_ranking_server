"""
Multi-Frontier Chart Module

This module handles multi-frontier chart data preparation and contamination analysis.
Extracted from report_charts_legacy.py for better modularity.

Functions:
- prepare_multi_frontier_chart_data: Multi-feature frontier regression visualization
- prepare_contamination_comparison_data: Contamination problem visualization
- prepare_frontier_plan_matrix_data: Frontier plan matrix visualization
"""

import logging

# Configure logging
logger = logging.getLogger(__name__)

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