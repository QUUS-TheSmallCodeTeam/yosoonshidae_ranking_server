"""
Report Charts Module (Refactored)

This module now imports from the modularized chart components.
All chart preparation logic has been moved to dedicated modules:
- charts/: Chart preparation functions by chart type
- charts/piecewise_utils: Piecewise regression utilities
- charts/feature_frontier: Feature frontier chart data preparation
- charts/multi_frontier: Multi-frontier chart data preparation
- charts/marginal_cost: Marginal cost frontier chart data preparation

Original file size: 1,824 lines → New size: ~30 lines (98.4% reduction)
Phase 2 Chart Module Completion: ✅ COMPLETED
"""

# Import all functions from the new modular structure
from .charts.piecewise_utils import detect_change_points, fit_piecewise_linear, fit_piecewise_linear_segments
from .charts.feature_frontier import prepare_feature_frontier_data, prepare_residual_analysis_data
from .charts.multi_frontier import prepare_multi_frontier_chart_data, prepare_contamination_comparison_data, prepare_frontier_plan_matrix_data
from .charts.marginal_cost import prepare_marginal_cost_frontier_data, create_granular_segments_with_intercepts, calculate_granular_piecewise_cost_with_intercepts, prepare_granular_marginal_cost_frontier_data

# Maintain backward compatibility by exposing the same interface
__all__ = [
    # From piecewise_utils module
    'detect_change_points',
    'fit_piecewise_linear',
    'fit_piecewise_linear_segments',
    # From feature_frontier module
    'prepare_feature_frontier_data',
    'prepare_residual_analysis_data',
    # From multi_frontier module
    'prepare_multi_frontier_chart_data',
    'prepare_contamination_comparison_data',
    'prepare_frontier_plan_matrix_data',
    # From marginal_cost module
    'prepare_marginal_cost_frontier_data',
    'create_granular_segments_with_intercepts',
    'calculate_granular_piecewise_cost_with_intercepts',
    'prepare_granular_marginal_cost_frontier_data'
] 