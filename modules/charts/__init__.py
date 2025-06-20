"""
Charts Module

This module contains chart preparation functions organized by chart type.
Extracted from report_charts_legacy.py for better modularity.

Modules:
- piecewise_utils: Piecewise regression utilities
- feature_frontier: Feature frontier chart data preparation and residual analysis
- multi_frontier: Multi-frontier chart data preparation and contamination analysis
- marginal_cost: Marginal cost frontier chart data preparation and granular cost analysis
"""

# Import from piecewise_utils
from .piecewise_utils import (
    detect_change_points,
    fit_piecewise_linear,
    fit_piecewise_linear_segments
)

# Import from feature_frontier
from .feature_frontier import (
    prepare_feature_frontier_data,
    prepare_residual_analysis_data
)

# Import from multi_frontier
from .multi_frontier import (
    prepare_multi_frontier_chart_data,
    prepare_contamination_comparison_data,
    prepare_frontier_plan_matrix_data
)

# Import from marginal_cost
from .marginal_cost import (
    prepare_marginal_cost_frontier_data,
    create_granular_segments_with_intercepts,
    calculate_granular_piecewise_cost_with_intercepts,
    prepare_granular_marginal_cost_frontier_data
)

# Export all functions
__all__ = [
    # Piecewise utilities
    'detect_change_points',
    'fit_piecewise_linear', 
    'fit_piecewise_linear_segments',
    # Feature frontier functions
    'prepare_feature_frontier_data',
    'prepare_residual_analysis_data',
    # Multi-frontier functions
    'prepare_multi_frontier_chart_data',
    'prepare_contamination_comparison_data',
    'prepare_frontier_plan_matrix_data',
    # Marginal cost functions
    'prepare_marginal_cost_frontier_data',
    'create_granular_segments_with_intercepts',
    'calculate_granular_piecewise_cost_with_intercepts',
    'prepare_granular_marginal_cost_frontier_data'
] 