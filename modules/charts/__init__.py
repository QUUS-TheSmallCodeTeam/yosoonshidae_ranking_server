"""
Charts Module

Provides chart data preparation functions for various visualization types.
Modularized for better maintainability and focused responsibilities.

Modules:
- feature_frontier: Feature frontier chart data preparation
- multi_frontier: Multi-frontier analysis and contamination comparison
- marginal_cost: Marginal cost frontier charts (refactored into 3 sub-modules)
  - basic_marginal_cost: Basic piecewise linear frontier charts
  - granular_segments: Granular segment creation and calculation
  - comprehensive_analysis: Comprehensive analysis using entire dataset
- piecewise_utils: Piecewise linear regression utilities
"""

from .feature_frontier import (
    prepare_feature_frontier_data,
    prepare_residual_analysis_data
)

from .multi_frontier import (
    prepare_multi_frontier_chart_data,
    prepare_contamination_comparison_data,
    prepare_frontier_plan_matrix_data
)

from .marginal_cost import (
    prepare_marginal_cost_frontier_data,
    create_granular_segments_with_intercepts,
    calculate_granular_piecewise_cost_with_intercepts,
    prepare_granular_marginal_cost_frontier_data
)

from .piecewise_utils import (
    detect_change_points,
    fit_piecewise_linear,
    fit_piecewise_linear_segments
)

# Also export individual sub-modules for direct access
from . import basic_marginal_cost
from . import granular_segments
from . import comprehensive_analysis

__all__ = [
    # Feature frontier functions
    'prepare_feature_frontier_data',
    'prepare_residual_analysis_data',
    
    # Multi-frontier functions
    'prepare_multi_frontier_chart_data',
    'prepare_contamination_comparison_data', 
    'prepare_frontier_plan_matrix_data',
    
    # Marginal cost functions (main interface)
    'prepare_marginal_cost_frontier_data',
    'create_granular_segments_with_intercepts',
    'calculate_granular_piecewise_cost_with_intercepts',
    'prepare_granular_marginal_cost_frontier_data',
    
    # Piecewise utilities
    'detect_change_points',
    'fit_piecewise_linear',
    'fit_piecewise_linear_segments',
    
    # Sub-modules for direct access
    'basic_marginal_cost',
    'granular_segments', 
    'comprehensive_analysis'
] 