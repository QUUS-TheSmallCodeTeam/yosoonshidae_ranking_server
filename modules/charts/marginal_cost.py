"""
Marginal Cost Chart Module (Refactored)

This module serves as a facade for marginal cost functionality, importing from smaller modules.
Refactored from original 960-line file into 3 focused modules for better maintainability.

Modules:
- basic_marginal_cost: Basic piecewise linear marginal cost frontier charts
- granular_segments: Granular segment creation and calculation functions  
- comprehensive_analysis: Comprehensive granular analysis using entire dataset
"""

# Import functions from refactored modules
from .basic_marginal_cost import prepare_marginal_cost_frontier_data
from .granular_segments import (
    create_granular_segments_with_intercepts,
    calculate_granular_piecewise_cost_with_intercepts
)
from .comprehensive_analysis import prepare_granular_marginal_cost_frontier_data

# Re-export all functions for backward compatibility
__all__ = [
    'prepare_marginal_cost_frontier_data',
    'create_granular_segments_with_intercepts', 
    'calculate_granular_piecewise_cost_with_intercepts',
    'prepare_granular_marginal_cost_frontier_data'
] 