"""
Cost-Spec analysis modules for MVNO plan ranking.

This package contains functions for calculating cost-spec ratios
and ranking plans based on various methodologies.
"""

from .ratio import (
    calculate_cs_ratio,
    rank_plans_by_cs,
    calculate_cs_ratio_enhanced,
    rank_plans_by_cs_enhanced
)

__all__ = [
    'calculate_cs_ratio',
    'rank_plans_by_cs',
    'calculate_cs_ratio_enhanced',
    'rank_plans_by_cs_enhanced'
] 