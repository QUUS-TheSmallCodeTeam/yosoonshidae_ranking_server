"""
Regression Analysis Module

This module contains various regression analysis techniques and multicollinearity handling.

Key Components:
- FullDatasetMultiFeatureRegression: Main facade class for full dataset analysis
- FullDatasetRegressionCore: Core regression analysis and outlier removal
- MulticollinearityHandler: Handles correlation analysis and coefficient redistribution
- CommonalityAnalyzer: All Possible Subsets Regression for variance decomposition
- ModelValidator: Comprehensive model validation and quality metrics
- EfficiencyFrontierRegression: Pareto-optimal plan extraction and regression

Functions:
- All major regression and analysis functions accessible through the main classes
"""

from .full_dataset import FullDatasetMultiFeatureRegression
from .regression_core import FullDatasetRegressionCore
from .multicollinearity_handler import MulticollinearityHandler
from .commonality_analysis import CommonalityAnalyzer
from .model_validation import ModelValidator
from .efficiency_frontier import EfficiencyFrontierRegression

# Import frontier analysis module
from .frontier_analysis import *
from .multi_regression import *

__all__ = [
    # Main classes
    'FullDatasetMultiFeatureRegression',
    'FullDatasetRegressionCore', 
    'MulticollinearityHandler',
    'CommonalityAnalyzer',
    'ModelValidator',
    'EfficiencyFrontierRegression',
    
    # Frontier analysis functions
    'collect_feature_frontiers',
    'solve_multi_frontier_coefficients',
    'MultiFeatureFrontierRegression'
] 