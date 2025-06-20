"""
Regression Analysis Module

This module contains regression analysis functionality for mobile plan ranking.

Classes:
- FullDatasetMultiFeatureRegression: Main facade class for full dataset analysis
- MultiFeatureFrontierRegression: Multi-feature frontier regression facade class
- FullDatasetRegressionCore: Core regression functionality
- MulticollinearityHandler: Multicollinearity detection and handling
- ModelValidator: Comprehensive model validation
- FrontierAnalyzer: Frontier plan collection and analysis
- MultiFeatureRegressor: Multi-feature regression analysis

Both main classes have been refactored into facade patterns using focused sub-modules.
"""

from .full_dataset import FullDatasetMultiFeatureRegression
from .multi_feature import MultiFeatureFrontierRegression

# Export refactored sub-modules for full_dataset
from .regression_core import FullDatasetRegressionCore
from .multicollinearity_handler import MulticollinearityHandler
from .model_validation import ModelValidator

# Export refactored sub-modules for multi_feature
from .frontier_analysis import FrontierAnalyzer
from .multi_regression import MultiFeatureRegressor

__all__ = [
    'FullDatasetMultiFeatureRegression',
    'MultiFeatureFrontierRegression',
    'FullDatasetRegressionCore',
    'MulticollinearityHandler',
    'ModelValidator',
    'FrontierAnalyzer',
    'MultiFeatureRegressor'
] 