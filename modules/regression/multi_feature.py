"""
Multi-Feature Frontier Regression Module (Facade)

This module serves as a facade for the refactored multi-feature frontier regression functionality.
The original large class has been decomposed into focused modules for better maintainability.

Modules:
- frontier_analysis: Frontier plan collection and feature analysis
- multi_regression: Multi-feature regression analysis and coefficient calculation

Original class maintained for backward compatibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Import from refactored modules
from .frontier_analysis import FrontierAnalyzer
from .multi_regression import MultiFeatureRegressor
from .model_validation import ModelValidator

# Configure logging
logger = logging.getLogger(__name__)

# Import constants from parent modules
from ..config import CORE_FEATURES, UNLIMITED_FLAGS

class MultiFeatureFrontierRegression:
    """
    Multi-Feature Frontier Regression implementation (Facade).
    
    This class serves as a facade for the refactored functionality, maintaining
    backward compatibility while using the new modular architecture.
    
    Solves the cross-contamination problem by:
    1. Collecting plans from all feature frontiers
    2. Performing multi-feature regression on complete feature vectors
    3. Extracting pure marginal costs for each feature
    """
    
    def __init__(self, features=None):
        """
        Initialize the multi-feature frontier regression analyzer.
        
        Args:
            features: List of features to analyze. If None, uses CORE_FEATURES.
        """
        # Initialize component modules
        self.frontier_analyzer = FrontierAnalyzer(features)
        self.regressor = MultiFeatureRegressor()
        self.model_validator = ModelValidator()
        
        # Maintain backward compatibility with existing attributes
        self.features = self.frontier_analyzer.features
        self.frontier_plans = None
        self.coefficients = None
        self.min_increments = {}
        self.feature_frontiers = {}
        self.multicollinearity_detected = False
        self.correlation_matrix = None
        self.multicollinearity_fixes = {}
        self.unconstrained_coefficients = None
        self.coefficient_bounds = None
        
    def collect_all_frontier_plans(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Collect all plans that appear in any feature frontier.
        Delegated to frontier analyzer module.
        """
        result = self.frontier_analyzer.collect_all_frontier_plans(df)
        
        # Update instance attributes for backward compatibility
        self.frontier_plans = self.frontier_analyzer.frontier_plans
        self.feature_frontiers = self.frontier_analyzer.feature_frontiers
        
        return result
    
    def calculate_min_increments(self, df: pd.DataFrame):
        """
        Calculate minimum increments for feature normalization.
        Delegated to frontier analyzer module.
        """
        self.frontier_analyzer.calculate_min_increments(df)
        
        # Update instance attributes for backward compatibility
        self.min_increments = self.frontier_analyzer.min_increments
    
    def solve_multi_feature_coefficients(self, df: pd.DataFrame) -> np.ndarray:
        """
        Solve for pure marginal costs using multi-feature regression.
        
        Args:
            df: DataFrame with plan data
            
        Returns:
            Array of coefficients [β₀, β₁, β₂, ...]
        """
        # Step 1: Collect frontier plans
        frontier_plans = self.collect_all_frontier_plans(df)
        
        if len(frontier_plans) < len(self.features) + 1:
            raise ValueError(f"Insufficient frontier plans ({len(frontier_plans)}) for {len(self.features)} features")
        
        # Step 2: Calculate minimum increments
        self.calculate_min_increments(df)
        
        # Step 3: Prepare feature matrix
        X, y, analysis_features = self.frontier_analyzer.prepare_feature_matrix(frontier_plans, df)
        
        # Step 4: Solve regression
        coefficients = self.regressor.solve_multi_feature_coefficients(X, y, analysis_features)
        
        # Update instance attributes for backward compatibility
        self.coefficients = self.regressor.coefficients
        self.multicollinearity_detected = self.regressor.multicollinearity_detected
        self.correlation_matrix = self.regressor.correlation_matrix
        self.multicollinearity_fixes = self.regressor.multicollinearity_fixes
        self.unconstrained_coefficients = self.regressor.unconstrained_coefficients
        self.coefficient_bounds = self.regressor.coefficient_bounds
        
        return self.coefficients

    def _solve_constrained_regression(self, X: np.ndarray, y: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Solve using constrained optimization.
        Delegated to regressor module.
        """
        return self.regressor._solve_constrained_regression(X, y, features)
    
    def _fix_multicollinearity_coefficients(self, coefficients: np.ndarray, features: List[str], 
                                           X: np.ndarray = None, y: np.ndarray = None) -> np.ndarray:
        """
        Fix multicollinearity by redistributing coefficients.
        Delegated to regressor module.
        """
        return self.regressor._fix_multicollinearity_coefficients(coefficients, features, X, y)
    
    def get_coefficient_breakdown(self) -> dict:
        """
        Get coefficient breakdown for visualization.
        
        Returns:
            Dictionary with coefficient information including both raw and constrained values
        """
        if self.coefficients is None:
            raise ValueError("Must solve coefficients first")
            
        analysis_features = [f for f in self.features if f not in UNLIMITED_FLAGS.values()]
        breakdown = self.regressor.get_coefficient_breakdown(self.frontier_plans, analysis_features)
        
        # Add multicollinearity information
        if self.multicollinearity_fixes:
            breakdown['multicollinearity_fixes'] = self.multicollinearity_fixes
            
        return breakdown

    # Model validation methods - delegated to model validator
    def validate_optimization_quality(self, X, y, coefficients, bounds):
        """최적화 품질 검증"""
        return self.model_validator.validate_optimization_quality(X, y, coefficients, bounds)

    def validate_economic_logic(self, df, coefficients, features):
        """경제적 타당성 검증"""
        return self.model_validator.validate_economic_logic(df, coefficients, features)

    def validate_prediction_power(self, df, features):
        """예측력 검증"""
        return self.model_validator.validate_prediction_power(df, features, self._solve_constrained_regression)

    def analyze_residuals(self, df, predicted, actual):
        """잔차 분석"""
        return self.model_validator.analyze_residuals(df, predicted, actual)

    def calculate_overall_validation_score(self, validation_report):
        """종합 검증 점수 계산"""
        return self.model_validator.calculate_overall_validation_score(validation_report)

    def comprehensive_model_validation(self, df, X, y, coefficients, features, bounds):
        """종합적인 모델 검증"""
        return self.model_validator.comprehensive_model_validation(
            df, X, y, coefficients, features, bounds, self._solve_constrained_regression
        )



