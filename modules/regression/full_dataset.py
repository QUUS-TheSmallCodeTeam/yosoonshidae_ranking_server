"""
Full Dataset Multi-Feature Regression Module (Facade)

This module serves as a facade for the refactored full dataset regression functionality.
The original large class has been decomposed into focused modules for better maintainability.

Modules:
- regression_core: Core regression analysis and outlier removal
- multicollinearity_handler: Multicollinearity detection and coefficient redistribution
- model_validation: Comprehensive model validation functionality

Original class maintained for backward compatibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Import from refactored modules
from .regression_core import FullDatasetRegressionCore
from .multicollinearity_handler import MulticollinearityHandler
from .model_validation import ModelValidator

# Import decorators
try:
    from ..performance.profiler import memory_monitor
    from ..performance.cache import cache_dataframe_operation
except ImportError:
    # Fallback decorators if performance module not available
    def memory_monitor(func):
        return func
    def cache_dataframe_operation(func):
        return func

# Configure logging
logger = logging.getLogger(__name__)

class FullDatasetMultiFeatureRegression:
    """
    Full dataset multi-feature regression analysis with comprehensive validation.
    
    This class serves as a facade for the refactored functionality, maintaining
    backward compatibility while using the new modular architecture.
    """
    
    def __init__(self, features=None, outlier_threshold=3.0, alpha=0.0, use_efficiency_frontier=True):
        """
        Initialize the full dataset regression analyzer.
        
        Args:
            features: List of feature names to use in regression
            outlier_threshold: Z-score threshold for outlier removal
            alpha: Regularization parameter (currently unused)
            use_efficiency_frontier: Use efficiency frontier regression (recommended)
        """
        # Initialize component modules
        self.regression_core = FullDatasetRegressionCore(features, outlier_threshold, alpha)
        self.multicollinearity_handler = MulticollinearityHandler(use_commonality_analysis=False)  # DISABLED
        self.model_validator = ModelValidator()
        
        # NEW: Efficiency frontier regression
        self.use_efficiency_frontier = use_efficiency_frontier
        if use_efficiency_frontier:
            from .efficiency_frontier import EfficiencyFrontierRegression
            self.efficiency_regressor = EfficiencyFrontierRegression(features=features, alpha=1.0)
        else:
            self.efficiency_regressor = None
        
        # Maintain backward compatibility with existing attributes
        self.features = self.regression_core.features
        self.outlier_threshold = outlier_threshold
        self.alpha = alpha
        self.coefficients = None
        self.unconstrained_coefficients = None
        self.coefficient_bounds = None
        self.all_plans = None
        self.outliers_removed = 0
        self.correlation_matrix = None
        self.multicollinearity_fixes = {}
        
    def detect_multicollinearity(self, X: np.ndarray, feature_names: List[str], threshold: float = 0.8) -> Dict:
        """
        Detect multicollinearity using correlation matrix.
        
        Args:
            X: Feature matrix (without intercept)
            feature_names: List of feature names
            threshold: Correlation threshold for multicollinearity detection
            
        Returns:
            Dictionary with multicollinearity analysis results
        """
        self.multicollinearity_handler.threshold = threshold
        result = self.multicollinearity_handler.detect_multicollinearity(X, feature_names)
        
        # Update instance attributes for backward compatibility
        self.correlation_matrix = self.multicollinearity_handler.correlation_matrix
        
        return result
    
    @memory_monitor
    @cache_dataframe_operation
    def remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove obvious pricing outliers that would skew regression.
        
        Args:
            df: DataFrame with plan data
            
        Returns:
            DataFrame with outliers removed
        """
        result = self.regression_core.remove_outliers(df)
        
        # Update instance attributes for backward compatibility
        self.outliers_removed = self.regression_core.outliers_removed
        
        return result
        
    @memory_monitor
    def solve_full_dataset_coefficients(self, df: pd.DataFrame) -> np.ndarray:
        """
        Solve for coefficients using ALL plans in the dataset with constrained optimization.
        
        Args:
            df: DataFrame with plan data
            
        Returns:
            Array of coefficients [β₀, β₁, β₂, ...]
        """
        # NEW: Use efficiency frontier if enabled
        if self.use_efficiency_frontier and self.efficiency_regressor is not None:
            logger.info("🎯 Using Efficiency Frontier Regression (Pareto-optimal plans only)")
            
            try:
                # Use fee or original_fee as target
                price_col = 'original_fee' if 'original_fee' in df.columns else 'fee'
                coefficients = self.efficiency_regressor.solve_efficiency_frontier_coefficients(df, price_col)
                
                # Update instance attributes for backward compatibility
                self.coefficients = coefficients
                self.all_plans = self.efficiency_regressor.efficient_plans
                self.outliers_removed = len(df) - len(self.efficiency_regressor.efficient_plans)
                
                logger.info(f"✅ Efficiency Frontier Regression completed successfully")
                logger.info(f"   Efficiency ratio: {self.efficiency_regressor.efficiency_ratio:.1%}")
                logger.info(f"   Used {len(self.efficiency_regressor.efficient_plans)} efficient plans")
                
                return coefficients
                
            except Exception as e:
                logger.error(f"❌ Efficiency Frontier Regression failed: {e}")
                logger.info("   Falling back to traditional full dataset regression")
                self.use_efficiency_frontier = False
        
        # FALLBACK: Traditional full dataset regression
        logger.info("📊 Using Traditional Full Dataset Regression")
        
        # Step 1: Core regression analysis
        coefficients = self.regression_core.solve_full_dataset_coefficients(df)
        
        # Step 2: Skip multicollinearity handling (commonality analysis disabled)
        logger.info("⚠️  Multicollinearity handling disabled (Commonality Analysis OFF)")
        
        # Update instance attributes for backward compatibility
        self.coefficients = coefficients
        self.unconstrained_coefficients = self.regression_core.unconstrained_coefficients
        self.coefficient_bounds = self.regression_core.coefficient_bounds
        self.all_plans = self.regression_core.all_plans
        self.outliers_removed = self.regression_core.outliers_removed
        # self.multicollinearity_fixes = {}  # No multicollinearity processing
        
        return coefficients
    
    def _solve_constrained_regression(self, X: np.ndarray, y: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Solve using constrained optimization.
        Delegated to regression core module.
        """
        return self.regression_core._solve_constrained_regression(X, y, features)
    
    def _fix_multicollinearity_coefficients(self, coefficients: np.ndarray, features: List[str], 
                                           X: np.ndarray = None, y: np.ndarray = None) -> np.ndarray:
        """
        Fix multicollinearity by redistributing coefficients.
        Delegated to multicollinearity handler module.
        """
        return self.multicollinearity_handler.fix_multicollinearity_coefficients(coefficients, features, X, y)
    
    def get_coefficient_breakdown(self) -> dict:
        """
        Get coefficient breakdown for visualization.
        
        Returns:
            Dictionary with coefficient information including both raw and constrained values
        """
        if self.coefficients is None:
            raise ValueError("Must solve coefficients first")
        
        # Use efficiency frontier breakdown if available
        if self.use_efficiency_frontier and self.efficiency_regressor is not None:
            try:
                return self.efficiency_regressor.get_coefficient_breakdown()
            except:
                # Fallback to traditional breakdown
                pass
        
        # Traditional breakdown
        breakdown = self.regression_core.get_coefficient_breakdown()
        
        # Add multicollinearity information (empty since disabled)
        breakdown['multicollinearity_fixes'] = {}
        breakdown['multicollinearity_disabled'] = True
            
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
