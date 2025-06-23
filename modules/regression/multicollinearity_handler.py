"""
Multicollinearity Handler Module

Contains multicollinearity detection and coefficient redistribution functionality.
Now uses true Commonality Analysis instead of simple averaging.

Classes:
- MulticollinearityHandler: Handles correlation analysis and coefficient redistribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from .commonality_analysis import CommonalityAnalyzer

# Configure logging
logger = logging.getLogger(__name__)

class MulticollinearityHandler:
    """
    Handles multicollinearity detection and coefficient redistribution using true Commonality Analysis.
    """
    
    def __init__(self, threshold: float = 0.8, use_commonality_analysis: bool = True):
        self.threshold = threshold
        self.use_commonality_analysis = use_commonality_analysis
        self.correlation_matrix = None
        self.multicollinearity_detected = False
        self.multicollinearity_fixes = {}
        self.commonality_analyzer = CommonalityAnalyzer() if use_commonality_analysis else None
        
    def detect_multicollinearity(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Detect multicollinearity using correlation matrix and variance inflation factor.
        
        Args:
            X: Feature matrix (without intercept)
            feature_names: List of feature names
            
        Returns:
            Dictionary with multicollinearity analysis results
        """
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(X, rowvar=False)
        self.correlation_matrix = pd.DataFrame(corr_matrix, 
                                             index=feature_names, 
                                             columns=feature_names)
        
        # Find high correlations
        high_correlations = []
        for i in range(len(feature_names)):
            for j in range(i+1, len(feature_names)):
                corr_val = corr_matrix[i, j]
                if abs(corr_val) > self.threshold:
                    high_correlations.append({
                        'feature1': feature_names[i],
                        'feature2': feature_names[j],
                        'correlation': corr_val
                    })
        
        self.multicollinearity_detected = len(high_correlations) > 0
        
        analysis = {
            'high_correlations': high_correlations,
            'multicollinearity_detected': self.multicollinearity_detected,
            'correlation_matrix': self.correlation_matrix
        }
        
        if self.multicollinearity_detected:
            logger.warning(f"Multicollinearity detected: {len(high_correlations)} high correlations found")
            for hc in high_correlations:
                logger.warning(f"  {hc['feature1']} ↔ {hc['feature2']}: {hc['correlation']:.3f}")
        
        return analysis

    def fix_multicollinearity_coefficients(self, coefficients: np.ndarray, features: List[str], 
                                         X: Optional[np.ndarray] = None, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fix multicollinearity using true Commonality Analysis or simple averaging fallback.
        
        Args:
            coefficients: Original coefficients [β₀, β₁, β₂, ...]
            features: List of feature names
            X: Feature matrix (required for Commonality Analysis)
            y: Target variable (required for Commonality Analysis)
            
        Returns:
            Adjusted coefficients with redistributed values
        """
        if self.correlation_matrix is None:
            logger.warning("No correlation matrix available - run detect_multicollinearity first")
            return coefficients
        
        fixed_coefficients = coefficients.copy()
        
        # Try Commonality Analysis if enabled and data is available
        if self.use_commonality_analysis and X is not None and y is not None:
            try:
                logger.info("🔬 Applying TRUE Commonality Analysis for coefficient redistribution")
                return self._fix_with_commonality_analysis(coefficients, features, X, y)
            except Exception as e:
                logger.warning(f"Commonality Analysis failed: {e}, falling back to simple averaging")
                self.use_commonality_analysis = False
        
        # Fallback to simple averaging method
        logger.info("📊 Applying simple averaging method for coefficient redistribution")
        return self._fix_with_simple_averaging(coefficients, features)
    
    def _fix_with_commonality_analysis(self, coefficients: np.ndarray, features: List[str], 
                                     X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply true Commonality Analysis to redistribute coefficients.
        """
        # Perform Commonality Analysis
        results = self.commonality_analyzer.fit(X, y, features)
        
        # Get the adjusted coefficients from commonality analysis
        adjusted_coeffs = results['final_coefficients']
        
        # Create new coefficient array
        fixed_coefficients = coefficients.copy()
        
        # Update coefficients based on commonality analysis
        for i, feature in enumerate(features):
            if feature in adjusted_coeffs:
                old_coeff = fixed_coefficients[i+1]  # Skip intercept
                new_coeff = adjusted_coeffs[feature]
                
                # Store detailed information for reporting
                unique_effect = results['commonality_coefficients'].get(f"{feature}_unique", {}).get('value', 0)
                
                # Find common effects involving this feature
                common_effects = []
                total_common = 0
                for key, coeff_info in results['commonality_coefficients'].items():
                    if coeff_info['type'] == 'common' and feature in coeff_info['features']:
                        common_val = coeff_info['value'] / len(coeff_info['features'])
                        common_effects.append({
                            'paired_with': [f for f in coeff_info['features'] if f != feature][0],
                            'value': common_val,
                            'correlation': self._get_correlation(feature, coeff_info['features'])
                        })
                        total_common += common_val
                
                # Update coefficient
                fixed_coefficients[i+1] = new_coeff
                
                # Store fix information for HTML reporting
                if len(common_effects) > 0:
                    paired_feature = common_effects[0]['paired_with']
                    correlation = common_effects[0]['correlation']
                    
                    self.multicollinearity_fixes[feature] = {
                        'paired_with': paired_feature,
                        'correlation': correlation,
                        'original_value': old_coeff,
                        'redistributed_value': new_coeff,
                        'unique_effect': unique_effect,
                        'common_effect': total_common,
                        'total_contribution': unique_effect + total_common,
                        'method': 'true_commonality_analysis',
                        'calculation_formula': f"unique({unique_effect:.4f}) + common({total_common:.4f}) = {new_coeff:.4f}"
                    }
                    
                    logger.info(f"  {feature}: {old_coeff:.4f} → {new_coeff:.4f} "
                               f"(unique={unique_effect:.4f}, common={total_common:.4f})")
        
        return fixed_coefficients
    
    def _fix_with_simple_averaging(self, coefficients: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Apply simple averaging method (original fallback approach).
        """
        fixed_coefficients = coefficients.copy()
        self.multicollinearity_fixes = {}
        
        # Find high correlation pairs (threshold)
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                feature1 = features[i]
                feature2 = features[j]
                
                if feature1 in self.correlation_matrix.index and feature2 in self.correlation_matrix.columns:
                    corr_val = abs(self.correlation_matrix.loc[feature1, feature2])
                    
                    if corr_val > self.threshold:  # High correlation
                        # Get current coefficients (skip base cost at index 0)
                        coeff1 = fixed_coefficients[i+1]
                        coeff2 = fixed_coefficients[j+1]
                        
                        # Calculate total value and redistribute equally
                        total_value = coeff1 + coeff2
                        redistributed_value = total_value / 2
                        
                        logger.info(f"Redistributing coefficients for {feature1} ↔ {feature2} (correlation: {corr_val:.3f})")
                        logger.info(f"  Before: {feature1}=₩{coeff1:,.2f}, {feature2}=₩{coeff2:,.2f}")
                        logger.info(f"  After: {feature1}=₩{redistributed_value:,.2f}, {feature2}=₩{redistributed_value:,.2f}")
                        
                        # Store detailed calculation steps for HTML display
                        self.multicollinearity_fixes[feature1] = {
                            'paired_with': feature2,
                            'correlation': corr_val,
                            'original_value': coeff1,
                            'partner_original_value': coeff2,
                            'total_value': total_value,
                            'redistributed_value': redistributed_value,
                            'method': 'simple_averaging',
                            'calculation_formula': f"({coeff1:.2f} + {coeff2:.2f}) / 2 = {redistributed_value:.2f}"
                        }
                        
                        self.multicollinearity_fixes[feature2] = {
                            'paired_with': feature1,
                            'correlation': corr_val,
                            'original_value': coeff2,
                            'partner_original_value': coeff1,
                            'total_value': total_value,
                            'redistributed_value': redistributed_value,
                            'method': 'simple_averaging',
                            'calculation_formula': f"({coeff1:.2f} + {coeff2:.2f}) / 2 = {redistributed_value:.2f}"
                        }
                        
                        # Apply redistribution
                        fixed_coefficients[i+1] = redistributed_value
                        fixed_coefficients[j+1] = redistributed_value
        
        return fixed_coefficients
    
    def _get_correlation(self, feature: str, feature_list: List[str]) -> float:
        """Get correlation between feature and other features in the list."""
        for other_feature in feature_list:
            if other_feature != feature:
                if (feature in self.correlation_matrix.index and 
                    other_feature in self.correlation_matrix.columns):
                    return abs(self.correlation_matrix.loc[feature, other_feature])
        return 0.0
    
    def get_multicollinearity_report(self) -> Dict:
        """
        Get detailed multicollinearity analysis report.
        
        Returns:
            Dictionary with multicollinearity analysis and fixes
        """
        return {
            'detected': self.multicollinearity_detected,
            'threshold': self.threshold,
            'method': 'true_commonality_analysis' if self.use_commonality_analysis else 'simple_averaging',
            'correlation_matrix': self.correlation_matrix,
            'fixes_applied': self.multicollinearity_fixes,
            'total_fixes': len(self.multicollinearity_fixes)
        } 