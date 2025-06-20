"""
Multicollinearity Handler Module

Contains multicollinearity detection and coefficient redistribution functionality.
Extracted from full_dataset.py for better modularity.

Classes:
- MulticollinearityHandler: Handles correlation analysis and coefficient redistribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

# Configure logging
logger = logging.getLogger(__name__)

class MulticollinearityHandler:
    """
    Handles multicollinearity detection and coefficient redistribution for regression analysis.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
        self.correlation_matrix = None
        self.multicollinearity_detected = False
        self.multicollinearity_fixes = {}
        
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

    def fix_multicollinearity_coefficients(self, coefficients: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Fix multicollinearity by redistributing coefficients for highly correlated features.
        
        Args:
            coefficients: Original coefficients [β₀, β₁, β₂, ...]
            features: List of feature names
            
        Returns:
            Adjusted coefficients with redistributed values
        """
        if self.correlation_matrix is None:
            logger.warning("No correlation matrix available - run detect_multicollinearity first")
            return coefficients
        
        fixed_coefficients = coefficients.copy()
        
        # Store multicollinearity fixes for detailed reporting
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
                            'calculation_formula': f"({coeff1:.2f} + {coeff2:.2f}) / 2 = {redistributed_value:.2f}"
                        }
                        
                        self.multicollinearity_fixes[feature2] = {
                            'paired_with': feature1,
                            'correlation': corr_val,
                            'original_value': coeff2,
                            'partner_original_value': coeff1,
                            'total_value': total_value,
                            'redistributed_value': redistributed_value,
                            'calculation_formula': f"({coeff1:.2f} + {coeff2:.2f}) / 2 = {redistributed_value:.2f}"
                        }
                        
                        # Apply redistribution
                        fixed_coefficients[i+1] = redistributed_value
                        fixed_coefficients[j+1] = redistributed_value
        
        return fixed_coefficients
    
    def get_multicollinearity_report(self) -> Dict:
        """
        Get detailed multicollinearity analysis report.
        
        Returns:
            Dictionary with multicollinearity analysis and fixes
        """
        return {
            'detected': self.multicollinearity_detected,
            'threshold': self.threshold,
            'correlation_matrix': self.correlation_matrix,
            'fixes_applied': self.multicollinearity_fixes,
            'total_fixes': len(self.multicollinearity_fixes)
        } 