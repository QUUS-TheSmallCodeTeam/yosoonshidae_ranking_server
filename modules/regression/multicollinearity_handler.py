"""
Multicollinearity Handler Module

Contains multicollinearity detection and coefficient redistribution functionality.
Now uses enhanced Commonality Analysis with intelligent redistribution.

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
    Handles multicollinearity detection and coefficient redistribution using enhanced Commonality Analysis.
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
        🔬 Enhanced Commonality Analysis with Intelligent Redistribution:
        - Perform true commonality analysis for variance decomposition
        - Apply intelligent coefficient redistribution based on commonality results
        - Maintain economic constraints to prevent unrealistic values
        
        Args:
            coefficients: Original constrained regression coefficients
            features: List of feature names
            X: Feature matrix (required for commonality analysis)
            y: Target variable (required for commonality analysis)
            
        Returns:
            Redistributed coefficients based on commonality analysis
        """
        logger.info("=== 🔬 Enhanced Commonality Analysis Multicollinearity Handler ===")
        
        if self.correlation_matrix is None:
            logger.warning("No correlation matrix available - run detect_multicollinearity first")
            return coefficients
        
        # Try Enhanced Commonality Analysis with redistribution
        if X is not None and y is not None and self.use_commonality_analysis:
            try:
                logger.info("🔬 Applying Enhanced Commonality Analysis with intelligent redistribution")
                return self._apply_enhanced_commonality_redistribution(coefficients, features, X, y)
            except Exception as e:
                logger.warning(f"Enhanced Commonality Analysis failed: {e}, falling back to simple redistribution")
                self.use_commonality_analysis = False
        
        # Fallback to simple redistribution method
        logger.info("📊 Applying simple redistribution method")
        return self._apply_simple_redistribution(coefficients, features)
    
    def _apply_enhanced_commonality_redistribution(self, coefficients: np.ndarray, features: List[str], 
                                                  X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Apply Enhanced Commonality Analysis for variance decomposition and coefficient redistribution.
        
        **UPDATED**: This method now applies the redistributed coefficients for actual CS calculation.
        
        Args:
            coefficients: Original coefficients (including base cost at index 0)
            features: List of feature names
            X: Feature matrix for commonality analysis
            y: Target variable
            
        Returns:
            Redistributed coefficients based on commonality analysis results
        """
        logger.info("Applying Enhanced Commonality Analysis with coefficient redistribution...")
        
        # **KEY CHANGE: Use redistributed coefficients for actual calculation**
        # Start with original coefficients but apply redistribution
        redistributed_coefficients = coefficients.copy()
        
        # Initialize commonality analyzer
        analyzer = CommonalityAnalyzer()
        
        try:
            # Perform commonality analysis to get variance decomposition
            results = analyzer.fit(X, y, features)
            
            # Get economic bounds for applying constraints
            bounds = self._get_economic_bounds(features)
            
            # Process each feature for variance decomposition and redistribution
            for i, feature in enumerate(features):
                original_coeff = coefficients[i+1]  # Skip base cost at index 0
                
                # Extract variance decomposition information
                unique_effect = 0.0
                total_common_effect = 0.0
                common_effects = []
                
                # Get unique effect for this feature
                unique_key = f"{feature}_unique"
                if unique_key in results['commonality_coefficients']:
                    unique_effect = results['commonality_coefficients'][unique_key]['value']
                
                # Get common effects involving this feature
                for key, coeff_info in results['commonality_coefficients'].items():
                    if coeff_info['type'] == 'common' and feature in coeff_info['features']:
                        # This is a common effect involving this feature
                        common_r2 = max(0, coeff_info['value'])  # Ensure non-negative
                        total_common_effect += common_r2
                        
                        # Find the paired feature
                        other_features = [f for f in coeff_info['features'] if f != feature]
                        if other_features:
                            correlation = self._get_correlation(feature, coeff_info['features'])
                            common_effects.append({
                                'paired_with': other_features[0],
                                'correlation': correlation,
                                'r2_contribution': common_r2
                            })
                
                # **CORE LOGIC: True Redistribution for actual calculation**
                # Redistribute coefficients between correlated features while preserving total value
                total_r2 = results.get('total_r2', 1.0)
                
                # **IMPORTANT: Apply redistributed values to actual coefficients**
                final_coeff = original_coeff  # Start with original
                
                # Find correlated features for redistribution
                if len(common_effects) > 0:
                    paired_feature = common_effects[0]['paired_with']
                    correlation = common_effects[0]['correlation']
                    
                    # Get partner's coefficient for redistribution calculation
                    partner_idx = None
                    for idx, feat in enumerate(features):
                        if feat == paired_feature:
                            partner_idx = idx
                            break
                    
                    if partner_idx is not None:
                        partner_coeff = coefficients[partner_idx + 1]  # +1 for intercept
                        
                        # **REDISTRIBUTION LOGIC**: 
                        # Total value = original_coeff + partner_coeff
                        # Redistribute based on unique variance contributions
                        total_value = original_coeff + partner_coeff
                        
                        if (unique_effect + total_common_effect) > 0:
                            # Calculate redistribution ratio based on unique contribution
                            unique_ratio = unique_effect / (unique_effect + total_common_effect)
                            
                            # Redistribute total value based on unique contributions
                            # Higher unique contribution = larger share of total value
                            redistributed_value = total_value * unique_ratio
                            
                            # Apply economic constraints to redistributed value
                            min_bound, max_bound = bounds.get(feature, (0.1, None))
                            
                            if redistributed_value < min_bound:
                                final_coeff = min_bound
                                redistribution_method = f"redistribution_capped_min (unique_ratio: {unique_ratio:.3f}, total: {total_value:.2f})"
                            elif max_bound and redistributed_value > max_bound:
                                final_coeff = max_bound
                                redistribution_method = f"redistribution_capped_max (unique_ratio: {unique_ratio:.3f}, total: {total_value:.2f})"
                            else:
                                final_coeff = redistributed_value
                                redistribution_method = f"variance_redistribution (unique_ratio: {unique_ratio:.3f}, total: {total_value:.2f})"
                        else:
                            # Fallback: equal redistribution
                            final_coeff = (original_coeff + partner_coeff) / 2
                            redistribution_method = "equal_redistribution (no variance data)"
                    else:
                        redistribution_method = "no_partner_found (coefficient preserved)"
                else:
                    # No correlation detected - keep original coefficient
                    redistribution_method = "no_correlation (coefficient preserved)"
                
                # **CRITICAL: Apply redistributed coefficient to actual array**
                redistributed_coefficients[i+1] = final_coeff
                
                # Store detailed variance decomposition information for display
                if len(common_effects) > 0:
                    paired_feature = common_effects[0]['paired_with']
                    correlation = common_effects[0]['correlation']
                    
                    # Calculate meaningful percentage breakdowns
                    total_variance_effect = unique_effect + total_common_effect
                    if total_variance_effect > 0:
                        unique_pct = (unique_effect / total_variance_effect) * 100
                        common_pct = (total_common_effect / total_variance_effect) * 100
                    else:
                        unique_pct = 100.0
                        common_pct = 0.0
                    
                    self.multicollinearity_fixes[feature] = {
                        'paired_with': paired_feature,
                        'correlation': correlation,
                        'original_value': original_coeff,
                        'redistributed_value': final_coeff,
                        'unique_effect': unique_effect,
                        'common_effect': total_common_effect,
                        'total_variance_effect': total_variance_effect,
                        'redistribution_method': redistribution_method,
                        'variance_breakdown': f"고유: {unique_effect:.4f} ({unique_pct:.1f}%) + 공통: {total_common_effect:.4f} ({common_pct:.1f}%)",
                        'method': 'enhanced_commonality_analysis',
                        'calculation_formula': f"unique({unique_effect:.4f}) + common({total_common_effect:.4f}) = {total_variance_effect:.4f} → {final_coeff:.4f}"
                    }
                    
                    logger.info(f"  {feature}: Coefficient redistributed from {original_coeff:.4f} to {final_coeff:.4f} using {redistribution_method}")
                    logger.info(f"    Variance decomposition: unique={unique_effect:.4f} ({unique_pct:.1f}%), common={total_common_effect:.4f} ({common_pct:.1f}%)")
                    logger.info(f"    Paired with: {paired_feature} (r={correlation:.3f})")
                else:
                    # No multicollinearity detected for this feature
                    self.multicollinearity_fixes[feature] = {
                        'paired_with': 'none',
                        'correlation': 0.0,
                        'original_value': original_coeff,
                        'redistributed_value': final_coeff,
                        'unique_effect': unique_effect,
                        'common_effect': 0.0,
                        'total_variance_effect': unique_effect,
                        'redistribution_method': redistribution_method,
                        'variance_breakdown': f"고유: {unique_effect:.4f} (100.0%) + 공통: 0.0000 (0.0%)",
                        'method': 'enhanced_commonality_analysis',
                        'calculation_formula': f"unique({unique_effect:.4f}) + common(0.0000) = {unique_effect:.4f} → {final_coeff:.4f}"
                    }
                    
                    logger.info(f"  {feature}: No multicollinearity detected, coefficient applied as {final_coeff:.4f} using {redistribution_method}")
        
        except Exception as e:
            logger.error(f"Enhanced Commonality Analysis failed: {e}")
            # Fallback: no multicollinearity fixes, preserve all coefficients
            for i, feature in enumerate(features):
                original_coeff = coefficients[i+1]
                self.multicollinearity_fixes[feature] = {
                    'paired_with': 'analysis_failed',
                    'correlation': 0.0,
                    'original_value': original_coeff,
                    'redistributed_value': original_coeff,
                    'unique_effect': 0.0,
                    'common_effect': 0.0,
                    'total_variance_effect': 0.0,
                    'redistribution_method': 'fallback_preserve_original',
                    'variance_breakdown': 'Analysis failed - coefficient preserved',
                    'method': 'enhanced_commonality_analysis',
                    'calculation_formula': f'Analysis failed → {original_coeff:.4f} (preserved)'
                }
        
        logger.info("Enhanced Commonality Analysis completed - redistributed coefficients applied to calculation")
        return redistributed_coefficients

    def _apply_simple_redistribution(self, coefficients: np.ndarray, features: List[str]) -> np.ndarray:
        """
        Apply simple redistribution method for high correlation pairs.
        
        Args:
            coefficients: Original coefficients
            features: List of feature names
            
        Returns:
            Redistributed coefficients using simple averaging
        """
        redistributed_coefficients = coefficients.copy()
        self.multicollinearity_fixes = {}
        
        # Find high correlation pairs
        for i in range(len(features)):
            for j in range(i+1, len(features)):
                feature1 = features[i]
                feature2 = features[j]
                
                if feature1 in self.correlation_matrix.index and feature2 in self.correlation_matrix.columns:
                    corr_val = abs(self.correlation_matrix.loc[feature1, feature2])
                    
                    if corr_val > self.threshold:  # High correlation
                        # Get current coefficients (skip base cost at index 0)
                        coeff1 = redistributed_coefficients[i+1]
                        coeff2 = redistributed_coefficients[j+1]
                        
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
                        redistributed_coefficients[i+1] = redistributed_value
                        redistributed_coefficients[j+1] = redistributed_value
        
        return redistributed_coefficients

    def _get_economic_bounds(self, features: List[str]) -> Dict[str, tuple]:
        """
        Get economic bounds for each feature type.
        
        Args:
            features: List of feature names
            
        Returns:
            Dictionary mapping feature names to (min_bound, max_bound) tuples
        """
        bounds = {}
        
        for feature in features:
            if 'unlimited' in feature or feature == 'is_5g':
                # Unlimited services and 5G: higher minimum, reasonable maximum
                bounds[feature] = (100.0, 20000.0)
            else:
                # Usage-based features: small minimum, no maximum
                bounds[feature] = (0.1, None)
        
        return bounds
    
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
            Dictionary with multicollinearity analysis and redistribution results
        """
        return {
            'detected': self.multicollinearity_detected,
            'threshold': self.threshold,
            'method': 'enhanced_commonality_analysis' if self.use_commonality_analysis else 'simple_averaging',
            'correlation_matrix': self.correlation_matrix,
            'fixes_applied': self.multicollinearity_fixes,
            'total_fixes': len(self.multicollinearity_fixes),
            'note': 'Coefficients redistributed based on commonality analysis with economic constraints'
        } 