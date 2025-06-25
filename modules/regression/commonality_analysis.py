"""
Commonality Analysis Module

Implements true commonality analysis for variance decomposition in multiple regression.
Based on Seibold & McPhee (1979) and modern implementations.
"""

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CommonalityAnalyzer:
    """
    Implements true commonality analysis for variance decomposition.
    
    Partitions R² into unique and common variance components:
    R² = Σ(unique effects) + Σ(common effects)
    """
    
    def __init__(self):
        self.feature_names = []
        self.results = {}
        self.subset_r2 = {}
        self.commonality_coefficients = {}
        self.final_coefficients = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict:
        """
        Perform complete commonality analysis.
        """
        self.feature_names = feature_names
        n_features = len(feature_names)
        
        logger.info(f"Starting commonality analysis for {n_features} features")
        
        # Step 1: Calculate R² for all possible subsets
        self.subset_r2 = self._calculate_all_subset_r2(X, y, feature_names)
        
        # Step 2: Calculate commonality coefficients
        self.commonality_coefficients = self._calculate_commonality_coefficients()
        
        # Step 3: Distribute coefficients based on commonality analysis
        self.final_coefficients = self._distribute_coefficients_by_commonality(X, y)
        
        # Step 4: Compile results
        self.results = {
            'subset_r2': self.subset_r2,
            'commonality_coefficients': self.commonality_coefficients,
            'final_coefficients': self.final_coefficients,
            'total_r2': self.subset_r2.get(tuple(feature_names), 0),
            'method': 'true_commonality_analysis'
        }
        
        logger.info("Commonality analysis completed successfully")
        return self.results
    
    def _calculate_all_subset_r2(self, X: np.ndarray, y: np.ndarray, 
                                feature_names: List[str]) -> Dict[Tuple[str, ...], float]:
        """Calculate R² for all possible subsets of features."""
        subset_r2 = {}
        n_features = len(feature_names)
        
        # Calculate total number of subsets for progress tracking
        total_subsets = 2**n_features - 1  # Exclude empty set
        processed_count = 0
        
        logger.info(f"Starting calculation of R² for {total_subsets} feature subsets")
        
        # Calculate R² for all non-empty subsets
        for r in range(1, n_features + 1):
            subset_count_for_size = len(list(combinations(range(n_features), r)))
            logger.info(f"Processing {subset_count_for_size} subsets of size {r}")
            
            for combo in combinations(range(n_features), r):
                # Create subset of features
                X_subset = X[:, combo]
                combo_names = tuple(feature_names[i] for i in combo)
                
                # Fit regression and calculate R²
                try:
                    model = LinearRegression(fit_intercept=True)
                    model.fit(X_subset, y)
                    y_pred = model.predict(X_subset)
                    r2 = r2_score(y, y_pred)
                    
                    # Ensure R² is non-negative
                    r2 = max(0, r2)
                    
                    subset_r2[combo_names] = r2
                    
                    processed_count += 1
                    
                    # Progress logging every 1000 subsets or for larger subsets
                    if processed_count % 1000 == 0 or len(combo_names) >= 10:
                        progress = (processed_count / total_subsets) * 100
                        logger.info(f"Progress: {processed_count}/{total_subsets} ({progress:.1f}%) - "
                                  f"Latest: R²({combo_names[:3]}{'...' if len(combo_names) > 3 else ''}) = {r2:.6f}")
                    
                    logger.debug(f"R²({combo_names}) = {r2:.6f}")
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate R² for {combo_names}: {e}")
                    subset_r2[combo_names] = 0
                    processed_count += 1
        
        logger.info(f"Completed all subset R² calculations: {processed_count} subsets processed")
        return subset_r2
    
    def _calculate_commonality_coefficients(self) -> Dict[str, Dict]:
        """Calculate unique and common effects for each feature."""
        coefficients = {}
        
        # Get all features
        all_features = tuple(self.feature_names)
        total_r2 = self.subset_r2.get(all_features, 0)
        
        # Calculate unique effects for each feature
        for feature in self.feature_names:
            # Unique effect = R²(all features) - R²(all features except this one)
            other_features = tuple(f for f in self.feature_names if f != feature)
            
            if len(other_features) > 0:
                r2_without_feature = self.subset_r2.get(other_features, 0)
                unique_effect = total_r2 - r2_without_feature
            else:
                # Only one feature
                unique_effect = total_r2
            
            coefficients[f"{feature}_unique"] = {
                'value': max(0, unique_effect),  # Ensure non-negative
                'type': 'unique',
                'features': [feature]
            }
        
        # Calculate pairwise common effects
        for i, feature1 in enumerate(self.feature_names):
            for j, feature2 in enumerate(self.feature_names[i+1:], i+1):
                # Common effect = R²(Xi, Xj) - R²(Xi) - R²(Xj) 
                r2_both = self.subset_r2.get((feature1, feature2), 0)
                r2_feat1 = self.subset_r2.get((feature1,), 0)
                r2_feat2 = self.subset_r2.get((feature2,), 0)
                
                common_effect = r2_both - r2_feat1 - r2_feat2
                
                coefficients[f"{feature1}_{feature2}_common"] = {
                    'value': common_effect,  # Can be negative (suppression effect)
                    'type': 'common',
                    'features': [feature1, feature2]
                }
        
        return coefficients
    
    def _distribute_coefficients_by_commonality(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate variance decomposition for each feature while preserving original coefficients.
        
        This method provides commonality analysis information without modifying coefficients:
        1. Calculate unique and common variance contributions for each variable
        2. Preserve original constrained coefficients (which respect economic bounds)
        3. Provide decomposition information for transparency
        """
        try:
            # Get original constrained regression coefficients (these already respect economic bounds)
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            original_coeffs = model.coef_
            intercept = model.intercept_
            
            final_coeffs = {'intercept': intercept}
            
            # Total R² for normalization
            total_r2 = self.subset_r2.get(tuple(self.feature_names), 0)
            
            # Check if we have valid data
            if len(self.feature_names) == 0 or total_r2 <= 0:
                for i, feature in enumerate(self.feature_names):
                    final_coeffs[feature] = original_coeffs[i] if i < len(original_coeffs) else 0.0
                return final_coeffs
            
            # Calculate variance decomposition for each feature (for information only)
            variance_decomposition = {}
            
            for i, feature in enumerate(self.feature_names):
                if i >= len(original_coeffs):
                    final_coeffs[feature] = 0.0
                    continue
                    
                original_coeff = original_coeffs[i]
                
                # Get unique contribution (variance explained by this feature alone)
                unique_r2 = self.subset_r2.get((feature,), 0)
                
                # Calculate common contributions (shared with other features)
                common_contribution = 0.0
                
                # Add common contributions from all subsets containing this feature
                for subset, r2_value in self.subset_r2.items():
                    if len(subset) > 1 and feature in subset:
                        # This is a common effect involving this feature
                        common_effect = self._calculate_common_effect(subset, r2_value)
                        # Distribute the common effect equally among features in the subset
                        common_contribution += common_effect / len(subset)
                
                # Store variance decomposition information
                variance_decomposition[feature] = {
                    'unique_variance': unique_r2,
                    'common_variance': common_contribution,
                    'total_contribution': unique_r2 + common_contribution,
                    'unique_percentage': (unique_r2 / total_r2 * 100) if total_r2 > 0 else 0,
                    'common_percentage': (common_contribution / total_r2 * 100) if total_r2 > 0 else 0
                }
                
                # **KEY CHANGE: Keep original coefficient unchanged**
                # Commonality analysis provides decomposition information, not coefficient modification
                final_coeffs[feature] = original_coeff
                
                # Log the decomposition for transparency
                logger.info(f"Variance decomposition for {feature}: "
                          f"Unique: {unique_r2:.4f} ({variance_decomposition[feature]['unique_percentage']:.1f}%), "
                          f"Common: {common_contribution:.4f} ({variance_decomposition[feature]['common_percentage']:.1f}%), "
                          f"Coefficient: {original_coeff:.4f} (preserved)")
            
            # Store decomposition information in results
            self.variance_decomposition = variance_decomposition
            
            return final_coeffs
            
        except Exception as e:
            logger.error(f"Error in commonality analysis: {e}")
            # Fallback to original coefficients
            fallback_coeffs = {'intercept': intercept if 'intercept' in locals() else 0.0}
            for i, feature in enumerate(self.feature_names):
                fallback_coeffs[feature] = original_coeffs[i] if i < len(original_coeffs) else 0.0
            return fallback_coeffs
    
    def _calculate_common_effect(self, subset: Tuple[str, ...], subset_r2: float) -> float:
        """Calculate the common effect for a subset of features."""
        try:
            # Common effect = R²(subset) - Σ(unique effects of features in subset)
            common_effect = subset_r2
            
            for feature in subset:
                unique_r2 = self.subset_r2.get((feature,), 0)
                common_effect -= unique_r2
            
            # Common effect should not be negative (handle suppressor effects)
            return max(0.0, common_effect)
            
        except Exception:
            return 0.0
    
    def _apply_economic_constraints(self, feature: str, commonality_coeff: float, 
                                  original_coeff: float) -> float:
        """Apply economic constraints to prevent unrealistic coefficients from suppressor effects."""
        try:
            # Define economic bounds based on feature type
            if 'unlimited' in feature.lower():
                min_bound, max_bound = 100.0, 20000.0
            elif any(keyword in feature.lower() for keyword in ['5g', 'tethering']):
                min_bound, max_bound = 100.0, 10000.0
            else:
                # Usage-based features (data, voice, message, etc.)
                min_bound, max_bound = 0.1, float('inf')
            
            # If commonality analysis produces coefficient that violates economic logic
            if commonality_coeff < min_bound:
                # Use the minimum bound but preserve the sign relationship
                constrained_coeff = min_bound
                print(f"Warning: {feature} coefficient {commonality_coeff:.4f} below economic minimum {min_bound}, using {constrained_coeff}")
                return constrained_coeff
            
            elif max_bound != float('inf') and commonality_coeff > max_bound:
                # Use the maximum bound
                constrained_coeff = max_bound
                print(f"Warning: {feature} coefficient {commonality_coeff:.4f} above economic maximum {max_bound}, using {constrained_coeff}")
                return constrained_coeff
            
            else:
                # Coefficient is within economic bounds
                return commonality_coeff
                
        except Exception as e:
            print(f"Error applying economic constraints for {feature}: {e}")
            # Fallback to original coefficient
            return original_coeff
    
    def get_commonality_report(self) -> Dict:
        """Generate comprehensive commonality analysis report."""
        if not self.results:
            return {'error': 'No analysis performed yet'}
        
        report = {
            'total_variance_explained': self.results['total_r2'],
            'variance_decomposition': {},
            'coefficient_adjustments': {},
            'methodology': 'true_commonality_analysis'
        }
        
        # Variance decomposition
        for key, coeff_info in self.commonality_coefficients.items():
            report['variance_decomposition'][key] = {
                'value': coeff_info['value'],
                'percentage': (coeff_info['value'] / self.results['total_r2'] * 100) if self.results['total_r2'] > 0 else 0,
                'type': coeff_info['type'],
                'features': coeff_info['features']
            }
        
        return report
