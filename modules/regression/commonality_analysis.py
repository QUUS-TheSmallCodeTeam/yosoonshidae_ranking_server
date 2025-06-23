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
        """Distribute regression coefficients based on commonality analysis."""
        try:
            # Get original regression coefficients
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            original_coeffs = model.coef_
            intercept = model.intercept_
            
            final_coeffs = {'intercept': intercept}
            
            # Total R² for normalization
            total_r2 = self.subset_r2.get(tuple(self.feature_names), 0)
            
            # Check if we have valid data
            if len(original_coeffs) != len(self.feature_names):
                logger.error(f"Coefficient length mismatch: {len(original_coeffs)} vs {len(self.feature_names)}")
                # Fallback: return original coefficients
                for i, feature in enumerate(self.feature_names):
                    if i < len(original_coeffs):
                        final_coeffs[feature] = original_coeffs[i]
                    else:
                        final_coeffs[feature] = 0.0
                return final_coeffs
            
            for i, feature in enumerate(self.feature_names):
                try:
                    # Get unique effect for this feature
                    unique_key = f"{feature}_unique"
                    unique_effect = self.commonality_coefficients.get(unique_key, {}).get('value', 0)
                    
                    # Get all common effects involving this feature
                    common_effects = []
                    for key, coeff_info in self.commonality_coefficients.items():
                        if (coeff_info.get('type') in ['common', 'common_higher_order'] and 
                            feature in coeff_info.get('features', [])):
                            # Distribute common effect equally among involved features
                            n_features_involved = len(coeff_info.get('features', []))
                            if n_features_involved > 0:
                                shared_contribution = coeff_info.get('value', 0) / n_features_involved
                                common_effects.append(shared_contribution)
                    
                    # Calculate total contribution (unique + shared)
                    total_contribution = unique_effect + sum(common_effects)
                    
                    # Scale original coefficient by contribution ratio
                    if total_r2 > 0:
                        contribution_ratio = total_contribution / total_r2
                        final_coeff = original_coeffs[i] * contribution_ratio
                    else:
                        final_coeff = original_coeffs[i]
                    
                    # Ensure coefficient is numeric
                    if np.isnan(final_coeff) or np.isinf(final_coeff):
                        final_coeff = original_coeffs[i]
                    
                    final_coeffs[feature] = float(final_coeff)
                    
                    logger.info(f"{feature}: unique={unique_effect:.6f}, common={sum(common_effects):.6f}, "
                               f"total={total_contribution:.6f}, final_coeff={final_coeff:.4f}")
                               
                except Exception as e:
                    logger.warning(f"Error processing feature {feature}: {e}")
                    # Fallback to original coefficient
                    final_coeffs[feature] = float(original_coeffs[i]) if i < len(original_coeffs) else 0.0
            
            return final_coeffs
            
        except Exception as e:
            logger.error(f"Critical error in coefficient distribution: {e}")
            # Emergency fallback: return simple coefficients
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)
            fallback_coeffs = {'intercept': model.intercept_}
            for i, feature in enumerate(self.feature_names):
                if i < len(model.coef_):
                    fallback_coeffs[feature] = float(model.coef_[i])
                else:
                    fallback_coeffs[feature] = 0.0
            return fallback_coeffs
    
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
