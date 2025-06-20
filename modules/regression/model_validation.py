"""
Model Validation Module

Contains comprehensive model validation functionality for regression analysis.
Extracted from full_dataset.py for better modularity.

Classes:
- ModelValidator: Handles optimization quality, economic logic, prediction power, and residual analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from scipy.optimize import minimize
import scipy.stats as stats
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

# Configure logging
logger = logging.getLogger(__name__)

class ModelValidator:
    """
    Comprehensive model validation for regression analysis.
    Handles optimization quality, economic logic, prediction power, and residual analysis.
    """
    
    def __init__(self):
        self.validation_report = {}
        
    def validate_optimization_quality(self, X, y, coefficients, bounds):
        """
        최적화가 local minima에 빠졌는지 검증
        """
        def objective(beta):
            return np.sum((X @ beta - y) ** 2)
        
        # 여러 다른 초기값으로 최적화 재실행
        initial_guesses = [
            np.ones(len(coefficients)-1) * 1.0,    # 작은 값 (intercept 제외)
            np.ones(len(coefficients)-1) * 100.0,  # 중간 값
            np.ones(len(coefficients)-1) * 1000.0, # 큰 값
            np.random.uniform(1, 1000, len(coefficients)-1),  # 랜덤
            np.random.uniform(10, 500, len(coefficients)-1),  # 다른 랜덤
        ]
        
        results = []
        convergence_info = []
        for i, guess in enumerate(initial_guesses):
            try:
                result = minimize(
                    objective, 
                    guess, 
                    bounds=bounds,
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-6}
                )
                if result.success:
                    results.append(result.fun)  # 목적함수 값
                    convergence_info.append({
                        'initial_guess': i,
                        'objective_value': result.fun,
                        'converged': True,
                        'iterations': result.nit
                    })
                else:
                    convergence_info.append({
                        'initial_guess': i,
                        'objective_value': float('inf'),
                        'converged': False,
                        'message': result.message
                    })
            except Exception as e:
                convergence_info.append({
                    'initial_guess': i,
                    'objective_value': float('inf'),
                    'converged': False,
                    'error': str(e)
                })
        
        # 모든 결과가 비슷하면 global optimum 가능성 높음
        if len(results) > 1:
            objective_values = np.array(results)
            consistency = np.std(objective_values) / np.mean(objective_values) if np.mean(objective_values) > 0 else float('inf')
        else:
            consistency = float('inf')
        
        return {
            'is_consistent': consistency < 0.01,  # 1% 이내 변동
            'objective_std': consistency,
            'all_results': results,
            'convergence_details': convergence_info,
            'successful_optimizations': len(results)
        }

    def validate_economic_logic(self, df, coefficients, features):
        """
        계수들이 경제적으로 말이 되는지 검증
        """
        validation_results = {}
        
        try:
            # 1. 스케일 검증: 데이터 1GB vs 5G 지원 비용 비교
            data_coeff = None
            fiveg_coeff = None
            
            if 'basic_data_clean' in features:
                data_idx = features.index('basic_data_clean')
                data_coeff = coefficients[data_idx + 1]  # Skip intercept
            
            if 'is_5g' in features:
                fiveg_idx = features.index('is_5g')
                fiveg_coeff = coefficients[fiveg_idx + 1]  # Skip intercept
            
            if data_coeff is not None and fiveg_coeff is not None:
                validation_results['scale_check'] = {
                    'data_per_gb': data_coeff,
                    'fiveg_premium': fiveg_coeff,
                    'ratio': fiveg_coeff / data_coeff if data_coeff != 0 else float('inf'),
                    'makes_sense': fiveg_coeff > data_coeff * 10,  # 5G가 10GB 데이터보다 비싸야 함
                    'economic_reasoning': '5G 지원 비용이 데이터 1GB 비용의 10배 이상이어야 경제적으로 타당함'
                }
            
            # 2. 순서 검증: 기본 기능 < 프리미엄 기능
            voice_coeff = None
            tethering_coeff = None
            
            if 'voice_clean' in features:
                voice_idx = features.index('voice_clean')
                voice_coeff = coefficients[voice_idx + 1]
            
            if 'tethering_gb' in features:
                tethering_idx = features.index('tethering_gb')
                tethering_coeff = coefficients[tethering_idx + 1]
            
            if voice_coeff is not None and tethering_coeff is not None:
                validation_results['premium_check'] = {
                    'voice_per_min': voice_coeff,
                    'tethering_per_gb': tethering_coeff,
                    'ratio': tethering_coeff / voice_coeff if voice_coeff != 0 else float('inf'),
                    'makes_sense': tethering_coeff > voice_coeff,  # 테더링이 음성보다 비싸야 함
                    'economic_reasoning': '테더링이 일반 음성통화보다 단위당 비용이 높아야 함'
                }
            
            # 3. 양수 검증: 모든 계수가 경제적으로 의미있는 값인지
            positive_check = {}
            negative_coefficients = []
            zero_coefficients = []
            
            for i, feature in enumerate(features):
                coeff = coefficients[i + 1]  # Skip intercept
                if coeff < 0:
                    negative_coefficients.append((feature, coeff))
                elif coeff == 0:
                    zero_coefficients.append((feature, coeff))
            
            positive_check = {
                'negative_count': len(negative_coefficients),
                'zero_count': len(zero_coefficients),
                'negative_features': negative_coefficients,
                'zero_features': zero_coefficients,
                'all_positive': len(negative_coefficients) == 0 and len(zero_coefficients) == 0,
                'economic_reasoning': '모든 기능은 비용을 증가시켜야 하므로 양수 계수를 가져야 함'
            }
            
            validation_results['positive_check'] = positive_check
            
        except Exception as e:
            validation_results['error'] = f"Economic logic validation failed: {str(e)}"
        
        return validation_results

    def validate_prediction_power(self, df, features, solve_regression_func):
        """
        모델이 실제로 시장 가격을 잘 예측하는지 검증 (Cross-Validation)
        """
        try:
            X = df[features].values
            y = df['fee'].values
            
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            r2_scores = []
            mae_scores = []
            fold_details = []
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # 학습 데이터로 계수 계산
                train_coeffs = solve_regression_func(X_train, y_train, features)
                
                # 테스트 데이터로 예측
                y_pred = X_test @ train_coeffs[1:]  # Skip intercept
                
                r2 = r2_score(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                
                r2_scores.append(r2)
                mae_scores.append(mae)
                
                fold_details.append({
                    'fold': fold + 1,
                    'r2_score': r2,
                    'mae': mae,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                })
            
            return {
                'mean_r2': np.mean(r2_scores),
                'std_r2': np.std(r2_scores),
                'mean_mae': np.mean(mae_scores),
                'std_mae': np.std(mae_scores),
                'is_stable': np.std(r2_scores) < 0.1,  # R² 표준편차가 0.1 미만
                'fold_details': fold_details,
                'all_r2_scores': r2_scores,
                'all_mae_scores': mae_scores
            }
            
        except Exception as e:
            return {
                'error': f"Prediction power validation failed: {str(e)}",
                'mean_r2': 0.0,
                'std_r2': 0.0,
                'mean_mae': float('inf'),
                'std_mae': 0.0,
                'is_stable': False
            }

    def analyze_residuals(self, df, predicted, actual):
        """
        잔차 패턴을 분석하여 모델의 적합성 검증
        """
        try:
            residuals = actual - predicted
            
            # 1. 잔차의 패턴 검사
            # 정규성 검정
            normality_test = stats.jarque_bera(residuals)
            
            # 이분산성 검정 (가격 구간별로 잔차 분산이 다른지)
            price_quartiles = np.quantile(actual, [0.25, 0.5, 0.75])
            residual_vars = []
            quartile_info = []
            
            for i in range(len(price_quartiles) + 1):
                if i == 0:
                    mask = actual <= price_quartiles[0]
                    quartile_name = f"Q1 (≤ ₩{price_quartiles[0]:,.0f})"
                elif i == len(price_quartiles):
                    mask = actual > price_quartiles[-1]
                    quartile_name = f"Q4 (> ₩{price_quartiles[-1]:,.0f})"
                else:
                    mask = (actual > price_quartiles[i-1]) & (actual <= price_quartiles[i])
                    quartile_name = f"Q{i+1} (₩{price_quartiles[i-1]:,.0f} - ₩{price_quartiles[i]:,.0f})"
                
                if np.sum(mask) > 0:
                    quartile_residuals = residuals[mask]
                    residual_var = np.var(quartile_residuals)
                    residual_vars.append(residual_var)
                    quartile_info.append({
                        'quartile': quartile_name,
                        'count': np.sum(mask),
                        'residual_variance': residual_var,
                        'residual_std': np.std(quartile_residuals),
                        'mean_residual': np.mean(quartile_residuals)
                    })
            
            # 잔차 분산의 일관성
            heteroscedasticity = np.std(residual_vars) / np.mean(residual_vars) if np.mean(residual_vars) > 0 else float('inf')
            
            # 이상치 잔차 분석
            residual_std = np.std(residuals)
            outlier_threshold = 3 * residual_std
            outlier_residuals = np.sum(np.abs(residuals) > outlier_threshold)
            
            return {
                'mean_residual': np.mean(residuals),
                'residual_std': residual_std,
                'residual_normality': {
                    'statistic': normality_test.statistic,
                    'p_value': normality_test.pvalue,
                    'is_normal': normality_test.pvalue > 0.05
                },
                'heteroscedasticity': {
                    'coefficient': heteroscedasticity,
                    'is_homoscedastic': heteroscedasticity < 0.5,
                    'quartile_analysis': quartile_info
                },
                'outlier_analysis': {
                    'count': outlier_residuals,
                    'percentage': (outlier_residuals / len(residuals)) * 100,
                    'threshold': outlier_threshold
                }
            }
            
        except Exception as e:
            return {
                'error': f"Residual analysis failed: {str(e)}",
                'mean_residual': 0.0,
                'residual_normality': {'is_normal': False},
                'heteroscedasticity': {'is_homoscedastic': False},
                'outlier_analysis': {'count': 0, 'percentage': 0.0}
            }

    def calculate_overall_validation_score(self, validation_report):
        """
        종합 검증 점수 계산 (0-100점)
        """
        score = 0
        detailed_scoring = {}
        
        # 최적화 일관성 (25점)
        optimization_score = 0
        if validation_report.get('optimization', {}).get('is_consistent', False):
            optimization_score = 25
        elif validation_report.get('optimization', {}).get('successful_optimizations', 0) >= 3:
            optimization_score = 15  # 부분 점수
        detailed_scoring['optimization'] = optimization_score
        score += optimization_score
        
        # 경제적 타당성 (25점)
        economic_score = 0
        econ = validation_report.get('economic_logic', {})
        
        # 스케일 검증
        scale_ok = econ.get('scale_check', {}).get('makes_sense', False)
        premium_ok = econ.get('premium_check', {}).get('makes_sense', False)
        positive_ok = econ.get('positive_check', {}).get('all_positive', False)
        
        if scale_ok and premium_ok and positive_ok:
            economic_score = 25
        elif (scale_ok and premium_ok) or (scale_ok and positive_ok) or (premium_ok and positive_ok):
            economic_score = 17
        elif scale_ok or premium_ok or positive_ok:
            economic_score = 8
        detailed_scoring['economic_logic'] = economic_score
        score += economic_score
        
        # 예측력 (30점)
        prediction_score = 0
        pred = validation_report.get('prediction_power', {})
        mean_r2 = pred.get('mean_r2', 0)
        is_stable = pred.get('is_stable', False)
        
        if mean_r2 > 0.8 and is_stable:
            prediction_score = 30
        elif mean_r2 > 0.6 and is_stable:
            prediction_score = 22
        elif mean_r2 > 0.6:
            prediction_score = 20
        elif mean_r2 > 0.4:
            prediction_score = 10
        elif mean_r2 > 0.2:
            prediction_score = 5
        detailed_scoring['prediction_power'] = prediction_score
        score += prediction_score
        
        # 잔차 품질 (20점)
        residual_score = 0
        resid = validation_report.get('residual_analysis', {})
        is_normal = resid.get('residual_normality', {}).get('is_normal', False)
        is_homoscedastic = resid.get('heteroscedasticity', {}).get('is_homoscedastic', False)
        outlier_pct = resid.get('outlier_analysis', {}).get('percentage', 100)
        
        if is_normal and is_homoscedastic and outlier_pct < 5:
            residual_score = 20
        elif (is_normal and is_homoscedastic) or (is_normal and outlier_pct < 5) or (is_homoscedastic and outlier_pct < 5):
            residual_score = 12
        elif is_normal or is_homoscedastic or outlier_pct < 10:
            residual_score = 6
        detailed_scoring['residual_quality'] = residual_score
        score += residual_score
        
        return {
            'total_score': score,
            'grade': 'A' if score >= 85 else 'B' if score >= 70 else 'C' if score >= 55 else 'D' if score >= 40 else 'F',
            'detailed_scoring': detailed_scoring,
            'score_breakdown': {
                'optimization_consistency': f"{optimization_score}/25",
                'economic_logic': f"{economic_score}/25", 
                'prediction_power': f"{prediction_score}/30",
                'residual_quality': f"{residual_score}/20"
            }
        }

    def comprehensive_model_validation(self, df, X, y, coefficients, features, bounds, solve_regression_func):
        """
        종합적인 모델 검증
        """
        validation_report = {}
        
        logger.info("Starting comprehensive model validation...")
        
        # 1. 최적화 품질
        try:
            validation_report['optimization'] = self.validate_optimization_quality(X, y, coefficients, bounds)
            logger.info("✓ Optimization quality validation completed")
        except Exception as e:
            logger.error(f"✗ Optimization validation failed: {e}")
            validation_report['optimization'] = {'error': str(e), 'is_consistent': False}
        
        # 2. 경제적 타당성
        try:
            validation_report['economic_logic'] = self.validate_economic_logic(df, coefficients, features)
            logger.info("✓ Economic logic validation completed")
        except Exception as e:
            logger.error(f"✗ Economic logic validation failed: {e}")
            validation_report['economic_logic'] = {'error': str(e)}
        
        # 3. 예측력
        try:
            validation_report['prediction_power'] = self.validate_prediction_power(df, features, solve_regression_func)
            logger.info("✓ Prediction power validation completed")
        except Exception as e:
            logger.error(f"✗ Prediction power validation failed: {e}")
            validation_report['prediction_power'] = {'error': str(e), 'mean_r2': 0.0, 'is_stable': False}
        
        # 4. 잔차 분석
        try:
            predicted = X @ coefficients[1:]  # Skip intercept
            validation_report['residual_analysis'] = self.analyze_residuals(df, predicted, y)
            logger.info("✓ Residual analysis completed")
        except Exception as e:
            logger.error(f"✗ Residual analysis failed: {e}")
            validation_report['residual_analysis'] = {'error': str(e)}
        
        # 5. 종합 점수
        try:
            validation_report['overall_score'] = self.calculate_overall_validation_score(validation_report)
            logger.info(f"✓ Overall validation score: {validation_report['overall_score']['total_score']}/100 ({validation_report['overall_score']['grade']})")
        except Exception as e:
            logger.error(f"✗ Overall scoring failed: {e}")
            validation_report['overall_score'] = {'error': str(e), 'total_score': 0, 'grade': 'F'}
        
        self.validation_report = validation_report
        return validation_report 