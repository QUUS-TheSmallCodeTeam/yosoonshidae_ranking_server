"""
Chart Data Module

This module handles chart data preparation for visualization.
"""

import pandas as pd

def prepare_cost_structure_chart_data(cost_structure):
    """
    Prepare chart data for cost structure visualization.
    
    Args:
        cost_structure: Dictionary with cost structure from linear decomposition
        
    Returns:
        Dictionary with chart data for JavaScript rendering
    """
    if not cost_structure:
        return None
    
    # Chart 1: Overall cost breakdown (pie chart)
    overall_data = {
        'labels': [],
        'data': [],
        'colors': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    }
    
    # Chart 2: Per-unit costs (bar chart)
    unit_cost_data = {
        'labels': [],
        'data': [],
        'units': [],
        'colors': ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    }
    
    # Base cost (infrastructure)
    base_cost = cost_structure.get('base_cost', 0)
    if base_cost > 0:
        overall_data['labels'].append('기본 인프라 (Base Infrastructure)')
        overall_data['data'].append(base_cost)
    
    # Feature costs with Korean labels and units
    feature_labels = {
        'basic_data_clean': {'label': '데이터 비용 (Data Cost)', 'unit': 'per GB'},
        'voice_clean': {'label': '음성 비용 (Voice Cost)', 'unit': 'per 100min'},
        'message_clean': {'label': 'SMS 비용 (SMS Cost)', 'unit': 'per 100msg'},
        'tethering_gb': {'label': '테더링 프리미엄 (Tethering Premium)', 'unit': 'per GB'},
        'is_5g': {'label': '5G 기술료 (5G Technology)', 'unit': 'per plan'}
    }
    
    # Handle both flat and nested cost structure formats
    feature_costs = cost_structure.get('feature_costs', {})
    if not feature_costs:
        # Fallback to flat structure (direct keys excluding base_cost)
        feature_costs = {k: v for k, v in cost_structure.items() if k != 'base_cost'}
    
    for feature, cost in feature_costs.items():
        if feature in feature_labels:
            info = feature_labels[feature]
            
            # Extract numeric value from cost (handle both dict and numeric formats)
            if isinstance(cost, dict):
                # Multi-frontier method returns nested structure
                cost_value = cost.get('coefficient', cost.get('cost_per_unit', 0))
            else:
                # Simple numeric value
                cost_value = cost
            
            # For overall breakdown - use normalized values for comparison
            overall_data['labels'].append(info['label'])
            overall_data['data'].append(abs(cost_value))  # Use absolute value for visualization
            
            # For unit costs - only meaningful marginal costs
            if cost_value > 0:  # Only positive marginal costs
                unit_cost_data['labels'].append(info['label'])
                unit_cost_data['data'].append(cost_value)
                unit_cost_data['units'].append(info['unit'])
    
    # Chart 3: Detailed marginal cost analysis with business interpretation
    marginal_analysis = {
        'features': [],
        'coefficients': [],
        'interpretations': [],
        'colors': ['#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#34495e'],
        'base_cost': base_cost
    }
    
    # Feature analysis with business context
    feature_analysis = {
        'basic_data_clean': {
            'name': '데이터 (Data)', 
            'unit': '₩/GB',
            'interpretation': lambda x: f"데이터 1GB 추가시 ₩{x:.0f} 비용 증가"
        },
        'voice_clean': {
            'name': '음성통화 (Voice)', 
            'unit': '₩/100분',
            'interpretation': lambda x: f"음성 100분 추가시 ₩{x:.0f} 비용 증가"
        },
        'message_clean': {
            'name': 'SMS 문자', 
            'unit': '₩/100건',
            'interpretation': lambda x: f"SMS 100건 추가시 ₩{x:.0f} 비용 증가"
        },
        'tethering_gb': {
            'name': '테더링 (Tethering)', 
            'unit': '₩/GB',
            'interpretation': lambda x: f"테더링 1GB 추가시 ₩{x:.0f} 비용 증가"
        },
        'is_5g': {
            'name': '5G 기술료', 
            'unit': '₩/요금제',
            'interpretation': lambda x: f"5G 지원시 ₩{x:.0f} 추가 비용"
        }
    }
    
    # Use the same feature_costs handling for marginal analysis
    for feature, cost in feature_costs.items():
        if feature in feature_analysis:
            analysis = feature_analysis[feature]
            
            # Extract numeric value from cost (handle both dict and numeric formats)
            if isinstance(cost, dict):
                # Multi-frontier method returns nested structure
                cost_value = cost.get('coefficient', cost.get('cost_per_unit', 0))
            else:
                # Simple numeric value
                cost_value = cost
            
            marginal_analysis['features'].append(analysis['name'])
            marginal_analysis['coefficients'].append(cost_value)
            marginal_analysis['interpretations'].append(analysis['interpretation'](cost_value))
    
    return {
        'overall': overall_data,
        'unit_costs': unit_cost_data,
        'marginal_analysis': marginal_analysis,
        'feature_costs': feature_costs,  # Add raw feature costs for marginal cost charts
        'base_cost': base_cost           # Add base cost for marginal cost charts
    }

def prepare_plan_efficiency_data(df, method):
    """
    Prepare data for Plan Value Efficiency Matrix visualization.
    
    Args:
        df: DataFrame with plan data including CS ratios and baselines
        method: Method used ('linear_decomposition', 'frontier', 'fixed_rates', or 'multi_frontier')
        
    Returns:
        Dictionary with chart data for JavaScript rendering
    """
    if df.empty:
        return None
    
    efficiency_data = {
        'plans': [],
        'diagonal': {'min': 0, 'max': 0}
    }
    
    # Get baseline and actual cost columns based on method
    if method == 'linear_decomposition' and 'B_decomposed' in df.columns:
        baseline_col = 'B_decomposed'
        cs_col = 'CS_decomposed'
    else:
        # For frontier, fixed_rates, multi_frontier methods, use standard columns
        baseline_col = 'B'
        cs_col = 'CS'
    
    actual_col = 'fee'
    
    if baseline_col not in df.columns or actual_col not in df.columns or cs_col not in df.columns:
        return None
    
    # Prepare plan data
    for _, row in df.iterrows():
        baseline = row[baseline_col] if pd.notna(row[baseline_col]) else 0
        actual = row[actual_col] if pd.notna(row[actual_col]) else 0
        cs_ratio = row[cs_col] if pd.notna(row[cs_col]) else 0
        
        # Calculate total features for bubble size
        feature_total = 0
        feature_cols = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
        for col in feature_cols:
            if col in row and pd.notna(row[col]):
                feature_total += row[col]
        
        plan_data = {
            'baseline': baseline,
            'actual': actual,
            'cs_ratio': cs_ratio,
            'feature_total': feature_total,
            'plan_name': row.get('plan_name', 'Unknown Plan'),
            'mvno': row.get('mvno', 'Unknown MVNO'),
            'is_good_value': cs_ratio >= 1.0
        }
        
        efficiency_data['plans'].append(plan_data)
    
    # Calculate diagonal range
    if efficiency_data['plans']:
        all_baselines = [p['baseline'] for p in efficiency_data['plans']]
        all_actuals = [p['actual'] for p in efficiency_data['plans']]
        
        min_val = min(min(all_baselines), min(all_actuals))
        max_val = max(max(all_baselines), max(all_actuals))
        
        efficiency_data['diagonal']['min'] = min_val
        efficiency_data['diagonal']['max'] = max_val
    
    return efficiency_data 