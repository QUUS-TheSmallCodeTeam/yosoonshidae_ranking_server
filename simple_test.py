import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from modules.cost_spec import LinearDecomposition

# Create preprocessed test data with the required *_clean columns
test_data = {
    'id': [1, 2, 3, 4, 5],
    'plan_name': ['Test Plan 1', 'Test Plan 2', 'Test Plan 3', 'Test Plan 4', 'Test Plan 5'],
    'basic_data_clean': [1.0, 6.0, 15.0, 2.0, 6.0],
    'daily_data_clean': [0.0, 0.0, 0.0, 0.0, 0.0],
    'voice_clean': [100, 300, 300, 100, 350],
    'message_clean': [50, 300, 300, 100, 100],
    'tethering_gb': [0.0, 6.0, 0.0, 2.0, 0.0],
    'fee': [3960, 6000, 11000, 5500, 9900]
}

df = pd.DataFrame(test_data)

print("Test data:")
print(df[['plan_name', 'basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb', 'fee']])

# Test LinearDecomposition directly
print("\nTesting LinearDecomposition...")

# Initialize with features that exist in our data
features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
decomposer = LinearDecomposition(tolerance=500, features=features)

# Extract representative plans
print("\nExtracting representative plans...")
representative_plans = decomposer.extract_representative_plans(df, 'diverse_segments')
print(f"Representative plans shape: {representative_plans.shape}")
print(representative_plans[['plan_name', 'basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb', 'fee']])

# Solve for coefficients
print("\nSolving for coefficients...")
coefficients = decomposer.solve_coefficients(representative_plans, 'fee')
print(f"Coefficients: {coefficients}")

print(f"\nCost structure:")
print(f"Base cost: ₩{coefficients[0]:.2f}")
for i, feature in enumerate(features):
    print(f"{feature}: ₩{coefficients[i+1]:.2f}")

# Calculate baselines
print("\nCalculating baselines...")
baselines = decomposer.calculate_decomposed_baselines(df)
print(f"Baselines: {baselines}")

# Calculate CS ratios
cs_ratios = baselines / df['fee']
print(f"\nCS ratios: {cs_ratios.values}")

# Add to dataframe and set attrs
df['B'] = baselines
df['CS'] = cs_ratios

# Test setting cost_structure in attrs
cost_structure = {
    'base_cost': coefficients[0],
    'feature_costs': dict(zip(features, coefficients[1:]))
}

df.attrs['cost_structure'] = cost_structure
df.attrs['decomposition_coefficients'] = cost_structure

print(f"\nDataFrame attrs keys: {list(df.attrs.keys())}")
print(f"Cost structure in attrs: {df.attrs.get('cost_structure', 'NOT FOUND')}")

print(f"\nFinal dataframe:")
print(df[['plan_name', 'B', 'CS', 'fee']]) 