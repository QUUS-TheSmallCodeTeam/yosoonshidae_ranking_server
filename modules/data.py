#!/usr/bin/env python
"""
Mobile Plan Data Loading and Processing Module

Functions for loading, preprocessing and preparing data for mobile plan ranking models.
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

def load_data_from_json(json_path):
    """Load plan data from a JSON file"""
    print(f"Loading data from JSON file: {json_path}")
    
    try:
        # Read JSON file
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Check if this is a logical test set (has base case and variants)
        is_logical_test = any(item.get('id') == 'base' for item in data)
        
        if is_logical_test:
            # Find the base case
            base_case = next(item for item in data if item.get('id') == 'base')
            base_features = base_case.get('features', {})
            
            # Process each variant
            processed_data = []
            
            # Add base case first
            base_row = {'id': 'base', 'name': base_case.get('name', '')}
            base_row.update(base_features)
            processed_data.append(base_row)
            
            # Process variants
            for item in data:
                if item.get('id') != 'base':
                    # Start with a copy of base features
                    variant = {'id': item.get('id'), 'name': item.get('name', '')}
                    variant.update(base_features)  # Fill with base features
                    
                    # Check if the variant has 'features' or 'changed_features'
                    if 'features' in item:
                        variant.update(item.get('features', {}))  # Override with variant features
                    elif 'changed_features' in item:
                        variant.update(item.get('changed_features', {}))  # Override with changed features
                    
                    processed_data.append(variant)
            
            # Convert to DataFrame
            df = pd.DataFrame(processed_data)
        else:
            # Regular data format
            df = pd.DataFrame(data)
        
        # Convert boolean columns to int
        bool_columns = ['is_5g', 'data_sharing', 'roaming_support', 
                       'micro_payment', 'signup_minor', 
                       'signup_foreigner', 'has_usim', 'has_nfc_usim',
                       'usim_is_free', 'nfc_usim_is_free', 'esim_is_free',
                       'usim_supported', 'nfc_usim_supported']
        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].fillna(False).astype(int)
        
        # Convert numeric strings to float - ensure these are always numeric
        numeric_columns = [
            'basic_data_clean', 'daily_data_clean', 'tethering_gb',
            'voice_clean', 'message_clean', 'additional_call',
            'agreement_period', 'fee', 'throttle_speed_normalized',
            'unlimited_type_numeric', 'usim_delivery_fee',
            'nfc_usim_delivery_fee', 'esim_fee'
        ]
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill NaN values
        df = df.fillna({
            'basic_data_clean': 0,
            'daily_data_clean': 0,
            'voice_clean': 0,
            'message_clean': 0,
            'additional_call': 0,
            'agreement_period': 0,
            'throttle_speed_normalized': 0,
            'unlimited_type_numeric': 0,
            'tethering_gb': 0,
            'usim_delivery_fee': 0,
            'nfc_usim_delivery_fee': 0,
            'esim_fee': 0,
            'usim_is_free': 0,
            'nfc_usim_is_free': 0,
            'esim_is_free': 0,
            'usim_supported': 0,
            'nfc_usim_supported': 0,
            'is_esim_numeric': 0
        })
        
        print(f"Loaded {len(df)} plans from JSON file")
        return df
        
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        traceback.print_exc()
        return None

def load_data_from_csv(filepath):
    """Load plan data from a CSV file"""
    print(f"Loading data from CSV file: {filepath}")
    
    try:
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Basic validation - check if this appears to be preprocessed data
        expected_cols = ['basic_data_clean', 'daily_data_clean', 'voice_clean', 'message_clean']
        preprocessed = any(col in df.columns for col in expected_cols)
        
        if not preprocessed:
            print("Warning: The CSV file doesn't appear to contain preprocessed data.")
            print("This may cause issues during model training/prediction.")
            print("Consider running preprocess_data.py on your raw data first.")
        
        # Ensure numeric columns are properly typed
        numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle any remaining NaN values
        df = df.fillna(0)
            
        print(f"Loaded {len(df)} plans from CSV file")
        print(f"Columns: {', '.join(df.columns)}")
        return df
        
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return None

def load_data(filepath=None):
    """Load data from a file.
    
    Args:
        filepath (str, optional): Path to the file to load. Defaults to None.
        
    Returns:
        pd.DataFrame: The loaded data.
    """
    if filepath is None:
        # Use a default path if none is specified
        filepath = 'data/processed/latest_processed_data.csv'
    
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"Error: File {filepath} does not exist.")
        return None
    
    # Check the file extension
    if filepath.endswith('.json'):
        return load_data_from_json(filepath)
    elif filepath.endswith('.csv'):
        return load_data_from_csv(filepath)
    else:
        print(f"Error: Unsupported file format for {filepath}. Use CSV or JSON.")
        return None

def load_default_data():
    """Load the default dataset from the standard location.
    
    Returns:
        pd.DataFrame: The default processed dataset.
    """
    default_path = 'data/processed/latest_processed_data.csv'
    if not os.path.exists(default_path):
        print(f"Error: Default data file {default_path} does not exist.")
        return None
    
    print(f"Loading default dataset from {default_path}")
    return load_data_from_csv(default_path)

def preprocess_data(df, use_gradient_penalty=False):
    """Preprocess the data for prediction.
    
    This function handles data type conversions and NaN value handling to ensure the data
    is properly formatted for model training and prediction.
    
    Args:
        df (pd.DataFrame): Input DataFrame with plan features.
        use_gradient_penalty (bool, optional): Whether to apply gradient penalties. Defaults to False.
        
    Returns:
        pd.DataFrame: Processed DataFrame ready for prediction.
    """
    print("\n--- PREPROCESSING DATA ---")
    
    # Make a copy of the input dataframe
    processed_df = df.copy()
    
    # Check for NaN values
    nan_columns = processed_df.columns[processed_df.isna().any()].tolist()
    if nan_columns:
        print(f"Warning: NaN values found in columns: {nan_columns}")
        # Fill NaN values
        processed_df = processed_df.fillna(0)
        print("Filled all NaN values with 0")
    else:
        print("No NaN values in the processed data")
    
    return processed_df 