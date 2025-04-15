import pandas as pd
import numpy as np

def prepare_features(df):
    """Prepare features for model development"""
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    
    # Print the incoming data types for debugging
    print("Data types before conversion:")
    print(processed_df.dtypes.head(10))
    
    # IMPORTANT: Ensure numeric fields are actually numeric
    # Convert basic_data and daily_data to numeric types first - ALWAYS DO THIS BEFORE ANY OPERATIONS
    for col in ['basic_data', 'daily_data', 'tethering_gb', 'mvno_rating', 'monthly_review_score', 'discount_percentage']:
        if col in processed_df.columns:
            # Check if the column is not already numeric
            if not pd.api.types.is_numeric_dtype(processed_df[col]):
                # Convert strings to numeric, coerce errors to NaN
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                # Fill NaN values with 0
                processed_df[col] = processed_df[col].fillna(0)
                print(f"Converted {col} to numeric type")
    
    # Print data types after conversion for verification
    print("Data types after conversion:")
    print(processed_df.dtypes.head(10))
    
    # === BEGIN ENHANCED LOGGING ===
    print("\n=== HF SERVER PREPROCESSING DETAILS ===")
    print(f"Input DataFrame shape: {df.shape}")
    print(f"Initial columns: {list(df.columns)}")
    # === END ENHANCED LOGGING ===
    
    # 1. Network type encoding
    processed_df['is_5g'] = (processed_df['network'] == '5G').astype(int)
    print(f"Created is_5g feature: {processed_df['is_5g'].sum()} 5G plans found")
    
    # 2. Process data_exhaustion field - do this first to identify throttled unlimited plans
    # Extract speed values from data_exhaustion field (e.g., "1Mbps" -> 1)
    def extract_speed(value):
        if pd.isna(value) or value is None:
            return 0
        if isinstance(value, str) and 'Mbps' in value:
            try:
                return float(value.replace('Mbps', '').strip())
            except:
                return 0
        return 0
    
    processed_df['speed_when_exhausted'] = processed_df['data_exhaustion'].apply(extract_speed)
    
    # Flag plans with throttled unlimited data (finite quota but continues at reduced speed)
    processed_df['has_throttled_data'] = (processed_df['speed_when_exhausted'] > 0).astype(int)
    print(f"Found {processed_df['has_throttled_data'].sum()} plans with throttled data")
    
    # 3. Data allowance processing
    # Handle special values in data fields
    # Special values: 999, 9999 = unlimited, -1 = not applicable
    
    # Manual check before isin() operation to ensure they're compared as numbers
    processed_df['basic_data_unlimited'] = processed_df['basic_data'].isin([999, 9999]).astype(int)
    # Extra safety for daily_data - ensure it's not None before comparison
    processed_df['daily_data_unlimited'] = processed_df['daily_data'].fillna(0).isin([999, 9999]).astype(int)
    
    print(f"Found {processed_df['basic_data_unlimited'].sum()} plans with unlimited basic data")
    print(f"Found {processed_df['daily_data_unlimited'].sum()} plans with unlimited daily data")
    
    # Find maximum finite values for basic_data and daily_data
    # These will be used as replacements for unlimited values
    # Use extra safety to avoid errors with mixed types
    try:
        max_basic_data = processed_df.loc[~processed_df['basic_data'].isin([999, 9999]), 'basic_data'].max()
    except Exception as e:
        print(f"Error getting max_basic_data: {e}")
        max_basic_data = 100  # Default value
    
    try:
        # Explicitly exclude nulls and handle with maximum care 
        valid_daily_data = processed_df.loc[
            (~processed_df['daily_data'].isin([999, 9999])) & 
            (processed_df['daily_data'].notna()), 
            'daily_data'
        ]
        max_daily_data = valid_daily_data.max() if not valid_daily_data.empty else 10
    except Exception as e:
        print(f"Error getting max_daily_data: {e}")
        max_daily_data = 10  # Default value
    
    # Set default values if max_basic_data or max_daily_data is NaN
    if pd.isna(max_basic_data):
        max_basic_data = 100  # Default if no finite values found
        print(f"Using default max_basic_data: {max_basic_data}")
    if pd.isna(max_daily_data):
        max_daily_data = 10   # Default if no finite values found
        print(f"Using default max_daily_data: {max_daily_data}")
    
    print(f"Maximum values: basic_data={max_basic_data}, daily_data={max_daily_data}")
    
    # Replace special values with maximum observed values for modeling
    processed_df['basic_data_clean'] = processed_df['basic_data'].replace({999: max_basic_data, 9999: max_basic_data, -1: 0})
    processed_df['daily_data_clean'] = processed_df['daily_data'].replace({999: max_daily_data, 9999: max_daily_data, -1: 0}).fillna(0)
    
    # Add binary feature for presence of daily data allocation
    processed_df['has_daily_allocation'] = (processed_df['daily_data_clean'] > 0).astype(int)
    
    # Calculate total monthly data with cleaned values
    # For unlimited basic data plans, use the maximum value
    # For plans with unlimited daily data, multiply by 30 to get monthly equivalent
    processed_df['total_data'] = np.where(
        processed_df['basic_data_unlimited'] == 1,
        max_basic_data,  # Maximum observed value for unlimited basic data
        processed_df['basic_data_clean'] + (processed_df['daily_data_clean'] * 30)
    )
    
    # Identify plans with unlimited data AND unlimited speed
    # (unlimited data AND no speed throttling specified = unlimited speed)
    processed_df['has_unlimited_data'] = ((processed_df['basic_data_unlimited'] == 1) | 
                                          (processed_df['daily_data_unlimited'] == 1)).astype(int)
    
    processed_df['has_unlimited_speed'] = ((processed_df['has_unlimited_data'] == 1) & 
                                          (processed_df['speed_when_exhausted'] == 0)).astype(int)
    
    # Flag for any type of unlimited data:
    # 1. Explicitly unlimited (basic or daily data is 999/9999)
    # 2. Throttled unlimited (has speed when exhausted)
    processed_df['any_unlimited_data'] = ((processed_df['basic_data_unlimited'] == 1) | 
                                         (processed_df['daily_data_unlimited'] == 1) |
                                         (processed_df['has_throttled_data'] == 1)).astype(int)
    
    # Create a more nuanced unlimited type classification
    processed_df['unlimited_type'] = np.where(
        (processed_df['has_unlimited_data'] == 1) & (processed_df['has_unlimited_speed'] == 1),
        'unlimited_speed',  # Both data and speed are unlimited
        np.where(
            processed_df['has_throttled_data'] == 1,
            'throttled_unlimited',  # Throttled after quota
            np.where(
                (processed_df['has_unlimited_data'] == 1),
                'unlimited_with_throttling',  # Unlimited data but with throttling
                'limited'  # Truly limited
            )
        )
    )
    
    # For regression, create numeric encoding of unlimited type
    # UPDATED ORDER:
    # 3 = unlimited_speed (unchanged) - Highest value: unlimited data AND speed
    # 2 = throttled_unlimited (was 1) - High value: full speed until quota, then throttled
    # 1 = unlimited_with_throttling (was 2) - Medium value: always throttled despite unlimited data
    # 0 = limited (unchanged) - Lowest value: service stops after quota
    processed_df['unlimited_type_numeric'] = np.where(
        (processed_df['has_unlimited_data'] == 1) & (processed_df['has_unlimited_speed'] == 1),
        3,  # Unlimited data and speed
        np.where(
            processed_df['has_throttled_data'] == 1,
            2,  # Throttled after quota (was 1, now 2)
            np.where(
                (processed_df['has_unlimited_data'] == 1),
                1,  # Unlimited data with throttling (was 2, now 1)
                0   # Truly limited
            )
        )
    )
    
    # 4. Additional feature for throttled speed - higher values are better
    # Create a normalized speed value (0-1 range) for throttled plans
    # Common speeds: 0.4, 1, 3, 5, 10 Mbps
    max_throttle_speed = 10.0  # Maximum typical throttle speed in dataset
    
    processed_df['throttle_speed_normalized'] = np.where(
        processed_df['has_throttled_data'] == 1,
        np.minimum(processed_df['speed_when_exhausted'] / max_throttle_speed, 1.0),
        np.where(
            processed_df['has_unlimited_speed'] == 1,
            1.0,  # Unlimited speed gets maximum value
            0  # Not applicable for other plans
        )
    )
    
    # 5. Feature flags - ensure consistent numeric representation (0/1) instead of boolean (T/F)
    # IMPORTANT: We use astype(int) for all boolean features to ensure consistent 0/1 representation
    # This is critical for machine learning models, especially when applying domain knowledge
    # or monotonic constraints. Boolean True/False can cause issues with some ML libraries
    # and makes feature importance interpretation more difficult.
    for col in ['data_sharing', 'roaming_support', 'micro_payment', 'is_esim', 'signup_minor', 'signup_foreigner']:
        if col in processed_df.columns:
            # First convert to boolean (to handle any string 'True'/'False' values)
            processed_df[col] = processed_df[col].fillna(False).astype(bool)
            # Then convert to integer (0/1)
            processed_df[f'{col}_numeric'] = processed_df[col].astype(int)
    
    # 6. Voice and message processing
    # Handle special values:
    # - Any number containing 3 or more consecutive 9s = unlimited (999, 9999, 9996, etc.)
    # - -1 = not available/not applicable
    # - Other values are treated as actual quotas
    
    # Convert voice and message to numeric if they're strings
    for col in ['voice', 'message']:
        if col in processed_df.columns and processed_df[col].dtype == 'object':
            # Try to convert strings to float/int, keeping NaN values
            try:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                # Fill NaN values with 0
                processed_df[col] = processed_df[col].fillna(0)
            except Exception as e:
                print(f"Error converting {col} to numeric: {e}")
                # If conversion fails, create a new column with defaults
                processed_df[f'{col}_clean'] = 0
                processed_df[f'{col}_unlimited'] = 0
                processed_df[f'{col}_na'] = 1
                continue
    
    def is_unlimited_marker(value):
        """Check if a value contains 3 or more consecutive 9s (999 and up)"""
        if pd.isna(value) or not isinstance(value, (int, float)):
            return False
        # Simple check - if it has '999' in it, it's unlimited
        # This catches 999, 9999, 9996, etc.
        return '999' in str(int(value))
    
    # Find maximum finite values for voice and message (excluding unlimited markers)
    max_voice = processed_df.loc[~processed_df['voice'].apply(is_unlimited_marker) & (processed_df['voice'] != -1), 'voice'].max()
    max_message = processed_df.loc[~processed_df['message'].apply(is_unlimited_marker) & (processed_df['message'] != -1), 'message'].max()
    
    # Create flags for unlimited and not applicable
    for col in ['voice', 'message']:
        if col in processed_df.columns:
            # Create flag for unlimited (anything with 999 in it)
            processed_df[f'{col}_unlimited'] = processed_df[col].apply(is_unlimited_marker).astype(int)
            # Create flag for not applicable
            processed_df[f'{col}_na'] = (processed_df[col] == -1).astype(int)
            
            # Clean values for modeling - replace special values
            # For unlimited, use maximum observed values
            # For not applicable (-1), use 0
            # All other values are kept as is
            max_val = max_voice if col == 'voice' else max_message
            processed_df[f'{col}_clean'] = processed_df[col].copy()
            # Replace unlimited markers with max value
            unlimited_mask = processed_df[col].apply(is_unlimited_marker)
            processed_df.loc[unlimited_mask, f'{col}_clean'] = max_val
            # Replace NA with 0
            na_mask = (processed_df[col] == -1)
            processed_df.loc[na_mask, f'{col}_clean'] = 0
    
    # 7. USIM fee status processing
    # Create binary features to distinguish between zero fees due to being unsupported vs. free
    
    # Process USIM delivery fee status
    if 'usim_delivery_fee_status' in processed_df.columns:
        # Create binary feature for whether USIM is supported
        processed_df['usim_supported'] = (~(processed_df['usim_delivery_fee_status'] == 'UNSUPPORTED')).astype(int)
        
        # Create binary feature for whether USIM is free (when supported)
        processed_df['usim_is_free'] = ((processed_df['usim_delivery_fee_status'].isin(['FREE', 'FREE_ON_ACTIVATION'])) & 
                                        (processed_df['usim_supported'] == 1)).astype(int)
    
    # Process NFC USIM delivery fee status
    if 'nfc_usim_delivery_fee_status' in processed_df.columns:
        # Create binary feature for whether NFC USIM is supported
        processed_df['nfc_usim_supported'] = (~(processed_df['nfc_usim_delivery_fee_status'] == 'UNSUPPORTED')).astype(int)
        
        # Create binary feature for whether NFC USIM is free (when supported)
        processed_df['nfc_usim_is_free'] = ((processed_df['nfc_usim_delivery_fee_status'].isin(['FREE', 'FREE_ON_ACTIVATION'])) & 
                                           (processed_df['nfc_usim_supported'] == 1)).astype(int)
    
    # Process eSIM fee status
    if 'esim_fee_status' in processed_df.columns:
        # Create binary feature for whether eSIM is supported (but we'll prefer is_esim_numeric)
        # We still compute this for backward compatibility, but it will be removed during feature selection
        processed_df['esim_supported'] = (~(processed_df['esim_fee_status'] == 'UNSUPPORTED')).astype(int)
        
        # Create binary feature for whether eSIM is free (when supported)
        processed_df['esim_is_free'] = ((processed_df['esim_fee_status'].isin(['FREE', 'FREE_ON_ACTIVATION'])) & 
                                       (processed_df['esim_supported'] == 1)).astype(int)
    
    # 8. Price features
    # Calculate price per GB (only for plans with finite data)
    processed_df['price_per_gb'] = np.where(
        (processed_df['total_data'] > 0) & (processed_df['any_unlimited_data'] == 0),
        processed_df['fee'] / processed_df['total_data'],
        np.nan  # Don't calculate for unlimited plans
    )
    
    # Calculate a different value metric for unlimited plans
    # Price adjusted for unlimited type (more premium unlimited types are more valuable)
    processed_df['price_unlimited_adjusted'] = np.where(
        processed_df['unlimited_type'] == 'unlimited_speed',
        processed_df['fee'] * 0.7,  # Better value for truly unlimited plans
        np.where(
            processed_df['unlimited_type'] == 'throttled_unlimited',
            processed_df['fee'] * 0.8,  # Good value for throttled after quota (full speed initially)
            np.where(
                processed_df['unlimited_type'] == 'unlimited_with_throttling',
                processed_df['fee'] * 0.9,  # Less attractive: always throttled despite unlimited data
                np.nan  # Not applicable for limited plans
            )
        )
    )
    
    # For all plans, calculate price per month directly
    processed_df['monthly_price'] = processed_df['fee']
    
    # Calculate discount ratio
    processed_df['discount_ratio'] = np.where(
        processed_df['original_fee'] > 0,
        1 - (processed_df['fee'] / processed_df['original_fee']),
        0
    )
    
    # --- START: MODIFICATION TO NORMALIZE original_fee ---
    # Ensure 'original_fee' consistently represents the base price.
    # Condition 1: If discount_period is 0, assume no discount, so original_fee = fee.
    # Condition 2: If discount_period is not 0 AND original_fee is missing/invalid (<=0), use 'fee' as base price.
    # Condition 3: If discount_period is not 0 AND original_fee is valid (>0), keep original_fee.
    
    # Ensure discount_period exists and handle potential NaNs
    if 'discount_period' in processed_df.columns:
        discount_period_filled = processed_df['discount_period'].fillna(-1) # Fill NaN to avoid comparison issues
        
        processed_df['original_fee'] = np.where(
            discount_period_filled == 0, # Condition 1: discount_period is 0
            processed_df['fee'], # Set original_fee = fee
            # Condition 2 & 3: discount_period is not 0
            np.where(
                processed_df['original_fee'] <= 0, # Condition 2: original_fee invalid
                processed_df['fee'], # Use fee as base price
                processed_df['original_fee'] # Condition 3: original_fee valid, keep it
            )
        )
    else:
        # Fallback to simpler logic if discount_period column is missing
        processed_df['original_fee'] = np.where(
            processed_df['original_fee'] <= 0,
            processed_df['fee'], 
            processed_df['original_fee'] 
        )
    # --- END: MODIFICATION TO NORMALIZE original_fee ---
    
    # 9. Agreement features
    processed_df['has_agreement'] = processed_df['agreement'].fillna(False).astype(int)
    processed_df['agreement_period'] = processed_df['agreement_period'].fillna(0)
    
    # 10. Clean up and handle missing values
    # Replace infinities with NaN
    processed_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Basic imputation for remaining NaNs
    numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
    nan_count_before = processed_df[numeric_cols].isna().sum().sum()
    for col in numeric_cols:
        if col in processed_df.columns and processed_df[col].isna().any():
            processed_df[col] = processed_df[col].fillna(processed_df[col].median())
    nan_count_after = processed_df[numeric_cols].isna().sum().sum()
    print(f"NaN values imputed: {nan_count_before} -> {nan_count_after}")
    
    # === BEGIN ENHANCED LOGGING ===
    print("\n=== FINAL PREPROCESSING STATS ===")
    print(f"Output DataFrame shape: {processed_df.shape}")
    print(f"Features generated: {len(processed_df.columns)}")
    
    # Log the basic feature list that will be used for modeling
    basic_features = [
        'is_5g',
        'basic_data_clean',
        'basic_data_unlimited',
        'daily_data_clean',
        'daily_data_unlimited',
        'voice_clean',
        'voice_unlimited',
        'message_clean',
        'message_unlimited',
        'throttle_speed_normalized',
        'tethering_gb',
        'unlimited_type_numeric',
        'additional_call'
    ]
    
    available_basic_features = [f for f in basic_features if f in processed_df.columns]
    print(f"\nBasic features available ({len(available_basic_features)}/{len(basic_features)}):")
    for feature in basic_features:
        status = "✓" if feature in processed_df.columns else "✗" 
        print(f"  {status} {feature}")
    
    # Check data quality of key features
    print("\nKey feature statistics:")
    key_features = ['basic_data_clean', 'daily_data_clean', 'voice_clean', 'message_clean', 
                   'throttle_speed_normalized', 'unlimited_type_numeric',
                   'original_fee']
    
    for feature in key_features:
        if feature in processed_df.columns:
            stats = processed_df[feature].describe()
            print(f"  {feature}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}, median={stats['50%']:.2f}")
    # === END ENHANCED LOGGING ===
    
    return processed_df 