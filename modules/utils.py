import os
import json
from pathlib import Path
from datetime import datetime

def ensure_directories():
    """
    Ensure that all necessary directories exist.
    This function creates the minimal directory structure needed for the server.
    """
    # Define the base directory (relative to this file)
    base_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "../.."
    
    # Define required directories - removed archive directories
    directories = [
        base_dir / "data" / "raw",
        base_dir / "data" / "processed",
        base_dir / "trained_models" / "xgboost" / "with_domain" / "basic" / "standard" / "model",
        base_dir / "trained_models" / "xgboost" / "with_domain" / "basic" / "standard" / "config",
        base_dir / "results" / "latest"
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    return directories

def save_raw_data(data, timestamp=None):
    """
    Save raw data as JSON file.
    
    Args:
        data: Data to save
        timestamp: Optional timestamp (default: current time)
        
    Returns:
        Path to the saved file
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define file path
    data_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "../../data/raw"
    os.makedirs(data_dir, exist_ok=True)
    file_path = data_dir / f"received_data_{timestamp}.json"
    
    # Save file
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    
    return str(file_path)

def save_processed_data(df, timestamp=None, save_latest=True):
    """
    Save processed data as CSV file.
    Simplified to focus on saving only the latest version.
    
    Args:
        df: DataFrame to save
        timestamp: Ignored in this implementation
        save_latest: Ignored in this implementation
        
    Returns:
        Path to the saved file
    """
    # Define file path
    processed_dir = Path(os.path.dirname(os.path.abspath(__file__))) / "../../data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    latest_path = processed_dir / "latest_processed_data.csv"
    
    # Save file
    df.to_csv(latest_path, index=False)
    
    return str(latest_path), str(latest_path)  # Return the same path twice for compatibility

def get_basic_feature_list():
    """
    Return the basic feature list used for modeling.
    
    Returns:
        List of feature names in the basic feature set
    """
    return [
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

def format_model_config(model, feature_names, dataset_info):
    """
    Format model configuration for saving.
    
    Args:
        model: The trained model
        feature_names: List of feature names used
        dataset_info: Information about the dataset
        
    Returns:
        Dictionary with model configuration
    """
    return {
        "model_type": "xgboost",
        "feature_set": "basic",
        "training_date": datetime.now().isoformat(),
        "hyperparameters": getattr(model, 'get_params', lambda: "Not Available")(),
        "feature_columns_used": feature_names,
        "monotonicity_constraints_applied": getattr(model, 'get_monotonicity_constraints', lambda: "Not Available")(),
        "use_domain_knowledge": True,
        "dataset_type": "standard",
        "relaxed_constraints": False,
        "use_gradient_penalty": False,
        "training_data_info": dataset_info
    }

def save_model_config(config, path=None):
    """
    Save model configuration as JSON file.
    
    Args:
        config: Configuration dictionary
        path: Optional path to save (default: standard location)
        
    Returns:
        Path to the saved file
    """
    if path is None:
        # Default path for model config
        config_dir = Path(os.path.dirname(os.path.abspath(__file__))) / \
                     "../../trained_models/xgboost/with_domain/basic/standard/config"
        os.makedirs(config_dir, exist_ok=True)
        config_path = config_dir / "xgboost_with_domain_basic_standard_config.json"
    else:
        config_path = Path(path)
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Save config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    
    return str(config_path) 