import os
import json
from pathlib import Path
from datetime import datetime

def ensure_directories():
    """
    Ensure that all necessary directories exist.
    This function creates the minimal directory structure needed for the server.
    
    In Hugging Face Spaces, we need to use paths within the app directory,
    which is the only writable location.
    """
    # Define directories directly under the app directory
    # Using Path('.') to get the current working directory - in HF this will be /app
    directories = [
        Path('./data/raw'),
        Path('./data/processed'),
        Path('./trained_models/xgboost/model'),
        Path('./trained_models/xgboost/config'),
        Path('./results'),
        Path('./reports')  # For HTML reports
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"Successfully created directory: {directory}")
        except PermissionError as e:
            print(f"Permission error creating directory {directory}: {e}")
            # Try to create in /tmp as fallback
            fallback_dir = Path(f"/tmp{directory}")
            try:
                os.makedirs(fallback_dir, exist_ok=True)
                print(f"Created fallback directory: {fallback_dir}")
                # Replace the original directory with the fallback
                # Find the index of the original directory
                idx = directories.index(directory)
                directories[idx] = fallback_dir
            except Exception as e2:
                print(f"Failed to create fallback directory: {e2}")
    
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
    
    # Define file path - using a simpler, direct path
    data_dir = Path("./data/raw")
    
    try:
        os.makedirs(data_dir, exist_ok=True)
        file_path = data_dir / f"received_data_{timestamp}.json"
    except PermissionError:
        # Fallback to /tmp if app directory is not writable
        data_dir = Path("/tmp/data/raw")
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
        timestamp: Optional timestamp (default: current time)
        save_latest: Whether to save as latest (default: True)
        
    Returns:
        Tuple of (timestamped_path, latest_path)
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define file paths - using simpler, direct paths
    processed_dir = Path("./data/processed")
    
    try:
        os.makedirs(processed_dir, exist_ok=True)
        timestamped_path = processed_dir / f"processed_data_{timestamp}.csv"
        latest_path = processed_dir / "latest_processed_data.csv"
    except PermissionError:
        # Fallback to /tmp if app directory is not writable
        processed_dir = Path("/tmp/data/processed") 
        os.makedirs(processed_dir, exist_ok=True)
        timestamped_path = processed_dir / f"processed_data_{timestamp}.csv"
        latest_path = processed_dir / "latest_processed_data.csv"
    
    # Save files
    df.to_csv(timestamped_path, index=False)
    if save_latest:
        df.to_csv(latest_path, index=False)
    
    return str(timestamped_path), str(latest_path)

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
        # Simplified path for model config
        config_dir = Path("./trained_models/xgboost/config")
        try:
            os.makedirs(config_dir, exist_ok=True)
            config_path = config_dir / "xgboost_model_config.json"
        except PermissionError:
            # Fallback to /tmp if app directory is not writable
            config_dir = Path("/tmp/trained_models/xgboost/config")
            os.makedirs(config_dir, exist_ok=True)
            config_path = config_dir / "xgboost_model_config.json"
    else:
        config_path = Path(path)
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
        except PermissionError:
            # If we can't write to the specified path, use a fallback in /tmp
            fallback_path = Path(f"/tmp/{os.path.basename(path)}")
            config_path = fallback_path
    
    # Save config
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    
    return str(config_path) 