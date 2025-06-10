import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Configure logging
logger = logging.getLogger(__name__)

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

def cleanup_old_files(directory: Path, pattern: str = "*", max_files: int = 5, max_age_days: int = 7):
    """
    Clean up old files in a directory to prevent disk space issues.
    
    Args:
        directory: Directory to clean up
        pattern: File pattern to match (default: "*" for all files)
        max_files: Maximum number of files to keep (keeps newest)
        max_age_days: Maximum age in days for files to keep
        
    Returns:
        Number of files deleted
    """
    if not directory.exists():
        return 0
    
    try:
        # Get all files matching pattern
        files = list(directory.glob(pattern))
        if not files:
            return 0
        
        # Sort by modification time (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Calculate cutoff date
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        cutoff_timestamp = cutoff_date.timestamp()
        
        deleted_count = 0
        
        # Delete files beyond max_files or older than max_age_days
        for i, file_path in enumerate(files):
            should_delete = False
            
            # Delete if beyond max_files limit
            if i >= max_files:
                should_delete = True
                reason = f"beyond limit ({i+1}/{max_files})"
            
            # Delete if older than max_age_days
            elif file_path.stat().st_mtime < cutoff_timestamp:
                should_delete = True
                file_age = (datetime.now() - datetime.fromtimestamp(file_path.stat().st_mtime)).days
                reason = f"too old ({file_age} days)"
            
            if should_delete:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path.name} ({reason})")
                    deleted_count += 1
                except Exception as e:
                    logger.error(f"Failed to delete {file_path}: {e}")
        
        if deleted_count > 0:
            logger.info(f"Cleanup completed: deleted {deleted_count} files from {directory}")
        
        return deleted_count
        
    except Exception as e:
        logger.error(f"Error during cleanup of {directory}: {e}")
        return 0

def cleanup_all_datasets(max_files: int = 5, max_age_days: int = 7):
    """
    Clean up all old dataset files comprehensively since pipeline recalculates everything from scratch.
    Removes raw data, processed data, reports, and any intermediate files.
    
    Args:
        max_files: Maximum number of files to keep per category
        max_age_days: Maximum age in days for files to keep
        
    Returns:
        Dictionary with cleanup statistics
    """
    logger.info("Starting comprehensive cleanup of old dataset files...")
    
    cleanup_stats = {
        'raw_data': 0,
        'processed_data': 0,
        'reports': 0,
        'intermediate': 0,
        'total': 0
    }
    
    # Define comprehensive cleanup targets - including all intermediate files
    cleanup_targets = [
        # Raw data files (timestamped)
        {
            'directory': Path('./data/raw'),
            'pattern': 'raw_data_*.json',
            'category': 'raw_data'
        },
        {
            'directory': Path('/tmp/data/raw'),
            'pattern': 'raw_data_*.json',
            'category': 'raw_data'
        },
        # Raw data files (any received data)
        {
            'directory': Path('./data/raw'),
            'pattern': 'received_data_*.json',
            'category': 'raw_data'
        },
        {
            'directory': Path('/tmp/data/raw'),
            'pattern': 'received_data_*.json',
            'category': 'raw_data'
        },
        
        # Processed data files (timestamped)
        {
            'directory': Path('./data/processed'),
            'pattern': 'processed_data_*.csv',
            'category': 'processed_data'
        },
        {
            'directory': Path('/tmp/data/processed'),
            'pattern': 'processed_data_*.csv',
            'category': 'processed_data'
        },
        # Latest processed data files (these become obsolete too)
        {
            'directory': Path('./data/processed'),
            'pattern': 'latest_processed_data.csv',
            'category': 'processed_data'
        },
        {
            'directory': Path('/tmp/data/processed'),
            'pattern': 'latest_processed_data.csv',
            'category': 'processed_data'
        },
        
        # HTML reports (all ranking reports)
        {
            'directory': Path('./reports'),
            'pattern': 'cs_ranking_*.html',
            'category': 'reports'
        },
        {
            'directory': Path('/tmp/reports'),
            'pattern': 'cs_ranking_*.html',
            'category': 'reports'
        },
        {
            'directory': Path('./reports'),
            'pattern': '*ranking_*.html',
            'category': 'reports'
        },
        {
            'directory': Path('/tmp/reports'),
            'pattern': '*ranking_*.html',
            'category': 'reports'
        },
        
        # Any other intermediate files that might accumulate
        {
            'directory': Path('./results'),
            'pattern': '*.csv',
            'category': 'intermediate'
        },
        {
            'directory': Path('./results'),
            'pattern': '*.json',
            'category': 'intermediate'
        },
        {
            'directory': Path('/tmp/results'),
            'pattern': '*.csv',
            'category': 'intermediate'
        },
        {
            'directory': Path('/tmp/results'),
            'pattern': '*.json',
            'category': 'intermediate'
        }
    ]
    
    # Import config to get proper directories
    try:
        from .config import config
        if hasattr(config, 'cs_raw_dir'):
            cleanup_targets.extend([
                {
                    'directory': config.cs_raw_dir,
                    'pattern': 'raw_data_*.json',
                    'category': 'raw_data'
                },
                {
                    'directory': config.cs_raw_dir,
                    'pattern': 'received_data_*.json',
                    'category': 'raw_data'
                }
            ])
        if hasattr(config, 'cs_processed_dir'):
            cleanup_targets.extend([
                {
                    'directory': config.cs_processed_dir,
                    'pattern': 'processed_data_*.csv',
                    'category': 'processed_data'
                },
                {
                    'directory': config.cs_processed_dir,
                    'pattern': 'latest_processed_data.csv',
                    'category': 'processed_data'
                }
            ])
        if hasattr(config, 'cs_report_dir'):
            cleanup_targets.extend([
                {
                    'directory': config.cs_report_dir,
                    'pattern': 'cs_ranking_*.html',
                    'category': 'reports'
                },
                {
                    'directory': config.cs_report_dir,
                    'pattern': '*ranking_*.html',
                    'category': 'reports'
                }
            ])
    except ImportError:
        logger.warning("Could not import config, using default paths only")
    
    # Perform cleanup for each target
    for target in cleanup_targets:
        deleted = cleanup_old_files(
            target['directory'], 
            target['pattern'], 
            max_files, 
            max_age_days
        )
        cleanup_stats[target['category']] += deleted
        cleanup_stats['total'] += deleted
    
    logger.info(f"Comprehensive cleanup completed: {cleanup_stats}")
    return cleanup_stats

# XGBoost-related functions have been removed