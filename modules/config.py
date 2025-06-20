"""
Configuration module for the ranking model server.
"""

import os
from pathlib import Path
import logging
import pandas as pd
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define Paths
APP_DIR = Path(__file__).parent.parent  # Should resolve to /app
DATA_DIR = APP_DIR / "data"
REPORT_DIR_BASE = Path("/tmp/reports")  # Use /tmp for reports in container

# Create necessary directories
os.makedirs(REPORT_DIR_BASE, exist_ok=True)
os.makedirs(DATA_DIR / "raw", exist_ok=True)
os.makedirs(DATA_DIR / "processed", exist_ok=True)
os.makedirs(DATA_DIR / "test", exist_ok=True)
os.makedirs(DATA_DIR / "results" / "latest", exist_ok=True)
os.makedirs(DATA_DIR / "results" / "archive", exist_ok=True)

# Set environment variables
os.environ["PYTHONPATH"] = str(APP_DIR)

# Configuration class
from dataclasses import dataclass

class Config:
    """Application configuration."""
    def __init__(self):
        # Base paths
        self.app_dir: Path = APP_DIR
        self.data_dir: Path = DATA_DIR
        self.report_dir: Path = REPORT_DIR_BASE
        
        # CS-specific paths
        self.cs_input_dir: Path = DATA_DIR / "cs_input"
        self.cs_raw_dir: Path = DATA_DIR / "raw"
        self.cs_processed_dir: Path = DATA_DIR / "processed"
        self.cs_report_dir: Path = REPORT_DIR_BASE / "cs_reports"
        
        # Global state - initialize as None but allow assignment
        self.df_with_rankings: Optional[pd.DataFrame] = None
        self.latest_logical_test_results: Optional[Dict[str, Any]] = None
        
        # Call post-init setup
        self._post_init()
    
    def _post_init(self):
        """Initialize configuration after dataclass creation."""
        # Ensure all directories exist
        for path in [
            self.app_dir,
            self.data_dir,
            self.report_dir,
            self.cs_input_dir,
            self.cs_raw_dir,
            self.cs_processed_dir,
            self.cs_report_dir
        ]:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directory: {path}")

        # Set environment variables
        os.environ["PYTHONPATH"] = str(self.app_dir)
        logger.info(f"Set PYTHONPATH to: {self.app_dir}")

# Feature set definitions
FEATURE_SETS = {
    'basic': [
        'basic_data_clean', 'basic_data_unlimited',
        'daily_data_clean', 'daily_data_unlimited',
        'voice_clean', 'voice_unlimited',
        'message_clean', 'message_unlimited',
        'additional_call', 'is_5g',
        'tethering_gb', 'speed_when_exhausted',
        'data_throttled_after_quota', 'data_unlimited_speed', 'has_unlimited_speed'
        # REMOVED: only 'data_stops_after_quota' per user request
    ]
}

# Unlimited flag mappings
UNLIMITED_FLAGS = {
    'basic_data_clean': 'basic_data_unlimited',
    'daily_data_clean': 'daily_data_unlimited',
    'voice_clean': 'voice_unlimited',
    'message_clean': 'message_unlimited',
    'speed_when_exhausted': 'has_unlimited_speed'
}

# Use all features for frontier calculation, not just core ones
CORE_FEATURES = FEATURE_SETS['basic']

# Create global config instance
config = Config()
