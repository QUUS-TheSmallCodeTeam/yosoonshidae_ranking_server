"""
Configuration module for the ranking model server.
"""

import os
from pathlib import Path
import logging

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
os.makedirs(DATA_DIR / "trained_models" / "xgboost" / "with_domain" / "basic" / "standard" / "model", exist_ok=True)
os.makedirs(DATA_DIR / "trained_models" / "xgboost" / "with_domain" / "basic" / "standard" / "config", exist_ok=True)
os.makedirs(DATA_DIR / "results" / "latest", exist_ok=True)
os.makedirs(DATA_DIR / "results" / "archive", exist_ok=True)

# Set environment variables
os.environ["PYTHONPATH"] = str(APP_DIR)

# Configuration class
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Application configuration."""
    app_dir: Path = APP_DIR
    data_dir: Path = DATA_DIR
    report_dir: Path = REPORT_DIR_BASE
    
    # Global state
    df_with_rankings: Optional[pd.DataFrame] = None
    latest_logical_test_results: Optional[dict] = None

# Create global config instance
config = Config()
