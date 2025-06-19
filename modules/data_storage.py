"""
File-based data storage module for sharing data across processes
"""
import json
import os
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# Storage paths
STORAGE_DIR = Path("/app/data/shared")
RANKINGS_FILE = STORAGE_DIR / "rankings.json"
COST_STRUCTURE_FILE = STORAGE_DIR / "cost_structure.json"
METADATA_FILE = STORAGE_DIR / "metadata.json"

# Ensure storage directory exists
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def save_rankings_data(df_with_rankings: pd.DataFrame, cost_structure: Dict[str, Any], method: str = "fixed_rates") -> bool:
    """
    Save rankings data and cost structure to files
    """
    try:
        # Save DataFrame as JSON
        rankings_data = {
            "data": df_with_rankings.to_dict('records'),
            "columns": list(df_with_rankings.columns),
            "shape": df_with_rankings.shape,
            "attrs": getattr(df_with_rankings, 'attrs', {})
        }
        
        with open(RANKINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(rankings_data, f, ensure_ascii=False, indent=2, default=str)
        
        # Save cost structure
        with open(COST_STRUCTURE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cost_structure, f, ensure_ascii=False, indent=2, default=str)
        
        # Save metadata
        metadata = {
            "method": method,
            "timestamp": pd.Timestamp.now().isoformat(),
            "has_data": True,
            "num_plans": len(df_with_rankings)
        }
        
        with open(METADATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Successfully saved rankings data: {len(df_with_rankings)} plans, method: {method}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save rankings data: {e}")
        return False

def load_rankings_data() -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str]]:
    """
    Load rankings data and cost structure from files
    Returns: (df_with_rankings, cost_structure, method)
    """
    try:
        # Check if files exist
        if not all(f.exists() for f in [RANKINGS_FILE, COST_STRUCTURE_FILE, METADATA_FILE]):
            logger.info("Rankings data files not found")
            return None, None, None
        
        # Load metadata first
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if not metadata.get("has_data", False):
            logger.info("No rankings data available")
            return None, None, None
        
        # Load DataFrame
        with open(RANKINGS_FILE, 'r', encoding='utf-8') as f:
            rankings_data = json.load(f)
        
        df_with_rankings = pd.DataFrame(rankings_data["data"])
        
        # Restore DataFrame attributes
        if "attrs" in rankings_data and rankings_data["attrs"]:
            df_with_rankings.attrs = rankings_data["attrs"]
        
        # Load cost structure
        with open(COST_STRUCTURE_FILE, 'r', encoding='utf-8') as f:
            cost_structure = json.load(f)
        
        method = metadata.get("method", "fixed_rates")
        
        logger.info(f"Successfully loaded rankings data: {len(df_with_rankings)} plans, method: {method}")
        return df_with_rankings, cost_structure, method
        
    except Exception as e:
        logger.error(f"Failed to load rankings data: {e}")
        return None, None, None

def has_rankings_data() -> bool:
    """
    Check if rankings data is available
    """
    try:
        if not METADATA_FILE.exists():
            return False
        
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        return metadata.get("has_data", False)
    
    except Exception as e:
        logger.error(f"Failed to check rankings data: {e}")
        return False

def get_data_info() -> Dict[str, Any]:
    """
    Get information about stored data
    """
    try:
        if not METADATA_FILE.exists():
            return {"has_data": False, "error": "No metadata file"}
        
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Add file sizes
        file_info = {}
        for name, path in [("rankings", RANKINGS_FILE), ("cost_structure", COST_STRUCTURE_FILE)]:
            if path.exists():
                file_info[f"{name}_file_size"] = path.stat().st_size
                file_info[f"{name}_file_exists"] = True
            else:
                file_info[f"{name}_file_exists"] = False
        
        return {**metadata, **file_info}
    
    except Exception as e:
        logger.error(f"Failed to get data info: {e}")
        return {"has_data": False, "error": str(e)}

def clear_rankings_data() -> bool:
    """
    Clear all stored rankings data
    """
    try:
        for file_path in [RANKINGS_FILE, COST_STRUCTURE_FILE, METADATA_FILE]:
            if file_path.exists():
                file_path.unlink()
        
        logger.info("Successfully cleared rankings data")
        return True
    
    except Exception as e:
        logger.error(f"Failed to clear rankings data: {e}")
        return False 