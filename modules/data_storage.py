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
CHARTS_FILE = STORAGE_DIR / "charts.json"

# Ensure storage directory exists
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

def save_rankings_data(df_with_rankings: pd.DataFrame, cost_structure: Dict[str, Any], method: str = "fixed_rates", charts_data: Optional[Dict[str, Any]] = None) -> bool:
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
        
        # Save charts data if provided
        if charts_data:
            with open(CHARTS_FILE, 'w', encoding='utf-8') as f:
                json.dump(charts_data, f, ensure_ascii=False, indent=2, default=str)
            logger.info(f"Successfully saved charts data: {list(charts_data.keys())}")
        
        logger.info(f"Successfully saved rankings data: {len(df_with_rankings)} plans, method: {method}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save rankings data: {e}")
        return False

def load_rankings_data() -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, Any]], Optional[str], Optional[Dict[str, Any]]]:
    """
    Load rankings data and cost structure from files
    Returns: (df_with_rankings, cost_structure, method, charts_data)
    """
    try:
        # Check if files exist
        if not all(f.exists() for f in [RANKINGS_FILE, COST_STRUCTURE_FILE, METADATA_FILE]):
            logger.info("Rankings data files not found")
            return None, None, None, None
        
        # Load metadata first
        with open(METADATA_FILE, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        if not metadata.get("has_data", False):
            logger.info("No rankings data available")
            return None, None, None, None
        
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
        
        # Load charts data if available
        charts_data = None
        if CHARTS_FILE.exists():
            with open(CHARTS_FILE, 'r', encoding='utf-8') as f:
                charts_data = json.load(f)
            logger.info(f"Successfully loaded charts data: {list(charts_data.keys()) if charts_data else 'empty'}")
        
        logger.info(f"Successfully loaded rankings data: {len(df_with_rankings)} plans, method: {method}")
        return df_with_rankings, cost_structure, method, charts_data
        
    except Exception as e:
        logger.error(f"Failed to load rankings data: {e}")
        return None, None, None, None

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
        for file_path in [RANKINGS_FILE, COST_STRUCTURE_FILE, METADATA_FILE, CHARTS_FILE]:
            if file_path.exists():
                file_path.unlink()
        
        logger.info("Successfully cleared rankings data")
        return True
    
    except Exception as e:
        logger.error(f"Failed to clear rankings data: {e}")
        return False

def calculate_and_save_charts(df_ranked: pd.DataFrame, method: str, cost_structure: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate all charts data synchronously and return for saving
    """
    charts_data = {}
    
    try:
        # Import chart calculation functions
        from .report_charts import prepare_feature_frontier_data, prepare_marginal_cost_frontier_data, prepare_granular_marginal_cost_frontier_data
        from .report_html import prepare_plan_efficiency_data
        
        logger.info("Starting synchronous chart calculations...")
        
        # 1. Feature Frontier Charts
        try:
            core_features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
            charts_data['feature_frontier'] = prepare_feature_frontier_data(df_ranked, core_features)
            logger.info("✓ Feature frontier chart data calculated")
        except Exception as e:
            logger.error(f"Failed to calculate feature frontier chart: {e}")
            charts_data['feature_frontier'] = None
        
        # 2. Marginal Cost Frontier Charts (if data available)
        try:
            if hasattr(df_ranked, 'attrs') and 'multi_frontier_breakdown' in df_ranked.attrs:
                multi_frontier_breakdown = df_ranked.attrs['multi_frontier_breakdown']
                core_features = ['basic_data_clean', 'voice_clean', 'message_clean', 'tethering_gb']
                charts_data['marginal_cost_frontier'] = prepare_granular_marginal_cost_frontier_data(
                    df_ranked, multi_frontier_breakdown, core_features
                )
                logger.info("✓ Marginal cost frontier chart data calculated")
            else:
                charts_data['marginal_cost_frontier'] = None
                logger.info("No multi_frontier_breakdown data available for marginal cost frontier")
        except Exception as e:
            logger.error(f"Failed to calculate marginal cost frontier chart: {e}")
            charts_data['marginal_cost_frontier'] = None
        
        # 3. Plan Efficiency Charts
        try:
            charts_data['plan_efficiency'] = prepare_plan_efficiency_data(df_ranked, method)
            logger.info("✓ Plan efficiency chart data calculated")
        except Exception as e:
            logger.error(f"Failed to calculate plan efficiency chart: {e}")
            charts_data['plan_efficiency'] = None
        
        logger.info(f"Charts calculation completed. Available charts: {[k for k, v in charts_data.items() if v is not None]}")
        return charts_data
        
    except Exception as e:
        logger.error(f"Failed to calculate charts: {e}")
        return {}