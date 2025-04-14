import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import xgboost as xgb
import json
from datetime import datetime

class XGBoostModel:
    """
    A simplified XGBoost model implementation without domain knowledge support.
    """
    
    def __init__(self, use_domain_knowledge=False, feature_names=None):
        self.model_type = 'xgboost'
        self.use_domain_knowledge = use_domain_knowledge
        self.feature_names = feature_names
        self.model = None
        
        # Default XGBoost parameters
        self.params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'nthread': -1,
            'seed': 42
        }
    
    def train(self, X, y):
        """Train the XGBoost model."""
        # Ensure X is using the expected features
        if self.feature_names:
            X = X[self.feature_names].copy()
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X, label=y, feature_names=X.columns.tolist())
        
        # Train the model
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=500,
            verbose_eval=False
        )
        
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        
        # Ensure X is using the expected features
        if self.feature_names and all(f in X.columns for f in self.feature_names):
            X = X[self.feature_names].copy()
        
        # Create DMatrix for prediction
        dtest = xgb.DMatrix(X, feature_names=X.columns.tolist())
        
        # Make predictions
        return self.model.predict(dtest)
    
    def save(self, path=None):
        """
        Save the model to disk using XGBoost's native JSON format for better interoperability.
        
        Args:
            path: Optional path to save (default: standard location)
            
        Returns:
            str: Path to the saved model file
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Define paths for model and metadata
        if path is None:
            # Try to use default path in the current directory structure
            try:
                base_path = Path(os.path.dirname(os.path.abspath(__file__))) / "../../trained_models/xgboost"
                os.makedirs(base_path, exist_ok=True)
                model_path = base_path / "model.json"
                metadata_path = base_path / "model_metadata.json"
            except PermissionError:
                # Fallback to /tmp if we don't have permission to write to the app directory
                print("Permission denied for app directory. Falling back to /tmp for model storage.")
                base_path = Path("/tmp/trained_models/xgboost")
                os.makedirs(base_path, exist_ok=True)
                model_path = base_path / "model.json"
                metadata_path = base_path / "model_metadata.json"
        else:
            try:
                base_path = Path(path)
                os.makedirs(base_path, exist_ok=True)
                model_path = base_path / "model.json"
                metadata_path = base_path / "model_metadata.json"
            except PermissionError:
                # Fallback to /tmp if specified path isn't writable
                print(f"Permission denied for path {path}. Falling back to /tmp for model storage.")
                base_path = Path("/tmp/trained_models/xgboost")
                os.makedirs(base_path, exist_ok=True)
                model_path = base_path / "model.json"
                metadata_path = base_path / "model_metadata.json"
        
        # Save the XGBoost model in native JSON format
        self.model.save_model(str(model_path))
        
        # Save metadata separately (feature names, etc.)
        metadata = {
            'feature_names': self.feature_names,
            'use_domain_knowledge': self.use_domain_knowledge,
            'params': self.params,
            'model_type': self.model_type,
            'saved_date': datetime.now().isoformat()
        }
        
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            print(f"Error saving metadata: {e}. Model saved but metadata may be missing.")
        
        return str(model_path)
    
    @classmethod
    def load(cls, model_path=None):
        """
        Load a saved model from disk using XGBoost's native format.
        
        Args:
            model_path: Optional path to load from (default: standard location)
            
        Returns:
            XGBoostModel: Loaded model instance
        """
        # Define paths for model and metadata
        if model_path is None:
            # Try default path in current directory structure
            base_path = Path(os.path.dirname(os.path.abspath(__file__))) / "../../trained_models/xgboost"
            model_path = base_path / "model.json"
            metadata_path = base_path / "model_metadata.json"
            
            # If not found, try /tmp fallback location
            if not os.path.exists(model_path):
                base_path = Path("/tmp/trained_models/xgboost")
                model_path = base_path / "model.json"
                metadata_path = base_path / "model_metadata.json"
        else:
            base_path = Path(model_path)
            model_path = base_path / "model.json"
            metadata_path = base_path / "model_metadata.json"
        
        # Check if model file exists
        if not os.path.exists(model_path):
            # Try .pkl extension for backward compatibility
            pkl_path = str(model_path).replace('.json', '.pkl')
            if os.path.exists(pkl_path):
                # Load using old pickle method
                return cls._load_legacy_model(pkl_path)
            else:
                raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Create a new instance
        instance = cls()
        
        # Create a new booster and load the model
        instance.model = xgb.Booster()
        instance.model.load_model(str(model_path))
        
        # Load metadata if available
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Set attributes from metadata
            instance.feature_names = metadata.get('feature_names')
            instance.use_domain_knowledge = metadata.get('use_domain_knowledge', False)
            instance.params = metadata.get('params', instance.params)
        
        return instance
    
    @classmethod
    def _load_legacy_model(cls, pkl_path):
        """Load a model using the legacy pickle format for backward compatibility."""
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
        
        # Create a new instance
        instance = cls(
            use_domain_knowledge=data.get('use_domain_knowledge', False),
            feature_names=data.get('feature_names', None)
        )
        
        # Set attributes from pickled data
        instance.model = data.get('model')
        instance.params = data.get('params', instance.params)
        
        return instance
    
    def get_params(self):
        """Get the model hyperparameters."""
        return self.params
    
    def get_monotonicity_constraints(self):
        """Get the monotonicity constraints used by the model."""
        return {}  # No constraints in this simplified version

def get_model(model_type='xgboost', use_domain_knowledge=False, feature_names=None, **kwargs):
    """
    Factory function to get the requested model type.
    Currently only supports XGBoost.
    """
    if model_type.lower() == 'xgboost':
        return XGBoostModel(use_domain_knowledge=use_domain_knowledge, feature_names=feature_names)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

def get_basic_feature_list():
    """Return a list of basic feature names used for modeling."""
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