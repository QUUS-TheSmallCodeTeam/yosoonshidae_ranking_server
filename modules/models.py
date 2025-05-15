import numpy as np
import pandas as pd
import os
from pathlib import Path
import json
from datetime import datetime

# This file previously contained the XGBoostModel class and related functions.
# Both the XGBoost and Spearman implementations have been removed in favor
# of the Cost-Spec ratio method.

# If model functionality is needed in the future, it should be implemented here.

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