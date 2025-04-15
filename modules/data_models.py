from pydantic import BaseModel
from typing import List, Optional, Union

# NOTE: This is an inferred structure based on app.py usage.
# It might need refinement based on the original data_models.py.

class PlanInput(BaseModel):
    # Infer structure from PlanData in app.py, assuming prediction input might be similar
    # Or it might just be the feature list. Let's assume it's the features for now.
    # Using the basic feature list as a likely default structure for prediction input.
    is_5g: Optional[int] = None
    basic_data_clean: Optional[float] = None
    basic_data_unlimited: Optional[int] = None
    daily_data_clean: Optional[float] = None
    daily_data_unlimited: Optional[int] = None
    voice_clean: Optional[float] = None
    voice_unlimited: Optional[int] = None
    message_clean: Optional[float] = None
    message_unlimited: Optional[int] = None
    throttle_speed_normalized: Optional[float] = None
    tethering_gb: Optional[float] = None
    unlimited_type_numeric: Optional[int] = None
    additional_call: Optional[int] = None
    # Add other potential fields if needed based on how PlanInput is actually used

    class Config:
        extra = 'allow' # Allow extra fields if the input has more than defined

# Placeholder for FeatureDefinitions, as its usage isn't clear from app.py
class FeatureDefinitions(BaseModel):
    pass 