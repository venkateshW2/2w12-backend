# models/requests.py
from pydantic import BaseModel
from typing import Optional

class AudioAnalysisRequest(BaseModel):
    """Request model for audio analysis"""
    analysis_type: Optional[str] = "basic"
    include_features: Optional[bool] = True
    
class BatchAnalysisRequest(BaseModel):
    """Request model for batch analysis"""
    analysis_type: Optional[str] = "basic"
    include_statistics: Optional[bool] = True