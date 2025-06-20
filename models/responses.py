# models/responses.py

from pydantic import BaseModel
from typing import Optional, Dict, List, Any
from datetime import datetime

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    version: str
    endpoints: List[str]
    features: List[str]

class BasicAudioResponse(BaseModel):
    """Basic audio analysis response"""
    filename: str
    duration: float
    tempo: float
    key: str
    rms_energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    sample_rate: int
    status: str

class AdvancedAudioResponse(BaseModel):
    """Advanced audio analysis response"""
    filename: str
    key_detection: Dict[str, Any]
    tempo_analysis: Dict[str, Any] 
    spectral_features: Dict[str, Any]
    duration: float
    sample_rate: int
    status: str

class GenreResponse(BaseModel):
    """Genre classification response"""
    filename: str
    genre: str
    confidence: float
    features: Dict[str, Any]
    status: str

class MoodResponse(BaseModel):
    """Mood detection response"""
    filename: str
    mood: str
    confidence: float
    features: Dict[str, Any]
    status: str

class LoudnessResponse(BaseModel):
    """Loudness analysis response"""
    filename: str
    loudness_analysis: Dict[str, Any]
    recommendations: List[str]
    status: str

class MFCCResponse(BaseModel):
    """MFCC analysis response"""
    filename: str
    mfcc_analysis: Dict[str, Any]
    status: str

class AudioTaggingResponse(BaseModel):
    """Audio tagging response"""
    filename: str
    content_analysis: Dict[str, Any]
    auto_tags: List[str]
    status: str

class BatchResponse(BaseModel):
    """Batch processing response"""
    total_files: int
    successful: int
    failed: int
    results: List[Dict[str, Any]]
    batch_statistics: Optional[Dict[str, Any]] = None