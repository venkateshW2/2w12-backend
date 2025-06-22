# models/responses.py - Enhanced with new fields

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
    """Enhanced basic audio analysis response"""
    filename: str
    duration: float
    tempo: float
    key: str
    rms_energy: float
    spectral_centroid: float
    zero_crossing_rate: float
    sample_rate: int
    status: str
    # NEW: Enhanced fields (optional for backward compatibility)
    tempo_analysis: Optional[Dict[str, Any]] = None
    scale_analysis: Optional[Dict[str, Any]] = None

class AdvancedAudioResponse(BaseModel):
    """Enhanced advanced audio analysis response"""
    filename: str
    key_detection: Dict[str, Any]
    tempo_analysis: Dict[str, Any] 
    spectral_features: Dict[str, Any]
    duration: float
    sample_rate: int
    status: str
    # NEW: Enhanced fields
    mfcc_features: Optional[Dict[str, Any]] = None
    transient_markers: Optional[List[Dict[str, Any]]] = None

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

# NEW: Enhanced responses for new endpoints
class TransientResponse(BaseModel):
    """Transient timeline response"""
    filename: str
    status: str
    file_info: Dict[str, Any]
    transient_markers: List[Dict[str, Any]]
    total_transients: int

class MemoryStatusResponse(BaseModel):
    """Memory usage response"""
    rss_mb: float
    vms_mb: float
    percent: float