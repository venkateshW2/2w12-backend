"""
Pydantic models for visualization API responses
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from datetime import datetime

class CompleteAnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    visualization: Dict[str, Any]
    playback: Dict[str, Any]
    processing_time: float

class FileStatusResponse(BaseModel):
    file_id: str
    status: str
    created: datetime
    last_accessed: datetime
    expires_at: datetime
    original_size: int
    compressed_size: Optional[int] = None
    error_message: Optional[str] = None
    
class WaveformData(BaseModel):
    data: List[float]
    time: List[float]
    points: int

class SpectrogramData(BaseModel):
    data: List[List[int]]  # 2D array as list of lists
    frequencies: List[float]
    time_bins: List[float]
    shape: List[int]
    db_range: List[float]

class VisualizationMetadata(BaseModel):
    duration: float
    sample_rate: int
    samples: int
    channels: int

class VisualizationResponse(BaseModel):
    waveform: Dict[str, WaveformData]
    spectrogram: SpectrogramData
    metadata: VisualizationMetadata

class PlaybackInfo(BaseModel):
    file_id: str
    stream_url: str
    compressed_size: int
    format: str
    expires_at: datetime

class CompleteAnalysisResponse(BaseModel):
    analysis: Dict[str, Any]  # Your existing analysis results
    visualization: VisualizationResponse
    playback: PlaybackInfo
    processing_time: float
    
class FileStatusResponse(BaseModel):
    file_id: str
    status: str  # "processing", "ready", "expired", "error"
    created: datetime
    last_accessed: datetime
    expires_at: datetime
    original_size: int
    compressed_size: Optional[int] = None
    error_message: Optional[str] = None

class CompleteAnalysisResponse(BaseModel):
    analysis: Dict[str, Any]
    visualization: Dict[str, Any]
    playback: Dict[str, Any]
    processing_time: float

class FileStatusResponse(BaseModel):
    file_id: str
    status: str
    created: datetime
    last_accessed: datetime
    expires_at: datetime
    original_size: int
    compressed_size: Optional[int] = None
    error_message: Optional[str] = None
